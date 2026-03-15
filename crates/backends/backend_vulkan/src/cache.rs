//! Vulkan weight cache: save pre-quantized buffers to disk, load via mmap + upload.
//!
//! First load: BF16→Q4/Int8 quantization + save to cache.
//! Subsequent loads: mmap cache file + upload to device-local (skips quantization).
//! Cache files stored in `~/.cache/herbert-vulkan/<model_hash>/<quant>.vulkancache`.

use ash::vk;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapMut};

use crate::context::VulkanContext;
use crate::loader::QuantMode;
use crate::memory::VulkanBuffer;
use crate::model::*;
use herbert_backend_common::loader_common::compute_rope_cache;
use herbert_core::config::Config;
use herbert_core::error::{HerbertError, Result};

const MAGIC: &[u8; 8] = b"VLKCACHE";
const HEADER_SIZE: usize = 64;
const CACHE_VERSION: u32 = 1;
const PAGE_SIZE: usize = 4096;

fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

// ============================================================================
// Cache path management
// ============================================================================

fn model_hash(model_dir: &Path) -> u64 {
    let mut hasher = DefaultHasher::new();
    if let Ok(data) = fs::read(model_dir.join("config.json")) {
        data.hash(&mut hasher);
    }
    let index_path = model_dir.join("model.safetensors.index.json");
    if let Ok(data) = fs::read(&index_path) {
        data.hash(&mut hasher);
    } else if let Ok(meta) = fs::metadata(model_dir.join("model.safetensors")) {
        meta.len().hash(&mut hasher);
    }
    hasher.finish()
}

fn quant_str(mode: QuantMode) -> &'static str {
    match mode {
        QuantMode::Q4 => "q4",
        QuantMode::Int8 => "int8",
        QuantMode::BF16 => "bf16",
    }
}

fn quant_id(mode: QuantMode) -> u32 {
    match mode {
        QuantMode::Int8 => 0,
        QuantMode::BF16 => 1,
        QuantMode::Q4 => 2,
    }
}

fn quant_from_id(id: u32) -> Option<QuantMode> {
    match id {
        0 => Some(QuantMode::Int8),
        1 => Some(QuantMode::BF16),
        2 => Some(QuantMode::Q4),
        _ => None,
    }
}

fn cache_path(model_dir: &Path, quant_mode: QuantMode) -> PathBuf {
    let hash = model_hash(model_dir);
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("herbert-vulkan")
        .join(format!("{:016x}", hash))
        .join(format!("{}.vulkancache", quant_str(quant_mode)))
}

// ============================================================================
// Buffer collection (serialize — readback device-local → CPU)
// ============================================================================

struct BufferEntry {
    size: usize,
    data: Vec<u8>,
}

struct BufferCollector {
    entries: Vec<BufferEntry>,
}

impl BufferCollector {
    fn new() -> Self {
        Self { entries: Vec::new() }
    }

    fn push_buf(&mut self, ctx: &VulkanContext, buf: &VulkanBuffer) -> Result<()> {
        let size = buf.size as usize;
        let data = if size > 0 {
            buf.read_bytes(ctx, size)?
        } else {
            Vec::new()
        };
        self.entries.push(BufferEntry { size, data });
        Ok(())
    }

    fn push_empty(&mut self) {
        self.entries.push(BufferEntry { size: 0, data: Vec::new() });
    }

    fn push_weight(&mut self, ctx: &VulkanContext, w: &VulkanWeight) -> Result<()> {
        match w {
            VulkanWeight::Q4(q) => {
                self.push_buf(ctx, &q.packed)?;
                self.push_buf(ctx, &q.scales)?;
            }
            VulkanWeight::Int8(q) => {
                self.push_buf(ctx, &q.packed)?;
                self.push_buf(ctx, &q.scales)?;
            }
            VulkanWeight::BF16(q) => {
                self.push_buf(ctx, &q.packed)?;
                self.push_empty();
            }
        }
        Ok(())
    }
}

fn collect_buffers(ctx: &VulkanContext, model: &VulkanModel, config: &Config) -> Result<BufferCollector> {
    let mut c = BufferCollector::new();

    c.push_buf(ctx, &model.embed_tokens)?;
    c.push_buf(ctx, &model.final_norm)?;
    c.push_weight(ctx, &model.lm_head)?;

    for (i, layer) in model.layers.iter().enumerate() {
        c.push_buf(ctx, &layer.input_layernorm)?;
        c.push_buf(ctx, &layer.post_attention_layernorm)?;
        c.push_weight(ctx, &layer.q_proj)?;
        c.push_weight(ctx, &layer.k_proj)?;
        c.push_weight(ctx, &layer.v_proj)?;
        c.push_weight(ctx, &layer.o_proj)?;
        c.push_buf(ctx, &layer.q_norm)?;
        c.push_buf(ctx, &layer.k_norm)?;

        if config.is_moe_layer(i) {
            match &layer.mlp {
                VulkanLayerMLP::MoE { router, experts, .. } => {
                    c.push_buf(ctx, router)?;
                    for expert in experts {
                        c.push_weight(ctx, &expert.gate_proj)?;
                        c.push_weight(ctx, &expert.up_proj)?;
                        c.push_weight(ctx, &expert.down_proj)?;
                    }
                }
                _ => unreachable!(),
            }
        } else {
            match &layer.mlp {
                VulkanLayerMLP::Dense { gate_proj, up_proj, down_proj } => {
                    c.push_weight(ctx, gate_proj)?;
                    c.push_weight(ctx, up_proj)?;
                    c.push_weight(ctx, down_proj)?;
                }
                _ => unreachable!(),
            }
        }
    }

    Ok(c)
}

// ============================================================================
// Save cache
// ============================================================================

pub fn save_cache(
    ctx: &VulkanContext,
    model: &VulkanModel,
    config: &Config,
    model_dir: &Path,
    quant_mode: QuantMode,
) -> Result<()> {
    let path = cache_path(model_dir, quant_mode);

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            HerbertError::Backend(format!("Failed to create cache dir: {}", e))
        })?;
    }

    let c = collect_buffers(ctx, model, config)?;
    let num_buffers = c.entries.len();
    let hash = model_hash(model_dir);

    let toc_size = num_buffers * 8;
    let data_start = align_up(HEADER_SIZE + toc_size, PAGE_SIZE);

    let mut file_size = data_start;
    for e in &c.entries {
        if e.size > 0 {
            file_size += align_up(e.size, PAGE_SIZE);
        }
    }

    let tmp_path = path.with_extension("vulkancache.tmp");
    let file = fs::File::options()
        .read(true).write(true).create(true).truncate(true)
        .open(&tmp_path)
        .map_err(|e| HerbertError::Backend(format!("Create cache: {}", e)))?;
    file.set_len(file_size as u64)
        .map_err(|e| HerbertError::Backend(format!("Set cache size: {}", e)))?;

    let mut mmap = unsafe { MmapMut::map_mut(&file) }
        .map_err(|e| HerbertError::Backend(format!("Mmap cache for write: {}", e)))?;

    let buf = &mut mmap[..];

    // Header
    buf[0..8].copy_from_slice(MAGIC);
    buf[8..12].copy_from_slice(&CACHE_VERSION.to_le_bytes());
    buf[12..16].copy_from_slice(&quant_id(quant_mode).to_le_bytes());
    buf[16..24].copy_from_slice(&hash.to_le_bytes());
    buf[24..28].copy_from_slice(&(num_buffers as u32).to_le_bytes());
    buf[28..32].copy_from_slice(&(data_start as u32).to_le_bytes());

    // Size table
    for (i, e) in c.entries.iter().enumerate() {
        let off = HEADER_SIZE + i * 8;
        buf[off..off + 8].copy_from_slice(&(e.size as u64).to_le_bytes());
    }

    // Buffer data (page-aligned)
    let mut pos = data_start;
    for e in &c.entries {
        if e.size > 0 {
            buf[pos..pos + e.size].copy_from_slice(&e.data);
            pos += align_up(e.size, PAGE_SIZE);
        }
    }

    mmap.flush().map_err(|e| HerbertError::Backend(format!("Flush cache: {}", e)))?;
    drop(mmap);
    drop(file);

    fs::rename(&tmp_path, &path).map_err(|e| {
        HerbertError::Backend(format!("Rename cache: {}", e))
    })?;

    let total_mb = file_size as f64 / 1_048_576.0;
    eprintln!(
        "[vulkan] Cache saved: {} buffers, {:.1} MB → {}",
        num_buffers, total_mb, path.display()
    );

    Ok(())
}

// ============================================================================
// Load from cache (mmap + upload to device-local)
// ============================================================================

pub fn try_load_cache(
    ctx: &VulkanContext,
    model_dir: &Path,
    max_tokens: usize,
    quant_mode: QuantMode,
) -> Option<(Config, VulkanModel)> {
    let path = cache_path(model_dir, quant_mode);
    if !path.exists() {
        return None;
    }

    let config = match Config::from_file(&model_dir.join("config.json")) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[vulkan] Cache: config load failed: {}", e);
            return None;
        }
    };

    match load_cache_inner(ctx, &path, model_dir, &config, max_tokens, quant_mode) {
        Ok(model) => Some((config, model)),
        Err(e) => {
            eprintln!("[vulkan] Cache invalid, will rebuild: {}", e);
            let _ = fs::remove_file(&path);
            None
        }
    }
}

struct BufferReader {
    sizes: Vec<usize>,
    offsets: Vec<usize>,
    idx: usize,
}

impl BufferReader {
    fn new(sizes: Vec<usize>, data_start: usize) -> Self {
        let mut offsets = Vec::with_capacity(sizes.len());
        let mut pos = data_start;
        for &sz in &sizes {
            offsets.push(pos);
            if sz > 0 {
                pos += align_up(sz, PAGE_SIZE);
            }
        }
        Self { sizes, offsets, idx: 0 }
    }

    fn next_buf(&mut self, ctx: &VulkanContext, mmap: &[u8]) -> Result<VulkanBuffer> {
        if self.idx >= self.sizes.len() {
            return Err(HerbertError::Backend("Cache: out of entries".into()));
        }
        let sz = self.sizes[self.idx];
        let off = self.offsets[self.idx];
        self.idx += 1;

        if sz == 0 {
            VulkanBuffer::device_local(ctx, 4, vk::BufferUsageFlags::empty())
        } else {
            VulkanBuffer::upload_to_device_local(ctx, &mmap[off..off + sz], vk::BufferUsageFlags::empty())
        }
    }

    fn next_weight(
        &mut self,
        ctx: &VulkanContext,
        mmap: &[u8],
        n: usize,
        k: usize,
        mode: QuantMode,
    ) -> Result<VulkanWeight> {
        let packed = self.next_buf(ctx, mmap)?;
        match mode {
            QuantMode::Q4 => {
                let scales = self.next_buf(ctx, mmap)?;
                Ok(VulkanWeight::Q4(VulkanQ4Weight { packed, scales, n, k }))
            }
            QuantMode::Int8 => {
                let scales = self.next_buf(ctx, mmap)?;
                Ok(VulkanWeight::Int8(VulkanInt8Weight { packed, scales, n, k }))
            }
            QuantMode::BF16 => {
                // Skip empty placeholder
                if self.idx < self.sizes.len() && self.sizes[self.idx] == 0 {
                    self.idx += 1;
                }
                Ok(VulkanWeight::BF16(VulkanBF16Weight { packed, n, k }))
            }
        }
    }

    fn entries_read(&self) -> usize {
        self.idx
    }
}

fn load_cache_inner(
    ctx: &VulkanContext,
    path: &Path,
    model_dir: &Path,
    config: &Config,
    max_tokens: usize,
    quant_mode: QuantMode,
) -> Result<VulkanModel> {
    let file = fs::File::open(path).map_err(|e| {
        HerbertError::Backend(format!("Open cache: {}", e))
    })?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
        HerbertError::Backend(format!("Mmap cache: {}", e))
    })?;

    let bytes = &mmap[..];
    if bytes.len() < HEADER_SIZE {
        return Err(HerbertError::Backend("Cache too small".into()));
    }

    // Validate header
    if &bytes[0..8] != MAGIC {
        return Err(HerbertError::Backend("Bad magic".into()));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != CACHE_VERSION {
        return Err(HerbertError::Backend(format!("Cache version {version} != {CACHE_VERSION}")));
    }
    let stored_quant = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    if quant_from_id(stored_quant) != Some(quant_mode) {
        return Err(HerbertError::Backend("Quant mismatch".into()));
    }
    let stored_hash = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    if stored_hash != model_hash(model_dir) {
        return Err(HerbertError::Backend("Model hash mismatch".into()));
    }
    let num_buffers = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
    let data_start = u32::from_le_bytes(bytes[28..32].try_into().unwrap()) as usize;

    // Read size table
    let toc_end = HEADER_SIZE + num_buffers * 8;
    if bytes.len() < toc_end {
        return Err(HerbertError::Backend("Cache truncated (TOC)".into()));
    }

    let mut sizes = Vec::with_capacity(num_buffers);
    for i in 0..num_buffers {
        let off = HEADER_SIZE + i * 8;
        sizes.push(u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()) as usize);
    }

    let mut r = BufferReader::new(sizes, data_start);
    let hidden_size = config.hidden_size;

    // 1. embed_tokens
    let embed_tokens = r.next_buf(ctx, bytes)?;
    // 2. final_norm
    let final_norm = r.next_buf(ctx, bytes)?;
    // 3. lm_head
    let lm_head = r.next_weight(ctx, bytes, config.vocab_size, hidden_size, quant_mode)?;

    // 4. Per-layer
    let num_layers = config.num_layers;
    let mut layers = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let input_layernorm = r.next_buf(ctx, bytes)?;
        let post_attention_layernorm = r.next_buf(ctx, bytes)?;

        let q_proj = r.next_weight(ctx, bytes, config.q_dim(), hidden_size, quant_mode)?;
        let k_proj = r.next_weight(ctx, bytes, config.kv_dim(), hidden_size, quant_mode)?;
        let v_proj = r.next_weight(ctx, bytes, config.kv_dim(), hidden_size, quant_mode)?;
        let o_proj = r.next_weight(ctx, bytes, hidden_size, config.q_dim(), quant_mode)?;

        let q_norm = r.next_buf(ctx, bytes)?;
        let k_norm = r.next_buf(ctx, bytes)?;

        let mlp = if config.is_moe_layer(i) {
            let ne = config.num_experts.unwrap();
            let nept = config.num_experts_per_tok.unwrap();
            let moe_inter = config.moe_intermediate_size.unwrap();
            let norm_topk = config.norm_topk_prob;

            let router = r.next_buf(ctx, bytes)?;
            let mut experts = Vec::with_capacity(ne);
            for _ in 0..ne {
                let gate_proj = r.next_weight(ctx, bytes, moe_inter, hidden_size, quant_mode)?;
                let up_proj = r.next_weight(ctx, bytes, moe_inter, hidden_size, quant_mode)?;
                let down_proj = r.next_weight(ctx, bytes, hidden_size, moe_inter, quant_mode)?;
                experts.push(VulkanMoEExpert { gate_proj, up_proj, down_proj });
            }

            VulkanLayerMLP::MoE {
                router,
                experts,
                num_experts: ne,
                num_experts_per_tok: nept,
                moe_intermediate_size: moe_inter,
                norm_topk_prob: norm_topk,
            }
        } else {
            let gate_proj = r.next_weight(ctx, bytes, config.intermediate_size, hidden_size, quant_mode)?;
            let up_proj = r.next_weight(ctx, bytes, config.intermediate_size, hidden_size, quant_mode)?;
            let down_proj = r.next_weight(ctx, bytes, hidden_size, config.intermediate_size, quant_mode)?;
            VulkanLayerMLP::Dense { gate_proj, up_proj, down_proj }
        };

        layers.push(VulkanLayer {
            input_layernorm, post_attention_layernorm,
            q_proj, k_proj, v_proj, o_proj,
            q_norm, k_norm,
            mlp,
        });
    }

    // RoPE cache (always recomputed — cheap)
    let max_seq_len = config.max_position_embeddings.min(max_tokens);
    let rope_config = Config {
        max_position_embeddings: max_seq_len,
        ..config.clone()
    };
    let (cos_data, sin_data) = compute_rope_cache(&rope_config);

    let cos_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(cos_data.as_ptr() as *const u8, cos_data.len() * 4)
    };
    let sin_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(sin_data.as_ptr() as *const u8, sin_data.len() * 4)
    };
    let cos_cache = VulkanBuffer::upload_to_device_local(ctx, cos_bytes, vk::BufferUsageFlags::empty())?;
    let sin_cache = VulkanBuffer::upload_to_device_local(ctx, sin_bytes, vk::BufferUsageFlags::empty())?;

    eprintln!(
        "[vulkan] Cache loaded: {} layers, {} buffers from {}",
        num_layers, r.entries_read(), path.display()
    );

    Ok(VulkanModel {
        embed_tokens,
        embed_tokens_f32: None,
        layers,
        final_norm,
        lm_head,
        cos_cache,
        sin_cache,
    })
}
