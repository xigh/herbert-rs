//! Metal weight cache: save pre-quantized buffers to disk, load via mmap + zero-copy.
//!
//! First load: BF16→Q4 quantization + save to cache (~12s + write).
//! Subsequent loads: mmap cache file + newBufferWithBytesNoCopy (zero memcpy, <1s).
//! Cache files stored in `~/.cache/herbert-metal/<model_hash>/<quant>.metalcache`.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapMut};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer as _, MTLDevice};

use crate::loader::QuantMode;
use crate::memory::MetalBuffer;
use crate::model::*;
use herbert_backend_common::loader_common::compute_rope_cache;
use herbert_core::config::Config;
use herbert_core::error::{HerbertError, Result};

const MAGIC: &[u8; 8] = b"MTLCACHE";
const HEADER_SIZE: usize = 64;
const CACHE_VERSION: u32 = 2; // v2: page-aligned data for zero-copy mmap
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
        .join("herbert-metal")
        .join(format!("{:016x}", hash))
        .join(format!("{}.metalcache", quant_str(quant_mode)))
}

// ============================================================================
// Buffer collection (serialize)
// ============================================================================

struct BufferEntry {
    size: usize,
    data: Vec<u8>, // empty if size==0 (BF16 scales placeholder)
}

struct BufferCollector {
    entries: Vec<BufferEntry>,
}

impl BufferCollector {
    fn new() -> Self {
        Self { entries: Vec::new() }
    }

    fn push_buf(&mut self, buf: &MetalBuffer) {
        let size = buf.size as usize;
        let data = if size > 0 {
            unsafe {
                std::slice::from_raw_parts(
                    buf.buffer.contents().as_ptr() as *const u8,
                    size,
                )
            }.to_vec()
        } else {
            Vec::new()
        };
        self.entries.push(BufferEntry { size, data });
    }

    fn push_empty(&mut self) {
        self.entries.push(BufferEntry { size: 0, data: Vec::new() });
    }

    fn push_weight(&mut self, w: &MetalWeight) {
        match w {
            MetalWeight::Q4(q) => {
                self.push_buf(&q.packed);
                self.push_buf(&q.scales);
            }
            MetalWeight::Int8(q) => {
                self.push_buf(&q.packed);
                self.push_buf(&q.scales);
            }
            MetalWeight::BF16(q) => {
                self.push_buf(&q.packed);
                self.push_empty();
            }
        }
    }
}

fn collect_buffers(model: &MetalModel, config: &Config) -> BufferCollector {
    let mut c = BufferCollector::new();

    c.push_buf(&model.embed_tokens);
    c.push_buf(&model.final_norm);
    c.push_weight(&model.lm_head);

    for (i, layer) in model.layers.iter().enumerate() {
        c.push_buf(&layer.input_layernorm);
        c.push_buf(&layer.post_attention_layernorm);
        c.push_weight(&layer.q_proj);
        c.push_weight(&layer.k_proj);
        c.push_weight(&layer.v_proj);
        c.push_weight(&layer.o_proj);
        if let Some(ref qn) = layer.q_norm { c.push_buf(qn); } else { c.push_empty(); }
        if let Some(ref kn) = layer.k_norm { c.push_buf(kn); } else { c.push_empty(); }

        if config.is_moe_layer(i) {
            match &layer.mlp {
                MetalLayerMLP::MoE { router, weights, .. } => {
                    c.push_buf(router);
                    c.push_buf(&weights.gate_packed);
                    if let Some(ref s) = weights.gate_scales { c.push_buf(s); } else { c.push_empty(); }
                    c.push_buf(&weights.up_packed);
                    if let Some(ref s) = weights.up_scales { c.push_buf(s); } else { c.push_empty(); }
                    c.push_buf(&weights.down_packed);
                    if let Some(ref s) = weights.down_scales { c.push_buf(s); } else { c.push_empty(); }
                }
                _ => unreachable!(),
            }
        } else {
            match &layer.mlp {
                MetalLayerMLP::Dense { gate_proj, up_proj, down_proj } => {
                    c.push_weight(gate_proj);
                    c.push_weight(up_proj);
                    c.push_weight(down_proj);
                }
                _ => unreachable!(),
            }
        }
    }

    c
}

// ============================================================================
// Save cache (page-aligned format)
// ============================================================================
//
// File layout:
//   [Header: 64 bytes]
//     magic "MTLCACHE" (8), version (4), quant_id (4), model_hash (8),
//     num_buffers (4), data_start (4), reserved (32)
//   [Size table: num_buffers * 8 bytes]
//   [Padding to PAGE_SIZE boundary]
//   [Buffer 0 data, padded to PAGE_SIZE]
//   [Buffer 1 data, padded to PAGE_SIZE]
//   ...
//
// Each buffer's data starts at a page boundary, enabling newBufferWithBytesNoCopy.

pub fn save_cache(
    model: &MetalModel,
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

    let c = collect_buffers(model, config);
    let num_buffers = c.entries.len();
    let hash = model_hash(model_dir);

    // Compute file layout
    let toc_size = num_buffers * 8;
    let data_start = align_up(HEADER_SIZE + toc_size, PAGE_SIZE);

    // Compute total file size
    let mut file_size = data_start;
    for e in &c.entries {
        if e.size > 0 {
            file_size += align_up(e.size, PAGE_SIZE);
        }
    }

    // Create file and set size
    let tmp_path = path.with_extension("metalcache.tmp");
    let file = fs::File::options()
        .read(true).write(true).create(true).truncate(true)
        .open(&tmp_path)
        .map_err(|e| HerbertError::Backend(format!("Create cache: {}", e)))?;
    file.set_len(file_size as u64)
        .map_err(|e| HerbertError::Backend(format!("Set cache size: {}", e)))?;

    // mmap for writing (faster than sequential writes for random-offset layout)
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
        "[metal] Cache saved: {} buffers, {:.1} MB → {}",
        num_buffers, total_mb, path.display()
    );

    Ok(())
}

// ============================================================================
// Load from cache (zero-copy via mmap + newBufferWithBytesNoCopy)
// ============================================================================

/// Try to load model from cache. Returns None if cache doesn't exist or is invalid.
/// On success, returns (Config, MetalModel, CacheMmap). The CacheMmap must be kept
/// alive for the lifetime of the MetalModel (it owns the mmap backing the buffers).
pub fn try_load_cache(
    device: &ProtocolObject<dyn MTLDevice>,
    model_dir: &Path,
    max_tokens: usize,
    quant_mode: QuantMode,
) -> Option<(Config, MetalModel, CacheMmap)> {
    let path = cache_path(model_dir, quant_mode);
    if !path.exists() {
        return None;
    }

    let config = match Config::from_file(&model_dir.join("config.json")) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[metal] Cache: config load failed: {}", e);
            return None;
        }
    };

    match load_cache_inner(device, &path, model_dir, &config, max_tokens, quant_mode) {
        Ok((model, mmap)) => Some((config, model, mmap)),
        Err(e) => {
            eprintln!("[metal] Cache invalid, will rebuild: {}", e);
            let _ = fs::remove_file(&path);
            None
        }
    }
}

/// Opaque handle to the mmap backing cached Metal buffers.
/// MUST be kept alive as long as the MetalModel is in use.
pub struct CacheMmap {
    _mmap: Mmap,
}

// SAFETY: The mmap is read-only and Metal buffers created from it are only
// accessed via the Metal command queue (sequential submission).
unsafe impl Send for CacheMmap {}
unsafe impl Sync for CacheMmap {}

struct BufferReader {
    sizes: Vec<usize>,
    offsets: Vec<usize>, // file offsets (page-aligned)
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

    fn next_buf_nocopy(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        mmap_ptr: *mut u8,
    ) -> Result<MetalBuffer> {
        if self.idx >= self.sizes.len() {
            return Err(HerbertError::Backend("Cache: out of entries".into()));
        }
        let sz = self.sizes[self.idx];
        let off = self.offsets[self.idx];
        self.idx += 1;

        if sz == 0 {
            MetalBuffer::new(device, 4) // minimal placeholder
        } else {
            let ptr = unsafe { mmap_ptr.add(off) };
            unsafe { MetalBuffer::from_ptr_nocopy(device, ptr, sz) }
        }
    }

    fn next_optional_buf_nocopy(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        mmap_ptr: *mut u8,
    ) -> Result<Option<MetalBuffer>> {
        if self.idx >= self.sizes.len() {
            return Err(HerbertError::Backend("Cache: out of entries".into()));
        }
        let sz = self.sizes[self.idx];
        let off = self.offsets[self.idx];
        self.idx += 1;

        if sz == 0 {
            Ok(None)
        } else {
            let ptr = unsafe { mmap_ptr.add(off) };
            Ok(Some(unsafe { MetalBuffer::from_ptr_nocopy(device, ptr, sz)? }))
        }
    }

    fn next_weight_nocopy(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        mmap_ptr: *mut u8,
        n: usize,
        k: usize,
        mode: QuantMode,
    ) -> Result<MetalWeight> {
        let packed = self.next_buf_nocopy(device, mmap_ptr)?;
        match mode {
            QuantMode::Q4 => {
                let scales = self.next_buf_nocopy(device, mmap_ptr)?;
                Ok(MetalWeight::Q4(MetalQ4Weight { packed, scales, n, k }))
            }
            QuantMode::Int8 => {
                let scales = self.next_buf_nocopy(device, mmap_ptr)?;
                Ok(MetalWeight::Int8(MetalInt8Weight { packed, scales, n, k }))
            }
            QuantMode::BF16 => {
                // Skip the empty placeholder entry
                if self.idx < self.sizes.len() && self.sizes[self.idx] == 0 {
                    self.idx += 1;
                }
                Ok(MetalWeight::BF16(MetalBF16Weight { packed, n, k }))
            }
        }
    }

    fn entries_read(&self) -> usize {
        self.idx
    }
}

fn load_cache_inner(
    device: &ProtocolObject<dyn MTLDevice>,
    path: &Path,
    model_dir: &Path,
    config: &Config,
    max_tokens: usize,
    quant_mode: QuantMode,
) -> Result<(MetalModel, CacheMmap)> {
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

    // Verify the mmap pointer is page-aligned (it should be, mmap guarantees this)
    let mmap_ptr = mmap.as_ptr() as *mut u8;
    debug_assert!(
        (mmap_ptr as usize) % PAGE_SIZE == 0,
        "mmap base not page-aligned: {:?}", mmap_ptr
    );

    let mut r = BufferReader::new(sizes, data_start);
    let hidden_size = config.hidden_size;

    // 1. embed_tokens
    let embed_tokens = r.next_buf_nocopy(device, mmap_ptr)?;
    // 2. final_norm
    let final_norm = r.next_buf_nocopy(device, mmap_ptr)?;
    // 3. lm_head
    let lm_head = r.next_weight_nocopy(device, mmap_ptr, config.vocab_size, hidden_size, quant_mode)?;

    // 4. Per-layer
    let num_layers = config.num_layers;
    let mut layers = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let input_layernorm = r.next_buf_nocopy(device, mmap_ptr)?;
        let post_attention_layernorm = r.next_buf_nocopy(device, mmap_ptr)?;

        let q_proj = r.next_weight_nocopy(device, mmap_ptr, config.q_dim(), hidden_size, quant_mode)?;
        let k_proj = r.next_weight_nocopy(device, mmap_ptr, config.kv_dim(), hidden_size, quant_mode)?;
        let v_proj = r.next_weight_nocopy(device, mmap_ptr, config.kv_dim(), hidden_size, quant_mode)?;
        let o_proj = r.next_weight_nocopy(device, mmap_ptr, hidden_size, config.q_dim(), quant_mode)?;

        let q_norm = r.next_optional_buf_nocopy(device, mmap_ptr)?;
        let k_norm = r.next_optional_buf_nocopy(device, mmap_ptr)?;

        let mlp = if config.is_moe_layer(i) {
            let ne = config.num_experts.unwrap();
            let nept = config.num_experts_per_tok.unwrap();
            let moe_inter = config.moe_intermediate_size.unwrap();
            let norm_topk = config.norm_topk_prob;

            let router = r.next_buf_nocopy(device, mmap_ptr)?;
            let gate_packed = r.next_buf_nocopy(device, mmap_ptr)?;
            let gate_scales = r.next_optional_buf_nocopy(device, mmap_ptr)?;
            let up_packed = r.next_buf_nocopy(device, mmap_ptr)?;
            let up_scales = r.next_optional_buf_nocopy(device, mmap_ptr)?;
            let down_packed = r.next_buf_nocopy(device, mmap_ptr)?;
            let down_scales = r.next_optional_buf_nocopy(device, mmap_ptr)?;

            let format = match quant_mode {
                QuantMode::Q4 => MoEQuantFormat::Q4,
                QuantMode::Int8 => MoEQuantFormat::Int8,
                QuantMode::BF16 => MoEQuantFormat::BF16,
            };

            MetalLayerMLP::MoE {
                router,
                weights: MetalMoEContiguous {
                    gate_packed, gate_scales,
                    up_packed, up_scales,
                    down_packed, down_scales,
                    format,
                    gate_n: moe_inter, gate_k: hidden_size,
                    down_n: hidden_size, down_k: moe_inter,
                },
                num_experts: ne,
                num_experts_per_tok: nept,
                moe_intermediate_size: moe_inter,
                norm_topk_prob: norm_topk,
            }
        } else {
            let gate_proj = r.next_weight_nocopy(device, mmap_ptr, config.intermediate_size, hidden_size, quant_mode)?;
            let up_proj = r.next_weight_nocopy(device, mmap_ptr, config.intermediate_size, hidden_size, quant_mode)?;
            let down_proj = r.next_weight_nocopy(device, mmap_ptr, hidden_size, config.intermediate_size, quant_mode)?;
            MetalLayerMLP::Dense { gate_proj, up_proj, down_proj }
        };

        layers.push(MetalLayer {
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
    let cos_cache = MetalBuffer::from_f32(device, &cos_data)?;
    let sin_cache = MetalBuffer::from_f32(device, &sin_data)?;

    eprintln!(
        "[metal] Cache loaded (zero-copy): {} layers, {} buffers from {}",
        num_layers, r.entries_read(), path.display()
    );

    let dummy_norm_buf = MetalBuffer::from_f32(device, &[0.0f32])?;
    let model = MetalModel {
        embed_tokens,
        embed_tokens_f32: None,
        layers,
        final_norm,
        lm_head,
        cos_cache,
        sin_cache,
        dummy_norm_buf,
    };

    Ok((model, CacheMmap { _mmap: mmap }))
}
