//! Quantized weight cache: save/load pre-quantized weights to skip quantization on subsequent loads.
//!
//! Cache files are stored in `~/.cache/herbert/<model_name>/<backend_id>_v<version>.herbcache`.

use crate::expert_pool::ExpertPool;
use crate::generic_attention::GenericAttention;
use crate::generic_layer::{GenericConvolution, GenericDecoderLayer, GenericGatedDeltaNet, LayerBlock};
use crate::generic_mlp::{GenericMLP, MlpActivation};
use crate::generic_moe::{GenericFFN, GenericMoE};
use crate::generic_model::GenericModel;
use crate::linear_ops::LinearOps;
use crate::loader_common::compute_rope_cache;
use crate::progress;
use herbert_core::config::Config;
use herbert_core::tensor::BF16;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, Read, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

// ============================================================================
// CacheWeight trait
// ============================================================================

/// Trait for types that can be serialized/deserialized to/from a binary cache.
pub trait CacheWeight: Sized {
    /// Increment this when the binary layout changes.
    const CACHE_VERSION: u32;

    /// Write the weight data to a writer.
    fn cache_write(&self, w: &mut impl Write) -> io::Result<()>;

    /// Read the weight data from a reader.
    fn cache_read(r: &mut impl Read) -> io::Result<Self>;
}

// ============================================================================
// Cache file header
// ============================================================================

const MAGIC: &[u8; 8] = b"QW3CACHE";
const HEADER_SIZE: usize = 64;

struct CacheHeader {
    header_version: u32,
    cache_format_version: u32,
    backend_id: [u8; 32],
    model_hash: [u64; 2],
}

impl CacheHeader {
    fn write(&self, w: &mut impl Write) -> io::Result<()> {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..8].copy_from_slice(MAGIC);
        buf[8..12].copy_from_slice(&self.header_version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.cache_format_version.to_le_bytes());
        buf[16..48].copy_from_slice(&self.backend_id);
        buf[48..56].copy_from_slice(&self.model_hash[0].to_le_bytes());
        buf[56..64].copy_from_slice(&self.model_hash[1].to_le_bytes());
        w.write_all(&buf)
    }

    fn read(r: &mut impl Read) -> io::Result<Self> {
        let mut buf = [0u8; HEADER_SIZE];
        r.read_exact(&mut buf)?;
        if &buf[0..8] != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        let header_version = u32::from_le_bytes(buf[8..12].try_into().expect("4-byte slice"));
        if header_version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported header version {}", header_version),
            ));
        }
        let cache_format_version = u32::from_le_bytes(buf[12..16].try_into().expect("4-byte slice"));
        let mut backend_id = [0u8; 32];
        backend_id.copy_from_slice(&buf[16..48]);
        let h0 = u64::from_le_bytes(buf[48..56].try_into().expect("8-byte slice"));
        let h1 = u64::from_le_bytes(buf[56..64].try_into().expect("8-byte slice"));
        Ok(Self {
            header_version,
            cache_format_version,
            backend_id,
            model_hash: [h0, h1],
        })
    }
}

fn make_backend_id_bytes(backend_id: &str) -> [u8; 32] {
    let mut buf = [0u8; 32];
    let bytes = backend_id.as_bytes();
    let len = bytes.len().min(32);
    buf[..len].copy_from_slice(&bytes[..len]);
    buf
}

// ============================================================================
// Model identity hash
// ============================================================================

/// Compute a 128-bit identity hash for a model directory.
///
/// Hashes: config.json content, model.safetensors.index.json content (if exists),
/// and sorted safetensor file sizes.
pub fn compute_model_hash(model_dir: &Path) -> io::Result<[u64; 2]> {
    let mut hasher1 = DefaultHasher::new();
    let mut hasher2 = DefaultHasher::new();

    // Hash config.json
    let config_bytes = fs::read(model_dir.join("config.json"))?;
    config_bytes.hash(&mut hasher1);
    config_bytes.hash(&mut hasher2);

    // Hash index file if present
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_bytes = fs::read(&index_path)?;
        index_bytes.hash(&mut hasher1);
        // Use different seed for second hash
        index_bytes.len().hash(&mut hasher2);
        index_bytes.hash(&mut hasher2);
    }

    // Hash safetensor file sizes (sorted by name)
    let mut st_files: Vec<(String, u64)> = Vec::new();
    for entry in fs::read_dir(model_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".safetensors") {
            let meta = entry.metadata()?;
            st_files.push((name, meta.len()));
        }
    }
    st_files.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, size) in &st_files {
        name.hash(&mut hasher1);
        size.hash(&mut hasher1);
        name.hash(&mut hasher2);
        size.hash(&mut hasher2);
    }

    // Mix in a different constant for the second hash so they're independent
    0xDEADBEEFu64.hash(&mut hasher2);

    Ok([hasher1.finish(), hasher2.finish()])
}

// ============================================================================
// Cache path helpers
// ============================================================================

/// Derive the model name from the model directory path.
fn model_name_from_dir(model_dir: &Path) -> String {
    model_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Get the cache directory for a model: ~/.cache/herbert/<model_name>/
fn cache_dir_for_model(model_dir: &Path) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let model_name = model_name_from_dir(model_dir);
    Some(PathBuf::from(home).join(".cache").join("herbert").join(model_name))
}

/// Get the full cache file path.
fn cache_file_path(model_dir: &Path, backend_id: &str, version: u32) -> Option<PathBuf> {
    let dir = cache_dir_for_model(model_dir)?;
    Some(dir.join(format!("{}_v{}.herbcache", backend_id, version)))
}

// ============================================================================
// I/O helpers
// ============================================================================

fn write_u32(w: &mut impl Write, val: u32) -> io::Result<()> {
    w.write_all(&val.to_le_bytes())
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_u64(w: &mut impl Write, val: u64) -> io::Result<()> {
    w.write_all(&val.to_le_bytes())
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn write_u8(w: &mut impl Write, val: u8) -> io::Result<()> {
    w.write_all(&[val])
}

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn write_f32_slice(w: &mut impl Write, data: &[f32]) -> io::Result<()> {
    write_u64(w, data.len() as u64)?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    w.write_all(bytes)
}

fn read_f32_vec(r: &mut impl Read) -> io::Result<Vec<f32>> {
    let len = read_u64(r)? as usize;
    let mut out = vec![0.0f32; len];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, len * 4) };
    r.read_exact(bytes)?;
    // x86 is little-endian, so the bytes are already in the correct order
    Ok(out)
}

fn write_bf16_slice(w: &mut impl Write, data: &[BF16]) -> io::Result<()> {
    write_u64(w, data.len() as u64)?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
    w.write_all(bytes)
}

fn read_bf16_vec(r: &mut impl Read) -> io::Result<Vec<BF16>> {
    let len = read_u64(r)? as usize;
    let mut out = vec![0u16; len];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, len * 2) };
    r.read_exact(bytes)?;
    Ok(out)
}

fn write_optional_bf16(w: &mut impl Write, data: &Option<Vec<BF16>>) -> io::Result<()> {
    match data {
        Some(v) => {
            write_u8(w, 1)?;
            write_bf16_slice(w, v)
        }
        None => write_u8(w, 0),
    }
}

fn read_optional_bf16(r: &mut impl Read) -> io::Result<Option<Vec<BF16>>> {
    let flag = read_u8(r)?;
    if flag == 0 {
        Ok(None)
    } else {
        Ok(Some(read_bf16_vec(r)?))
    }
}

fn write_optional_f32(w: &mut impl Write, data: &Option<Vec<f32>>) -> io::Result<()> {
    match data {
        Some(v) => {
            write_u8(w, 1)?;
            write_f32_slice(w, v)
        }
        None => write_u8(w, 0),
    }
}

fn read_optional_f32(r: &mut impl Read) -> io::Result<Option<Vec<f32>>> {
    let flag = read_u8(r)?;
    if flag == 0 {
        Ok(None)
    } else {
        Ok(Some(read_f32_vec(r)?))
    }
}

// ============================================================================
// Save / Load full model cache
// ============================================================================

/// Save a GenericModel to the weight cache.
///
/// Returns `Some(cache_path)` on success, `None` if the cache path cannot be determined.
pub fn save_model_cache<L: LinearOps>(
    model: &GenericModel<L>,
    _config: &Config,
    model_dir: &Path,
    backend_id: &str,
    model_hash: [u64; 2],
) -> io::Result<Option<PathBuf>>
where
    L::Weight: CacheWeight,
{
    let path = match cache_file_path(model_dir, backend_id, L::Weight::CACHE_VERSION) {
        Some(p) => p,
        None => return Ok(None), // can't determine cache path, silently skip
    };

    // Create cache directory
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp_path = path.with_extension("tmp");
    let file = fs::File::create(&tmp_path)?;
    let mut w = io::BufWriter::with_capacity(4 * 1024 * 1024, file); // 4 MB buffer

    // Header
    let header = CacheHeader {
        header_version: 1,
        cache_format_version: L::Weight::CACHE_VERSION,
        backend_id: make_backend_id_bytes(backend_id),
        model_hash,
    };
    header.write(&mut w)?;

    // embed_tokens (bf16)
    write_bf16_slice(&mut w, &model.embed_tokens)?;

    // norm (bf16)
    write_bf16_slice(&mut w, &model.norm)?;

    // lm_head (CacheWeight)
    model.lm_head.cache_write(&mut w)?;

    // num_layers
    let num_layers = model.decoder_layers.len() as u32;
    write_u32(&mut w, num_layers)?;

    // Each layer
    for layer in &model.decoder_layers {
        // Norms
        write_bf16_slice(&mut w, &layer.input_layernorm)?;
        write_bf16_slice(&mut w, &layer.post_attention_layernorm)?;

        // Gemma3 pre/post feedforward layernorms
        write_optional_bf16(&mut w, &layer.pre_feedforward_layernorm)?;
        write_optional_bf16(&mut w, &layer.post_feedforward_layernorm)?;

        // Block type: 0 = Attention, 1 = Convolution, 2 = GatedDeltaNet
        match &layer.block {
            LayerBlock::Attention(attn) => {
                write_u8(&mut w, 0)?;
                attn.q_proj.cache_write(&mut w)?;
                attn.k_proj.cache_write(&mut w)?;
                attn.v_proj.cache_write(&mut w)?;
                attn.o_proj.cache_write(&mut w)?;

                // Optional q/k norms
                write_optional_bf16(&mut w, &attn.q_norm)?;
                write_optional_bf16(&mut w, &attn.k_norm)?;

                // Optional q/k/v biases
                write_optional_f32(&mut w, &attn.q_bias)?;
                write_optional_f32(&mut w, &attn.k_bias)?;
                write_optional_f32(&mut w, &attn.v_bias)?;
            }
            LayerBlock::Convolution(conv) => {
                write_u8(&mut w, 1)?;
                conv.in_proj.cache_write(&mut w)?;
                write_f32_slice(&mut w, &conv.conv_kernel)?;
                conv.out_proj.cache_write(&mut w)?;
                write_u32(&mut w, conv.hidden_size as u32)?;
                write_u32(&mut w, conv.conv_l_cache as u32)?;
            }
            LayerBlock::GatedDeltaNet(dn) => {
                write_u8(&mut w, 2)?;
                dn.in_proj_qkv.cache_write(&mut w)?;
                dn.in_proj_z.cache_write(&mut w)?;
                write_f32_slice(&mut w, &dn.in_proj_a)?;
                write_f32_slice(&mut w, &dn.in_proj_b)?;
                write_f32_slice(&mut w, &dn.conv1d_weight)?;
                write_f32_slice(&mut w, &dn.a_log)?;
                write_f32_slice(&mut w, &dn.dt_bias)?;
                write_bf16_slice(&mut w, &dn.norm_weight)?;
                dn.out_proj.cache_write(&mut w)?;
            }
        }

        // FFN
        match &layer.ffn {
            GenericFFN::Dense(mlp) => {
                write_u8(&mut w, 0)?; // Dense
                mlp.gate_proj.cache_write(&mut w)?;
                mlp.up_proj.cache_write(&mut w)?;
                mlp.down_proj.cache_write(&mut w)?;
            }
            GenericFFN::MoE(moe) => {
                write_u8(&mut w, 1)?; // MoE
                moe.gate.cache_write(&mut w)?;
                write_u32(&mut w, moe.num_experts as u32)?;
                let pool = moe.expert_pool.lock().unwrap_or_else(|e| e.into_inner());
                for eid in 0..moe.num_experts {
                    let expert = pool.get(eid);
                    expert.gate_proj.cache_write(&mut w)?;
                    expert.up_proj.cache_write(&mut w)?;
                    expert.down_proj.cache_write(&mut w)?;
                }
                // Expert bias (LFM2)
                write_optional_f32(&mut w, &moe.expert_bias)?;
                // Shared expert (Qwen3.5)
                let has_shared = moe.shared_expert.is_some();
                write_u8(&mut w, has_shared as u8)?;
                if let Some(ref shared) = moe.shared_expert {
                    shared.gate_proj.cache_write(&mut w)?;
                    shared.up_proj.cache_write(&mut w)?;
                    shared.down_proj.cache_write(&mut w)?;
                }
                write_optional_f32(&mut w, &moe.shared_expert_gate)?;
            }
        }
    }

    w.flush()?;
    drop(w);

    // Atomic rename
    fs::rename(&tmp_path, &path)?;
    Ok(Some(path))
}

/// Try to load a GenericModel from the weight cache.
///
/// Returns `None` if the cache doesn't exist or is invalid.
/// On I/O errors, the corrupt cache file is removed.
pub fn load_model_cache<L: LinearOps>(
    config: &Config,
    model_dir: &Path,
    backend_id: &str,
    model_hash: [u64; 2],
    show_progress: bool,
) -> Option<GenericModel<L>>
where
    L::Weight: CacheWeight,
{
    let path = cache_file_path(model_dir, backend_id, L::Weight::CACHE_VERSION)?;
    if !path.exists() {
        return None;
    }

    match load_model_cache_inner::<L>(config, &path, backend_id, model_hash, show_progress) {
        Ok(model) => Some(model),
        Err(e) => {
            tracing::warn!(error = %e, "weight cache invalid, re-quantizing");
            let _ = fs::remove_file(&path);
            None
        }
    }
}

fn load_model_cache_inner<L: LinearOps>(
    config: &Config,
    path: &Path,
    backend_id: &str,
    model_hash: [u64; 2],
    show_progress: bool,
) -> io::Result<GenericModel<L>>
where
    L::Weight: CacheWeight,
{
    let file = fs::File::open(path)?;
    let mut r = io::BufReader::with_capacity(4 * 1024 * 1024, file); // 4 MB buffer

    // Header
    let header = CacheHeader::read(&mut r)?;

    // Validate backend_id
    let expected_bid = make_backend_id_bytes(backend_id);
    if header.backend_id != expected_bid {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "backend_id mismatch",
        ));
    }

    // Validate cache_format_version
    if header.cache_format_version != L::Weight::CACHE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "cache version mismatch: file={}, expected={}",
                header.cache_format_version,
                L::Weight::CACHE_VERSION
            ),
        ));
    }

    // Validate model hash
    if header.model_hash != model_hash {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "model hash mismatch (model files changed)",
        ));
    }

    // embed_tokens (bf16)
    let embed_tokens = read_bf16_vec(&mut r)?;

    // norm
    let norm = read_bf16_vec(&mut r)?;

    // lm_head
    let lm_head = L::Weight::cache_read(&mut r)?;

    // layers
    let num_layers = read_u32(&mut r)? as usize;
    let mut decoder_layers = Vec::with_capacity(num_layers);

    let pb = if show_progress {
        Some(progress::cache_bar(num_layers))
    } else {
        None
    };

    for layer_idx in 0..num_layers {
        let input_layernorm = read_bf16_vec(&mut r)?;
        let post_attention_layernorm = read_bf16_vec(&mut r)?;

        // Gemma3 pre/post feedforward layernorms
        let pre_feedforward_layernorm = read_optional_bf16(&mut r)?;
        let post_feedforward_layernorm = read_optional_bf16(&mut r)?;

        // Block type: 0 = Attention, 1 = Convolution, 2 = GatedDeltaNet
        let block_type = read_u8(&mut r)?;
        let block = if block_type == 0 {
            let q_proj = L::Weight::cache_read(&mut r)?;
            let k_proj = L::Weight::cache_read(&mut r)?;
            let v_proj = L::Weight::cache_read(&mut r)?;
            let o_proj = L::Weight::cache_read(&mut r)?;

            let q_norm = read_optional_bf16(&mut r)?;
            let k_norm = read_optional_bf16(&mut r)?;

            let q_bias = read_optional_f32(&mut r)?;
            let k_bias = read_optional_f32(&mut r)?;
            let v_bias = read_optional_f32(&mut r)?;

            {
                let use_rope = true;
                LayerBlock::Attention(GenericAttention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    q_bias,
                    k_bias,
                    v_bias,
                    o_bias: None,
                    num_heads: config.num_attention_heads,
                    num_kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                    rotary_ndims: config.rotary_ndims,
                    head_to_kv_head: (0..config.num_attention_heads)
                        .map(|h| h / config.num_kv_groups())
                        .collect(),
                    hidden_size: config.hidden_size,
                    q_dim: config.q_dim(),
                    kv_dim: config.kv_dim(),
                    rms_norm_eps: config.rms_norm_eps,
                    use_rope,
                    attn_scale: config.attn_scale,
                    use_gemma_qk_norm: config.norm_type == herbert_core::config::NormType::GemmaRMSNorm,
                    has_output_gate: config.attn_output_gate,
                    use_q_bf16: false,
                    _marker: PhantomData,
                })
            }
        } else if block_type == 2 {
            // GatedDeltaNet
            let in_proj_qkv = L::Weight::cache_read(&mut r)?;
            let in_proj_z = L::Weight::cache_read(&mut r)?;
            let in_proj_a = read_f32_vec(&mut r)?;
            let in_proj_b = read_f32_vec(&mut r)?;
            let conv1d_weight = read_f32_vec(&mut r)?;
            let a_log = read_f32_vec(&mut r)?;
            let dt_bias = read_f32_vec(&mut r)?;
            let norm_weight = read_bf16_vec(&mut r)?;
            let out_proj = L::Weight::cache_read(&mut r)?;

            LayerBlock::GatedDeltaNet(GenericGatedDeltaNet {
                in_proj_qkv,
                in_proj_z,
                in_proj_a,
                in_proj_b,
                conv1d_weight,
                a_log,
                dt_bias,
                norm_weight,
                out_proj,
                num_k_heads: config.linear_num_key_heads.unwrap_or(0),
                num_v_heads: config.linear_num_value_heads.unwrap_or(0),
                head_k_dim: config.linear_key_head_dim.unwrap_or(0),
                head_v_dim: config.linear_value_head_dim.unwrap_or(0),
                conv_kernel_size: config.linear_conv_kernel_dim.unwrap_or(4),
                hidden_size: config.hidden_size,
                rms_norm_eps: config.rms_norm_eps,
            })
        } else {
            let in_proj = L::Weight::cache_read(&mut r)?;
            let conv_kernel = read_f32_vec(&mut r)?;
            let out_proj = L::Weight::cache_read(&mut r)?;
            let hidden_size = read_u32(&mut r)? as usize;
            let conv_l_cache = read_u32(&mut r)? as usize;

            LayerBlock::Convolution(GenericConvolution {
                in_proj,
                conv_kernel,
                out_proj,
                hidden_size,
                conv_l_cache,
            })
        };

        let ffn_type = read_u8(&mut r)?;
        let ffn = if ffn_type == 0 {
            // Dense
            let gate_proj = L::Weight::cache_read(&mut r)?;
            let up_proj = L::Weight::cache_read(&mut r)?;
            let down_proj = L::Weight::cache_read(&mut r)?;
            let activation = MlpActivation::SwiGLU;
            GenericFFN::Dense(GenericMLP {
                gate_proj,
                up_proj,
                down_proj,
                hidden_size: config.hidden_size,
                intermediate_size: config.intermediate_size,
                activation,
                _marker: PhantomData,
            })
        } else {
            // MoE
            let activation = MlpActivation::SwiGLU;
            let gate = L::Weight::cache_read(&mut r)?;
            let num_experts = read_u32(&mut r)? as usize;
            let mut experts = Vec::with_capacity(num_experts);
            let moe_intermediate_size = config.moe_intermediate_size.unwrap_or(config.intermediate_size);
            for _ in 0..num_experts {
                let gate_proj = L::Weight::cache_read(&mut r)?;
                let up_proj = L::Weight::cache_read(&mut r)?;
                let down_proj = L::Weight::cache_read(&mut r)?;
                experts.push(GenericMLP {
                    gate_proj,
                    up_proj,
                    down_proj,
                    hidden_size: config.hidden_size,
                    intermediate_size: moe_intermediate_size,
                    activation,
                    _marker: PhantomData,
                });
            }
            // Expert bias (LFM2)
            let expert_bias = read_optional_f32(&mut r)?;
            // Shared expert (Qwen3.5)
            let has_shared = read_u8(&mut r)?;
            let shared_expert = if has_shared != 0 {
                let shared_gate_proj = L::Weight::cache_read(&mut r)?;
                let shared_up_proj = L::Weight::cache_read(&mut r)?;
                let shared_down_proj = L::Weight::cache_read(&mut r)?;
                let shared_inter = config.shared_expert_intermediate_size.unwrap_or(config.intermediate_size);
                Some(GenericMLP {
                    gate_proj: shared_gate_proj,
                    up_proj: shared_up_proj,
                    down_proj: shared_down_proj,
                    hidden_size: config.hidden_size,
                    intermediate_size: shared_inter,
                    activation,
                    _marker: PhantomData,
                })
            } else {
                None
            };
            let shared_expert_gate = read_optional_f32(&mut r)?;
            GenericFFN::MoE(GenericMoE {
                gate,
                expert_pool: std::sync::Mutex::new(ExpertPool::new_all_loaded(experts, layer_idx)),
                num_experts,
                num_experts_per_tok: config.num_experts_per_tok.unwrap_or(0),
                hidden_size: config.hidden_size,
                moe_intermediate_size,
                norm_topk_prob: config.norm_topk_prob,
                layer_id: layer_idx,
                expert_bias,
                shared_expert,
                shared_expert_gate,
            })
        };

        decoder_layers.push(GenericDecoderLayer {
            input_layernorm,
            block,
            post_attention_layernorm,
            ffn,
            hidden_size: config.hidden_size,
            rms_norm_eps: config.rms_norm_eps,
            input_layernorm_bias: None,
            post_attention_layernorm_bias: None,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            norm_type: config.norm_type,
        });

        if let Some(ref pb) = pb {
            pb.set_position((layer_idx + 1) as u64);
        }
    }

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    // Recompute RoPE tables (deterministic, fast)
    let (cos_cache, sin_cache, inv_freq, mrope_pattern) =
        if let Some(section) = config.mrope_section {
            let inv_freq =
                crate::mrope::compute_inv_freq(config.head_dim, config.rope_theta);
            let pattern = crate::mrope::build_mrope_pattern(section);
            (Vec::new(), Vec::new(), inv_freq, pattern)
        } else {
            let (cos, sin) = compute_rope_cache(config);
            (cos, sin, Vec::new(), Vec::new())
        };

    // Gemma3 VL: dual RoPE caches
    let (cos_cache, sin_cache, sliding_cos_cache, sliding_sin_cache) = if let Some(sliding_theta) = config.sliding_rope_theta {
        let factor = config.gemma3_rope_factor.unwrap_or(8.0);
        let (main_cos, main_sin) = crate::loader_common::compute_linear_rope_cache(config, factor);
        let mut sliding_config = config.clone();
        sliding_config.rope_theta = sliding_theta;
        let (sl_cos, sl_sin) = crate::loader_common::compute_standard_rope_cache(&sliding_config);
        (main_cos, main_sin, sl_cos, sl_sin)
    } else {
        (cos_cache, sin_cache, Vec::new(), Vec::new())
    };

    Ok(GenericModel {
        embed_tokens,
        decoder_layers,
        norm,
        norm_bias: None,
        lm_head,
        lm_head_bias: None,
        config: config.clone(),
        cos_cache,
        sin_cache,
        sliding_cos_cache,
        sliding_sin_cache,
        inv_freq,
        mrope_pattern,
        _marker: PhantomData,
    })
}
