//! SafeTensors weight loading into Metal GPU buffers for Qwen3.
//!
//! Streams safetensors shards sequentially, converts BF16 weights to f32,
//! quantizes projection matrices to Int8, BF16, or Q4 format, and copies all
//! buffers into Metal shared memory (unified CPU/GPU on Apple Silicon).

use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use crate::memory::MetalBuffer;
use crate::model::{
    MetalBF16Weight, MetalInt8Weight, MetalQ4Weight, MetalLayer, MetalLayerMLP,
    MetalMoEContiguous, MoEQuantFormat, MetalModel, MetalWeight,
};

use herbert_backend_common::loader_common::{
    classify_tensor, compute_rope_cache, dequant_fp8_to_f32, load_f32_from_view,
    normalize_shard_name, AttnNormWhich, AttnProjWhich, MlpProjWhich, NormWhich,
    SafetensorsIndex, TensorKind,
};
use herbert_core::config::Config;
use herbert_core::error::{HerbertError, Result};
use herbert_core::tensor::f32_to_bf16;

/// Quantization mode for the Metal GPU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    Int8,
    BF16,
    Q4,
}

// ============================================================================
// Int8 quantization (row-major, per-channel, for Metal GPU shaders)
// ============================================================================

pub fn quantize_int8_rowmajor(f32_data: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(f32_data.len(), n * k, "quantize_int8_rowmajor: data length mismatch");
    assert_eq!(k % 4, 0, "quantize_int8_rowmajor: K must be divisible by 4");

    let mut packed = vec![0u8; n * k];
    let mut scales = vec![0.0f32; n];

    for row in 0..n {
        let row_offset = row * k;
        let row_data = &f32_data[row_offset..row_offset + k];

        let absmax = row_data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if absmax == 0.0 { 0.0 } else { absmax / 127.0 };
        scales[row] = scale;

        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        for j in 0..k {
            let q = (row_data[j] * inv_scale).round().clamp(-127.0, 127.0) as i8;
            packed[row_offset + j] = q as u8;
        }
    }

    (packed, scales)
}

fn make_int8_weight(
    device: &ProtocolObject<dyn MTLDevice>,
    f32_data: &[f32],
    n: usize,
    k: usize,
) -> Result<MetalInt8Weight> {
    let (packed, scales) = quantize_int8_rowmajor(f32_data, n, k);

    let packed_buf = MetalBuffer::from_data(device, &packed)?;

    let scales_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4) };
    let scales_buf = MetalBuffer::from_data(device, scales_bytes)?;

    Ok(MetalInt8Weight {
        packed: packed_buf,
        scales: scales_buf,
        n,
        k,
    })
}

// ============================================================================
// BF16 conversion
// ============================================================================

fn pack_f32_to_bf16_bytes(f32_data: &[f32]) -> Vec<u8> {
    let bf16_vals: Vec<u16> = f32_data.iter().map(|&v| f32_to_bf16(v)).collect();
    bf16_vals.iter().flat_map(|&v| v.to_le_bytes()).collect()
}

fn make_bf16_weight(
    device: &ProtocolObject<dyn MTLDevice>,
    f32_data: &[f32],
    n: usize,
    k: usize,
) -> Result<MetalBF16Weight> {
    assert_eq!(f32_data.len(), n * k, "make_bf16_weight: data length mismatch");
    assert_eq!(k % 2, 0, "make_bf16_weight: K must be even");

    let packed_bytes = pack_f32_to_bf16_bytes(f32_data);
    let packed = MetalBuffer::from_data(device, &packed_bytes)?;

    Ok(MetalBF16Weight { packed, n, k })
}

// ============================================================================
// Q4 quantization (row-major, symmetric, group_size=32)
// ============================================================================

/// Quantize an f32 matrix [N, K] to Q4 format.
///
/// Returns (packed_bytes, scales) where:
/// - packed_bytes: [N, K/2] bytes, each byte = nib_k0 | (nib_k1 << 4)
/// - scales: [N, n_groups] f32, n_groups = ceil(K/32)
///
/// Nibble values: unsigned [0,15] representing symmetric Q4 (q+8)
/// Dequant: value = (float(nibble) - 8.0) * scales[row * n_groups + col/32]
pub fn quantize_q4_rowmajor(f32_data: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(f32_data.len(), n * k, "quantize_q4_rowmajor: data length mismatch");
    assert_eq!(k % 2, 0, "quantize_q4_rowmajor: K must be even");

    let group_size = 32usize;
    let n_groups = (k + group_size - 1) / group_size;
    let mut packed = vec![0u8; n * (k / 2)];
    let mut scales = vec![0.0f32; n * n_groups];

    for row in 0..n {
        let row_offset = row * k;
        let row_data = &f32_data[row_offset..row_offset + k];

        // Compute per-group scales
        for g in 0..n_groups {
            let start = g * group_size;
            let end = (start + group_size).min(k);
            let absmax = row_data[start..end].iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let scale = if absmax == 0.0 { 0.0 } else { absmax / 7.0 };
            scales[row * n_groups + g] = scale;
        }

        // Quantize to 4-bit nibbles and pack pairs
        let pack_offset = row * (k / 2);
        for j in (0..k).step_by(2) {
            let g0 = j / group_size;
            let g1 = (j + 1) / group_size;
            let inv_s0 = if scales[row * n_groups + g0] == 0.0 { 0.0 } else { 1.0 / scales[row * n_groups + g0] };
            let inv_s1 = if scales[row * n_groups + g1] == 0.0 { 0.0 } else { 1.0 / scales[row * n_groups + g1] };

            let q0 = (row_data[j] * inv_s0).round().clamp(-8.0, 7.0) as i8;
            let q1 = (row_data[j + 1] * inv_s1).round().clamp(-8.0, 7.0) as i8;

            let nib0 = (q0 + 8) as u8;  // [0, 15]
            let nib1 = (q1 + 8) as u8;  // [0, 15]

            packed[pack_offset + j / 2] = nib0 | (nib1 << 4);
        }
    }

    (packed, scales)
}

fn make_q4_weight(
    device: &ProtocolObject<dyn MTLDevice>,
    f32_data: &[f32],
    n: usize,
    k: usize,
) -> Result<MetalQ4Weight> {
    let (packed, scales) = quantize_q4_rowmajor(f32_data, n, k);

    let packed_buf = MetalBuffer::from_data(device, &packed)?;

    let scales_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4) };
    let scales_buf = MetalBuffer::from_data(device, scales_bytes)?;

    Ok(MetalQ4Weight {
        packed: packed_buf,
        scales: scales_buf,
        n,
        k,
    })
}

/// Create a MetalWeight from f32 data using the selected quantization mode.
fn make_weight(
    device: &ProtocolObject<dyn MTLDevice>,
    f32_data: &[f32],
    n: usize,
    k: usize,
    mode: QuantMode,
) -> Result<MetalWeight> {
    match mode {
        QuantMode::Int8 => Ok(MetalWeight::Int8(make_int8_weight(device, f32_data, n, k)?)),
        QuantMode::BF16 => Ok(MetalWeight::BF16(make_bf16_weight(device, f32_data, n, k)?)),
        QuantMode::Q4 => Ok(MetalWeight::Q4(make_q4_weight(device, f32_data, n, k)?)),
    }
}

fn upload_f32(device: &ProtocolObject<dyn MTLDevice>, data: &[f32]) -> Result<MetalBuffer> {
    MetalBuffer::from_f32(device, data)
}

// ============================================================================
// Partial layer accumulator structures
// ============================================================================

/// CPU-side quantized data for a single expert projection, awaiting contiguous upload.
struct ExpertQuantData {
    packed: Vec<u8>,
    scales: Vec<u8>, // f32 as bytes; empty for BF16
}

/// Quantize f32 data to the selected format, returning CPU-side byte arrays.
fn quantize_expert_data(f32_data: &[f32], n: usize, k: usize, mode: QuantMode) -> ExpertQuantData {
    match mode {
        QuantMode::Q4 => {
            let (packed, scales) = quantize_q4_rowmajor(f32_data, n, k);
            let scales_bytes = unsafe {
                std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4)
            };
            ExpertQuantData { packed, scales: scales_bytes.to_vec() }
        }
        QuantMode::Int8 => {
            let (packed, scales) = quantize_int8_rowmajor(f32_data, n, k);
            let scales_bytes = unsafe {
                std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4)
            };
            ExpertQuantData { packed, scales: scales_bytes.to_vec() }
        }
        QuantMode::BF16 => {
            let packed = pack_f32_to_bf16_bytes(f32_data);
            ExpertQuantData { packed, scales: Vec::new() }
        }
    }
}

struct PartialExpert {
    gate_proj: Option<ExpertQuantData>,
    up_proj: Option<ExpertQuantData>,
    down_proj: Option<ExpertQuantData>,
}

struct PartialLayer {
    input_layernorm: Option<MetalBuffer>,
    post_attention_layernorm: Option<MetalBuffer>,
    q_proj: Option<MetalWeight>,
    k_proj: Option<MetalWeight>,
    v_proj: Option<MetalWeight>,
    o_proj: Option<MetalWeight>,
    q_norm: Option<MetalBuffer>,
    k_norm: Option<MetalBuffer>,
    // Dense MLP fields
    gate_proj: Option<MetalWeight>,
    up_proj: Option<MetalWeight>,
    down_proj: Option<MetalWeight>,
    // MoE fields
    moe_router: Option<MetalBuffer>,
    experts: Vec<PartialExpert>,
}

fn new_partial_layer(num_experts_total: usize) -> PartialLayer {
    PartialLayer {
        input_layernorm: None,
        post_attention_layernorm: None,
        q_proj: None,
        k_proj: None,
        v_proj: None,
        o_proj: None,
        q_norm: None,
        k_norm: None,
        gate_proj: None,
        up_proj: None,
        down_proj: None,
        moe_router: None,
        experts: (0..num_experts_total)
            .map(|_| PartialExpert {
                gate_proj: None,
                up_proj: None,
                down_proj: None,
            })
            .collect(),
    }
}

fn finalize_layer(
    pl: PartialLayer,
    i: usize,
    config: &Config,
    device: &ProtocolObject<dyn MTLDevice>,
    quant_mode: QuantMode,
) -> Result<MetalLayer> {
    let mlp = if config.is_moe_layer(i) {
        let router = pl.moe_router.ok_or_else(|| {
            HerbertError::ModelLoad(format!("layer {}: missing MoE router gate", i))
        })?;
        let ne = config.num_experts.unwrap();
        let nept = config.num_experts_per_tok.unwrap();
        let moe_inter = config.moe_intermediate_size.unwrap();
        let hidden_size = config.hidden_size;
        let norm_topk = config.norm_topk_prob;

        // Concatenate all experts' quantized data into contiguous arrays
        let mut gate_packed_all = Vec::new();
        let mut gate_scales_all = Vec::new();
        let mut up_packed_all = Vec::new();
        let mut up_scales_all = Vec::new();
        let mut down_packed_all = Vec::new();
        let mut down_scales_all = Vec::new();

        for (e, pe) in pl.experts.into_iter().enumerate() {
            let gp = pe.gate_proj.ok_or_else(|| {
                HerbertError::ModelLoad(format!("layer {}: expert {}: missing gate_proj", i, e))
            })?;
            gate_packed_all.extend_from_slice(&gp.packed);
            gate_scales_all.extend_from_slice(&gp.scales);

            let up = pe.up_proj.ok_or_else(|| {
                HerbertError::ModelLoad(format!("layer {}: expert {}: missing up_proj", i, e))
            })?;
            up_packed_all.extend_from_slice(&up.packed);
            up_scales_all.extend_from_slice(&up.scales);

            let dp = pe.down_proj.ok_or_else(|| {
                HerbertError::ModelLoad(format!("layer {}: expert {}: missing down_proj", i, e))
            })?;
            down_packed_all.extend_from_slice(&dp.packed);
            down_scales_all.extend_from_slice(&dp.scales);
        }

        let format = match quant_mode {
            QuantMode::Q4 => MoEQuantFormat::Q4,
            QuantMode::Int8 => MoEQuantFormat::Int8,
            QuantMode::BF16 => MoEQuantFormat::BF16,
        };

        let weights = MetalMoEContiguous {
            gate_packed: MetalBuffer::from_data(device, &gate_packed_all)?,
            gate_scales: if gate_scales_all.is_empty() { None }
                         else { Some(MetalBuffer::from_data(device, &gate_scales_all)?) },
            up_packed: MetalBuffer::from_data(device, &up_packed_all)?,
            up_scales: if up_scales_all.is_empty() { None }
                       else { Some(MetalBuffer::from_data(device, &up_scales_all)?) },
            down_packed: MetalBuffer::from_data(device, &down_packed_all)?,
            down_scales: if down_scales_all.is_empty() { None }
                         else { Some(MetalBuffer::from_data(device, &down_scales_all)?) },
            format,
            gate_n: moe_inter,
            gate_k: hidden_size,
            down_n: hidden_size,
            down_k: moe_inter,
        };

        if i == 0 {
            let gate_mb = gate_packed_all.len() as f64 / 1_048_576.0;
            let total_mb = (gate_packed_all.len() + gate_scales_all.len()
                + up_packed_all.len() + up_scales_all.len()
                + down_packed_all.len() + down_scales_all.len()) as f64 / 1_048_576.0;
            eprintln!(
                "[metal] MoE contiguous: {} experts, gate_packed={:.1} MB, total={:.1} MB/layer",
                ne, gate_mb, total_mb,
            );
        }

        MetalLayerMLP::MoE {
            router,
            weights,
            num_experts: ne,
            num_experts_per_tok: nept,
            moe_intermediate_size: moe_inter,
            norm_topk_prob: norm_topk,
        }
    } else {
        MetalLayerMLP::Dense {
            gate_proj: pl.gate_proj.ok_or_else(|| {
                HerbertError::ModelLoad(format!("layer {}: missing gate_proj", i))
            })?,
            up_proj: pl.up_proj.ok_or_else(|| {
                HerbertError::ModelLoad(format!("layer {}: missing up_proj", i))
            })?,
            down_proj: pl.down_proj.ok_or_else(|| {
                HerbertError::ModelLoad(format!("layer {}: missing down_proj", i))
            })?,
        }
    };

    Ok(MetalLayer {
        input_layernorm: pl.input_layernorm.ok_or_else(|| {
            HerbertError::ModelLoad(format!("layer {}: missing input_layernorm", i))
        })?,
        post_attention_layernorm: pl.post_attention_layernorm.ok_or_else(|| {
            HerbertError::ModelLoad(format!("layer {}: missing post_attention_layernorm", i))
        })?,
        q_proj: pl.q_proj.ok_or_else(|| {
            HerbertError::ModelLoad(format!("layer {}: missing q_proj", i))
        })?,
        k_proj: pl.k_proj.ok_or_else(|| {
            HerbertError::ModelLoad(format!("layer {}: missing k_proj", i))
        })?,
        v_proj: pl.v_proj.ok_or_else(|| {
            HerbertError::ModelLoad(format!("layer {}: missing v_proj", i))
        })?,
        o_proj: pl.o_proj.ok_or_else(|| {
            HerbertError::ModelLoad(format!("layer {}: missing o_proj", i))
        })?,
        q_norm: if config.has_qk_norm {
            Some(pl.q_norm.ok_or_else(|| {
                HerbertError::ModelLoad(format!("layer {}: missing q_norm", i))
            })?)
        } else {
            pl.q_norm
        },
        k_norm: if config.has_qk_norm {
            Some(pl.k_norm.ok_or_else(|| {
                HerbertError::ModelLoad(format!("layer {}: missing k_norm", i))
            })?)
        } else {
            pl.k_norm
        },
        mlp,
    })
}

// ============================================================================
// Model loader
// ============================================================================

pub fn load_model(
    model_dir: &Path,
    device: &ProtocolObject<dyn MTLDevice>,
    max_tokens: usize,
    quant_mode: QuantMode,
) -> Result<(Config, MetalModel)> {
    let config = Config::from_file(&model_dir.join("config.json"))?;

    let num_layers = config.num_layers;
    let hidden_size = config.hidden_size;

    // Determine shard list
    let index_path = model_dir.join("model.safetensors.index.json");
    let (shard_names, weight_to_shard) = if index_path.exists() {
        let index_data = fs::read(&index_path)?;
        let index: SafetensorsIndex = serde_json::from_slice(&index_data)?;
        let mut shard_set: HashSet<String> = HashSet::new();
        for shard_file in index.weight_map.values() {
            shard_set.insert(normalize_shard_name(shard_file).to_string());
        }
        let mut shard_names: Vec<String> = shard_set.into_iter().collect();
        shard_names.sort();
        let mut w2s = HashMap::new();
        for (weight, shard_file) in &index.weight_map {
            w2s.insert(weight.clone(), normalize_shard_name(shard_file).to_string());
        }
        (shard_names, Some(w2s))
    } else {
        (vec!["model.safetensors".to_string()], None)
    };

    // Detect FP8 block quantization from config.json (weight_block_size field)
    let fp8_block_size: Option<[usize; 2]> = {
        let raw: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(model_dir.join("config.json"))
                .unwrap_or_default(),
        ).unwrap_or_default();
        let wbs = raw.get("weight_block_size")
            .or_else(|| raw.get("text_config").and_then(|tc| tc.get("weight_block_size")))
            .or_else(|| raw.get("quantization_config").and_then(|qc| qc.get("weight_block_size")));
        match wbs {
            Some(serde_json::Value::Array(arr)) if arr.len() == 2 => {
                let r = arr[0].as_u64().unwrap_or(128) as usize;
                let c = arr[1].as_u64().unwrap_or(128) as usize;
                Some([r, c])
            }
            _ => None,
        }
    };

    if let Some(bs) = fp8_block_size {
        eprintln!("[metal] FP8 model detected: weight_block_size={:?}", bs);
    }

    eprintln!(
        "[metal] Loading model: {} layers, hidden_size={}, shards={}, quant={:?}",
        num_layers, hidden_size, shard_names.len(), quant_mode
    );

    // Accumulators
    let mut embed_tokens: Option<MetalBuffer> = None;
    let mut embed_tokens_f32: Option<Vec<f32>> = None;
    let mut final_norm: Option<MetalBuffer> = None;
    let mut lm_head: Option<MetalWeight> = None;

    let num_experts_total = config.num_experts.unwrap_or(0);

    let mut partial_layers: Vec<PartialLayer> = (0..num_layers)
        .map(|_| new_partial_layer(num_experts_total))
        .collect();

    let mut layers_seen = vec![false; num_layers];

    // Stream shards
    for shard_name in &shard_names {
        let shard_path = model_dir.join(shard_name);
        let shard_file = fs::File::open(&shard_path).map_err(|e| {
            HerbertError::ModelLoad(format!("Failed to open shard {}: {}", shard_name, e))
        })?;
        let shard_mmap = unsafe { Mmap::map(&shard_file) }.map_err(|e| {
            HerbertError::ModelLoad(format!("Failed to mmap shard {}: {}", shard_name, e))
        })?;
        let tensors = SafeTensors::deserialize(&shard_mmap[..]).map_err(|e| {
            HerbertError::ModelLoad(format!("Failed to parse shard {}: {}", shard_name, e))
        })?;

        // FP8-aware f32 loader: dequantizes FP8 E4M3 tensors using companion _scale_inv
        let load_f32_auto = |view: &safetensors::tensor::TensorView<'_>, tensor_name: &str| -> Result<Vec<f32>> {
            if view.dtype() == safetensors::Dtype::F8_E4M3 {
                let scale_name = format!("{}_scale_inv", tensor_name);
                let scale_view = tensors.tensor(&scale_name).map_err(|e| {
                    HerbertError::ModelLoad(format!(
                        "FP8 tensor {} requires companion {} but not found: {}",
                        tensor_name, scale_name, e
                    ))
                })?;
                let scale_data = load_f32_from_view(&scale_view, &scale_name)?;
                let shape = view.shape();
                let rows = shape[..shape.len() - 1].iter().product::<usize>().max(1);
                let cols = *shape.last().unwrap_or(&1);
                let bs = fp8_block_size.unwrap_or([rows, cols]);
                Ok(dequant_fp8_to_f32(view.data(), &scale_data, rows, cols, bs))
            } else {
                load_f32_from_view(view, tensor_name)
            }
        };

        let tensor_names: Vec<String> = if let Some(ref w2s) = weight_to_shard {
            w2s.iter()
                .filter(|(_, s)| s.as_str() == shard_name.as_str())
                .map(|(w, _)| w.clone())
                .collect()
        } else {
            tensors.names().into_iter().map(|s| s.to_string()).collect()
        };

        for name in &tensor_names {
            let view = tensors.tensor(name).map_err(|e| {
                HerbertError::ModelLoad(format!("Tensor {} not found: {}", name, e))
            })?;

            let kind = classify_tensor(name);

            match kind {
                TensorKind::Embedding => {
                    let f32_data = load_f32_auto(&view, name)?;
                    let buf = upload_f32(device, &f32_data)?;
                    eprintln!(
                        "[metal] embed_tokens: {}x{} f32 ({:.1} MB)",
                        config.vocab_size, hidden_size,
                        (f32_data.len() * 4) as f64 / 1_048_576.0
                    );
                    embed_tokens = Some(buf);
                    if config.tie_word_embeddings {
                        embed_tokens_f32 = Some(f32_data);
                    }
                }

                TensorKind::FinalNorm => {
                    let f32_data = load_f32_auto(&view, name)?;
                    final_norm = Some(upload_f32(device, &f32_data)?);
                }

                TensorKind::LmHead => {
                    let f32_data = load_f32_auto(&view, name)?;
                    lm_head = Some(make_weight(
                        device, &f32_data, config.vocab_size, hidden_size, quant_mode,
                    )?);
                }

                TensorKind::LayerNorm { layer, which } => {
                    if !layers_seen[layer] {
                        layers_seen[layer] = true;
                        eprintln!("[metal] Loading layer {}/{}", layer + 1, num_layers);
                    }
                    let f32_data = load_f32_auto(&view, name)?;
                    let buf = upload_f32(device, &f32_data)?;
                    let pl = &mut partial_layers[layer];
                    match which {
                        NormWhich::Input => pl.input_layernorm = Some(buf),
                        NormWhich::PostAttention => pl.post_attention_layernorm = Some(buf),
                        _ => {} // Gemma3-only norms, skip for Qwen3
                    }
                }

                TensorKind::LayerAttnProj { layer, which } => {
                    if !layers_seen[layer] {
                        layers_seen[layer] = true;
                        eprintln!("[metal] Loading layer {}/{}", layer + 1, num_layers);
                    }
                    let f32_data = load_f32_auto(&view, name)?;
                    let (out_f, in_f) = match which {
                        AttnProjWhich::Q => (config.q_dim(), hidden_size),
                        AttnProjWhich::K => (config.kv_dim(), hidden_size),
                        AttnProjWhich::V => (config.kv_dim(), hidden_size),
                        AttnProjWhich::O => (hidden_size, config.q_dim()),
                    };
                    let w = make_weight(device, &f32_data, out_f, in_f, quant_mode)?;
                    let pl = &mut partial_layers[layer];
                    match which {
                        AttnProjWhich::Q => pl.q_proj = Some(w),
                        AttnProjWhich::K => pl.k_proj = Some(w),
                        AttnProjWhich::V => pl.v_proj = Some(w),
                        AttnProjWhich::O => pl.o_proj = Some(w),
                    }
                }

                TensorKind::LayerAttnNorm { layer, which } => {
                    if !layers_seen[layer] {
                        layers_seen[layer] = true;
                        eprintln!("[metal] Loading layer {}/{}", layer + 1, num_layers);
                    }
                    let f32_data = load_f32_auto(&view, name)?;
                    let buf = upload_f32(device, &f32_data)?;
                    let pl = &mut partial_layers[layer];
                    match which {
                        AttnNormWhich::Q => pl.q_norm = Some(buf),
                        AttnNormWhich::K => pl.k_norm = Some(buf),
                    }
                }

                TensorKind::LayerMlpProj { layer, which } => {
                    if !layers_seen[layer] {
                        layers_seen[layer] = true;
                        eprintln!("[metal] Loading layer {}/{}", layer + 1, num_layers);
                    }
                    let f32_data = load_f32_auto(&view, name)?;
                    let (out_f, in_f) = match which {
                        MlpProjWhich::Gate => (config.intermediate_size, hidden_size),
                        MlpProjWhich::Up => (config.intermediate_size, hidden_size),
                        MlpProjWhich::Down => (hidden_size, config.intermediate_size),
                    };
                    let w = make_weight(device, &f32_data, out_f, in_f, quant_mode)?;
                    let pl = &mut partial_layers[layer];
                    match which {
                        MlpProjWhich::Gate => pl.gate_proj = Some(w),
                        MlpProjWhich::Up => pl.up_proj = Some(w),
                        MlpProjWhich::Down => pl.down_proj = Some(w),
                    }
                }

                TensorKind::LayerMoEGate { layer } => {
                    if !layers_seen[layer] {
                        layers_seen[layer] = true;
                        eprintln!("[metal] Loading layer {}/{}", layer + 1, num_layers);
                    }
                    let f32_data = load_f32_auto(&view, name)?;
                    let buf = upload_f32(device, &f32_data)?;
                    partial_layers[layer].moe_router = Some(buf);
                }

                TensorKind::LayerExpertProj { layer, expert, which } => {
                    if !layers_seen[layer] {
                        layers_seen[layer] = true;
                        eprintln!("[metal] Loading layer {}/{}", layer + 1, num_layers);
                    }
                    let moe_inter = config.moe_intermediate_size.unwrap_or(0);
                    let f32_data = load_f32_auto(&view, name)?;
                    let (out_f, in_f) = match which {
                        MlpProjWhich::Gate => (moe_inter, hidden_size),
                        MlpProjWhich::Up => (moe_inter, hidden_size),
                        MlpProjWhich::Down => (hidden_size, moe_inter),
                    };
                    let qd = quantize_expert_data(&f32_data, out_f, in_f, quant_mode);
                    let pl = &mut partial_layers[layer];
                    if expert >= pl.experts.len() {
                        return Err(HerbertError::ModelLoad(format!(
                            "layer {}: expert {} exceeds num_experts {}",
                            layer, expert, pl.experts.len()
                        )));
                    }
                    match which {
                        MlpProjWhich::Gate => pl.experts[expert].gate_proj = Some(qd),
                        MlpProjWhich::Up => pl.experts[expert].up_proj = Some(qd),
                        MlpProjWhich::Down => pl.experts[expert].down_proj = Some(qd),
                    }
                }

                TensorKind::FusedExpertGateUp { layer } => {
                    if !layers_seen[layer] {
                        layers_seen[layer] = true;
                        eprintln!("[metal] Loading layer {}/{}", layer + 1, num_layers);
                    }
                    let moe_inter = config.moe_intermediate_size.unwrap_or(0);
                    let shape = view.shape();
                    let num_experts_in_tensor = shape[0];
                    let expected_double_inter = moe_inter * 2;
                    let transposed = shape[1] != expected_double_inter && shape[2] == expected_double_inter;
                    let (double_inter, in_features) = if transposed {
                        (shape[2], shape[1])
                    } else {
                        (shape[1], shape[2])
                    };
                    let moe_inter_actual = double_inter / 2;
                    let f32_data = load_f32_auto(&view, name)?;

                    let pl = &mut partial_layers[layer];
                    for e in 0..num_experts_in_tensor {
                        if e >= pl.experts.len() {
                            return Err(HerbertError::ModelLoad(format!(
                                "layer {}: fused expert {} exceeds num_experts {}",
                                layer, e, pl.experts.len()
                            )));
                        }
                        let expert_offset = e * double_inter * in_features;
                        if transposed {
                            let expert_data = &f32_data[expert_offset..expert_offset + in_features * double_inter];
                            let mut gate_t = vec![0.0f32; moe_inter_actual * in_features];
                            let mut up_t = vec![0.0f32; moe_inter_actual * in_features];
                            for row in 0..in_features {
                                for col in 0..moe_inter_actual {
                                    gate_t[col * in_features + row] = expert_data[row * double_inter + col];
                                }
                                for col in 0..moe_inter_actual {
                                    up_t[col * in_features + row] = expert_data[row * double_inter + moe_inter_actual + col];
                                }
                            }
                            pl.experts[e].gate_proj = Some(quantize_expert_data(&gate_t, moe_inter_actual, in_features, quant_mode));
                            pl.experts[e].up_proj = Some(quantize_expert_data(&up_t, moe_inter_actual, in_features, quant_mode));
                        } else {
                            let gate_data = &f32_data[expert_offset..expert_offset + moe_inter_actual * in_features];
                            let up_data = &f32_data[expert_offset + moe_inter_actual * in_features..expert_offset + double_inter * in_features];
                            pl.experts[e].gate_proj = Some(quantize_expert_data(gate_data, moe_inter_actual, in_features, quant_mode));
                            pl.experts[e].up_proj = Some(quantize_expert_data(up_data, moe_inter_actual, in_features, quant_mode));
                        }
                    }
                }

                TensorKind::FusedExpertDown { layer } => {
                    if !layers_seen[layer] {
                        layers_seen[layer] = true;
                        eprintln!("[metal] Loading layer {}/{}", layer + 1, num_layers);
                    }
                    let shape = view.shape();
                    let num_experts_in_tensor = shape[0];
                    let transposed = shape[1] != hidden_size && shape[2] == hidden_size;
                    let (out_features, in_features) = if transposed {
                        (shape[2], shape[1])
                    } else {
                        (shape[1], shape[2])
                    };
                    let f32_data = load_f32_auto(&view, name)?;

                    let pl = &mut partial_layers[layer];
                    for e in 0..num_experts_in_tensor {
                        if e >= pl.experts.len() {
                            return Err(HerbertError::ModelLoad(format!(
                                "layer {}: fused expert {} exceeds num_experts {}",
                                layer, e, pl.experts.len()
                            )));
                        }
                        let expert_offset = e * out_features * in_features;
                        if transposed {
                            let expert_data = &f32_data[expert_offset..expert_offset + in_features * out_features];
                            let mut down_t = vec![0.0f32; out_features * in_features];
                            for row in 0..in_features {
                                for col in 0..out_features {
                                    down_t[col * in_features + row] = expert_data[row * out_features + col];
                                }
                            }
                            pl.experts[e].down_proj = Some(quantize_expert_data(&down_t, out_features, in_features, quant_mode));
                        } else {
                            let expert_data = &f32_data[expert_offset..expert_offset + out_features * in_features];
                            pl.experts[e].down_proj = Some(quantize_expert_data(expert_data, out_features, in_features, quant_mode));
                        }
                    }
                }

                // Skip unsupported tensor kinds
                TensorKind::FinalNormBias
                | TensorKind::LmHeadBias
                | TensorKind::LayerNormBias { .. }
                | TensorKind::LayerAttnBias { .. } => {}

                _ => {
                    // Silently skip unknown tensors
                }
            }
        }
    }

    // Handle tied embeddings for lm_head
    if lm_head.is_none() {
        if config.tie_word_embeddings {
            let f32_data = embed_tokens_f32.as_ref().ok_or_else(|| {
                HerbertError::ModelLoad(
                    "tie_word_embeddings is true but embed_tokens not loaded".into(),
                )
            })?;
            lm_head = Some(make_weight(
                device, f32_data, config.vocab_size, hidden_size, quant_mode,
            )?);
            eprintln!("[metal] lm_head: tied to embed_tokens ({:?})", quant_mode);
        } else {
            return Err(HerbertError::ModelLoad(
                "missing lm_head.weight and tie_word_embeddings is false".into(),
            ));
        }
    }

    // Finalize layers
    let mut layers = Vec::with_capacity(num_layers);
    for (i, pl) in partial_layers.into_iter().enumerate() {
        layers.push(finalize_layer(pl, i, &config, device, quant_mode)?);
    }

    // RoPE cache
    let max_seq_len = config.max_position_embeddings.min(max_tokens);
    let rope_config = Config {
        max_position_embeddings: max_seq_len,
        ..config.clone()
    };
    let (cos_data, sin_data) = compute_rope_cache(&rope_config);

    let cos_cache = upload_f32(device, &cos_data)?;
    let sin_cache = upload_f32(device, &sin_data)?;

    let half_dim = config.rotary_ndims / 2;
    eprintln!(
        "[metal] RoPE cache: {} positions x {} half_dim ({:.1} MB)",
        max_seq_len, half_dim,
        (cos_data.len() * 4 * 2) as f64 / 1_048_576.0
    );

    // Dummy norm buffer (placeholder for skip_norm path in Mistral3)
    let dummy_norm_buf = upload_f32(device, &[0.0f32])?;

    // Build final model
    let model = MetalModel {
        embed_tokens: embed_tokens
            .ok_or_else(|| HerbertError::ModelLoad("missing embed_tokens".into()))?,
        embed_tokens_f32,
        layers,
        final_norm: final_norm
            .ok_or_else(|| HerbertError::ModelLoad("missing model.norm.weight".into()))?,
        lm_head: lm_head.unwrap(),
        cos_cache,
        sin_cache,
        dummy_norm_buf,
    };

    eprintln!(
        "[metal] Model loaded: {} layers on GPU (unified memory)",
        num_layers,
    );

    Ok((config, model))
}
