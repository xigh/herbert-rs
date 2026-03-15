//! SafeTensors weight loading, quantization, and upload to GPU device memory for Qwen3.

use ash::vk;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use crate::context::VulkanContext;
use crate::memory::VulkanBuffer;
use crate::model::{
    VulkanBF16Weight, VulkanInt8Weight, VulkanQ4Weight, VulkanLayer, VulkanLayerMLP,
    VulkanModel, VulkanMoEExpert, VulkanWeight,
};

use herbert_backend_common::loader_common::{
    classify_tensor, compute_rope_cache, load_f32_from_view,
    normalize_shard_name, AttnNormWhich, AttnProjWhich, MlpProjWhich, NormWhich,
    SafetensorsIndex, TensorKind,
};
use herbert_core::config::Config;
use herbert_core::error::{HerbertError, Result};

/// Quantization mode for the Vulkan GPU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    Int8,
    BF16,
    Q4,
}

// ============================================================================
// Int8 quantization
// ============================================================================

pub fn quantize_int8_rowmajor(f32_data: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(f32_data.len(), n * k);
    assert_eq!(k % 4, 0);

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
    ctx: &VulkanContext,
    f32_data: &[f32],
    n: usize,
    k: usize,
) -> Result<VulkanInt8Weight> {
    let (packed, scales) = quantize_int8_rowmajor(f32_data, n, k);

    let packed_buf =
        VulkanBuffer::upload_to_device_local(ctx, &packed, vk::BufferUsageFlags::empty())?;

    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4)
    };
    let scales_buf =
        VulkanBuffer::upload_to_device_local(ctx, scales_bytes, vk::BufferUsageFlags::empty())?;

    Ok(VulkanInt8Weight {
        packed: packed_buf,
        scales: scales_buf,
        n,
        k,
    })
}

// ============================================================================
// BF16 conversion
// ============================================================================

fn f32_to_bf16(data: &[f32]) -> Vec<u16> {
    data.iter()
        .map(|&v| (v.to_bits() >> 16) as u16)
        .collect()
}

fn make_bf16_weight(
    ctx: &VulkanContext,
    f32_data: &[f32],
    n: usize,
    k: usize,
) -> Result<VulkanBF16Weight> {
    assert_eq!(f32_data.len(), n * k);
    assert_eq!(k % 2, 0);

    let bf16_data = f32_to_bf16(f32_data);
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(bf16_data.as_ptr() as *const u8, bf16_data.len() * 2)
    };
    let packed_buf =
        VulkanBuffer::upload_to_device_local(ctx, bytes, vk::BufferUsageFlags::empty())?;

    Ok(VulkanBF16Weight {
        packed: packed_buf,
        n,
        k,
    })
}

// ============================================================================
// Q4 quantization (symmetric, group_size=32)
// ============================================================================

/// Quantize an f32 matrix [N, K] to Q4 format.
///
/// Returns (packed_bytes, scales) where:
/// - packed_bytes: N * K/8 uint32 words (8 nibbles per word)
///   Each nibble is unsigned [0,15] = (signed_q4 + 8)
/// - scales: [N * n_groups] f32, n_groups = ceil(K/32)
///
/// Dequant: value = (float(nibble) - 8.0) * scales[row * n_groups + col/32]
pub fn quantize_q4_rowmajor(f32_data: &[f32], n: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(f32_data.len(), n * k);
    assert_eq!(k % 8, 0, "K must be divisible by 8 for Q4 uint32 packing");

    let group_size = 32usize;
    let n_groups = (k + group_size - 1) / group_size;
    let k8 = k / 8;
    let mut packed = vec![0u32; n * k8];
    let mut scales = vec![0.0f32; n * n_groups];

    for row in 0..n {
        let row_data = &f32_data[row * k..(row + 1) * k];

        // Compute per-group scales
        for g in 0..n_groups {
            let start = g * group_size;
            let end = (start + group_size).min(k);
            let absmax = row_data[start..end].iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let scale = if absmax == 0.0 { 0.0 } else { absmax / 7.0 };
            scales[row * n_groups + g] = scale;
        }

        // Quantize and pack 8 nibbles per uint32
        for word_idx in 0..k8 {
            let mut word = 0u32;
            for nib in 0..8 {
                let col = word_idx * 8 + nib;
                if col >= k { break; }
                let g = col / group_size;
                let scale = scales[row * n_groups + g];
                let inv_s = if scale == 0.0 { 0.0 } else { 1.0 / scale };
                let q = (row_data[col] * inv_s).round().clamp(-8.0, 7.0) as i8;
                let nibble = (q + 8) as u8; // [0, 15]
                word |= (nibble as u32) << (nib * 4);
            }
            packed[row * k8 + word_idx] = word;
        }
    }

    (packed, scales)
}

fn make_q4_weight(
    ctx: &VulkanContext,
    f32_data: &[f32],
    n: usize,
    k: usize,
) -> Result<VulkanQ4Weight> {
    let (packed, scales) = quantize_q4_rowmajor(f32_data, n, k);

    let packed_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4)
    };
    let packed_buf =
        VulkanBuffer::upload_to_device_local(ctx, packed_bytes, vk::BufferUsageFlags::empty())?;

    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4)
    };
    let scales_buf =
        VulkanBuffer::upload_to_device_local(ctx, scales_bytes, vk::BufferUsageFlags::empty())?;

    Ok(VulkanQ4Weight {
        packed: packed_buf,
        scales: scales_buf,
        n,
        k,
    })
}

// ============================================================================
// Generic weight creation
// ============================================================================

fn make_weight(
    ctx: &VulkanContext,
    f32_data: &[f32],
    n: usize,
    k: usize,
    mode: QuantMode,
) -> Result<VulkanWeight> {
    match mode {
        QuantMode::Int8 => Ok(VulkanWeight::Int8(make_int8_weight(ctx, f32_data, n, k)?)),
        QuantMode::BF16 => Ok(VulkanWeight::BF16(make_bf16_weight(ctx, f32_data, n, k)?)),
        QuantMode::Q4 => Ok(VulkanWeight::Q4(make_q4_weight(ctx, f32_data, n, k)?)),
    }
}

fn upload_f32(ctx: &VulkanContext, data: &[f32]) -> Result<VulkanBuffer> {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    VulkanBuffer::upload_to_device_local(ctx, bytes, vk::BufferUsageFlags::empty())
}

// ============================================================================
// Partial layer accumulator
// ============================================================================

struct PartialExpert {
    gate_proj: Option<VulkanWeight>,
    up_proj: Option<VulkanWeight>,
    down_proj: Option<VulkanWeight>,
}

struct PartialLayer {
    input_layernorm: Option<VulkanBuffer>,
    post_attention_layernorm: Option<VulkanBuffer>,
    q_proj: Option<VulkanWeight>,
    k_proj: Option<VulkanWeight>,
    v_proj: Option<VulkanWeight>,
    o_proj: Option<VulkanWeight>,
    q_norm: Option<VulkanBuffer>,
    k_norm: Option<VulkanBuffer>,
    // Dense MLP
    gate_proj: Option<VulkanWeight>,
    up_proj: Option<VulkanWeight>,
    down_proj: Option<VulkanWeight>,
    // MoE
    moe_router: Option<VulkanBuffer>,
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

fn finalize_layer(pl: PartialLayer, config: &Config, layer_idx: usize) -> Result<VulkanLayer> {
    let mlp = if config.is_moe_layer(layer_idx) {
        let router = pl.moe_router.ok_or_else(|| {
            HerbertError::ModelLoad(format!("Layer {} missing MoE router", layer_idx))
        })?;
        let num_experts = config.num_experts.unwrap();
        let num_experts_per_tok = config.num_experts_per_tok.unwrap();
        let moe_intermediate_size = config.moe_intermediate_size.unwrap();
        let norm_topk_prob = config.norm_topk_prob;

        let mut experts = Vec::with_capacity(num_experts);
        for (e_idx, pe) in pl.experts.into_iter().enumerate() {
            experts.push(VulkanMoEExpert {
                gate_proj: pe.gate_proj.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("Layer {} expert {} missing gate_proj", layer_idx, e_idx))
                })?,
                up_proj: pe.up_proj.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("Layer {} expert {} missing up_proj", layer_idx, e_idx))
                })?,
                down_proj: pe.down_proj.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("Layer {} expert {} missing down_proj", layer_idx, e_idx))
                })?,
            });
        }

        VulkanLayerMLP::MoE {
            router,
            experts,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            norm_topk_prob,
        }
    } else {
        VulkanLayerMLP::Dense {
            gate_proj: pl.gate_proj.ok_or_else(|| {
                HerbertError::ModelLoad(format!("Layer {} missing gate_proj", layer_idx))
            })?,
            up_proj: pl.up_proj.ok_or_else(|| {
                HerbertError::ModelLoad(format!("Layer {} missing up_proj", layer_idx))
            })?,
            down_proj: pl.down_proj.ok_or_else(|| {
                HerbertError::ModelLoad(format!("Layer {} missing down_proj", layer_idx))
            })?,
        }
    };

    Ok(VulkanLayer {
        input_layernorm: pl.input_layernorm.ok_or_else(|| {
            HerbertError::ModelLoad(format!("Layer {} missing input_layernorm", layer_idx))
        })?,
        post_attention_layernorm: pl.post_attention_layernorm.ok_or_else(|| {
            HerbertError::ModelLoad(format!("Layer {} missing post_attention_layernorm", layer_idx))
        })?,
        q_proj: pl.q_proj.ok_or_else(|| {
            HerbertError::ModelLoad(format!("Layer {} missing q_proj", layer_idx))
        })?,
        k_proj: pl.k_proj.ok_or_else(|| {
            HerbertError::ModelLoad(format!("Layer {} missing k_proj", layer_idx))
        })?,
        v_proj: pl.v_proj.ok_or_else(|| {
            HerbertError::ModelLoad(format!("Layer {} missing v_proj", layer_idx))
        })?,
        o_proj: pl.o_proj.ok_or_else(|| {
            HerbertError::ModelLoad(format!("Layer {} missing o_proj", layer_idx))
        })?,
        q_norm: pl.q_norm.ok_or_else(|| {
            HerbertError::ModelLoad(format!("Layer {} missing q_norm", layer_idx))
        })?,
        k_norm: pl.k_norm.ok_or_else(|| {
            HerbertError::ModelLoad(format!("Layer {} missing k_norm", layer_idx))
        })?,
        mlp,
    })
}

// ============================================================================
// Model loader
// ============================================================================

pub fn load_model(
    model_dir: &Path,
    ctx: &VulkanContext,
    _max_tokens: usize,
    quant_mode: QuantMode,
) -> Result<(Config, VulkanModel)> {
    let config = Config::from_file(&model_dir.join("config.json"))?;

    let num_layers = config.num_layers;
    let hidden_size = config.hidden_size;
    let num_experts_total = config.num_experts.unwrap_or(0);

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
            w2s.insert(
                weight.clone(),
                normalize_shard_name(shard_file).to_string(),
            );
        }
        (shard_names, Some(w2s))
    } else {
        (vec!["model.safetensors".to_string()], None)
    };

    eprintln!(
        "[vulkan] Loading model: {} layers, hidden_size={}, shards={}, quant={:?}",
        num_layers, hidden_size, shard_names.len(), quant_mode
    );

    // Accumulators
    let mut embed_tokens: Option<VulkanBuffer> = None;
    let mut embed_tokens_f32: Option<Vec<f32>> = None;
    let mut final_norm: Option<VulkanBuffer> = None;
    let mut lm_head: Option<VulkanWeight> = None;

    let mut partial_layers: Vec<PartialLayer> = (0..num_layers)
        .map(|_| new_partial_layer(num_experts_total))
        .collect();

    // Stream shards
    for (shard_idx, shard_name) in shard_names.iter().enumerate() {
        let shard_path = model_dir.join(shard_name);
        eprintln!(
            "[vulkan] Loading shard {}/{}: {}",
            shard_idx + 1,
            shard_names.len(),
            shard_name
        );

        let file = fs::File::open(&shard_path)?;
        let mmap = unsafe { Mmap::map(&file) }?;
        let tensors = SafeTensors::deserialize(&mmap).map_err(|e| {
            HerbertError::ModelLoad(format!("Failed to parse {}: {:?}", shard_name, e))
        })?;

        // Collect tensor names for this shard
        let tensor_names: Vec<String> = if let Some(ref w2s) = weight_to_shard {
            w2s.iter()
                .filter(|(_, shard)| shard.as_str() == shard_name.as_str())
                .map(|(name, _)| name.clone())
                .collect()
        } else {
            tensors.names().into_iter().map(|s| s.to_string()).collect()
        };

        for name in &tensor_names {
            let view = match tensors.tensor(name) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let f32_data = match load_f32_from_view(&view, name) {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("[vulkan] Warning: skipping {}: {}", name, e);
                    continue;
                }
            };

            let kind = classify_tensor(name);

            match kind {
                TensorKind::Embedding => {
                    let n = config.vocab_size;
                    let k = hidden_size;
                    embed_tokens_f32 = Some(f32_data.clone());
                    embed_tokens = Some(upload_f32(ctx, &f32_data)?);
                    eprintln!("[vulkan]   embed_tokens [{}, {}]", n, k);
                }
                TensorKind::FinalNorm => {
                    final_norm = Some(upload_f32(ctx, &f32_data)?);
                }
                TensorKind::LmHead => {
                    let n = config.vocab_size;
                    let k = hidden_size;
                    lm_head = Some(make_weight(ctx, &f32_data, n, k, quant_mode)?);
                    eprintln!("[vulkan]   lm_head [{}, {}]", n, k);
                }
                TensorKind::LayerNorm { layer, which } => {
                    match which {
                        NormWhich::Input => {
                            partial_layers[layer].input_layernorm = Some(upload_f32(ctx, &f32_data)?);
                        }
                        NormWhich::PostAttention => {
                            partial_layers[layer].post_attention_layernorm = Some(upload_f32(ctx, &f32_data)?);
                        }
                        _ => {} // skip other norm types
                    }
                }
                TensorKind::LayerAttnProj { layer, which } => {
                    let (n, k) = match which {
                        AttnProjWhich::Q => (config.q_dim(), hidden_size),
                        AttnProjWhich::K => (config.kv_dim(), hidden_size),
                        AttnProjWhich::V => (config.kv_dim(), hidden_size),
                        AttnProjWhich::O => (hidden_size, config.q_dim()),
                    };
                    let w = make_weight(ctx, &f32_data, n, k, quant_mode)?;
                    match which {
                        AttnProjWhich::Q => partial_layers[layer].q_proj = Some(w),
                        AttnProjWhich::K => partial_layers[layer].k_proj = Some(w),
                        AttnProjWhich::V => partial_layers[layer].v_proj = Some(w),
                        AttnProjWhich::O => partial_layers[layer].o_proj = Some(w),
                    }
                }
                TensorKind::LayerAttnNorm { layer, which } => {
                    let buf = upload_f32(ctx, &f32_data)?;
                    match which {
                        AttnNormWhich::Q => partial_layers[layer].q_norm = Some(buf),
                        AttnNormWhich::K => partial_layers[layer].k_norm = Some(buf),
                    }
                }
                TensorKind::LayerMlpProj { layer, which } => {
                    let inter_size = if config.is_moe_layer(layer) {
                        config.moe_intermediate_size.unwrap()
                    } else {
                        config.intermediate_size
                    };
                    let (n, k) = match which {
                        MlpProjWhich::Gate => (inter_size, hidden_size),
                        MlpProjWhich::Up => (inter_size, hidden_size),
                        MlpProjWhich::Down => (hidden_size, inter_size),
                    };
                    let w = make_weight(ctx, &f32_data, n, k, quant_mode)?;
                    match which {
                        MlpProjWhich::Gate => partial_layers[layer].gate_proj = Some(w),
                        MlpProjWhich::Up => partial_layers[layer].up_proj = Some(w),
                        MlpProjWhich::Down => partial_layers[layer].down_proj = Some(w),
                    }
                }
                TensorKind::LayerMoEGate { layer } => {
                    partial_layers[layer].moe_router = Some(upload_f32(ctx, &f32_data)?);
                }
                TensorKind::LayerExpertProj { layer, expert, which } => {
                    let inter_size = config.moe_intermediate_size.unwrap();
                    let (n, k) = match which {
                        MlpProjWhich::Gate => (inter_size, hidden_size),
                        MlpProjWhich::Up => (inter_size, hidden_size),
                        MlpProjWhich::Down => (hidden_size, inter_size),
                    };
                    let w = make_weight(ctx, &f32_data, n, k, quant_mode)?;
                    let pe = &mut partial_layers[layer].experts[expert];
                    match which {
                        MlpProjWhich::Gate => pe.gate_proj = Some(w),
                        MlpProjWhich::Up => pe.up_proj = Some(w),
                        MlpProjWhich::Down => pe.down_proj = Some(w),
                    }
                }
                TensorKind::FusedExpertGateUp { layer } => {
                    // Shape: [num_experts, 2*moe_inter, hidden_size] (normal)
                    //    or: [num_experts, hidden_size, 2*moe_inter] (transposed)
                    let shape = view.shape();
                    let num_experts_in_tensor = shape[0];
                    let moe_inter = config.moe_intermediate_size.unwrap();
                    let expected_double_inter = moe_inter * 2;
                    let transposed = shape[1] != expected_double_inter && shape[2] == expected_double_inter;
                    let (double_inter, in_features) = if transposed {
                        (shape[2], shape[1])
                    } else {
                        (shape[1], shape[2])
                    };
                    let moe_inter_actual = double_inter / 2;
                    eprintln!("[vulkan]   fused gate_up layer {} [{}, {}, {}] transposed={}",
                        layer, num_experts_in_tensor, double_inter, in_features, transposed);

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
                            pl.experts[e].gate_proj = Some(make_weight(ctx, &gate_t, moe_inter_actual, in_features, quant_mode)?);
                            pl.experts[e].up_proj = Some(make_weight(ctx, &up_t, moe_inter_actual, in_features, quant_mode)?);
                        } else {
                            let gate_data = &f32_data[expert_offset..expert_offset + moe_inter_actual * in_features];
                            let up_data = &f32_data[expert_offset + moe_inter_actual * in_features..expert_offset + double_inter * in_features];
                            pl.experts[e].gate_proj = Some(make_weight(ctx, gate_data, moe_inter_actual, in_features, quant_mode)?);
                            pl.experts[e].up_proj = Some(make_weight(ctx, up_data, moe_inter_actual, in_features, quant_mode)?);
                        }
                    }
                }
                TensorKind::FusedExpertDown { layer } => {
                    // Shape: [num_experts, hidden_size, moe_inter] (normal)
                    //    or: [num_experts, moe_inter, hidden_size] (transposed)
                    let shape = view.shape();
                    let num_experts_in_tensor = shape[0];
                    let transposed = shape[1] != hidden_size && shape[2] == hidden_size;
                    let (out_features, in_features) = if transposed {
                        (shape[2], shape[1])
                    } else {
                        (shape[1], shape[2])
                    };
                    eprintln!("[vulkan]   fused down layer {} [{}, {}, {}] transposed={}",
                        layer, num_experts_in_tensor, out_features, in_features, transposed);

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
                            pl.experts[e].down_proj = Some(make_weight(ctx, &down_t, out_features, in_features, quant_mode)?);
                        } else {
                            let expert_data = &f32_data[expert_offset..expert_offset + out_features * in_features];
                            pl.experts[e].down_proj = Some(make_weight(ctx, expert_data, out_features, in_features, quant_mode)?);
                        }
                    }
                }
                _ => {
                    // Skip unrecognized tensors (visual encoder, etc.)
                }
            }
        }
    }

    // Finalize layers
    let mut layers = Vec::with_capacity(num_layers);
    for (i, pl) in partial_layers.into_iter().enumerate() {
        layers.push(finalize_layer(pl, &config, i)?);
    }

    // Handle tied lm_head (shared embed_tokens weights)
    if lm_head.is_none() {
        if let Some(ref f32_data) = embed_tokens_f32 {
            eprintln!("[vulkan] Using tied lm_head = embed_tokens");
            let n = config.vocab_size;
            let k = hidden_size;
            lm_head = Some(make_weight(ctx, f32_data, n, k, quant_mode)?);
        }
    }

    // RoPE cache
    let (cos_data, sin_data) = compute_rope_cache(&config);
    let cos_cache = upload_f32(ctx, &cos_data)?;
    let sin_cache = upload_f32(ctx, &sin_data)?;

    let model = VulkanModel {
        embed_tokens: embed_tokens.ok_or_else(|| {
            HerbertError::ModelLoad("Missing embed_tokens".into())
        })?,
        embed_tokens_f32,
        layers,
        final_norm: final_norm.ok_or_else(|| {
            HerbertError::ModelLoad("Missing final_norm".into())
        })?,
        lm_head: lm_head.ok_or_else(|| {
            HerbertError::ModelLoad("Missing lm_head".into())
        })?,
        cos_cache,
        sin_cache,
    };

    eprintln!("[vulkan] Model loaded successfully ({} layers)", num_layers);

    Ok((config, model))
}
