//! Shared loader utilities used by all CPU backend loaders.

use crate::expert_pool::ExpertPool;
use crate::generic_attention::GenericAttention;
use crate::generic_layer::{GenericConvolution, GenericDecoderLayer, GenericGatedDeltaNet, LayerBlock};
use crate::generic_mlp::{GenericMLP, MlpActivation};
use crate::generic_moe::{GenericFFN, GenericMoE};
use crate::generic_model::GenericModel;
use crate::linear_ops::LinearOps;
use crate::progress;
use herbert_core::config::{Config, RopeType};
use herbert_core::error::{HerbertError, Result};
use herbert_core::tensor::{bf16_to_f32, f32_to_bf16, fp8e4m3_to_f32, BF16};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

/// Deserialized safetensors index mapping weight names to shard filenames.
#[derive(serde::Deserialize)]
pub struct SafetensorsIndex {
    pub weight_map: HashMap<String, String>,
}

/// Extract the bare filename from a potentially path-qualified shard name.
pub fn normalize_shard_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Load a BF16 vector from a safetensors tensor view, converting F32 → BF16 if needed.
pub fn load_bf16_from_view(
    view: &safetensors::tensor::TensorView<'_>,
    name: &str,
) -> Result<Vec<BF16>> {
    let dtype = view.dtype();
    let bytes = view.data();

    match dtype {
        safetensors::Dtype::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect()),
        safetensors::Dtype::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|b| f32_to_bf16(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect()),
        _ => Err(HerbertError::ModelLoad(format!(
            "Unsupported dtype {:?} for {}",
            dtype, name
        ))),
    }
}

/// Load an F32 vector from a safetensors tensor view, converting BF16 → F32 if needed.
pub fn load_f32_from_view(
    view: &safetensors::tensor::TensorView<'_>,
    name: &str,
) -> Result<Vec<f32>> {
    let dtype = view.dtype();
    let bytes = view.data();

    match dtype {
        safetensors::Dtype::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        safetensors::Dtype::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect()),
        _ => Err(HerbertError::ModelLoad(format!(
            "Unsupported dtype {:?} for {}",
            dtype, name
        ))),
    }
}

/// Dequantize FP8 E4M3 data to BF16 using block-wise scale_inv factors.
///
/// `fp8_data`: raw FP8 bytes in row-major order, `[rows, cols]`
/// `scale_inv`: f32 scale factors in row-major order, `[ceil(rows/block_r), ceil(cols/block_c)]`
/// `rows`, `cols`: logical tensor shape
/// `block_size`: `[block_r, block_c]` — block dimensions for scale quantization
pub fn dequant_fp8_to_bf16(
    fp8_data: &[u8],
    scale_inv: &[f32],
    rows: usize,
    cols: usize,
    block_size: [usize; 2],
) -> Vec<BF16> {
    let [block_r, block_c] = block_size;
    let scale_cols = cols.div_ceil(block_c);
    let mut out = vec![0u16; rows * cols];
    for r in 0..rows {
        let sr = r / block_r;
        for c in 0..cols {
            let sc = c / block_c;
            let scale = scale_inv[sr * scale_cols + sc];
            let fp8_byte = fp8_data[r * cols + c];
            let val = fp8e4m3_to_f32(fp8_byte) * scale;
            out[r * cols + c] = f32_to_bf16(val);
        }
    }
    out
}

/// Dequantize FP8 E4M3 data directly to f32 using block-wise scale_inv factors.
///
/// Same as `dequant_fp8_to_bf16` but outputs f32 directly (avoids BF16 intermediate precision loss).
pub fn dequant_fp8_to_f32(
    fp8_data: &[u8],
    scale_inv: &[f32],
    rows: usize,
    cols: usize,
    block_size: [usize; 2],
) -> Vec<f32> {
    let [block_r, block_c] = block_size;
    let scale_cols = cols.div_ceil(block_c);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let sr = r / block_r;
        for c in 0..cols {
            let sc = c / block_c;
            let scale = scale_inv[sr * scale_cols + sc];
            let fp8_byte = fp8_data[r * cols + c];
            out[r * cols + c] = fp8e4m3_to_f32(fp8_byte) * scale;
        }
    }
    out
}

/// Load a BF16 vector from a safetensors tensor view, with optional FP8 dequantization.
///
/// If the tensor dtype is F8_E4M3, `scale_info` must provide the companion
/// scale_inv tensor data along with the block size for dequantization.
pub fn load_bf16_from_view_fp8(
    view: &safetensors::tensor::TensorView<'_>,
    name: &str,
    scale_info: Option<(&[f32], [usize; 2])>,
) -> Result<Vec<BF16>> {
    let dtype = view.dtype();
    match dtype {
        safetensors::Dtype::F8_E4M3 => {
            let (scale_inv, block_size) = scale_info.ok_or_else(|| {
                HerbertError::ModelLoad(format!(
                    "FP8 tensor {} requires scale_inv but none provided",
                    name
                ))
            })?;
            let shape = view.shape();
            if shape.len() < 2 {
                return Err(HerbertError::ModelLoad(format!(
                    "FP8 tensor {} has < 2 dims: {:?}",
                    name, shape
                )));
            }
            let rows = shape[..shape.len() - 1].iter().product::<usize>();
            let cols = *shape.last().expect("shape has >= 2 dims");
            Ok(dequant_fp8_to_bf16(view.data(), scale_inv, rows, cols, block_size))
        }
        _ => load_bf16_from_view(view, name),
    }
}

/// Precompute RoPE cos/sin tables for all positions up to `max_position_embeddings`.
/// Dispatches on `config.rope_type` to select Standard, Linear, or YARN frequency scaling.
pub fn compute_rope_cache(config: &Config) -> (Vec<f32>, Vec<f32>) {
    match &config.rope_type {
        RopeType::Standard => compute_standard_rope_cache(config),
        RopeType::Linear { factor } => compute_linear_rope_cache(config, *factor),
        RopeType::Yarn { factor, original_max_position_embeddings, beta_fast, beta_slow } => {
            compute_yarn_rope_cache(config, *factor, *original_max_position_embeddings, *beta_fast, *beta_slow)
        }
    }
}

pub fn compute_standard_rope_cache(config: &Config) -> (Vec<f32>, Vec<f32>) {
    let rotary_ndims = config.rotary_ndims;
    let max_seq_len = config.max_position_embeddings;
    let theta = config.rope_theta;
    let half_dim = rotary_ndims / 2;

    let mut inv_freq = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        inv_freq[i] = 1.0 / theta.powf((2 * i) as f32 / rotary_ndims as f32);
    }

    let mut cos_data = vec![0.0f32; max_seq_len * half_dim];
    let mut sin_data = vec![0.0f32; max_seq_len * half_dim];

    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let freq = pos as f32 * inv_freq[i];
            cos_data[pos * half_dim + i] = freq.cos();
            sin_data[pos * half_dim + i] = freq.sin();
        }
    }

    (cos_data, sin_data)
}

/// Linear RoPE scaling: inv_freq /= factor (extends effective context).
pub fn compute_linear_rope_cache(config: &Config, factor: f32) -> (Vec<f32>, Vec<f32>) {
    let rotary_ndims = config.rotary_ndims;
    let max_seq_len = config.max_position_embeddings;
    let theta = config.rope_theta;
    let half_dim = rotary_ndims / 2;

    let mut inv_freq = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        inv_freq[i] = 1.0 / (theta.powf((2 * i) as f32 / rotary_ndims as f32) * factor);
    }

    let mut cos_data = vec![0.0f32; max_seq_len * half_dim];
    let mut sin_data = vec![0.0f32; max_seq_len * half_dim];

    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let freq = pos as f32 * inv_freq[i];
            cos_data[pos * half_dim + i] = freq.cos();
            sin_data[pos * half_dim + i] = freq.sin();
        }
    }

    (cos_data, sin_data)
}

/// YARN RoPE (from HuggingFace LlamaYarnScaledRotaryEmbedding).
///
/// 1. Compute standard inv_freq: inv_freq[i] = 1 / (theta ^ (2i / head_dim))
/// 2. Compute wavelength boundaries from beta_fast / beta_slow
/// 3. For each frequency:
///    - wavelen < low_freq_wavelen: keep original (high frequency → extrapolation)
///    - wavelen > high_freq_wavelen: divide by factor (low frequency → interpolation)
///    - Else: smooth ramp blend between the two
/// 4. Precompute cos/sin tables using YARN-scaled frequencies
fn compute_yarn_rope_cache(
    config: &Config,
    factor: f32,
    original_max_pos: usize,
    beta_fast: f32,
    beta_slow: f32,
) -> (Vec<f32>, Vec<f32>) {
    let rotary_ndims = config.rotary_ndims;
    let max_seq_len = config.max_position_embeddings;
    let theta = config.rope_theta;
    let half_dim = rotary_ndims / 2;

    // Step 1: standard inv_freq
    let mut inv_freq = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        inv_freq[i] = 1.0 / theta.powf((2 * i) as f32 / rotary_ndims as f32);
    }

    // Step 2: wavelength boundaries
    let low_freq_wavelen = original_max_pos as f32 / beta_fast;
    let high_freq_wavelen = original_max_pos as f32 / beta_slow;

    // Step 3: apply YARN scaling per frequency
    for i in 0..half_dim {
        let freq = inv_freq[i];
        let wavelen = 2.0 * std::f32::consts::PI / freq;

        if wavelen < low_freq_wavelen {
            // High frequency: keep original (extrapolation region)
        } else if wavelen > high_freq_wavelen {
            // Low frequency: scale down by factor (interpolation region)
            inv_freq[i] = freq / factor;
        } else {
            // Transition region: smooth ramp blend
            let smooth = (original_max_pos as f32 / wavelen - beta_slow)
                / (beta_fast - beta_slow);
            let scaled_freq = freq / factor;
            inv_freq[i] = (1.0 - smooth) * scaled_freq + smooth * freq;
        }
    }

    // Step 4: precompute cos/sin tables
    let mut cos_data = vec![0.0f32; max_seq_len * half_dim];
    let mut sin_data = vec![0.0f32; max_seq_len * half_dim];

    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let angle = pos as f32 * inv_freq[i];
            cos_data[pos * half_dim + i] = angle.cos();
            sin_data[pos * half_dim + i] = angle.sin();
        }
    }

    (cos_data, sin_data)
}

// ============================================================================
// Generic streaming loader
// ============================================================================

/// Classify tensor names for routing during streaming load.
pub enum TensorKind {
    Embedding,
    FinalNorm,
    FinalNormBias,
    LmHead,
    LmHeadBias,
    LayerNorm { layer: usize, which: NormWhich },
    LayerNormBias { layer: usize, which: NormWhich },
    LayerAttnProj { layer: usize, which: AttnProjWhich },
    LayerAttnNorm { layer: usize, which: AttnNormWhich },
    LayerAttnBias { layer: usize, which: AttnProjWhich },
    LayerMlpProj { layer: usize, which: MlpProjWhich },
    LayerMoEGate { layer: usize },
    LayerExpertProj { layer: usize, expert: usize, which: MlpProjWhich },
    /// Fused expert gate+up projection: shape [num_experts, in_features, 2*moe_intermediate]
    FusedExpertGateUp { layer: usize },
    /// Fused expert down projection: shape [num_experts, moe_intermediate, hidden_size]
    FusedExpertDown { layer: usize },
    /// LFM2 convolution projections
    LayerConvProj { layer: usize, which: ConvProjWhich },
    /// LFM2 MoE expert bias
    LayerExpertBias { layer: usize },
    /// Fused QKV projection (Phi-4): [q_dim + 2*kv_dim, hidden_size]
    FusedQKV { layer: usize },
    /// Fused gate+up MLP projection (Phi-4): [2*intermediate_size, hidden_size]
    FusedGateUp { layer: usize },
    // GatedDeltaNet projections (Qwen3.5)
    DeltaNetProjQKV { layer: usize },
    DeltaNetProjZ { layer: usize },
    DeltaNetProjA { layer: usize },
    DeltaNetProjB { layer: usize },
    DeltaNetConv1d { layer: usize },
    DeltaNetALog { layer: usize },
    DeltaNetDtBias { layer: usize },
    DeltaNetNorm { layer: usize },
    DeltaNetOutProj { layer: usize },
    // Shared expert (Qwen3.5)
    SharedExpertProj { layer: usize, which: MlpProjWhich },
    SharedExpertGate { layer: usize },
    Unknown,
}

#[derive(Debug, Clone, Copy)]
pub enum NormWhich {
    Input,
    PostAttention,
    /// Gemma3: pre-feedforward layernorm
    PreFeedForward,
    /// Gemma3: post-feedforward layernorm
    PostFeedForward,
}

pub enum AttnProjWhich {
    Q,
    K,
    V,
    O,
}

pub enum AttnNormWhich {
    Q,
    K,
}

pub enum MlpProjWhich {
    Gate,
    Up,
    Down,
}

pub enum ConvProjWhich {
    In,
    Conv,
    Out,
}

/// Pre-merge LoRA weights into base weight (BF16).
///
/// Computes: `merged = base + scaling * (lora_B @ lora_A)`
/// where lora_A is [r, in_features] and lora_B is [out_features, r], both row-major BF16.
/// Returns a new BF16 vec of size [out_features * in_features].
fn merge_lora_bf16(
    base: &[BF16],
    lora_a: &[BF16], // [r, in_features]
    lora_b: &[BF16], // [out_features, r]
    scaling: f32,
    out_features: usize,
    r: usize,
    in_features: usize,
) -> Vec<BF16> {
    assert_eq!(base.len(), out_features * in_features);
    assert_eq!(lora_a.len(), r * in_features);
    assert_eq!(lora_b.len(), out_features * r);

    // Convert lora_A and lora_B to f32 for matmul
    let a_f32: Vec<f32> = lora_a.iter().map(|&x| bf16_to_f32(x)).collect();
    let b_f32: Vec<f32> = lora_b.iter().map(|&x| bf16_to_f32(x)).collect();

    let mut merged = Vec::with_capacity(out_features * in_features);
    for i in 0..out_features {
        for j in 0..in_features {
            let base_val = bf16_to_f32(base[i * in_features + j]);
            // dot product: B[i, :] @ A[:, j] = sum_k B[i,k] * A[k,j]
            let mut dot = 0.0f32;
            for k in 0..r {
                dot += b_f32[i * r + k] * a_f32[k * in_features + j];
            }
            merged.push(f32_to_bf16(base_val + scaling * dot));
        }
    }
    merged
}

pub fn classify_tensor(name: &str) -> TensorKind {
    // Skip scale_inv companion tensors (used for FP8 dequant, loaded separately)
    if name.ends_with("_scale_inv") {
        return TensorKind::Unknown;
    }
    // Skip FP8 activation scale tensors (not needed for our inference)
    if name.ends_with(".activation_scale") {
        return TensorKind::Unknown;
    }

    // Normalize VL prefixes:
    // Mistral3 VL: language_model.model.X → model.X, language_model.lm_head.X → lm_head.X
    // Qwen3 VL:    model.language_model.X → model.X
    let name: Cow<'_, str> = if let Some(rest) = name.strip_prefix("language_model.model.") {
        Cow::Owned(format!("model.{}", rest))
    } else if let Some(rest) = name.strip_prefix("language_model.") {
        Cow::Borrowed(rest)
    } else if let Some(rest) = name.strip_prefix("model.language_model.") {
        Cow::Owned(format!("model.{}", rest))
    } else if !name.starts_with("model.") && !name.starts_with("lm_head.")
        && (name.starts_with("embed_tokens.")
            || name.starts_with("embedding_norm.")
            || name.starts_with("layers.")
            || name.starts_with("norm."))
    {
        // Normalize bare prefix (e.g. LFM2-ColBERT): embed_tokens.X → model.embed_tokens.X
        Cow::Owned(format!("model.{}", name))
    } else {
        Cow::Borrowed(name)
    };
    let name = name.as_ref();

    // Strip `.base_layer.weight` suffix from LoRA models (Phi-4) → `.weight`
    let name: Cow<'_, str> = if let Some(rest) = name.strip_suffix(".base_layer.weight") {
        Cow::Owned(format!("{}.weight", rest))
    } else {
        Cow::Borrowed(name)
    };
    let name = name.as_ref();

    if name == "model.embed_tokens.weight" {
        return TensorKind::Embedding;
    }
    if name == "model.norm.weight" || name == "model.embedding_norm.weight" {
        return TensorKind::FinalNorm;
    }
    if name == "model.norm.bias" {
        return TensorKind::FinalNormBias;
    }
    if name == "lm_head.weight" {
        return TensorKind::LmHead;
    }
    if name == "lm_head.bias" {
        return TensorKind::LmHeadBias;
    }

    // model.layers.{idx}.{subpath}
    if let Some(rest) = name.strip_prefix("model.layers.") {
        let dot_pos = rest.find('.').unwrap_or(rest.len());
        if let Ok(layer) = rest[..dot_pos].parse::<usize>() {
            let subpath = &rest[dot_pos + 1..];
            return match subpath {
                "input_layernorm.weight" | "operator_norm.weight" => TensorKind::LayerNorm {
                    layer,
                    which: NormWhich::Input,
                },
                "post_attention_layernorm.weight" | "ffn_norm.weight" => TensorKind::LayerNorm {
                    layer,
                    which: NormWhich::PostAttention,
                },
                "pre_feedforward_layernorm.weight" => TensorKind::LayerNorm {
                    layer,
                    which: NormWhich::PreFeedForward,
                },
                "post_feedforward_layernorm.weight" => TensorKind::LayerNorm {
                    layer,
                    which: NormWhich::PostFeedForward,
                },
                "input_layernorm.bias" | "operator_norm.bias" => TensorKind::LayerNormBias {
                    layer,
                    which: NormWhich::Input,
                },
                "post_attention_layernorm.bias" | "ffn_norm.bias" => TensorKind::LayerNormBias {
                    layer,
                    which: NormWhich::PostAttention,
                },
                "self_attn.q_proj.weight" => TensorKind::LayerAttnProj {
                    layer,
                    which: AttnProjWhich::Q,
                },
                "self_attn.k_proj.weight" => TensorKind::LayerAttnProj {
                    layer,
                    which: AttnProjWhich::K,
                },
                "self_attn.v_proj.weight" => TensorKind::LayerAttnProj {
                    layer,
                    which: AttnProjWhich::V,
                },
                "self_attn.o_proj.weight" | "self_attn.out_proj.weight" => TensorKind::LayerAttnProj {
                    layer,
                    which: AttnProjWhich::O,
                },
                "self_attn.q_norm.weight" | "self_attn.q_layernorm.weight" => TensorKind::LayerAttnNorm {
                    layer,
                    which: AttnNormWhich::Q,
                },
                "self_attn.k_norm.weight" | "self_attn.k_layernorm.weight" => TensorKind::LayerAttnNorm {
                    layer,
                    which: AttnNormWhich::K,
                },
                "self_attn.q_proj.bias" => TensorKind::LayerAttnBias {
                    layer,
                    which: AttnProjWhich::Q,
                },
                "self_attn.k_proj.bias" => TensorKind::LayerAttnBias {
                    layer,
                    which: AttnProjWhich::K,
                },
                "self_attn.v_proj.bias" => TensorKind::LayerAttnBias {
                    layer,
                    which: AttnProjWhich::V,
                },
                "self_attn.o_proj.bias" => TensorKind::LayerAttnBias {
                    layer,
                    which: AttnProjWhich::O,
                },
                // Fused QKV projection (Phi-4): self_attn.qkv_proj.weight
                "self_attn.qkv_proj.weight" => TensorKind::FusedQKV { layer },
                // Convolution projections (LFM2)
                "conv.in_proj.weight" => TensorKind::LayerConvProj {
                    layer,
                    which: ConvProjWhich::In,
                },
                "conv.conv.weight" => TensorKind::LayerConvProj {
                    layer,
                    which: ConvProjWhich::Conv,
                },
                "conv.out_proj.weight" => TensorKind::LayerConvProj {
                    layer,
                    which: ConvProjWhich::Out,
                },
                // Dense MLP (Qwen and LFM2 aliases)
                "mlp.gate_proj.weight" | "feed_forward.w1.weight" => TensorKind::LayerMlpProj {
                    layer,
                    which: MlpProjWhich::Gate,
                },
                "mlp.up_proj.weight" | "feed_forward.w3.weight" => TensorKind::LayerMlpProj {
                    layer,
                    which: MlpProjWhich::Up,
                },
                "mlp.down_proj.weight" | "feed_forward.w2.weight" => TensorKind::LayerMlpProj {
                    layer,
                    which: MlpProjWhich::Down,
                },
                // Fused gate+up MLP projection (Phi-4): mlp.gate_up_proj.weight
                "mlp.gate_up_proj.weight" => TensorKind::FusedGateUp { layer },
                // MoE router gate (Qwen, LFM2, PhiMoE aliases)
                "mlp.gate.weight" | "feed_forward.gate.weight" | "block_sparse_moe.gate.weight"
                    => TensorKind::LayerMoEGate { layer },
                // LFM2 expert bias
                "feed_forward.expert_bias" => TensorKind::LayerExpertBias { layer },
                // GatedDeltaNet (Qwen3.5 linear attention)
                "linear_attn.in_proj_qkv.weight" => TensorKind::DeltaNetProjQKV { layer },
                "linear_attn.in_proj_z.weight" => TensorKind::DeltaNetProjZ { layer },
                "linear_attn.in_proj_a.weight" => TensorKind::DeltaNetProjA { layer },
                "linear_attn.in_proj_b.weight" => TensorKind::DeltaNetProjB { layer },
                "linear_attn.conv1d.weight" => TensorKind::DeltaNetConv1d { layer },
                "linear_attn.A_log" => TensorKind::DeltaNetALog { layer },
                "linear_attn.dt_bias" => TensorKind::DeltaNetDtBias { layer },
                "linear_attn.norm.weight" => TensorKind::DeltaNetNorm { layer },
                "linear_attn.out_proj.weight" => TensorKind::DeltaNetOutProj { layer },
                // Shared expert (Qwen3.5)
                "mlp.shared_expert.gate_proj.weight" => TensorKind::SharedExpertProj { layer, which: MlpProjWhich::Gate },
                "mlp.shared_expert.up_proj.weight" => TensorKind::SharedExpertProj { layer, which: MlpProjWhich::Up },
                "mlp.shared_expert.down_proj.weight" => TensorKind::SharedExpertProj { layer, which: MlpProjWhich::Down },
                "mlp.shared_expert_gate.weight" => TensorKind::SharedExpertGate { layer },
                _ => {
                    // Fused expert tensors: mlp.experts.gate_up_proj[.weight] / mlp.experts.down_proj[.weight]
                    if subpath == "mlp.experts.gate_up_proj"
                        || subpath == "mlp.experts.gate_up_proj.weight"
                    {
                        return TensorKind::FusedExpertGateUp { layer };
                    }
                    if subpath == "mlp.experts.down_proj"
                        || subpath == "mlp.experts.down_proj.weight"
                    {
                        return TensorKind::FusedExpertDown { layer };
                    }
                    // MoE expert projections: mlp.experts.{j}.{gate,up,down}_proj.weight
                    if let Some(expert_rest) = subpath.strip_prefix("mlp.experts.") {
                        if let Some(dot2) = expert_rest.find('.') {
                            if let Ok(expert) = expert_rest[..dot2].parse::<usize>() {
                                let proj_part = &expert_rest[dot2 + 1..];
                                return match proj_part {
                                    "gate_proj.weight" => TensorKind::LayerExpertProj {
                                        layer,
                                        expert,
                                        which: MlpProjWhich::Gate,
                                    },
                                    "up_proj.weight" => TensorKind::LayerExpertProj {
                                        layer,
                                        expert,
                                        which: MlpProjWhich::Up,
                                    },
                                    "down_proj.weight" => TensorKind::LayerExpertProj {
                                        layer,
                                        expert,
                                        which: MlpProjWhich::Down,
                                    },
                                    _ => TensorKind::Unknown,
                                };
                            }
                        }
                    }
                    // PhiMoE expert projections: block_sparse_moe.experts.{j}.w1/w2/w3.weight
                    if let Some(expert_rest) = subpath.strip_prefix("block_sparse_moe.experts.") {
                        if let Some(dot2) = expert_rest.find('.') {
                            if let Ok(expert) = expert_rest[..dot2].parse::<usize>() {
                                let proj_part = &expert_rest[dot2 + 1..];
                                return match proj_part {
                                    "w1.weight" => TensorKind::LayerExpertProj {
                                        layer,
                                        expert,
                                        which: MlpProjWhich::Gate,
                                    },
                                    "w3.weight" => TensorKind::LayerExpertProj {
                                        layer,
                                        expert,
                                        which: MlpProjWhich::Up,
                                    },
                                    "w2.weight" => TensorKind::LayerExpertProj {
                                        layer,
                                        expert,
                                        which: MlpProjWhich::Down,
                                    },
                                    _ => TensorKind::Unknown,
                                };
                            }
                        }
                    }
                    // LFM2 MoE expert projections: feed_forward.experts.{j}.w1/w2/w3.weight
                    if let Some(expert_rest) = subpath.strip_prefix("feed_forward.experts.") {
                        if let Some(dot2) = expert_rest.find('.') {
                            if let Ok(expert) = expert_rest[..dot2].parse::<usize>() {
                                let proj_part = &expert_rest[dot2 + 1..];
                                return match proj_part {
                                    "w1.weight" => TensorKind::LayerExpertProj {
                                        layer,
                                        expert,
                                        which: MlpProjWhich::Gate,
                                    },
                                    "w3.weight" => TensorKind::LayerExpertProj {
                                        layer,
                                        expert,
                                        which: MlpProjWhich::Up,
                                    },
                                    "w2.weight" => TensorKind::LayerExpertProj {
                                        layer,
                                        expert,
                                        which: MlpProjWhich::Down,
                                    },
                                    _ => TensorKind::Unknown,
                                };
                            }
                        }
                    }
                    TensorKind::Unknown
                }
            };
        }
    }

    TensorKind::Unknown
}

/// Partial expert accumulator for MoE loading.
pub struct PartialExpert<W> {
    pub gate_proj: Option<W>,
    pub up_proj: Option<W>,
    pub down_proj: Option<W>,
}

impl<W> Default for PartialExpert<W> {
    fn default() -> Self {
        Self {
            gate_proj: None,
            up_proj: None,
            down_proj: None,
        }
    }
}

/// Partial layer accumulator used during streaming load.
/// Fields are filled as tensors arrive from different shards.
pub struct PartialLayer<W> {
    pub input_layernorm: Option<Vec<BF16>>,
    pub post_attention_layernorm: Option<Vec<BF16>>,
    pub q_proj: Option<W>,
    pub k_proj: Option<W>,
    pub v_proj: Option<W>,
    pub o_proj: Option<W>,
    pub q_norm: Option<Vec<BF16>>,
    pub k_norm: Option<Vec<BF16>>,
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    pub o_bias: Option<Vec<f32>>,
    // Norm biases (LayerNorm models like PhiMoE)
    pub input_layernorm_bias: Option<Vec<f32>>,
    pub post_attention_layernorm_bias: Option<Vec<f32>>,
    // Convolution fields (LFM2)
    pub conv_in_proj: Option<W>,
    pub conv_kernel: Option<Vec<f32>>,
    pub conv_out_proj: Option<W>,
    // Dense MLP fields
    pub gate_proj: Option<W>,
    pub up_proj: Option<W>,
    pub down_proj: Option<W>,
    // MoE fields
    pub moe_gate: Option<W>,
    pub experts: Option<Vec<PartialExpert<W>>>,
    // MoE expert bias (LFM2)
    pub expert_bias: Option<Vec<f32>>,
    // Gemma3: pre/post feedforward norms
    pub pre_feedforward_layernorm: Option<Vec<BF16>>,
    pub post_feedforward_layernorm: Option<Vec<BF16>>,
    // GatedDeltaNet fields (Qwen3.5)
    pub deltanet_proj_qkv: Option<W>,
    pub deltanet_proj_z: Option<W>,
    pub deltanet_proj_a: Option<Vec<f32>>,
    pub deltanet_proj_b: Option<Vec<f32>>,
    pub deltanet_conv1d: Option<Vec<f32>>,
    pub deltanet_a_log: Option<Vec<f32>>,
    pub deltanet_dt_bias: Option<Vec<f32>>,
    pub deltanet_norm: Option<Vec<BF16>>,
    pub deltanet_out_proj: Option<W>,
    // Shared expert (Qwen3.5)
    pub shared_expert_gate_proj: Option<W>,
    pub shared_expert_up_proj: Option<W>,
    pub shared_expert_down_proj: Option<W>,
    pub shared_expert_gate: Option<Vec<f32>>,
}

impl<W> Default for PartialLayer<W> {
    fn default() -> Self {
        Self {
            input_layernorm: None,
            post_attention_layernorm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            o_proj: None,
            q_norm: None,
            k_norm: None,
            q_bias: None,
            k_bias: None,
            v_bias: None,
            o_bias: None,
            input_layernorm_bias: None,
            post_attention_layernorm_bias: None,
            conv_in_proj: None,
            conv_kernel: None,
            conv_out_proj: None,
            gate_proj: None,
            up_proj: None,
            down_proj: None,
            moe_gate: None,
            experts: None,
            expert_bias: None,
            pre_feedforward_layernorm: None,
            post_feedforward_layernorm: None,
            deltanet_proj_qkv: None,
            deltanet_proj_z: None,
            deltanet_proj_a: None,
            deltanet_proj_b: None,
            deltanet_conv1d: None,
            deltanet_a_log: None,
            deltanet_dt_bias: None,
            deltanet_norm: None,
            deltanet_out_proj: None,
            shared_expert_gate_proj: None,
            shared_expert_up_proj: None,
            shared_expert_down_proj: None,
            shared_expert_gate: None,
        }
    }
}

/// Result of processing a single tensor in parallel.
/// Collected from parallel tasks and merged sequentially into accumulators.
enum TensorResult<W> {
    Embedding(Vec<BF16>),
    FinalNorm(Vec<BF16>),
    FinalNormBias(Vec<f32>),
    LmHead(Vec<BF16>),
    LmHeadBias(Vec<f32>),
    LayerNorm { layer: usize, which: NormWhich, data: Vec<BF16> },
    LayerNormBias { layer: usize, which: NormWhich, data: Vec<f32> },
    LayerAttnProj { layer: usize, which: AttnProjWhich, weight: W },
    LayerAttnNorm { layer: usize, which: AttnNormWhich, data: Vec<BF16> },
    LayerAttnBias { layer: usize, which: AttnProjWhich, data: Vec<f32> },
    LayerMlpProj { layer: usize, which: MlpProjWhich, weight: W },
    LayerMoEGate { layer: usize, weight: W },
    LayerExpertProj { layer: usize, expert: usize, which: MlpProjWhich, weight: W },
    FusedExpertGateUp { layer: usize, experts: Vec<(W, W)> },
    FusedExpertDown { layer: usize, experts: Vec<W> },
    LayerConvProj { layer: usize, which: ConvProjWhich, weight: W },
    LayerConvKernel { layer: usize, data: Vec<f32> },
    LayerExpertBias { layer: usize, data: Vec<f32> },
    /// Fused QKV split into separate Q, K, V weights
    FusedQKVSplit { layer: usize, q: W, k: W, v: W },
    /// Fused gate_up split into separate gate and up weights
    FusedGateUpSplit { layer: usize, gate: W, up: W },
    // GatedDeltaNet (Qwen3.5)
    DeltaNetProjQKV { layer: usize, weight: W },
    DeltaNetProjZ { layer: usize, weight: W },
    DeltaNetProjA { layer: usize, data: Vec<f32> },
    DeltaNetProjB { layer: usize, data: Vec<f32> },
    DeltaNetConv1d { layer: usize, data: Vec<f32> },
    DeltaNetALog { layer: usize, data: Vec<f32> },
    DeltaNetDtBias { layer: usize, data: Vec<f32> },
    DeltaNetNorm { layer: usize, data: Vec<BF16> },
    DeltaNetOutProj { layer: usize, weight: W },
    // Shared expert (Qwen3.5)
    SharedExpertProj { layer: usize, which: MlpProjWhich, weight: W },
    SharedExpertGate { layer: usize, data: Vec<f32> },
    Unknown,
}

impl<W> PartialLayer<W> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn finalize<L: LinearOps<Weight = W>>(
        self,
        config: &Config,
        layer_idx: usize,
    ) -> Result<GenericDecoderLayer<L>> {
        let hidden_size = config.hidden_size;
        let activation = MlpActivation::SwiGLU;

        // Build FFN: MoE or Dense depending on layer and config
        let ffn = if config.is_moe_layer(layer_idx) {
            let moe_gate = self.moe_gate.ok_or_else(|| {
                HerbertError::ModelLoad(format!("layer {}: missing MoE router gate", layer_idx))
            })?;
            let partial_experts = self.experts.ok_or_else(|| {
                HerbertError::ModelLoad(format!("layer {}: missing MoE experts", layer_idx))
            })?;
            let num_experts = config.num_experts.ok_or_else(|| {
                HerbertError::ModelLoad(format!("layer {}: MoE config missing num_experts", layer_idx))
            })?;
            let moe_intermediate_size = config.moe_intermediate_size.ok_or_else(|| {
                HerbertError::ModelLoad(format!("layer {}: MoE config missing moe_intermediate_size", layer_idx))
            })?;
            if partial_experts.len() != num_experts {
                return Err(HerbertError::ModelLoad(format!(
                    "layer {}: expected {} experts, got {}",
                    layer_idx,
                    num_experts,
                    partial_experts.len()
                )));
            }
            let mut experts = Vec::with_capacity(num_experts);
            for (eidx, pe) in partial_experts.into_iter().enumerate() {
                experts.push(GenericMLP {
                    gate_proj: pe.gate_proj.ok_or_else(|| {
                        HerbertError::ModelLoad(format!(
                            "layer {} expert {}: missing gate_proj",
                            layer_idx, eidx
                        ))
                    })?,
                    up_proj: pe.up_proj.ok_or_else(|| {
                        HerbertError::ModelLoad(format!(
                            "layer {} expert {}: missing up_proj",
                            layer_idx, eidx
                        ))
                    })?,
                    down_proj: pe.down_proj.ok_or_else(|| {
                        HerbertError::ModelLoad(format!(
                            "layer {} expert {}: missing down_proj",
                            layer_idx, eidx
                        ))
                    })?,
                    hidden_size,
                    intermediate_size: moe_intermediate_size,
                    activation,
                    _marker: std::marker::PhantomData,
                });
            }
            // Build shared expert if present (Qwen3.5)
            let shared_expert = if self.shared_expert_gate_proj.is_some() {
                let shared_inter = config.shared_expert_intermediate_size.unwrap_or(config.intermediate_size);
                Some(GenericMLP {
                    gate_proj: self.shared_expert_gate_proj.ok_or_else(|| {
                        HerbertError::ModelLoad(format!("layer {}: missing shared_expert gate_proj", layer_idx))
                    })?,
                    up_proj: self.shared_expert_up_proj.ok_or_else(|| {
                        HerbertError::ModelLoad(format!("layer {}: missing shared_expert up_proj", layer_idx))
                    })?,
                    down_proj: self.shared_expert_down_proj.ok_or_else(|| {
                        HerbertError::ModelLoad(format!("layer {}: missing shared_expert down_proj", layer_idx))
                    })?,
                    hidden_size,
                    intermediate_size: shared_inter,
                    activation,
                    _marker: std::marker::PhantomData,
                })
            } else {
                None
            };

            GenericFFN::MoE(GenericMoE {
                gate: moe_gate,
                expert_pool: std::sync::Mutex::new(ExpertPool::new_all_loaded(experts, layer_idx)),
                num_experts,
                num_experts_per_tok: config.num_experts_per_tok.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("layer {}: MoE config missing num_experts_per_tok", layer_idx))
                })?,
                hidden_size,
                moe_intermediate_size,
                norm_topk_prob: config.norm_topk_prob,
                layer_id: layer_idx,
                expert_bias: self.expert_bias,
                shared_expert,
                shared_expert_gate: self.shared_expert_gate,
            })
        } else {
            // Dense MLP
            GenericFFN::Dense(GenericMLP {
                gate_proj: self
                    .gate_proj
                    .ok_or_else(|| HerbertError::ModelLoad(format!("layer {}: missing gate_proj", layer_idx)))?,
                up_proj: self
                    .up_proj
                    .ok_or_else(|| HerbertError::ModelLoad(format!("layer {}: missing up_proj", layer_idx)))?,
                down_proj: self
                    .down_proj
                    .ok_or_else(|| HerbertError::ModelLoad(format!("layer {}: missing down_proj", layer_idx)))?,
                hidden_size,
                intermediate_size: config.intermediate_size,
                activation,
                _marker: std::marker::PhantomData,
            })
        };

        // Build the layer block: attention or convolution
        let block = if config.is_attn_layer(layer_idx) {
            // q_norm / k_norm: required for qwen3/lfm2 (has_qk_norm), optional otherwise
            let q_norm = if config.has_qk_norm {
                Some(self.q_norm.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("layer {}: missing q_norm", layer_idx))
                })?)
            } else {
                self.q_norm // pass through if present, None otherwise
            };
            let k_norm = if config.has_qk_norm {
                Some(self.k_norm.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("layer {}: missing k_norm", layer_idx))
                })?)
            } else {
                self.k_norm
            };

            let use_rope = true;

            LayerBlock::Attention(GenericAttention {
                q_proj: self
                    .q_proj
                    .ok_or_else(|| HerbertError::ModelLoad(format!("layer {}: missing q_proj", layer_idx)))?,
                k_proj: self
                    .k_proj
                    .ok_or_else(|| HerbertError::ModelLoad(format!("layer {}: missing k_proj", layer_idx)))?,
                v_proj: self
                    .v_proj
                    .ok_or_else(|| HerbertError::ModelLoad(format!("layer {}: missing v_proj", layer_idx)))?,
                o_proj: self
                    .o_proj
                    .ok_or_else(|| HerbertError::ModelLoad(format!("layer {}: missing o_proj", layer_idx)))?,
                q_norm,
                k_norm,
                q_bias: self.q_bias,
                k_bias: self.k_bias,
                v_bias: self.v_bias,
                o_bias: self.o_bias,
                num_heads: config.num_attention_heads,
                num_kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                rotary_ndims: config.rotary_ndims,
                head_to_kv_head: (0..config.num_attention_heads)
                    .map(|h| h / config.num_kv_groups())
                    .collect(),
                hidden_size,
                q_dim: config.q_dim(),
                kv_dim: config.kv_dim(),
                rms_norm_eps: config.rms_norm_eps,
                use_rope,
                attn_scale: config.attn_scale,
                use_gemma_qk_norm: config.norm_type == herbert_core::config::NormType::GemmaRMSNorm,
                has_output_gate: config.attn_output_gate,
                use_q_bf16: false,
                _marker: std::marker::PhantomData,
            })
        } else if config.is_deltanet_layer(layer_idx) {
            // GatedDeltaNet layer (Qwen3.5)
            LayerBlock::GatedDeltaNet(GenericGatedDeltaNet {
                in_proj_qkv: self.deltanet_proj_qkv.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("layer {}: missing deltanet in_proj_qkv", layer_idx))
                })?,
                in_proj_z: self.deltanet_proj_z.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("layer {}: missing deltanet in_proj_z", layer_idx))
                })?,
                in_proj_a: self.deltanet_proj_a.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("layer {}: missing deltanet in_proj_a", layer_idx))
                })?,
                in_proj_b: self.deltanet_proj_b.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("layer {}: missing deltanet in_proj_b", layer_idx))
                })?,
                conv1d_weight: self.deltanet_conv1d.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("layer {}: missing deltanet conv1d", layer_idx))
                })?,
                a_log: self.deltanet_a_log.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("layer {}: missing deltanet A_log", layer_idx))
                })?,
                dt_bias: self.deltanet_dt_bias.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("layer {}: missing deltanet dt_bias", layer_idx))
                })?,
                norm_weight: self.deltanet_norm.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("layer {}: missing deltanet norm", layer_idx))
                })?,
                out_proj: self.deltanet_out_proj.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("layer {}: missing deltanet out_proj", layer_idx))
                })?,
                num_k_heads: config.linear_num_key_heads.unwrap_or(0),
                num_v_heads: config.linear_num_value_heads.unwrap_or(0),
                head_k_dim: config.linear_key_head_dim.unwrap_or(0),
                head_v_dim: config.linear_value_head_dim.unwrap_or(0),
                conv_kernel_size: config.linear_conv_kernel_dim.unwrap_or(4),
                hidden_size,
                rms_norm_eps: config.rms_norm_eps,
            })
        } else {
            // Convolution layer
            let conv_kernel = self.conv_kernel.ok_or_else(|| {
                HerbertError::ModelLoad(format!("layer {}: missing conv_kernel", layer_idx))
            })?;
            let expected_kernel = config.hidden_size * config.conv_l_cache;
            if conv_kernel.len() != expected_kernel {
                return Err(HerbertError::ModelLoad(format!(
                    "layer {}: conv_kernel has {} elems, expected {} (hidden_size={} * conv_l_cache={})",
                    layer_idx,
                    conv_kernel.len(),
                    expected_kernel,
                    config.hidden_size,
                    config.conv_l_cache
                )));
            }
            LayerBlock::Convolution(GenericConvolution {
                in_proj: self.conv_in_proj.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("layer {}: missing conv_in_proj", layer_idx))
                })?,
                conv_kernel,
                out_proj: self.conv_out_proj.ok_or_else(|| {
                    HerbertError::ModelLoad(format!("layer {}: missing conv_out_proj", layer_idx))
                })?,
                hidden_size: config.hidden_size,
                conv_l_cache: config.conv_l_cache,
            })
        };

        Ok(GenericDecoderLayer {
            input_layernorm: self
                .input_layernorm
                .ok_or_else(|| HerbertError::ModelLoad(format!("layer {}: missing input_layernorm", layer_idx)))?,
            input_layernorm_bias: self.input_layernorm_bias,
            post_attention_layernorm: self
                .post_attention_layernorm
                .ok_or_else(|| HerbertError::ModelLoad(format!("layer {}: missing post_attention_layernorm", layer_idx)))?,
            post_attention_layernorm_bias: self.post_attention_layernorm_bias,
            pre_feedforward_layernorm: self.pre_feedforward_layernorm,
            post_feedforward_layernorm: self.post_feedforward_layernorm,
            block,
            ffn,
            hidden_size,
            rms_norm_eps: config.rms_norm_eps,
            norm_type: config.norm_type,
        })
    }
}

/// Generic streaming model loader with optional progress reporting.
pub fn load_model_streaming_progress<L: LinearOps>(
    model_dir: &Path,
    quantize_fn: impl Fn(&[BF16], usize, usize, &str) -> Result<L::Weight> + Send + Sync,
    show_progress: bool,
) -> Result<(Config, GenericModel<L>)> {
    let config = Config::from_file(&model_dir.join("config.json"))?;

    // Detect FP8 block quantization from config.json (weight_block_size field)
    let block_size: Option<[usize; 2]> = {
        let raw: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(model_dir.join("config.json"))?,
        )?;
        // Check top-level, text_config, or quantization_config for weight_block_size
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


    // Accumulators
    let mut embed_tokens_bf16: Option<Vec<BF16>> = None;
    let mut final_norm: Option<Vec<BF16>> = None;
    let mut final_norm_bias: Option<Vec<f32>> = None;
    let mut lm_head_raw: Option<Vec<BF16>> = None;
    let mut lm_head_bias: Option<Vec<f32>> = None;
    let mut partial_layers: Vec<PartialLayer<L::Weight>> = (0..config.num_layers)
        .map(|layer_idx| {
            let mut pl = PartialLayer::new();
            if config.is_moe_layer(layer_idx) {
                let ne = config.num_experts.expect("MoE layer requires num_experts in config");
                pl.experts = Some((0..ne).map(|_| PartialExpert::default()).collect());
            }
            pl
        })
        .collect();

    // Count total tensors for the progress bar
    let total_tensors: u64 = if let Some(ref w2s) = weight_to_shard {
        w2s.len() as u64
    } else {
        0 // single-shard: updated after opening
    };
    let tensor_pb = if show_progress {
        Some(progress::model_bar(total_tensors, config.num_layers))
    } else {
        None
    };
    let mut max_layer_seen: usize = 0;
    // Stream shards one at a time
    for shard_name in &shard_names {
        let shard_path = model_dir.join(shard_name);
        let shard_file = fs::File::open(&shard_path).map_err(|e| {
            HerbertError::ModelLoad(format!("Failed to open shard {}: {}", shard_name, e))
        })?;
        // SAFETY: the file is read-only and not modified while mapped.
        let shard_mmap = unsafe { Mmap::map(&shard_file) }.map_err(|e| {
            HerbertError::ModelLoad(format!("Failed to mmap shard {}: {}", shard_name, e))
        })?;
        let tensors = SafeTensors::deserialize(&shard_mmap[..]).map_err(|e| {
            HerbertError::ModelLoad(format!("Failed to parse shard {}: {}", shard_name, e))
        })?;

        // Determine which tensors are in this shard
        let tensor_names: Vec<String> = if let Some(ref w2s) = weight_to_shard {
            w2s.iter()
                .filter(|(_, s)| s.as_str() == shard_name.as_str())
                .map(|(w, _)| w.clone())
                .collect()
        } else {
            let names: Vec<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();
            // Single-shard: set the progress bar length now that we know the count
            if let Some(ref pb) = tensor_pb {
                pb.set_length(names.len() as u64);
            }
            names
        };

        // Helper: load BF16 data with automatic FP8 dequantization
        let load_bf16_auto = |view: &safetensors::tensor::TensorView<'_>,
                              tensor_name: &str|
         -> Result<Vec<BF16>> {
            if view.dtype() == safetensors::Dtype::F8_E4M3 {
                let scale_name = format!("{}_scale_inv", tensor_name);
                let scale_view = tensors.tensor(&scale_name).map_err(|_| {
                    HerbertError::ModelLoad(format!(
                        "Missing scale tensor {} for FP8 tensor {}",
                        scale_name, tensor_name
                    ))
                })?;
                let scale_data = load_f32_from_view(&scale_view, &scale_name)?;
                // Determine block size: explicit from config, or per-tensor (full dimensions)
                let bs = block_size.unwrap_or_else(|| {
                    let shape = view.shape();
                    let rows = shape[..shape.len() - 1].iter().product::<usize>().max(1);
                    let cols = *shape.last().unwrap_or(&1);
                    [rows, cols]
                });
                load_bf16_from_view_fp8(view, tensor_name, Some((&scale_data, bs)))
            } else {
                load_bf16_from_view(view, tensor_name)
            }
        };

        let lora_map: HashMap<String, Vec<BF16>> = HashMap::new();
        let lora_scaling: f32 = 2.0;
        let lora_r: usize = 256;

        // --- Parallel tensor processing (quantization is CPU-bound) ---
        let num_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(tensor_names.len().max(1));
        let chunk_size = tensor_names.len().div_ceil(num_workers);

        let process_tensor = |name: &String| -> Result<TensorResult<L::Weight>> {
            let view = tensors
                .tensor(name)
                .map_err(|e| HerbertError::ModelLoad(format!("Tensor {} not found: {}", name, e)))?;

            let kind = classify_tensor(name);

            let result = match kind {
                TensorKind::Embedding => {
                    TensorResult::Embedding(load_bf16_auto(&view, name)?)
                }
                TensorKind::FinalNorm => {
                    TensorResult::FinalNorm(load_bf16_auto(&view, name)?)
                }
                TensorKind::LmHead => {
                    TensorResult::LmHead(load_bf16_auto(&view, name)?)
                }
                TensorKind::LayerNorm { layer, which } => {
                    let data = load_bf16_auto(&view, name)?;
                    TensorResult::LayerNorm { layer, which, data }
                }
                TensorKind::LayerAttnProj { layer, which } => {
                    let bf16_data = load_bf16_auto(&view, name)?;
                    let (out_f, in_f) = match which {
                        AttnProjWhich::Q => {
                            let q_out = if config.attn_output_gate {
                                config.q_dim() * 2 // fused Q + output gate
                            } else {
                                config.q_dim()
                            };
                            (q_out, config.hidden_size)
                        }
                        AttnProjWhich::K => (config.kv_dim(), config.hidden_size),
                        AttnProjWhich::V => (config.kv_dim(), config.hidden_size),
                        AttnProjWhich::O => (config.hidden_size, config.q_dim()),
                    };
                    // Merge vision LoRA for o_proj if available (Phi-4)
                    let bf16_data = if !lora_map.is_empty() && matches!(which, AttnProjWhich::O) {
                        let lora_a_name = format!("model.layers.{}.self_attn.o_proj.lora_A.vision.weight", layer);
                        let lora_b_name = format!("model.layers.{}.self_attn.o_proj.lora_B.vision.weight", layer);
                        if let (Some(a), Some(b)) = (lora_map.get(&lora_a_name), lora_map.get(&lora_b_name)) {
                            merge_lora_bf16(&bf16_data, a, b, lora_scaling, out_f, lora_r, in_f)
                        } else {
                            bf16_data
                        }
                    } else {
                        bf16_data
                    };
                    let weight = quantize_fn(&bf16_data, out_f, in_f, name)?;
                    TensorResult::LayerAttnProj { layer, which, weight }
                }
                TensorKind::LayerAttnNorm { layer, which } => {
                    let data = load_bf16_auto(&view, name)?;
                    TensorResult::LayerAttnNorm { layer, which, data }
                }
                TensorKind::LayerAttnBias { layer, which } => {
                    let data = load_f32_from_view(&view, name)?;
                    TensorResult::LayerAttnBias { layer, which, data }
                }
                TensorKind::LayerMlpProj { layer, which } => {
                    let bf16_data = load_bf16_auto(&view, name)?;
                    let (out_f, in_f) = match which {
                        MlpProjWhich::Gate => (config.intermediate_size, config.hidden_size),
                        MlpProjWhich::Up => (config.intermediate_size, config.hidden_size),
                        MlpProjWhich::Down => (config.hidden_size, config.intermediate_size),
                    };
                    // Merge vision LoRA for down_proj if available (Phi-4)
                    let bf16_data = if !lora_map.is_empty() && matches!(which, MlpProjWhich::Down) {
                        let lora_a_name = format!("model.layers.{}.mlp.down_proj.lora_A.vision.weight", layer);
                        let lora_b_name = format!("model.layers.{}.mlp.down_proj.lora_B.vision.weight", layer);
                        if let (Some(a), Some(b)) = (lora_map.get(&lora_a_name), lora_map.get(&lora_b_name)) {
                            merge_lora_bf16(&bf16_data, a, b, lora_scaling, out_f, lora_r, in_f)
                        } else {
                            bf16_data
                        }
                    } else {
                        bf16_data
                    };
                    let weight = quantize_fn(&bf16_data, out_f, in_f, name)?;
                    TensorResult::LayerMlpProj { layer, which, weight }
                }
                TensorKind::LayerMoEGate { layer } => {
                    let bf16_data = load_bf16_auto(&view, name)?;
                    let shape = view.shape();
                    let out_features = shape[0]; // num_experts
                    let in_features = shape[1]; // hidden_size
                    let weight = quantize_fn(&bf16_data, out_features, in_features, name)?;
                    TensorResult::LayerMoEGate { layer, weight }
                }
                TensorKind::LayerExpertProj { layer, expert, which } => {
                    let bf16_data = load_bf16_auto(&view, name)?;
                    let moe_intermediate = config.moe_intermediate_size.ok_or_else(|| {
                        HerbertError::ModelLoad("MoE config missing moe_intermediate_size".into())
                    })?;
                    let (out_f, in_f) = match which {
                        MlpProjWhich::Gate => (moe_intermediate, config.hidden_size),
                        MlpProjWhich::Up => (moe_intermediate, config.hidden_size),
                        MlpProjWhich::Down => (config.hidden_size, moe_intermediate),
                    };
                    let weight = quantize_fn(&bf16_data, out_f, in_f, name)?;
                    TensorResult::LayerExpertProj { layer, expert, which, weight }
                }
                TensorKind::FusedExpertGateUp { layer } => {
                    // Expected shape: [num_experts, 2*moe_inter, hidden_size] (out×in)
                    // Some FP8 models store: [num_experts, hidden_size, 2*moe_inter] (in×out, transposed)
                    let shape = view.shape();
                    let num_experts = shape[0];
                    let moe_inter = config.moe_intermediate_size.unwrap_or(config.intermediate_size);
                    let expected_double_inter = moe_inter * 2;
                    let transposed = shape[1] != expected_double_inter && shape[2] == expected_double_inter;
                    let (double_inter, in_features) = if transposed {
                        (shape[2], shape[1]) // [experts, in, out] → swap
                    } else {
                        (shape[1], shape[2]) // [experts, out, in] → normal
                    };
                    let moe_inter = double_inter / 2;
                    let fused_bf16 = load_bf16_auto(&view, name)?;

                    let mut experts = Vec::with_capacity(num_experts);
                    for e in 0..num_experts {
                        let expert_offset = e * double_inter * in_features;
                        if transposed {
                            // Transpose each expert from [in, out] to [out, in]
                            let expert_data = &fused_bf16[expert_offset..expert_offset + in_features * double_inter];
                            let mut gate_t = vec![0u16; moe_inter * in_features];
                            let mut up_t = vec![0u16; moe_inter * in_features];
                            for row in 0..in_features {
                                for col in 0..moe_inter {
                                    gate_t[col * in_features + row] = expert_data[row * double_inter + col];
                                }
                                for col in 0..moe_inter {
                                    up_t[col * in_features + row] = expert_data[row * double_inter + moe_inter + col];
                                }
                            }
                            let gate_name = format!("model.layers.{}.mlp.experts.{}.gate_proj.weight", layer, e);
                            let up_name = format!("model.layers.{}.mlp.experts.{}.up_proj.weight", layer, e);
                            let gate = quantize_fn(&gate_t, moe_inter, in_features, &gate_name)?;
                            let up = quantize_fn(&up_t, moe_inter, in_features, &up_name)?;
                            experts.push((gate, up));
                        } else {
                            let gate_data = &fused_bf16[expert_offset..expert_offset + moe_inter * in_features];
                            let up_data = &fused_bf16[expert_offset + moe_inter * in_features..expert_offset + double_inter * in_features];
                            let gate_name = format!("model.layers.{}.mlp.experts.{}.gate_proj.weight", layer, e);
                            let up_name = format!("model.layers.{}.mlp.experts.{}.up_proj.weight", layer, e);
                            let gate = quantize_fn(gate_data, moe_inter, in_features, &gate_name)?;
                            let up = quantize_fn(up_data, moe_inter, in_features, &up_name)?;
                            experts.push((gate, up));
                        }
                    }
                    TensorResult::FusedExpertGateUp { layer, experts }
                }
                TensorKind::FusedExpertDown { layer } => {
                    // Expected shape: [num_experts, hidden_size, moe_inter] (out×in)
                    // Some FP8 models store: [num_experts, moe_inter, hidden_size] (in×out, transposed)
                    let shape = view.shape();
                    let num_experts = shape[0];
                    let hidden = config.hidden_size;
                    let transposed = shape[1] != hidden && shape[2] == hidden;
                    let (out_features, in_features) = if transposed {
                        (shape[2], shape[1]) // swap
                    } else {
                        (shape[1], shape[2]) // normal
                    };
                    let fused_bf16 = load_bf16_auto(&view, name)?;

                    let mut experts = Vec::with_capacity(num_experts);
                    for e in 0..num_experts {
                        let expert_offset = e * out_features * in_features;
                        if transposed {
                            // Transpose from [in, out] to [out, in]
                            let expert_data = &fused_bf16[expert_offset..expert_offset + in_features * out_features];
                            let mut down_t = vec![0u16; out_features * in_features];
                            for row in 0..in_features {
                                for col in 0..out_features {
                                    down_t[col * in_features + row] = expert_data[row * out_features + col];
                                }
                            }
                            let down_name = format!("model.layers.{}.mlp.experts.{}.down_proj.weight", layer, e);
                            let down = quantize_fn(&down_t, out_features, in_features, &down_name)?;
                            experts.push(down);
                        } else {
                            let expert_data = &fused_bf16[expert_offset..expert_offset + out_features * in_features];
                            let down_name = format!("model.layers.{}.mlp.experts.{}.down_proj.weight", layer, e);
                            let down = quantize_fn(expert_data, out_features, in_features, &down_name)?;
                            experts.push(down);
                        }
                    }
                    TensorResult::FusedExpertDown { layer, experts }
                }
                TensorKind::LayerConvProj { layer, which } => match which {
                    ConvProjWhich::Conv => {
                        let bf16_data = load_bf16_auto(&view, name)?;
                        let data = bf16_data.iter().map(|&x| bf16_to_f32(x)).collect();
                        TensorResult::LayerConvKernel { layer, data }
                    }
                    ConvProjWhich::In | ConvProjWhich::Out => {
                        let bf16_data = load_bf16_auto(&view, name)?;
                        let (out_f, in_f) = match which {
                            ConvProjWhich::In => (config.hidden_size * 3, config.hidden_size),
                            ConvProjWhich::Out => (config.hidden_size, config.hidden_size),
                            ConvProjWhich::Conv => unreachable!(),
                        };
                        let weight = quantize_fn(&bf16_data, out_f, in_f, name)?;
                        TensorResult::LayerConvProj { layer, which, weight }
                    }
                },
                TensorKind::LayerExpertBias { layer } => {
                    let data = load_f32_from_view(&view, name)?;
                    TensorResult::LayerExpertBias { layer, data }
                }
                TensorKind::FinalNormBias => {
                    let data = load_f32_from_view(&view, name)?;
                    TensorResult::FinalNormBias(data)
                }
                TensorKind::LmHeadBias => {
                    let data = load_f32_from_view(&view, name)?;
                    TensorResult::LmHeadBias(data)
                }
                TensorKind::LayerNormBias { layer, which } => {
                    let data = load_f32_from_view(&view, name)?;
                    TensorResult::LayerNormBias { layer, which, data }
                }
                TensorKind::FusedQKV { layer } => {
                    // Split [q_dim + 2*kv_dim, hidden_size] into Q, K, V before quantization
                    let bf16_data = load_bf16_auto(&view, name)?;
                    let shape = view.shape();
                    let fused_out = shape[0]; // q_dim + kv_dim + kv_dim
                    let in_features = shape[1]; // hidden_size
                    let q_dim = config.q_dim();
                    let kv_dim = config.kv_dim();
                    assert_eq!(fused_out, q_dim + 2 * kv_dim,
                        "FusedQKV shape mismatch: {} != {} + 2*{}", fused_out, q_dim, kv_dim);
                    // Merge vision LoRA if available (Phi-4)
                    let bf16_data = if !lora_map.is_empty() {
                        let lora_a_name = format!("model.layers.{}.self_attn.qkv_proj.lora_A.vision.weight", layer);
                        let lora_b_name = format!("model.layers.{}.self_attn.qkv_proj.lora_B.vision.weight", layer);
                        if let (Some(a), Some(b)) = (lora_map.get(&lora_a_name), lora_map.get(&lora_b_name)) {
                            merge_lora_bf16(&bf16_data, a, b, lora_scaling, fused_out, lora_r, in_features)
                        } else {
                            bf16_data
                        }
                    } else {
                        bf16_data
                    };
                    let q_data = &bf16_data[..q_dim * in_features];
                    let k_data = &bf16_data[q_dim * in_features..(q_dim + kv_dim) * in_features];
                    let v_data = &bf16_data[(q_dim + kv_dim) * in_features..];
                    let q_name = format!("model.layers.{}.self_attn.q_proj.weight", layer);
                    let k_name = format!("model.layers.{}.self_attn.k_proj.weight", layer);
                    let v_name = format!("model.layers.{}.self_attn.v_proj.weight", layer);
                    let q = quantize_fn(q_data, q_dim, in_features, &q_name)?;
                    let k = quantize_fn(k_data, kv_dim, in_features, &k_name)?;
                    let v = quantize_fn(v_data, kv_dim, in_features, &v_name)?;
                    TensorResult::FusedQKVSplit { layer, q, k, v }
                }
                TensorKind::FusedGateUp { layer } => {
                    // Split [2*intermediate_size, hidden_size] into gate, up before quantization
                    let bf16_data = load_bf16_auto(&view, name)?;
                    let shape = view.shape();
                    let fused_out = shape[0]; // 2 * intermediate_size
                    let in_features = shape[1]; // hidden_size
                    // Merge vision LoRA if available (Phi-4)
                    let bf16_data = if !lora_map.is_empty() {
                        let lora_a_name = format!("model.layers.{}.mlp.gate_up_proj.lora_A.vision.weight", layer);
                        let lora_b_name = format!("model.layers.{}.mlp.gate_up_proj.lora_B.vision.weight", layer);
                        if let (Some(a), Some(b)) = (lora_map.get(&lora_a_name), lora_map.get(&lora_b_name)) {
                            merge_lora_bf16(&bf16_data, a, b, lora_scaling, fused_out, lora_r, in_features)
                        } else {
                            bf16_data
                        }
                    } else {
                        bf16_data
                    };
                    let half = fused_out / 2;
                    let gate_data = &bf16_data[..half * in_features];
                    let up_data = &bf16_data[half * in_features..];
                    let gate_name = format!("model.layers.{}.mlp.gate_proj.weight", layer);
                    let up_name = format!("model.layers.{}.mlp.up_proj.weight", layer);
                    let gate = quantize_fn(gate_data, half, in_features, &gate_name)?;
                    let up = quantize_fn(up_data, half, in_features, &up_name)?;
                    TensorResult::FusedGateUpSplit { layer, gate, up }
                }
                // GatedDeltaNet projections (Qwen3.5)
                TensorKind::DeltaNetProjQKV { layer } => {
                    let bf16_data = load_bf16_auto(&view, name)?;
                    let shape = view.shape();
                    let weight = quantize_fn(&bf16_data, shape[0], shape[1], name)?;
                    TensorResult::DeltaNetProjQKV { layer, weight }
                }
                TensorKind::DeltaNetProjZ { layer } => {
                    let bf16_data = load_bf16_auto(&view, name)?;
                    let shape = view.shape();
                    let weight = quantize_fn(&bf16_data, shape[0], shape[1], name)?;
                    TensorResult::DeltaNetProjZ { layer, weight }
                }
                TensorKind::DeltaNetOutProj { layer } => {
                    let bf16_data = load_bf16_auto(&view, name)?;
                    let shape = view.shape();
                    let weight = quantize_fn(&bf16_data, shape[0], shape[1], name)?;
                    TensorResult::DeltaNetOutProj { layer, weight }
                }
                TensorKind::DeltaNetProjA { layer } => {
                    let data = load_f32_from_view(&view, name)?;
                    TensorResult::DeltaNetProjA { layer, data }
                }
                TensorKind::DeltaNetProjB { layer } => {
                    let data = load_f32_from_view(&view, name)?;
                    TensorResult::DeltaNetProjB { layer, data }
                }
                TensorKind::DeltaNetConv1d { layer } => {
                    let data = load_f32_from_view(&view, name)?;
                    TensorResult::DeltaNetConv1d { layer, data }
                }
                TensorKind::DeltaNetALog { layer } => {
                    let data = load_f32_from_view(&view, name)?;
                    TensorResult::DeltaNetALog { layer, data }
                }
                TensorKind::DeltaNetDtBias { layer } => {
                    let data = load_f32_from_view(&view, name)?;
                    TensorResult::DeltaNetDtBias { layer, data }
                }
                TensorKind::DeltaNetNorm { layer } => {
                    let data = load_bf16_auto(&view, name)?;
                    TensorResult::DeltaNetNorm { layer, data }
                }
                // Shared expert (Qwen3.5)
                TensorKind::SharedExpertProj { layer, which } => {
                    let bf16_data = load_bf16_auto(&view, name)?;
                    let shared_inter = config.shared_expert_intermediate_size.unwrap_or(config.intermediate_size);
                    let (out_f, in_f) = match which {
                        MlpProjWhich::Gate => (shared_inter, config.hidden_size),
                        MlpProjWhich::Up => (shared_inter, config.hidden_size),
                        MlpProjWhich::Down => (config.hidden_size, shared_inter),
                    };
                    let weight = quantize_fn(&bf16_data, out_f, in_f, name)?;
                    TensorResult::SharedExpertProj { layer, which, weight }
                }
                TensorKind::SharedExpertGate { layer } => {
                    let data = load_f32_from_view(&view, name)?;
                    TensorResult::SharedExpertGate { layer, data }
                }
                TensorKind::Unknown => TensorResult::Unknown,
            };

            if let Some(ref pb) = tensor_pb {
                pb.inc(1);
            }

            Ok(result)
        };

        let results: Vec<TensorResult<L::Weight>> = std::thread::scope(|s| {
            let handles: Vec<_> = tensor_names
                .chunks(chunk_size.max(1))
                .map(|chunk| {
                    s.spawn(|| {
                        chunk
                            .iter()
                            .map(&process_tensor)
                            .collect::<Result<Vec<_>>>()
                    })
                })
                .collect();

            let mut all_results = Vec::with_capacity(tensor_names.len());
            for handle in handles {
                match handle.join() {
                    Ok(chunk_result) => all_results.extend(chunk_result?),
                    Err(_) => {
                        return Err(HerbertError::ModelLoad(
                            "thread panicked during tensor quantization".into(),
                        ))
                    }
                }
            }
            Ok(all_results)
        })?;

        // --- Sequential merge: store results into accumulators ---
        for result in results {
            let layer_idx = match &result {
                TensorResult::LayerNorm { layer, .. }
                | TensorResult::LayerNormBias { layer, .. }
                | TensorResult::LayerAttnProj { layer, .. }
                | TensorResult::LayerAttnNorm { layer, .. }
                | TensorResult::LayerAttnBias { layer, .. }
                | TensorResult::LayerConvProj { layer, .. }
                | TensorResult::LayerConvKernel { layer, .. }
                | TensorResult::LayerExpertBias { layer, .. }
                | TensorResult::LayerMlpProj { layer, .. }
                | TensorResult::LayerMoEGate { layer, .. }
                | TensorResult::LayerExpertProj { layer, .. }
                | TensorResult::FusedExpertGateUp { layer, .. }
                | TensorResult::FusedExpertDown { layer, .. }
                | TensorResult::FusedQKVSplit { layer, .. }
                | TensorResult::FusedGateUpSplit { layer, .. }
                | TensorResult::DeltaNetProjQKV { layer, .. }
                | TensorResult::DeltaNetProjZ { layer, .. }
                | TensorResult::DeltaNetProjA { layer, .. }
                | TensorResult::DeltaNetProjB { layer, .. }
                | TensorResult::DeltaNetConv1d { layer, .. }
                | TensorResult::DeltaNetALog { layer, .. }
                | TensorResult::DeltaNetDtBias { layer, .. }
                | TensorResult::DeltaNetNorm { layer, .. }
                | TensorResult::DeltaNetOutProj { layer, .. }
                | TensorResult::SharedExpertProj { layer, .. }
                | TensorResult::SharedExpertGate { layer, .. } => Some(*layer),
                _ => None,
            };
            if let Some(l) = layer_idx {
                if l + 1 > max_layer_seen {
                    max_layer_seen = l + 1;
                    if let Some(ref pb) = tensor_pb {
                        pb.set_message(format!(
                            "Loading model (layer {}/{})",
                            max_layer_seen, config.num_layers
                        ));
                    }
                }
            }

            match result {
                TensorResult::Embedding(bf16_data) => {
                    embed_tokens_bf16 = Some(bf16_data);
                }
                TensorResult::FinalNorm(data) => {
                    final_norm = Some(data);
                }
                TensorResult::FinalNormBias(data) => {
                    final_norm_bias = Some(data);
                }
                TensorResult::LmHead(data) => {
                    lm_head_raw = Some(data);
                }
                TensorResult::LmHeadBias(data) => {
                    lm_head_bias = Some(data);
                }
                TensorResult::LayerNorm { layer, which, data } => {
                    let pl = &mut partial_layers[layer];
                    match which {
                        NormWhich::Input => pl.input_layernorm = Some(data),
                        NormWhich::PostAttention => pl.post_attention_layernorm = Some(data),
                        NormWhich::PreFeedForward => pl.pre_feedforward_layernorm = Some(data),
                        NormWhich::PostFeedForward => pl.post_feedforward_layernorm = Some(data),
                    }
                }
                TensorResult::LayerAttnProj { layer, which, weight } => {
                    let pl = &mut partial_layers[layer];
                    match which {
                        AttnProjWhich::Q => pl.q_proj = Some(weight),
                        AttnProjWhich::K => pl.k_proj = Some(weight),
                        AttnProjWhich::V => pl.v_proj = Some(weight),
                        AttnProjWhich::O => pl.o_proj = Some(weight),
                    }
                }
                TensorResult::LayerAttnNorm { layer, which, data } => {
                    let pl = &mut partial_layers[layer];
                    match which {
                        AttnNormWhich::Q => pl.q_norm = Some(data),
                        AttnNormWhich::K => pl.k_norm = Some(data),
                    }
                }
                TensorResult::LayerAttnBias { layer, which, data } => {
                    let pl = &mut partial_layers[layer];
                    match which {
                        AttnProjWhich::Q => pl.q_bias = Some(data),
                        AttnProjWhich::K => pl.k_bias = Some(data),
                        AttnProjWhich::V => pl.v_bias = Some(data),
                        AttnProjWhich::O => pl.o_bias = Some(data),
                    }
                }
                TensorResult::LayerMlpProj { layer, which, weight } => {
                    let pl = &mut partial_layers[layer];
                    match which {
                        MlpProjWhich::Gate => pl.gate_proj = Some(weight),
                        MlpProjWhich::Up => pl.up_proj = Some(weight),
                        MlpProjWhich::Down => pl.down_proj = Some(weight),
                    }
                }
                TensorResult::LayerMoEGate { layer, weight } => {
                    partial_layers[layer].moe_gate = Some(weight);
                }
                TensorResult::LayerExpertProj { layer, expert, which, weight } => {
                    if let Some(ref mut experts) = partial_layers[layer].experts {
                        match which {
                            MlpProjWhich::Gate => experts[expert].gate_proj = Some(weight),
                            MlpProjWhich::Up => experts[expert].up_proj = Some(weight),
                            MlpProjWhich::Down => experts[expert].down_proj = Some(weight),
                        }
                    }
                }
                TensorResult::FusedExpertGateUp { layer, experts: expert_weights } => {
                    let num_experts = expert_weights.len();
                    let pl_experts = partial_layers[layer]
                        .experts
                        .get_or_insert_with(|| (0..num_experts).map(|_| PartialExpert::default()).collect());
                    for (e, (gate, up)) in expert_weights.into_iter().enumerate() {
                        pl_experts[e].gate_proj = Some(gate);
                        pl_experts[e].up_proj = Some(up);
                    }
                }
                TensorResult::FusedExpertDown { layer, experts: expert_weights } => {
                    let num_experts = expert_weights.len();
                    let pl_experts = partial_layers[layer]
                        .experts
                        .get_or_insert_with(|| (0..num_experts).map(|_| PartialExpert::default()).collect());
                    for (e, down) in expert_weights.into_iter().enumerate() {
                        pl_experts[e].down_proj = Some(down);
                    }
                }
                TensorResult::LayerConvProj { layer, which, weight } => {
                    let pl = &mut partial_layers[layer];
                    match which {
                        ConvProjWhich::In => pl.conv_in_proj = Some(weight),
                        ConvProjWhich::Conv => {} // conv kernel handled as LayerConvKernel
                        ConvProjWhich::Out => pl.conv_out_proj = Some(weight),
                    }
                }
                TensorResult::LayerConvKernel { layer, data } => {
                    partial_layers[layer].conv_kernel = Some(data);
                }
                TensorResult::LayerExpertBias { layer, data } => {
                    partial_layers[layer].expert_bias = Some(data);
                }
                TensorResult::LayerNormBias { layer, which, data } => {
                    let pl = &mut partial_layers[layer];
                    match which {
                        NormWhich::Input => pl.input_layernorm_bias = Some(data),
                        NormWhich::PostAttention => pl.post_attention_layernorm_bias = Some(data),
                        NormWhich::PreFeedForward | NormWhich::PostFeedForward => {
                            // Gemma3 pre/post FFN norms don't use bias
                        }
                    }
                }
                TensorResult::FusedQKVSplit { layer, q, k, v } => {
                    let pl = &mut partial_layers[layer];
                    pl.q_proj = Some(q);
                    pl.k_proj = Some(k);
                    pl.v_proj = Some(v);
                }
                TensorResult::FusedGateUpSplit { layer, gate, up } => {
                    let pl = &mut partial_layers[layer];
                    pl.gate_proj = Some(gate);
                    pl.up_proj = Some(up);
                }
                // GatedDeltaNet (Qwen3.5)
                TensorResult::DeltaNetProjQKV { layer, weight } => {
                    partial_layers[layer].deltanet_proj_qkv = Some(weight);
                }
                TensorResult::DeltaNetProjZ { layer, weight } => {
                    partial_layers[layer].deltanet_proj_z = Some(weight);
                }
                TensorResult::DeltaNetOutProj { layer, weight } => {
                    partial_layers[layer].deltanet_out_proj = Some(weight);
                }
                TensorResult::DeltaNetProjA { layer, data } => {
                    partial_layers[layer].deltanet_proj_a = Some(data);
                }
                TensorResult::DeltaNetProjB { layer, data } => {
                    partial_layers[layer].deltanet_proj_b = Some(data);
                }
                TensorResult::DeltaNetConv1d { layer, data } => {
                    partial_layers[layer].deltanet_conv1d = Some(data);
                }
                TensorResult::DeltaNetALog { layer, data } => {
                    partial_layers[layer].deltanet_a_log = Some(data);
                }
                TensorResult::DeltaNetDtBias { layer, data } => {
                    partial_layers[layer].deltanet_dt_bias = Some(data);
                }
                TensorResult::DeltaNetNorm { layer, data } => {
                    partial_layers[layer].deltanet_norm = Some(data);
                }
                // Shared expert (Qwen3.5)
                TensorResult::SharedExpertProj { layer, which, weight } => {
                    let pl = &mut partial_layers[layer];
                    match which {
                        MlpProjWhich::Gate => pl.shared_expert_gate_proj = Some(weight),
                        MlpProjWhich::Up => pl.shared_expert_up_proj = Some(weight),
                        MlpProjWhich::Down => pl.shared_expert_down_proj = Some(weight),
                    }
                }
                TensorResult::SharedExpertGate { layer, data } => {
                    partial_layers[layer].shared_expert_gate = Some(data);
                }
                TensorResult::Unknown => {}
            }
        }
        // shard_mmap is dropped here, releasing the memory mapping
    }

    if let Some(pb) = tensor_pb {
        pb.finish_and_clear();
    }
    // Finalize layers
    let mut decoder_layers = Vec::with_capacity(config.num_layers);
    for (i, pl) in partial_layers.into_iter().enumerate() {
        decoder_layers.push(
            pl.finalize::<L>(&config, i)
                .map_err(|e| HerbertError::ModelLoad(format!("Layer {} incomplete: {}", i, e)))?,
        );
    }

    // LM head
    let lm_head = if config.tie_word_embeddings {
        let emb_bf16 = embed_tokens_bf16
            .as_ref()
            .ok_or_else(|| HerbertError::ModelLoad("missing embed_tokens for tied LM head".into()))?;
        quantize_fn(
            emb_bf16,
            config.vocab_size,
            config.hidden_size,
            "lm_head.weight (tied)",
        )?
    } else if let Some(raw) = lm_head_raw {
        quantize_fn(
            &raw,
            config.vocab_size,
            config.hidden_size,
            "lm_head.weight",
        )?
    } else {
        return Err(HerbertError::ModelLoad(
            "missing lm_head.weight and tie_word_embeddings is false".into(),
        ));
    };

    let embed_tokens =
        embed_tokens_bf16.ok_or_else(|| HerbertError::ModelLoad("missing embed_tokens".into()))?;
    let norm =
        final_norm.ok_or_else(|| HerbertError::ModelLoad("missing model.norm.weight".into()))?;

    // For VL models with MRoPE, compute inv_freq + pattern instead of static cos/sin cache.
    // For text-only models, compute the static cache as before.
    let (cos_cache, sin_cache, inv_freq, mrope_pattern) = if let Some(section) = config.mrope_section {
        let inv_freq = crate::mrope::compute_inv_freq(config.head_dim, config.rope_theta);
        let pattern = crate::mrope::build_mrope_pattern(section);
        (Vec::new(), Vec::new(), inv_freq, pattern)
    } else {
        let (cos, sin) = compute_rope_cache(&config);
        (cos, sin, Vec::new(), Vec::new())
    };

    // Gemma3 VL: dual RoPE caches — sliding layers (theta=10K) vs full layers (theta=1M, linear scaled)
    let (cos_cache, sin_cache, sliding_cos_cache, sliding_sin_cache) = if let Some(sliding_theta) = config.sliding_rope_theta {
        let factor = config.gemma3_rope_factor.unwrap_or(8.0);
        // Main cache: linear-scaled RoPE with theta=1M for full attention layers (every 6th)
        let (main_cos, main_sin) = compute_linear_rope_cache(&config, factor);
        // Sliding cache: standard RoPE with theta=10K for sliding attention layers
        let mut sliding_config = config.clone();
        sliding_config.rope_theta = sliding_theta;
        let (sl_cos, sl_sin) = compute_standard_rope_cache(&sliding_config);
        tracing::info!(sliding_theta, main_theta = config.rope_theta, factor, "Gemma3 VL: dual RoPE caches");
        (main_cos, main_sin, sl_cos, sl_sin)
    } else {
        (cos_cache, sin_cache, Vec::new(), Vec::new())
    };

    let model = GenericModel {
        embed_tokens,
        decoder_layers,
        norm,
        norm_bias: final_norm_bias,
        lm_head,
        lm_head_bias,
        config: config.clone(),
        cos_cache,
        sin_cache,
        sliding_cos_cache,
        sliding_sin_cache,
        inv_freq,
        mrope_pattern,
        _marker: std::marker::PhantomData,
    };

    Ok((config, model))
}
