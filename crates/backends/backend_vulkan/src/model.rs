//! Vulkan GPU model: weight buffer structures for Qwen3 inference.
//!
//! Supports BF16, Int8, and Q4 weight formats. Per-expert separate buffers for MoE.

use crate::memory::VulkanBuffer;

/// An Int8 per-channel quantized weight matrix on GPU.
pub struct VulkanInt8Weight {
    pub packed: VulkanBuffer,   // i8 as bytes [N * K]
    pub scales: VulkanBuffer,   // f32 per-channel scales [N]
    pub n: usize,
    pub k: usize,
}

/// A BF16 weight matrix on GPU.
pub struct VulkanBF16Weight {
    pub packed: VulkanBuffer,   // bf16 as u16 [N * K]
    pub n: usize,
    pub k: usize,
}

/// A Q4 (4-bit) quantized weight matrix on GPU.
///
/// Layout: row-major, packed nibble pairs.
/// - packed: [N, K/2] bytes stored as uint32 words [N * K/8]
///   Each word = 8 nibbles, nibble[i] = (word >> (i*4)) & 0xF, unsigned [0,15]
/// - scales: [N, n_groups] f32, row-major, n_groups = ceil(K/32)
/// - Dequant: value = (float(nibble) - 8.0) * scales[row * n_groups + col/32]
pub struct VulkanQ4Weight {
    pub packed: VulkanBuffer,   // [N * K/8] uint32 words
    pub scales: VulkanBuffer,   // [N * n_groups] f32
    pub n: usize,
    pub k: usize,
}

/// A weight matrix on GPU (Int8, BF16, or Q4).
pub enum VulkanWeight {
    Int8(VulkanInt8Weight),
    BF16(VulkanBF16Weight),
    Q4(VulkanQ4Weight),
}

impl VulkanWeight {
    pub fn n(&self) -> usize {
        match self {
            VulkanWeight::Int8(w) => w.n,
            VulkanWeight::BF16(w) => w.n,
            VulkanWeight::Q4(w) => w.n,
        }
    }
    pub fn k(&self) -> usize {
        match self {
            VulkanWeight::Int8(w) => w.k,
            VulkanWeight::BF16(w) => w.k,
            VulkanWeight::Q4(w) => w.k,
        }
    }
}

/// A single MoE expert's weights (per-expert separate buffers).
pub struct VulkanMoEExpert {
    pub gate_proj: VulkanWeight,
    pub up_proj: VulkanWeight,
    pub down_proj: VulkanWeight,
}

/// MLP variant: dense or MoE.
pub enum VulkanLayerMLP {
    Dense {
        gate_proj: VulkanWeight,
        up_proj: VulkanWeight,
        down_proj: VulkanWeight,
    },
    MoE {
        router: VulkanBuffer,  // [num_experts, hidden_size] f32
        experts: Vec<VulkanMoEExpert>,
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_intermediate_size: usize,
        norm_topk_prob: bool,
    },
}

/// A single Qwen3 decoder layer's weight buffers on GPU.
///
/// All layers have attention (no conv). QK norms always present for Qwen3.
pub struct VulkanLayer {
    pub input_layernorm: VulkanBuffer,
    pub post_attention_layernorm: VulkanBuffer,

    // Attention (always present)
    pub q_proj: VulkanWeight,
    pub k_proj: VulkanWeight,
    pub v_proj: VulkanWeight,
    pub o_proj: VulkanWeight,

    // Per-head QK norms (Qwen3 always has these)
    pub q_norm: VulkanBuffer,
    pub k_norm: VulkanBuffer,

    // MLP (Dense or MoE)
    pub mlp: VulkanLayerMLP,
}

/// Complete Qwen3 model on Vulkan GPU.
pub struct VulkanModel {
    pub embed_tokens: VulkanBuffer,
    pub embed_tokens_f32: Option<Vec<f32>>,
    pub layers: Vec<VulkanLayer>,
    pub final_norm: VulkanBuffer,
    pub lm_head: VulkanWeight,
    pub cos_cache: VulkanBuffer,
    pub sin_cache: VulkanBuffer,
}
