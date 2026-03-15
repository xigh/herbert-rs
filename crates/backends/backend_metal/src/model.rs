//! Metal GPU model: weight buffer structures for Qwen3 inference.
//!
//! Supports BF16, Int8, and Q4 weight formats. No convolution fields (Qwen3 is pure attention).

use crate::memory::MetalBuffer;

/// A BF16 weight matrix on GPU.
pub struct MetalBF16Weight {
    pub packed: MetalBuffer,   // bf16 stored as uint16 pairs in uint32 [N * K / 2]
    pub n: usize,               // output dim
    pub k: usize,               // input dim
}

/// An Int8 per-channel quantized weight matrix on GPU.
pub struct MetalInt8Weight {
    pub packed: MetalBuffer,   // i8 packed as uint32 [N * K / 4]
    pub scales: MetalBuffer,   // f32 per-channel scales [N]
    pub n: usize,               // output dim
    pub k: usize,               // input dim
}

/// A Q4 (4-bit) quantized weight matrix on GPU.
///
/// Layout: row-major, Metal-friendly (different from CPU Q4 AVX-512 layout).
/// - packed: [N, K/2] bytes, each byte = nib_k0 | (nib_k1 << 4), consecutive K
/// - scales: [N, n_groups] f32, row-major, n_groups = ceil(K/32)
/// - Dequant: value = (float(nibble) - 8.0) * scales[row * n_groups + col/32]
pub struct MetalQ4Weight {
    pub packed: MetalBuffer,   // [N, K/2] nibble pairs
    pub scales: MetalBuffer,   // [N, n_groups] f32
    pub n: usize,               // output dim
    pub k: usize,               // input dim
}

/// A weight matrix on GPU (Int8, BF16, or Q4).
pub enum MetalWeight {
    Int8(MetalInt8Weight),
    BF16(MetalBF16Weight),
    Q4(MetalQ4Weight),
}

impl MetalWeight {
    pub fn n(&self) -> usize {
        match self {
            MetalWeight::Int8(w) => w.n,
            MetalWeight::BF16(w) => w.n,
            MetalWeight::Q4(w) => w.n,
        }
    }
    pub fn k(&self) -> usize {
        match self {
            MetalWeight::Int8(w) => w.k,
            MetalWeight::BF16(w) => w.k,
            MetalWeight::Q4(w) => w.k,
        }
    }
}

/// Quantization format tag for MoE contiguous weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoEQuantFormat {
    Q4,
    Int8,
    BF16,
}

/// Contiguous MoE expert weights — all experts packed into flat arrays.
///
/// Memory layout: expert[0] || expert[1] || ... || expert[N-1]
/// Each expert's weight matrix is stored row-major in the same format
/// as individual `MetalWeight` buffers (Q4/Int8/BF16).
///
/// This layout enables batched GPU shaders to index by expert_id
/// without CPU intervention, eliminating per-layer GPU→CPU syncs.
pub struct MetalMoEContiguous {
    /// gate_proj packed: all experts concatenated.
    /// Q4: [ne * gate_n, gate_k/2] bytes; Int8: [ne * gate_n, gate_k]; BF16: [ne * gate_n * gate_k * 2]
    pub gate_packed: MetalBuffer,
    /// gate_proj scales: Q4 [ne * gate_n, n_groups] f32; Int8 [ne * gate_n] f32; None for BF16.
    pub gate_scales: Option<MetalBuffer>,
    /// up_proj packed: same layout as gate.
    pub up_packed: MetalBuffer,
    pub up_scales: Option<MetalBuffer>,
    /// down_proj packed: [ne * down_n, down_k/2] (Q4) etc.
    pub down_packed: MetalBuffer,
    pub down_scales: Option<MetalBuffer>,
    /// Quantization format.
    pub format: MoEQuantFormat,
    /// Per-expert row count for gate/up projections (= moe_intermediate_size).
    pub gate_n: usize,
    /// Per-expert column count for gate/up projections (= hidden_size).
    pub gate_k: usize,
    /// Per-expert row count for down projection (= hidden_size).
    pub down_n: usize,
    /// Per-expert column count for down projection (= moe_intermediate_size).
    pub down_k: usize,
}

impl MetalMoEContiguous {
    /// Create a `MetalWeight` view for a specific expert's gate projection.
    ///
    /// The view shares the underlying contiguous buffer via refcounting.
    pub fn expert_gate(&self, expert_id: usize) -> MetalWeight {
        self.make_expert_weight(
            &self.gate_packed, &self.gate_scales,
            expert_id, self.gate_n, self.gate_k,
        )
    }

    /// Create a `MetalWeight` view for a specific expert's up projection.
    pub fn expert_up(&self, expert_id: usize) -> MetalWeight {
        self.make_expert_weight(
            &self.up_packed, &self.up_scales,
            expert_id, self.gate_n, self.gate_k,
        )
    }

    /// Create a `MetalWeight` view for a specific expert's down projection.
    pub fn expert_down(&self, expert_id: usize) -> MetalWeight {
        self.make_expert_weight(
            &self.down_packed, &self.down_scales,
            expert_id, self.down_n, self.down_k,
        )
    }

    fn make_expert_weight(
        &self,
        packed: &MetalBuffer,
        scales: &Option<MetalBuffer>,
        expert_id: usize,
        n: usize,
        k: usize,
    ) -> MetalWeight {
        match self.format {
            MoEQuantFormat::Q4 => {
                let packed_per_expert = n * (k / 2);
                let n_groups = (k + 31) / 32;
                let scales_per_expert = n * n_groups * 4; // f32 bytes
                MetalWeight::Q4(MetalQ4Weight {
                    packed: packed.slice(expert_id * packed_per_expert, packed_per_expert),
                    scales: scales.as_ref().unwrap().slice(
                        expert_id * scales_per_expert, scales_per_expert,
                    ),
                    n,
                    k,
                })
            }
            MoEQuantFormat::Int8 => {
                let packed_per_expert = n * k;
                let scales_per_expert = n * 4; // f32 per row
                MetalWeight::Int8(MetalInt8Weight {
                    packed: packed.slice(expert_id * packed_per_expert, packed_per_expert),
                    scales: scales.as_ref().unwrap().slice(
                        expert_id * scales_per_expert, scales_per_expert,
                    ),
                    n,
                    k,
                })
            }
            MoEQuantFormat::BF16 => {
                let packed_per_expert = n * k * 2; // 2 bytes per BF16 element
                MetalWeight::BF16(MetalBF16Weight {
                    packed: packed.slice(expert_id * packed_per_expert, packed_per_expert),
                    n,
                    k,
                })
            }
        }
    }
}

/// MLP variant: dense or MoE.
pub enum MetalLayerMLP {
    Dense {
        gate_proj: MetalWeight,
        up_proj: MetalWeight,
        down_proj: MetalWeight,
    },
    MoE {
        router: MetalBuffer,  // [num_experts, hidden_size] f32
        weights: MetalMoEContiguous,
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_intermediate_size: usize,
        norm_topk_prob: bool,
    },
}

/// A single Qwen3 decoder layer's weight buffers on GPU.
///
/// All layers have attention (no conv layers). QK norms always present for Qwen3.
pub struct MetalLayer {
    pub input_layernorm: MetalBuffer,
    pub post_attention_layernorm: MetalBuffer,

    // Attention (always present for Qwen3)
    pub q_proj: MetalWeight,
    pub k_proj: MetalWeight,
    pub v_proj: MetalWeight,
    pub o_proj: MetalWeight,

    // Per-head QK norms (Qwen3 has these, Mistral3 does not)
    pub q_norm: Option<MetalBuffer>,
    pub k_norm: Option<MetalBuffer>,

    // MLP (Dense or MoE)
    pub mlp: MetalLayerMLP,
}

/// Complete model on Metal GPU.
pub struct MetalModel {
    pub embed_tokens: MetalBuffer,
    pub embed_tokens_f32: Option<Vec<f32>>,
    pub layers: Vec<MetalLayer>,
    pub final_norm: MetalBuffer,
    pub lm_head: MetalWeight,
    pub cos_cache: MetalBuffer,
    pub sin_cache: MetalBuffer,
    /// Dummy 4-byte buffer used as placeholder when QK norms are absent (Mistral3).
    pub dummy_norm_buf: MetalBuffer,
}
