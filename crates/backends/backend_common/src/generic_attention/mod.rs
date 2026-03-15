//! Generic attention module parameterized over LinearOps.

use std::marker::PhantomData;

pub(crate) use herbert_core::error::{HerbertError, Result};
pub(crate) use herbert_core::tensor::{bf16_to_f32, BF16};

pub(crate) use crate::attention_common::{
    decode_head_attention, decode_head_attention_f32, decode_head_attention_int8,
    decode_head_attention_int4,
    decode_head_parallel_workers, should_parallelize_decode_heads,
};
pub(crate) use crate::kv_cache::{CpuKvCache, KvLayerData};
pub(crate) use crate::linear_ops::LinearOps;
pub(crate) use crate::thread_pool::{global_pool, SendMutPtr, SendPtr};

mod ffi;
#[cfg(target_arch = "x86_64")]
use ffi::*;
#[cfg(target_arch = "x86_64")]
pub(crate) use crate::kernels::{has_avx512f, has_avx512bf16, has_avx2_fma};

mod decode;
pub mod prefill;

/// Attention module generic over the linear ops / weight type.
pub struct GenericAttention<L: LinearOps> {
    pub q_proj: L::Weight,
    pub k_proj: L::Weight,
    pub v_proj: L::Weight,
    pub o_proj: L::Weight,
    pub q_norm: Option<Vec<BF16>>,
    pub k_norm: Option<Vec<BF16>>,
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    pub o_bias: Option<Vec<f32>>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_ndims: usize,
    pub head_to_kv_head: Vec<usize>,
    pub hidden_size: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub rms_norm_eps: f32,
    /// SmolLM3 NoPE: false = skip RoPE for this layer.
    pub use_rope: bool,
    /// Gemma3: explicit attention scale (None = 1/sqrt(head_dim)).
    pub attn_scale: Option<f32>,
    /// Gemma3: use (1 + weight) * rms_norm(x) for QK norms.
    pub use_gemma_qk_norm: bool,
    /// Qwen3.5: q_proj is fused Q+gate, apply sigmoid output gating.
    pub has_output_gate: bool,
    /// Use Q BF16 format for attention dot products (VDPBF16PS optimization).
    /// Set true when --q-format bf16 is passed AND avx512bf16 is available.
    pub use_q_bf16: bool,
    pub _marker: PhantomData<L>,
}

impl<L: LinearOps> GenericAttention<L> {
    pub(super) fn head_rms_norm_inplace(&self, x: &mut [f32], weight: &[BF16], num_heads: usize) {
        for h in 0..num_heads {
            let offset = h * self.head_dim;
            let slice = &mut x[offset..offset + self.head_dim];
            let mut sum_sq = 0.0f32;
            for &v in slice.iter() {
                sum_sq += v * v;
            }
            let rms = (sum_sq / self.head_dim as f32 + self.rms_norm_eps).sqrt();
            let inv_rms = 1.0 / rms;
            for (i, v) in slice.iter_mut().enumerate() {
                *v = *v * inv_rms * bf16_to_f32(weight[i]);
            }
        }
    }

    /// Gemma-style per-head RMS norm: (1 + weight) * rms_norm(x).
    pub(super) fn head_rms_norm_gemma_inplace(&self, x: &mut [f32], weight: &[BF16], num_heads: usize) {
        for h in 0..num_heads {
            let offset = h * self.head_dim;
            let slice = &mut x[offset..offset + self.head_dim];
            let mut sum_sq = 0.0f32;
            for &v in slice.iter() {
                sum_sq += v * v;
            }
            let inv_rms = 1.0 / (sum_sq / self.head_dim as f32 + self.rms_norm_eps).sqrt();
            for (i, v) in slice.iter_mut().enumerate() {
                *v = *v * inv_rms * (1.0 + bf16_to_f32(weight[i]));
            }
        }
    }
}
