//! RoPE (Rotary Position Embedding) kernel dispatch wrappers.

use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::context::MetalContext;
use crate::memory::MetalBuffer;
use super::{dispatch_kernel, div_ceil};

/// RoPE for a single position (decode step, in-place).
pub fn rope_single(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    qk: &MetalBuffer,
    cos: &MetalBuffer,
    sin: &MetalBuffer,
    num_heads: u32,
    half_dim: u32,
) {
    let total = num_heads * half_dim;

    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    push[4..8].copy_from_slice(&half_dim.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.rope_single,
        &[qk, cos, sin],
        &push,
        MTLSize { width: div_ceil(total, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// RoPE for a batch of positions (prefill, in-place).
pub fn rope_batch(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    qk: &MetalBuffer,
    cos_cache: &MetalBuffer,
    sin_cache: &MetalBuffer,
    seq_len: u32,
    num_heads: u32,
    half_dim: u32,
    start_pos: u32,
) {
    let total = seq_len * num_heads * half_dim;

    let mut push = [0u8; 16];
    push[0..4].copy_from_slice(&seq_len.to_le_bytes());
    push[4..8].copy_from_slice(&num_heads.to_le_bytes());
    push[8..12].copy_from_slice(&half_dim.to_le_bytes());
    push[12..16].copy_from_slice(&start_pos.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.rope_batch,
        &[qk, cos_cache, sin_cache],
        &push,
        MTLSize { width: div_ceil(total, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Fused RoPE + KV cache append (decode, single token).
pub fn rope_kv_append(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    k: &MetalBuffer,
    cos_cache: &MetalBuffer,
    sin_cache: &MetalBuffer,
    k_cache: &MetalBuffer,
    num_heads: u32,
    half_dim: u32,
    kv_dim: u32,
    cache_pos: u32,
) {
    let total = num_heads * half_dim;

    let mut push = [0u8; 16];
    push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    push[4..8].copy_from_slice(&half_dim.to_le_bytes());
    push[8..12].copy_from_slice(&kv_dim.to_le_bytes());
    push[12..16].copy_from_slice(&cache_pos.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.rope_kv_append,
        &[k, cos_cache, sin_cache, k_cache],
        &push,
        MTLSize { width: div_ceil(total, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}
