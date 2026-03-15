//! RMS normalization kernel dispatch wrappers.

use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::context::MetalContext;
use crate::memory::MetalBuffer;
use super::dispatch_kernel;

/// RMS normalization (single vector, 256 threads with threadgroup reduction).
pub fn rms_norm(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &MetalBuffer,
    weight: &MetalBuffer,
    output: &MetalBuffer,
    dim: u32,
    eps: f32,
) {
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&dim.to_le_bytes());
    push[4..8].copy_from_slice(&eps.to_le_bytes());

    // RMS norm half accumulation causes numerical instability (half overflow on sum_sq
    // when activations > 256). Keep float accumulation.
    dispatch_kernel(
        encoder,
        &ctx.pipelines.rms_norm,
        &[input, weight, output],
        &push,
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Batched RMS normalization (for prefill).
pub fn rms_norm_batch(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &MetalBuffer,
    weight: &MetalBuffer,
    output: &MetalBuffer,
    dim: u32,
    eps: f32,
    batch_size: u32,
) {
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&dim.to_le_bytes());
    push[4..8].copy_from_slice(&eps.to_le_bytes());
    push[8..12].copy_from_slice(&batch_size.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.rms_norm_batch,
        &[input, weight, output],
        &push,
        MTLSize { width: batch_size as usize, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// Fused residual add + RMS normalization (decode, single vector, 256 threads).
pub fn rms_norm_residual(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    a: &MetalBuffer,
    b: &MetalBuffer,
    weight: &MetalBuffer,
    output: &MetalBuffer,
    dim: u32,
    eps: f32,
) {
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&dim.to_le_bytes());
    push[4..8].copy_from_slice(&eps.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.rms_norm_residual,
        &[a, b, weight, output],
        &push,
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Per-head RMS normalization (for QK norms, in-place).
pub fn head_rms_norm(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    data: &MetalBuffer,
    weight: &MetalBuffer,
    num_heads: u32,
    head_dim: u32,
    eps: f32,
) {
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    push[4..8].copy_from_slice(&head_dim.to_le_bytes());
    push[8..12].copy_from_slice(&eps.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.head_rms_norm,
        &[data, weight],
        &push,
        MTLSize { width: num_heads as usize, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// Batched per-head RMS normalization for prefill.
pub fn head_rms_norm_batch(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    data: &MetalBuffer,
    weight: &MetalBuffer,
    num_heads: u32,
    head_dim: u32,
    seq_len: u32,
    eps: f32,
) {
    let total_heads = seq_len * num_heads;

    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&total_heads.to_le_bytes());
    push[4..8].copy_from_slice(&head_dim.to_le_bytes());
    push[8..12].copy_from_slice(&eps.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.head_rms_norm,
        &[data, weight],
        &push,
        MTLSize { width: total_heads as usize, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// Fused per-head RMS norm + RoPE for prefill (batch of seq_len tokens, in-place).
/// Replaces separate head_rms_norm_batch + rope_batch dispatches.
/// When `skip_norm` is true, RMS norm is skipped (Mistral3 has no QK norms).
pub fn head_norm_rope_batch(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    data: &MetalBuffer,
    weight: &MetalBuffer,
    cos_cache: &MetalBuffer,
    sin_cache: &MetalBuffer,
    num_heads: u32,
    head_dim: u32,
    half_dim: u32,
    seq_len: u32,
    start_pos: u32,
    eps: f32,
    skip_norm: bool,
) {
    let total_heads = seq_len * num_heads;
    let skip_norm_u32: u32 = if skip_norm { 1 } else { 0 };

    let mut push = [0u8; 28];
    push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    push[4..8].copy_from_slice(&head_dim.to_le_bytes());
    push[8..12].copy_from_slice(&half_dim.to_le_bytes());
    push[12..16].copy_from_slice(&seq_len.to_le_bytes());
    push[16..20].copy_from_slice(&start_pos.to_le_bytes());
    push[20..24].copy_from_slice(&eps.to_le_bytes());
    push[24..28].copy_from_slice(&skip_norm_u32.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.head_norm_rope_batch,
        &[data, weight, cos_cache, sin_cache],
        &push,
        MTLSize { width: total_heads as usize, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// Fused per-head RMS norm + RoPE for Q vectors (decode only).
/// When `skip_norm` is true, RMS norm is skipped (Mistral3 has no QK norms).
pub fn head_norm_rope(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    data: &MetalBuffer,
    weight: &MetalBuffer,
    cos_cache: &MetalBuffer,
    sin_cache: &MetalBuffer,
    num_heads: u32,
    head_dim: u32,
    half_dim: u32,
    pos: u32,
    eps: f32,
    skip_norm: bool,
) {
    let skip_norm_u32: u32 = if skip_norm { 1 } else { 0 };

    let mut push = [0u8; 24];
    push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    push[4..8].copy_from_slice(&head_dim.to_le_bytes());
    push[8..12].copy_from_slice(&half_dim.to_le_bytes());
    push[12..16].copy_from_slice(&pos.to_le_bytes());
    push[16..20].copy_from_slice(&eps.to_le_bytes());
    push[20..24].copy_from_slice(&skip_norm_u32.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.head_norm_rope,
        &[data, weight, cos_cache, sin_cache],
        &push,
        MTLSize { width: num_heads as usize, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// Fused per-head RMS norm + RoPE + KV cache append for K and V vectors (decode only).
///
/// `cache_pos`: KV cache write slot. `rope_pos`: RoPE cos/sin lookup position.
/// For text-only models these are always equal. For VL models after VL prefill,
/// `cache_pos = seq_len` (total tokens) while `rope_pos = text_token_count`.
/// When `skip_norm` is true, RMS norm is skipped (Mistral3 has no QK norms).
pub fn head_norm_rope_kv_append(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    k_data: &MetalBuffer,
    weight: &MetalBuffer,
    cos_cache: &MetalBuffer,
    sin_cache: &MetalBuffer,
    k_cache: &MetalBuffer,
    v_data: &MetalBuffer,
    v_cache: &MetalBuffer,
    num_heads: u32,
    head_dim: u32,
    half_dim: u32,
    kv_dim: u32,
    cache_pos: u32,
    rope_pos: u32,
    eps: f32,
    skip_norm: bool,
) {
    let skip_norm_u32: u32 = if skip_norm { 1 } else { 0 };

    let mut push = [0u8; 32];
    push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    push[4..8].copy_from_slice(&head_dim.to_le_bytes());
    push[8..12].copy_from_slice(&half_dim.to_le_bytes());
    push[12..16].copy_from_slice(&kv_dim.to_le_bytes());
    push[16..20].copy_from_slice(&cache_pos.to_le_bytes());
    push[20..24].copy_from_slice(&rope_pos.to_le_bytes());
    push[24..28].copy_from_slice(&eps.to_le_bytes());
    push[28..32].copy_from_slice(&skip_norm_u32.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.head_norm_rope_kv_append,
        &[k_data, weight, cos_cache, sin_cache, k_cache, v_data, v_cache],
        &push,
        MTLSize { width: num_heads as usize, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// Fused per-head RMS norm + RoPE + INT8 KV cache append for K and V vectors.
///
/// Same as `head_norm_rope_kv_append` but quantizes to INT8 with per-head scales.
/// When `skip_norm` is true, RMS norm is skipped (Mistral3 has no QK norms).
pub fn head_norm_rope_kv_append_i8(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    k_data: &MetalBuffer,
    weight: &MetalBuffer,
    cos_cache: &MetalBuffer,
    sin_cache: &MetalBuffer,
    k_cache: &MetalBuffer,
    k_scales: &MetalBuffer,
    v_data: &MetalBuffer,
    v_cache: &MetalBuffer,
    v_scales: &MetalBuffer,
    num_heads: u32,
    head_dim: u32,
    half_dim: u32,
    kv_dim: u32,
    cache_pos: u32,
    rope_pos: u32,
    eps: f32,
    skip_norm: bool,
) {
    let skip_norm_u32: u32 = if skip_norm { 1 } else { 0 };

    let mut push = [0u8; 32];
    push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    push[4..8].copy_from_slice(&head_dim.to_le_bytes());
    push[8..12].copy_from_slice(&half_dim.to_le_bytes());
    push[12..16].copy_from_slice(&kv_dim.to_le_bytes());
    push[16..20].copy_from_slice(&cache_pos.to_le_bytes());
    push[20..24].copy_from_slice(&rope_pos.to_le_bytes());
    push[24..28].copy_from_slice(&eps.to_le_bytes());
    push[28..32].copy_from_slice(&skip_norm_u32.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.head_norm_rope_kv_append_i8,
        &[k_data, weight, cos_cache, sin_cache,
          k_cache, k_scales, v_data, v_cache, v_scales],
        &push,
        MTLSize { width: num_heads as usize, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

