//! Matrix-vector multiply kernel dispatch wrappers (decode path, M=1).
//!
//! All matvec kernels use 4-row-per-threadgroup dispatch (128 threads = 4 warps).

use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::context::MetalContext;
use crate::memory::MetalBuffer;
use super::{dispatch_kernel_with_tgmem, div_ceil};

/// BF16 matrix-vector multiply.
pub fn bf16_matvec(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &MetalBuffer,
    w_packed: &MetalBuffer,
    y: &MetalBuffer,
    n: u32,
    k: u32,
) {
    let push = k.to_le_bytes();

    dispatch_kernel_with_tgmem(
        encoder,
        &ctx.pipelines.bf16_matvec,
        &[x, w_packed, y],
        &push,
        k as usize * 4,
        MTLSize { width: div_ceil(n, 4) as usize, height: 1, depth: 1 },
        MTLSize { width: 128, height: 1, depth: 1 },
    );
}

/// Float32 matrix-vector multiply (for MoE router gate).
pub fn f32_matvec(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &MetalBuffer,
    w: &MetalBuffer,
    y: &MetalBuffer,
    n: u32,
    k: u32,
) {
    let push = k.to_le_bytes();

    dispatch_kernel_with_tgmem(
        encoder,
        &ctx.pipelines.f32_matvec,
        &[x, w, y],
        &push,
        k as usize * 4,
        MTLSize { width: div_ceil(n, 4) as usize, height: 1, depth: 1 },
        MTLSize { width: 128, height: 1, depth: 1 },
    );
}

/// Int8 per-channel matrix-vector multiply.
pub fn int8_matvec(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &MetalBuffer,
    w_packed: &MetalBuffer,
    scales: &MetalBuffer,
    y: &MetalBuffer,
    n: u32,
    k: u32,
) {
    let push = k.to_le_bytes();

    dispatch_kernel_with_tgmem(
        encoder,
        &ctx.pipelines.int8_matvec,
        &[x, w_packed, scales, y],
        &push,
        k as usize * 4,
        MTLSize { width: div_ceil(n, 4) as usize, height: 1, depth: 1 },
        MTLSize { width: 128, height: 1, depth: 1 },
    );
}

/// Fused Q/K/V projection for Q4 weights in a single dispatch.
///
/// Eliminates 2 dispatch overheads per layer.
/// Grid: (ceil(max(n_q,n_k,n_v)/4), 3, 1) — gid.y selects Q(0)/K(1)/V(2).
pub fn q4_matvec_qkv(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &MetalBuffer,
    q_packed: &MetalBuffer, q_scales: &MetalBuffer, out_q: &MetalBuffer, n_q: u32,
    k_packed: &MetalBuffer, k_scales: &MetalBuffer, out_k: &MetalBuffer, n_k: u32,
    v_packed: &MetalBuffer, v_scales: &MetalBuffer, out_v: &MetalBuffer, n_v: u32,
    k_dim: u32,
) {
    let mut push = [0u8; 16];
    push[0..4].copy_from_slice(&n_q.to_le_bytes());
    push[4..8].copy_from_slice(&n_k.to_le_bytes());
    push[8..12].copy_from_slice(&n_v.to_le_bytes());
    push[12..16].copy_from_slice(&k_dim.to_le_bytes());

    let max_n = n_q.max(n_k).max(n_v);

    dispatch_kernel_with_tgmem(
        encoder,
        &ctx.pipelines.q4_matvec_qkv,
        &[x, q_packed, q_scales, k_packed, k_scales, v_packed, v_scales, out_q, out_k, out_v],
        &push,
        k_dim as usize * 2,
        MTLSize { width: div_ceil(max_n, 4) as usize, height: 3, depth: 1 },
        MTLSize { width: 128, height: 1, depth: 1 },
    );
}

/// Fused RMS norm + Q/K/V projection for Q4 weights in a single dispatch.
///
/// Eliminates norm1 dispatch per layer.
/// Grid: (ceil(max(n_q,n_k,n_v)/4), 3, 1) — gid.y selects Q(0)/K(1)/V(2).
pub fn q4_matvec_qkv_normed(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &MetalBuffer,
    norm_weight: &MetalBuffer,
    q_packed: &MetalBuffer, q_scales: &MetalBuffer, out_q: &MetalBuffer, n_q: u32,
    k_packed: &MetalBuffer, k_scales: &MetalBuffer, out_k: &MetalBuffer, n_k: u32,
    v_packed: &MetalBuffer, v_scales: &MetalBuffer, out_v: &MetalBuffer, n_v: u32,
    k_dim: u32,
    eps: f32,
) {
    let mut push = [0u8; 20];
    push[0..4].copy_from_slice(&n_q.to_le_bytes());
    push[4..8].copy_from_slice(&n_k.to_le_bytes());
    push[8..12].copy_from_slice(&n_v.to_le_bytes());
    push[12..16].copy_from_slice(&k_dim.to_le_bytes());
    push[16..20].copy_from_slice(&eps.to_le_bytes());

    let max_n = n_q.max(n_k).max(n_v);

    dispatch_kernel_with_tgmem(
        encoder,
        &ctx.pipelines.q4_matvec_qkv_normed,
        &[x, norm_weight, q_packed, q_scales, k_packed, k_scales, v_packed, v_scales, out_q, out_k, out_v],
        &push,
        k_dim as usize * 4,
        MTLSize { width: div_ceil(max_n, 4) as usize, height: 3, depth: 1 },
        MTLSize { width: 128, height: 1, depth: 1 },
    );
}

/// Q4 (4-bit) matrix-vector multiply.
///
/// Buffers: x (f32), packed (nibble pairs), scales (f32), y (f32).
/// Push constants: K (u32).
pub fn q4_matvec(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &MetalBuffer,
    packed: &MetalBuffer,
    scales: &MetalBuffer,
    y: &MetalBuffer,
    n: u32,
    k: u32,
) {
    let push = k.to_le_bytes();

    dispatch_kernel_with_tgmem(
        encoder,
        &ctx.pipelines.q4_matvec,
        &[x, packed, scales, y],
        &push,
        k as usize * 2,
        MTLSize { width: div_ceil(n, 4) as usize, height: 1, depth: 1 },
        MTLSize { width: 128, height: 1, depth: 1 },
    );
}

/// Q4 matrix-vector multiply with fused residual add: output = residual + W*x.
pub fn q4_matvec_residual_add(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &MetalBuffer,
    packed: &MetalBuffer,
    scales: &MetalBuffer,
    residual: &MetalBuffer,
    output: &MetalBuffer,
    n: u32,
    k: u32,
) {
    let push = k.to_le_bytes();

    dispatch_kernel_with_tgmem(
        encoder,
        &ctx.pipelines.q4_matvec_residual_add,
        &[x, packed, scales, residual, output],
        &push,
        k as usize * 2,
        MTLSize { width: div_ceil(n, 4) as usize, height: 1, depth: 1 },
        MTLSize { width: 128, height: 1, depth: 1 },
    );
}

/// Q4 matvec v2: 8 rows/TG, arithmetic dequant, uint4 loads.
pub fn q4_matvec_v2(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &MetalBuffer,
    packed: &MetalBuffer,
    scales: &MetalBuffer,
    y: &MetalBuffer,
    n: u32,
    k: u32,
) {
    let pipeline = ctx.pipelines.q4_matvec_v2.as_ref().unwrap();
    let push = k.to_le_bytes();

    dispatch_kernel_with_tgmem(
        encoder,
        pipeline,
        &[x, packed, scales, y],
        &push,
        k as usize * 2,
        MTLSize { width: div_ceil(n, 8) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Q4 matvec v2 with fused residual add: output = residual + W*x.
pub fn q4_matvec_residual_add_v2(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &MetalBuffer,
    packed: &MetalBuffer,
    scales: &MetalBuffer,
    residual: &MetalBuffer,
    output: &MetalBuffer,
    n: u32,
    k: u32,
) {
    let pipeline = ctx.pipelines.q4_matvec_residual_add_v2.as_ref().unwrap();
    let push = k.to_le_bytes();

    dispatch_kernel_with_tgmem(
        encoder,
        pipeline,
        &[x, packed, scales, residual, output],
        &push,
        k as usize * 2,
        MTLSize { width: div_ceil(n, 8) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Q4 matrix-vector multiply with fused residual add, 8 rows per threadgroup.
pub fn q4_matvec_residual_add_8row(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &MetalBuffer,
    packed: &MetalBuffer,
    scales: &MetalBuffer,
    residual: &MetalBuffer,
    output: &MetalBuffer,
    n: u32,
    k: u32,
) {
    let push = k.to_le_bytes();

    dispatch_kernel_with_tgmem(
        encoder,
        &ctx.pipelines.q4_matvec_residual_add_8row,
        &[x, packed, scales, residual, output],
        &push,
        k as usize * 2,
        MTLSize { width: div_ceil(n, 8) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}
