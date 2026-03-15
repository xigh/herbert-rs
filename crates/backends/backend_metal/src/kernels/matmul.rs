//! Matrix-matrix multiply kernel dispatch wrappers (prefill path, M>1).

use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::context::MetalContext;
use crate::memory::MetalBuffer;
use super::{dispatch_kernel, dispatch_kernel_with_tgmem, div_ceil};

/// BF16 matrix-matrix multiply.
pub fn bf16_matmul(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    a: &MetalBuffer,
    w_packed: &MetalBuffer,
    c: &MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&m.to_le_bytes());
    push[4..8].copy_from_slice(&k.to_le_bytes());
    push[8..12].copy_from_slice(&n.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.bf16_matmul,
        &[a, w_packed, c],
        &push,
        MTLSize { width: n as usize, height: m as usize, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// Float32 matrix-matrix multiply.
pub fn f32_matmul(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    a: &MetalBuffer,
    w: &MetalBuffer,
    c: &MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&m.to_le_bytes());
    push[4..8].copy_from_slice(&k.to_le_bytes());
    push[8..12].copy_from_slice(&n.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.f32_matmul,
        &[a, w, c],
        &push,
        MTLSize { width: n as usize, height: m as usize, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// Int8 per-channel matrix-matrix multiply.
pub fn int8_matmul(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    a: &MetalBuffer,
    w_packed: &MetalBuffer,
    scales: &MetalBuffer,
    c: &MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&m.to_le_bytes());
    push[4..8].copy_from_slice(&k.to_le_bytes());
    push[8..12].copy_from_slice(&n.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.int8_matmul,
        &[a, w_packed, scales, c],
        &push,
        MTLSize { width: n as usize, height: m as usize, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// Q4 (4-bit) matrix-matrix multiply.
///
/// Buffers: A (f32), packed (nibble pairs), scales (f32), C (f32).
/// Push constants: M (u32), K (u32), N (u32).
pub fn q4_matmul(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    a: &MetalBuffer,
    packed: &MetalBuffer,
    scales: &MetalBuffer,
    c: &MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&m.to_le_bytes());
    push[4..8].copy_from_slice(&k.to_le_bytes());
    push[8..12].copy_from_slice(&n.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.q4_matmul,
        &[a, packed, scales, c],
        &push,
        MTLSize { width: n as usize, height: m as usize, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// Tiled Q4 matrix-matrix multiply (shared memory tiling for better data reuse).
///
/// On Apple11+ with Metal 4: cooperative_tensor 32×32, 128 threads.
/// With `simdgroup-matmul` feature: 16×32 tiles, 256 threads, simdgroup_matrix.
/// Without (default): 4×4 tiles, 128 threads, classic shared memory tiling.
pub fn q4_matmul_tiled(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    a: &MetalBuffer,
    packed: &MetalBuffer,
    scales: &MetalBuffer,
    c: &MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&m.to_le_bytes());
    push[4..8].copy_from_slice(&n.to_le_bytes());
    push[8..12].copy_from_slice(&k.to_le_bytes());

    // Metal 4 MPP path: matmul2d Neural Accelerators, 32×32×32 tiles
    // Double-buffered + dequant 4-by-4: 1.7-1.9× faster than coop32
    // 4 simdgroups × 32 threads = 128 threads
    // TG mem: 2×A(2048) + 2×W(2048) + C(4096) = 12288 bytes
    if let Some(ref pipeline) = ctx.pipelines.q4_matmul_tiled_mpp {
        dispatch_kernel_with_tgmem(
            encoder,
            pipeline,
            &[a, packed, scales, c],
            &push,
            12288,
            MTLSize { width: div_ceil(n, 32) as usize, height: div_ceil(m, 32) as usize, depth: 1 },
            MTLSize { width: 128, height: 1, depth: 1 },
        );
        return;
    }

    // Metal 4 coop32 path: 32×16 output tile, 256 threads (8 simdgroups: 4×2)
    if let Some(ref pipeline) = ctx.pipelines.q4_matmul_tiled_coop32 {
        dispatch_kernel(
            encoder,
            pipeline,
            &[a, packed, scales, c],
            &push,
            MTLSize { width: div_ceil(n, 16) as usize, height: div_ceil(m, 32) as usize, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        return;
    }

    // Fallback: simdgroup_matrix or classic tiling
    #[cfg(feature = "simdgroup-matmul")]
    let (grid, threads) = (
        MTLSize { width: div_ceil(n, 32) as usize, height: div_ceil(m, 16) as usize, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
    #[cfg(not(feature = "simdgroup-matmul"))]
    let (grid, threads) = (
        MTLSize { width: div_ceil(n, 4) as usize, height: div_ceil(m, 4) as usize, depth: 1 },
        MTLSize { width: 128, height: 1, depth: 1 },
    );

    dispatch_kernel(
        encoder,
        &ctx.pipelines.q4_matmul_tiled,
        &[a, packed, scales, c],
        &push,
        grid,
        threads,
    );
}

/// Tiled BF16 matrix-matrix multiply (shared memory tiling for better data reuse).
///
/// On Apple11+ with Metal 4: cooperative_tensor 32×32, 128 threads.
/// Fallback: 4×4 tiling, 128 threads.
pub fn bf16_matmul_tiled(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    a: &MetalBuffer,
    w_packed: &MetalBuffer,
    c: &MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&m.to_le_bytes());
    push[4..8].copy_from_slice(&n.to_le_bytes());
    push[8..12].copy_from_slice(&k.to_le_bytes());

    // Metal 4 MPP path: matmul2d 32×32×32, double-buffered, 128 threads
    if let Some(ref pipeline) = ctx.pipelines.bf16_matmul_tiled_mpp {
        dispatch_kernel_with_tgmem(
            encoder,
            pipeline,
            &[a, w_packed, c],
            &push,
            12288,
            MTLSize { width: div_ceil(n, 32) as usize, height: div_ceil(m, 32) as usize, depth: 1 },
            MTLSize { width: 128, height: 1, depth: 1 },
        );
        return;
    }

    // Metal 4 coop32 path: 32×16 output tile, 256 threads (8 simdgroups: 4×2)
    if let Some(ref pipeline) = ctx.pipelines.bf16_matmul_tiled_coop32 {
        dispatch_kernel(
            encoder,
            pipeline,
            &[a, w_packed, c],
            &push,
            MTLSize { width: div_ceil(n, 16) as usize, height: div_ceil(m, 32) as usize, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        return;
    }

    dispatch_kernel(
        encoder,
        &ctx.pipelines.bf16_matmul_tiled,
        &[a, w_packed, c],
        &push,
        MTLSize { width: div_ceil(n, 4) as usize, height: div_ceil(m, 4) as usize, depth: 1 },
        MTLSize { width: 128, height: 1, depth: 1 },
    );
}
