//! Matrix-matrix multiply kernel dispatch wrappers (prefill path, M>1).

use ash::vk;
use crate::context::VulkanContext;
use crate::memory::VulkanBuffer;

pub fn int8_matmul(ctx: &VulkanContext, cb: vk::CommandBuffer, a: &VulkanBuffer, w_packed: &VulkanBuffer, scales: &VulkanBuffer, c: &VulkanBuffer, m: u32, n: u32, k: u32) {
    let p = &ctx.pipelines.int8_matmul;
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&m.to_le_bytes());
    push[4..8].copy_from_slice(&k.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(a.buffer, a.size), (w_packed.buffer, w_packed.size), (scales.buffer, scales.size), (c.buffer, c.size)],
        &push, n, m, 1);
}

pub fn bf16_matmul(ctx: &VulkanContext, cb: vk::CommandBuffer, a: &VulkanBuffer, w_packed: &VulkanBuffer, c: &VulkanBuffer, m: u32, n: u32, k: u32) {
    let p = &ctx.pipelines.bf16_matmul;
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&m.to_le_bytes());
    push[4..8].copy_from_slice(&k.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(a.buffer, a.size), (w_packed.buffer, w_packed.size), (c.buffer, c.size)],
        &push, n, m, 1);
}

pub fn f32_matmul(ctx: &VulkanContext, cb: vk::CommandBuffer, a: &VulkanBuffer, w: &VulkanBuffer, c: &VulkanBuffer, m: u32, n: u32, k: u32) {
    let p = &ctx.pipelines.f32_matmul;
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&m.to_le_bytes());
    push[4..8].copy_from_slice(&k.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(a.buffer, a.size), (w.buffer, w.size), (c.buffer, c.size)],
        &push, n, m, 1);
}

/// Minimum M for cooperative matrix dispatch (tiles are 16×16, below this threshold
/// the tile waste makes scalar v5 faster). Benchmark-validated on AMD 780M RDNA3.
const COOPMAT_M_THRESHOLD: u32 = 16;

/// Maximum N/K ratio for cooperative matrix dispatch.
///
/// When N >> K (e.g. gate_up: N=9728, K=2560, ratio=3.8), each 16×16 tile iterates
/// only K/16 times — too few iterations to amortize tile setup and shared memory loads.
/// v5 vectorized is faster in this regime.
///
/// When N ≈ K (hidden: ratio=1.0) or K >> N (down: ratio=0.26), coopmat wins 1.3-3×
/// thanks to better data reuse across the longer K-loop.
///
/// Threshold of 2.0 validated on AMD 780M RDNA3 (bench-vulkan-q4, M=16/64/128):
///   hidden  (2560²,     ratio=1.0) → coopmat 2-3× faster
///   down    (2560×9728, ratio=0.26) → coopmat 1.3-2× faster
///   gate_up (9728×2560, ratio=3.8) → v5 wins
const COOPMAT_NK_RATIO_MAX: u32 = 2;

/// Q4 per-group matrix-matrix multiply.
///
/// C[m,n] = sum_k( A[m*K+k] * dequant(W_packed[n,k/2], scales[n,k/32]) )
///
/// Selects the best shader variant at runtime:
/// - M >= 16 + N < 2*K + coopmat available: WMMA 16×16×16 tiled dispatch
/// - Otherwise: v5 vectorized or v0 baseline with (N, M, 1) dispatch
///
/// Push constants: M (u32) + N (u32) + K (u32) = 12 bytes.
pub fn q4_matmul(ctx: &VulkanContext, cb: vk::CommandBuffer, a: &VulkanBuffer, w_packed: &VulkanBuffer, scales: &VulkanBuffer, c: &VulkanBuffer, m: u32, n: u32, k: u32) {
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&m.to_le_bytes());
    push[4..8].copy_from_slice(&n.to_le_bytes());
    push[8..12].copy_from_slice(&k.to_le_bytes());

    let bufs = [(a.buffer, a.size), (w_packed.buffer, w_packed.size), (scales.buffer, scales.size), (c.buffer, c.size)];

    // Use cooperative matrix for prefill (M >= 16) on favorable dimensions (N < 2*K).
    // Skip coopmat when N >> K (gate_up pattern): too few K-loop iterations per tile.
    if m >= COOPMAT_M_THRESHOLD && n < k * COOPMAT_NK_RATIO_MAX {
        // Prefer INT8 WMMA on NVIDIA (2× throughput vs fp16)
        if let Some(ref i8_p) = ctx.pipelines.q4_matmul_coopmat_i8 {
            let groups_x = (n + 15) / 16;
            let groups_y = (m + 15) / 16;
            ctx.cmd_dispatch(cb, i8_p.pipeline, i8_p.layout,
                &bufs, &push, groups_x, groups_y, 1);
            return;
        }
        // Fallback: fp16 coopmat
        if let Some(ref coopmat_p) = ctx.pipelines.q4_matmul_coopmat {
            let groups_x = (n + 15) / 16;
            let groups_y = (m + 15) / 16;
            ctx.cmd_dispatch(cb, coopmat_p.pipeline, coopmat_p.layout,
                &bufs, &push, groups_x, groups_y, 1);
            return;
        }
    }

    // Fallback: v5 vectorized or v0 baseline
    let p = if ctx.pipelines.use_q4_v5 {
        &ctx.pipelines.q4_matmul_v5
    } else {
        &ctx.pipelines.q4_matmul
    };
    ctx.cmd_dispatch(cb, p.pipeline, p.layout, &bufs, &push, n, m, 1);
}
