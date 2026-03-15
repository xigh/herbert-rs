//! Matrix-vector multiply kernel dispatch wrappers (decode path, M=1).

use ash::vk;
use crate::context::VulkanContext;
use crate::memory::VulkanBuffer;

pub fn int8_matvec(ctx: &VulkanContext, cb: vk::CommandBuffer, x: &VulkanBuffer, w_packed: &VulkanBuffer, scales: &VulkanBuffer, y: &VulkanBuffer, n: u32, k: u32) {
    let p = &ctx.pipelines.int8_matvec;
    let push = k.to_le_bytes();
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(x.buffer, x.size), (w_packed.buffer, w_packed.size), (scales.buffer, scales.size), (y.buffer, y.size)],
        &push, n, 1, 1);
}

pub fn bf16_matvec(ctx: &VulkanContext, cb: vk::CommandBuffer, x: &VulkanBuffer, w_packed: &VulkanBuffer, y: &VulkanBuffer, n: u32, k: u32) {
    let p = &ctx.pipelines.bf16_matvec;
    let push = k.to_le_bytes();
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(x.buffer, x.size), (w_packed.buffer, w_packed.size), (y.buffer, y.size)],
        &push, n, 1, 1);
}

pub fn f32_matvec(ctx: &VulkanContext, cb: vk::CommandBuffer, x: &VulkanBuffer, w: &VulkanBuffer, y: &VulkanBuffer, n: u32, k: u32) {
    let p = &ctx.pipelines.f32_matvec;
    let push = k.to_le_bytes();
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(x.buffer, x.size), (w.buffer, w.size), (y.buffer, y.size)],
        &push, n, 1, 1);
}

/// Q4 per-group matrix-vector multiply.
///
/// y[n] = sum_k( x[k] * dequant(w_packed[n,k/2], scales[n,k/32]) )
///
/// Dispatch: (N, 1, 1) workgroups of (SUBGROUP_SIZE, 1, 1) threads.
/// Push constants: N (u32) + K (u32) = 8 bytes.
pub fn q4_matvec(ctx: &VulkanContext, cb: vk::CommandBuffer, x: &VulkanBuffer, w_packed: &VulkanBuffer, scales: &VulkanBuffer, y: &VulkanBuffer, n: u32, k: u32) {
    let p = if ctx.pipelines.use_q4_v5 {
        &ctx.pipelines.q4_matvec_v5
    } else {
        &ctx.pipelines.q4_matvec
    };
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&n.to_le_bytes());
    push[4..8].copy_from_slice(&k.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(x.buffer, x.size), (w_packed.buffer, w_packed.size), (scales.buffer, scales.size), (y.buffer, y.size)],
        &push, n, 1, 1);
}
