//! RMS normalization kernel dispatch wrappers.

use ash::vk;
use crate::context::VulkanContext;
use crate::memory::VulkanBuffer;

/// RMS normalization (single vector).
pub fn rms_norm(
    ctx: &VulkanContext,
    cb: vk::CommandBuffer,
    input: &VulkanBuffer,
    weight: &VulkanBuffer,
    output: &VulkanBuffer,
    dim: u32,
    eps: f32,
) {
    let p = &ctx.pipelines.rms_norm;
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&dim.to_le_bytes());
    push[4..8].copy_from_slice(&eps.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(input.buffer, input.size), (weight.buffer, weight.size), (output.buffer, output.size)],
        &push, 1, 1, 1);
}

/// Batched RMS normalization (for prefill).
pub fn rms_norm_batch(
    ctx: &VulkanContext,
    cb: vk::CommandBuffer,
    input: &VulkanBuffer,
    weight: &VulkanBuffer,
    output: &VulkanBuffer,
    dim: u32,
    eps: f32,
    batch_size: u32,
) {
    let p = &ctx.pipelines.rms_norm_batch;
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&dim.to_le_bytes());
    push[4..8].copy_from_slice(&eps.to_le_bytes());
    push[8..12].copy_from_slice(&batch_size.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(input.buffer, input.size), (weight.buffer, weight.size), (output.buffer, output.size)],
        &push, batch_size, 1, 1);
}

/// Per-head RMS normalization (for QK norms, in-place).
pub fn head_rms_norm(
    ctx: &VulkanContext,
    cb: vk::CommandBuffer,
    data: &VulkanBuffer,
    weight: &VulkanBuffer,
    num_heads: u32,
    head_dim: u32,
    eps: f32,
) {
    let p = &ctx.pipelines.head_rms_norm;
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    push[4..8].copy_from_slice(&head_dim.to_le_bytes());
    push[8..12].copy_from_slice(&eps.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(data.buffer, data.size), (weight.buffer, weight.size)],
        &push, num_heads, 1, 1);
}

/// Batched per-head RMS normalization for prefill.
pub fn head_rms_norm_batch(
    ctx: &VulkanContext,
    cb: vk::CommandBuffer,
    data: &VulkanBuffer,
    weight: &VulkanBuffer,
    num_heads: u32,
    head_dim: u32,
    seq_len: u32,
    eps: f32,
) {
    let total_heads = seq_len * num_heads;
    let p = &ctx.pipelines.head_rms_norm;
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&total_heads.to_le_bytes());
    push[4..8].copy_from_slice(&head_dim.to_le_bytes());
    push[8..12].copy_from_slice(&eps.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(data.buffer, data.size), (weight.buffer, weight.size)],
        &push, total_heads, 1, 1);
}
