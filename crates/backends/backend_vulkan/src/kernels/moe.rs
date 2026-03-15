//! MoE (Mixture of Experts) kernel dispatch wrappers.

use ash::vk;
use crate::context::VulkanContext;
use crate::memory::VulkanBuffer;
use super::div_ceil;

pub fn gather(ctx: &VulkanContext, cb: vk::CommandBuffer, output: &VulkanBuffer, input: &VulkanBuffer, indices: &VulkanBuffer, dim: u32, count: u32) {
    let p = &ctx.pipelines.moe_gather;
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&dim.to_le_bytes());
    push[4..8].copy_from_slice(&count.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(output.buffer, output.size), (input.buffer, input.size), (indices.buffer, indices.size)],
        &push, div_ceil(count * dim, 256), 1, 1);
}

pub fn scatter_weighted_add(ctx: &VulkanContext, cb: vk::CommandBuffer, output: &VulkanBuffer, input: &VulkanBuffer, indices: &VulkanBuffer, weights: &VulkanBuffer, dim: u32, count: u32) {
    let p = &ctx.pipelines.moe_scatter_add;
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&dim.to_le_bytes());
    push[4..8].copy_from_slice(&count.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(output.buffer, output.size), (input.buffer, input.size), (indices.buffer, indices.size), (weights.buffer, weights.size)],
        &push, div_ceil(count * dim, 256), 1, 1);
}
