//! RoPE (Rotary Position Embedding) kernel dispatch wrappers.

use ash::vk;
use crate::context::VulkanContext;
use crate::memory::VulkanBuffer;
use super::div_ceil;

pub fn rope_single(ctx: &VulkanContext, cb: vk::CommandBuffer, qk: &VulkanBuffer, cos_vals: &VulkanBuffer, sin_vals: &VulkanBuffer, num_heads: u32, half_dim: u32) {
    let p = &ctx.pipelines.rope_single;
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    push[4..8].copy_from_slice(&half_dim.to_le_bytes());
    let total = num_heads * half_dim;
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(qk.buffer, qk.size), (cos_vals.buffer, cos_vals.size), (sin_vals.buffer, sin_vals.size)],
        &push, div_ceil(total, 256), 1, 1);
}

pub fn rope_batch(ctx: &VulkanContext, cb: vk::CommandBuffer, qk: &VulkanBuffer, cos_cache: &VulkanBuffer, sin_cache: &VulkanBuffer, seq_len: u32, num_heads: u32, half_dim: u32, start_pos: u32) {
    let p = &ctx.pipelines.rope_batch;
    let mut push = [0u8; 16];
    push[0..4].copy_from_slice(&seq_len.to_le_bytes());
    push[4..8].copy_from_slice(&num_heads.to_le_bytes());
    push[8..12].copy_from_slice(&half_dim.to_le_bytes());
    push[12..16].copy_from_slice(&start_pos.to_le_bytes());
    let total = seq_len * num_heads * half_dim;
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(qk.buffer, qk.size), (cos_cache.buffer, cos_cache.size), (sin_cache.buffer, sin_cache.size)],
        &push, div_ceil(total, 256), 1, 1);
}
