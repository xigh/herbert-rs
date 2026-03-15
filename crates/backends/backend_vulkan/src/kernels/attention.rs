//! Attention and KV cache kernel dispatch wrappers.

use ash::vk;
use crate::context::VulkanContext;
use crate::memory::VulkanBuffer;
use super::div_ceil;

pub fn attention_decode(ctx: &VulkanContext, cb: vk::CommandBuffer, q: &VulkanBuffer, k_cache: &VulkanBuffer, v_cache: &VulkanBuffer, output: &VulkanBuffer, scores: &VulkanBuffer, num_heads: u32, num_kv_heads: u32, head_dim: u32, kv_dim: u32, cached_len: u32, scale: f32) {
    let p = &ctx.pipelines.attention_decode;
    let mut push = [0u8; 24];
    push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    push[4..8].copy_from_slice(&num_kv_heads.to_le_bytes());
    push[8..12].copy_from_slice(&head_dim.to_le_bytes());
    push[12..16].copy_from_slice(&kv_dim.to_le_bytes());
    push[16..20].copy_from_slice(&cached_len.to_le_bytes());
    push[20..24].copy_from_slice(&scale.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(q.buffer, q.size), (k_cache.buffer, k_cache.size), (v_cache.buffer, v_cache.size), (output.buffer, output.size), (scores.buffer, scores.size)],
        &push, num_heads, 1, 1);
}

pub fn attention_prefill(ctx: &VulkanContext, cb: vk::CommandBuffer, q: &VulkanBuffer, k_cache: &VulkanBuffer, v_cache: &VulkanBuffer, output: &VulkanBuffer, scores: &VulkanBuffer, seq_len: u32, num_heads: u32, num_kv_heads: u32, head_dim: u32, kv_dim: u32, q_dim: u32, cached_len: u32, start_pos: u32, scale: f32) {
    let p = &ctx.pipelines.attention_prefill;
    let mut push = [0u8; 36];
    push[0..4].copy_from_slice(&seq_len.to_le_bytes());
    push[4..8].copy_from_slice(&num_heads.to_le_bytes());
    push[8..12].copy_from_slice(&num_kv_heads.to_le_bytes());
    push[12..16].copy_from_slice(&head_dim.to_le_bytes());
    push[16..20].copy_from_slice(&kv_dim.to_le_bytes());
    push[20..24].copy_from_slice(&q_dim.to_le_bytes());
    push[24..28].copy_from_slice(&cached_len.to_le_bytes());
    push[28..32].copy_from_slice(&start_pos.to_le_bytes());
    push[32..36].copy_from_slice(&scale.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(q.buffer, q.size), (k_cache.buffer, k_cache.size), (v_cache.buffer, v_cache.size), (output.buffer, output.size), (scores.buffer, scores.size)],
        &push, num_heads, seq_len, 1);
}

pub fn kv_cache_append(ctx: &VulkanContext, cb: vk::CommandBuffer, cache: &VulkanBuffer, new_kv: &VulkanBuffer, kv_dim: u32, seq_len: u32) {
    let p = &ctx.pipelines.kv_cache_append;
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&kv_dim.to_le_bytes());
    push[4..8].copy_from_slice(&seq_len.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(cache.buffer, cache.size), (new_kv.buffer, new_kv.size)],
        &push, div_ceil(kv_dim, 256), 1, 1);
}

pub fn kv_cache_append_batch(ctx: &VulkanContext, cb: vk::CommandBuffer, cache: &VulkanBuffer, new_kv: &VulkanBuffer, kv_dim: u32, start: u32, count: u32) {
    let p = &ctx.pipelines.kv_cache_append_batch;
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&kv_dim.to_le_bytes());
    push[4..8].copy_from_slice(&start.to_le_bytes());
    push[8..12].copy_from_slice(&count.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(cache.buffer, cache.size), (new_kv.buffer, new_kv.size)],
        &push, div_ceil(count * kv_dim, 256), 1, 1);
}
