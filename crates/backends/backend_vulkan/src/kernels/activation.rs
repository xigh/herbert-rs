//! Activation, embedding, and utility kernel dispatch wrappers.

use ash::vk;
use crate::context::VulkanContext;
use crate::memory::VulkanBuffer;
use super::div_ceil;

pub fn swiglu(ctx: &VulkanContext, cb: vk::CommandBuffer, gate: &VulkanBuffer, up: &VulkanBuffer, n: u32) {
    let p = &ctx.pipelines.swiglu;
    let push = n.to_le_bytes();
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(gate.buffer, gate.size), (up.buffer, up.size)],
        &push, div_ceil(n, 256), 1, 1);
}

pub fn softmax(ctx: &VulkanContext, cb: vk::CommandBuffer, data: &VulkanBuffer, n: u32) {
    let p = &ctx.pipelines.softmax;
    let push = n.to_le_bytes();
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(data.buffer, data.size)],
        &push, 1, 1, 1);
}

pub fn argmax(ctx: &VulkanContext, cb: vk::CommandBuffer, data: &VulkanBuffer, result: &VulkanBuffer, n: u32) {
    let p = &ctx.pipelines.argmax;
    let push = n.to_le_bytes();
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(data.buffer, data.size), (result.buffer, result.size)],
        &push, 1, 1, 1);
}

pub fn embedding(ctx: &VulkanContext, cb: vk::CommandBuffer, embed_table: &VulkanBuffer, tokens: &VulkanBuffer, output: &VulkanBuffer, seq_len: u32, hidden_size: u32) {
    let p = &ctx.pipelines.embedding;
    let total = seq_len * hidden_size;
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&hidden_size.to_le_bytes());
    push[4..8].copy_from_slice(&total.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(embed_table.buffer, embed_table.size), (tokens.buffer, tokens.size), (output.buffer, output.size)],
        &push, div_ceil(total, 256), 1, 1);
}

pub fn residual_add(ctx: &VulkanContext, cb: vk::CommandBuffer, a: &VulkanBuffer, b: &VulkanBuffer, len: u32) {
    let p = &ctx.pipelines.residual_add;
    let push = len.to_le_bytes();
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(a.buffer, a.size), (b.buffer, b.size)],
        &push, div_ceil(len, 256), 1, 1);
}

pub fn bias_add_batch(ctx: &VulkanContext, cb: vk::CommandBuffer, data: &VulkanBuffer, bias: &VulkanBuffer, dim: u32, batch_size: u32) {
    let p = &ctx.pipelines.bias_add_batch;
    let total = dim * batch_size;
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&dim.to_le_bytes());
    push[4..8].copy_from_slice(&total.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(data.buffer, data.size), (bias.buffer, bias.size)],
        &push, div_ceil(total, 256), 1, 1);
}

pub fn scaled_add(ctx: &VulkanContext, cb: vk::CommandBuffer, a: &VulkanBuffer, b: &VulkanBuffer, len: u32, scale: f32) {
    let p = &ctx.pipelines.scaled_add;
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&len.to_le_bytes());
    push[4..8].copy_from_slice(&scale.to_le_bytes());
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(a.buffer, a.size), (b.buffer, b.size)],
        &push, div_ceil(len, 256), 1, 1);
}

pub fn copy_buffer(ctx: &VulkanContext, cb: vk::CommandBuffer, src: &VulkanBuffer, dst: &VulkanBuffer, n: u32) {
    let p = &ctx.pipelines.copy_buffer;
    let push = n.to_le_bytes();
    ctx.cmd_dispatch(cb, p.pipeline, p.layout,
        &[(src.buffer, src.size), (dst.buffer, dst.size)],
        &push, div_ceil(n, 256), 1, 1);
}
