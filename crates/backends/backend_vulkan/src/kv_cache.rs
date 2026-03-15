//! KV cache stored in GPU device memory.
//!
//! Pre-allocates per-layer key/value caches and decode working buffers.
//! Qwen3-specific: no conv buffers.

use ash::vk;
use herbert_core::config::Config;
use herbert_core::error::Result;

use crate::context::VulkanContext;
use crate::memory::VulkanBuffer;

/// KV cache and decode working buffers on GPU.
///
/// K and V caches are stored as `f32[max_tokens * kv_dim]` per layer.
/// Decode working buffers are pre-allocated to avoid per-token allocation.
pub struct VulkanKvCache {
    // Per-layer KV caches
    pub keys: Vec<VulkanBuffer>,
    pub values: Vec<VulkanBuffer>,
    pub seq_len: usize,
    pub max_tokens: usize,
    pub kv_dim: usize,

    // Per-layer decode working buffers
    pub decode_q: Vec<VulkanBuffer>,
    pub decode_k: Vec<VulkanBuffer>,
    pub decode_v: Vec<VulkanBuffer>,
    pub decode_attn_out: Vec<VulkanBuffer>,
    pub decode_norm1: Vec<VulkanBuffer>,
    pub decode_norm2: Vec<VulkanBuffer>,
    pub decode_scores: Vec<VulkanBuffer>,
    pub decode_mlp_gate: Vec<VulkanBuffer>,
    pub decode_mlp_up: Vec<VulkanBuffer>,
    pub decode_mlp_out: Vec<VulkanBuffer>,

    // Shared working buffers (not per-layer)
    pub decode_embed: VulkanBuffer,
    pub decode_final_norm: VulkanBuffer,
    pub decode_logits: VulkanBuffer,
    pub argmax_result: VulkanBuffer,
    pub token_buf: VulkanBuffer,

    // MoE decode working buffers (allocated only for MoE models)
    pub moe_router_logits: Option<VulkanBuffer>,  // host-visible [num_experts] f32
    pub moe_gate_buf: Option<VulkanBuffer>,        // [moe_intermediate_size]
    pub moe_up_buf: Option<VulkanBuffer>,          // [moe_intermediate_size]
    pub moe_expert_out: Option<VulkanBuffer>,      // [hidden_size]
    pub moe_output: Option<VulkanBuffer>,          // [hidden_size] accumulator
}

// SAFETY: VulkanKvCache holds Vulkan buffer handles that are not accessed
// concurrently. The backend ensures single-threaded GPU access via sequential
// command buffer submission. Required for KvHandle (Box<dyn Any + Send + Sync>).
unsafe impl Send for VulkanKvCache {}
unsafe impl Sync for VulkanKvCache {}

impl VulkanKvCache {
    /// Allocate KV cache and all decode working buffers.
    pub fn new(ctx: &VulkanContext, config: &Config, max_tokens: usize) -> Result<Self> {
        let num_layers = config.num_layers;
        let hidden_size = config.hidden_size;
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let num_heads = config.num_attention_heads;

        let kv_buf_bytes = (max_tokens * kv_dim * 4) as u64;
        let scores_bytes = (num_heads * max_tokens * 4) as u64;
        let usage = vk::BufferUsageFlags::empty();

        let moe_inter = config.moe_intermediate_size.unwrap_or(0);

        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);
        let mut decode_q = Vec::with_capacity(num_layers);
        let mut decode_k = Vec::with_capacity(num_layers);
        let mut decode_v = Vec::with_capacity(num_layers);
        let mut decode_attn_out = Vec::with_capacity(num_layers);
        let mut decode_norm1 = Vec::with_capacity(num_layers);
        let mut decode_norm2 = Vec::with_capacity(num_layers);
        let mut decode_scores = Vec::with_capacity(num_layers);
        let mut decode_mlp_gate = Vec::with_capacity(num_layers);
        let mut decode_mlp_up = Vec::with_capacity(num_layers);
        let mut decode_mlp_out = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            keys.push(VulkanBuffer::device_local_zeroed(ctx, kv_buf_bytes, usage)?);
            values.push(VulkanBuffer::device_local_zeroed(ctx, kv_buf_bytes, usage)?);
            decode_q.push(VulkanBuffer::device_local_zeroed(ctx, (q_dim * 4) as u64, usage)?);
            decode_k.push(VulkanBuffer::device_local_zeroed(ctx, (kv_dim * 4) as u64, usage)?);
            decode_v.push(VulkanBuffer::device_local_zeroed(ctx, (kv_dim * 4) as u64, usage)?);
            decode_attn_out.push(VulkanBuffer::device_local_zeroed(ctx, (q_dim * 4) as u64, usage)?);
            decode_norm1.push(VulkanBuffer::device_local_zeroed(ctx, (hidden_size * 4) as u64, usage)?);
            decode_norm2.push(VulkanBuffer::device_local_zeroed(ctx, (hidden_size * 4) as u64, usage)?);
            decode_scores.push(VulkanBuffer::device_local_zeroed(ctx, scores_bytes, usage)?);

            let mlp_inter = if config.is_moe_layer(i) { moe_inter } else { intermediate_size };
            decode_mlp_gate.push(VulkanBuffer::device_local_zeroed(ctx, (mlp_inter * 4) as u64, usage)?);
            decode_mlp_up.push(VulkanBuffer::device_local_zeroed(ctx, (mlp_inter * 4) as u64, usage)?);
            decode_mlp_out.push(VulkanBuffer::device_local_zeroed(ctx, (hidden_size * 4) as u64, usage)?);
        }

        let decode_embed = VulkanBuffer::device_local_zeroed(ctx, (hidden_size * 4) as u64, usage)?;
        let decode_final_norm = VulkanBuffer::device_local_zeroed(ctx, (hidden_size * 4) as u64, usage)?;
        let decode_logits = VulkanBuffer::device_local_zeroed(ctx, (vocab_size * 4) as u64, usage)?;
        let argmax_result = VulkanBuffer::host_visible(
            ctx, 8,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        let token_buf = VulkanBuffer::host_visible(
            ctx, 4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        )?;

        let (moe_router_logits, moe_gate_buf, moe_up_buf, moe_expert_out, moe_output) =
            if config.is_moe() {
                let ne = config.num_experts.unwrap();
                let router_buf = VulkanBuffer::host_visible(
                    ctx, (ne * 4) as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                )?;
                let gate_buf = VulkanBuffer::device_local_zeroed(ctx, (moe_inter * 4) as u64, usage)?;
                let up_buf = VulkanBuffer::device_local_zeroed(ctx, (moe_inter * 4) as u64, usage)?;
                let expert_out = VulkanBuffer::device_local_zeroed(ctx, (hidden_size * 4) as u64, usage)?;
                let output = VulkanBuffer::device_local_zeroed(ctx, (hidden_size * 4) as u64, usage)?;
                (Some(router_buf), Some(gate_buf), Some(up_buf), Some(expert_out), Some(output))
            } else {
                (None, None, None, None, None)
            };

        Ok(Self {
            keys,
            values,
            seq_len: 0,
            max_tokens,
            kv_dim,
            decode_q,
            decode_k,
            decode_v,
            decode_attn_out,
            decode_norm1,
            decode_norm2,
            decode_scores,
            decode_mlp_gate,
            decode_mlp_up,
            decode_mlp_out,
            decode_embed,
            decode_final_norm,
            decode_logits,
            argmax_result,
            token_buf,
            moe_router_logits,
            moe_gate_buf,
            moe_up_buf,
            moe_expert_out,
            moe_output,
        })
    }
}
