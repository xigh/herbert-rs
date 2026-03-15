//! Vulkan inference backend -- full decode and prefill pipeline for Qwen3.
//!
//! Simplified vs. LFM2 backend:
//! - No conv layers: Qwen3 is pure attention+MLP.
//! - QK norms always present (non-optional).
//! - MoE routing: softmax only.
//! - Three weight formats: BF16, INT8, Q4.

use ash::vk;

use herbert_core::backend::{DecodeOutput, PrefillOutput, RunOpts};
use herbert_core::config::Config;
use herbert_core::error::{HerbertError, Result};

use crate::context::VulkanContext;
use crate::kernels;
use crate::kv_cache::VulkanKvCache;
use crate::memory::VulkanBuffer;
use crate::model::{VulkanLayerMLP, VulkanModel, VulkanWeight};

/// Inner state of the Vulkan backend (populated after load).
///
/// Field order matters: `ctx` MUST be last because its Drop destroys the Vulkan device,
/// and all VulkanBuffers in `model` must be destroyed first.
pub struct VulkanBackendInner {
    pub model: VulkanModel,
    pub config: Config,
    pub max_tokens: usize,
    pub ctx: VulkanContext, // MUST be last (drop order)
}

impl VulkanBackendInner {
    // ========================================================================
    // Helper: matrix-vector multiply (dispatch BF16, Int8, or Q4)
    // ========================================================================

    fn matvec_weight(
        ctx: &VulkanContext,
        cb: vk::CommandBuffer,
        input: &VulkanBuffer,
        weight: &VulkanWeight,
        output: &VulkanBuffer,
    ) {
        match weight {
            VulkanWeight::BF16(w) => {
                kernels::matvec::bf16_matvec(
                    ctx, cb, input, &w.packed, output,
                    w.n as u32, w.k as u32,
                );
            }
            VulkanWeight::Int8(w) => {
                kernels::matvec::int8_matvec(
                    ctx, cb, input, &w.packed, &w.scales, output,
                    w.n as u32, w.k as u32,
                );
            }
            VulkanWeight::Q4(w) => {
                kernels::matvec::q4_matvec(
                    ctx, cb, input, &w.packed, &w.scales, output,
                    w.n as u32, w.k as u32,
                );
            }
        }
    }

    // ========================================================================
    // Helper: matrix-matrix multiply (dispatch BF16, Int8, or Q4)
    // ========================================================================

    fn matmul_weight(
        ctx: &VulkanContext,
        cb: vk::CommandBuffer,
        input: &VulkanBuffer,
        weight: &VulkanWeight,
        output: &VulkanBuffer,
        m: u32,
    ) {
        match weight {
            VulkanWeight::BF16(w) => {
                kernels::matmul::bf16_matmul(
                    ctx, cb, input, &w.packed, output,
                    m, w.n as u32, w.k as u32,
                );
            }
            VulkanWeight::Int8(w) => {
                kernels::matmul::int8_matmul(
                    ctx, cb, input, &w.packed, &w.scales, output,
                    m, w.n as u32, w.k as u32,
                );
            }
            VulkanWeight::Q4(w) => {
                kernels::matmul::q4_matmul(
                    ctx, cb, input, &w.packed, &w.scales, output,
                    m, w.n as u32, w.k as u32,
                );
            }
        }
    }

    // ========================================================================
    // CPU-side MoE routing helpers
    // ========================================================================

    fn cpu_softmax(logits: &mut [f32]) {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in logits.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for v in logits.iter_mut() {
                *v *= inv;
            }
        }
    }

    fn cpu_top_k(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    fn cpu_route_token(logits: &mut [f32], k: usize, norm_topk: bool) -> Vec<(usize, f32)> {
        Self::cpu_softmax(logits);
        let mut selected = Self::cpu_top_k(logits, k);
        if norm_topk {
            let sum: f32 = selected.iter().map(|(_, w)| *w).sum();
            if sum > 0.0 {
                for (_, w) in &mut selected {
                    *w /= sum;
                }
            }
        }
        selected
    }

    // ========================================================================
    // Decode step: process one token, return predicted next token ID
    // ========================================================================

    fn decode_step(&self, kv: &mut VulkanKvCache, token: u32) -> Result<u32> {
        let ctx = &self.ctx;
        let model = &self.model;
        let config = &self.config;

        let hidden_size = config.hidden_size as u32;
        let num_heads = config.num_attention_heads as u32;
        let num_kv_heads = config.num_key_value_heads as u32;
        let head_dim = config.head_dim as u32;
        let half_dim = (config.rotary_ndims / 2) as u32;
        let kv_dim = config.kv_dim() as u32;
        let vocab_size = config.vocab_size as u32;
        let eps = config.rms_norm_eps;
        let cache_pos = kv.seq_len as u32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // 1. Write token to host-visible buffer
        {
            let ptr = unsafe {
                ctx.device.map_memory(kv.token_buf.memory, 0, 4, vk::MemoryMapFlags::empty())
            }.map_err(|e| HerbertError::Backend(format!("map token_buf: {:?}", e)))?;
            unsafe {
                *(ptr as *mut u32) = token;
                ctx.device.unmap_memory(kv.token_buf.memory);
            }
        }

        // 2. Begin command buffer
        let mut cb = ctx.cmd_begin()?;

        // 3. Embedding lookup
        kernels::activation::embedding(
            ctx, cb, &model.embed_tokens, &kv.token_buf,
            &kv.decode_embed, 1, hidden_size,
        );
        ctx.cmd_pipeline_barrier(cb);

        // 4. Transformer layers
        for l in 0..config.num_layers {
            let layer = &model.layers[l];

            // (a) RMS norm 1
            kernels::norm::rms_norm(
                ctx, cb, &kv.decode_embed, &layer.input_layernorm,
                &kv.decode_norm1[l], hidden_size, eps,
            );
            ctx.cmd_pipeline_barrier(cb);

            // (b) Q/K/V matvec projections
            Self::matvec_weight(ctx, cb, &kv.decode_norm1[l], &layer.q_proj, &kv.decode_q[l]);
            Self::matvec_weight(ctx, cb, &kv.decode_norm1[l], &layer.k_proj, &kv.decode_k[l]);
            Self::matvec_weight(ctx, cb, &kv.decode_norm1[l], &layer.v_proj, &kv.decode_v[l]);
            ctx.cmd_pipeline_barrier(cb);

            // (c) QK head RMS norm (always present for Qwen3)
            kernels::norm::head_rms_norm(
                ctx, cb, &kv.decode_q[l], &layer.q_norm,
                num_heads, head_dim, eps,
            );
            kernels::norm::head_rms_norm(
                ctx, cb, &kv.decode_k[l], &layer.k_norm,
                num_kv_heads, head_dim, eps,
            );
            ctx.cmd_pipeline_barrier(cb);

            // (d) RoPE on Q and K (use rope_batch with seq_len=1 for position offset)
            kernels::rope::rope_batch(
                ctx, cb, &kv.decode_q[l],
                &model.cos_cache, &model.sin_cache,
                1, num_heads, half_dim, cache_pos,
            );
            kernels::rope::rope_batch(
                ctx, cb, &kv.decode_k[l],
                &model.cos_cache, &model.sin_cache,
                1, num_kv_heads, half_dim, cache_pos,
            );
            ctx.cmd_pipeline_barrier(cb);

            // (e) KV cache append
            kernels::attention::kv_cache_append(
                ctx, cb, &kv.keys[l], &kv.decode_k[l], kv_dim, cache_pos,
            );
            kernels::attention::kv_cache_append(
                ctx, cb, &kv.values[l], &kv.decode_v[l], kv_dim, cache_pos,
            );
            ctx.cmd_pipeline_barrier(cb);

            // (f) Attention decode
            let cached_len = cache_pos + 1;
            kernels::attention::attention_decode(
                ctx, cb,
                &kv.decode_q[l], &kv.keys[l], &kv.values[l],
                &kv.decode_attn_out[l], &kv.decode_scores[l],
                num_heads, num_kv_heads, head_dim, kv_dim,
                cached_len, scale,
            );
            ctx.cmd_pipeline_barrier(cb);

            // (g) O projection
            Self::matvec_weight(
                ctx, cb, &kv.decode_attn_out[l],
                &layer.o_proj, &kv.decode_mlp_out[l],
            );
            ctx.cmd_pipeline_barrier(cb);

            // (h) Residual add
            kernels::activation::residual_add(
                ctx, cb, &kv.decode_embed, &kv.decode_mlp_out[l], hidden_size,
            );
            ctx.cmd_pipeline_barrier(cb);

            // (i) RMS norm 2
            kernels::norm::rms_norm(
                ctx, cb, &kv.decode_embed, &layer.post_attention_layernorm,
                &kv.decode_norm2[l], hidden_size, eps,
            );
            ctx.cmd_pipeline_barrier(cb);

            // (j) MLP
            match &layer.mlp {
                VulkanLayerMLP::Dense { gate_proj, up_proj, down_proj } => {
                    Self::matvec_weight(ctx, cb, &kv.decode_norm2[l], gate_proj, &kv.decode_mlp_gate[l]);
                    Self::matvec_weight(ctx, cb, &kv.decode_norm2[l], up_proj, &kv.decode_mlp_up[l]);
                    ctx.cmd_pipeline_barrier(cb);

                    let inter = config.intermediate_size as u32;
                    kernels::activation::swiglu(ctx, cb, &kv.decode_mlp_gate[l], &kv.decode_mlp_up[l], inter);
                    ctx.cmd_pipeline_barrier(cb);

                    Self::matvec_weight(ctx, cb, &kv.decode_mlp_gate[l], down_proj, &kv.decode_mlp_out[l]);
                    ctx.cmd_pipeline_barrier(cb);

                    kernels::activation::residual_add(
                        ctx, cb, &kv.decode_embed, &kv.decode_mlp_out[l], hidden_size,
                    );
                    ctx.cmd_pipeline_barrier(cb);
                }

                VulkanLayerMLP::MoE {
                    router, experts, num_experts, num_experts_per_tok,
                    moe_intermediate_size, norm_topk_prob,
                } => {
                    let ne = *num_experts;
                    let top_k = *num_experts_per_tok;
                    let moe_inter = *moe_intermediate_size as u32;
                    let router_logits_buf = kv.moe_router_logits.as_ref().unwrap();
                    let moe_gate = kv.moe_gate_buf.as_ref().unwrap();
                    let moe_up = kv.moe_up_buf.as_ref().unwrap();
                    let moe_expert_out = kv.moe_expert_out.as_ref().unwrap();
                    let moe_output = kv.moe_output.as_ref().unwrap();

                    // Router: hidden -> [ne] logits
                    kernels::matvec::f32_matvec(
                        ctx, cb, &kv.decode_norm2[l], router,
                        router_logits_buf, ne as u32, hidden_size,
                    );
                    ctx.cmd_compute_to_transfer_barrier(cb);

                    // Submit to readback router logits on CPU
                    ctx.cmd_end_submit_wait(cb)?;

                    // CPU: read router logits
                    let mut logits = router_logits_buf.read_f32(ctx, ne)?;
                    let selected = Self::cpu_route_token(&mut logits, top_k, *norm_topk_prob);

                    // Zero the MoE output accumulator
                    cb = ctx.cmd_begin()?;
                    unsafe {
                        ctx.device.cmd_fill_buffer(cb, moe_output.buffer, 0, vk::WHOLE_SIZE, 0);
                    }
                    ctx.cmd_pipeline_barrier(cb);

                    // Process each selected expert
                    for (expert_id, expert_weight) in &selected {
                        let expert = &experts[*expert_id];

                        // gate_proj
                        Self::matvec_weight(ctx, cb, &kv.decode_norm2[l], &expert.gate_proj, moe_gate);
                        // up_proj
                        Self::matvec_weight(ctx, cb, &kv.decode_norm2[l], &expert.up_proj, moe_up);
                        ctx.cmd_pipeline_barrier(cb);

                        // SwiGLU
                        kernels::activation::swiglu(ctx, cb, moe_gate, moe_up, moe_inter);
                        ctx.cmd_pipeline_barrier(cb);

                        // down_proj
                        Self::matvec_weight(ctx, cb, moe_gate, &expert.down_proj, moe_expert_out);
                        ctx.cmd_pipeline_barrier(cb);

                        // Weighted accumulate: moe_output += weight * expert_out
                        kernels::activation::scaled_add(
                            ctx, cb, moe_output, moe_expert_out,
                            hidden_size, *expert_weight,
                        );
                        ctx.cmd_pipeline_barrier(cb);
                    }

                    // Residual add: embed += moe_output
                    kernels::activation::residual_add(
                        ctx, cb, &kv.decode_embed, moe_output, hidden_size,
                    );
                    ctx.cmd_pipeline_barrier(cb);
                }
            }
        }

        // 5. Final RMS norm
        kernels::norm::rms_norm(
            ctx, cb, &kv.decode_embed, &model.final_norm,
            &kv.decode_final_norm, hidden_size, eps,
        );
        ctx.cmd_pipeline_barrier(cb);

        // 6. LM head matvec
        Self::matvec_weight(
            ctx, cb, &kv.decode_final_norm, &model.lm_head,
            &kv.decode_logits,
        );
        ctx.cmd_pipeline_barrier(cb);

        // 7. Argmax
        kernels::activation::argmax(
            ctx, cb, &kv.decode_logits, &kv.argmax_result, vocab_size,
        );

        // 8. Submit and wait
        ctx.cmd_compute_to_transfer_barrier(cb);
        ctx.cmd_end_submit_wait(cb)?;

        // 9. Read result
        let result = kv.argmax_result.read_u32(ctx, 1)?;
        let token_id = result[0];

        // 10. Advance sequence position
        kv.seq_len += 1;

        Ok(token_id)
    }

    // ========================================================================
    // Prefill step: process all prompt tokens
    // ========================================================================

    fn prefill_step(&self, kv: &mut VulkanKvCache, tokens: &[u32]) -> Result<u32> {
        let ctx = &self.ctx;
        let model = &self.model;
        let config = &self.config;

        let seq_len = tokens.len() as u32;
        let hidden_size = config.hidden_size as u32;
        let num_heads = config.num_attention_heads as u32;
        let num_kv_heads = config.num_key_value_heads as u32;
        let head_dim = config.head_dim as u32;
        let half_dim = (config.rotary_ndims / 2) as u32;
        let q_dim = config.q_dim() as u32;
        let kv_dim = config.kv_dim() as u32;
        let vocab_size = config.vocab_size as u32;
        let eps = config.rms_norm_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let usage = vk::BufferUsageFlags::empty();

        // 1. Allocate temporary buffers
        let embed = VulkanBuffer::device_local(ctx, (seq_len * hidden_size * 4) as u64, usage)?;
        let norm = VulkanBuffer::device_local(ctx, (seq_len * hidden_size * 4) as u64, usage)?;
        let q_buf = VulkanBuffer::device_local(ctx, (seq_len * q_dim * 4) as u64, usage)?;
        let k_buf = VulkanBuffer::device_local(ctx, (seq_len * kv_dim * 4) as u64, usage)?;
        let v_buf = VulkanBuffer::device_local(ctx, (seq_len * kv_dim * 4) as u64, usage)?;
        let attn_out = VulkanBuffer::device_local(ctx, (seq_len * q_dim * 4) as u64, usage)?;
        let mlp_out = VulkanBuffer::device_local(ctx, (seq_len * hidden_size * 4) as u64, usage)?;

        let max_inter = if config.is_moe() {
            config.moe_intermediate_size.unwrap().max(config.intermediate_size)
        } else {
            config.intermediate_size
        };
        let mlp_gate = VulkanBuffer::device_local(ctx, (seq_len as usize * max_inter * 4) as u64, usage)?;
        let mlp_up = VulkanBuffer::device_local(ctx, (seq_len as usize * max_inter * 4) as u64, usage)?;
        let norm2 = VulkanBuffer::device_local(ctx, (seq_len * hidden_size * 4) as u64, usage)?;
        let scores = VulkanBuffer::device_local(ctx, (num_heads as u64 * seq_len as u64 * seq_len as u64 * 4) as u64, usage)?;

        // Last-token extraction buffer
        let last_hidden = VulkanBuffer::device_local(ctx, (hidden_size * 4) as u64, usage)?;
        let logits = VulkanBuffer::device_local(ctx, (vocab_size * 4) as u64, usage)?;
        let argmax_result = VulkanBuffer::host_visible(
            ctx, 8,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        // 2. Upload token IDs
        let token_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        let token_buf = VulkanBuffer::host_visible_with_data(
            ctx, &token_bytes, vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        // 3. Begin command buffer
        let mut cb = ctx.cmd_begin()?;

        // 4. Embedding
        kernels::activation::embedding(
            ctx, cb, &model.embed_tokens, &token_buf, &embed,
            seq_len, hidden_size,
        );
        ctx.cmd_pipeline_barrier(cb);

        // DEBUG: Check embedding output
        if std::env::var("VK_DEBUG").is_ok() {
            ctx.cmd_end_submit_wait(cb)?;
            let embed_vals = embed.read_f32(ctx, 8)?;
            eprintln!("[vk-debug] embed[0..8] = {:?}", embed_vals);
            cb = ctx.cmd_begin()?;
        }

        // 5. Transformer layers
        for l in 0..config.num_layers {
            let layer = &model.layers[l];

            // (a) RMS norm: embed -> norm
            kernels::norm::rms_norm_batch(
                ctx, cb, &embed, &layer.input_layernorm, &norm,
                hidden_size, eps, seq_len,
            );
            ctx.cmd_pipeline_barrier(cb);

            // DEBUG: after first layer's norm
            if l == 0 && std::env::var("VK_DEBUG").is_ok() {
                ctx.cmd_end_submit_wait(cb)?;
                let norm_vals = norm.read_f32(ctx, 8)?;
                eprintln!("[vk-debug] layer0 norm[0..8] = {:?}", norm_vals);
                cb = ctx.cmd_begin()?;
            }

            // (b) Q/K/V projections (matmul, M=seq_len)
            Self::matmul_weight(ctx, cb, &norm, &layer.q_proj, &q_buf, seq_len);
            Self::matmul_weight(ctx, cb, &norm, &layer.k_proj, &k_buf, seq_len);
            Self::matmul_weight(ctx, cb, &norm, &layer.v_proj, &v_buf, seq_len);
            ctx.cmd_pipeline_barrier(cb);

            // (c) QK head RMS norm (batched)
            kernels::norm::head_rms_norm_batch(
                ctx, cb, &q_buf, &layer.q_norm,
                num_heads, head_dim, seq_len, eps,
            );
            kernels::norm::head_rms_norm_batch(
                ctx, cb, &k_buf, &layer.k_norm,
                num_kv_heads, head_dim, seq_len, eps,
            );
            ctx.cmd_pipeline_barrier(cb);

            // (d) RoPE batch
            kernels::rope::rope_batch(
                ctx, cb, &q_buf, &model.cos_cache, &model.sin_cache,
                seq_len, num_heads, half_dim, 0,
            );
            kernels::rope::rope_batch(
                ctx, cb, &k_buf, &model.cos_cache, &model.sin_cache,
                seq_len, num_kv_heads, half_dim, 0,
            );
            ctx.cmd_pipeline_barrier(cb);

            // (e) KV cache append batch
            kernels::attention::kv_cache_append_batch(
                ctx, cb, &kv.keys[l], &k_buf, kv_dim, 0, seq_len,
            );
            kernels::attention::kv_cache_append_batch(
                ctx, cb, &kv.values[l], &v_buf, kv_dim, 0, seq_len,
            );
            ctx.cmd_pipeline_barrier(cb);

            // (f) Attention prefill (cached_len = seq_len after kv_cache_append_batch)
            kernels::attention::attention_prefill(
                ctx, cb,
                &q_buf, &kv.keys[l], &kv.values[l],
                &attn_out, &scores,
                seq_len, num_heads, num_kv_heads, head_dim,
                kv_dim, q_dim, seq_len, 0, scale,
            );
            ctx.cmd_pipeline_barrier(cb);

            // (g) O projection
            Self::matmul_weight(ctx, cb, &attn_out, &layer.o_proj, &mlp_out, seq_len);
            ctx.cmd_pipeline_barrier(cb);

            // (h) Residual add
            kernels::activation::residual_add(
                ctx, cb, &embed, &mlp_out, seq_len * hidden_size,
            );
            ctx.cmd_pipeline_barrier(cb);

            // (i) RMS norm 2
            kernels::norm::rms_norm_batch(
                ctx, cb, &embed, &layer.post_attention_layernorm, &norm2,
                hidden_size, eps, seq_len,
            );
            ctx.cmd_pipeline_barrier(cb);

            // (j) MLP
            match &layer.mlp {
                VulkanLayerMLP::Dense { gate_proj, up_proj, down_proj } => {
                    Self::matmul_weight(ctx, cb, &norm2, gate_proj, &mlp_gate, seq_len);
                    Self::matmul_weight(ctx, cb, &norm2, up_proj, &mlp_up, seq_len);
                    ctx.cmd_pipeline_barrier(cb);

                    let inter = config.intermediate_size as u32;
                    kernels::activation::swiglu(ctx, cb, &mlp_gate, &mlp_up, seq_len * inter);
                    ctx.cmd_pipeline_barrier(cb);

                    Self::matmul_weight(ctx, cb, &mlp_gate, down_proj, &mlp_out, seq_len);
                    ctx.cmd_pipeline_barrier(cb);

                    kernels::activation::residual_add(
                        ctx, cb, &embed, &mlp_out, seq_len * hidden_size,
                    );
                    ctx.cmd_pipeline_barrier(cb);
                }

                VulkanLayerMLP::MoE {
                    router, experts, num_experts, num_experts_per_tok,
                    moe_intermediate_size, norm_topk_prob,
                } => {
                    // MoE prefill: per-token CPU routing (submit CB per layer for readback)
                    let ne = *num_experts;
                    let top_k = *num_experts_per_tok;
                    let moe_inter = *moe_intermediate_size as u32;

                    // Router logits: [seq_len, ne]
                    let router_buf = VulkanBuffer::device_local(
                        ctx, (seq_len as usize * ne * 4) as u64, usage,
                    )?;
                    kernels::matmul::f32_matmul(
                        ctx, cb, &norm2, router, &router_buf,
                        seq_len, ne as u32, hidden_size,
                    );
                    ctx.cmd_compute_to_transfer_barrier(cb);
                    ctx.cmd_end_submit_wait(cb)?;

                    // CPU readback and routing
                    let all_logits = router_buf.read_f32(ctx, seq_len as usize * ne)?;

                    // Per-token expert routing + GPU dispatch
                    // Allocate MoE working buffers (reused across all tokens)
                    let moe_gate_buf = VulkanBuffer::device_local(ctx, (moe_inter * 4) as u64, usage)?;
                    let moe_up_buf = VulkanBuffer::device_local(ctx, (moe_inter * 4) as u64, usage)?;
                    let moe_expert_out = VulkanBuffer::device_local(ctx, (hidden_size * 4) as u64, usage)?;
                    let moe_token_out = VulkanBuffer::device_local(ctx, (hidden_size * 4) as u64, usage)?;
                    let token_norm2 = VulkanBuffer::device_local(ctx, (hidden_size * 4) as u64, usage)?;

                    cb = ctx.cmd_begin()?;

                    for t in 0..seq_len as usize {
                        let mut token_logits = all_logits[t * ne..(t + 1) * ne].to_vec();
                        let selected = Self::cpu_route_token(&mut token_logits, top_k, *norm_topk_prob);

                        let token_offset = (t as u64) * (hidden_size as u64) * 4;

                        // Copy token t's norm2 slice into reusable buffer
                        let region = vk::BufferCopy {
                            src_offset: token_offset,
                            dst_offset: 0,
                            size: (hidden_size * 4) as u64,
                        };
                        unsafe {
                            ctx.device.cmd_copy_buffer(cb, norm2.buffer, token_norm2.buffer, &[region]);
                        }
                        ctx.cmd_pipeline_barrier(cb);

                        // Zero moe_token_out
                        unsafe {
                            ctx.device.cmd_fill_buffer(cb, moe_token_out.buffer, 0, vk::WHOLE_SIZE, 0);
                        }
                        ctx.cmd_pipeline_barrier(cb);

                        for (expert_id, expert_weight) in &selected {
                            let expert = &experts[*expert_id];

                            Self::matvec_weight(ctx, cb, &token_norm2, &expert.gate_proj, &moe_gate_buf);
                            Self::matvec_weight(ctx, cb, &token_norm2, &expert.up_proj, &moe_up_buf);
                            ctx.cmd_pipeline_barrier(cb);

                            kernels::activation::swiglu(ctx, cb, &moe_gate_buf, &moe_up_buf, moe_inter);
                            ctx.cmd_pipeline_barrier(cb);

                            Self::matvec_weight(ctx, cb, &moe_gate_buf, &expert.down_proj, &moe_expert_out);
                            ctx.cmd_pipeline_barrier(cb);

                            kernels::activation::scaled_add(
                                ctx, cb, &moe_token_out, &moe_expert_out,
                                hidden_size, *expert_weight,
                            );
                            ctx.cmd_pipeline_barrier(cb);
                        }

                        // Copy moe_token_out into mlp_out[t]
                        let dst_region = vk::BufferCopy {
                            src_offset: 0,
                            dst_offset: token_offset,
                            size: (hidden_size * 4) as u64,
                        };
                        unsafe {
                            ctx.device.cmd_copy_buffer(cb, moe_token_out.buffer, mlp_out.buffer, &[dst_region]);
                        }
                        ctx.cmd_pipeline_barrier(cb);

                        // Submit per-token work to avoid huge command buffers
                        ctx.cmd_end_submit_wait(cb)?;
                        cb = ctx.cmd_begin()?;
                    }

                    // Residual add: embed += mlp_out (all tokens)
                    kernels::activation::residual_add(
                        ctx, cb, &embed, &mlp_out, seq_len * hidden_size,
                    );
                    ctx.cmd_pipeline_barrier(cb);
                }
            }
        }

        // 6. Final RMS norm (only last token for lm_head)
        // Extract last token's hidden state
        let last_offset = ((seq_len - 1) as u64) * (hidden_size as u64) * 4;
        let region = vk::BufferCopy {
            src_offset: last_offset,
            dst_offset: 0,
            size: (hidden_size * 4) as u64,
        };
        unsafe {
            ctx.device.cmd_copy_buffer(cb, embed.buffer, last_hidden.buffer, &[region]);
        }
        ctx.cmd_pipeline_barrier(cb);

        kernels::norm::rms_norm(
            ctx, cb, &last_hidden, &model.final_norm,
            &kv.decode_final_norm, hidden_size, eps,
        );
        ctx.cmd_pipeline_barrier(cb);

        // 7. LM head
        Self::matvec_weight(
            ctx, cb, &kv.decode_final_norm, &model.lm_head,
            &logits,
        );
        ctx.cmd_pipeline_barrier(cb);

        // DEBUG: Check logits before argmax
        if std::env::var("VK_DEBUG").is_ok() {
            ctx.cmd_end_submit_wait(cb)?;
            let logit_vals = logits.read_f32(ctx, 8)?;
            eprintln!("[vk-debug] logits[0..8] = {:?}", logit_vals);
            let last_vals = last_hidden.read_f32(ctx, 8)?;
            eprintln!("[vk-debug] last_hidden[0..8] = {:?}", last_vals);
            cb = ctx.cmd_begin()?;
        }

        // 8. Argmax
        kernels::activation::argmax(ctx, cb, &logits, &argmax_result, vocab_size);

        // 9. Submit and wait
        ctx.cmd_compute_to_transfer_barrier(cb);
        ctx.cmd_end_submit_wait(cb)?;

        // 10. Read result
        let result = argmax_result.read_u32(ctx, 2)?;
        let token_id = result[0];

        if std::env::var("VK_DEBUG").is_ok() {
            eprintln!("[vk-debug] prefill argmax: token_id={}, val_bits=0x{:08x}", token_id, result[1]);
            // Find top-5 logits on CPU for comparison
            let all_logits = logits.read_f32(ctx, vocab_size as usize)?;
            let mut indexed: Vec<(usize, f32)> = all_logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[vk-debug] prefill top-5 logits (CPU check):");
            for (i, (idx, val)) in indexed.iter().take(5).enumerate() {
                eprintln!("[vk-debug]   #{}: token {} = {:.4}", i, idx, val);
            }
        }

        // 11. Set sequence length
        kv.seq_len = tokens.len();

        Ok(token_id)
    }

    // ========================================================================
    // Public entry points
    // ========================================================================

    pub fn do_prefill(
        &self,
        input_tokens: &[u32],
        _opts: RunOpts,
    ) -> Result<(herbert_core::kv_cache::KvHandle, PrefillOutput)> {
        let mut kv = VulkanKvCache::new(&self.ctx, &self.config, self.max_tokens)?;
        let first_token = self.prefill_step(&mut kv, input_tokens)?;

        let output = PrefillOutput {
            logits: None,
            first_token,
            timings: None,
            cached_prefix_len: 0,
        };

        let handle = herbert_core::kv_cache::KvHandle::new(kv);
        Ok((handle, output))
    }

    pub fn do_decode(
        &self,
        kv: &mut herbert_core::kv_cache::KvHandle,
        token: u32,
        _opts: RunOpts,
    ) -> Result<DecodeOutput> {
        let kv_cache = kv.get_mut::<VulkanKvCache>()?;

        let next_token = self.decode_step(kv_cache, token)?;

        Ok(DecodeOutput {
            logits: None,
            token: next_token,
            timings: None,
        })
    }
}
