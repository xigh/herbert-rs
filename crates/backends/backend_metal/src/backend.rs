//! Metal inference backend — full decode and prefill pipeline for Qwen3.
//!
//! Simplified vs. the LFM2 backend:
//! - No conv layers: Qwen3 is pure attention+MLP.
//! - QK norms always present (non-optional).
//! - MoE routing: softmax only (no sigmoid+bias, no expert_bias).
//! - Three weight formats: BF16, INT8, Q4.

use std::sync::Arc;
use core::ptr::NonNull;

use block2::RcBlock;
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use herbert_core::backend::{DecodeOutput, PrefillOutput, RunOpts, VisionEmbedding};
use herbert_core::config::Config;
use herbert_core::error::{HerbertError, Result};

use herbert_backend_common::mrope;
use herbert_backend_common::position_ids::{self, ImageInfo};

use crate::context::MetalContext;
use crate::kernels;
use crate::kv_cache::MetalKvCache;
use crate::memory::MetalBuffer;
use crate::model::{MetalLayerMLP, MetalModel, MetalWeight, MoEQuantFormat};
use crate::profiler::ProfiledEncoder;

// ============================================================================
// Metal command buffer / encoder management
// ============================================================================

/// Manages a Metal command buffer and its current compute encoder.
///
/// MoE routing requires CPU readback mid-inference, which forces us to submit
/// the current command buffer, wait, read back router logits on the CPU, and
/// then start a new command buffer. This struct encapsulates that lifecycle.
struct EncoderState {
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

impl EncoderState {
    /// Create a new command buffer + compute encoder.
    fn new(ctx: &MetalContext) -> Result<Self> {
        let cb = ctx.begin_command_buffer()?;
        let encoder = MetalContext::new_compute_encoder(&cb)?;
        Ok(Self { cb, encoder })
    }

    /// End the current encoder, submit the command buffer, and wait for
    /// completion.
    fn submit_and_wait(self) -> Result<()> {
        self.encoder.endEncoding();
        MetalContext::submit_and_wait(&self.cb)
    }
}

// ============================================================================
// Prefill per-kernel profiling (cargo feature "prefill-profile")
// ============================================================================

/// Split the current CB, wait for GPU completion, record GPU timing,
/// then start a new CB + encoder.
///
/// Usage: `cargo build --release -p herbert-server --features herbert-backend-metal/prefill-profile`
/// then:  `METAL_PREFILL_PROFILE=5 herbert-server ...`
#[cfg(feature = "prefill-profile")]
fn prefill_profile_mark(
    ctx: &MetalContext,
    cb: &mut Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    enc: &mut Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    label: &str,
    entries: &mut Vec<(String, f64)>,
) -> Result<()> {
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    if let Some(error) = cb.error() {
        return Err(HerbertError::Backend(
            format!("Metal command buffer error (profile): {:?}", error),
        ));
    }
    let ms = unsafe {
        let s: f64 = objc2::msg_send![&**cb, GPUStartTime];
        let e: f64 = objc2::msg_send![&**cb, GPUEndTime];
        (e - s) * 1000.0
    };
    entries.push((label.to_string(), ms));
    *cb = ctx.begin_command_buffer()?;
    *enc = MetalContext::new_compute_encoder(cb)?;
    Ok(())
}

// ============================================================================
// Transformer forward output (shared between prefill and embed)
// ============================================================================

/// Result of the shared transformer forward pass.
///
/// `_keep_alive` retains all scratch buffers referenced by the encoded GPU work
/// until the caller submits the command buffer.
struct TransformerOutput {
    embed: MetalBuffer,
    hidden_size: u32,
    seq_len: u32,
    _keep_alive: Vec<MetalBuffer>,
}

// ============================================================================
// Metal backend inner state
// ============================================================================

/// Inner state of the Metal backend (populated after load).
///
/// Field order matters: Rust drops fields in declaration order.
/// `ctx` MUST be last because its Drop destroys the Metal device/queue,
/// and all MetalBuffers in `model` must be destroyed first.
pub struct MetalBackendInner {
    pub model: MetalModel,
    pub config: Config,
    pub max_tokens: usize,
    pub kv_quant: herbert_core::config::KvQuantType,
    pub kv_budget: Option<usize>,
    pub ctx: MetalContext, // MUST be last (drop order)
}

impl MetalBackendInner {
    // ========================================================================
    // Helper: matrix-vector multiply (dispatch BF16, Int8, or Q4)
    // ========================================================================

    fn matvec_weight(
        ctx: &MetalContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &MetalBuffer,
        weight: &MetalWeight,
        output: &MetalBuffer,
    ) {
        match weight {
            MetalWeight::BF16(w) => {
                kernels::matvec::bf16_matvec(
                    ctx, encoder, input, &w.packed, output,
                    w.n as u32, w.k as u32,
                );
            }
            MetalWeight::Int8(w) => {
                kernels::matvec::int8_matvec(
                    ctx, encoder, input, &w.packed, &w.scales, output,
                    w.n as u32, w.k as u32,
                );
            }
            MetalWeight::Q4(w) => {
                let n = w.n as u32;
                let use_v2 = ctx.pipelines.q4_matvec_v2.is_some()
                    && n.is_multiple_of(8)
                    && std::env::var("METAL_DISABLE_MATVEC_V2")
                        .map(|v| v != "1")
                        .unwrap_or(true);
                if use_v2 {
                    kernels::matvec::q4_matvec_v2(
                        ctx, encoder, input, &w.packed, &w.scales, output,
                        n, w.k as u32,
                    );
                } else {
                    kernels::matvec::q4_matvec(
                        ctx, encoder, input, &w.packed, &w.scales, output,
                        n, w.k as u32,
                    );
                }
            }
        }
    }

    fn matvec_weight_residual_add(
        ctx: &MetalContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &MetalBuffer,
        weight: &MetalWeight,
        residual: &MetalBuffer,
        output: &MetalBuffer,
    ) -> bool {
        match weight {
            MetalWeight::Q4(w) => {
                let n = w.n as u32;
                let disable_v2 = std::env::var("METAL_DISABLE_MATVEC_V2")
                    .map(|v| v == "1")
                    .unwrap_or(false);
                let use_v2 = ctx.pipelines.q4_matvec_residual_add_v2.is_some()
                    && n.is_multiple_of(8)
                    && !disable_v2;
                if use_v2 {
                    kernels::matvec::q4_matvec_residual_add_v2(
                        ctx, encoder, input, &w.packed, &w.scales, residual, output,
                        n, w.k as u32,
                    );
                } else {
                    let use_8row = ctx.device.supportsFamily(MTLGPUFamily::Apple9)
                        && n.is_multiple_of(8)
                        && std::env::var("METAL_DISABLE_Q4_DOWN_8ROW")
                            .map(|v| v != "1")
                            .unwrap_or(true);
                    if use_8row {
                        kernels::matvec::q4_matvec_residual_add_8row(
                            ctx, encoder, input, &w.packed, &w.scales, residual, output,
                            n, w.k as u32,
                        );
                    } else {
                        kernels::matvec::q4_matvec_residual_add(
                            ctx, encoder, input, &w.packed, &w.scales, residual, output,
                            n, w.k as u32,
                        );
                    }
                }
                true
            }
            _ => false,
        }
    }

    // ========================================================================
    // Helper: matrix-matrix multiply (dispatch BF16, Int8, or Q4)
    // ========================================================================

    #[allow(dead_code)]
    fn matmul_weight(
        ctx: &MetalContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &MetalBuffer,
        weight: &MetalWeight,
        output: &MetalBuffer,
        m: u32,
    ) {
        match weight {
            MetalWeight::BF16(w) => {
                kernels::matmul::bf16_matmul(
                    ctx, encoder, input, &w.packed, output,
                    m, w.n as u32, w.k as u32,
                );
            }
            MetalWeight::Int8(w) => {
                kernels::matmul::int8_matmul(
                    ctx, encoder, input, &w.packed, &w.scales, output,
                    m, w.n as u32, w.k as u32,
                );
            }
            MetalWeight::Q4(w) => {
                kernels::matmul::q4_matmul(
                    ctx, encoder, input, &w.packed, &w.scales, output,
                    m, w.n as u32, w.k as u32,
                );
            }
        }
    }

    /// Tiled matmul variant with shared memory (Phase 5). Only for Q4 and BF16;
    /// falls back to regular matmul for Int8.
    fn matmul_weight_tiled(
        ctx: &MetalContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &MetalBuffer,
        weight: &MetalWeight,
        output: &MetalBuffer,
        m: u32,
    ) {
        match weight {
            MetalWeight::BF16(w) => {
                kernels::matmul::bf16_matmul_tiled(
                    ctx, encoder, input, &w.packed, output,
                    m, w.n as u32, w.k as u32,
                );
            }
            MetalWeight::Q4(w) => {
                kernels::matmul::q4_matmul_tiled(
                    ctx, encoder, input, &w.packed, &w.scales, output,
                    m, w.n as u32, w.k as u32,
                );
            }
            MetalWeight::Int8(w) => {
                // No tiled variant for Int8 yet, fall back to regular
                kernels::matmul::int8_matmul(
                    ctx, encoder, input, &w.packed, &w.scales, output,
                    m, w.n as u32, w.k as u32,
                );
            }
        }
    }

    // ========================================================================
    // Decode step: process one token, return predicted next token ID
    // ========================================================================

    fn decode_step(&self, kv: &mut MetalKvCache, token: u32) -> Result<u32> {
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
        let rope_pos = kv.rope_pos as u32;    // for RoPE cos/sin lookup
        let cache_pos = kv.seq_len as u32;    // for KV cache slot + attention range
        let scale = 1.0 / (head_dim as f32).sqrt();

        // 1. Write token ID (unified memory, zero-copy)
        kv.token_buf.write_u32(token);

        // 2. Create profiled encoder (single CB when profiling disabled,
        //    per-phase CBs when METAL_PROFILE=1)
        let mut pe = ProfiledEncoder::new(ctx, cache_pos)?;

        // 3. Embedding lookup
        kernels::activation::embedding(
            ctx, pe.enc(), &model.embed_tokens, &kv.token_buf,
            &kv.decode_embed, 1, hidden_size,
        );
        pe.mark("embed")?;

        // 4. Transformer layers
        let detailed = pe.is_detailed();
        let use_dense_q4_gateup = ctx.device.supportsFamily(MTLGPUFamily::Apple9)
            && std::env::var("METAL_DISABLE_DENSE_Q4_GATEUP")
                .map(|v| v != "1")
                .unwrap_or(true);
        for l in 0..config.num_layers {
            let layer = &model.layers[l];

            // (a) Norm1: embed -> norm1
            kernels::norm::rms_norm(
                ctx, pe.enc(), &kv.decode_embed, &layer.input_layernorm,
                &kv.decode_norm1[l], hidden_size, eps,
            );

            // (b,c,d) Separate Q/K/V matvecs from normed input.
            // Separate dispatches are faster than fused q4_matvec_qkv_normed
            // because the fused kernel recomputes RMS norm in every threadgroup
            // (768x redundant on 2B), wasting ~60% of time on norm overhead.
            Self::matvec_weight(
                ctx, pe.enc(), &kv.decode_norm1[l],
                &layer.q_proj, &kv.decode_q[l],
            );
            Self::matvec_weight(
                ctx, pe.enc(), &kv.decode_norm1[l],
                &layer.k_proj, &kv.decode_k[l],
            );
            Self::matvec_weight(
                ctx, pe.enc(), &kv.decode_norm1[l],
                &layer.v_proj, &kv.decode_v[l],
            );
            pe.mark_detail(&format!("L{l}_qkv"))?;

            // (e+f) Fused QK norms + RoPE (Phase 4)
            let skip_norm = !config.has_qk_norm;
            let q_norm_buf = layer.q_norm.as_ref().unwrap_or(&model.dummy_norm_buf);
            let k_norm_buf = layer.k_norm.as_ref().unwrap_or(&model.dummy_norm_buf);
            kernels::norm::head_norm_rope(
                ctx, pe.enc(), &kv.decode_q[l], q_norm_buf,
                &model.cos_cache, &model.sin_cache,
                num_heads, head_dim, half_dim, rope_pos, eps, skip_norm,
            );

            // (g) Fused K norm + RoPE + KV cache append (K and V)
            let use_i8_kv = kv.i8_ready;
            if use_i8_kv {
                let ki = kv.keys_i8.as_ref().unwrap();
                let vi = kv.values_i8.as_ref().unwrap();
                let ks = kv.keys_scales.as_ref().unwrap();
                let vs = kv.values_scales.as_ref().unwrap();
                kernels::norm::head_norm_rope_kv_append_i8(
                    ctx, pe.enc(), &kv.decode_k[l], k_norm_buf,
                    &model.cos_cache, &model.sin_cache,
                    &ki[l], &ks[l],
                    &kv.decode_v[l], &vi[l], &vs[l],
                    num_kv_heads, head_dim, half_dim, kv_dim, cache_pos, rope_pos, eps, skip_norm,
                );
            } else {
                kernels::norm::head_norm_rope_kv_append(
                    ctx, pe.enc(), &kv.decode_k[l], k_norm_buf,
                    &model.cos_cache, &model.sin_cache,
                    &kv.keys[l],
                    &kv.decode_v[l], &kv.values[l],
                    num_kv_heads, head_dim, half_dim, kv_dim, cache_pos, rope_pos, eps, skip_norm,
                );
            }
            pe.mark_detail(&format!("L{l}_rope"))?;

            // (h) Attention decode
            let cached_len = cache_pos + 1;
            if use_i8_kv {
                let ki = kv.keys_i8.as_ref().unwrap();
                let vi = kv.values_i8.as_ref().unwrap();
                let ks = kv.keys_scales.as_ref().unwrap();
                let vs = kv.values_scales.as_ref().unwrap();
                kernels::attention::attention_decode_i8(
                    ctx, pe.enc(),
                    &kv.decode_q[l], &ki[l], &vi[l],
                    &ks[l], &vs[l],
                    &kv.decode_attn_out[l],
                    &kv.flash_decode_partials,
                    num_heads, num_kv_heads, head_dim, kv_dim,
                    cached_len, scale,
                );
            } else {
                kernels::attention::attention_decode(
                    ctx, pe.enc(),
                    &kv.decode_q[l], &kv.keys[l], &kv.values[l],
                    &kv.decode_attn_out[l],
                    &kv.flash_decode_partials,
                    num_heads, num_kv_heads, head_dim, kv_dim,
                    cached_len, scale,
                );
            }
            pe.mark_detail(&format!("L{l}_attn_k"))?;

            // (i) O projection
            Self::matvec_weight(
                ctx, pe.enc(), &kv.decode_attn_out[l],
                &layer.o_proj, &kv.decode_mlp_out[l],
            );

            // Coarse mode: single mark for entire attention phase
            // Detailed mode: mark O proj separately
            if detailed {
                pe.mark_detail(&format!("L{l}_oproj"))?;
            } else {
                pe.mark(&format!("L{l}_attn"))?;
            }

            // (j+k) Fused residual add + RMS norm 2
            kernels::norm::rms_norm_residual(
                ctx, pe.enc(), &kv.decode_embed,
                &kv.decode_mlp_out[l], &layer.post_attention_layernorm,
                &kv.decode_norm2[l], hidden_size, eps,
            );
            pe.mark_detail(&format!("L{l}_norm2"))?;

            // MLP
            match &layer.mlp {
                MetalLayerMLP::Dense { gate_proj, up_proj, down_proj } => {
                    // Keep the Q4 dense gate/up fast path scoped to Apple9+, which
                    // is the configuration where we benchmarked and retained it.
                    let fused_q4 = use_dense_q4_gateup
                        && matches!((gate_proj, up_proj), (MetalWeight::Q4(_), MetalWeight::Q4(_)))
                        && kernels::moe::fused_gate_up_swiglu_weight(
                            ctx, pe.enc(), &kv.decode_norm2[l],
                            gate_proj, up_proj, &kv.decode_mlp_gate[l],
                        );
                    let fused_bf16 = !fused_q4 && kernels::activation::fused_gate_up_swiglu(
                        ctx, pe.enc(), &kv.decode_norm2[l],
                        gate_proj, up_proj, &kv.decode_mlp_gate[l],
                    );

                    if !fused_q4 && !fused_bf16 {
                        Self::matvec_weight(
                            ctx, pe.enc(), &kv.decode_norm2[l], gate_proj,
                            &kv.decode_mlp_gate[l],
                        );
                        if detailed {
                            pe.mark_detail(&format!("L{l}_gate"))?;
                        }
                        Self::matvec_weight(
                            ctx, pe.enc(), &kv.decode_norm2[l], up_proj,
                            &kv.decode_mlp_up[l],
                        );
                        if detailed {
                            pe.mark_detail(&format!("L{l}_up"))?;
                        }
                        kernels::activation::swiglu(
                            ctx, pe.enc(), &kv.decode_mlp_gate[l],
                            &kv.decode_mlp_up[l],
                            config.intermediate_size as u32,
                        );
                        if detailed {
                            pe.mark_detail(&format!("L{l}_swiglu"))?;
                        }
                    } else if detailed {
                        pe.mark_detail(&format!("L{l}_gateup"))?;
                    }

                    let fused_down = Self::matvec_weight_residual_add(
                        ctx, pe.enc(), &kv.decode_mlp_gate[l], down_proj,
                        &kv.decode_embed, &kv.decode_embed,
                    );
                    if !fused_down {
                        Self::matvec_weight(
                            ctx, pe.enc(), &kv.decode_mlp_gate[l], down_proj,
                            &kv.decode_mlp_out[l],
                        );
                    }
                    if detailed {
                        pe.mark_detail(&format!("L{l}_down"))?;
                    }

                    if !fused_down {
                        kernels::activation::residual_add(
                            ctx, pe.enc(), &kv.decode_embed,
                            &kv.decode_mlp_out[l], hidden_size,
                        );
                    }
                }

                MetalLayerMLP::MoE {
                    router, weights, num_experts, num_experts_per_tok,
                    moe_intermediate_size, norm_topk_prob,
                } => {
                    let ne = *num_experts as u32;
                    let k = *num_experts_per_tok as u32;
                    let moe_inter = *moe_intermediate_size as u32;
                    let router_logits = kv.moe_router_logits.as_ref().unwrap();
                    let expert_ids_buf = kv.moe_expert_ids.as_ref().unwrap();
                    let expert_weights_buf = kv.moe_expert_weights.as_ref().unwrap();
                    let moe_inter_batched = kv.moe_inter_batched.as_ref().unwrap();
                    let moe_down_output = kv.moe_down_output.as_ref().unwrap();

                    let is_q4 = weights.format == MoEQuantFormat::Q4;
                    let is_int8 = weights.format == MoEQuantFormat::Int8;

                    kernels::matvec::f32_matvec(
                        ctx, pe.enc(), &kv.decode_norm2[l], router,
                        router_logits, ne, hidden_size,
                    );

                    kernels::moe::softmax_topk(
                        ctx, pe.enc(), router_logits,
                        expert_ids_buf, expert_weights_buf,
                        ne, k, *norm_topk_prob,
                    );
                    pe.mark_detail(&format!("L{l}_router"))?;

                    kernels::moe::batched_gate_up_swiglu(
                        ctx, pe.enc(),
                        &kv.decode_norm2[l],
                        &weights.gate_packed,
                        weights.gate_scales.as_ref(),
                        &weights.up_packed,
                        weights.up_scales.as_ref(),
                        expert_ids_buf,
                        moe_inter_batched,
                        moe_inter, hidden_size, k,
                        is_q4, is_int8,
                    );
                    pe.mark_detail(&format!("L{l}_gateup"))?;

                    kernels::moe::batched_down(
                        ctx, pe.enc(),
                        moe_inter_batched,
                        &weights.down_packed,
                        weights.down_scales.as_ref(),
                        expert_ids_buf,
                        expert_weights_buf,
                        moe_down_output,
                        hidden_size, moe_inter, k,
                        is_q4, is_int8,
                    );
                    pe.mark_detail(&format!("L{l}_down"))?;

                    kernels::moe::moe_reduce_residual(
                        ctx, pe.enc(),
                        moe_down_output, &kv.decode_embed,
                        hidden_size, k,
                    );
                }
            }

            // Coarse mode: single mark for entire MLP phase
            // Detailed mode: mark reduce+residual separately
            if detailed {
                pe.mark_detail(&format!("L{l}_resid"))?;
            } else {
                pe.mark(&format!("L{l}_mlp"))?;
            }
        }

        // 5. Final RMS norm
        kernels::norm::rms_norm(
            ctx, pe.enc(), &kv.decode_embed, &model.final_norm,
            &kv.decode_final_norm, hidden_size, eps,
        );
        pe.mark("final_norm")?;

        // 6. LM head
        Self::matvec_weight(
            ctx, pe.enc(), &kv.decode_final_norm, &model.lm_head,
            &kv.decode_logits,
        );
        pe.mark("lm_head")?;

        // 7. Argmax (two-stage parallel reduction)
        kernels::activation::argmax(
            ctx, pe.enc(), &kv.decode_logits,
            &kv.argmax_intermediate, &kv.argmax_result,
            vocab_size,
        );

        // 8. Submit, wait, and print profiling (if enabled)
        pe.finish("argmax")?;

        // 9. Read result
        let token_id = kv.argmax_result.read_u32();

        // 10. Advance sequence position
        kv.seq_len += 1;
        kv.rope_pos += 1;

        Ok(token_id)
    }

    // ========================================================================
    // Prefill step: process all prompt tokens in a single pass
    // ========================================================================

    // ========================================================================
    // Shared transformer forward pass (used by prefill and embed)
    // ========================================================================

    fn encode_transformer_forward(
        &self,
        cb: &mut Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        enc: &mut Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        kv: &mut MetalKvCache,
        tokens: &[u32],
        prebuilt_embeds: Option<&[f32]>,
        cos_sin_override: Option<(&MetalBuffer, &MetalBuffer)>,
        deepstack: Option<(&[MetalBuffer], &MetalBuffer, u32)>,
        progress_split: Option<(&ProtocolObject<dyn MTLEvent>, usize)>,
    ) -> Result<TransformerOutput> {
        let ctx = &self.ctx;
        let model = &self.model;
        let config = &self.config;
        let device = &*ctx.device;

        #[cfg(feature = "prefill-profile")]
        let prefill_profile_layer: Option<usize> = std::env::var("METAL_PREFILL_PROFILE")
            .ok()
            .and_then(|v| v.parse().ok());
        #[cfg(feature = "prefill-profile")]
        let mut profile_entries: Vec<(String, f64)> = Vec::new();

        // Max layers per command buffer — prevents GPU watchdog timeout on long prefills.
        // At 22K tokens × 4 layers per CB → ~15s per CB on M3 Ultra (well within ~60s limit).
        //
        // Metal 4 (M5): GPU is ~1.75× faster (MPP matmul2d), so we can safely fit more
        // layers per CB. For sequences ≤ 8192 tokens on a 4B model, all 36 layers complete
        // in ~20-30s — well within the 60s watchdog limit.
        // This eliminates all CB split overhead (8× CPU-GPU roundtrips for 36 layers).
        let default_layers = if ctx.metal4 { config.num_layers } else { 4 };
        let max_safe = if ctx.metal4 {
            config.num_layers  // Metal 4: no hard cap, trust the faster GPU
        } else {
            (config.num_layers / 2).max(2)  // Pre-Metal 4: conservative cap
        };
        let layers_per_cb: usize = std::env::var("METAL_LAYERS_PER_CB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_layers)
            .min(max_safe);

        let seq_len = tokens.len() as u32;
        let kv_offset = kv.seq_len as u32;
        let hidden_size = config.hidden_size as u32;
        let num_heads = config.num_attention_heads as u32;
        let num_kv_heads = config.num_key_value_heads as u32;
        let head_dim = config.head_dim as u32;
        let half_dim = (config.rotary_ndims / 2) as u32;
        let q_dim = config.q_dim() as u32;
        let kv_dim = config.kv_dim() as u32;
        let eps = config.rms_norm_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let embed = MetalBuffer::new_uninit(device, (seq_len * hidden_size * 4) as u64)?;
        let norm = MetalBuffer::new_uninit(device, (seq_len * hidden_size * 4) as u64)?;
        let q_buf = MetalBuffer::new_uninit(device, (seq_len * q_dim * 4) as u64)?;
        let k_buf = MetalBuffer::new_uninit(device, (seq_len * kv_dim * 4) as u64)?;
        let v_buf = MetalBuffer::new_uninit(device, (seq_len * kv_dim * 4) as u64)?;
        let attn_out = MetalBuffer::new_uninit(device, (seq_len * q_dim * 4) as u64)?;
        let mlp_out = MetalBuffer::new_uninit(device, (seq_len * hidden_size * 4) as u64)?;
        let mlp_gate = MetalBuffer::new_uninit(device, (seq_len * config.intermediate_size as u32 * 4) as u64)?;
        let mlp_up = MetalBuffer::new_uninit(device, (seq_len * config.intermediate_size as u32 * 4) as u64)?;
        let norm2 = MetalBuffer::new_uninit(device, (seq_len * hidden_size * 4) as u64)?;

        let moe_bufs = if config.is_moe() {
            let ne = config.num_experts.unwrap();
            let top_k = config.num_experts_per_tok.unwrap();
            let moe_inter = config.moe_intermediate_size.unwrap();
            Some((
                MetalBuffer::new_uninit(device, (seq_len as usize * ne * 4) as u64)?,
                MetalBuffer::new_uninit(device, (seq_len as usize * top_k * 4) as u64)?,
                MetalBuffer::new_uninit(device, (seq_len as usize * top_k * 4) as u64)?,
                MetalBuffer::new_uninit(device, (seq_len as usize * top_k * moe_inter * 4) as u64)?,
                MetalBuffer::new_uninit(device, (seq_len as usize * top_k * config.hidden_size as usize * 4) as u64)?,
                MetalBuffer::new(device, (seq_len as usize * config.hidden_size as usize * 4) as u64)?,
                MetalBuffer::new_uninit(device, (ne * 4) as u64)?,
                MetalBuffer::new_uninit(device, (ne * 4) as u64)?,
                MetalBuffer::new_uninit(device, (seq_len as usize * top_k * 4) as u64)?,
                MetalBuffer::new_uninit(device, 24u64)?,
            ))
        } else {
            None
        };

        let embed_input = if let Some(embeds_f32) = prebuilt_embeds {
            let embed_bytes: Vec<u8> = embeds_f32.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            MetalBuffer::from_data(device, &embed_bytes)?
        } else {
            let token_bytes: Vec<u8> = tokens.iter()
                .flat_map(|t| t.to_le_bytes())
                .collect();
            MetalBuffer::from_data(device, &token_bytes)?
        };

        if prebuilt_embeds.is_some() {
            kernels::activation::copy_buffer(
                ctx, enc, &embed_input, &embed,
                seq_len * hidden_size,
            );
        } else {
            kernels::activation::embedding(
                ctx, enc, &model.embed_tokens, &embed_input, &embed,
                seq_len, hidden_size,
            );
        }

        let mut signal_idx = 0usize;
        for l in 0..config.num_layers {
            let layer = &model.layers[l];
            #[cfg(feature = "prefill-profile")]
            let profiling = prefill_profile_layer == Some(l);

            // Flush all prior GPU work before starting per-kernel timing
            #[cfg(feature = "prefill-profile")]
            if profiling {
                prefill_profile_mark(ctx, cb, enc, "_flush", &mut profile_entries)?;
            }

            kernels::norm::rms_norm_batch(
                ctx, enc, &embed, &layer.input_layernorm, &norm,
                hidden_size, eps, seq_len,
            );
            #[cfg(feature = "prefill-profile")]
            if profiling {
                prefill_profile_mark(ctx, cb, enc, "norm1", &mut profile_entries)?;
            }

            Self::matmul_weight_tiled(
                ctx, enc, &norm, &layer.q_proj, &q_buf, seq_len,
            );
            Self::matmul_weight_tiled(
                ctx, enc, &norm, &layer.k_proj, &k_buf, seq_len,
            );
            Self::matmul_weight_tiled(
                ctx, enc, &norm, &layer.v_proj, &v_buf, seq_len,
            );
            #[cfg(feature = "prefill-profile")]
            if profiling {
                prefill_profile_mark(ctx, cb, enc, "qkv", &mut profile_entries)?;
            }

            let (cos_buf, sin_buf, rope_start) = match cos_sin_override {
                Some((c, s)) => (c, s, 0u32),
                None => (&model.cos_cache, &model.sin_cache, kv_offset),
            };
            let skip_norm = !config.has_qk_norm;
            let q_norm_buf = layer.q_norm.as_ref().unwrap_or(&model.dummy_norm_buf);
            let k_norm_buf = layer.k_norm.as_ref().unwrap_or(&model.dummy_norm_buf);
            kernels::norm::head_norm_rope_batch(
                ctx, enc, &q_buf, q_norm_buf,
                cos_buf, sin_buf,
                num_heads, head_dim, half_dim, seq_len, rope_start, eps, skip_norm,
            );
            kernels::norm::head_norm_rope_batch(
                ctx, enc, &k_buf, k_norm_buf,
                cos_buf, sin_buf,
                num_kv_heads, head_dim, half_dim, seq_len, rope_start, eps, skip_norm,
            );
            #[cfg(feature = "prefill-profile")]
            if profiling {
                prefill_profile_mark(ctx, cb, enc, "rope", &mut profile_entries)?;
            }

            // Prefill always writes to half-precision KV cache (existing attention shaders read half*)
            kernels::attention::kv_cache_append_batch(
                ctx, enc, &kv.keys[l], &k_buf, kv_dim, kv_offset, seq_len,
            );
            kernels::attention::kv_cache_append_batch(
                ctx, enc, &kv.values[l], &v_buf, kv_dim, kv_offset, seq_len,
            );
            #[cfg(feature = "prefill-profile")]
            if profiling {
                prefill_profile_mark(ctx, cb, enc, "kv_append", &mut profile_entries)?;
            }

            kernels::attention::attention_prefill(
                ctx, enc, &q_buf, &kv.keys[l], &kv.values[l],
                &attn_out,
                seq_len, num_heads, num_kv_heads, head_dim, kv_dim, q_dim,
                kv_offset + seq_len,
                kv_offset,
                scale,
            );
            #[cfg(feature = "prefill-profile")]
            if profiling {
                prefill_profile_mark(ctx, cb, enc, "attention", &mut profile_entries)?;
            }

            Self::matmul_weight_tiled(
                ctx, enc, &attn_out, &layer.o_proj, &mlp_out, seq_len,
            );
            #[cfg(feature = "prefill-profile")]
            if profiling {
                prefill_profile_mark(ctx, cb, enc, "oproj", &mut profile_entries)?;
            }

            kernels::activation::residual_add(
                ctx, enc, &embed, &mlp_out, seq_len * hidden_size,
            );
            #[cfg(feature = "prefill-profile")]
            if profiling {
                prefill_profile_mark(ctx, cb, enc, "resid1", &mut profile_entries)?;
            }

            kernels::norm::rms_norm_batch(
                ctx, enc, &embed, &layer.post_attention_layernorm,
                &norm2, hidden_size, eps, seq_len,
            );
            #[cfg(feature = "prefill-profile")]
            if profiling {
                prefill_profile_mark(ctx, cb, enc, "norm2", &mut profile_entries)?;
            }

            match &layer.mlp {
                MetalLayerMLP::Dense { gate_proj, up_proj, down_proj } => {
                    Self::matmul_weight_tiled(
                        ctx, enc, &norm2, gate_proj, &mlp_gate, seq_len,
                    );
                    #[cfg(feature = "prefill-profile")]
                    if profiling {
                        prefill_profile_mark(ctx, cb, enc, "gate", &mut profile_entries)?;
                    }
                    Self::matmul_weight_tiled(
                        ctx, enc, &norm2, up_proj, &mlp_up, seq_len,
                    );
                    #[cfg(feature = "prefill-profile")]
                    if profiling {
                        prefill_profile_mark(ctx, cb, enc, "up", &mut profile_entries)?;
                    }

                    kernels::activation::swiglu(
                        ctx, enc, &mlp_gate, &mlp_up,
                        seq_len * config.intermediate_size as u32,
                    );
                    #[cfg(feature = "prefill-profile")]
                    if profiling {
                        prefill_profile_mark(ctx, cb, enc, "swiglu", &mut profile_entries)?;
                    }

                    Self::matmul_weight_tiled(
                        ctx, enc, &mlp_gate, down_proj, &mlp_out, seq_len,
                    );
                    #[cfg(feature = "prefill-profile")]
                    if profiling {
                        prefill_profile_mark(ctx, cb, enc, "down", &mut profile_entries)?;
                    }

                    kernels::activation::residual_add(
                        ctx, enc, &embed, &mlp_out, seq_len * hidden_size,
                    );
                    #[cfg(feature = "prefill-profile")]
                    if profiling {
                        prefill_profile_mark(ctx, cb, enc, "resid2", &mut profile_entries)?;
                    }
                }

                MetalLayerMLP::MoE {
                    router, weights, num_experts, num_experts_per_tok,
                    moe_intermediate_size, norm_topk_prob,
                } => {
                    let ne = *num_experts as u32;
                    let top_k = *num_experts_per_tok as u32;
                    let moe_inter = *moe_intermediate_size as u32;
                    let is_q4 = weights.format == MoEQuantFormat::Q4;
                    let is_int8 = weights.format == MoEQuantFormat::Int8;
                    let (router_buf, expert_ids_buf, expert_weights_buf,
                         gate_up_out, down_out, counters,
                         expert_counts, expert_offsets, sorted_src_idx,
                         indirect_args)
                        = moe_bufs.as_ref().unwrap();

                    kernels::matmul::f32_matmul(
                        ctx, enc, &norm2, router, router_buf,
                        seq_len, ne, hidden_size,
                    );
                    #[cfg(feature = "prefill-profile")]
                    if profiling {
                        prefill_profile_mark(ctx, cb, enc, "router", &mut profile_entries)?;
                    }

                    kernels::moe::softmax_topk_batch(
                        ctx, enc, router_buf,
                        expert_ids_buf, expert_weights_buf,
                        ne, top_k, *norm_topk_prob, seq_len,
                    );
                    #[cfg(feature = "prefill-profile")]
                    if profiling {
                        prefill_profile_mark(ctx, cb, enc, "softmax_topk", &mut profile_entries)?;
                    }

                    kernels::moe::prefill_sort(
                        ctx, enc, expert_ids_buf,
                        expert_counts, expert_offsets, sorted_src_idx,
                        indirect_args,
                        seq_len * top_k, ne, top_k, moe_inter, hidden_size,
                        is_q4,
                    );
                    #[cfg(feature = "prefill-profile")]
                    if profiling {
                        prefill_profile_mark(ctx, cb, enc, "sort", &mut profile_entries)?;
                    }

                    kernels::moe::prefill_tiled_gate_up_swiglu(
                        ctx, enc, &norm2,
                        &weights.gate_packed,
                        weights.gate_scales.as_ref(),
                        &weights.up_packed,
                        weights.up_scales.as_ref(),
                        expert_counts, expert_offsets, sorted_src_idx,
                        gate_up_out,
                        indirect_args,
                        moe_inter, hidden_size, top_k,
                        is_q4, is_int8,
                    );
                    #[cfg(feature = "prefill-profile")]
                    if profiling {
                        prefill_profile_mark(ctx, cb, enc, "gate_up_swiglu", &mut profile_entries)?;
                    }

                    kernels::moe::prefill_tiled_down_residual(
                        ctx, enc, gate_up_out,
                        &weights.down_packed,
                        weights.down_scales.as_ref(),
                        expert_weights_buf,
                        expert_counts, expert_offsets, sorted_src_idx,
                        down_out, counters, &embed,
                        indirect_args,
                        hidden_size, moe_inter, top_k,
                        is_q4, is_int8,
                    );
                    #[cfg(feature = "prefill-profile")]
                    if profiling {
                        prefill_profile_mark(ctx, cb, enc, "down_residual", &mut profile_entries)?;
                    }
                }
            }

            if let Some((ds_features, ds_indices, ds_count)) = deepstack {
                if l < ds_features.len() {
                    kernels::activation::deepstack_add(
                        ctx, enc,
                        &embed, &ds_features[l], ds_indices,
                        hidden_size, ds_count,
                    );
                }
            }

            // Split command buffer to stay under macOS GPU watchdog timeout.
            // For short sequences (< 256 tokens), skip splitting — overhead not worth it.
            let needs_cb_split = seq_len >= 256
                && (l + 1) % layers_per_cb == 0
                && l + 1 < config.num_layers;

            // At group boundary: signal event for async progress.
            if let Some((event_ref, group_size)) = progress_split {
                if (l + 1) % group_size == 0 && l + 1 < config.num_layers {
                    enc.endEncoding();
                    signal_idx += 1;
                    cb.encodeSignalEvent_value(event_ref, signal_idx as u64);
                    if needs_cb_split {
                        // Combine progress signal + CB split: commit current CB, start fresh.
                        MetalContext::submit_and_wait(cb)?;
                        *cb = ctx.begin_command_buffer()?;
                    }
                    *enc = MetalContext::new_compute_encoder(cb)?;
                }
            }

            // CB split without progress (when no progress_split or different boundary)
            if needs_cb_split {
                let already_split = progress_split.map_or(false, |(_, gs)| {
                    (l + 1) % gs == 0 && l + 1 < config.num_layers
                });
                if !already_split {
                    enc.endEncoding();
                    MetalContext::submit_and_wait(cb)?;
                    *cb = ctx.begin_command_buffer()?;
                    *enc = MetalContext::new_compute_encoder(cb)?;
                }
            }
        }

        // Print prefill profiling report
        #[cfg(feature = "prefill-profile")]
        if !profile_entries.is_empty() {
            // Exclude _flush (drains prior layers) from the total
            let total_ms: f64 = profile_entries.iter()
                .filter(|(label, _)| !label.starts_with('_'))
                .map(|(_, ms)| ms)
                .sum();
            let layer = prefill_profile_layer.unwrap();
            let layer_type = if config.is_moe_layer(layer) { "MoE" } else { "Dense" };
            eprintln!(
                "\n[PREFILL-PROFILE] Layer {} ({}), seq_len={}, total={:.1}ms:",
                layer, layer_type, seq_len, total_ms,
            );
            for (label, ms) in &profile_entries {
                if label.starts_with('_') {
                    eprintln!("  {:<20} {:>8.1}ms (prior layers)", label, ms);
                } else {
                    let pct = if total_ms > 0.0 { ms / total_ms * 100.0 } else { 0.0 };
                    eprintln!("  {:<20} {:>8.1}ms ({:>5.1}%)", label, ms, pct);
                }
            }
            eprintln!();
        }

        let mut keep_alive = vec![
            embed_input,
            norm,
            q_buf,
            k_buf,
            v_buf,
            attn_out,
            mlp_out,
            mlp_gate,
            mlp_up,
            norm2,
        ];
        if let Some((
            router_buf,
            expert_ids_buf,
            expert_weights_buf,
            gate_up_out,
            down_out,
            counters,
            expert_counts,
            expert_offsets,
            sorted_src_idx,
            indirect_args,
        )) = moe_bufs {
            keep_alive.extend([
                router_buf,
                expert_ids_buf,
                expert_weights_buf,
                gate_up_out,
                down_out,
                counters,
                expert_counts,
                expert_offsets,
                sorted_src_idx,
                indirect_args,
            ]);
        }

        Ok(TransformerOutput {
            embed,
            hidden_size,
            seq_len,
            _keep_alive: keep_alive,
        })
    }

    fn transformer_forward(
        &self,
        kv: &mut MetalKvCache,
        tokens: &[u32],
        prebuilt_embeds: Option<&[f32]>,
        cos_sin_override: Option<(&MetalBuffer, &MetalBuffer)>,
        deepstack: Option<(&[MetalBuffer], &MetalBuffer, u32)>,
    ) -> Result<TransformerOutput> {
        let mut cb = self.ctx.begin_command_buffer()?;
        let mut enc = MetalContext::new_compute_encoder(&cb)?;
        let output = self.encode_transformer_forward(
            &mut cb,
            &mut enc,
            kv,
            tokens,
            prebuilt_embeds,
            cos_sin_override,
            deepstack,
            None,
        )?;
        enc.endEncoding();
        MetalContext::submit_and_wait(&cb)?;
        Ok(output)
    }

    fn prefill_step(
        &self,
        kv: &mut MetalKvCache,
        tokens: &[u32],
        prebuilt_embeds: Option<&[f32]>,
        cos_sin_override: Option<(&MetalBuffer, &MetalBuffer)>,
        deepstack: Option<(&[MetalBuffer], &MetalBuffer, u32)>,
        progress: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
    ) -> Result<u32> {
        let ctx = &self.ctx;
        let model = &self.model;
        let config = &self.config;
        let hidden_size = config.hidden_size as u32;
        let vocab_size = config.vocab_size as u32;
        let eps = config.rms_norm_eps;
        let num_layers = config.num_layers;

        // --- MTLSharedEvent setup for async per-layer progress ---
        let group_size = if progress.is_some() {
            std::env::var("PREFILL_PROGRESS_GROUP")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(2) // every 2 layers → ~24 progress events (no CB split overhead)
        } else {
            num_layers + 1 // no splitting
        };

        let boundaries: Vec<usize> = (1..=num_layers)
            .filter(|&i| i % group_size == 0 && i < num_layers)
            .collect();

        type NotifyBlock = RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTLSharedEvent>>, u64)>;
        let async_state: Option<(
            Retained<ProtocolObject<dyn MTLSharedEvent>>,
            Retained<MTLSharedEventListener>,
            Vec<NotifyBlock>,
        )> = if let Some(ref progress_fn) = progress {
            if boundaries.is_empty() {
                None
            } else {
                let shared_event = ctx.device.newSharedEvent()
                    .ok_or_else(|| HerbertError::Backend("Failed to create MTLSharedEvent".into()))?;

                let queue = dispatch2::DispatchQueue::new("herbert.prefill.progress", None);
                let listener = unsafe {
                    MTLSharedEventListener::initWithDispatchQueue(
                        MTLSharedEventListener::alloc(),
                        &queue,
                    )
                };

                let total_tokens = tokens.len();
                let mut blocks_to_keep: Vec<NotifyBlock> = Vec::with_capacity(boundaries.len());

                for (idx, &layers_done) in boundaries.iter().enumerate() {
                    let signal_val = (idx + 1) as u64;
                    let p = progress_fn.clone();
                    let rc_block: NotifyBlock = RcBlock::new(
                        move |_event: NonNull<ProtocolObject<dyn MTLSharedEvent>>, _value: u64| {
                            // Report progress proportional to layers completed
                            let tokens_approx = layers_done * total_tokens / num_layers;
                            p(tokens_approx, total_tokens);
                        },
                    );
                    unsafe {
                        shared_event.notifyListener_atValue_block(
                            &listener,
                            signal_val,
                            &*rc_block as *const _ as *mut _,
                        );
                    }
                    blocks_to_keep.push(rc_block);
                }

                Some((shared_event, listener, blocks_to_keep))
            }
        } else {
            None
        };

        // Build event reference for progress splitting
        let progress_split = async_state.as_ref().map(|(shared_event, _, _)| {
            let event_ref: &ProtocolObject<dyn MTLEvent> =
                ProtocolObject::from_ref(&**shared_event);
            (event_ref, group_size)
        });

        // --- Transformer forward + lm_head + argmax ---
        let mut cb = ctx.begin_command_buffer()?;
        let mut enc = MetalContext::new_compute_encoder(&cb)?;

        let output = self.encode_transformer_forward(
            &mut cb,
            &mut enc,
            kv,
            tokens,
            prebuilt_embeds,
            cos_sin_override,
            deepstack,
            progress_split,
        )?;

        let last_hidden = output.embed.slice(
            (output.seq_len - 1) as usize * output.hidden_size as usize * 4,
            output.hidden_size as usize * 4,
        );

        kernels::norm::rms_norm(
            ctx, &enc, &last_hidden, &model.final_norm,
            &kv.decode_final_norm, hidden_size, eps,
        );

        Self::matvec_weight(
            ctx, &enc, &kv.decode_final_norm, &model.lm_head,
            &kv.decode_logits,
        );

        kernels::activation::argmax(
            ctx, &enc, &kv.decode_logits,
            &kv.argmax_intermediate, &kv.argmax_result,
            vocab_size,
        );

        enc.endEncoding();
        MetalContext::submit_and_wait(&cb)?;

        // Drop async state after final wait (listener, RcBlocks, shared event)
        drop(async_state);

        let token_id = kv.argmax_result.read_u32();
        kv.seq_len += tokens.len();

        Ok(token_id)
    }


    // ========================================================================
    // Embed step: stateless forward pass → L2-normalised hidden state
    // ========================================================================

    fn embed_step(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let ctx = &self.ctx;
        let model = &self.model;
        let config = &self.config;
        let device = &*ctx.device;
        let hidden_size = config.hidden_size;
        let eps = config.rms_norm_eps;

        // Temporary KV cache sized to seq_len (minimal allocation, stateless)
        // Embed doesn't do decode, so always use half precision
        let mut kv = MetalKvCache::new(device, config, tokens.len(), herbert_core::config::KvQuantType::BF16, None)?;

        // Shared transformer forward
        let output = self.transformer_forward(&mut kv, tokens, None, None, None)?;

        // Bind the last token hidden state as a sub-buffer view
        let last_hidden = output.embed.slice(
            (output.seq_len - 1) as usize * hidden_size * 4,
            hidden_size * 4,
        );

        // Final RMS norm on GPU
        let normed = MetalBuffer::new_uninit(device, (hidden_size * 4) as u64)?;
        let es = EncoderState::new(ctx)?;
        kernels::norm::rms_norm(
            ctx, &es.encoder, &last_hidden, &model.final_norm,
            &normed, hidden_size as u32, eps,
        );
        es.submit_and_wait()?;

        // Read back + L2 normalize on CPU (trivial for ~2048 floats)
        let mut result = normed.read_f32(hidden_size);
        let norm_sq: f32 = result.iter().map(|v| v * v).sum();
        let norm_val = norm_sq.sqrt();
        if norm_val > 0.0 {
            let inv = 1.0 / norm_val;
            for v in &mut result {
                *v *= inv;
            }
        }

        Ok(result)
    }

    pub fn do_embed(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(HerbertError::Backend("embed: empty token sequence".into()));
        }
        self.embed_step(tokens)
    }

    pub fn do_embed_vl(
        &self,
        tokens: &[u32],
        images: &[VisionEmbedding],
        image_positions: &[usize],
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(HerbertError::Backend("embed_vl: empty token sequence".into()));
        }
        if images.len() != image_positions.len() {
            return Err(HerbertError::Backend(format!(
                "embed_vl: images ({}) and image_positions ({}) must have same length",
                images.len(), image_positions.len(),
            )));
        }

        let config = &self.config;
        let model = &self.model;
        let device = &*self.ctx.device;
        let hidden_size = config.hidden_size;
        let total_len = tokens.len();

        // 1. Build ImageInfo list
        let image_infos: Vec<ImageInfo> = images
            .iter()
            .zip(image_positions.iter())
            .map(|(img, &pos)| ImageInfo {
                start_idx: pos,
                grid_h: img.grid_h,
                grid_w: img.grid_w,
            })
            .collect();

        // 2. Build combined embeddings on CPU (text lookup + image hidden states)
        let embed_f32 = unsafe {
            let ptr = model.embed_tokens.contents_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, config.vocab_size * hidden_size)
        };

        let mut combined = vec![0.0f32; total_len * hidden_size];
        let mut tok_idx = 0usize;
        let mut img_idx = 0usize;
        while tok_idx < total_len {
            if img_idx < image_infos.len() && tok_idx == image_infos[img_idx].start_idx {
                let n = image_infos[img_idx].num_tokens();
                let dst = &mut combined[tok_idx * hidden_size..(tok_idx + n) * hidden_size];
                dst.copy_from_slice(&images[img_idx].hidden_states[..n * hidden_size]);
                tok_idx += n;
                img_idx += 1;
            } else {
                let t = tokens[tok_idx] as usize;
                let dst = &mut combined[tok_idx * hidden_size..(tok_idx + 1) * hidden_size];
                dst.copy_from_slice(&embed_f32[t * hidden_size..(t + 1) * hidden_size]);
                tok_idx += 1;
            }
        }

        // 3. Build MRoPE cos/sin on CPU
        let half_dim = config.rotary_ndims / 2;
        let positions = position_ids::build_vl_position_ids(total_len, &image_infos);
        let inv_freq = mrope::compute_inv_freq(config.head_dim, config.rope_theta);
        let mrope_section = config.mrope_section.unwrap_or([
            half_dim, 0, 0,
        ]);
        let pattern = mrope::build_mrope_pattern(mrope_section);

        let mut cos_data = vec![0.0f32; total_len * half_dim];
        let mut sin_data = vec![0.0f32; total_len * half_dim];
        for (i, pos) in positions.iter().enumerate() {
            let offset = i * half_dim;
            mrope::compute_mrope_cos_sin(
                pos.as_array(),
                &inv_freq,
                &pattern,
                &mut cos_data[offset..offset + half_dim],
                &mut sin_data[offset..offset + half_dim],
            );
        }

        // 4. Upload cos/sin to GPU
        let cos_buf = MetalBuffer::from_f32(device, &cos_data)?;
        let sin_buf = MetalBuffer::from_f32(device, &sin_data)?;

        // 5. Build DeepStack GPU buffers
        let image_token_indices: Vec<u32> = image_infos
            .iter()
            .flat_map(|img| (img.start_idx..img.end_idx()).map(|i| i as u32))
            .collect();
        let num_image_tokens = image_token_indices.len() as u32;

        let max_ds_layers = images.iter().map(|img| img.deepstack_features.len()).max().unwrap_or(0);
        let mut ds_feature_bufs: Vec<MetalBuffer> = Vec::with_capacity(max_ds_layers);
        for layer_idx in 0..max_ds_layers {
            let mut combined_feats = Vec::new();
            for img in images.iter() {
                if layer_idx < img.deepstack_features.len() {
                    combined_feats.extend_from_slice(&img.deepstack_features[layer_idx]);
                }
            }
            ds_feature_bufs.push(MetalBuffer::from_f32(device, &combined_feats)?);
        }

        let ds_indices_buf = if !image_token_indices.is_empty() {
            let bytes: Vec<u8> = image_token_indices.iter()
                .flat_map(|i| i.to_le_bytes())
                .collect();
            Some(MetalBuffer::from_data(device, &bytes)?)
        } else {
            None
        };

        // 6. Build deepstack param tuple
        let deepstack = if !ds_feature_bufs.is_empty() && ds_indices_buf.is_some() {
            Some((
                ds_feature_bufs.as_slice(),
                ds_indices_buf.as_ref().unwrap(),
                num_image_tokens,
            ))
        } else {
            None
        };

        // 7. Temporary KV cache (stateless — discarded after forward pass)
        let mut kv = MetalKvCache::new(device, config, total_len, herbert_core::config::KvQuantType::BF16, None)?;

        // 8. Shared transformer forward with prebuilt embeds + MRoPE + DeepStack
        let output = self.transformer_forward(
            &mut kv, tokens, Some(&combined),
            Some((&cos_buf, &sin_buf)),
            deepstack,
        )?;

        // 9. Bind last hidden → RMS norm → L2 normalize
        let eps = config.rms_norm_eps;
        let last_hidden = output.embed.slice(
            (output.seq_len - 1) as usize * hidden_size * 4,
            hidden_size * 4,
        );

        let normed = MetalBuffer::new_uninit(device, (hidden_size * 4) as u64)?;
        let es = EncoderState::new(&self.ctx)?;
        kernels::norm::rms_norm(
            &self.ctx, &es.encoder, &last_hidden, &model.final_norm,
            &normed, hidden_size as u32, eps,
        );
        es.submit_and_wait()?;

        let mut result = normed.read_f32(hidden_size);
        let norm_sq: f32 = result.iter().map(|v| v * v).sum();
        let norm_val = norm_sq.sqrt();
        if norm_val > 0.0 {
            let inv = 1.0 / norm_val;
            for v in &mut result {
                *v *= inv;
            }
        }

        Ok(result)
    }

    // ========================================================================
    // INT8 KV: post-prefill bulk quantization (half → INT8)
    // ========================================================================

    /// Quantize the half-precision KV cache to INT8 for decode.
    ///
    /// Uses the kv_cache_append_batch_i8 shader to read from half cache and
    /// write to INT8 cache with per-position per-head scales.
    fn quantize_kv_half_to_i8(&self, kv: &mut MetalKvCache) -> Result<()> {
        let ctx = &self.ctx;
        let config = &self.config;
        let seq_len = kv.seq_len as u32;
        if seq_len == 0 {
            kv.i8_ready = true;
            return Ok(());
        }

        let kv_dim = config.kv_dim() as u32;
        let head_dim = config.head_dim as u32;
        let num_kv_heads = config.num_key_value_heads as u32;

        let t0 = std::time::Instant::now();
        let cb = ctx.begin_command_buffer()?;
        let enc = MetalContext::new_compute_encoder(&cb)?;

        let ki = kv.keys_i8.as_ref().unwrap();
        let vi = kv.values_i8.as_ref().unwrap();
        let ks = kv.keys_scales.as_ref().unwrap();
        let vs = kv.values_scales.as_ref().unwrap();

        for l in 0..config.num_layers {
            // Quantize half K → INT8 K
            kernels::attention::kv_cache_quantize_half_to_i8(
                ctx, &enc, &ki[l], &kv.keys[l], &ks[l],
                kv_dim, 0, seq_len, head_dim, num_kv_heads,
            );
            // Quantize half V → INT8 V
            kernels::attention::kv_cache_quantize_half_to_i8(
                ctx, &enc, &vi[l], &kv.values[l], &vs[l],
                kv_dim, 0, seq_len, head_dim, num_kv_heads,
            );
        }

        enc.endEncoding();
        MetalContext::submit_and_wait(&cb)?;

        kv.i8_ready = true;
        eprintln!(
            "[metal] INT8 KV quantization: {} layers × {} positions in {:.1}ms",
            config.num_layers, seq_len, t0.elapsed().as_secs_f64() * 1000.0,
        );
        Ok(())
    }

    // ========================================================================
    // Public entry points called from the Backend trait implementation
    // ========================================================================

    pub fn do_prefill(
        &self,
        tokens: &[u32],
        _opts: RunOpts,
    ) -> Result<(herbert_core::kv_cache::KvHandle, PrefillOutput)> {
        if tokens.is_empty() {
            return Err(HerbertError::Backend(
                "prefill: empty token sequence".into(),
            ));
        }

        let mut kv = MetalKvCache::new(&self.ctx.device, &self.config, self.max_tokens, self.kv_quant, self.kv_budget)?;
        let total = tokens.len();

        // Extract thread-local progress callback for async MTLSharedEvent notifications.
        // The callback is taken (not borrowed) so it can be wrapped in Arc for the dispatch queue.
        let progress: Option<Arc<dyn Fn(usize, usize) + Send + Sync>> =
            herbert_backend_common::prefill_progress::take_callback()
                .map(|cb| Arc::from(cb));

        if let Some(ref p) = progress {
            p(0, total);
        }

        let first_token = self.prefill_step(
            &mut kv, tokens, None, None, None, progress.clone(),
        )?;

        if let Some(ref p) = progress {
            p(total, total);
        }
        kv.rope_pos = total;

        // Post-prefill: quantize half KV cache → INT8 for decode
        if kv.kv_quant == herbert_core::config::KvQuantType::INT8 {
            self.quantize_kv_half_to_i8(&mut kv)?;
        }

        let kv_handle = herbert_core::kv_cache::KvHandle::new(kv);
        Ok((
            kv_handle,
            PrefillOutput {
                first_token,
                logits: None,
                timings: None,
                cached_prefix_len: 0,
            },
        ))
    }

    pub fn do_prefill_vl(
        &self,
        tokens: &[u32],
        images: &[VisionEmbedding],
        image_positions: &[usize],
        _opts: RunOpts,
    ) -> Result<(herbert_core::kv_cache::KvHandle, PrefillOutput)> {
        if images.len() != image_positions.len() {
            return Err(HerbertError::Backend(format!(
                "prefill_vl: images ({}) and image_positions ({}) must have same length",
                images.len(), image_positions.len(),
            )));
        }

        let config = &self.config;
        let model = &self.model;
        let device = &*self.ctx.device;
        let hidden_size = config.hidden_size;
        let total_len = tokens.len();

        // 1. Build ImageInfo list
        let image_infos: Vec<ImageInfo> = images
            .iter()
            .zip(image_positions.iter())
            .map(|(img, &pos)| ImageInfo {
                start_idx: pos,
                grid_h: img.grid_h,
                grid_w: img.grid_w,
            })
            .collect();

        // 2. Build combined embeddings on CPU (text lookup + image hidden states)
        //    embed_tokens is f32 in unified memory — read directly.
        let embed_f32 = unsafe {
            let ptr = model.embed_tokens.contents_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, config.vocab_size * hidden_size)
        };

        let mut combined = vec![0.0f32; total_len * hidden_size];
        let mut tok_idx = 0usize;
        let mut img_idx = 0usize;
        while tok_idx < total_len {
            if img_idx < image_infos.len() && tok_idx == image_infos[img_idx].start_idx {
                let n = image_infos[img_idx].num_tokens();
                let dst = &mut combined[tok_idx * hidden_size..(tok_idx + n) * hidden_size];
                dst.copy_from_slice(&images[img_idx].hidden_states[..n * hidden_size]);
                tok_idx += n;
                img_idx += 1;
            } else {
                let t = tokens[tok_idx] as usize;
                let dst = &mut combined[tok_idx * hidden_size..(tok_idx + 1) * hidden_size];
                dst.copy_from_slice(&embed_f32[t * hidden_size..(t + 1) * hidden_size]);
                tok_idx += 1;
            }
        }

        // 3. Build MRoPE cos/sin on CPU
        let half_dim = config.rotary_ndims / 2;
        let positions = position_ids::build_vl_position_ids(total_len, &image_infos);
        let inv_freq = mrope::compute_inv_freq(config.head_dim, config.rope_theta);
        let mrope_section = config.mrope_section.unwrap_or([
            half_dim, 0, 0, // fallback: all T (standard RoPE equivalent)
        ]);
        let pattern = mrope::build_mrope_pattern(mrope_section);

        let mut cos_data = vec![0.0f32; total_len * half_dim];
        let mut sin_data = vec![0.0f32; total_len * half_dim];
        for (i, pos) in positions.iter().enumerate() {
            let offset = i * half_dim;
            mrope::compute_mrope_cos_sin(
                pos.as_array(),
                &inv_freq,
                &pattern,
                &mut cos_data[offset..offset + half_dim],
                &mut sin_data[offset..offset + half_dim],
            );
        }

        // 4. Upload cos/sin to GPU
        let cos_buf = MetalBuffer::from_f32(device, &cos_data)?;
        let sin_buf = MetalBuffer::from_f32(device, &sin_data)?;

        // 5. Build DeepStack GPU buffers
        let image_token_indices: Vec<u32> = image_infos
            .iter()
            .flat_map(|img| (img.start_idx..img.end_idx()).map(|i| i as u32))
            .collect();
        let num_image_tokens = image_token_indices.len() as u32;

        // Concatenate per-layer DeepStack features from all images
        let max_ds_layers = images.iter().map(|img| img.deepstack_features.len()).max().unwrap_or(0);
        let mut ds_feature_bufs: Vec<MetalBuffer> = Vec::with_capacity(max_ds_layers);
        for layer_idx in 0..max_ds_layers {
            let mut combined_feats = Vec::new();
            for img in images.iter() {
                if layer_idx < img.deepstack_features.len() {
                    combined_feats.extend_from_slice(&img.deepstack_features[layer_idx]);
                }
            }
            ds_feature_bufs.push(MetalBuffer::from_f32(device, &combined_feats)?);
        }

        let ds_indices_buf = if !image_token_indices.is_empty() {
            let bytes: Vec<u8> = image_token_indices.iter()
                .flat_map(|i| i.to_le_bytes())
                .collect();
            Some(MetalBuffer::from_data(device, &bytes)?)
        } else {
            None
        };

        // 6. Build deepstack param tuple
        let deepstack = if !ds_feature_bufs.is_empty() && ds_indices_buf.is_some() {
            Some((
                ds_feature_bufs.as_slice(),
                ds_indices_buf.as_ref().unwrap(),
                num_image_tokens,
            ))
        } else {
            None
        };

        // 7. Run prefill with per-layer progress
        let mut kv = MetalKvCache::new(&self.ctx.device, &self.config, self.max_tokens, self.kv_quant, self.kv_budget)?;
        let progress: Option<Arc<dyn Fn(usize, usize) + Send + Sync>> =
            herbert_backend_common::prefill_progress::take_callback()
                .map(|cb| Arc::from(cb));
        if let Some(ref p) = progress {
            p(0, tokens.len());
        }
        let first_token = self.prefill_step(
            &mut kv, tokens, Some(&combined),
            Some((&cos_buf, &sin_buf)),
            deepstack,
            progress.clone(),
        )?;
        if let Some(ref p) = progress {
            p(tokens.len(), tokens.len());
        }

        // 8. Set rope_pos = text_token_count (for decode RoPE)
        let text_token_count = total_len - images.iter().map(|img| img.num_tokens).sum::<usize>();
        kv.rope_pos = text_token_count;

        // Post-prefill: quantize half KV cache → INT8 for decode
        if kv.kv_quant == herbert_core::config::KvQuantType::INT8 {
            self.quantize_kv_half_to_i8(&mut kv)?;
        }

        let kv_handle = herbert_core::kv_cache::KvHandle::new(kv);
        Ok((
            kv_handle,
            PrefillOutput {
                first_token,
                logits: None,
                timings: None,
                cached_prefix_len: 0,
            },
        ))
    }

    pub fn do_decode(
        &self,
        kv_handle: &mut herbert_core::kv_cache::KvHandle,
        token: u32,
        opts: RunOpts,
    ) -> Result<DecodeOutput> {
        let kv = kv_handle.get_mut::<MetalKvCache>()?;
        let next_token = self.decode_step(kv, token)?;

        // Read back full logit vector if requested (speculative decoding draft model)
        let logits = if opts.return_logits {
            Some(kv.decode_logits.read_f32(self.config.vocab_size))
        } else {
            None
        };

        // H2O eviction: if budget is set and sequence exceeds budget + hysteresis
        if let Some(budget) = self.kv_budget {
            let hysteresis = (budget / 8).min(512);
            if kv.seq_len > budget + hysteresis {
                self.h2o_evict(kv, budget)?;
            }
        }

        Ok(DecodeOutput {
            logits,
            token: next_token,
            timings: None,
        })
    }

    /// Verify draft tokens: run prefill-like forward pass and extract logits at ALL positions.
    ///
    /// Used by speculative decoding to verify K draft tokens in one GPU pass.
    pub fn do_verify_draft(
        &self,
        kv_handle: &mut herbert_core::kv_cache::KvHandle,
        draft_tokens: &[u32],
    ) -> Result<herbert_core::backend::VerifyOutput> {
        let kv = kv_handle.get_mut::<MetalKvCache>()?;
        let seq_len = draft_tokens.len() as u32;
        let hidden_size = self.config.hidden_size as u32;
        let vocab_size = self.config.vocab_size;
        let eps = self.config.rms_norm_eps;
        let device = &*self.ctx.device;
        let ctx = &self.ctx;
        let model = &self.model;

        // Phase 1: Transformer forward (reuses prefill path — appends to KV cache)
        let output = self.transformer_forward(kv, draft_tokens, None, None, None)?;

        // Phase 2: Batch rms_norm + lm_head matmul for ALL positions (not just last)
        let normed_buf = MetalBuffer::new_uninit(device, (seq_len * hidden_size * 4) as u64)?;
        let logits_buf = MetalBuffer::new_uninit(device, (seq_len as usize * vocab_size * 4) as u64)?;

        let cb = ctx.begin_command_buffer()?;
        let enc = MetalContext::new_compute_encoder(&cb)?;

        kernels::norm::rms_norm_batch(
            ctx, &enc, &output.embed, &model.final_norm,
            &normed_buf, hidden_size, eps, seq_len,
        );

        Self::matmul_weight_tiled(
            ctx, &enc, &normed_buf, &model.lm_head,
            &logits_buf, seq_len,
        );

        enc.endEncoding();
        MetalContext::submit_and_wait(&cb)?;

        // Phase 3: Read back logits from GPU (unified memory)
        let all_logits = logits_buf.read_f32(seq_len as usize * vocab_size);
        let mut logits_per_position = Vec::with_capacity(seq_len as usize);
        for pos in 0..seq_len as usize {
            let start = pos * vocab_size;
            logits_per_position.push(all_logits[start..start + vocab_size].to_vec());
        }

        // Update KV cache state
        kv.seq_len += draft_tokens.len();
        kv.rope_pos += draft_tokens.len();

        // Re-quantize KV to INT8 for subsequent decode steps
        if kv.kv_quant == herbert_core::config::KvQuantType::INT8 {
            self.quantize_kv_half_to_i8(kv)?;
        }

        Ok(herbert_core::backend::VerifyOutput {
            logits_per_position,
        })
    }

    /// H2O eviction: score → sort → compact the KV cache down to `budget` positions.
    ///
    /// 1. Score probe: run Q.K dot products on late probe layers
    /// 2. CPU sort: read scores, protect sinks + recent window, select top-budget
    /// 3. GPU compact: write index_map, compact all KV layers in-place
    /// 4. Update seq_len (rope_pos stays unchanged — positions are logical)
    fn h2o_evict(&self, kv: &mut MetalKvCache, budget: usize) -> Result<()> {
        let config = &self.config;
        let ctx = &self.ctx;
        let cached_len = kv.seq_len;

        if cached_len <= budget {
            return Ok(());
        }

        let scores_buf = kv.h2o_scores.as_ref().ok_or_else(|| {
            HerbertError::Backend("H2O eviction: scores buffer not allocated".into())
        })?;
        let index_map_buf = kv.h2o_index_map.as_ref().ok_or_else(|| {
            HerbertError::Backend("H2O eviction: index_map buffer not allocated".into())
        })?;

        let num_heads = config.num_attention_heads as u32;
        let num_kv_heads = config.num_key_value_heads as u32;
        let head_dim = config.head_dim as u32;
        let kv_dim = config.kv_dim() as u32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Probe layers: 3 late layers (empirically best for Qwen3 attention patterns)
        let nl = config.num_layers;
        let probe_layers = [
            nl.saturating_sub(8),   // e.g., layer 40 for 48-layer
            nl.saturating_sub(4),   // e.g., layer 44
            nl.saturating_sub(1),   // e.g., layer 47
        ];

        let t0 = std::time::Instant::now();

        // Zero the scores buffer (CPU, unified memory)
        scores_buf.zero_fill();

        // Score probe: GPU dispatch on probe layers
        let cb = ctx.begin_command_buffer()?;
        let encoder = MetalContext::new_compute_encoder(&cb)?;

        for &layer_idx in &probe_layers {
            if layer_idx >= nl { continue; }
            // Use the last Q vector (current decode position) for scoring
            let q = &kv.decode_q[layer_idx];

            if kv.i8_ready {
                let ki = &kv.keys_i8.as_ref().unwrap()[layer_idx];
                let ks = &kv.keys_scales.as_ref().unwrap()[layer_idx];
                kernels::attention::h2o_score_probe_i8(
                    ctx, &encoder, q, ki, ks, scores_buf,
                    num_heads, num_kv_heads, head_dim, kv_dim,
                    cached_len as u32, scale,
                );
            } else {
                kernels::attention::h2o_score_probe_half(
                    ctx, &encoder, q, &kv.keys[layer_idx], scores_buf,
                    num_heads, num_kv_heads, head_dim, kv_dim,
                    cached_len as u32, scale,
                );
            }
        }

        encoder.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();

        // CPU sort: read scores, determine which positions to keep
        let sink_count = 8usize;
        let recent_window = 128usize;

        let scores = scores_buf.read_f32(cached_len);

        // Build (position, score) pairs for the evictable middle region
        let evictable_start = sink_count;
        let evictable_end = cached_len.saturating_sub(recent_window);

        let mut kept_positions: Vec<u32> = Vec::with_capacity(budget);

        // Always keep sink positions
        for i in 0..sink_count.min(cached_len) {
            kept_positions.push(i as u32);
        }

        if evictable_start < evictable_end {
            // Sort middle region by score (descending) and keep top-N
            let middle_budget = budget.saturating_sub(sink_count).saturating_sub(recent_window);
            let mut scored: Vec<(u32, f32)> = (evictable_start..evictable_end)
                .map(|i| (i as u32, scores[i]))
                .collect();
            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(middle_budget);

            // Sort kept middle positions by position (ascending) for in-place compaction safety
            scored.sort_unstable_by_key(|&(pos, _)| pos);
            for (pos, _) in scored {
                kept_positions.push(pos);
            }
        }

        // Always keep recent window
        for i in evictable_end.max(sink_count)..cached_len {
            kept_positions.push(i as u32);
        }

        let num_kept = kept_positions.len();

        // Write index_map to GPU buffer
        unsafe {
            let dst = index_map_buf.contents_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(kept_positions.as_ptr(), dst, num_kept);
        }

        // GPU compact: compact all KV layers
        let cb2 = ctx.begin_command_buffer()?;
        let encoder2 = MetalContext::new_compute_encoder(&cb2)?;

        for l in 0..nl {
            if kv.i8_ready {
                let ki = &kv.keys_i8.as_ref().unwrap()[l];
                let vi = &kv.values_i8.as_ref().unwrap()[l];
                kernels::attention::kv_cache_compact_i8(
                    ctx, &encoder2, ki, index_map_buf, kv_dim, num_kept as u32,
                );
                kernels::attention::kv_cache_compact_i8(
                    ctx, &encoder2, vi, index_map_buf, kv_dim, num_kept as u32,
                );
                let ks = &kv.keys_scales.as_ref().unwrap()[l];
                let vs = &kv.values_scales.as_ref().unwrap()[l];
                kernels::attention::kv_cache_compact_scales(
                    ctx, &encoder2, ks, index_map_buf, num_kv_heads, num_kept as u32,
                );
                kernels::attention::kv_cache_compact_scales(
                    ctx, &encoder2, vs, index_map_buf, num_kv_heads, num_kept as u32,
                );
            } else {
                kernels::attention::kv_cache_compact_half(
                    ctx, &encoder2, &kv.keys[l], index_map_buf, kv_dim, num_kept as u32,
                );
                kernels::attention::kv_cache_compact_half(
                    ctx, &encoder2, &kv.values[l], index_map_buf, kv_dim, num_kept as u32,
                );
            }
        }

        encoder2.endEncoding();
        cb2.commit();
        cb2.waitUntilCompleted();

        let evicted = cached_len - num_kept;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[h2o] Evicted {} positions: {} → {} (budget={}, {:.1}ms)",
            evicted, cached_len, num_kept, budget, elapsed_ms
        );

        // Update sequence length (rope_pos stays unchanged — RoPE positions are absolute)
        kv.seq_len = num_kept;

        Ok(())
    }
}
