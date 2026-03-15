//! KV cache stored in Metal GPU buffers (unified memory).
//!
//! Pre-allocates per-layer key/value caches and decode working buffers.
//! No convolution state buffers (Qwen3 is pure attention).
//!
//! Supports two KV cache formats:
//! - Half (f16): 2 bytes/element, default
//! - INT8: 1 byte/element + f32 scale per position per head (~2× bandwidth reduction)
//!
//! When INT8 is selected, BOTH half and INT8 buffers are allocated:
//! - Half buffers (`keys`/`values`) are used during prefill (existing shaders read half*)
//! - INT8 buffers (`keys_i8`/`values_i8` + scales) are used during decode
//! - After prefill completes, a bulk quantize converts half → INT8
//! - Decode appends directly to INT8 via fused norm+rope+i8 shader

use herbert_core::config::{Config, KvQuantType};
use herbert_core::error::Result;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use crate::memory::MetalBuffer;

/// KV cache and decode working buffers on Metal GPU.
pub struct MetalKvCache {
    // Per-layer KV caches (half-precision, always allocated for prefill)
    pub keys: Vec<MetalBuffer>,
    pub values: Vec<MetalBuffer>,
    // INT8 KV caches (only allocated when kv_quant == INT8, used for decode)
    pub keys_i8: Option<Vec<MetalBuffer>>,
    pub values_i8: Option<Vec<MetalBuffer>>,
    pub keys_scales: Option<Vec<MetalBuffer>>,
    pub values_scales: Option<Vec<MetalBuffer>>,
    pub kv_quant: KvQuantType,
    /// Whether INT8 cache has been populated (set after post-prefill quantization)
    pub i8_ready: bool,
    pub seq_len: usize,
    /// RoPE position counter — equals seq_len for text-only prefill.
    /// After VL prefill, rope_pos = text_token_count (< seq_len) so that
    /// decode RoPE uses the correct text-only position.
    pub rope_pos: usize,
    pub max_tokens: usize,
    pub kv_dim: usize,
    pub num_kv_heads: usize,

    // Per-layer decode working buffers
    pub decode_q: Vec<MetalBuffer>,
    pub decode_k: Vec<MetalBuffer>,
    pub decode_v: Vec<MetalBuffer>,
    pub decode_attn_out: Vec<MetalBuffer>,
    pub decode_norm1: Vec<MetalBuffer>,
    pub decode_norm2: Vec<MetalBuffer>,
    pub decode_mlp_gate: Vec<MetalBuffer>,
    pub decode_mlp_up: Vec<MetalBuffer>,
    pub decode_mlp_out: Vec<MetalBuffer>,

    // Shared working buffers
    pub decode_embed: MetalBuffer,
    pub decode_final_norm: MetalBuffer,
    pub decode_logits: MetalBuffer,
    pub argmax_result: MetalBuffer,
    pub argmax_intermediate: MetalBuffer,  // N_TG × 2 uint for two-stage argmax
    pub token_buf: MetalBuffer,

    // FlashDecoding temporary buffer for tile partials
    // Layout: [max_tiles, num_heads, 2 + head_dim] float
    pub flash_decode_partials: MetalBuffer,

    // H2O eviction buffers (allocated when kv_budget is set)
    pub h2o_scores: Option<MetalBuffer>,     // [max_tokens] f32 — cumulative attention scores
    pub h2o_index_map: Option<MetalBuffer>,  // [max_tokens] u32 — kept position indices

    // MoE working buffers
    pub moe_router_logits: Option<MetalBuffer>,
    pub moe_output: Option<MetalBuffer>,
    // GPU-side MoE routing output buffers
    pub moe_expert_ids: Option<MetalBuffer>,
    pub moe_expert_weights: Option<MetalBuffer>,
    // Batched MoE working buffers (for sync-free decode)
    pub moe_inter_batched: Option<MetalBuffer>,  // [top_k * moe_inter] f32
    pub moe_down_output: Option<MetalBuffer>,    // [top_k * hidden] f32
}

// SAFETY: MetalKvCache holds Metal buffer handles that are not accessed
// concurrently. The backend ensures single-threaded GPU access via sequential
// command buffer submission.
unsafe impl Send for MetalKvCache {}
unsafe impl Sync for MetalKvCache {}

impl MetalKvCache {
    /// Allocate KV cache and all decode working buffers for Qwen3.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        config: &Config,
        max_tokens: usize,
        kv_quant: KvQuantType,
        kv_budget: Option<usize>,
    ) -> Result<Self> {
        let num_layers = config.num_layers;
        let hidden_size = config.hidden_size;
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let num_kv_heads = config.num_key_value_heads;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let moe_inter = config.moe_intermediate_size.unwrap_or(0);

        // Metal KV cache supports half (f16) or int8. Map F32/BF16 → half.
        let effective_quant = match kv_quant {
            KvQuantType::INT8 => KvQuantType::INT8,
            _ => KvQuantType::BF16,
        };
        let use_i8 = effective_quant == KvQuantType::INT8;

        // Always allocate half-precision KV cache (used by prefill attention shaders)
        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);
        let mut keys_i8 = if use_i8 { Some(Vec::with_capacity(num_layers)) } else { None };
        let mut values_i8 = if use_i8 { Some(Vec::with_capacity(num_layers)) } else { None };
        let mut keys_scales = if use_i8 { Some(Vec::with_capacity(num_layers)) } else { None };
        let mut values_scales = if use_i8 { Some(Vec::with_capacity(num_layers)) } else { None };

        let mut decode_q = Vec::with_capacity(num_layers);
        let mut decode_k = Vec::with_capacity(num_layers);
        let mut decode_v = Vec::with_capacity(num_layers);
        let mut decode_attn_out = Vec::with_capacity(num_layers);
        let mut decode_norm1 = Vec::with_capacity(num_layers);
        let mut decode_norm2 = Vec::with_capacity(num_layers);
        let mut decode_mlp_gate = Vec::with_capacity(num_layers);
        let mut decode_mlp_up = Vec::with_capacity(num_layers);
        let mut decode_mlp_out = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            // Half-precision KV cache: 2 bytes per element (always allocated)
            keys.push(MetalBuffer::new_uninit(device, (max_tokens * kv_dim * 2) as u64)?);
            values.push(MetalBuffer::new_uninit(device, (max_tokens * kv_dim * 2) as u64)?);

            // INT8 KV cache: 1 byte per element + f32 scale per position per head
            if let Some(ref mut ki) = keys_i8 {
                ki.push(MetalBuffer::new_uninit(device, (max_tokens * kv_dim) as u64)?);
            }
            if let Some(ref mut vi) = values_i8 {
                vi.push(MetalBuffer::new_uninit(device, (max_tokens * kv_dim) as u64)?);
            }
            if let Some(ref mut ks) = keys_scales {
                ks.push(MetalBuffer::new_uninit(device, (max_tokens * num_kv_heads * 4) as u64)?);
            }
            if let Some(ref mut vs) = values_scales {
                vs.push(MetalBuffer::new_uninit(device, (max_tokens * num_kv_heads * 4) as u64)?);
            }

            decode_q.push(MetalBuffer::new_uninit(device, (q_dim * 4) as u64)?);
            decode_k.push(MetalBuffer::new_uninit(device, (kv_dim * 4) as u64)?);
            decode_v.push(MetalBuffer::new_uninit(device, (kv_dim * 4) as u64)?);
            decode_attn_out.push(MetalBuffer::new_uninit(device, (q_dim * 4) as u64)?);
            decode_norm1.push(MetalBuffer::new_uninit(device, (hidden_size * 4) as u64)?);
            decode_norm2.push(MetalBuffer::new_uninit(device, (hidden_size * 4) as u64)?);

            let mlp_inter = if config.is_moe_layer(i) { moe_inter } else { intermediate_size };
            decode_mlp_gate.push(MetalBuffer::new_uninit(device, (mlp_inter * 4) as u64)?);
            decode_mlp_up.push(MetalBuffer::new_uninit(device, (mlp_inter * 4) as u64)?);
            decode_mlp_out.push(MetalBuffer::new_uninit(device, (hidden_size * 4) as u64)?);
        }

        let decode_embed = MetalBuffer::new_uninit(device, (hidden_size * 4) as u64)?;
        let decode_final_norm = MetalBuffer::new_uninit(device, (hidden_size * 4) as u64)?;
        let decode_logits = MetalBuffer::new_uninit(device, (vocab_size * 4) as u64)?;
        let argmax_result = MetalBuffer::new_uninit(device, 8)?;
        let argmax_intermediate = MetalBuffer::new_uninit(device, 256 * 8)?;
        let token_buf = MetalBuffer::new_uninit(device, 4)?;

        // FlashDecoding partials buffer: [max_tiles][num_heads][2 + head_dim] float
        let num_heads = config.num_attention_heads;
        let head_dim = kv_dim / num_kv_heads;
        let flash_tile_size = 256usize;
        let max_tiles = (max_tokens + flash_tile_size - 1) / flash_tile_size;
        let partials_size = max_tiles * num_heads * (2 + head_dim) * 4;
        let flash_decode_partials = MetalBuffer::new_uninit(device, partials_size as u64)?;

        let num_experts_per_tok = config.num_experts_per_tok.unwrap_or(0);
        let (moe_router_logits, moe_output,
             moe_expert_ids, moe_expert_weights,
             moe_inter_batched, moe_down_output) =
            if config.is_moe() {
                let ne = config.num_experts.unwrap();
                let k = num_experts_per_tok;
                (
                    Some(MetalBuffer::new_uninit(device, (ne * 4) as u64)?),
                    Some(MetalBuffer::new_uninit(device, (hidden_size * 4) as u64)?),
                    Some(MetalBuffer::new_uninit(device, (k * 4) as u64)?),
                    Some(MetalBuffer::new_uninit(device, (k * 4) as u64)?),
                    Some(MetalBuffer::new_uninit(device, (k * moe_inter * 4) as u64)?),
                    Some(MetalBuffer::new_uninit(device, (k * hidden_size * 4) as u64)?),
                )
            } else {
                (None, None, None, None, None, None)
            };

        let (h2o_scores, h2o_index_map) = if kv_budget.is_some() {
            (
                Some(MetalBuffer::new_uninit(device, (max_tokens * 4) as u64)?),
                Some(MetalBuffer::new_uninit(device, (max_tokens * 4) as u64)?),
            )
        } else {
            (None, None)
        };

        {
            let half_bytes = num_layers * max_tokens * kv_dim * 2 * 2; // K+V, 2 bytes each
            if use_i8 {
                let i8_bytes = num_layers * max_tokens * kv_dim * 2;
                let scale_bytes = num_layers * max_tokens * num_kv_heads * 4 * 2;
                let total = half_bytes + i8_bytes + scale_bytes;
                eprintln!(
                    "[metal] KV alloc: max_tokens={}, half={:.1}MB + i8={:.1}MB + scales={:.1}MB = {:.1}GB total",
                    max_tokens,
                    half_bytes as f64 / 1e6,
                    i8_bytes as f64 / 1e6,
                    scale_bytes as f64 / 1e6,
                    total as f64 / 1e9,
                );
            } else {
                eprintln!(
                    "[metal] KV alloc: max_tokens={}, half={:.1}MB ({:.1}GB)",
                    max_tokens,
                    half_bytes as f64 / 1e6,
                    half_bytes as f64 / 1e9,
                );
            }
        }

        Ok(Self {
            keys, values,
            keys_i8, values_i8, keys_scales, values_scales,
            kv_quant: effective_quant,
            i8_ready: false,
            seq_len: 0, rope_pos: 0, max_tokens, kv_dim, num_kv_heads,
            decode_q, decode_k, decode_v, decode_attn_out,
            decode_norm1, decode_norm2,
            decode_mlp_gate, decode_mlp_up, decode_mlp_out,
            decode_embed, decode_final_norm, decode_logits,
            argmax_result, argmax_intermediate, token_buf,
            flash_decode_partials,
            h2o_scores, h2o_index_map,
            moe_router_logits, moe_output,
            moe_expert_ids, moe_expert_weights,
            moe_inter_batched, moe_down_output,
        })
    }
}
