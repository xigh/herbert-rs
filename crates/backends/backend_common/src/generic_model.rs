//! Generic model implementation parameterized over LinearOps.

use std::marker::PhantomData;

use herbert_core::config::{Config, NormType};
use herbert_core::error::{HerbertError, Result};
use herbert_core::tensor::BF16;

use crate::generic_layer::GenericDecoderLayer;
use crate::kernels as common_kernels;
use crate::kv_cache::CpuKvCache;
use crate::linear_ops::LinearOps;
use crate::mrope;
use crate::position_ids::Position3D;
use crate::profiler::LayerProfiler;
use indicatif::{ProgressBar, ProgressStyle};

/// Info needed to inject DeepStack features into the LLM forward pass.
pub struct DeepStackInfo<'a> {
    /// Indices of image tokens in the sequence (sorted, ascending).
    pub image_token_indices: &'a [usize],
    /// One feature slice per LLM layer to inject: `[num_image_tokens * hidden_size]`.
    pub layer_features: &'a [Vec<f32>],
}

/// Complete model generic over the linear ops / weight type.
pub struct GenericModel<L: LinearOps> {
    pub embed_tokens: Vec<BF16>,
    pub decoder_layers: Vec<GenericDecoderLayer<L>>,
    pub norm: Vec<BF16>,
    pub norm_bias: Option<Vec<f32>>,
    pub lm_head: L::Weight,
    pub lm_head_bias: Option<Vec<f32>>,
    pub config: Config,
    pub cos_cache: Vec<f32>,
    pub sin_cache: Vec<f32>,
    /// Gemma3 VL: sliding attention layers' RoPE cache (theta=10K, standard).
    /// Empty for non-Gemma3 models (zero-cost: always takes main path).
    pub sliding_cos_cache: Vec<f32>,
    pub sliding_sin_cache: Vec<f32>,
    /// Precomputed inverse frequencies for MRoPE (empty for text-only models).
    pub inv_freq: Vec<f32>,
    /// Interleaved MRoPE pattern (empty for text-only models).
    pub mrope_pattern: Vec<u8>,
    pub _marker: PhantomData<L>,
}

impl<L: LinearOps> GenericModel<L> {
    /// Whether this model uses MRoPE (VL model).
    #[inline]
    pub fn is_mrope(&self) -> bool {
        !self.inv_freq.is_empty()
    }

    /// Whether this model has dual RoPE caches (Gemma3 VL).
    #[inline]
    fn has_dual_rope(&self) -> bool {
        !self.sliding_cos_cache.is_empty()
    }

    /// Whether a layer uses sliding attention RoPE (Gemma3 VL: 5 out of every 6 layers).
    /// Full attention layers are every 6th layer (layer_idx+1 divisible by 6).
    #[inline]
    fn is_sliding_layer(&self, layer_idx: usize) -> bool {
        self.has_dual_rope() && !(layer_idx + 1).is_multiple_of(6)
    }

    /// Select the appropriate cos/sin cache for a layer (dual RoPE dispatch).
    #[inline]
    pub fn rope_cache_for_layer(&self, layer_idx: usize) -> (&[f32], &[f32]) {
        if self.is_sliding_layer(layer_idx) {
            (&self.sliding_cos_cache, &self.sliding_sin_cache)
        } else {
            (&self.cos_cache, &self.sin_cache)
        }
    }

    /// Apply final normalization (RMSNorm or GemmaRMSNorm depending on config).
    #[inline]
    pub fn apply_final_norm(&self, input: &[f32], output: &mut [f32]) {
        match self.config.norm_type {
            NormType::GemmaRMSNorm => {
                common_kernels::rms_norm_gemma_bf16(input, &self.norm, output, self.config.rms_norm_eps);
            }
            NormType::RMSNorm => {
                L::rms_norm(input, &self.norm, output, self.config.rms_norm_eps);
            }
        }
    }

    /// No-op: embedding scaling was only needed by Gemma3.
    #[inline]
    fn scale_embeddings(&self, _embeddings: &mut [f32]) {
    }

    /// Get the embed_tokens table as a BF16 slice.
    #[inline]
    pub fn embed_tokens(&self) -> &[BF16] {
        &self.embed_tokens
    }

    /// Copy the BF16 embedding for `token_idx` into `dst`, converting to f32.
    #[inline]
    pub fn embed_lookup(&self, token_idx: usize, dst: &mut [f32]) {
        let h = self.config.hidden_size;
        let start = token_idx * h;
        let src = &self.embed_tokens[start..start + h];
        for i in 0..h {
            dst[i] = herbert_core::tensor::bf16_to_f32(src[i]);
        }
    }

    /// Apply lm_head bias if present.
    #[inline]
    pub fn apply_lm_head_bias(&self, logits: &mut [f32]) {
        if let Some(ref bias) = self.lm_head_bias {
            for i in 0..logits.len().min(bias.len()) {
                logits[i] += bias[i];
            }
        }
    }

    pub fn create_kv_cache(&self, reserve_tokens: Option<usize>) -> CpuKvCache {
        CpuKvCache::new(&self.config, reserve_tokens)
    }

    pub fn create_kv_cache_with_quant(&self, reserve_tokens: Option<usize>, kv_quant: herbert_core::config::KvQuantType) -> CpuKvCache {
        CpuKvCache::new_with_quant(&self.config, reserve_tokens, kv_quant)
    }

    pub fn prefill_with_token(
        &self,
        input_tokens: &[u32],
        kv_cache: &mut CpuKvCache,
        start_pos: usize,
        return_logits: bool,
        show_progress: bool,
    ) -> Result<(Option<Vec<f32>>, u32, Vec<f64>)> {
        let seq_len = input_tokens.len();
        if seq_len == 0 {
            return Err(HerbertError::Backend(
                "prefill requires at least one input token".to_string(),
            ));
        }
        let end_pos = start_pos + seq_len;
        let max_pos = self.config.max_position_embeddings;
        if end_pos > max_pos {
            return Err(HerbertError::Backend(format!(
                "sequence length {} (start_pos={} + {} tokens) exceeds context window (max_position_embeddings={})",
                end_pos, start_pos, seq_len, max_pos
            )));
        }
        let hidden_size = self.config.hidden_size;
        let n = seq_len * hidden_size;

        // Ping-pong buffers — zero allocation after first prefill
        let mut buf_a = std::mem::take(&mut kv_cache.ctx.prefill_buf_a);
        buf_a.resize(n, 0.0);
        let mut buf_b = std::mem::take(&mut kv_cache.ctx.prefill_buf_b);
        buf_b.resize(n, 0.0);

        // Get embeddings into buf_a
        for (i, &token_id) in input_tokens.iter().enumerate() {
            let token_idx = token_id as usize;
            if token_idx >= self.config.vocab_size {
                kv_cache.ctx.prefill_buf_a = buf_a;
                kv_cache.ctx.prefill_buf_b = buf_b;
                return Err(HerbertError::Backend(format!(
                    "token_id {} out of range (vocab_size={})",
                    token_idx, self.config.vocab_size
                )));
            }
            self.embed_lookup(token_idx, &mut buf_a[i * hidden_size..(i + 1) * hidden_size]);
        }
        self.scale_embeddings(&mut buf_a[..n]);

        // Forward through layers (ping-pong: even layers read A→write B, odd read B→write A)

        let num_layers = self.decoder_layers.len();
        let pb = if show_progress {
            tracing::info!(seq_len, "Prefill");
            let pb = ProgressBar::new(num_layers as u64);
            pb.set_style(
                ProgressStyle::with_template("  Prefill [{bar:30}] {pos}/{len} layers")
                    .expect("valid progress template")
                    .progress_chars("█░░"),
            );
            Some(pb)
        } else {
            None
        };
        let prefill_start = std::time::Instant::now();
        #[cfg(feature = "profile-layer")]
        crate::profiler::reset(seq_len, num_layers);
        #[cfg(feature = "profile-prefill-layer")]
        crate::profiler::reset_prefill_layer_detail();
        #[cfg(feature = "profile-attn-kernel")]
        crate::profiler::reset_attn_kernel_breakdown();
        let dump_hidden = std::env::var("HERBERT_DUMP_HIDDEN").is_ok();
        let dump_hidden_max = std::env::var("HERBERT_DUMP_HIDDEN_MAX")
            .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(9);
        let mut profiler = LayerProfiler::new(num_layers);
        for (layer_idx, layer) in self.decoder_layers.iter().enumerate() {
            let (layer_cos, layer_sin) = self.rope_cache_for_layer(layer_idx);
            if layer_idx % 2 == 0 {
                profiler.measure(|| {
                    layer.forward_prefill(
                        &buf_a, layer_cos, layer_sin,
                        kv_cache, layer_idx, seq_len, start_pos, &mut buf_b,
                    )
                })?;
            } else {
                profiler.measure(|| {
                    layer.forward_prefill(
                        &buf_b, layer_cos, layer_sin,
                        kv_cache, layer_idx, seq_len, start_pos, &mut buf_a,
                    )
                })?;
            }

            // Debug: dump last-token hidden state after each layer (set HERBERT_DUMP_HIDDEN=1)
            if dump_hidden && layer_idx <= dump_hidden_max {
                let output = if layer_idx % 2 == 0 { &buf_b } else { &buf_a };
                let last_hidden = &output[(seq_len - 1) * hidden_size..seq_len * hidden_size];
                // Compute stats: norm, first 8 values
                let norm: f64 = last_hidden.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                let first8: Vec<f32> = last_hidden.iter().take(8).copied().collect();
                let abs_max = last_hidden.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let mean: f64 = last_hidden.iter().map(|&v| v as f64).sum::<f64>() / hidden_size as f64;
                eprintln!("[HIDDEN_DUMP] layer {:2} seq_len={} norm={:.6} abs_max={:.6} mean={:.8} first8={:?}",
                    layer_idx, seq_len, norm, abs_max, mean, first8);
            }

            #[cfg(feature = "profile-prefill-layer")]
            if layer_idx == 1 {
                crate::profiler::report_prefill_layer_detail(seq_len);
                #[cfg(feature = "profile-attn-kernel")]
                crate::profiler::report_attn_kernel_breakdown();
                std::process::exit(0);
            }
            if let Some(ref pb) = pb {
                pb.set_position((layer_idx + 1) as u64);
            }
            if show_progress {
                let elapsed = prefill_start.elapsed().as_secs_f64();
                tracing::info!(
                    layer = layer_idx + 1,
                    total = num_layers,
                    pct = format_args!("{:.1}", (layer_idx + 1) as f64 / num_layers as f64 * 100.0),
                    elapsed_s = format_args!("{:.2}", elapsed),
                    "prefill progress",
                );
            }
        }
        if let Some(pb) = pb {
            pb.finish_and_clear();
        }
        #[cfg(feature = "profile-layer")]
        crate::profiler::report();
        #[cfg(feature = "profile-attn-kernel")]
        crate::profiler::report_attn_kernel_breakdown();
        let layer_times = profiler.finish();
        kv_cache.advance_seq_len(seq_len);

        // Result is in buf_a if num_layers is even, buf_b if odd
        let result = if num_layers.is_multiple_of(2) { &buf_a } else { &buf_b };

        // Final norm with BF16 weights
        let last_hidden = &result[(seq_len - 1) * hidden_size..seq_len * hidden_size];
        let mut final_norm = std::mem::take(&mut kv_cache.ctx.decode_final_norm);
        if final_norm.len() != hidden_size {
            final_norm.resize(hidden_size, 0.0);
        }
        self.apply_final_norm(last_hidden, &mut final_norm);

        // LM head
        let vocab_size = self.config.vocab_size;
        let mut logits = std::mem::take(&mut kv_cache.ctx.decode_logits);
        if logits.len() != vocab_size {
            logits.resize(vocab_size, 0.0);
        }
        L::matvec(&final_norm, &self.lm_head, &mut logits)?;
        self.apply_lm_head_bias(&mut logits);

        // Debug: dump top-20 logits after prefill (set HERBERT_DUMP_LOGITS=1)
        if std::env::var("HERBERT_DUMP_LOGITS").is_ok() {
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[LOGIT_DUMP] prefill seq_len={} start_pos={}", seq_len, start_pos);
            eprintln!("[LOGIT_DUMP] top-20 logits after prefill:");
            for (rank, (idx, val)) in indexed.iter().take(20).enumerate() {
                eprintln!("[LOGIT_DUMP]   #{:2}: token_id={:6} logit={:12.6}", rank, idx, val);
            }
            // Also dump logits stats
            let sum: f64 = logits.iter().map(|&v| v as f64).sum();
            let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min_val = logits.iter().copied().fold(f32::INFINITY, f32::min);
            eprintln!("[LOGIT_DUMP] stats: min={:.6} max={:.6} mean={:.6} vocab_size={}", min_val, max_val, sum / logits.len() as f64, logits.len());
        }

        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| HerbertError::Backend("empty logits in prefill argmax".into()))?;
        let logits_out = if return_logits {
            Some(logits.clone())
        } else {
            None
        };

        kv_cache.ctx.prefill_buf_a = buf_a;
        kv_cache.ctx.prefill_buf_b = buf_b;
        kv_cache.ctx.decode_final_norm = final_norm;
        kv_cache.ctx.decode_logits = logits;

        Ok((logits_out, next_token, layer_times))
    }

    /// Prefill with per-layer callbacks for pipeline overlap.
    ///
    /// `before_layer(layer_idx, kv_cache)` is called before each layer (mutable: can prepend KV).
    /// `after_layer(layer_idx, kv_cache)` is called after each layer (immutable: can read KV).
    ///
    /// This is a stripped-down version of `prefill_with_token()` without progress bar,
    /// debug dump, or profiler features — intended for distributed prefill pipeline.
    pub fn prefill_with_token_layer_hooks<BH, AH>(
        &self,
        input_tokens: &[u32],
        kv_cache: &mut CpuKvCache,
        start_pos: usize,
        return_logits: bool,
        before_layer: &mut BH,
        after_layer: &mut AH,
    ) -> Result<(Option<Vec<f32>>, u32, Vec<f64>)>
    where
        BH: FnMut(usize, &mut CpuKvCache) -> Result<()>,
        AH: FnMut(usize, &mut CpuKvCache) -> Result<()>,
    {
        let seq_len = input_tokens.len();
        if seq_len == 0 {
            return Err(HerbertError::Backend(
                "prefill requires at least one input token".to_string(),
            ));
        }
        let end_pos = start_pos + seq_len;
        let max_pos = self.config.max_position_embeddings;
        if end_pos > max_pos {
            return Err(HerbertError::Backend(format!(
                "sequence length {} (start_pos={} + {} tokens) exceeds context window (max_position_embeddings={})",
                end_pos, start_pos, seq_len, max_pos
            )));
        }
        let hidden_size = self.config.hidden_size;
        let n = seq_len * hidden_size;

        // Ping-pong buffers
        let mut buf_a = std::mem::take(&mut kv_cache.ctx.prefill_buf_a);
        buf_a.resize(n, 0.0);
        let mut buf_b = std::mem::take(&mut kv_cache.ctx.prefill_buf_b);
        buf_b.resize(n, 0.0);

        // Get embeddings into buf_a
        for (i, &token_id) in input_tokens.iter().enumerate() {
            let token_idx = token_id as usize;
            if token_idx >= self.config.vocab_size {
                kv_cache.ctx.prefill_buf_a = buf_a;
                kv_cache.ctx.prefill_buf_b = buf_b;
                return Err(HerbertError::Backend(format!(
                    "token_id {} out of range (vocab_size={})",
                    token_idx, self.config.vocab_size
                )));
            }
            self.embed_lookup(token_idx, &mut buf_a[i * hidden_size..(i + 1) * hidden_size]);
        }
        self.scale_embeddings(&mut buf_a[..n]);

        // Forward through layers with hooks
        let num_layers = self.decoder_layers.len();
        let mut profiler = LayerProfiler::new(num_layers);
        for (layer_idx, layer) in self.decoder_layers.iter().enumerate() {
            before_layer(layer_idx, kv_cache)?;
            let (layer_cos, layer_sin) = self.rope_cache_for_layer(layer_idx);
            if layer_idx % 2 == 0 {
                profiler.measure(|| {
                    layer.forward_prefill(
                        &buf_a, layer_cos, layer_sin,
                        kv_cache, layer_idx, seq_len, start_pos, &mut buf_b,
                    )
                })?;
            } else {
                profiler.measure(|| {
                    layer.forward_prefill(
                        &buf_b, layer_cos, layer_sin,
                        kv_cache, layer_idx, seq_len, start_pos, &mut buf_a,
                    )
                })?;
            }
            after_layer(layer_idx, kv_cache)?;
        }
        let layer_times = profiler.finish();
        kv_cache.advance_seq_len(seq_len);

        // Result is in buf_a if num_layers is even, buf_b if odd
        let result = if num_layers.is_multiple_of(2) { &buf_a } else { &buf_b };

        // Final norm
        let last_hidden = &result[(seq_len - 1) * hidden_size..seq_len * hidden_size];
        let mut final_norm = std::mem::take(&mut kv_cache.ctx.decode_final_norm);
        if final_norm.len() != hidden_size {
            final_norm.resize(hidden_size, 0.0);
        }
        self.apply_final_norm(last_hidden, &mut final_norm);

        // LM head
        let vocab_size = self.config.vocab_size;
        let mut logits = std::mem::take(&mut kv_cache.ctx.decode_logits);
        if logits.len() != vocab_size {
            logits.resize(vocab_size, 0.0);
        }
        L::matvec(&final_norm, &self.lm_head, &mut logits)?;
        self.apply_lm_head_bias(&mut logits);

        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| HerbertError::Backend("empty logits in prefill argmax".into()))?;
        let logits_out = if return_logits {
            Some(logits.clone())
        } else {
            None
        };

        kv_cache.ctx.prefill_buf_a = buf_a;
        kv_cache.ctx.prefill_buf_b = buf_b;
        kv_cache.ctx.decode_final_norm = final_norm;
        kv_cache.ctx.decode_logits = logits;

        Ok((logits_out, next_token, layer_times))
    }

    /// Prefill with chunking: split long prompts into chunks of `chunk_size` tokens.
    ///
    /// Each chunk calls `prefill_with_token()` with an incremental `start_pos`.
    /// The existing code already supports non-zero `start_pos` (RoPE, causal mask, KV append).
    ///
    /// If `chunk_size` is 0 or >= the input length, falls back to a single prefill call.
    pub fn prefill_with_token_chunked(
        &self,
        input_tokens: &[u32],
        kv_cache: &mut CpuKvCache,
        chunk_size: usize,
        return_logits: bool,
        show_progress: bool,
    ) -> Result<(Option<Vec<f32>>, u32, Vec<f64>)> {
        let total_tokens = input_tokens.len();
        let end_pos = kv_cache.kv.seq_len + total_tokens;
        let max_pos = self.config.max_position_embeddings;
        if end_pos > max_pos {
            return Err(HerbertError::Backend(format!(
                "sequence length {} (kv_len={} + {} new tokens) exceeds context window (max_position_embeddings={})",
                end_pos, kv_cache.kv.seq_len, total_tokens, max_pos
            )));
        }

        // Fall back to single call if chunking is disabled or unnecessary
        if chunk_size == 0 || chunk_size >= total_tokens {
            return self.prefill_with_token(
                input_tokens,
                kv_cache,
                0,
                return_logits,
                show_progress,
            );
        }

        let mut start = 0;
        let mut all_layer_times: Vec<f64> = Vec::new();
        let mut last_logits = None;
        let mut last_token = 0u32;

        while start < total_tokens {
            let end = (start + chunk_size).min(total_tokens);
            let chunk = &input_tokens[start..end];
            let is_last = end == total_tokens;

            let (logits, token, layer_times) = self.prefill_with_token(
                chunk,
                kv_cache,
                start,
                is_last && return_logits,
                // Only show progress bar for the last chunk to avoid clutter
                show_progress && is_last,
            )?;

            // Report progress after each chunk
            crate::prefill_progress::report(end, total_tokens);

            // Accumulate layer times
            if all_layer_times.is_empty() {
                all_layer_times = layer_times;
            } else {
                for (i, &t) in layer_times.iter().enumerate() {
                    if i < all_layer_times.len() {
                        all_layer_times[i] += t;
                    }
                }
            }

            if is_last {
                last_logits = logits;
                last_token = token;
            }

            start = end;
        }

        Ok((last_logits, last_token, all_layer_times))
    }

    pub fn decode_step(
        &self,
        token: u32,
        kv_cache: &mut CpuKvCache,
        return_logits: bool,
    ) -> Result<(Option<Vec<f32>>, u32, Vec<f64>)> {
        let pos = kv_cache.kv.seq_len;
        let max_pos = self.config.max_position_embeddings;
        if pos >= max_pos {
            return Err(HerbertError::Backend(format!(
                "sequence position {} exceeds context window (max_position_embeddings={})",
                pos, max_pos
            )));
        }
        let hidden_size = self.config.hidden_size;

        // Get embedding (F32)
        #[cfg(feature = "profile-decode-layer")]
        let t_embed = std::time::Instant::now();
        let token_idx = token as usize;
        let mut x = std::mem::take(&mut kv_cache.ctx.decode_embed);
        if x.len() != hidden_size {
            x.resize(hidden_size, 0.0);
        }
        x.fill(0.0);
        if token_idx >= self.config.vocab_size {
            return Err(HerbertError::Backend(format!(
                "token_id {} out of range (vocab_size={})",
                token_idx, self.config.vocab_size
            )));
        }
        self.embed_lookup(token_idx, &mut x);
        self.scale_embeddings(&mut x);
        #[cfg(feature = "profile-decode-layer")]
        let embed_us = t_embed.elapsed().as_micros() as u64;

        // Snapshot: trigger check and dump embed output
        #[cfg(feature = "bench-decode-snapshot")]
        let snapshot_active = crate::snapshot::begin_decode_step();
        #[cfg(feature = "bench-decode-snapshot")]
        if snapshot_active {
            crate::snapshot::dump_f32("embed_output", &x);
            crate::snapshot::dump_u32("token_id", token);
            crate::snapshot::dump_usize("pos", pos);
        }

        // Get RoPE
        let half_dim = self.config.rotary_ndims / 2;
        let rope_start = pos
            .checked_mul(half_dim)
            .ok_or_else(|| HerbertError::Backend("RoPE index overflow".to_string()))?;
        let rope_end = rope_start
            .checked_add(half_dim)
            .ok_or_else(|| HerbertError::Backend("RoPE index overflow".to_string()))?;
        if rope_end > self.cos_cache.len() || rope_end > self.sin_cache.len() {
            return Err(HerbertError::Backend(format!(
                "sequence position {} exceeds RoPE cache (max positions: {})",
                pos, self.config.max_position_embeddings
            )));
        }

        // Forward through layers

        // Snapshot: dump RoPE cache slice
        #[cfg(feature = "bench-decode-snapshot")]
        if snapshot_active {
            let cos_slice = &self.cos_cache[rope_start..rope_end];
            let sin_slice = &self.sin_cache[rope_start..rope_end];
            crate::snapshot::dump_f32("rope_cos", cos_slice);
            crate::snapshot::dump_f32("rope_sin", sin_slice);
        }

        #[cfg(feature = "profile-decode-layer")]
        crate::profiler::reset_decode_layer_detail();
        let mut profiler = LayerProfiler::new(self.decoder_layers.len());
        for (layer_idx, layer) in self.decoder_layers.iter().enumerate() {
            let (cos_cache, sin_cache) = self.rope_cache_for_layer(layer_idx);
            let cos = &cos_cache[rope_start..rope_end];
            let sin = &sin_cache[rope_start..rope_end];

            // Snapshot: dump layer input before forward
            #[cfg(feature = "bench-decode-snapshot")]
            if snapshot_active && crate::snapshot::should_snapshot_layer(layer_idx) {
                layer.dump_snapshot(layer_idx, &x, kv_cache);
            }

            profiler
                .measure(|| layer.forward_decode(&mut x, cos, sin, kv_cache, layer_idx, pos))?;

            // EAGLE-3: capture hidden states at extraction layers
            if kv_cache.ctx.eagle3_active {
                for slot in 0..3 {
                    if layer_idx == kv_cache.ctx.eagle3_extract_layers[slot] {
                        let hs = &mut kv_cache.ctx.eagle3_hidden_states[slot];
                        hs.resize(x.len(), 0.0);
                        hs.copy_from_slice(&x);
                    }
                }
            }

            // Snapshot: dump layer output after forward
            #[cfg(feature = "bench-decode-snapshot")]
            if snapshot_active && crate::snapshot::should_snapshot_layer(layer_idx) {
                crate::snapshot::dump_f32(
                    &format!("layer{}_output", layer_idx),
                    &x,
                );
            }

        }
        let layer_times = profiler.finish();
        kv_cache.advance_seq_len(1);


        // Final norm
        #[cfg(feature = "profile-decode-layer")]
        let t_fnorm = std::time::Instant::now();
        let mut final_norm = std::mem::take(&mut kv_cache.ctx.decode_final_norm);
        if final_norm.len() != hidden_size {
            final_norm.resize(hidden_size, 0.0);
        }

        // Snapshot: dump final_norm input/weight
        #[cfg(feature = "bench-decode-snapshot")]
        if snapshot_active {
            crate::snapshot::dump_f32("final_norm_input", &x);
            crate::snapshot::dump_bf16("final_norm_weight", &self.norm);
        }

        self.apply_final_norm(&x, &mut final_norm);
        #[cfg(feature = "profile-decode-layer")]
        let fnorm_us = t_fnorm.elapsed().as_micros() as u64;

        // LM head
        #[cfg(feature = "profile-decode-layer")]
        let t_lmhead = std::time::Instant::now();
        let vocab_size = self.config.vocab_size;
        let mut logits = std::mem::take(&mut kv_cache.ctx.decode_logits);
        if logits.len() != vocab_size {
            logits.resize(vocab_size, 0.0);
        }

        // Snapshot: dump lm_head weight (optional, ~300MB)
        #[cfg(feature = "bench-decode-snapshot")]
        if snapshot_active {
            if std::env::var("BENCH_SNAPSHOT_LM_HEAD").is_ok() {
                crate::snapshot::dump_weight::<L>("lm_head", &self.lm_head);
            } else {
                eprintln!("[SNAPSHOT]   lm_head: SKIPPED (set BENCH_SNAPSHOT_LM_HEAD=1 to include)");
            }
        }

        L::matvec(&final_norm, &self.lm_head, &mut logits)?;
        self.apply_lm_head_bias(&mut logits);
        #[cfg(feature = "profile-decode-layer")]
        let lmhead_us = t_lmhead.elapsed().as_micros() as u64;

        // Snapshot: dump logits output
        #[cfg(feature = "bench-decode-snapshot")]
        if snapshot_active {
            crate::snapshot::dump_f32("logits_output", &logits);
        }

        #[cfg(feature = "profile-decode-layer")]
        let t_argmax = std::time::Instant::now();
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| HerbertError::Backend("empty logits in decode argmax".into()))?;
        #[cfg(feature = "profile-decode-layer")]
        let argmax_us = t_argmax.elapsed().as_micros() as u64;

        #[cfg(feature = "profile-decode-layer")]
        {
            crate::profiler::report_decode_layer_detail();
            eprintln!("\n[PROFILE-DECODE] === Global decode components ===");
            eprintln!("  {:<20} {:>8.1} \u{00b5}s", "Embedding", embed_us as f64);
            eprintln!("  {:<20} {:>8.1} \u{00b5}s", "Final norm", fnorm_us as f64);
            eprintln!("  {:<20} {:>8.1} \u{00b5}s", "LM head", lmhead_us as f64);
            eprintln!("  {:<20} {:>8.1} \u{00b5}s", "Argmax", argmax_us as f64);
            let total_global = embed_us + fnorm_us + lmhead_us + argmax_us;
            eprintln!("  {:<20} {:>8.1} \u{00b5}s", "Total global", total_global as f64);
            std::process::exit(0);
        }

        let logits_out = if return_logits {
            Some(logits.clone())
        } else {
            None
        };

        kv_cache.ctx.decode_embed = x;
        kv_cache.ctx.decode_final_norm = final_norm;
        kv_cache.ctx.decode_logits = logits;

        // Snapshot: finalize
        #[cfg(feature = "bench-decode-snapshot")]
        crate::snapshot::end_decode_step();

        Ok((logits_out, next_token, layer_times))
    }

    // ====================================================================
    // Speculative decoding: verify draft tokens
    // ====================================================================

    /// Verify K draft tokens in a single forward pass, returning logits at
    /// all K+1 positions (K draft positions + 1 bonus).
    ///
    /// This is like `prefill_with_token` but extracts hidden states at every
    /// position (not just the last one) and runs norm + lm_head for each.
    pub fn verify_draft_tokens(
        &self,
        draft_tokens: &[u32],
        kv_cache: &mut CpuKvCache,
        start_pos: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let seq_len = draft_tokens.len();
        if seq_len == 0 {
            return Err(HerbertError::Backend(
                "verify_draft requires at least one token".to_string(),
            ));
        }
        let end_pos = start_pos + seq_len;
        let max_pos = self.config.max_position_embeddings;
        if end_pos > max_pos {
            return Err(HerbertError::Backend(format!(
                "verify_draft: position {} exceeds context window ({})",
                end_pos, max_pos
            )));
        }
        let hidden_size = self.config.hidden_size;
        let n = seq_len * hidden_size;

        // Ping-pong buffers
        let mut buf_a = std::mem::take(&mut kv_cache.ctx.prefill_buf_a);
        buf_a.resize(n, 0.0);
        let mut buf_b = std::mem::take(&mut kv_cache.ctx.prefill_buf_b);
        buf_b.resize(n, 0.0);

        // Get embeddings into buf_a
        for (i, &token_id) in draft_tokens.iter().enumerate() {
            let token_idx = token_id as usize;
            if token_idx >= self.config.vocab_size {
                kv_cache.ctx.prefill_buf_a = buf_a;
                kv_cache.ctx.prefill_buf_b = buf_b;
                return Err(HerbertError::Backend(format!(
                    "token_id {} out of range (vocab_size={})",
                    token_idx, self.config.vocab_size
                )));
            }
            self.embed_lookup(token_idx, &mut buf_a[i * hidden_size..(i + 1) * hidden_size]);
        }
        self.scale_embeddings(&mut buf_a[..n]);

        // Forward through layers (ping-pong)
        let num_layers = self.decoder_layers.len();
        let mut profiler = LayerProfiler::new(num_layers);
        for (layer_idx, layer) in self.decoder_layers.iter().enumerate() {
            let (layer_cos, layer_sin) = self.rope_cache_for_layer(layer_idx);
            if layer_idx % 2 == 0 {
                profiler.measure(|| {
                    layer.forward_prefill(
                        &buf_a, layer_cos, layer_sin,
                        kv_cache, layer_idx, seq_len, start_pos, &mut buf_b,
                    )
                })?;
            } else {
                profiler.measure(|| {
                    layer.forward_prefill(
                        &buf_b, layer_cos, layer_sin,
                        kv_cache, layer_idx, seq_len, start_pos, &mut buf_a,
                    )
                })?;
            }
        }
        let _layer_times = profiler.finish();
        kv_cache.advance_seq_len(seq_len);

        // Result is in buf_a if num_layers is even, buf_b if odd
        let result = if num_layers.is_multiple_of(2) { &buf_a } else { &buf_b };

        // Extract logits at ALL positions (not just last)
        let vocab_size = self.config.vocab_size;
        let mut all_logits = Vec::with_capacity(seq_len);
        let mut norm_buf = std::mem::take(&mut kv_cache.ctx.decode_final_norm);
        if norm_buf.len() != hidden_size {
            norm_buf.resize(hidden_size, 0.0);
        }
        let mut logits_buf = std::mem::take(&mut kv_cache.ctx.decode_logits);
        if logits_buf.len() != vocab_size {
            logits_buf.resize(vocab_size, 0.0);
        }

        for pos in 0..seq_len {
            let hidden = &result[pos * hidden_size..(pos + 1) * hidden_size];
            self.apply_final_norm(hidden, &mut norm_buf);
            L::matvec(&norm_buf, &self.lm_head, &mut logits_buf)?;
            self.apply_lm_head_bias(&mut logits_buf);
            all_logits.push(logits_buf.clone());
        }

        kv_cache.ctx.prefill_buf_a = buf_a;
        kv_cache.ctx.prefill_buf_b = buf_b;
        kv_cache.ctx.decode_final_norm = norm_buf;
        kv_cache.ctx.decode_logits = logits_buf;

        Ok(all_logits)
    }

    /// MRoPE variant of verify_draft_tokens for VL models.
    ///
    /// Like verify_draft_tokens, but computes MRoPE cos/sin caches dynamically
    /// from 3D text positions instead of using precomputed standard RoPE cache.
    pub fn verify_draft_tokens_mrope(
        &self,
        draft_tokens: &[u32],
        positions: &[Position3D],
        kv_cache: &mut CpuKvCache,
        start_pos: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let seq_len = draft_tokens.len();
        if seq_len == 0 {
            return Err(HerbertError::Backend(
                "verify_draft_mrope requires at least one token".to_string(),
            ));
        }
        if positions.len() != seq_len {
            return Err(HerbertError::Backend(format!(
                "positions length {} != draft_tokens length {}",
                positions.len(), seq_len
            )));
        }
        let end_pos = start_pos + seq_len;
        let max_pos = self.config.max_position_embeddings;
        if end_pos > max_pos {
            return Err(HerbertError::Backend(format!(
                "verify_draft_mrope: position {} exceeds context window ({})",
                end_pos, max_pos
            )));
        }
        let hidden_size = self.config.hidden_size;
        let n = seq_len * hidden_size;

        // Compute MRoPE cos/sin cache from 3D positions
        let half_dim = self.config.rotary_ndims / 2;
        let pos_arrays: Vec<[u32; 3]> = positions.iter().map(|p| p.as_array()).collect();
        let total_positions = start_pos + seq_len;
        let mut cos_cache = vec![0.0f32; total_positions * half_dim];
        let mut sin_cache = vec![0.0f32; total_positions * half_dim];
        for (i, pos_3d) in pos_arrays.iter().enumerate() {
            let offset = (start_pos + i) * half_dim;
            mrope::compute_mrope_cos_sin(
                *pos_3d,
                &self.inv_freq,
                &self.mrope_pattern,
                &mut cos_cache[offset..offset + half_dim],
                &mut sin_cache[offset..offset + half_dim],
            );
        }

        // Ping-pong buffers
        let mut buf_a = std::mem::take(&mut kv_cache.ctx.prefill_buf_a);
        buf_a.resize(n, 0.0);
        let mut buf_b = std::mem::take(&mut kv_cache.ctx.prefill_buf_b);
        buf_b.resize(n, 0.0);

        // Get embeddings into buf_a
        for (i, &token_id) in draft_tokens.iter().enumerate() {
            let token_idx = token_id as usize;
            if token_idx >= self.config.vocab_size {
                kv_cache.ctx.prefill_buf_a = buf_a;
                kv_cache.ctx.prefill_buf_b = buf_b;
                return Err(HerbertError::Backend(format!(
                    "token_id {} out of range (vocab_size={})",
                    token_idx, self.config.vocab_size
                )));
            }
            self.embed_lookup(token_idx, &mut buf_a[i * hidden_size..(i + 1) * hidden_size]);
        }
        self.scale_embeddings(&mut buf_a[..n]);

        // Forward through layers (ping-pong) using dynamic MRoPE cos/sin
        let num_layers = self.decoder_layers.len();
        let mut profiler = LayerProfiler::new(num_layers);
        for (layer_idx, layer) in self.decoder_layers.iter().enumerate() {
            if layer_idx % 2 == 0 {
                profiler.measure(|| {
                    layer.forward_prefill(
                        &buf_a, &cos_cache, &sin_cache,
                        kv_cache, layer_idx, seq_len, start_pos, &mut buf_b,
                    )
                })?;
            } else {
                profiler.measure(|| {
                    layer.forward_prefill(
                        &buf_b, &cos_cache, &sin_cache,
                        kv_cache, layer_idx, seq_len, start_pos, &mut buf_a,
                    )
                })?;
            }
        }
        let _layer_times = profiler.finish();
        kv_cache.advance_seq_len(seq_len);

        // Result is in buf_a if num_layers is even, buf_b if odd
        let result = if num_layers.is_multiple_of(2) { &buf_a } else { &buf_b };

        // Extract logits at ALL positions
        let vocab_size = self.config.vocab_size;
        let mut all_logits = Vec::with_capacity(seq_len);
        let mut norm_buf = std::mem::take(&mut kv_cache.ctx.decode_final_norm);
        if norm_buf.len() != hidden_size {
            norm_buf.resize(hidden_size, 0.0);
        }
        let mut logits_buf = std::mem::take(&mut kv_cache.ctx.decode_logits);
        if logits_buf.len() != vocab_size {
            logits_buf.resize(vocab_size, 0.0);
        }

        for pos in 0..seq_len {
            let hidden = &result[pos * hidden_size..(pos + 1) * hidden_size];
            self.apply_final_norm(hidden, &mut norm_buf);
            L::matvec(&norm_buf, &self.lm_head, &mut logits_buf)?;
            self.apply_lm_head_bias(&mut logits_buf);
            all_logits.push(logits_buf.clone());
        }

        kv_cache.ctx.prefill_buf_a = buf_a;
        kv_cache.ctx.prefill_buf_b = buf_b;
        kv_cache.ctx.decode_final_norm = norm_buf;
        kv_cache.ctx.decode_logits = logits_buf;

        Ok(all_logits)
    }

    // ====================================================================
    // MRoPE methods (for VL models)
    // ====================================================================

    /// Prefill with precomputed embeddings and 3D positions (MRoPE).
    ///
    /// Used for VL models where embeddings come from mixed text+vision tokens.
    /// `start_pos` is the KV cache offset (0 for initial prefill, >0 for continue_prefill).
    pub fn prefill_with_embeds_mrope(
        &self,
        embeds: &[f32],
        positions: &[Position3D],
        kv_cache: &mut CpuKvCache,
        return_logits: bool,
        deepstack: Option<&DeepStackInfo<'_>>,
        show_progress: bool,
        start_pos: usize,
    ) -> Result<(Option<Vec<f32>>, u32, Vec<f64>)> {
        let hidden_size = self.config.hidden_size;
        let seq_len = positions.len();
        if seq_len == 0 {
            return Err(HerbertError::Backend(
                "prefill_mrope requires at least one position".to_string(),
            ));
        }
        let end_pos = start_pos + seq_len;
        let max_pos = self.config.max_position_embeddings;
        if end_pos > max_pos {
            return Err(HerbertError::Backend(format!(
                "sequence length {} (start_pos={} + {} tokens) exceeds context window (max_position_embeddings={})",
                end_pos, start_pos, seq_len, max_pos
            )));
        }
        if embeds.len() != seq_len * hidden_size {
            return Err(HerbertError::Backend(format!(
                "embeds length {} != seq_len({}) * hidden_size({})",
                embeds.len(), seq_len, hidden_size
            )));
        }

        // Compute cos/sin cache from 3D positions.
        // The cache must cover indices [start_pos..start_pos+seq_len] because
        // apply_rope_batch indexes at (start_pos + pos) * half_dim.
        let half_dim = self.config.rotary_ndims / 2;
        let pos_arrays: Vec<[u32; 3]> = positions.iter().map(|p| p.as_array()).collect();
        let total_positions = start_pos + seq_len;
        let mut cos_cache = vec![0.0f32; total_positions * half_dim];
        let mut sin_cache = vec![0.0f32; total_positions * half_dim];
        // Fill only [start_pos..start_pos+seq_len]; earlier positions are never read.
        for (i, pos_3d) in pos_arrays.iter().enumerate() {
            let offset = (start_pos + i) * half_dim;
            mrope::compute_mrope_cos_sin(
                *pos_3d,
                &self.inv_freq,
                &self.mrope_pattern,
                &mut cos_cache[offset..offset + half_dim],
                &mut sin_cache[offset..offset + half_dim],
            );
        }

        // Forward through layers (ping-pong)

        let num_layers = self.decoder_layers.len();
        let n = seq_len * hidden_size;
        let pb = if show_progress {
            tracing::info!(seq_len, "Prefill");
            let pb = ProgressBar::new(num_layers as u64);
            pb.set_style(
                ProgressStyle::with_template("  Prefill [{bar:30}] {pos}/{len} layers")
                    .expect("valid progress template")
                    .progress_chars("█░░"),
            );
            Some(pb)
        } else {
            None
        };

        // Ping-pong buffers
        let mut buf_a = std::mem::take(&mut kv_cache.ctx.prefill_buf_a);
        buf_a.resize(n, 0.0);
        let mut buf_b = std::mem::take(&mut kv_cache.ctx.prefill_buf_b);
        buf_b.resize(n, 0.0);
        buf_a[..n].copy_from_slice(embeds);

        #[cfg(feature = "profile-layer")]
        crate::profiler::reset(seq_len, num_layers);
        #[cfg(feature = "profile-prefill-layer")]
        crate::profiler::reset_prefill_layer_detail();
        #[cfg(feature = "profile-attn-kernel")]
        crate::profiler::reset_attn_kernel_breakdown();
        let mut profiler = LayerProfiler::new(num_layers);
        for (layer_idx, layer) in self.decoder_layers.iter().enumerate() {
            if layer_idx % 2 == 0 {
                profiler.measure(|| {
                    layer.forward_prefill(
                        &buf_a, &cos_cache, &sin_cache,
                        kv_cache, layer_idx, seq_len, start_pos, &mut buf_b,
                    )
                })?;
                // DeepStack injection into buf_b (the output of this layer)
                if let Some(ds) = deepstack {
                    if layer_idx < ds.layer_features.len() {
                        let feats = &ds.layer_features[layer_idx];
                        for (i, &tok_idx) in ds.image_token_indices.iter().enumerate() {
                            let base = tok_idx * hidden_size;
                            let feat_base = i * hidden_size;
                            for j in 0..hidden_size {
                                buf_b[base + j] += feats[feat_base + j];
                            }
                        }
                    }
                }
            } else {
                profiler.measure(|| {
                    layer.forward_prefill(
                        &buf_b, &cos_cache, &sin_cache,
                        kv_cache, layer_idx, seq_len, start_pos, &mut buf_a,
                    )
                })?;
                // DeepStack injection into buf_a (the output of this layer)
                if let Some(ds) = deepstack {
                    if layer_idx < ds.layer_features.len() {
                        let feats = &ds.layer_features[layer_idx];
                        for (i, &tok_idx) in ds.image_token_indices.iter().enumerate() {
                            let base = tok_idx * hidden_size;
                            let feat_base = i * hidden_size;
                            for j in 0..hidden_size {
                                buf_a[base + j] += feats[feat_base + j];
                            }
                        }
                    }
                }
            }
            #[cfg(feature = "profile-prefill-layer")]
            if layer_idx == 1 {
                crate::profiler::report_prefill_layer_detail(seq_len);
                #[cfg(feature = "profile-attn-kernel")]
                crate::profiler::report_attn_kernel_breakdown();
                std::process::exit(0);
            }
            if let Some(ref pb) = pb {
                pb.set_position((layer_idx + 1) as u64);
            }
        }
        if let Some(pb) = pb {
            pb.finish_and_clear();
        }
        #[cfg(feature = "profile-layer")]
        crate::profiler::report();
        #[cfg(feature = "profile-attn-kernel")]
        crate::profiler::report_attn_kernel_breakdown();
        let layer_times = profiler.finish();
        kv_cache.advance_seq_len(seq_len);

        // Result is in buf_a if num_layers is even, buf_b if odd
        let result = if num_layers.is_multiple_of(2) { &buf_a } else { &buf_b };

        // Final norm + LM head
        let last_hidden = &result[(seq_len - 1) * hidden_size..seq_len * hidden_size];
        let mut final_norm = std::mem::take(&mut kv_cache.ctx.decode_final_norm);
        if final_norm.len() != hidden_size {
            final_norm.resize(hidden_size, 0.0);
        }
        self.apply_final_norm(last_hidden, &mut final_norm);

        let vocab_size = self.config.vocab_size;
        let mut logits = std::mem::take(&mut kv_cache.ctx.decode_logits);
        if logits.len() != vocab_size {
            logits.resize(vocab_size, 0.0);
        }
        L::matvec(&final_norm, &self.lm_head, &mut logits)?;
        self.apply_lm_head_bias(&mut logits);

        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| HerbertError::Backend("empty logits in mrope prefill argmax".into()))?;
        let logits_out = if return_logits {
            Some(logits.clone())
        } else {
            None
        };

        kv_cache.ctx.prefill_buf_a = buf_a;
        kv_cache.ctx.prefill_buf_b = buf_b;
        kv_cache.ctx.decode_final_norm = final_norm;
        kv_cache.ctx.decode_logits = logits;

        Ok((logits_out, next_token, layer_times))
    }

    /// MRoPE prefill with per-layer callbacks for pipeline overlap.
    ///
    /// `before_layer(layer_idx, kv_cache)` is called before each layer (mutable: can prepend KV).
    /// `after_layer(layer_idx, kv_cache)` is called after each layer (immutable: can read KV).
    ///
    /// Stripped-down version of `prefill_with_embeds_mrope()` without progress bar,
    /// profiler, or DeepStack injection — intended for distributed prefill pipeline.
    pub fn prefill_with_embeds_mrope_layer_hooks<BH, AH>(
        &self,
        embeds: &[f32],
        positions: &[Position3D],
        kv_cache: &mut CpuKvCache,
        return_logits: bool,
        start_pos: usize,
        before_layer: &mut BH,
        after_layer: &mut AH,
    ) -> Result<(Option<Vec<f32>>, u32, Vec<f64>)>
    where
        BH: FnMut(usize, &mut CpuKvCache) -> Result<()>,
        AH: FnMut(usize, &mut CpuKvCache) -> Result<()>,
    {
        let hidden_size = self.config.hidden_size;
        let seq_len = positions.len();
        if seq_len == 0 {
            return Err(HerbertError::Backend(
                "prefill_mrope requires at least one position".to_string(),
            ));
        }
        let end_pos = start_pos + seq_len;
        let max_pos = self.config.max_position_embeddings;
        if end_pos > max_pos {
            return Err(HerbertError::Backend(format!(
                "sequence length {} (start_pos={} + {} tokens) exceeds context window (max_position_embeddings={})",
                end_pos, start_pos, seq_len, max_pos
            )));
        }
        if embeds.len() != seq_len * hidden_size {
            return Err(HerbertError::Backend(format!(
                "embeds length {} != seq_len({}) * hidden_size({})",
                embeds.len(), seq_len, hidden_size
            )));
        }

        // Compute cos/sin cache from 3D positions
        let half_dim = self.config.rotary_ndims / 2;
        let pos_arrays: Vec<[u32; 3]> = positions.iter().map(|p| p.as_array()).collect();
        let total_positions = start_pos + seq_len;
        let mut cos_cache = vec![0.0f32; total_positions * half_dim];
        let mut sin_cache = vec![0.0f32; total_positions * half_dim];
        for (i, pos_3d) in pos_arrays.iter().enumerate() {
            let offset = (start_pos + i) * half_dim;
            mrope::compute_mrope_cos_sin(
                *pos_3d,
                &self.inv_freq,
                &self.mrope_pattern,
                &mut cos_cache[offset..offset + half_dim],
                &mut sin_cache[offset..offset + half_dim],
            );
        }

        // Ping-pong buffers
        let num_layers = self.decoder_layers.len();
        let n = seq_len * hidden_size;
        let mut buf_a = std::mem::take(&mut kv_cache.ctx.prefill_buf_a);
        buf_a.resize(n, 0.0);
        let mut buf_b = std::mem::take(&mut kv_cache.ctx.prefill_buf_b);
        buf_b.resize(n, 0.0);
        buf_a[..n].copy_from_slice(embeds);

        // Forward through layers with hooks
        let mut profiler = LayerProfiler::new(num_layers);
        for (layer_idx, layer) in self.decoder_layers.iter().enumerate() {
            before_layer(layer_idx, kv_cache)?;
            if layer_idx % 2 == 0 {
                profiler.measure(|| {
                    layer.forward_prefill(
                        &buf_a, &cos_cache, &sin_cache,
                        kv_cache, layer_idx, seq_len, start_pos, &mut buf_b,
                    )
                })?;
            } else {
                profiler.measure(|| {
                    layer.forward_prefill(
                        &buf_b, &cos_cache, &sin_cache,
                        kv_cache, layer_idx, seq_len, start_pos, &mut buf_a,
                    )
                })?;
            }
            after_layer(layer_idx, kv_cache)?;
        }
        let layer_times = profiler.finish();
        kv_cache.advance_seq_len(seq_len);

        // Result is in buf_a if num_layers is even, buf_b if odd
        let result = if num_layers.is_multiple_of(2) { &buf_a } else { &buf_b };

        // Final norm + LM head
        let last_hidden = &result[(seq_len - 1) * hidden_size..seq_len * hidden_size];
        let mut final_norm = std::mem::take(&mut kv_cache.ctx.decode_final_norm);
        if final_norm.len() != hidden_size {
            final_norm.resize(hidden_size, 0.0);
        }
        self.apply_final_norm(last_hidden, &mut final_norm);

        let vocab_size = self.config.vocab_size;
        let mut logits = std::mem::take(&mut kv_cache.ctx.decode_logits);
        if logits.len() != vocab_size {
            logits.resize(vocab_size, 0.0);
        }
        L::matvec(&final_norm, &self.lm_head, &mut logits)?;
        self.apply_lm_head_bias(&mut logits);

        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| HerbertError::Backend("empty logits in mrope prefill argmax".into()))?;
        let logits_out = if return_logits {
            Some(logits.clone())
        } else {
            None
        };

        kv_cache.ctx.prefill_buf_a = buf_a;
        kv_cache.ctx.prefill_buf_b = buf_b;
        kv_cache.ctx.decode_final_norm = final_norm;
        kv_cache.ctx.decode_logits = logits;

        Ok((logits_out, next_token, layer_times))
    }

    /// Prefill with precomputed embeddings and standard 1D RoPE positions.
    ///
    /// Used for VL models that don't use MRoPE (e.g. Mistral3/Pixtral, LFM2).
    /// All tokens get sequential positions [start_pos..start_pos+seq_len] using
    /// the pre-computed cos/sin cache (standard RoPE), no DeepStack injection.
    pub fn prefill_with_embeds(
        &self,
        embeds: &[f32],
        kv_cache: &mut CpuKvCache,
        start_pos: usize,
        return_logits: bool,
        show_progress: bool,
    ) -> Result<(Option<Vec<f32>>, u32, Vec<f64>)> {
        let hidden_size = self.config.hidden_size;
        let seq_len = embeds.len() / hidden_size;
        if seq_len == 0 {
            return Err(HerbertError::Backend(
                "prefill_with_embeds requires at least one token".to_string(),
            ));
        }
        if embeds.len() != seq_len * hidden_size {
            return Err(HerbertError::Backend(format!(
                "embeds length {} not divisible by hidden_size ({})",
                embeds.len(), hidden_size
            )));
        }
        let end_pos = start_pos + seq_len;
        let max_pos = self.config.max_position_embeddings;
        if end_pos > max_pos {
            return Err(HerbertError::Backend(format!(
                "sequence length {} (start_pos={} + {} tokens) exceeds context window (max_position_embeddings={})",
                end_pos, start_pos, seq_len, max_pos
            )));
        }

        // Forward through layers using standard cos/sin cache (same as prefill_with_token)
        let num_layers = self.decoder_layers.len();
        let n = seq_len * hidden_size;
        let pb = if show_progress {
            tracing::info!(seq_len, "Prefill (VL, 1D RoPE)");
            let pb = ProgressBar::new(num_layers as u64);
            pb.set_style(
                ProgressStyle::with_template("  Prefill [{bar:30}] {pos}/{len} layers")
                    .expect("valid progress template")
                    .progress_chars("█░░"),
            );
            Some(pb)
        } else {
            None
        };

        // Ping-pong buffers
        let mut buf_a = std::mem::take(&mut kv_cache.ctx.prefill_buf_a);
        buf_a.resize(n, 0.0);
        let mut buf_b = std::mem::take(&mut kv_cache.ctx.prefill_buf_b);
        buf_b.resize(n, 0.0);
        buf_a[..n].copy_from_slice(embeds);

        let mut profiler = LayerProfiler::new(num_layers);
        for (layer_idx, layer) in self.decoder_layers.iter().enumerate() {
            let (layer_cos, layer_sin) = self.rope_cache_for_layer(layer_idx);
            if layer_idx % 2 == 0 {
                profiler.measure(|| {
                    layer.forward_prefill(
                        &buf_a, layer_cos, layer_sin,
                        kv_cache, layer_idx, seq_len, start_pos, &mut buf_b,
                    )
                })?;
            } else {
                profiler.measure(|| {
                    layer.forward_prefill(
                        &buf_b, layer_cos, layer_sin,
                        kv_cache, layer_idx, seq_len, start_pos, &mut buf_a,
                    )
                })?;
            }
            if let Some(ref pb) = pb {
                pb.set_position((layer_idx + 1) as u64);
            }
        }
        if let Some(pb) = pb {
            pb.finish_and_clear();
        }
        let layer_times = profiler.finish();
        kv_cache.advance_seq_len(seq_len);

        // Result is in buf_a if num_layers is even, buf_b if odd
        let result = if num_layers.is_multiple_of(2) { &buf_a } else { &buf_b };

        // Final norm + LM head
        let last_hidden = &result[(seq_len - 1) * hidden_size..seq_len * hidden_size];
        let mut final_norm = std::mem::take(&mut kv_cache.ctx.decode_final_norm);
        if final_norm.len() != hidden_size {
            final_norm.resize(hidden_size, 0.0);
        }
        self.apply_final_norm(last_hidden, &mut final_norm);

        let vocab_size = self.config.vocab_size;
        let mut logits = std::mem::take(&mut kv_cache.ctx.decode_logits);
        if logits.len() != vocab_size {
            logits.resize(vocab_size, 0.0);
        }
        L::matvec(&final_norm, &self.lm_head, &mut logits)?;
        self.apply_lm_head_bias(&mut logits);

        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| HerbertError::Backend("empty logits in embeds prefill argmax".into()))?;
        let logits_out = if return_logits {
            Some(logits.clone())
        } else {
            None
        };

        kv_cache.ctx.prefill_buf_a = buf_a;
        kv_cache.ctx.prefill_buf_b = buf_b;
        kv_cache.ctx.decode_final_norm = final_norm;
        kv_cache.ctx.decode_logits = logits;

        Ok((logits_out, next_token, layer_times))
    }

    /// Stateless embedding forward pass: prefill tokens, extract the last-token
    /// hidden state, apply RMS norm, L2-normalise → `Vec<f32>` of `hidden_size` dims.
    ///
    /// The KV cache is NOT advanced — this is a one-shot, disposable forward pass.
    pub fn prefill_for_embedding(
        &self,
        input_tokens: &[u32],
        kv_cache: &mut CpuKvCache,
    ) -> Result<Vec<f32>> {
        let seq_len = input_tokens.len();
        if seq_len == 0 {
            return Err(HerbertError::Backend(
                "embed requires at least one input token".to_string(),
            ));
        }
        let max_pos = self.config.max_position_embeddings;
        if seq_len > max_pos {
            return Err(HerbertError::Backend(format!(
                "embedding input length {} exceeds context window (max_position_embeddings={})",
                seq_len, max_pos
            )));
        }
        let hidden_size = self.config.hidden_size;
        let n = seq_len * hidden_size;

        // Ping-pong buffers
        let mut buf_a = std::mem::take(&mut kv_cache.ctx.prefill_buf_a);
        buf_a.resize(n, 0.0);
        let mut buf_b = std::mem::take(&mut kv_cache.ctx.prefill_buf_b);
        buf_b.resize(n, 0.0);

        // Get embeddings into buf_a
        for (i, &token_id) in input_tokens.iter().enumerate() {
            let token_idx = token_id as usize;
            if token_idx >= self.config.vocab_size {
                kv_cache.ctx.prefill_buf_a = buf_a;
                kv_cache.ctx.prefill_buf_b = buf_b;
                return Err(HerbertError::Backend(format!(
                    "token_id {} out of range (vocab_size={})",
                    token_idx, self.config.vocab_size
                )));
            }
            self.embed_lookup(token_idx, &mut buf_a[i * hidden_size..(i + 1) * hidden_size]);
        }
        self.scale_embeddings(&mut buf_a[..n]);

        // Compute cos/sin — standard RoPE or MRoPE
        let (cos_ref, sin_ref);
        let cos_mrope;
        let sin_mrope;
        if self.is_mrope() {
            let half_dim = self.config.rotary_ndims / 2;
            cos_mrope = vec![0.0f32; seq_len * half_dim];
            sin_mrope = vec![0.0f32; seq_len * half_dim];
            let cos_ptr = cos_mrope.as_ptr() as *mut f32;
            let sin_ptr = sin_mrope.as_ptr() as *mut f32;
            for i in 0..seq_len {
                let pos_3d = [i as u32; 3];
                let offset = i * half_dim;
                unsafe {
                    let cos_slice = std::slice::from_raw_parts_mut(cos_ptr.add(offset), half_dim);
                    let sin_slice = std::slice::from_raw_parts_mut(sin_ptr.add(offset), half_dim);
                    mrope::compute_mrope_cos_sin(
                        pos_3d,
                        &self.inv_freq,
                        &self.mrope_pattern,
                        cos_slice,
                        sin_slice,
                    );
                }
            }
            cos_ref = &cos_mrope;
            sin_ref = &sin_mrope;
        } else {
            cos_ref = &self.cos_cache;
            sin_ref = &self.sin_cache;
        }

        // Forward through layers (ping-pong)
        let num_layers = self.decoder_layers.len();
        for (layer_idx, layer) in self.decoder_layers.iter().enumerate() {
            if layer_idx % 2 == 0 {
                layer.forward_prefill(
                    &buf_a, cos_ref, sin_ref,
                    kv_cache, layer_idx, seq_len, 0, &mut buf_b,
                )?;
            } else {
                layer.forward_prefill(
                    &buf_b, cos_ref, sin_ref,
                    kv_cache, layer_idx, seq_len, 0, &mut buf_a,
                )?;
            }
        }
        // DO NOT advance kv_cache.kv.seq_len — embedding is stateless

        // Result is in buf_a if num_layers is even, buf_b if odd
        let result = if num_layers.is_multiple_of(2) { &buf_a } else { &buf_b };

        // Last-token pooling + final norm
        let last_hidden = &result[(seq_len - 1) * hidden_size..seq_len * hidden_size];
        let mut normed = vec![0.0f32; hidden_size];
        self.apply_final_norm(last_hidden, &mut normed);

        // L2 normalize
        let norm_sq: f32 = normed.iter().map(|v| v * v).sum();
        let norm_val = norm_sq.sqrt();
        if norm_val > 0.0 {
            let inv = 1.0 / norm_val;
            for v in &mut normed {
                *v *= inv;
            }
        }

        kv_cache.ctx.prefill_buf_a = buf_a;
        kv_cache.ctx.prefill_buf_b = buf_b;

        Ok(normed)
    }

    /// Stateless embedding forward pass returning ALL token hidden states.
    ///
    /// Returns `seq_len * hidden_size` floats: each position is RMS-normed and
    /// L2-normalised independently.  When `bidirectional` is true the attention
    /// layers use full (non-causal) masking.
    ///
    /// The KV cache is NOT advanced — this is a one-shot, disposable pass.
    pub fn prefill_all_embeddings(
        &self,
        input_tokens: &[u32],
        kv_cache: &mut CpuKvCache,
        bidirectional: bool,
        l2_normalize: bool,
    ) -> Result<Vec<f32>> {
        let seq_len = input_tokens.len();
        if seq_len == 0 {
            return Err(HerbertError::Backend(
                "embed requires at least one input token".to_string(),
            ));
        }
        let max_pos = self.config.max_position_embeddings;
        if seq_len > max_pos {
            return Err(HerbertError::Backend(format!(
                "embedding input length {} exceeds context window (max_position_embeddings={})",
                seq_len, max_pos
            )));
        }
        let hidden_size = self.config.hidden_size;
        let n = seq_len * hidden_size;

        // Set bidirectional flag for attention layers
        kv_cache.kv.bidirectional = bidirectional;

        // Ping-pong buffers
        let mut buf_a = std::mem::take(&mut kv_cache.ctx.prefill_buf_a);
        buf_a.resize(n, 0.0);
        let mut buf_b = std::mem::take(&mut kv_cache.ctx.prefill_buf_b);
        buf_b.resize(n, 0.0);

        // Get embeddings into buf_a
        for (i, &token_id) in input_tokens.iter().enumerate() {
            let token_idx = token_id as usize;
            if token_idx >= self.config.vocab_size {
                kv_cache.kv.bidirectional = false;
                kv_cache.ctx.prefill_buf_a = buf_a;
                kv_cache.ctx.prefill_buf_b = buf_b;
                return Err(HerbertError::Backend(format!(
                    "token_id {} out of range (vocab_size={})",
                    token_idx, self.config.vocab_size
                )));
            }
            self.embed_lookup(token_idx, &mut buf_a[i * hidden_size..(i + 1) * hidden_size]);
        }
        self.scale_embeddings(&mut buf_a[..n]);

        // Compute cos/sin — standard RoPE
        let cos_ref = &self.cos_cache;
        let sin_ref = &self.sin_cache;

        // Forward through layers (ping-pong)
        let num_layers = self.decoder_layers.len();
        for (layer_idx, layer) in self.decoder_layers.iter().enumerate() {
            if layer_idx % 2 == 0 {
                layer.forward_prefill(
                    &buf_a, cos_ref, sin_ref,
                    kv_cache, layer_idx, seq_len, 0, &mut buf_b,
                )?;
            } else {
                layer.forward_prefill(
                    &buf_b, cos_ref, sin_ref,
                    kv_cache, layer_idx, seq_len, 0, &mut buf_a,
                )?;
            }
        }
        // DO NOT advance kv_cache.kv.seq_len — embedding is stateless
        kv_cache.kv.bidirectional = false;

        // Result is in buf_a if num_layers is even, buf_b if odd
        let result = if num_layers.is_multiple_of(2) { &buf_a } else { &buf_b };

        // Final norm each position, optionally L2-normalise
        let mut out = vec![0.0f32; n];
        for pos in 0..seq_len {
            let src = &result[pos * hidden_size..(pos + 1) * hidden_size];
            let dst = &mut out[pos * hidden_size..(pos + 1) * hidden_size];
            self.apply_final_norm(src, dst);
            if l2_normalize {
                let norm_sq: f32 = dst.iter().map(|v| v * v).sum();
                let norm_val = norm_sq.sqrt();
                if norm_val > 0.0 {
                    let inv = 1.0 / norm_val;
                    for v in dst.iter_mut() {
                        *v *= inv;
                    }
                }
            }
        }

        kv_cache.ctx.prefill_buf_a = buf_a;
        kv_cache.ctx.prefill_buf_b = buf_b;

        Ok(out)
    }

    /// Decode a single text token on a VL model using MRoPE.
    ///
    /// Uses `kv_cache.kv.text_pos` for the text position coordinate (all 3 dims = text_pos).
    pub fn decode_step_mrope(
        &self,
        token: u32,
        kv_cache: &mut CpuKvCache,
        return_logits: bool,
    ) -> Result<(Option<Vec<f32>>, u32, Vec<f64>)> {
        let pos = kv_cache.kv.seq_len;
        let max_pos = self.config.max_position_embeddings;
        if pos >= max_pos {
            return Err(HerbertError::Backend(format!(
                "sequence position {} exceeds context window (max_position_embeddings={})",
                pos, max_pos
            )));
        }
        let hidden_size = self.config.hidden_size;
        let half_dim = self.config.rotary_ndims / 2;

        // Get embedding (F32)
        let token_idx = token as usize;
        let mut x = std::mem::take(&mut kv_cache.ctx.decode_embed);
        if x.len() != hidden_size {
            x.resize(hidden_size, 0.0);
        }
        x.fill(0.0);
        if token_idx >= self.config.vocab_size {
            return Err(HerbertError::Backend(format!(
                "token_id {} out of range (vocab_size={})",
                token_idx, self.config.vocab_size
            )));
        }
        self.embed_lookup(token_idx, &mut x);
        self.scale_embeddings(&mut x);

        // Compute MRoPE cos/sin for this text position
        let text_pos = kv_cache.kv.text_pos;
        let pos_3d = Position3D::text(text_pos);
        let mut cos_buf = std::mem::take(&mut kv_cache.ctx.mrope_cos);
        let mut sin_buf = std::mem::take(&mut kv_cache.ctx.mrope_sin);
        if cos_buf.len() != half_dim {
            cos_buf.resize(half_dim, 0.0);
        }
        if sin_buf.len() != half_dim {
            sin_buf.resize(half_dim, 0.0);
        }
        mrope::compute_mrope_cos_sin(
            pos_3d.as_array(),
            &self.inv_freq,
            &self.mrope_pattern,
            &mut cos_buf,
            &mut sin_buf,
        );

        // Forward through layers

        #[cfg(feature = "profile-decode-layer")]
        crate::profiler::reset_decode_layer_detail();
        let mut profiler = LayerProfiler::new(self.decoder_layers.len());
        for (layer_idx, layer) in self.decoder_layers.iter().enumerate() {
            profiler
                .measure(|| layer.forward_decode(&mut x, &cos_buf, &sin_buf, kv_cache, layer_idx, pos))?;

            // EAGLE-3: capture hidden states at extraction layers
            if kv_cache.ctx.eagle3_active {
                for slot in 0..3 {
                    if layer_idx == kv_cache.ctx.eagle3_extract_layers[slot] {
                        let hs = &mut kv_cache.ctx.eagle3_hidden_states[slot];
                        hs.resize(x.len(), 0.0);
                        hs.copy_from_slice(&x);
                    }
                }
            }

            #[cfg(feature = "profile-decode-layer")]
            if layer_idx == self.decoder_layers.len() - 1 {
                crate::profiler::report_decode_layer_detail();
                // Continue to measure global components before exiting
            }
        }
        let layer_times = profiler.finish();
        kv_cache.advance_seq_len(1);
        kv_cache.kv.text_pos += 1;


        // Final norm + LM head
        #[cfg(feature = "profile-decode-layer")]
        let t_fnorm = std::time::Instant::now();
        let mut final_norm = std::mem::take(&mut kv_cache.ctx.decode_final_norm);
        if final_norm.len() != hidden_size {
            final_norm.resize(hidden_size, 0.0);
        }
        self.apply_final_norm(&x, &mut final_norm);
        #[cfg(feature = "profile-decode-layer")]
        let fnorm_us = t_fnorm.elapsed().as_micros() as u64;

        #[cfg(feature = "profile-decode-layer")]
        let t_lmhead = std::time::Instant::now();
        let vocab_size = self.config.vocab_size;
        let mut logits = std::mem::take(&mut kv_cache.ctx.decode_logits);
        if logits.len() != vocab_size {
            logits.resize(vocab_size, 0.0);
        }
        L::matvec(&final_norm, &self.lm_head, &mut logits)?;
        self.apply_lm_head_bias(&mut logits);
        #[cfg(feature = "profile-decode-layer")]
        let lmhead_us = t_lmhead.elapsed().as_micros() as u64;

        #[cfg(feature = "profile-decode-layer")]
        let t_argmax = std::time::Instant::now();
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| HerbertError::Backend("empty logits in mrope decode argmax".into()))?;
        #[cfg(feature = "profile-decode-layer")]
        {
            let argmax_us = t_argmax.elapsed().as_micros() as u64;
            eprintln!("\n[PROFILE-DECODE] === Global decode components (MRoPE path) ===");
            eprintln!("  {:<20} {:>8.1} \u{00b5}s", "Final norm", fnorm_us as f64);
            eprintln!("  {:<20} {:>8.1} \u{00b5}s ({:.1} ms)", "LM head", lmhead_us as f64, lmhead_us as f64 / 1000.0);
            eprintln!("  {:<20} {:>8.1} \u{00b5}s", "Argmax", argmax_us as f64);
            let total_global = fnorm_us + lmhead_us + argmax_us;
            eprintln!("  {:<20} {:>8.1} \u{00b5}s ({:.1} ms)", "Total global", total_global as f64, total_global as f64 / 1000.0);
            std::process::exit(0);
        }
        let logits_out = if return_logits {
            Some(logits.clone())
        } else {
            None
        };

        kv_cache.ctx.decode_embed = x;
        kv_cache.ctx.decode_final_norm = final_norm;
        kv_cache.ctx.decode_logits = logits;
        kv_cache.ctx.mrope_cos = cos_buf;
        kv_cache.ctx.mrope_sin = sin_buf;

        Ok((logits_out, next_token, layer_times))
    }

    /// Decomposed prefill using Phase A/B split.
    ///
    /// Functionally identical to `prefill_with_token`, but separates each layer
    /// into Phase A (QKV + RoPE, independent of remote KV) and Phase B
    /// (attention kernel + FFN, depends on full KV cache).
    ///
    /// This enables distributed prefill: remote KV can be received between
    /// Phase A and Phase B of each layer.
    pub fn prefill_with_token_decomposed(
        &self,
        input_tokens: &[u32],
        kv_cache: &mut CpuKvCache,
        start_pos: usize,
        return_logits: bool,
    ) -> Result<(Option<Vec<f32>>, u32, Vec<f64>)> {
        let seq_len = input_tokens.len();
        if seq_len == 0 {
            return Err(HerbertError::Backend(
                "prefill requires at least one input token".to_string(),
            ));
        }
        let hidden_size = self.config.hidden_size;
        let n = seq_len * hidden_size;

        // Ping-pong buffers
        let mut buf_a = std::mem::take(&mut kv_cache.ctx.prefill_buf_a);
        buf_a.resize(n, 0.0);
        let mut buf_b = std::mem::take(&mut kv_cache.ctx.prefill_buf_b);
        buf_b.resize(n, 0.0);

        // Load embeddings
        for (i, &token_id) in input_tokens.iter().enumerate() {
            let token_idx = token_id as usize;
            if token_idx >= self.config.vocab_size {
                kv_cache.ctx.prefill_buf_a = buf_a;
                kv_cache.ctx.prefill_buf_b = buf_b;
                return Err(HerbertError::Backend(format!(
                    "token_id {} out of range (vocab_size={})",
                    token_idx, self.config.vocab_size
                )));
            }
            self.embed_lookup(token_idx, &mut buf_a[i * hidden_size..(i + 1) * hidden_size]);
        }
        self.scale_embeddings(&mut buf_a[..n]);

        // Forward through layers with Phase A/B split
        let num_layers = self.decoder_layers.len();
        let mut profiler = LayerProfiler::new(num_layers);

        for (layer_idx, layer) in self.decoder_layers.iter().enumerate() {
            let (layer_cos, layer_sin) = self.rope_cache_for_layer(layer_idx);

            if layer_idx % 2 == 0 {
                profiler.measure(|| {
                    // Phase A: QKV + RoPE (could overlap with KV transfer)
                    let phase_a = layer.forward_prefill_phase_a(
                        &buf_a, layer_cos, layer_sin, kv_cache, seq_len, start_pos,
                    )?;
                    // [HERE: inject remote KV into kv_cache before Phase B]
                    // Phase B: KV append + attention + FFN
                    layer.forward_prefill_phase_b(
                        &buf_a, phase_a, kv_cache, layer_idx, seq_len, start_pos, &mut buf_b,
                    )
                })?;
            } else {
                profiler.measure(|| {
                    let phase_a = layer.forward_prefill_phase_a(
                        &buf_b, layer_cos, layer_sin, kv_cache, seq_len, start_pos,
                    )?;
                    layer.forward_prefill_phase_b(
                        &buf_b, phase_a, kv_cache, layer_idx, seq_len, start_pos, &mut buf_a,
                    )
                })?;
            }
        }
        let layer_times = profiler.finish();
        kv_cache.advance_seq_len(seq_len);

        // Final norm + LM head (same as standard path)
        let result = if num_layers.is_multiple_of(2) { &buf_a } else { &buf_b };
        let last_hidden = &result[(seq_len - 1) * hidden_size..seq_len * hidden_size];
        let mut final_norm = std::mem::take(&mut kv_cache.ctx.decode_final_norm);
        if final_norm.len() != hidden_size {
            final_norm.resize(hidden_size, 0.0);
        }
        self.apply_final_norm(last_hidden, &mut final_norm);

        let vocab_size = self.config.vocab_size;
        let mut logits = std::mem::take(&mut kv_cache.ctx.decode_logits);
        if logits.len() != vocab_size {
            logits.resize(vocab_size, 0.0);
        }
        L::matvec(&final_norm, &self.lm_head, &mut logits)?;
        self.apply_lm_head_bias(&mut logits);

        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| HerbertError::Backend("empty logits in decomposed prefill argmax".into()))?;

        let logits_out = if return_logits { Some(logits.clone()) } else { None };

        kv_cache.ctx.prefill_buf_a = buf_a;
        kv_cache.ctx.prefill_buf_b = buf_b;
        kv_cache.ctx.decode_final_norm = final_norm;
        kv_cache.ctx.decode_logits = logits;

        Ok((logits_out, next_token, layer_times))
    }
}
