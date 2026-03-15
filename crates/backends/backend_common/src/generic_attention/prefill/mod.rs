//! Prefill forward pass (multiple tokens) for GenericAttention.

use super::*;

mod kernel_bf16;
mod kernel_f32;
mod kernel_int4;
mod kernel_int8;

/// Intermediate state produced by Phase A, consumed by Phase B.
/// Owns the Q, K, V buffers (taken from kv_cache scratch).
pub struct PrefillPhaseAOutput {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub gate: Vec<f32>,
    pub seq_len: usize,
}

impl<L: LinearOps> GenericAttention<L> {
    /// Prefill forward (multiple tokens)
    pub fn forward_prefill(
        &self,
        x: &[f32],
        cos_cache: &[f32],
        sin_cache: &[f32],
        kv_cache: &mut CpuKvCache,
        layer_idx: usize,
        seq_len: usize,
        start_pos: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let mut q = std::mem::take(&mut kv_cache.ctx.prefill_q);
        let mut k = std::mem::take(&mut kv_cache.ctx.prefill_k);
        k.resize(seq_len * self.kv_dim, 0.0);
        let mut v = std::mem::take(&mut kv_cache.ctx.prefill_v);
        v.resize(seq_len * self.kv_dim, 0.0);

        // QKV matmul
        #[cfg(feature = "profile-attn")]
        let t_qkv = std::time::Instant::now();
        #[cfg(feature = "profile-prefill-layer")]
        let t_qkv_ppl = std::time::Instant::now();

        // When output gating is enabled, q_proj produces [seq_len, 2*q_dim].
        // De-interleave per token×head into Q [seq_len, q_dim] and gate [seq_len, q_dim].
        let mut prefill_gate: Vec<f32> = Vec::new();
        if self.has_output_gate {
            let qg_dim = self.q_dim * 2;
            q.resize(seq_len * qg_dim, 0.0);
            L::matmul(x, &self.q_proj, &mut q, seq_len)?;
            prefill_gate.resize(seq_len * self.q_dim, 0.0);
            let hd = self.head_dim;
            // IMPORTANT: Extract ALL gates first before compacting Q, because
            // compacting Q in-place overwrites gate data for lower heads.
            for tok in 0..seq_len {
                for h in 0..self.num_heads {
                    let src = tok * qg_dim + h * 2 * hd;
                    let dst = tok * self.q_dim + h * hd;
                    prefill_gate[dst..dst + hd].copy_from_slice(&q[src + hd..src + 2 * hd]);
                }
            }
            // Now compact Q values in reverse order (safe: dst <= src)
            for tok in (0..seq_len).rev() {
                for h in (0..self.num_heads).rev() {
                    let src = tok * qg_dim + h * 2 * hd;
                    let dst_q = tok * self.q_dim + h * hd;
                    q.copy_within(src..src + hd, dst_q);
                }
            }
            q.truncate(seq_len * self.q_dim);
        } else {
            q.resize(seq_len * self.q_dim, 0.0);
            L::matmul(x, &self.q_proj, &mut q, seq_len)?;
        }
        L::matmul(x, &self.k_proj, &mut k, seq_len)?;
        L::matmul(x, &self.v_proj, &mut v, seq_len)?;

        // Add biases (Qwen2-VL style) — each row gets the same bias vector
        if let Some(ref bias) = self.q_bias {
            for row in 0..seq_len {
                let base = row * self.q_dim;
                for (j, &b) in bias.iter().enumerate() {
                    q[base + j] += b;
                }
            }
        }
        if let Some(ref bias) = self.k_bias {
            for row in 0..seq_len {
                let base = row * self.kv_dim;
                for (j, &b) in bias.iter().enumerate() {
                    k[base + j] += b;
                }
            }
        }
        if let Some(ref bias) = self.v_bias {
            for row in 0..seq_len {
                let base = row * self.kv_dim;
                for (j, &b) in bias.iter().enumerate() {
                    v[base + j] += b;
                }
            }
        }
        #[cfg(feature = "profile-attn")]
        let qkv_matmul_us = t_qkv.elapsed().as_micros() as u64;
        #[cfg(feature = "profile-prefill-layer")]
        let qkv_proj_us_ppl = t_qkv_ppl.elapsed().as_micros() as u64;

        // QK norm
        #[cfg(feature = "profile-attn")]
        let t_qk_norm = std::time::Instant::now();
        #[cfg(feature = "profile-prefill-layer")]
        let t_qk_norm_ppl = std::time::Instant::now();
        // Per-head RMS norm (parallelized across positions) — only for models with QK norms
        if let (Some(ref q_norm_vec), Some(ref k_norm_vec)) = (&self.q_norm, &self.k_norm) {
            let pool = global_pool();
            let q_ptr = SendMutPtr::new(q.as_mut_ptr());
            let k_ptr = SendMutPtr::new(k.as_mut_ptr());
            let q_norm_ptr = SendPtr::new(q_norm_vec.as_ptr());
            let k_norm_ptr = SendPtr::new(k_norm_vec.as_ptr());
            let num_heads = self.num_heads;
            let num_kv_heads = self.num_kv_heads;
            let head_dim = self.head_dim;
            let q_dim = self.q_dim;
            let kv_dim = self.kv_dim;
            let rms_norm_eps = self.rms_norm_eps;
            let gemma_norm = self.use_gemma_qk_norm;

            pool.parallel_for(seq_len, move |_worker_id, pos_start, pos_end| {
                let q_ptr = q_ptr.ptr();
                let k_ptr = k_ptr.ptr();
                let q_norm_ptr = q_norm_ptr.ptr();
                let k_norm_ptr = k_norm_ptr.ptr();

                for i in pos_start..pos_end {
                    for h in 0..num_heads {
                        let offset = i * q_dim + h * head_dim;
                        let mut sum_sq = 0.0f32;
                        for d in 0..head_dim {
                            unsafe {
                                let v = *q_ptr.add(offset + d);
                                sum_sq += v * v;
                            }
                        }
                        let inv_rms = 1.0 / (sum_sq / head_dim as f32 + rms_norm_eps).sqrt();
                        for d in 0..head_dim {
                            unsafe {
                                let v = *q_ptr.add(offset + d);
                                let w = bf16_to_f32(*q_norm_ptr.add(d));
                                let w = if gemma_norm { 1.0 + w } else { w };
                                *q_ptr.add(offset + d) = v * inv_rms * w;
                            }
                        }
                    }
                    for h in 0..num_kv_heads {
                        let offset = i * kv_dim + h * head_dim;
                        let mut sum_sq = 0.0f32;
                        for d in 0..head_dim {
                            unsafe {
                                let v = *k_ptr.add(offset + d);
                                sum_sq += v * v;
                            }
                        }
                        let inv_rms = 1.0 / (sum_sq / head_dim as f32 + rms_norm_eps).sqrt();
                        for d in 0..head_dim {
                            unsafe {
                                let v = *k_ptr.add(offset + d);
                                let w = bf16_to_f32(*k_norm_ptr.add(d));
                                let w = if gemma_norm { 1.0 + w } else { w };
                                *k_ptr.add(offset + d) = v * inv_rms * w;
                            }
                        }
                    }
                }
            })
            .map_err(|e| HerbertError::Backend(format!("prefill head norm parallel error: {}", e)))?;
        }

        #[cfg(feature = "profile-attn")]
        let qk_norm_us = t_qk_norm.elapsed().as_micros() as u64;
        #[cfg(feature = "profile-prefill-layer")]
        let qk_norm_us_ppl = t_qk_norm_ppl.elapsed().as_micros() as u64;

        // RoPE
        #[cfg(feature = "profile-attn")]
        let t_rope = std::time::Instant::now();
        #[cfg(feature = "profile-prefill-layer")]
        let t_rope_ppl = std::time::Instant::now();
        if self.use_rope {
            L::apply_rope_batch(
                &mut q,
                cos_cache,
                sin_cache,
                seq_len,
                self.num_heads,
                self.head_dim,
                self.rotary_ndims,
                start_pos,
            )?;
            L::apply_rope_batch(
                &mut k,
                cos_cache,
                sin_cache,
                seq_len,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_ndims,
                start_pos,
            )?;
        }
        #[cfg(feature = "profile-attn")]
        let rope_us = t_rope.elapsed().as_micros() as u64;
        #[cfg(feature = "profile-prefill-layer")]
        let rope_us_ppl = t_rope_ppl.elapsed().as_micros() as u64;

        // Q → BF16 conversion (once per layer, after RoPE) for VDPBF16PS path
        #[cfg(target_arch = "x86_64")]
        let prefill_q_bf16 = if self.use_q_bf16 && has_avx512bf16() {
            let mut buf = std::mem::take(&mut kv_cache.ctx.prefill_q_bf16);
            buf.resize(seq_len * self.q_dim, 0);
            crate::bf16_convert::convert_f32_to_bf16(&q, &mut buf);
            Some(buf)
        } else {
            None
        };

        // KV append
        #[cfg(feature = "profile-attn")]
        let t_kv_append = std::time::Instant::now();
        #[cfg(feature = "profile-prefill-layer")]
        let t_kv_append_ppl = std::time::Instant::now();
        kv_cache.append(layer_idx, &k, &v);
        #[cfg(feature = "profile-attn")]
        let kv_append_us = t_kv_append.elapsed().as_micros() as u64;
        #[cfg(feature = "profile-prefill-layer")]
        let kv_append_us_ppl = t_kv_append_ppl.elapsed().as_micros() as u64;

        let cached_len = kv_cache.cached_len(layer_idx);
        let bidirectional = kv_cache.kv.bidirectional;

        let scale = self.attn_scale.unwrap_or(1.0 / (self.head_dim as f32).sqrt());
        let mut attn_out = std::mem::take(&mut kv_cache.ctx.prefill_attn_out);
        attn_out.resize(seq_len * self.q_dim, 0.0);

        // ---- GQA-grouped attention with fused online softmax + per-head KV layout ----
        #[cfg(feature = "profile-attn")]
        let t_kernel = std::time::Instant::now();
        #[cfg(feature = "profile-prefill-layer")]
        let t_kernel_ppl = std::time::Instant::now();

        match kv_cache.kv.kv_quant {
            herbert_core::config::KvQuantType::F32 => self.prefill_kernel_f32(
                &q, &mut attn_out, kv_cache, layer_idx,
                seq_len, start_pos, cached_len, scale, bidirectional,
            )?,
            herbert_core::config::KvQuantType::BF16 => {
                #[cfg(target_arch = "x86_64")]
                let q_bf16_ref = prefill_q_bf16.as_deref();
                #[cfg(not(target_arch = "x86_64"))]
                let q_bf16_ref: Option<&[u16]> = None;
                self.prefill_kernel_bf16(
                    &q, q_bf16_ref, &mut attn_out, kv_cache, layer_idx,
                    seq_len, start_pos, cached_len, scale, bidirectional,
                )?
            },
            herbert_core::config::KvQuantType::INT8 => self.prefill_kernel_int8(
                &q, &mut attn_out, kv_cache, layer_idx,
                seq_len, start_pos, cached_len, scale, bidirectional,
            )?,
            herbert_core::config::KvQuantType::INT4 => self.prefill_kernel_int4(
                &q, &mut attn_out, kv_cache, layer_idx,
                seq_len, start_pos, cached_len, scale, bidirectional,
            )?,
        }

        #[cfg(feature = "profile-attn")]
        let kernel_us = t_kernel.elapsed().as_micros() as u64;
        #[cfg(feature = "profile-prefill-layer")]
        let attn_kernel_us_ppl = t_kernel_ppl.elapsed().as_micros() as u64;

        // Apply sigmoid output gating before O projection
        if self.has_output_gate {
            for (a, &g) in attn_out.iter_mut().zip(prefill_gate.iter()) {
                *a *= 1.0 / (1.0 + (-g).exp());
            }
        }

        // O projection — writes into caller-provided output buffer
        #[cfg(feature = "profile-attn")]
        let t_o_proj = std::time::Instant::now();
        #[cfg(feature = "profile-prefill-layer")]
        let t_o_proj_ppl = std::time::Instant::now();
        L::matmul(&attn_out, &self.o_proj, output, seq_len)?;
        if let Some(ref o_bias) = self.o_bias {
            for pos in 0..seq_len {
                let base = pos * self.hidden_size;
                for i in 0..self.hidden_size.min(o_bias.len()) {
                    output[base + i] += o_bias[i];
                }
            }
        }

        #[cfg(feature = "profile-attn")]
        let o_proj_us = t_o_proj.elapsed().as_micros() as u64;
        #[cfg(feature = "profile-prefill-layer")]
        let o_proj_us_ppl = t_o_proj_ppl.elapsed().as_micros() as u64;

        #[cfg(feature = "profile-attn")]
        crate::profiler::push_attn(crate::profiler::AttnStepProfile {
            qkv_matmul_us,
            qk_norm_us,
            rope_us,
            kv_append_us,
            kernel_us,
            o_proj_us,
        });

        #[cfg(feature = "profile-prefill-layer")]
        crate::profiler::set_attn_sub_timings(crate::profiler::AttnSubTimings {
            qkv_proj_us: qkv_proj_us_ppl,
            qk_norm_us: qk_norm_us_ppl,
            rope_us: rope_us_ppl,
            kv_append_us: kv_append_us_ppl,
            attn_kernel_us: attn_kernel_us_ppl,
            o_proj_us: o_proj_us_ppl,
        });

        // Put buffers back for reuse in subsequent calls
        kv_cache.ctx.prefill_q = q;
        kv_cache.ctx.prefill_k = k;
        kv_cache.ctx.prefill_v = v;
        kv_cache.ctx.prefill_attn_out = attn_out;
        #[cfg(target_arch = "x86_64")]
        if let Some(buf) = prefill_q_bf16 {
            kv_cache.ctx.prefill_q_bf16 = buf;
        }

        Ok(())
    }

    /// Phase A: QKV projections + QK norm + RoPE.
    /// Independent of remote KV — can overlap with network transfer.
    pub fn forward_prefill_phase_a(
        &self,
        x: &[f32],
        cos_cache: &[f32],
        sin_cache: &[f32],
        kv_cache: &mut CpuKvCache,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<PrefillPhaseAOutput> {
        let mut q = std::mem::take(&mut kv_cache.ctx.prefill_q);
        let mut k = std::mem::take(&mut kv_cache.ctx.prefill_k);
        k.resize(seq_len * self.kv_dim, 0.0);
        let mut v = std::mem::take(&mut kv_cache.ctx.prefill_v);
        v.resize(seq_len * self.kv_dim, 0.0);

        // QKV matmul
        let mut gate: Vec<f32> = Vec::new();
        if self.has_output_gate {
            let qg_dim = self.q_dim * 2;
            q.resize(seq_len * qg_dim, 0.0);
            L::matmul(x, &self.q_proj, &mut q, seq_len)?;
            gate.resize(seq_len * self.q_dim, 0.0);
            let hd = self.head_dim;
            for tok in 0..seq_len {
                for h in 0..self.num_heads {
                    let src = tok * qg_dim + h * 2 * hd;
                    let dst = tok * self.q_dim + h * hd;
                    gate[dst..dst + hd].copy_from_slice(&q[src + hd..src + 2 * hd]);
                }
            }
            for tok in (0..seq_len).rev() {
                for h in (0..self.num_heads).rev() {
                    let src = tok * qg_dim + h * 2 * hd;
                    let dst_q = tok * self.q_dim + h * hd;
                    q.copy_within(src..src + hd, dst_q);
                }
            }
            q.truncate(seq_len * self.q_dim);
        } else {
            q.resize(seq_len * self.q_dim, 0.0);
            L::matmul(x, &self.q_proj, &mut q, seq_len)?;
        }
        L::matmul(x, &self.k_proj, &mut k, seq_len)?;
        L::matmul(x, &self.v_proj, &mut v, seq_len)?;

        // Add biases
        if let Some(ref bias) = self.q_bias {
            for row in 0..seq_len {
                let base = row * self.q_dim;
                for (j, &b) in bias.iter().enumerate() {
                    q[base + j] += b;
                }
            }
        }
        if let Some(ref bias) = self.k_bias {
            for row in 0..seq_len {
                let base = row * self.kv_dim;
                for (j, &b) in bias.iter().enumerate() {
                    k[base + j] += b;
                }
            }
        }
        if let Some(ref bias) = self.v_bias {
            for row in 0..seq_len {
                let base = row * self.kv_dim;
                for (j, &b) in bias.iter().enumerate() {
                    v[base + j] += b;
                }
            }
        }

        // QK norm
        if let (Some(ref q_norm_vec), Some(ref k_norm_vec)) = (&self.q_norm, &self.k_norm) {
            let pool = global_pool();
            let q_ptr = SendMutPtr::new(q.as_mut_ptr());
            let k_ptr = SendMutPtr::new(k.as_mut_ptr());
            let q_norm_ptr = SendPtr::new(q_norm_vec.as_ptr());
            let k_norm_ptr = SendPtr::new(k_norm_vec.as_ptr());
            let num_heads = self.num_heads;
            let num_kv_heads = self.num_kv_heads;
            let head_dim = self.head_dim;
            let q_dim = self.q_dim;
            let kv_dim = self.kv_dim;
            let rms_norm_eps = self.rms_norm_eps;
            let gemma_norm = self.use_gemma_qk_norm;

            pool.parallel_for(seq_len, move |_worker_id, pos_start, pos_end| {
                let q_ptr = q_ptr.ptr();
                let k_ptr = k_ptr.ptr();
                let q_norm_ptr = q_norm_ptr.ptr();
                let k_norm_ptr = k_norm_ptr.ptr();
                for i in pos_start..pos_end {
                    for h in 0..num_heads {
                        let offset = i * q_dim + h * head_dim;
                        let mut sum_sq = 0.0f32;
                        for d in 0..head_dim {
                            unsafe { sum_sq += (*q_ptr.add(offset + d)) * (*q_ptr.add(offset + d)); }
                        }
                        let inv_rms = 1.0 / (sum_sq / head_dim as f32 + rms_norm_eps).sqrt();
                        for d in 0..head_dim {
                            unsafe {
                                let val = *q_ptr.add(offset + d);
                                let w = bf16_to_f32(*q_norm_ptr.add(d));
                                let w = if gemma_norm { 1.0 + w } else { w };
                                *q_ptr.add(offset + d) = val * inv_rms * w;
                            }
                        }
                    }
                    for h in 0..num_kv_heads {
                        let offset = i * kv_dim + h * head_dim;
                        let mut sum_sq = 0.0f32;
                        for d in 0..head_dim {
                            unsafe { sum_sq += (*k_ptr.add(offset + d)) * (*k_ptr.add(offset + d)); }
                        }
                        let inv_rms = 1.0 / (sum_sq / head_dim as f32 + rms_norm_eps).sqrt();
                        for d in 0..head_dim {
                            unsafe {
                                let val = *k_ptr.add(offset + d);
                                let w = bf16_to_f32(*k_norm_ptr.add(d));
                                let w = if gemma_norm { 1.0 + w } else { w };
                                *k_ptr.add(offset + d) = val * inv_rms * w;
                            }
                        }
                    }
                }
            })
            .map_err(|e| HerbertError::Backend(format!("prefill head norm parallel error: {}", e)))?;
        }

        // RoPE
        if self.use_rope {
            L::apply_rope_batch(&mut q, cos_cache, sin_cache, seq_len, self.num_heads, self.head_dim, self.rotary_ndims, start_pos)?;
            L::apply_rope_batch(&mut k, cos_cache, sin_cache, seq_len, self.num_kv_heads, self.head_dim, self.rotary_ndims, start_pos)?;
        }

        Ok(PrefillPhaseAOutput { q, k, v, gate, seq_len })
    }

    /// Phase B: KV append + attention kernel + output gating + O projection.
    /// Depends on full KV cache (local + remote partitions must be present).
    pub fn forward_prefill_phase_b(
        &self,
        phase_a: PrefillPhaseAOutput,
        kv_cache: &mut CpuKvCache,
        layer_idx: usize,
        start_pos: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let PrefillPhaseAOutput { q, k, v, gate, seq_len } = phase_a;

        // KV append
        kv_cache.append(layer_idx, &k, &v);

        let cached_len = kv_cache.cached_len(layer_idx);
        let bidirectional = kv_cache.kv.bidirectional;
        let scale = self.attn_scale.unwrap_or(1.0 / (self.head_dim as f32).sqrt());

        let mut attn_out = std::mem::take(&mut kv_cache.ctx.prefill_attn_out);
        attn_out.resize(seq_len * self.q_dim, 0.0);

        // Q → BF16 conversion for phase B VDPBF16PS path
        #[cfg(target_arch = "x86_64")]
        let prefill_q_bf16_b = if self.use_q_bf16 && has_avx512bf16() {
            let mut buf = std::mem::take(&mut kv_cache.ctx.prefill_q_bf16);
            buf.resize(seq_len * self.q_dim, 0);
            crate::bf16_convert::convert_f32_to_bf16(&q, &mut buf);
            Some(buf)
        } else {
            None
        };

        // Attention kernel
        match kv_cache.kv.kv_quant {
            herbert_core::config::KvQuantType::F32 => self.prefill_kernel_f32(
                &q, &mut attn_out, kv_cache, layer_idx,
                seq_len, start_pos, cached_len, scale, bidirectional,
            )?,
            herbert_core::config::KvQuantType::BF16 => {
                #[cfg(target_arch = "x86_64")]
                let q_bf16_ref = prefill_q_bf16_b.as_deref();
                #[cfg(not(target_arch = "x86_64"))]
                let q_bf16_ref: Option<&[u16]> = None;
                self.prefill_kernel_bf16(
                    &q, q_bf16_ref, &mut attn_out, kv_cache, layer_idx,
                    seq_len, start_pos, cached_len, scale, bidirectional,
                )?
            },
            herbert_core::config::KvQuantType::INT8 => self.prefill_kernel_int8(
                &q, &mut attn_out, kv_cache, layer_idx,
                seq_len, start_pos, cached_len, scale, bidirectional,
            )?,
            herbert_core::config::KvQuantType::INT4 => self.prefill_kernel_int4(
                &q, &mut attn_out, kv_cache, layer_idx,
                seq_len, start_pos, cached_len, scale, bidirectional,
            )?,
        }

        // Output gating
        if self.has_output_gate {
            for (a, &g) in attn_out.iter_mut().zip(gate.iter()) {
                *a *= 1.0 / (1.0 + (-g).exp());
            }
        }

        // O projection
        L::matmul(&attn_out, &self.o_proj, output, seq_len)?;
        if let Some(ref o_bias) = self.o_bias {
            for pos in 0..seq_len {
                let base = pos * self.hidden_size;
                for i in 0..self.hidden_size.min(o_bias.len()) {
                    output[base + i] += o_bias[i];
                }
            }
        }

        // Put buffers back
        kv_cache.ctx.prefill_q = q;
        kv_cache.ctx.prefill_k = k;
        kv_cache.ctx.prefill_v = v;
        kv_cache.ctx.prefill_attn_out = attn_out;
        #[cfg(target_arch = "x86_64")]
        if let Some(buf) = prefill_q_bf16_b {
            kv_cache.ctx.prefill_q_bf16 = buf;
        }

        Ok(())
    }
}
