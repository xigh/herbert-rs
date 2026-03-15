//! Decode forward pass (single token) for GenericAttention.

use super::*;

impl<L: LinearOps> GenericAttention<L> {
    /// Decode forward (single token)
    pub fn forward_decode(
        &self,
        x: &[f32],
        cos: &[f32],
        sin: &[f32],
        kv_cache: &mut CpuKvCache,
        layer_idx: usize,
        pos: usize,
        output: &mut [f32],
    ) -> Result<()> {
        debug_assert_eq!(output.len(), self.hidden_size);
        let mut q = std::mem::take(&mut kv_cache.ctx.decode_q[layer_idx]);
        let mut k = std::mem::take(&mut kv_cache.ctx.decode_k[layer_idx]);
        let mut v = std::mem::take(&mut kv_cache.ctx.decode_v[layer_idx]);

        // QKV projections + bias + deinterleave
        #[cfg(feature = "profile-decode-layer")]
        let t0 = std::time::Instant::now();

        // When output gating is enabled, q_proj is fused [Q; gate] with 2*q_dim output.
        // De-interleave per head: each head has [head_dim Q, head_dim gate].
        let mut gate_buf: Vec<f32> = Vec::new();
        if self.has_output_gate {
            let qg_dim = self.q_dim * 2;
            q.resize(qg_dim, 0.0);
            k.resize(self.kv_dim, 0.0);
            v.resize(self.kv_dim, 0.0);
            L::fused_3_matvec(x, &self.q_proj, &mut q, &self.k_proj, &mut k, &self.v_proj, &mut v)?;
            gate_buf.resize(self.q_dim, 0.0);
            let hd = self.head_dim;
            for h in 0..self.num_heads {
                let src = h * 2 * hd;
                let dst = h * hd;
                gate_buf[dst..dst + hd].copy_from_slice(&q[src + hd..src + 2 * hd]);
            }
            for h in (0..self.num_heads).rev() {
                let src = h * 2 * hd;
                let dst_q = h * hd;
                q.copy_within(src..src + hd, dst_q);
            }
            q.truncate(self.q_dim);
        } else {
            q.resize(self.q_dim, 0.0);
            k.resize(self.kv_dim, 0.0);
            v.resize(self.kv_dim, 0.0);
            L::fused_3_matvec(x, &self.q_proj, &mut q, &self.k_proj, &mut k, &self.v_proj, &mut v)?;
        }

        if let Some(ref bias) = self.q_bias {
            for (o, &b) in q.iter_mut().zip(bias.iter()) {
                *o += b;
            }
        }
        if let Some(ref bias) = self.k_bias {
            for (o, &b) in k.iter_mut().zip(bias.iter()) {
                *o += b;
            }
        }
        if let Some(ref bias) = self.v_bias {
            for (o, &b) in v.iter_mut().zip(bias.iter()) {
                *o += b;
            }
        }
        #[cfg(feature = "profile-decode-layer")]
        let qkv_proj_us = t0.elapsed().as_micros() as u64;

        // QK norms
        #[cfg(feature = "profile-decode-layer")]
        let t0 = std::time::Instant::now();
        if let Some(ref norm) = self.q_norm {
            if self.use_gemma_qk_norm {
                self.head_rms_norm_gemma_inplace(&mut q, norm, self.num_heads);
            } else {
                self.head_rms_norm_inplace(&mut q, norm, self.num_heads);
            }
        }
        if let Some(ref norm) = self.k_norm {
            if self.use_gemma_qk_norm {
                self.head_rms_norm_gemma_inplace(&mut k, norm, self.num_kv_heads);
            } else {
                self.head_rms_norm_inplace(&mut k, norm, self.num_kv_heads);
            }
        }
        #[cfg(feature = "profile-decode-layer")]
        let qk_norm_us = t0.elapsed().as_micros() as u64;

        // RoPE
        #[cfg(feature = "profile-decode-layer")]
        let t0 = std::time::Instant::now();
        if self.use_rope {
            L::apply_rope_single(&mut q, cos, sin, self.num_heads, self.head_dim, self.rotary_ndims)?;
            L::apply_rope_single(&mut k, cos, sin, self.num_kv_heads, self.head_dim, self.rotary_ndims)?;
        }
        #[cfg(feature = "profile-decode-layer")]
        let rope_us = t0.elapsed().as_micros() as u64;

        #[cfg(feature = "bench-decode-snapshot")]
        if crate::snapshot::is_snapshot_active() && crate::snapshot::should_snapshot_layer(layer_idx) {
            crate::snapshot::dump_f32(&format!("layer{}_q_output", layer_idx), &q);
            crate::snapshot::dump_f32(&format!("layer{}_k_output", layer_idx), &k);
            crate::snapshot::dump_f32(&format!("layer{}_v_output", layer_idx), &v);
        }

        // Q → BF16 conversion (once per layer, after RoPE) for VDPBF16PS path
        #[cfg(target_arch = "x86_64")]
        let q_bf16_buf = if self.use_q_bf16 && has_avx512bf16() {
            let mut buf = std::mem::take(&mut kv_cache.ctx.decode_q_bf16[layer_idx]);
            buf.resize(self.q_dim, 0);
            crate::bf16_convert::convert_f32_to_bf16(&q, &mut buf);
            Some(buf)
        } else {
            None
        };

        // KV cache append + attention kernel
        #[cfg(feature = "profile-decode-layer")]
        let t0 = std::time::Instant::now();
        kv_cache.append(layer_idx, &k, &v);

        let scale = self.attn_scale.unwrap_or(1.0 / (self.head_dim as f32).sqrt());
        let mut attn_out = std::mem::take(&mut kv_cache.ctx.decode_attn_out[layer_idx]);
        attn_out.resize(self.q_dim, 0.0);

        // Runtime dispatch on layer_data format
        match &kv_cache.kv.layer_data[layer_idx] {
            KvLayerData::F32 { keys_per_head, values_per_head } => {
                let cached_len = keys_per_head.first().map_or(0, |v| v.len() / self.head_dim);
                debug_assert_eq!(cached_len, pos.saturating_add(1));
                let pool = global_pool();
                if should_parallelize_decode_heads(self.num_heads, cached_len, pool.num_workers()) {
                    let workers = decode_head_parallel_workers(pool.num_workers(), self.num_heads);
                    let mut worker_scores = std::mem::take(&mut kv_cache.ctx.decode_scores_workers[layer_idx]);
                    if worker_scores.len() < workers {
                        worker_scores.resize_with(workers, Vec::new);
                    }
                    for scores in worker_scores.iter_mut().take(workers) {
                        if scores.len() < cached_len { scores.resize(cached_len, 0.0); }
                        else { scores.truncate(cached_len); }
                    }

                    let q_ptr = SendPtr::new(q.as_ptr());
                    let attn_out_ptr = SendMutPtr::new(attn_out.as_mut_ptr());
                    let head_to_kv_head_ptr = SendPtr::new(self.head_to_kv_head.as_ptr());
                    let worker_scores_ptr = SendMutPtr::new(worker_scores.as_mut_ptr());
                    let kph_ptrs: Vec<SendPtr<f32>> = keys_per_head.iter().map(|v| SendPtr::new(v.as_ptr())).collect();
                    let vph_ptrs: Vec<SendPtr<f32>> = values_per_head.iter().map(|v| SendPtr::new(v.as_ptr())).collect();
                    let kph_ptrs_ptr = SendPtr::new(kph_ptrs.as_ptr());
                    let vph_ptrs_ptr = SendPtr::new(vph_ptrs.as_ptr());
                    let head_dim = self.head_dim;
                    let q_len = q.len();
                    let attn_out_len = attn_out.len();

                    pool.parallel_for_with_max_workers(
                        self.num_heads,
                        workers,
                        move |worker_id, head_start, head_end| {
                            let q_ptr = q_ptr.ptr();
                            let attn_out_ptr = attn_out_ptr.ptr();
                            let head_to_kv_head_ptr = head_to_kv_head_ptr.ptr();
                            let worker_scores_ptr = worker_scores_ptr.ptr();
                            let scores = unsafe { &mut *worker_scores_ptr.add(worker_id) };
                            let scores = &mut scores[..cached_len];
                            for h in head_start..head_end {
                                let q_offset = h * head_dim;
                                let out_offset = h * head_dim;
                                let kv_head_idx = unsafe { *head_to_kv_head_ptr.add(h) };
                                let cached_k_head = unsafe { std::slice::from_raw_parts((*kph_ptrs_ptr.ptr().add(kv_head_idx)).ptr(), cached_len * head_dim) };
                                let cached_v_head = unsafe { std::slice::from_raw_parts((*vph_ptrs_ptr.ptr().add(kv_head_idx)).ptr(), cached_len * head_dim) };
                                decode_head_attention_f32(
                                    unsafe { &std::slice::from_raw_parts(q_ptr, q_len)[q_offset..q_offset + head_dim] },
                                    cached_k_head,
                                    cached_v_head,
                                    head_dim,
                                    scale,
                                    scores,
                                    unsafe { &mut std::slice::from_raw_parts_mut(attn_out_ptr, attn_out_len)[out_offset..out_offset + head_dim] },
                                );
                            }
                        },
                    )
                    .map_err(|e| HerbertError::Backend(format!("decode attention f32 parallel error: {}", e)))?;
                    kv_cache.ctx.decode_scores_workers[layer_idx] = worker_scores;
                } else {
                    let mut scores = std::mem::take(&mut kv_cache.ctx.decode_scores[layer_idx]);
                    scores.resize(cached_len, 0.0);
                    for h in 0..self.num_heads {
                        let q_offset = h * self.head_dim;
                        let out_offset = h * self.head_dim;
                        let kv_head_idx = self.head_to_kv_head[h];
                        decode_head_attention_f32(
                            &q[q_offset..q_offset + self.head_dim],
                            &keys_per_head[kv_head_idx],
                            &values_per_head[kv_head_idx],
                            self.head_dim,
                            scale,
                            &mut scores[..cached_len],
                            &mut attn_out[out_offset..out_offset + self.head_dim],
                        );
                    }
                    kv_cache.ctx.decode_scores[layer_idx] = scores;
                }
            }

            KvLayerData::BF16 { keys_per_head, values_per_head } => {
                // BF16 per-head path (replaces the old flat layout)
                let cached_len = keys_per_head.first().map_or(0, |v| v.len() / self.head_dim);
                debug_assert_eq!(cached_len, pos.saturating_add(1));
                let pool = global_pool();
                if should_parallelize_decode_heads(self.num_heads, cached_len, pool.num_workers()) {
                    let workers = decode_head_parallel_workers(pool.num_workers(), self.num_heads);
                    let mut worker_scores = std::mem::take(&mut kv_cache.ctx.decode_scores_workers[layer_idx]);
                    if worker_scores.len() < workers {
                        worker_scores.resize_with(workers, Vec::new);
                    }
                    for scores in worker_scores.iter_mut().take(workers) {
                        if scores.len() < cached_len { scores.resize(cached_len, 0.0); }
                        else { scores.truncate(cached_len); }
                    }

                    let q_ptr = SendPtr::new(q.as_ptr());
                    let attn_out_ptr = SendMutPtr::new(attn_out.as_mut_ptr());
                    let head_to_kv_head_ptr = SendPtr::new(self.head_to_kv_head.as_ptr());
                    let worker_scores_ptr = SendMutPtr::new(worker_scores.as_mut_ptr());
                    let kph_ptrs: Vec<SendPtr<u16>> = keys_per_head.iter().map(|v| SendPtr::new(v.as_ptr())).collect();
                    let vph_ptrs: Vec<SendPtr<u16>> = values_per_head.iter().map(|v| SendPtr::new(v.as_ptr())).collect();
                    let kph_ptrs_ptr = SendPtr::new(kph_ptrs.as_ptr());
                    let vph_ptrs_ptr = SendPtr::new(vph_ptrs.as_ptr());
                    let head_dim = self.head_dim;
                    let q_len = q.len();
                    let attn_out_len = attn_out.len();

                    pool.parallel_for_with_max_workers(
                        self.num_heads,
                        workers,
                        move |worker_id, head_start, head_end| {
                            let q_ptr = q_ptr.ptr();
                            let attn_out_ptr = attn_out_ptr.ptr();
                            let head_to_kv_head_ptr = head_to_kv_head_ptr.ptr();
                            let worker_scores_ptr = worker_scores_ptr.ptr();
                            let scores = unsafe { &mut *worker_scores_ptr.add(worker_id) };
                            let scores = &mut scores[..cached_len];

                            for h in head_start..head_end {
                                let q_offset = h * head_dim;
                                let out_offset = h * head_dim;
                                let kv_head_idx = unsafe { *head_to_kv_head_ptr.add(h) };
                                let cached_k = unsafe { std::slice::from_raw_parts((*kph_ptrs_ptr.ptr().add(kv_head_idx)).ptr(), cached_len * head_dim) };
                                let cached_v = unsafe { std::slice::from_raw_parts((*vph_ptrs_ptr.ptr().add(kv_head_idx)).ptr(), cached_len * head_dim) };

                                // Per-head BF16: reuse decode_head_attention but with per-head layout
                                // (kv_head_idx=0, kv_dim=head_dim since data is already per-head)
                                decode_head_attention(
                                    unsafe { &std::slice::from_raw_parts(q_ptr, q_len)[q_offset..q_offset + head_dim] },
                                    cached_k,
                                    cached_v,
                                    head_dim,  // kv_dim = head_dim for per-head layout
                                    head_dim,
                                    0,  // kv_head_idx = 0 for per-head layout
                                    scale,
                                    scores,
                                    unsafe { &mut std::slice::from_raw_parts_mut(attn_out_ptr, attn_out_len)[out_offset..out_offset + head_dim] },
                                );
                            }
                        },
                    )
                    .map_err(|e| HerbertError::Backend(format!("decode attention bf16 parallel error: {}", e)))?;
                    kv_cache.ctx.decode_scores_workers[layer_idx] = worker_scores;
                } else {
                    let mut scores = std::mem::take(&mut kv_cache.ctx.decode_scores[layer_idx]);
                    scores.resize(cached_len, 0.0);
                    for h in 0..self.num_heads {
                        let q_offset = h * self.head_dim;
                        let out_offset = h * self.head_dim;
                        let kv_head_idx = self.head_to_kv_head[h];
                        decode_head_attention(
                            &q[q_offset..q_offset + self.head_dim],
                            &keys_per_head[kv_head_idx],
                            &values_per_head[kv_head_idx],
                            self.head_dim,
                            self.head_dim,
                            0,
                            scale,
                            &mut scores[..cached_len],
                            &mut attn_out[out_offset..out_offset + self.head_dim],
                        );
                    }
                    kv_cache.ctx.decode_scores[layer_idx] = scores;
                }
            }

            KvLayerData::INT8 { keys_per_head, values_per_head, keys_scales, values_scales } => {
                let cached_len = keys_per_head.first().map_or(0, |v| v.len() / self.head_dim);
                debug_assert_eq!(cached_len, pos.saturating_add(1));
                let pool = global_pool();
                if should_parallelize_decode_heads(self.num_heads, cached_len, pool.num_workers()) {
                    let workers = decode_head_parallel_workers(pool.num_workers(), self.num_heads);
                    let mut worker_scores = std::mem::take(&mut kv_cache.ctx.decode_scores_workers[layer_idx]);
                    if worker_scores.len() < workers {
                        worker_scores.resize_with(workers, Vec::new);
                    }
                    for scores in worker_scores.iter_mut().take(workers) {
                        if scores.len() < cached_len { scores.resize(cached_len, 0.0); }
                        else { scores.truncate(cached_len); }
                    }

                    let q_ptr = SendPtr::new(q.as_ptr());
                    let attn_out_ptr = SendMutPtr::new(attn_out.as_mut_ptr());
                    let head_to_kv_head_ptr = SendPtr::new(self.head_to_kv_head.as_ptr());
                    let worker_scores_ptr = SendMutPtr::new(worker_scores.as_mut_ptr());
                    let kph_ptrs: Vec<SendPtr<i8>> = keys_per_head.iter().map(|v| SendPtr::new(v.as_ptr())).collect();
                    let vph_ptrs: Vec<SendPtr<i8>> = values_per_head.iter().map(|v| SendPtr::new(v.as_ptr())).collect();
                    let ks_ptrs: Vec<SendPtr<f32>> = keys_scales.iter().map(|v| SendPtr::new(v.as_ptr())).collect();
                    let vs_ptrs: Vec<SendPtr<f32>> = values_scales.iter().map(|v| SendPtr::new(v.as_ptr())).collect();
                    let kph_ptrs_ptr = SendPtr::new(kph_ptrs.as_ptr());
                    let vph_ptrs_ptr = SendPtr::new(vph_ptrs.as_ptr());
                    let ks_ptrs_ptr = SendPtr::new(ks_ptrs.as_ptr());
                    let vs_ptrs_ptr = SendPtr::new(vs_ptrs.as_ptr());
                    let head_dim = self.head_dim;
                    let q_len = q.len();
                    let attn_out_len = attn_out.len();

                    pool.parallel_for_with_max_workers(
                        self.num_heads,
                        workers,
                        move |worker_id, head_start, head_end| {
                            let q_ptr = q_ptr.ptr();
                            let attn_out_ptr = attn_out_ptr.ptr();
                            let head_to_kv_head_ptr = head_to_kv_head_ptr.ptr();
                            let worker_scores_ptr = worker_scores_ptr.ptr();
                            let scores = unsafe { &mut *worker_scores_ptr.add(worker_id) };
                            let scores = &mut scores[..cached_len];
                            for h in head_start..head_end {
                                let q_offset = h * head_dim;
                                let out_offset = h * head_dim;
                                let kv_head_idx = unsafe { *head_to_kv_head_ptr.add(h) };
                                let cached_k_head = unsafe { &*kph_ptrs_ptr.ptr().add(kv_head_idx) };
                                let cached_v_head = unsafe { &*vph_ptrs_ptr.ptr().add(kv_head_idx) };
                                let k_scales = unsafe { &*ks_ptrs_ptr.ptr().add(kv_head_idx) };
                                let v_scales_head = unsafe { &*vs_ptrs_ptr.ptr().add(kv_head_idx) };
                                let cached_k = unsafe { std::slice::from_raw_parts(cached_k_head.ptr(), cached_len * head_dim) };
                                let cached_v = unsafe { std::slice::from_raw_parts(cached_v_head.ptr(), cached_len * head_dim) };
                                let ks = unsafe { std::slice::from_raw_parts(k_scales.ptr(), cached_len) };
                                let vs = unsafe { std::slice::from_raw_parts(v_scales_head.ptr(), cached_len) };
                                decode_head_attention_int8(
                                    unsafe { &std::slice::from_raw_parts(q_ptr, q_len)[q_offset..q_offset + head_dim] },
                                    cached_k,
                                    cached_v,
                                    ks,
                                    vs,
                                    head_dim,
                                    scale,
                                    scores,
                                    unsafe { &mut std::slice::from_raw_parts_mut(attn_out_ptr, attn_out_len)[out_offset..out_offset + head_dim] },
                                );
                            }
                        },
                    )
                    .map_err(|e| HerbertError::Backend(format!("decode attention int8 parallel error: {}", e)))?;
                    kv_cache.ctx.decode_scores_workers[layer_idx] = worker_scores;
                } else {
                    let mut scores = std::mem::take(&mut kv_cache.ctx.decode_scores[layer_idx]);
                    scores.resize(cached_len, 0.0);
                    for h in 0..self.num_heads {
                        let q_offset = h * self.head_dim;
                        let out_offset = h * self.head_dim;
                        let kv_head_idx = self.head_to_kv_head[h];
                        decode_head_attention_int8(
                            &q[q_offset..q_offset + self.head_dim],
                            &keys_per_head[kv_head_idx],
                            &values_per_head[kv_head_idx],
                            &keys_scales[kv_head_idx],
                            &values_scales[kv_head_idx],
                            self.head_dim,
                            scale,
                            &mut scores[..cached_len],
                            &mut attn_out[out_offset..out_offset + self.head_dim],
                        );
                    }
                    kv_cache.ctx.decode_scores[layer_idx] = scores;
                }
            }

            KvLayerData::INT4 { keys_per_head, values_per_head, keys_scales, values_scales } => {
                let packed_per_pos = self.head_dim / 2;
                let num_groups = self.head_dim / 32;
                let cached_len = keys_scales.first().map_or(0, |v| {
                    if num_groups == 0 { 0 } else { v.len() / num_groups }
                });
                debug_assert_eq!(cached_len, pos.saturating_add(1));
                let pool = global_pool();
                if should_parallelize_decode_heads(self.num_heads, cached_len, pool.num_workers()) {
                    let workers = decode_head_parallel_workers(pool.num_workers(), self.num_heads);
                    let mut worker_scores = std::mem::take(&mut kv_cache.ctx.decode_scores_workers[layer_idx]);
                    if worker_scores.len() < workers {
                        worker_scores.resize_with(workers, Vec::new);
                    }
                    for scores in worker_scores.iter_mut().take(workers) {
                        if scores.len() < cached_len { scores.resize(cached_len, 0.0); }
                        else { scores.truncate(cached_len); }
                    }

                    let q_ptr = SendPtr::new(q.as_ptr());
                    let attn_out_ptr = SendMutPtr::new(attn_out.as_mut_ptr());
                    let head_to_kv_head_ptr = SendPtr::new(self.head_to_kv_head.as_ptr());
                    let worker_scores_ptr = SendMutPtr::new(worker_scores.as_mut_ptr());
                    let kph_ptrs: Vec<SendPtr<u8>> = keys_per_head.iter().map(|v| SendPtr::new(v.as_ptr())).collect();
                    let vph_ptrs: Vec<SendPtr<u8>> = values_per_head.iter().map(|v| SendPtr::new(v.as_ptr())).collect();
                    let ks_ptrs: Vec<SendPtr<f32>> = keys_scales.iter().map(|v| SendPtr::new(v.as_ptr())).collect();
                    let vs_ptrs: Vec<SendPtr<f32>> = values_scales.iter().map(|v| SendPtr::new(v.as_ptr())).collect();
                    let kph_ptrs_ptr = SendPtr::new(kph_ptrs.as_ptr());
                    let vph_ptrs_ptr = SendPtr::new(vph_ptrs.as_ptr());
                    let ks_ptrs_ptr = SendPtr::new(ks_ptrs.as_ptr());
                    let vs_ptrs_ptr = SendPtr::new(vs_ptrs.as_ptr());
                    let head_dim = self.head_dim;
                    let q_len = q.len();
                    let attn_out_len = attn_out.len();

                    pool.parallel_for_with_max_workers(
                        self.num_heads,
                        workers,
                        move |worker_id, head_start, head_end| {
                            let q_ptr = q_ptr.ptr();
                            let attn_out_ptr = attn_out_ptr.ptr();
                            let head_to_kv_head_ptr = head_to_kv_head_ptr.ptr();
                            let worker_scores_ptr = worker_scores_ptr.ptr();
                            let scores = unsafe { &mut *worker_scores_ptr.add(worker_id) };
                            let scores = &mut scores[..cached_len];
                            for h in head_start..head_end {
                                let q_offset = h * head_dim;
                                let out_offset = h * head_dim;
                                let kv_head_idx = unsafe { *head_to_kv_head_ptr.add(h) };
                                let cached_k_head = unsafe { &*kph_ptrs_ptr.ptr().add(kv_head_idx) };
                                let cached_v_head = unsafe { &*vph_ptrs_ptr.ptr().add(kv_head_idx) };
                                let k_scales_head = unsafe { &*ks_ptrs_ptr.ptr().add(kv_head_idx) };
                                let v_scales_head = unsafe { &*vs_ptrs_ptr.ptr().add(kv_head_idx) };
                                let cached_k = unsafe { std::slice::from_raw_parts(cached_k_head.ptr(), cached_len * packed_per_pos) };
                                let cached_v = unsafe { std::slice::from_raw_parts(cached_v_head.ptr(), cached_len * packed_per_pos) };
                                let num_groups_inner = head_dim / 32;
                                let ks = unsafe { std::slice::from_raw_parts(k_scales_head.ptr(), cached_len * num_groups_inner) };
                                let vs = unsafe { std::slice::from_raw_parts(v_scales_head.ptr(), cached_len * num_groups_inner) };
                                decode_head_attention_int4(
                                    unsafe { &std::slice::from_raw_parts(q_ptr, q_len)[q_offset..q_offset + head_dim] },
                                    cached_k,
                                    cached_v,
                                    ks,
                                    vs,
                                    head_dim,
                                    scale,
                                    scores,
                                    unsafe { &mut std::slice::from_raw_parts_mut(attn_out_ptr, attn_out_len)[out_offset..out_offset + head_dim] },
                                );
                            }
                        },
                    )
                    .map_err(|e| HerbertError::Backend(format!("decode attention int4 parallel error: {}", e)))?;
                    kv_cache.ctx.decode_scores_workers[layer_idx] = worker_scores;
                } else {
                    let mut scores = std::mem::take(&mut kv_cache.ctx.decode_scores[layer_idx]);
                    scores.resize(cached_len, 0.0);
                    for h in 0..self.num_heads {
                        let q_offset = h * self.head_dim;
                        let out_offset = h * self.head_dim;
                        let kv_head_idx = self.head_to_kv_head[h];
                        decode_head_attention_int4(
                            &q[q_offset..q_offset + self.head_dim],
                            &keys_per_head[kv_head_idx],
                            &values_per_head[kv_head_idx],
                            &keys_scales[kv_head_idx],
                            &values_scales[kv_head_idx],
                            self.head_dim,
                            scale,
                            &mut scores[..cached_len],
                            &mut attn_out[out_offset..out_offset + self.head_dim],
                        );
                    }
                    kv_cache.ctx.decode_scores[layer_idx] = scores;
                }
            }
        }

        #[cfg(feature = "profile-decode-layer")]
        let kv_cache_attn_us = t0.elapsed().as_micros() as u64;

        // Apply sigmoid output gating: attn_out *= sigmoid(gate)
        #[cfg(feature = "profile-decode-layer")]
        let t0 = std::time::Instant::now();
        if self.has_output_gate {
            for (a, &g) in attn_out.iter_mut().zip(gate_buf.iter()) {
                *a *= 1.0 / (1.0 + (-g).exp());
            }
        }
        #[cfg(feature = "profile-decode-layer")]
        let output_gate_us = t0.elapsed().as_micros() as u64;

        #[cfg(feature = "bench-decode-snapshot")]
        if crate::snapshot::is_snapshot_active() && crate::snapshot::should_snapshot_layer(layer_idx) {
            crate::snapshot::dump_f32(&format!("layer{}_o_proj_input", layer_idx), &attn_out);
        }

        #[cfg(feature = "profile-decode-layer")]
        let t0 = std::time::Instant::now();
        L::matvec(&attn_out, &self.o_proj, output)?;
        if let Some(ref o_bias) = self.o_bias {
            for i in 0..output.len().min(o_bias.len()) {
                output[i] += o_bias[i];
            }
        }
        #[cfg(feature = "profile-decode-layer")]
        let o_proj_us = t0.elapsed().as_micros() as u64;

        #[cfg(feature = "profile-decode-layer")]
        crate::profiler::set_decode_block_sub_timings(
            crate::profiler::DecodeBlockSubTimings::Attn {
                qkv_proj_us,
                qk_norm_us,
                rope_us,
                kv_cache_attn_us,
                output_gate_us,
                o_proj_us,
            },
        );

        kv_cache.ctx.decode_q[layer_idx] = q;
        kv_cache.ctx.decode_k[layer_idx] = k;
        kv_cache.ctx.decode_v[layer_idx] = v;
        kv_cache.ctx.decode_attn_out[layer_idx] = attn_out;
        #[cfg(target_arch = "x86_64")]
        if let Some(buf) = q_bf16_buf {
            kv_cache.ctx.decode_q_bf16[layer_idx] = buf;
        }

        Ok(())
    }
}
