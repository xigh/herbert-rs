//! Int8 KV cache prefill attention kernel.

use super::*;

impl<L: LinearOps> GenericAttention<L> {
    pub(super) fn prefill_kernel_int8(
        &self,
        q: &[f32],
        attn_out: &mut [f32],
        kv_cache: &CpuKvCache,
        layer_idx: usize,
        seq_len: usize,
        start_pos: usize,
        cached_len: usize,
        scale: f32,
        bidirectional: bool,
    ) -> Result<()> {
        // int8 prefill path: pure Rust implementation
        let pool = global_pool();
        let num_kv_heads = self.num_kv_heads;
        let heads_per_kv = self.num_heads / num_kv_heads;
        let head_dim = self.head_dim;
        let q_dim = self.q_dim;
        let q_ptr = SendPtr::new(q.as_ptr());
        let attn_out_ptr = SendMutPtr::new(attn_out.as_mut_ptr());

        // Per-head int8 KV cache + scales from layer_data
        let (kph, vph, ksc, vsc) = match &kv_cache.kv.layer_data[layer_idx] {
            KvLayerData::INT8 { keys_per_head, values_per_head, keys_scales, values_scales } =>
                (keys_per_head, values_per_head, keys_scales, values_scales),
            _ => unreachable!("prefill_kernel_int8 called with non-INT8 KV cache"),
        };
        let cached_k_head_ptrs: Vec<SendPtr<i8>> = kph
            .iter().map(|v| SendPtr::new(v.as_ptr())).collect();
        let cached_v_head_ptrs: Vec<SendPtr<i8>> = vph
            .iter().map(|v| SendPtr::new(v.as_ptr())).collect();
        let cached_k_head_ptrs_ptr = SendPtr::new(cached_k_head_ptrs.as_ptr());
        let cached_v_head_ptrs_ptr = SendPtr::new(cached_v_head_ptrs.as_ptr());

        let k_scales_ptrs: Vec<SendPtr<f32>> = ksc
            .iter().map(|v| SendPtr::new(v.as_ptr())).collect();
        let v_scales_ptrs: Vec<SendPtr<f32>> = vsc
            .iter().map(|v| SendPtr::new(v.as_ptr())).collect();
        let k_scales_ptrs_ptr = SendPtr::new(k_scales_ptrs.as_ptr());
        let v_scales_ptrs_ptr = SendPtr::new(v_scales_ptrs.as_ptr());

        pool.parallel_for(seq_len * num_kv_heads, move |_worker_id, work_start, work_end| {
            let q_ptr = q_ptr.ptr();
            let attn_out_ptr = attn_out_ptr.ptr();

            #[cfg(all(target_arch = "x86_64", not(feature = "no-sw-prefetch")))]
            const KV_PREFETCH_DISTANCE_I8: usize = 8;

            let mut o = vec![0.0f32; heads_per_kv * head_dim];

            // 2-pass scores buffer (pre-allocated per thread)
            #[cfg(all(target_arch = "x86_64", feature = "attn-2pass"))]
            let mut scores_buf = vec![0.0f32; cached_len * 8];

            // Sampled profiling accumulators (thread-local)
            #[cfg(feature = "profile-attn-kernel")]
            let mut prof_dot_ns: u64 = 0;
            #[cfg(feature = "profile-attn-kernel")]
            let mut prof_exp_ns: u64 = 0;
            #[cfg(feature = "profile-attn-kernel")]
            let mut prof_sv_ns: u64 = 0;
            #[cfg(feature = "profile-attn-kernel")]
            let mut prof_norm_ns: u64 = 0;
            #[cfg(feature = "profile-attn-kernel")]
            let mut prof_overhead_ns: u64 = 0;
            #[cfg(feature = "profile-attn-kernel")]
            let mut prof_total_iters: u64 = 0;
            #[cfg(feature = "profile-attn-kernel")]
            let mut prof_sampled_iters: u64 = 0;

            for work_idx in work_start..work_end {
                let kv_h = work_idx / seq_len;
                let local_idx = work_idx % seq_len;
                let i = if local_idx.is_multiple_of(2) {
                    local_idx / 2
                } else {
                    seq_len - 1 - local_idx / 2
                };
                let query_pos = start_pos + i;

                // Build Q pointer array for batch dot product
                #[cfg(target_arch = "x86_64")]
                let q_ptr_array: [*const f32; 8] = {
                    let first_h = kv_h * heads_per_kv;
                    let fill = unsafe { q_ptr.add(i * q_dim + first_h * head_dim) };
                    let mut arr = [fill; 8];
                    for qh_local in 0..heads_per_kv.min(8) {
                        let h = first_h + qh_local;
                        arr[qh_local] = unsafe { q_ptr.add(i * q_dim + h * head_dim) };
                    }
                    arr
                };

                let k_head_ptr: *const i8 = unsafe { (*cached_k_head_ptrs_ptr.ptr().add(kv_h)).ptr() };
                let v_head_ptr: *const i8 = unsafe { (*cached_v_head_ptrs_ptr.ptr().add(kv_h)).ptr() };
                let k_scales_ptr: *const f32 = unsafe { (*k_scales_ptrs_ptr.ptr().add(kv_h)).ptr() };
                let v_scales_ptr: *const f32 = unsafe { (*v_scales_ptrs_ptr.ptr().add(kv_h)).ptr() };

                #[cfg(not(target_arch = "x86_64"))]
                let mut m = [f32::NEG_INFINITY; 8];
                let mut l = [0.0f32; 8];
                o.fill(0.0);

                // Fused kernel state: [m[0..8], l[0..8]] combined
                // When profiling with rdtsc, extend to 22 f32 (3 u64 TSC deltas at offset 64)
                #[cfg(target_arch = "x86_64")]
                #[cfg(feature = "profile-attn-kernel")]
                let mut ml = [0.0f32; 22];
                #[cfg(target_arch = "x86_64")]
                #[cfg(not(feature = "profile-attn-kernel"))]
                let mut ml = [0.0f32; 16];
                #[cfg(target_arch = "x86_64")]
                for q in 0..8 { ml[q] = f32::NEG_INFINITY; }

                let valid_len = if bidirectional { cached_len } else { (query_pos + 1).min(cached_len) };

                // ---- 2-pass attention path (int8, AVX-512 only) ----
                #[cfg(all(target_arch = "x86_64", feature = "attn-2pass"))]
                if heads_per_kv == 8 && has_avx512f() {
                    #[cfg(feature = "profile-attn-kernel")]
                    let _prof_2p_t0 = std::time::Instant::now();

                    // Pass 1: DOT loop — compute all scores
                    let scores = &mut scores_buf[..valid_len * 8];
                    for j in 0..valid_len {
                        let kv_offset = j * head_dim;
                        let k_scale = unsafe { *k_scales_ptr.add(j) };
                        unsafe {
                            avx512_attn_dot8_qf32_ki8(
                                q_ptr_array.as_ptr(),
                                k_head_ptr.add(kv_offset),
                                head_dim as u64,
                                scores[j * 8..].as_mut_ptr(),
                            );
                        }
                        // Apply scale * k_scale
                        let combined_scale = scale * k_scale;
                        for h in 0..8 {
                            scores[j * 8 + h] *= combined_scale;
                        }
                    }

                    #[cfg(feature = "profile-attn-kernel")]
                    let _prof_2p_t1 = std::time::Instant::now();

                    // Softmax (3-pass AVX-512, already normalized)
                    unsafe {
                        avx512_softmax_8head_inplace(
                            scores.as_mut_ptr(),
                            valid_len as u64,
                        );
                    }

                    // Fold v_scale into softmax weights
                    for j in 0..valid_len {
                        let v_scale = unsafe { *v_scales_ptr.add(j) };
                        for h in 0..8 {
                            scores[j * 8 + h] *= v_scale;
                        }
                    }

                    #[cfg(feature = "profile-attn-kernel")]
                    let _prof_2p_t2 = std::time::Instant::now();

                    // Pass 2: SV kernel (dim-major, outputs held in registers)
                    unsafe {
                        avx512_2pass_sv8_i8(
                            scores.as_ptr(),
                            v_head_ptr,
                            o.as_mut_ptr(),
                            head_dim as u64,
                            valid_len as u64,
                        );
                    }

                    #[cfg(feature = "profile-attn-kernel")]
                    {
                        let _prof_2p_t3 = std::time::Instant::now();
                        prof_dot_ns += (_prof_2p_t1 - _prof_2p_t0).as_nanos() as u64;
                        prof_exp_ns += (_prof_2p_t2 - _prof_2p_t1).as_nanos() as u64;
                        prof_sv_ns += (_prof_2p_t3 - _prof_2p_t2).as_nanos() as u64;
                        prof_sampled_iters += 1;
                        prof_total_iters += valid_len as u64;
                    }

                    // Copy o[] directly to attn_out (no 1/l — softmax already normalized)
                    for qh_local in 0..8 {
                        let h = kv_h * heads_per_kv + qh_local;
                        let out_offset = i * q_dim + h * head_dim;
                        let o_base = qh_local * head_dim;
                        for d in 0..head_dim {
                            unsafe {
                                *attn_out_ptr.add(out_offset + d) = o[o_base + d];
                            }
                        }
                    }
                    continue;
                }

                for j in 0..valid_len {
                    #[cfg(feature = "profile-attn-kernel")]
                    let _prof_sample = j & 63 == 0;
                    #[cfg(feature = "profile-attn-kernel")]
                    let _prof_t0 = if _prof_sample { Some(std::time::Instant::now()) } else { None };

                    // Prefetch K and V
                    #[cfg(all(target_arch = "x86_64", not(feature = "no-sw-prefetch")))]
                    {
                        if j + KV_PREFETCH_DISTANCE_I8 < valid_len {
                            let pf_off = (j + KV_PREFETCH_DISTANCE_I8) * head_dim;
                            unsafe {
                                core::arch::x86_64::_mm_prefetch(
                                    (k_head_ptr as *const i8).add(pf_off),
                                    core::arch::x86_64::_MM_HINT_T0,
                                );
                                core::arch::x86_64::_mm_prefetch(
                                    (v_head_ptr as *const i8).add(pf_off),
                                    core::arch::x86_64::_MM_HINT_T0,
                                );
                            }
                        }
                    }
                    let kv_offset = j * head_dim;
                    let k_scale = unsafe { *k_scales_ptr.add(j) };
                    let v_scale = unsafe { *v_scales_ptr.add(j) };

                    // Fused DOT+EXP+SV path for x86_64 with 8 heads
                    #[cfg(target_arch = "x86_64")]
                    if heads_per_kv == 8 {
                        if has_avx512f() {
                            let dot_scale = scale * k_scale;
                            unsafe {
                                avx512_fused_attn_step8_i8(
                                    q_ptr_array.as_ptr(),
                                    k_head_ptr.add(kv_offset),
                                    v_head_ptr.add(kv_offset),
                                    o.as_mut_ptr(),
                                    ml.as_mut_ptr(),
                                    head_dim as u64,
                                    dot_scale,
                                    v_scale,
                                );
                            }
                            #[cfg(feature = "profile-attn-kernel")]
                            {
                                unsafe {
                                    let tsc_ptr = ml.as_ptr().add(16) as *const u64;
                                    prof_dot_ns += *tsc_ptr;
                                    prof_exp_ns += *tsc_ptr.add(1);
                                    prof_sv_ns += *tsc_ptr.add(2);
                                }
                                prof_sampled_iters += 1;
                                prof_total_iters += 1;
                            }
                            continue;
                        }
                        // Scalar fallback: per-head dot + online softmax + SV
                        for qh_local in 0..8 {
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                unsafe {
                                    dot += *q_ptr_array[qh_local].add(d)
                                        * (*k_head_ptr.add(kv_offset + d) as f32);
                                }
                            }
                            dot *= scale * k_scale;
                            let m_new = if dot > ml[qh_local] { dot } else { ml[qh_local] };
                            let correction = (ml[qh_local] - m_new).exp();
                            let alpha = (dot - m_new).exp();
                            l[qh_local] = correction * l[qh_local] + alpha;
                            ml[qh_local] = m_new;
                            let o_base = qh_local * head_dim;
                            for d in 0..head_dim {
                                unsafe {
                                    let v_val = *v_head_ptr.add(kv_offset + d) as f32 * v_scale;
                                    o[o_base + d] = correction * o[o_base + d] + alpha * v_val;
                                }
                            }
                        }
                        #[cfg(feature = "profile-attn-kernel")]
                        { prof_total_iters += 1; }
                        continue;
                    }

                    // Compute dot products (Q·K^T)
                    let mut s = [0.0f32; 8];
                    #[cfg(target_arch = "x86_64")]
                    {
                        if has_avx512f() {
                            unsafe {
                                avx512_attn_dot8_qf32_ki8(
                                    q_ptr_array.as_ptr(),
                                    k_head_ptr.add(kv_offset),
                                    head_dim as u64,
                                    s.as_mut_ptr(),
                                );
                            }
                        } else if has_avx2_fma() {
                            unsafe {
                                avx2_attn_dot8_qf32_ki8(
                                    q_ptr_array.as_ptr(),
                                    k_head_ptr.add(kv_offset),
                                    head_dim as u64,
                                    s.as_mut_ptr(),
                                );
                            }
                        } else {
                            for qh_local in 0..heads_per_kv {
                                let mut dot = 0.0f32;
                                for d in 0..head_dim {
                                    unsafe {
                                        dot += *q_ptr_array[qh_local].add(d)
                                            * (*k_head_ptr.add(kv_offset + d) as f32);
                                    }
                                }
                                s[qh_local] = dot;
                            }
                        }
                        for qh_local in 0..heads_per_kv {
                            s[qh_local] *= k_scale * scale;
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        for qh_local in 0..heads_per_kv {
                            let h = kv_h * heads_per_kv + qh_local;
                            let q_offset = i * q_dim + h * head_dim;
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                unsafe {
                                    let k_deq = *k_head_ptr.add(kv_offset + d) as f32 * k_scale;
                                    dot += *q_ptr.add(q_offset + d) * k_deq;
                                }
                            }
                            s[qh_local] = dot * scale;
                        }
                    }

                    // --- t1: end of DOT, start of EXP ---
                    #[cfg(feature = "profile-attn-kernel")]
                    let _prof_t1 = _prof_t0.map(|_| std::time::Instant::now());

                    // Online softmax + V accumulation
                    // (x86_64 heads_per_kv==8 handled by fused kernel above via continue)
                    #[cfg(target_arch = "x86_64")]
                    {
                        // Fallback for non-8 GQA ratios (uses ml[] for m state)
                        #[cfg(feature = "profile-attn-kernel")]
                        let _prof_t2 = _prof_t1.map(|_| std::time::Instant::now());

                        for qh_local in 0..heads_per_kv {
                            let m_new = if s[qh_local] > ml[qh_local] { s[qh_local] } else { ml[qh_local] };
                            let correction = (ml[qh_local] - m_new).exp();
                            let alpha = (s[qh_local] - m_new).exp();
                            l[qh_local] = correction * l[qh_local] + alpha;
                            ml[qh_local] = m_new;

                            let o_base = qh_local * head_dim;
                            if has_avx512f() {
                                unsafe {
                                    avx512_attn_online_sv_i8(
                                        o.as_mut_ptr().add(o_base),
                                        v_head_ptr.add(kv_offset),
                                        correction,
                                        alpha * v_scale,
                                        head_dim as u64,
                                    );
                                }
                            } else if has_avx2_fma() {
                                unsafe {
                                    crate::attention_common::avx2_online_sv_i8(
                                        &mut o[o_base..o_base + head_dim],
                                        v_head_ptr.add(kv_offset),
                                        correction,
                                        alpha * v_scale,
                                        head_dim,
                                    );
                                }
                            } else {
                                let weighted = alpha * v_scale;
                                for d in 0..head_dim {
                                    unsafe {
                                        let v_val = *v_head_ptr.add(kv_offset + d) as f32;
                                        o[o_base + d] = correction * o[o_base + d] + weighted * v_val;
                                    }
                                }
                            }
                        }

                        #[cfg(feature = "profile-attn-kernel")]
                        if let (Some(t0), Some(t1), Some(_t2)) = (_prof_t0, _prof_t1, _prof_t2) {
                            let t3 = std::time::Instant::now();
                            prof_dot_ns += (t1 - t0).as_nanos() as u64;
                            let combined = (t3 - t1).as_nanos() as u64;
                            prof_exp_ns += combined / 2;
                            prof_sv_ns += combined - combined / 2;
                            prof_sampled_iters += 1;
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        #[cfg(feature = "profile-attn-kernel")]
                        let _prof_t2 = _prof_t1.map(|_| std::time::Instant::now());

                        for qh_local in 0..heads_per_kv {
                            let m_new = if s[qh_local] > m[qh_local] { s[qh_local] } else { m[qh_local] };
                            let correction = (m[qh_local] - m_new).exp();
                            let alpha = (s[qh_local] - m_new).exp();
                            l[qh_local] = correction * l[qh_local] + alpha;
                            m[qh_local] = m_new;

                            let o_base = qh_local * head_dim;
                            for d in 0..head_dim {
                                unsafe {
                                    let v_deq = *v_head_ptr.add(kv_offset + d) as f32 * v_scale;
                                    o[o_base + d] = correction * o[o_base + d] + alpha * v_deq;
                                }
                            }
                        }

                        #[cfg(feature = "profile-attn-kernel")]
                        if let (Some(t0), Some(t1), Some(_t2)) = (_prof_t0, _prof_t1, _prof_t2) {
                            let t3 = std::time::Instant::now();
                            prof_dot_ns += (t1 - t0).as_nanos() as u64;
                            let combined = (t3 - t1).as_nanos() as u64;
                            prof_exp_ns += combined / 2;
                            prof_sv_ns += combined - combined / 2;
                            prof_sampled_iters += 1;
                        }
                    }

                    #[cfg(feature = "profile-attn-kernel")]
                    { prof_total_iters += 1; }
                }

                // Transfer fused kernel l state to l[] for normalization
                // (only when AVX-512 fused kernel was used — it stores l in ml[8..16])
                #[cfg(target_arch = "x86_64")]
                if heads_per_kv == 8 && has_avx512f() {
                    for q in 0..8 { l[q] = ml[8 + q]; }
                }

                // Normalize output
                #[cfg(feature = "profile-attn-kernel")]
                let _prof_norm_t0 = std::time::Instant::now();
                for qh_local in 0..heads_per_kv {
                    let h = kv_h * heads_per_kv + qh_local;
                    let out_offset = i * q_dim + h * head_dim;
                    let inv_l = if l[qh_local] > 0.0 { 1.0 / l[qh_local] } else { 0.0 };
                    let o_base = qh_local * head_dim;
                    for d in 0..head_dim {
                        unsafe {
                            *attn_out_ptr.add(out_offset + d) = o[o_base + d] * inv_l;
                        }
                    }
                }
                #[cfg(feature = "profile-attn-kernel")]
                { prof_norm_ns += _prof_norm_t0.elapsed().as_nanos() as u64; }
            }

            // Flush thread-local profiling accumulators
            #[cfg(feature = "profile-attn-kernel")]
            crate::profiler::push_attn_kernel_breakdown(
                prof_dot_ns, prof_exp_ns, prof_sv_ns, prof_norm_ns,
                prof_overhead_ns, prof_total_iters, prof_sampled_iters,
            );
        })
        .map_err(|e| HerbertError::Backend(format!("prefill attention int8 error: {}", e)))?;

        Ok(())
    }
}
