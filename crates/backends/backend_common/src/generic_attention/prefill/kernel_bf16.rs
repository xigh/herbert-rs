//! BF16 KV cache prefill attention kernel.

use super::*;

impl<L: LinearOps> GenericAttention<L> {
    #[allow(unused_variables)]
    pub(super) fn prefill_kernel_bf16(
        &self,
        q: &[f32],
        q_bf16: Option<&[u16]>,
        attn_out: &mut [f32],
        kv_cache: &CpuKvCache,
        layer_idx: usize,
        seq_len: usize,
        start_pos: usize,
        cached_len: usize,
        scale: f32,
        bidirectional: bool,
    ) -> Result<()> {
        let pool = global_pool();
        let num_kv_heads = self.num_kv_heads;
        let heads_per_kv = self.num_heads / num_kv_heads;
        let total_work = seq_len * num_kv_heads;
        let head_dim = self.head_dim;
        let q_dim = self.q_dim;
        let q_ptr = SendPtr::new(q.as_ptr());
        let attn_out_ptr = SendMutPtr::new(attn_out.as_mut_ptr());

        // Q BF16 pointer (None if not using DPBF16 path)
        #[cfg(target_arch = "x86_64")]
        let use_dpbf16 = q_bf16.is_some() && has_avx512bf16();
        #[cfg(target_arch = "x86_64")]
        let q_bf16_ptr = q_bf16.map(|b| SendPtr::new(b.as_ptr()));

        // Per-head KV cache pointers from layer_data
        let (kph, vph) = match &kv_cache.kv.layer_data[layer_idx] {
            KvLayerData::BF16 { keys_per_head, values_per_head } => (keys_per_head, values_per_head),
            _ => unreachable!("prefill_kernel_bf16 called with non-BF16 KV cache"),
        };
        let cached_k_head_ptrs: Vec<SendPtr<u16>> = kph
            .iter().map(|v| SendPtr::new(v.as_ptr())).collect();
        let cached_v_head_ptrs: Vec<SendPtr<u16>> = vph
            .iter().map(|v| SendPtr::new(v.as_ptr())).collect();
        let cached_k_head_ptrs_ptr = SendPtr::new(cached_k_head_ptrs.as_ptr());
        let cached_v_head_ptrs_ptr = SendPtr::new(cached_v_head_ptrs.as_ptr());

        pool.parallel_for(total_work, move |_worker_id, work_start, work_end| {
            let q_ptr = q_ptr.ptr();
            let attn_out_ptr = attn_out_ptr.ptr();

            #[cfg(all(target_arch = "x86_64", not(feature = "no-sw-prefetch")))]
            const KV_PREFETCH_DISTANCE: usize = 8;

            // Pre-allocate output accumulator, reused across work units
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
            let prof_overhead_ns: u64 = 0;
            #[cfg(feature = "profile-attn-kernel")]
            let mut prof_total_iters: u64 = 0;
            #[cfg(feature = "profile-attn-kernel")]
            let mut prof_sampled_iters: u64 = 0;

            for work_idx in work_start..work_end {
                // KV-head grouping: kv_h is the outer (slow-changing)
                // dimension so consecutive work units on the same thread
                // access the same KV head → L2 cache reuse instead of
                // thrashing across all 4 heads every iteration.
                let kv_h = work_idx / seq_len;
                let local_idx = work_idx % seq_len;
                // Fold-interleave within each KV head: pair cheap (early)
                // positions with expensive (late) for load balance.
                let i = if local_idx.is_multiple_of(2) {
                    local_idx / 2
                } else {
                    seq_len - 1 - local_idx / 2
                };
                let query_pos = start_pos + i;

                // Build Q pointer array for batch dot product (F32).
                // dot8 always reads 8 pointers — pad unused slots with
                // a valid pointer so it doesn't dereference null.
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

                // Build Q BF16 pointer array for DPBF16 kernels
                #[cfg(target_arch = "x86_64")]
                let q_bf16_ptr_array: [*const u16; 8] = if use_dpbf16 {
                    let q_bf16_raw = q_bf16_ptr.unwrap().ptr();
                    let first_h = kv_h * heads_per_kv;
                    let fill = unsafe { q_bf16_raw.add(i * q_dim + first_h * head_dim) };
                    let mut arr = [fill; 8];
                    for qh_local in 0..heads_per_kv.min(8) {
                        let h = first_h + qh_local;
                        arr[qh_local] = unsafe { q_bf16_raw.add(i * q_dim + h * head_dim) };
                    }
                    arr
                } else {
                    [std::ptr::null(); 8]
                };

                // Get per-head KV pointers
                let k_head_ptr: *const u16 = unsafe { (*cached_k_head_ptrs_ptr.ptr().add(kv_h)).ptr() };
                let v_head_ptr: *const u16 = unsafe { (*cached_v_head_ptrs_ptr.ptr().add(kv_h)).ptr() };

                // Online softmax state per Q head
                #[cfg(not(target_arch = "x86_64"))]
                let mut m = [f32::NEG_INFINITY; 8]; // running max
                let mut l = [0.0f32; 8];            // running sum of exp
                o.fill(0.0); // reset pre-allocated output accumulators

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

                // ---- 2-pass attention path (bf16, AVX-512 only) ----
                #[cfg(all(target_arch = "x86_64", feature = "attn-2pass"))]
                if heads_per_kv == 8 && has_avx512f() {
                    #[cfg(feature = "profile-attn-kernel")]
                    let _prof_2p_t0 = std::time::Instant::now();

                    // Pass 1: DOT loop — compute all scores
                    let scores = &mut scores_buf[..valid_len * 8];
                    if use_dpbf16 {
                        for j in 0..valid_len {
                            let kv_offset = j * head_dim;
                            unsafe {
                                avx512_attn_dot8_qbf16_kbf16(
                                    q_bf16_ptr_array.as_ptr(),
                                    k_head_ptr.add(kv_offset),
                                    head_dim as u64,
                                    scores[j * 8..].as_mut_ptr(),
                                );
                            }
                            for h in 0..8 {
                                scores[j * 8 + h] *= scale;
                            }
                        }
                    } else {
                        for j in 0..valid_len {
                            let kv_offset = j * head_dim;
                            unsafe {
                                avx512_attn_dot8_qf32_kbf16(
                                    q_ptr_array.as_ptr(),
                                    k_head_ptr.add(kv_offset),
                                    head_dim as u64,
                                    scores[j * 8..].as_mut_ptr(),
                                );
                            }
                            for h in 0..8 {
                                scores[j * 8 + h] *= scale;
                            }
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

                    #[cfg(feature = "profile-attn-kernel")]
                    let _prof_2p_t2 = std::time::Instant::now();

                    // Pass 2: SV kernel (dim-major, outputs held in registers)
                    unsafe {
                        avx512_2pass_sv8_bf16(
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

                    // Prefetch K and V together
                    #[cfg(all(target_arch = "x86_64", not(feature = "no-sw-prefetch")))]
                    {
                        if j + KV_PREFETCH_DISTANCE < valid_len {
                            let pf_off = (j + KV_PREFETCH_DISTANCE) * head_dim;
                            unsafe {
                                core::arch::x86_64::_mm_prefetch(
                                    (k_head_ptr as *const i8).add(pf_off * 2),
                                    core::arch::x86_64::_MM_HINT_T0,
                                );
                                core::arch::x86_64::_mm_prefetch(
                                    (v_head_ptr as *const i8).add(pf_off * 2),
                                    core::arch::x86_64::_MM_HINT_T0,
                                );
                            }
                        }
                    }
                    let kv_offset = j * head_dim;

                    // Fused DOT+EXP+SV path for x86_64 with 8 heads
                    #[cfg(target_arch = "x86_64")]
                    if heads_per_kv == 8 {
                        // DPBF16 fused path: Q BF16 + K BF16 + VDPBF16PS
                        if use_dpbf16 {
                            unsafe {
                                avx512_fused_attn_step8_bf16_dpbf16(
                                    q_bf16_ptr_array.as_ptr(),
                                    k_head_ptr.add(kv_offset),
                                    v_head_ptr.add(kv_offset),
                                    o.as_mut_ptr(),
                                    ml.as_mut_ptr(),
                                    head_dim as u64,
                                    scale,
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
                        if has_avx512f() {
                            unsafe {
                                avx512_fused_attn_step8_bf16(
                                    q_ptr_array.as_ptr(),
                                    k_head_ptr.add(kv_offset),
                                    v_head_ptr.add(kv_offset),
                                    o.as_mut_ptr(),
                                    ml.as_mut_ptr(),
                                    head_dim as u64,
                                    scale,
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
                                        * bf16_to_f32(*k_head_ptr.add(kv_offset + d));
                                }
                            }
                            dot *= scale;
                            let m_new = if dot > ml[qh_local] { dot } else { ml[qh_local] };
                            let correction = (ml[qh_local] - m_new).exp();
                            let alpha = (dot - m_new).exp();
                            l[qh_local] = correction * l[qh_local] + alpha;
                            ml[qh_local] = m_new;
                            let o_base = qh_local * head_dim;
                            for d in 0..head_dim {
                                unsafe {
                                    let v_val = bf16_to_f32(*v_head_ptr.add(kv_offset + d));
                                    o[o_base + d] = correction * o[o_base + d] + alpha * v_val;
                                }
                            }
                        }
                        #[cfg(feature = "profile-attn-kernel")]
                        { prof_total_iters += 1; }
                        continue;
                    }

                    // Compute dot products (Q·K^T) for non-8 GQA ratios
                    let mut s = [0.0f32; 8];
                    #[cfg(target_arch = "x86_64")]
                    {
                        if use_dpbf16 {
                            unsafe {
                                avx512_attn_dot8_qbf16_kbf16(
                                    q_bf16_ptr_array.as_ptr(),
                                    k_head_ptr.add(kv_offset),
                                    head_dim as u64,
                                    s.as_mut_ptr(),
                                );
                            }
                        } else if has_avx512f() {
                            unsafe {
                                avx512_attn_dot8_qf32_kbf16(
                                    q_ptr_array.as_ptr(),
                                    k_head_ptr.add(kv_offset),
                                    head_dim as u64,
                                    s.as_mut_ptr(),
                                );
                            }
                        } else if has_avx2_fma() {
                            unsafe {
                                avx2_attn_dot8_qf32_kbf16(
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
                                            * bf16_to_f32(*k_head_ptr.add(kv_offset + d));
                                    }
                                }
                                s[qh_local] = dot;
                            }
                        }
                        for qh_local in 0..heads_per_kv {
                            s[qh_local] *= scale;
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
                                    dot += *q_ptr.add(q_offset + d)
                                        * bf16_to_f32(*k_head_ptr.add(kv_offset + d));
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
                                    avx512_attn_online_sv_bf16(
                                        o.as_mut_ptr().add(o_base),
                                        v_head_ptr.add(kv_offset),
                                        correction,
                                        alpha,
                                        head_dim as u64,
                                    );
                                }
                            } else if has_avx2_fma() {
                                unsafe {
                                    crate::attention_common::avx2_online_sv_bf16(
                                        &mut o[o_base..o_base + head_dim],
                                        v_head_ptr.add(kv_offset),
                                        correction,
                                        alpha,
                                        head_dim,
                                    );
                                }
                            } else {
                                for d in 0..head_dim {
                                    unsafe {
                                        let v_val = bf16_to_f32(*v_head_ptr.add(kv_offset + d));
                                        o[o_base + d] = correction * o[o_base + d] + alpha * v_val;
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
                        // --- t2 for generic fallback ---
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
                                    let v_val = bf16_to_f32(*v_head_ptr.add(kv_offset + d));
                                    o[o_base + d] = correction * o[o_base + d] + alpha * v_val;
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
                if heads_per_kv == 8 && (has_avx512f() || use_dpbf16) {
                    for q in 0..8 { l[q] = ml[8 + q]; }
                }

                // Final normalize: output = o / l
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
        .map_err(|e| HerbertError::Backend(format!("prefill attention GQA-grouped error: {}", e)))?;

        Ok(())
    }
}
