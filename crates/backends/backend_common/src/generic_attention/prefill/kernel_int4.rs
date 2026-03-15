//! Int4 KV cache prefill attention kernel (Q4_0 group quantization, group_size=32).

use super::*;

impl<L: LinearOps> GenericAttention<L> {
    pub(super) fn prefill_kernel_int4(
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
        let pool = global_pool();
        let num_kv_heads = self.num_kv_heads;
        let heads_per_kv = self.num_heads / num_kv_heads;
        let head_dim = self.head_dim;
        let packed_per_pos = head_dim / 2;
        let num_groups = head_dim / 32;
        let q_dim = self.q_dim;
        let q_ptr = SendPtr::new(q.as_ptr());
        let attn_out_ptr = SendMutPtr::new(attn_out.as_mut_ptr());

        let (kph, vph, ksc, vsc) = match &kv_cache.kv.layer_data[layer_idx] {
            KvLayerData::INT4 { keys_per_head, values_per_head, keys_scales, values_scales } =>
                (keys_per_head, values_per_head, keys_scales, values_scales),
            _ => unreachable!("prefill_kernel_int4 called with non-INT4 KV cache"),
        };
        let cached_k_head_ptrs: Vec<SendPtr<u8>> = kph
            .iter().map(|v| SendPtr::new(v.as_ptr())).collect();
        let cached_v_head_ptrs: Vec<SendPtr<u8>> = vph
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

            let mut o = vec![0.0f32; heads_per_kv * head_dim];

            for work_idx in work_start..work_end {
                let kv_h = work_idx / seq_len;
                let local_idx = work_idx % seq_len;
                let i = if local_idx.is_multiple_of(2) {
                    local_idx / 2
                } else {
                    seq_len - 1 - local_idx / 2
                };
                let query_pos = start_pos + i;

                let k_head_ptr: *const u8 = unsafe { (*cached_k_head_ptrs_ptr.ptr().add(kv_h)).ptr() };
                let v_head_ptr: *const u8 = unsafe { (*cached_v_head_ptrs_ptr.ptr().add(kv_h)).ptr() };
                let k_scales_ptr: *const f32 = unsafe { (*k_scales_ptrs_ptr.ptr().add(kv_h)).ptr() };
                let v_scales_ptr: *const f32 = unsafe { (*v_scales_ptrs_ptr.ptr().add(kv_h)).ptr() };

                #[cfg(not(target_arch = "x86_64"))]
                let mut m = [f32::NEG_INFINITY; 8];
                let mut l = [0.0f32; 8];
                o.fill(0.0);

                #[cfg(target_arch = "x86_64")]
                let mut ml = [0.0f32; 16];
                #[cfg(target_arch = "x86_64")]
                for q_idx in 0..8 { ml[q_idx] = f32::NEG_INFINITY; }

                let valid_len = if bidirectional { cached_len } else { (query_pos + 1).min(cached_len) };

                for j in 0..valid_len {
                    let kv_offset = j * packed_per_pos;
                    // Per-group scales for this position
                    let k_gs_base = j * num_groups;
                    let v_gs_base = j * num_groups;

                    // Compute dot products (Q·K^T) with per-group scales
                    let mut s = [0.0f32; 8];
                    for qh_local in 0..heads_per_kv {
                        let h = kv_h * heads_per_kv + qh_local;
                        let q_offset = i * q_dim + h * head_dim;
                        let q_slice = unsafe { std::slice::from_raw_parts(q_ptr.add(q_offset), head_dim) };
                        let k_packed = unsafe { std::slice::from_raw_parts(k_head_ptr.add(kv_offset), packed_per_pos) };

                        let mut dot = 0.0f32;
                        for g in 0..num_groups {
                            let k_scale_g = unsafe { *k_scales_ptr.add(k_gs_base + g) };
                            let packed = &k_packed[g * 16..(g + 1) * 16];
                            let q_base = g * 32;
                            let mut group_dot = 0.0f32;
                            for ii in 0..16 {
                                let byte = packed[ii];
                                let lo = ((byte & 0x0F) as i32 - 8) as f32;
                                let hi = ((byte >> 4) as i32 - 8) as f32;
                                group_dot += q_slice[q_base + ii] * lo;
                                group_dot += q_slice[q_base + ii + 16] * hi;
                            }
                            dot += group_dot * k_scale_g;
                        }
                        s[qh_local] = dot * scale;
                    }

                    // Online softmax + V accumulation with per-group V scales
                    #[cfg(target_arch = "x86_64")]
                    {
                        for qh_local in 0..heads_per_kv {
                            let m_new = if s[qh_local] > ml[qh_local] { s[qh_local] } else { ml[qh_local] };
                            let correction = (ml[qh_local] - m_new).exp();
                            let alpha = (s[qh_local] - m_new).exp();
                            l[qh_local] = correction * l[qh_local] + alpha;
                            ml[qh_local] = m_new;

                            let o_base = qh_local * head_dim;
                            let v_packed = unsafe { std::slice::from_raw_parts(v_head_ptr.add(kv_offset), packed_per_pos) };
                            for g in 0..num_groups {
                                let v_scale_g = unsafe { *v_scales_ptr.add(v_gs_base + g) };
                                let packed = &v_packed[g * 16..(g + 1) * 16];
                                let base = g * 32;
                                let weighted = alpha * v_scale_g;
                                for ii in 0..16 {
                                    let byte = packed[ii];
                                    let lo = ((byte & 0x0F) as i32 - 8) as f32;
                                    let hi = ((byte >> 4) as i32 - 8) as f32;
                                    o[o_base + base + ii] = correction * o[o_base + base + ii] + weighted * lo;
                                    o[o_base + base + ii + 16] = correction * o[o_base + base + ii + 16] + weighted * hi;
                                }
                            }
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        for qh_local in 0..heads_per_kv {
                            let m_new = if s[qh_local] > m[qh_local] { s[qh_local] } else { m[qh_local] };
                            let correction = (m[qh_local] - m_new).exp();
                            let alpha = (s[qh_local] - m_new).exp();
                            l[qh_local] = correction * l[qh_local] + alpha;
                            m[qh_local] = m_new;

                            let o_base = qh_local * head_dim;
                            let v_packed = unsafe { std::slice::from_raw_parts(v_head_ptr.add(kv_offset), packed_per_pos) };
                            for g in 0..num_groups {
                                let v_scale_g = unsafe { *v_scales_ptr.add(v_gs_base + g) };
                                let packed = &v_packed[g * 16..(g + 1) * 16];
                                let base = g * 32;
                                let weighted = alpha * v_scale_g;
                                for ii in 0..16 {
                                    let byte = packed[ii];
                                    let lo = ((byte & 0x0F) as i32 - 8) as f32;
                                    let hi = ((byte >> 4) as i32 - 8) as f32;
                                    o[o_base + base + ii] = correction * o[o_base + base + ii] + weighted * lo;
                                    o[o_base + base + ii + 16] = correction * o[o_base + base + ii + 16] + weighted * hi;
                                }
                            }
                        }
                    }
                }

                // Normalize output
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
            }
        })
        .map_err(|e| HerbertError::Backend(format!("prefill attention int4 error: {}", e)))?;

        Ok(())
    }
}
