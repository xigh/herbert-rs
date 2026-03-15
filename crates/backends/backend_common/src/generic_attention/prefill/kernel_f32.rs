//! F32 KV cache prefill attention kernel (pure Rust, no ASM).

use super::*;

impl<L: LinearOps> GenericAttention<L> {
    pub(super) fn prefill_kernel_f32(
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
        let total_work = seq_len * num_kv_heads;
        let head_dim = self.head_dim;
        let q_dim = self.q_dim;
        let q_ptr = SendPtr::new(q.as_ptr());
        let attn_out_ptr = SendMutPtr::new(attn_out.as_mut_ptr());

        // Per-head F32 KV cache pointers from layer_data
        let (kph, vph) = match &kv_cache.kv.layer_data[layer_idx] {
            KvLayerData::F32 { keys_per_head, values_per_head } => (keys_per_head, values_per_head),
            _ => unreachable!("prefill_kernel_f32 called with non-F32 KV cache"),
        };
        let cached_k_head_ptrs: Vec<SendPtr<f32>> = kph
            .iter().map(|v| SendPtr::new(v.as_ptr())).collect();
        let cached_v_head_ptrs: Vec<SendPtr<f32>> = vph
            .iter().map(|v| SendPtr::new(v.as_ptr())).collect();
        let cached_k_head_ptrs_ptr = SendPtr::new(cached_k_head_ptrs.as_ptr());
        let cached_v_head_ptrs_ptr = SendPtr::new(cached_v_head_ptrs.as_ptr());

        pool.parallel_for(total_work, move |_worker_id, work_start, work_end| {
            let q_ptr = q_ptr.ptr();
            let attn_out_ptr = attn_out_ptr.ptr();

            // Pre-allocate output accumulator, reused across work units
            let mut o = vec![0.0f32; heads_per_kv * head_dim];

            for work_idx in work_start..work_end {
                let kv_h = work_idx / seq_len;
                let local_idx = work_idx % seq_len;
                // Fold-interleave for load balance
                let i = if local_idx.is_multiple_of(2) {
                    local_idx / 2
                } else {
                    seq_len - 1 - local_idx / 2
                };
                let query_pos = start_pos + i;

                let k_head_ptr: *const f32 = unsafe { (*cached_k_head_ptrs_ptr.ptr().add(kv_h)).ptr() };
                let v_head_ptr: *const f32 = unsafe { (*cached_v_head_ptrs_ptr.ptr().add(kv_h)).ptr() };

                // Reset state
                let mut l = [0.0f32; 8];
                let mut m = [f32::NEG_INFINITY; 8];
                o.fill(0.0);

                let valid_len = if bidirectional { cached_len } else { (query_pos + 1).min(cached_len) };

                for j in 0..valid_len {
                    let kv_offset = j * head_dim;

                    // Compute dot products (Q·K^T) for all Q heads in this KV group
                    for qh_local in 0..heads_per_kv {
                        let h = kv_h * heads_per_kv + qh_local;
                        let q_offset = i * q_dim + h * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            unsafe {
                                dot += *q_ptr.add(q_offset + d)
                                    * *k_head_ptr.add(kv_offset + d);
                            }
                        }
                        let s = dot * scale;

                        // Online softmax + V accumulation
                        let m_new = if s > m[qh_local] { s } else { m[qh_local] };
                        let correction = (m[qh_local] - m_new).exp();
                        let alpha = (s - m_new).exp();
                        l[qh_local] = correction * l[qh_local] + alpha;
                        m[qh_local] = m_new;

                        let o_base = qh_local * head_dim;
                        for d in 0..head_dim {
                            unsafe {
                                let v_val = *v_head_ptr.add(kv_offset + d);
                                o[o_base + d] = correction * o[o_base + d] + alpha * v_val;
                            }
                        }
                    }
                }

                // Final normalize: output = o / l
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
        .map_err(|e| HerbertError::Backend(format!("prefill attention f32 error: {}", e)))?;

        Ok(())
    }
}
