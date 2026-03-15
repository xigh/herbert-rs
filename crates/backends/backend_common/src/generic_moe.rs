//! MoE (Mixture of Experts) support for generic CPU backends.

use std::sync::Mutex;

use herbert_core::error::Result;
use herbert_core::moe_stats;

use crate::expert_pool::ExpertPool;
use crate::generic_mlp::GenericMLP;
use crate::linear_ops::LinearOps;
#[cfg(feature = "moe-v6")]
use crate::thread_pool::{global_pool, SendMutPtr, SendPtr};
#[cfg(feature = "moe-v6")]
use herbert_core::error::HerbertError;

/// A single router + pool of expert MLPs.
pub struct GenericMoE<L: LinearOps> {
    /// Router gate weights [num_experts, hidden_size], quantized like other projections.
    pub gate: L::Weight,
    /// Expert pool (Mutex for interior mutability through Arc<GenericModel>).
    pub expert_pool: Mutex<ExpertPool<L>>,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub norm_topk_prob: bool,
    /// Layer index in the model (set by loader).
    pub layer_id: usize,
    /// Optional per-layer expert bias added after MoE output (LFM2-8B-A1B).
    pub expert_bias: Option<Vec<f32>>,
    /// Shared expert MLP (always active, output added to MoE output).
    pub shared_expert: Option<GenericMLP<L>>,
    /// Shared expert gate weight [hidden_size] — sigmoid gating.
    pub shared_expert_gate: Option<Vec<f32>>,
}

/// Unified FFN enum: either a dense MLP or a MoE layer.
pub enum GenericFFN<L: LinearOps> {
    Dense(GenericMLP<L>),
    MoE(GenericMoE<L>),
}

impl<L: LinearOps> GenericFFN<L> {
    /// Whether this FFN is MoE (has experts to distribute).
    #[inline]
    pub fn is_moe(&self) -> bool {
        matches!(self, GenericFFN::MoE(_))
    }

    /// Get a reference to the inner MoE (panics if dense).
    #[inline]
    pub fn as_moe(&self) -> &GenericMoE<L> {
        match self {
            GenericFFN::MoE(moe) => moe,
            GenericFFN::Dense(_) => panic!("as_moe called on Dense FFN"),
        }
    }

    /// The intermediate size used by this FFN (for buffer sizing).
    #[inline]
    pub fn intermediate_size(&self) -> usize {
        match self {
            GenericFFN::Dense(mlp) => mlp.intermediate_size,
            GenericFFN::MoE(moe) => moe.moe_intermediate_size,
        }
    }

    /// Single-token forward (decode).
    pub fn forward_decode(
        &self,
        x: &[f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        output: &mut [f32],
        moe_router: &mut Vec<f32>,
        moe_expert_out: &mut Vec<f32>,
    ) -> Result<()> {
        match self {
            GenericFFN::Dense(mlp) => mlp.forward_decode(x, gate_buf, up_buf, output),
            GenericFFN::MoE(moe) => moe.forward_decode(x, gate_buf, up_buf, output, moe_router, moe_expert_out),
        }
    }

    /// Single-token forward (decode) with pre-computed quantized input.
    /// Skips redundant prequantize_input inside MoE/MLP when pq was
    /// already computed during fused RMSNorm+quantize.
    pub fn forward_decode_with_pq(
        &self,
        x: &[f32],
        pq: Option<crate::linear_ops::PreQuantizedInput>,
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        output: &mut [f32],
        moe_router: &mut Vec<f32>,
        moe_expert_out: &mut Vec<f32>,
    ) -> Result<()> {
        match self {
            GenericFFN::Dense(mlp) => mlp.forward_decode(x, gate_buf, up_buf, output),
            GenericFFN::MoE(moe) => moe.forward_decode_with_pq(x, pq, gate_buf, up_buf, output, moe_router, moe_expert_out),
        }
    }

    /// Dump snapshot for FFN (dense MLP or MoE).
    #[cfg(feature = "bench-decode-snapshot")]
    pub fn dump_snapshot(&self, prefix: &str) {
        match self {
            GenericFFN::Dense(mlp) => {
                use crate::snapshot;
                eprintln!("[SNAPSHOT]   {} FFN: Dense MLP", prefix);
                snapshot::dump_weight::<L>(&format!("{}_ffn_gate_proj", prefix), &mlp.gate_proj);
                snapshot::dump_weight::<L>(&format!("{}_ffn_up_proj", prefix), &mlp.up_proj);
                snapshot::dump_weight::<L>(&format!("{}_ffn_down_proj", prefix), &mlp.down_proj);
            }
            GenericFFN::MoE(moe) => {
                moe.dump_snapshot(prefix);
            }
        }
    }

    /// Multi-token forward (prefill).
    pub fn forward_prefill(
        &self,
        x: &[f32],
        seq_len: usize,
        kv_cache: &mut crate::kv_cache::CpuKvCache,
        output: &mut [f32],
    ) -> Result<()> {
        match self {
            GenericFFN::Dense(mlp) => {
                let mut gate = std::mem::take(&mut kv_cache.ctx.prefill_mlp_gate);
                gate.resize(seq_len * mlp.intermediate_size, 0.0);
                let mut up = std::mem::take(&mut kv_cache.ctx.prefill_mlp_up);
                up.resize(seq_len * mlp.intermediate_size, 0.0);
                let result = mlp.forward_prefill(x, seq_len, &mut gate, &mut up, output);
                kv_cache.ctx.prefill_mlp_gate = gate;
                kv_cache.ctx.prefill_mlp_up = up;
                result
            }
            GenericFFN::MoE(moe) => moe.forward_prefill(x, seq_len, kv_cache, output),
        }
    }
}

impl<L: LinearOps> GenericMoE<L> {
    /// Dump snapshot data for MoE (router, shared expert, active experts).
    #[cfg(feature = "bench-decode-snapshot")]
    pub fn dump_snapshot(&self, prefix: &str) {
        use crate::snapshot;
        eprintln!("[SNAPSHOT]   {} FFN: MoE ({} experts, {} active)",
            prefix, self.num_experts, self.num_experts_per_tok);

        // Router
        snapshot::dump_weight::<L>(&format!("{}_moe_gate", prefix), &self.gate);

        // Shared expert
        if let Some(ref shared) = self.shared_expert {
            snapshot::dump_weight::<L>(&format!("{}_shared_gate_proj", prefix), &shared.gate_proj);
            snapshot::dump_weight::<L>(&format!("{}_shared_up_proj", prefix), &shared.up_proj);
            snapshot::dump_weight::<L>(&format!("{}_shared_down_proj", prefix), &shared.down_proj);
        }
        if let Some(ref gate_w) = self.shared_expert_gate {
            snapshot::dump_f32(&format!("{}_shared_expert_gate_w", prefix), gate_w);
        }

        // Expert pool — dump all loaded experts
        let pool = self.expert_pool.lock().unwrap();
        let experts = pool.all_experts();
        for (expert_id, expert) in experts.iter().enumerate() {
            if let Some(ref mlp) = expert {
                snapshot::dump_weight::<L>(
                    &format!("{}_expert{}_gate_proj", prefix, expert_id),
                    &mlp.gate_proj,
                );
                snapshot::dump_weight::<L>(
                    &format!("{}_expert{}_up_proj", prefix, expert_id),
                    &mlp.up_proj,
                );
                snapshot::dump_weight::<L>(
                    &format!("{}_expert{}_down_proj", prefix, expert_id),
                    &mlp.down_proj,
                );
            }
        }
        eprintln!("[SNAPSHOT]   {} MoE: dumped {} experts", prefix, experts.iter().filter(|e| e.is_some()).count());

        // MoE config
        snapshot::dump_usize(&format!("{}_num_experts", prefix), self.num_experts);
        snapshot::dump_usize(&format!("{}_num_experts_per_tok", prefix), self.num_experts_per_tok);
        snapshot::dump_usize(&format!("{}_moe_intermediate_size", prefix), self.moe_intermediate_size);
    }

    /// Route a single token: compute activation (softmax or sigmoid), apply expert_bias
    /// for selection if present, and return top-K (expert_id, weight) pairs.
    ///
    /// LFM2: sigmoid activation, expert_bias added for selection only, unbiased weights returned.
    /// Qwen: softmax activation, no bias.
    fn route_token(&self, router_buf: &mut [f32], top_k_buf: &mut [(usize, f32); 8]) -> usize {
        let k = self.num_experts_per_tok;
        let ne = self.num_experts;
        if let Some(ref bias) = self.expert_bias {
            // LFM2: sigmoid routing with expert bias for selection
            sigmoid_inplace(router_buf);
            let mut biased = [0.0f32; 128];
            for i in 0..ne { biased[i] = router_buf[i] + bias[i]; }
            let n_sel = top_k_indices(&biased[..ne], k, top_k_buf);
            // Replace biased weights with unbiased sigmoid weights
            for i in 0..n_sel {
                top_k_buf[i].1 = router_buf[top_k_buf[i].0];
            }
            n_sel
        } else {
            // Qwen: softmax routing
            softmax_inplace(router_buf);
            top_k_indices(router_buf, k, top_k_buf)
        }
    }

    /// Add shared expert contribution to MoE output (decode, single token).
    fn add_shared_expert(&self, x: &[f32], output: &mut [f32]) -> Result<()> {
        if let Some(ref shared) = self.shared_expert {
            let inter = shared.intermediate_size;
            let h = self.hidden_size;
            let mut gate = vec![0.0f32; inter];
            let mut up = vec![0.0f32; inter];
            let mut shared_out = vec![0.0f32; h];
            shared.forward_decode(x, &mut gate, &mut up, &mut shared_out)?;

            if let Some(ref gate_w) = self.shared_expert_gate {
                // sigmoid gating: g = sigmoid(gate_w · x)
                let dot: f32 = x.iter().zip(gate_w.iter()).map(|(a, b)| a * b).sum();
                let g = 1.0 / (1.0 + (-dot).exp());
                for i in 0..h {
                    output[i] += g * shared_out[i];
                }
            } else {
                for i in 0..h {
                    output[i] += shared_out[i];
                }
            }
        }
        Ok(())
    }

    /// Add shared expert contribution to MoE output (prefill, multi-token).
    fn add_shared_expert_prefill(&self, x: &[f32], seq_len: usize, output: &mut [f32]) -> Result<()> {
        if let Some(ref shared) = self.shared_expert {
            let inter = shared.intermediate_size;
            let h = self.hidden_size;
            let mut gate = vec![0.0f32; seq_len * inter];
            let mut up = vec![0.0f32; seq_len * inter];
            let mut shared_out = vec![0.0f32; seq_len * h];
            shared.forward_prefill(x, seq_len, &mut gate, &mut up, &mut shared_out)?;

            if let Some(ref gate_w) = self.shared_expert_gate {
                for t in 0..seq_len {
                    let token_x = &x[t * h..(t + 1) * h];
                    let dot: f32 = token_x.iter().zip(gate_w.iter()).map(|(a, b)| a * b).sum();
                    let g = 1.0 / (1.0 + (-dot).exp());
                    let out_t = &mut output[t * h..(t + 1) * h];
                    let shared_t = &shared_out[t * h..(t + 1) * h];
                    for i in 0..h {
                        out_t[i] += g * shared_t[i];
                    }
                }
            } else {
                for i in 0..seq_len * h {
                    output[i] += shared_out[i];
                }
            }
        }
        Ok(())
    }

    /// Compute router logits and return top-K (expert_id, weight) pairs.
    /// Does NOT run any experts. Used by distributed decode to split routing
    /// from expert computation.
    pub fn compute_routing(
        &self,
        x: &[f32],
        router_buf: &mut Vec<f32>,
    ) -> Vec<(usize, f32)> {
        let ne = self.num_experts;

        // Router logits
        router_buf.resize(ne, 0.0);
        let pq = L::prequantize_input(x);
        if let Some(ref pq) = pq {
            L::matvec_pq_st(pq, x, &self.gate, router_buf).expect("router matvec failed");
        } else {
            L::matvec_float_dequant(x, &self.gate, router_buf).expect("router matvec failed");
        }

        // Activation + top-K
        let mut top_k_buf = [(0usize, 0.0f32); 8];
        let n_sel = self.route_token(router_buf, &mut top_k_buf);

        // Normalize weights
        let mut result = Vec::with_capacity(n_sel);
        if self.norm_topk_prob {
            let sum: f32 = top_k_buf[..n_sel].iter().map(|&(_, w)| w).sum();
            if sum > 0.0 {
                for i in 0..n_sel {
                    result.push((top_k_buf[i].0, top_k_buf[i].1 / sum));
                }
            } else {
                for i in 0..n_sel {
                    result.push(top_k_buf[i]);
                }
            }
        } else {
            for i in 0..n_sel {
                result.push(top_k_buf[i]);
            }
        }
        result
    }

    /// Compute a subset of experts and return weighted partial sum.
    /// Only runs experts whose IDs appear in `assignments`.
    /// `output` must be zeroed by the caller.
    pub fn compute_experts_partial(
        &self,
        x: &[f32],
        assignments: &[(usize, f32)],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        output: &mut [f32],
    ) -> Result<()> {
        let h = self.hidden_size;
        let mut pool = self.expert_pool.lock().unwrap_or_else(|e| e.into_inner());
        pool.ensure_loaded(&[])?;
        let mut expert_out = vec![0.0f32; h];

        for &(expert_id, weight) in assignments {
            gate_buf.iter_mut().for_each(|v| *v = 0.0);
            up_buf.iter_mut().for_each(|v| *v = 0.0);
            expert_out.iter_mut().for_each(|v| *v = 0.0);

            let expert = pool.get(expert_id);
            if L::PARALLEL_EXPERTS {
                expert.forward_decode_st(x, gate_buf, up_buf, &mut expert_out)?;
            } else {
                expert.forward_decode(x, gate_buf, up_buf, &mut expert_out)?;
            }

            for i in 0..h {
                output[i] += weight * expert_out[i];
            }
        }
        Ok(())
    }

    /// Add shared expert contribution to output (public wrapper for distributed decode).
    pub fn add_shared_expert_to(&self, x: &[f32], output: &mut [f32]) -> Result<()> {
        self.add_shared_expert(x, output)
    }

    /// Single-token MoE forward.
    pub fn forward_decode(
        &self,
        x: &[f32],
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        output: &mut [f32],
        router_buf: &mut Vec<f32>,
        expert_out_buf: &mut Vec<f32>,
    ) -> Result<()> {
        self.forward_decode_with_pq(x, None, gate_buf, up_buf, output, router_buf, expert_out_buf)
    }

    /// Single-token MoE forward with optional pre-computed quantized input.
    ///
    /// 1. Compute router logits via dot products with gate weights
    /// 2. Softmax over all experts
    /// 3. Select top-K experts
    /// 4. Optionally renormalize selected weights
    /// 5. Weighted sum of expert outputs
    pub fn forward_decode_with_pq(
        &self,
        x: &[f32],
        pq_in: Option<crate::linear_ops::PreQuantizedInput>,
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
        output: &mut [f32],
        router_buf: &mut Vec<f32>,
        expert_out_buf: &mut Vec<f32>,
    ) -> Result<()> {
        #[cfg(feature = "profile-moe-decode")]
        let t_total = std::time::Instant::now();

        let h = self.hidden_size;
        let _k = self.num_experts_per_tok;
        let ne = self.num_experts;

        // Use caller-provided pre-quantized input (from fused RMSNorm+quantize),
        // or quantize now if not provided.
        let pq = pq_in.or_else(|| L::prequantize_input(x));

        // 1. Router logits: use pre-quantized input if available, else float-dequant.
        router_buf.resize(ne, 0.0);
        if let Some(ref pq) = pq {
            L::matvec_pq_st(pq, x, &self.gate, router_buf)?;
        } else {
            L::matvec_float_dequant(x, &self.gate, router_buf)?;
        }

        // 2-3. Activation + top-K selection (sigmoid+bias for LFM2, softmax for Qwen)
        let mut top_k_buf = [(0usize, 0.0f32); 8];
        let n_sel = self.route_token(router_buf, &mut top_k_buf);

        // Lock expert pool, record usage, and sort experts by frequency (hottest first).
        // Hot-expert-first ordering improves L3 cache residency: if the same experts
        // are selected across consecutive tokens, the first-processed expert's weights
        // are most likely still in L3 from the previous token.
        let mut pool = self.expert_pool.lock().unwrap_or_else(|e| e.into_inner());
        pool.ensure_loaded(&[])?;
        pool.record_and_sort_by_frequency(&mut top_k_buf, n_sel);

        // 4. Renormalize weights (after sorting so weights[i] matches sorted order)
        let mut weights = [0.0f32; 8];
        if self.norm_topk_prob {
            let sum: f32 = top_k_buf[..n_sel].iter().map(|&(_, w)| w).sum();
            if sum > 0.0 {
                for i in 0..n_sel { weights[i] = top_k_buf[i].1 / sum; }
            } else {
                for i in 0..n_sel { weights[i] = top_k_buf[i].1; }
            }
        } else {
            for i in 0..n_sel { weights[i] = top_k_buf[i].1; }
        }

        #[cfg(feature = "profile-moe-decode")]
        let router_us = t_total.elapsed().as_micros() as u64;

        // 5. Weighted sum of expert outputs
        for o in output.iter_mut() {
            *o = 0.0;
        }

        #[cfg(feature = "moe-v6")]
        if L::PARALLEL_EXPERTS && n_sel > 1 {
            // Parallel path: run experts concurrently via thread pool
            let inter = self.moe_intermediate_size;
            let mut expert_outputs = vec![0.0f32; n_sel * h];

            // Collect expert IDs into a stack array for SendPtr
            let mut expert_ids = [0usize; 8];
            for i in 0..n_sel { expert_ids[i] = top_k_buf[i].0; }

            // Wrap pq in Arc for sharing across threads (already computed above).
            let pq = pq.map(std::sync::Arc::new);

            let x_ptr = SendPtr::new(x.as_ptr());
            let expert_out_ptr = SendMutPtr::new(expert_outputs.as_mut_ptr());
            let experts_opt_ptr = SendPtr::new(pool.experts_option_ptr());
            let expert_ids_ptr = SendPtr::new(expert_ids.as_ptr());

            // Seed hardware prefetcher: touch head of each expert's gate_proj
            // during parallel_for setup overhead (~2-5µs window).
            for i in 0..n_sel {
                let expert_id = expert_ids[i];
                if let Some(ref expert) = unsafe { &*pool.experts_option_ptr().add(expert_id) } {
                    L::prefetch_weight(&expert.gate_proj);
                }
            }

            #[cfg(feature = "profile-moe-decode")]
            let t_dispatch = std::time::Instant::now();

            let thread_pool = global_pool();
            thread_pool.parallel_for(n_sel, move |_, start, end| {
                // Per-worker scratch buffers
                let mut gate = vec![0.0f32; inter];
                let mut up = vec![0.0f32; inter];
                let mut out_buf = vec![0.0f32; h];

                for idx in start..end {
                    gate.iter_mut().for_each(|v| *v = 0.0);
                    up.iter_mut().for_each(|v| *v = 0.0);
                    out_buf.iter_mut().for_each(|v| *v = 0.0);

                    let expert_id = unsafe { *expert_ids_ptr.ptr().add(idx) };
                    let x_slice = unsafe { std::slice::from_raw_parts(x_ptr.ptr(), h) };
                    let expert = unsafe { &*experts_opt_ptr.ptr().add(expert_id) }
                        .as_ref()
                        .expect("expert must be loaded after ensure_loaded");

                    // Prefetch down_proj while gate+up computation runs.
                    L::prefetch_weight(&expert.down_proj);

                    #[cfg(feature = "profile-moe-decode")]
                    let t0 = std::time::Instant::now();
                    if let Some(ref pq) = pq {
                        L::fused_gate_up_matvec_pq_st(
                            pq, x_slice, &expert.gate_proj, &expert.up_proj, &mut gate, &mut up,
                        )
                        .expect("fused_gate_up_matvec_pq_st failed");
                    } else {
                        L::fused_gate_up_matvec_st(
                            x_slice, &expert.gate_proj, &expert.up_proj, &mut gate, &mut up,
                        )
                        .expect("fused_gate_up_matvec_st failed");
                    }
                    #[cfg(feature = "profile-moe-decode")]
                    let gate_us = t0.elapsed().as_micros() as u64;
                    #[cfg(feature = "profile-moe-decode")]
                    let up_us = 0u64;

                    #[cfg(feature = "profile-moe-decode")]
                    let t2 = std::time::Instant::now();
                    L::swiglu(&mut gate, &up);
                    #[cfg(feature = "profile-moe-decode")]
                    let swiglu_us = t2.elapsed().as_micros() as u64;

                    #[cfg(feature = "profile-moe-decode")]
                    let t3 = std::time::Instant::now();
                    L::matvec_st(&gate, &expert.down_proj, &mut out_buf)
                        .expect("down_proj matvec_st failed");
                    #[cfg(feature = "profile-moe-decode")]
                    let down_us = t3.elapsed().as_micros() as u64;

                    #[cfg(feature = "profile-moe-decode")]
                    crate::profiler::push_moe_decode_kernel(gate_us, up_us, swiglu_us, down_us);

                    // Write to disjoint slice of expert_outputs
                    unsafe {
                        let dst = expert_out_ptr.ptr().add(idx * h);
                        std::ptr::copy_nonoverlapping(out_buf.as_ptr(), dst, h);
                    }
                }
            })
            .map_err(|e| herbert_core::error::HerbertError::Backend(
                format!("parallel decode expert dispatch error: {}", e),
            ))?;

            #[cfg(feature = "profile-moe-decode")]
            let dispatch_us = t_dispatch.elapsed().as_micros() as u64;

            // Sequential weighted reduction
            #[cfg(feature = "profile-moe-decode")]
            let t_reduce = std::time::Instant::now();

            for idx in 0..n_sel {
                let w = weights[idx];
                let src = &expert_outputs[idx * h..(idx + 1) * h];
                for i in 0..h {
                    output[i] += w * src[i];
                }
            }

            #[cfg(feature = "profile-moe-decode")]
            {
                let reduce_us = t_reduce.elapsed().as_micros() as u64;
                let total_us = t_total.elapsed().as_micros() as u64;
                crate::profiler::push_moe_decode(router_us, dispatch_us, reduce_us, total_us, n_sel as u64);
            }

            self.add_shared_expert(x, output)?;
            return Ok(());
        }

        // Sequential fallback (non-parallel backends or single expert selected)
        expert_out_buf.resize(h, 0.0);
        for idx in 0..n_sel {
            let expert_id = top_k_buf[idx].0;
            let weight = weights[idx];
            gate_buf.iter_mut().for_each(|v| *v = 0.0);
            up_buf.iter_mut().for_each(|v| *v = 0.0);
            expert_out_buf.iter_mut().for_each(|v| *v = 0.0);

            let expert = pool.get(expert_id);
            if L::PARALLEL_EXPERTS {
                expert.forward_decode_st(x, gate_buf, up_buf, expert_out_buf)?;
            } else {
                expert.forward_decode(x, gate_buf, up_buf, expert_out_buf)?;
            }

            for i in 0..h {
                output[i] += weight * expert_out_buf[i];
            }
        }

        self.add_shared_expert(x, output)?;
        Ok(())
    }

    /// Multi-token MoE forward (prefill).
    pub fn forward_prefill(
        &self,
        x: &[f32],
        seq_len: usize,
        kv_cache: &mut crate::kv_cache::CpuKvCache,
        output: &mut [f32],
    ) -> Result<()> {
        #[cfg(feature = "moe-v6")]
        {
            if !L::PARALLEL_EXPERTS {
                self.forward_prefill_v6(x, seq_len, kv_cache, output)
            } else {
                self.forward_prefill_v8_parallel(x, seq_len, kv_cache, output)
            }
        }
        #[cfg(not(feature = "moe-v6"))]
        {
            self.forward_prefill_v5(x, seq_len, kv_cache, output)
        }
    }

    /// v5: per-token routing + per-token expert calls (matvec).
    #[cfg(not(feature = "moe-v6"))]
    fn forward_prefill_v5(
        &self,
        x: &[f32],
        seq_len: usize,
        kv_cache: &mut crate::kv_cache::CpuKvCache,
        output: &mut [f32],
    ) -> Result<()> {
        let h = self.hidden_size;
        let _k = self.num_experts_per_tok;
        let ne = self.num_experts;

        for o in output.iter_mut() { *o = 0.0; }

        let mut gate_buf = std::mem::take(&mut kv_cache.ctx.prefill_mlp_gate);
        gate_buf.resize(self.moe_intermediate_size, 0.0);
        let mut up_buf = std::mem::take(&mut kv_cache.ctx.prefill_mlp_up);
        up_buf.resize(self.moe_intermediate_size, 0.0);
        let mut expert_out = std::mem::take(&mut kv_cache.ctx.prefill_moe_expert_out);
        expert_out.resize(h, 0.0);
        let mut router_logits = std::mem::take(&mut kv_cache.ctx.prefill_moe_logits);
        router_logits.resize(ne, 0.0);

        let mut pool = self.expert_pool.lock().unwrap_or_else(|e| e.into_inner());

        for t in 0..seq_len {
            let token_x = &x[t * h..(t + 1) * h];

            L::matvec(token_x, &self.gate, &mut router_logits)?;
            let mut top_k_buf = [(0usize, 0.0f32); 8];
            let n_sel = self.route_token(&mut router_logits, &mut top_k_buf);

            let mut weights = [0.0f32; 8];
            if self.norm_topk_prob {
                let sum: f32 = top_k_buf[..n_sel].iter().map(|&(_, w)| w).sum();
                if sum > 0.0 {
                    for i in 0..n_sel { weights[i] = top_k_buf[i].1 / sum; }
                } else {
                    for i in 0..n_sel { weights[i] = top_k_buf[i].1; }
                }
            } else {
                for i in 0..n_sel { weights[i] = top_k_buf[i].1; }
            }

            let mut needed = [0usize; 8];
            for i in 0..n_sel { needed[i] = top_k_buf[i].0; }
            pool.ensure_loaded(&needed[..n_sel])?;

            let out_slice = &mut output[t * h..(t + 1) * h];
            for idx in 0..n_sel {
                let expert_id = top_k_buf[idx].0;
                let weight = weights[idx];
                gate_buf.iter_mut().for_each(|v| *v = 0.0);
                up_buf.iter_mut().for_each(|v| *v = 0.0);
                expert_out.iter_mut().for_each(|v| *v = 0.0);

                pool.get(expert_id).forward_decode(token_x, &mut gate_buf, &mut up_buf, &mut expert_out)?;

                for i in 0..h {
                    out_slice[i] += weight * expert_out[i];
                }
            }
        }

        drop(pool);
        self.add_shared_expert_prefill(x, seq_len, output)?;

        kv_cache.ctx.prefill_mlp_gate = gate_buf;
        kv_cache.ctx.prefill_mlp_up = up_buf;
        kv_cache.ctx.prefill_moe_expert_out = expert_out;
        kv_cache.ctx.prefill_moe_logits = router_logits;

        Ok(())
    }

    /// v6: batched router (f32_matmul) + expert batching (gather → matmul → scatter).
    #[cfg(feature = "moe-v6")]
    fn forward_prefill_v6(
        &self,
        x: &[f32],
        seq_len: usize,
        kv_cache: &mut crate::kv_cache::CpuKvCache,
        output: &mut [f32],
    ) -> Result<()> {
        let h = self.hidden_size;
        let inter = self.moe_intermediate_size;
        let _k = self.num_experts_per_tok;
        let ne = self.num_experts;

        // 1. Compute all router logits in one matmul
        #[cfg(feature = "profile-experts")]
        let t_router = std::time::Instant::now();
        let mut all_logits = std::mem::take(&mut kv_cache.ctx.prefill_moe_logits);
        all_logits.resize(seq_len * ne, 0.0);
        L::matmul(x, &self.gate, &mut all_logits, seq_len)?;

        // 2. Per-token activation + top-K + build expert assignments (reuse Vecs)
        let mut expert_assignments = std::mem::take(&mut kv_cache.ctx.prefill_moe_expert_assignments);
        if expert_assignments.len() < ne {
            expert_assignments.resize_with(ne, Vec::new);
        }
        for v in expert_assignments.iter_mut() { v.clear(); }

        for t in 0..seq_len {
            let logits = &mut all_logits[t * ne..(t + 1) * ne];
            let mut top_k_buf = [(0usize, 0.0f32); 8];
            let n_sel = self.route_token(logits, &mut top_k_buf);

            if self.norm_topk_prob {
                let sum: f32 = top_k_buf[..n_sel].iter().map(|&(_, w)| w).sum();
                if sum > 0.0 {
                    for i in 0..n_sel { top_k_buf[i].1 /= sum; }
                }
            }

            for i in 0..n_sel {
                let (eid, wt) = top_k_buf[i];
                expert_assignments[eid].push((t, wt));
            }
        }
        kv_cache.ctx.prefill_moe_logits = all_logits;
        #[cfg(feature = "profile-experts")]
        let router_us = t_router.elapsed().as_micros() as u64;

        // Record MoE stats (if enabled)
        #[cfg(feature = "profile-experts")]
        let t_routing = std::time::Instant::now();
        {
            let stats = moe_stats::global();
            if stats.is_enabled() {
                let tpe: Vec<(usize, usize)> = expert_assignments
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| !a.is_empty())
                    .map(|(eid, a)| (eid, a.len()))
                    .collect();
                let num_active = tpe.len();
                stats.record(moe_stats::MoeCallRecord {
                    num_experts_total: ne,
                    num_active,
                    tokens_per_expert: tpe,
                });
            }
        }

        #[cfg(feature = "profile-experts")]
        let num_active_experts = expert_assignments.iter().filter(|a| !a.is_empty()).count();

        // 3. For each active expert: gather → forward_prefill → scatter weighted
        for o in output.iter_mut() { *o = 0.0; }

        let pool = self.expert_pool.lock().unwrap_or_else(|e| e.into_inner());
        #[cfg(feature = "profile-experts")]
        let routing_us = t_routing.elapsed().as_micros() as u64;

        #[cfg(feature = "profile-experts")]
        let t_dispatch = std::time::Instant::now();

        // Reusable buffers for expert forward
        let mut batch_x = std::mem::take(&mut kv_cache.ctx.prefill_moe_batch_x);
        let mut gate_buf = std::mem::take(&mut kv_cache.ctx.prefill_mlp_gate);
        let mut up_buf = std::mem::take(&mut kv_cache.ctx.prefill_mlp_up);
        let mut expert_out_buf = std::mem::take(&mut kv_cache.ctx.prefill_moe_expert_out);

        for (expert_id, assignments) in expert_assignments.iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }
            let num_assigned = assignments.len();

            // Gather: build batch_x[num_assigned, h]
            batch_x.resize(num_assigned * h, 0.0);
            for (local_idx, &(token_idx, _)) in assignments.iter().enumerate() {
                batch_x[local_idx * h..(local_idx + 1) * h]
                    .copy_from_slice(&x[token_idx * h..(token_idx + 1) * h]);
            }

            // Forward through expert with pre-allocated buffers
            gate_buf.resize(num_assigned * inter, 0.0);
            up_buf.resize(num_assigned * inter, 0.0);
            expert_out_buf.resize(num_assigned * h, 0.0);
            pool.get(expert_id).forward_prefill(
                &batch_x, num_assigned, &mut gate_buf, &mut up_buf, &mut expert_out_buf,
            )?;

            // Scatter weighted
            for (local_idx, &(token_idx, weight)) in assignments.iter().enumerate() {
                let src = &expert_out_buf[local_idx * h..(local_idx + 1) * h];
                let dst = &mut output[token_idx * h..(token_idx + 1) * h];
                for i in 0..h {
                    dst[i] += weight * src[i];
                }
            }
        }

        self.add_shared_expert_prefill(x, seq_len, output)?;

        kv_cache.ctx.prefill_moe_batch_x = batch_x;
        kv_cache.ctx.prefill_mlp_gate = gate_buf;
        kv_cache.ctx.prefill_mlp_up = up_buf;
        kv_cache.ctx.prefill_moe_expert_out = expert_out_buf;
        kv_cache.ctx.prefill_moe_expert_assignments = expert_assignments;

        #[cfg(feature = "profile-experts")]
        let dispatch_us = t_dispatch.elapsed().as_micros() as u64;

        #[cfg(feature = "profile-experts")]
        crate::profiler::push_moe(crate::profiler::MoeStepProfile {
            router_us,
            routing_us,
            dispatch_us,
            scatter_us: 0,
            num_active_experts,
            num_experts_total: ne,
        });

        Ok(())
    }

    /// v8: parallel expert dispatch — experts run concurrently across the thread pool,
    /// each using single-threaded matmul_st to avoid pool nesting.
    #[cfg(feature = "moe-v6")]
    fn forward_prefill_v8_parallel(
        &self,
        x: &[f32],
        seq_len: usize,
        kv_cache: &mut crate::kv_cache::CpuKvCache,
        output: &mut [f32],
    ) -> Result<()> {
        let h = self.hidden_size;
        let inter = self.moe_intermediate_size;
        let _k = self.num_experts_per_tok;
        let ne = self.num_experts;

        // Phase 1: Batched router
        #[cfg(feature = "profile-experts")]
        let t_router = std::time::Instant::now();
        let mut all_logits = std::mem::take(&mut kv_cache.ctx.prefill_moe_logits);
        all_logits.resize(seq_len * ne, 0.0);
        L::matmul(x, &self.gate, &mut all_logits, seq_len)?;

        // Phase 2: Per-token activation + top-K + build expert_assignments (reuse Vecs)
        let mut expert_assignments = std::mem::take(&mut kv_cache.ctx.prefill_moe_expert_assignments);
        if expert_assignments.len() < ne {
            expert_assignments.resize_with(ne, Vec::new);
        }
        for v in expert_assignments.iter_mut() { v.clear(); }

        for t in 0..seq_len {
            let logits = &mut all_logits[t * ne..(t + 1) * ne];
            let mut top_k_buf = [(0usize, 0.0f32); 8];
            let n_sel = self.route_token(logits, &mut top_k_buf);

            if self.norm_topk_prob {
                let sum: f32 = top_k_buf[..n_sel].iter().map(|&(_, w)| w).sum();
                if sum > 0.0 {
                    for i in 0..n_sel { top_k_buf[i].1 /= sum; }
                }
            }

            for i in 0..n_sel {
                let (eid, wt) = top_k_buf[i];
                expert_assignments[eid].push((t, wt));
            }
        }
        kv_cache.ctx.prefill_moe_logits = all_logits;
        #[cfg(feature = "profile-experts")]
        let router_us = t_router.elapsed().as_micros() as u64;

        // Phase 3: Flatten assignments into contiguous arrays
        #[cfg(feature = "profile-experts")]
        let t_routing = std::time::Instant::now();
        let mut flat_token_indices = std::mem::take(&mut kv_cache.ctx.prefill_moe_flat_tok);
        flat_token_indices.clear();
        let mut flat_weights = std::mem::take(&mut kv_cache.ctx.prefill_moe_flat_wt);
        flat_weights.clear();
        let mut active_experts = std::mem::take(&mut kv_cache.ctx.prefill_moe_active);
        active_experts.clear();
        let mut total_output_tokens = 0usize;

        for (expert_id, assignments) in expert_assignments.iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }
            let offset_in_flat = flat_token_indices.len();
            let count = assignments.len();
            let offset_in_outputs = total_output_tokens;

            for &(token_idx, weight) in assignments {
                flat_token_indices.push(token_idx);
                flat_weights.push(weight);
            }

            active_experts.push([expert_id, offset_in_flat, count, offset_in_outputs]);
            total_output_tokens += count;
        }

        let num_active = active_experts.len();

        // Record MoE stats (if enabled)
        {
            let stats = moe_stats::global();
            if stats.is_enabled() {
                let tpe: Vec<(usize, usize)> = active_experts
                    .iter()
                    .map(|info| (info[0], info[2]))
                    .collect();
                stats.record(moe_stats::MoeCallRecord {
                    num_experts_total: ne,
                    num_active,
                    tokens_per_expert: tpe,
                });
            }
        }

        if num_active == 0 {
            #[cfg(feature = "profile-experts")]
            { let _ = t_routing; }
            for o in output.iter_mut() { *o = 0.0; }
            kv_cache.ctx.prefill_moe_flat_tok = flat_token_indices;
            kv_cache.ctx.prefill_moe_flat_wt = flat_weights;
            kv_cache.ctx.prefill_moe_active = active_experts;
            kv_cache.ctx.prefill_moe_expert_assignments = expert_assignments;
            return Ok(());
        }

        // Lock expert pool and ensure all active experts are loaded
        let active_ids: Vec<usize> = active_experts.iter().map(|info| info[0]).collect();
        let mut pool = self.expert_pool.lock().unwrap_or_else(|e| e.into_inner());
        pool.ensure_loaded(&active_ids)?;

        // LPT load-balanced scheduling
        let thread_pool = global_pool();
        let lpt_workers = thread_pool.num_workers().min(num_active);

        active_experts.sort_unstable_by(|a, b| b[2].cmp(&a[2]));

        let mut worker_load: Vec<usize> = vec![0; lpt_workers];
        let mut worker_lists: Vec<Vec<usize>> = vec![Vec::new(); lpt_workers];

        for (idx, info) in active_experts.iter().enumerate() {
            let count = info[2];
            let min_w = worker_load
                .iter()
                .enumerate()
                .min_by_key(|&(_, &load)| load)
                .expect("worker_load non-empty")
                .0;
            worker_lists[min_w].push(idx);
            worker_load[min_w] += count;
        }

        let mut flat_assigns = std::mem::take(&mut kv_cache.ctx.prefill_moe_flat_assigns);
        flat_assigns.clear();
        let mut worker_ranges = std::mem::take(&mut kv_cache.ctx.prefill_moe_worker_ranges);
        worker_ranges.clear();
        for list in &worker_lists {
            let offset = flat_assigns.len();
            flat_assigns.extend_from_slice(list);
            worker_ranges.push([offset, list.len()]);
        }

        #[cfg(feature = "profile-experts")]
        let routing_us = t_routing.elapsed().as_micros() as u64;

        // Phase 4: Parallel expert dispatch (LPT-scheduled)
        #[cfg(feature = "profile-experts")]
        let t_dispatch = std::time::Instant::now();
        let mut expert_outputs = std::mem::take(&mut kv_cache.ctx.prefill_moe_expert_out);
        expert_outputs.resize(total_output_tokens * h, 0.0);

        let x_ptr = SendPtr::new(x.as_ptr());
        let out_ptr = SendMutPtr::new(expert_outputs.as_mut_ptr());
        let flat_tok_ptr = SendPtr::new(flat_token_indices.as_ptr());
        let active_ptr = SendPtr::new(active_experts.as_ptr());
        let experts_opt_ptr = SendPtr::new(pool.experts_option_ptr());
        let flat_asgn_ptr = SendPtr::new(flat_assigns.as_ptr());
        let wr_ptr = SendPtr::new(worker_ranges.as_ptr());

        thread_pool.parallel_for(lpt_workers, move |_, w_start, _| {
            let range = unsafe { *wr_ptr.ptr().add(w_start) };
            let asgn_offset = range[0];
            let asgn_count = range[1];

            // Per-worker temp buffers (reused across experts in this worker's range)
            let mut batch_x = Vec::<f32>::new();
            let mut gate_buf = Vec::<f32>::new();
            let mut up_buf = Vec::<f32>::new();
            let mut down_buf = Vec::<f32>::new();

            for ai in 0..asgn_count {
                let e_idx = unsafe { *flat_asgn_ptr.ptr().add(asgn_offset + ai) };
                let info = unsafe { *active_ptr.ptr().add(e_idx) };
                let expert_id = info[0];
                let flat_offset = info[1];
                let count = info[2];
                let out_offset = info[3];

                let expert = unsafe { &*experts_opt_ptr.ptr().add(expert_id) }
                        .as_ref()
                        .expect("expert must be loaded after ensure_loaded");

                batch_x.clear();
                batch_x.resize(count * h, 0.0);
                for i in 0..count {
                    let tok_idx = unsafe { *flat_tok_ptr.ptr().add(flat_offset + i) };
                    unsafe {
                        let src = std::slice::from_raw_parts(x_ptr.ptr().add(tok_idx * h), h);
                        batch_x[i * h..(i + 1) * h].copy_from_slice(src);
                    }
                }

                gate_buf.clear();
                gate_buf.resize(count * inter, 0.0);
                up_buf.clear();
                up_buf.resize(count * inter, 0.0);
                L::fused_gate_up_matmul_st(&batch_x, &expert.gate_proj, &expert.up_proj, &mut gate_buf, &mut up_buf, count).expect("fused gate+up matmul_st failed");

                L::swiglu(&mut gate_buf, &up_buf);

                down_buf.clear();
                down_buf.resize(count * h, 0.0);
                L::matmul_st(&gate_buf, &expert.down_proj, &mut down_buf, count).expect("down_proj matmul_st failed");

                unsafe {
                    let dst = std::slice::from_raw_parts_mut(
                        out_ptr.ptr().add(out_offset * h),
                        count * h,
                    );
                    dst.copy_from_slice(&down_buf);
                }
            }
        })
        .map_err(|e| HerbertError::Backend(format!("v8 parallel expert dispatch error: {}", e)))?;

        drop(pool);
        #[cfg(feature = "profile-experts")]
        let dispatch_us = t_dispatch.elapsed().as_micros() as u64;

        // Phase 5: Weighted scatter
        #[cfg(feature = "profile-experts")]
        let t_scatter = std::time::Instant::now();
        for o in output.iter_mut() { *o = 0.0; }

        for info in &active_experts {
            let flat_offset = info[1];
            let count = info[2];
            let out_offset = info[3];

            for i in 0..count {
                let tok_idx = flat_token_indices[flat_offset + i];
                let weight = flat_weights[flat_offset + i];
                let src = &expert_outputs[(out_offset + i) * h..(out_offset + i + 1) * h];
                let dst = &mut output[tok_idx * h..(tok_idx + 1) * h];
                for j in 0..h {
                    dst[j] += weight * src[j];
                }
            }
        }

        self.add_shared_expert_prefill(x, seq_len, output)?;

        // Return buffers to kv_cache
        kv_cache.ctx.prefill_moe_expert_out = expert_outputs;
        kv_cache.ctx.prefill_moe_flat_tok = flat_token_indices;
        kv_cache.ctx.prefill_moe_flat_wt = flat_weights;
        kv_cache.ctx.prefill_moe_active = active_experts;
        kv_cache.ctx.prefill_moe_flat_assigns = flat_assigns;
        kv_cache.ctx.prefill_moe_worker_ranges = worker_ranges;
        kv_cache.ctx.prefill_moe_expert_assignments = expert_assignments;
        #[cfg(feature = "profile-experts")]
        let scatter_us = t_scatter.elapsed().as_micros() as u64;

        #[cfg(feature = "profile-experts")]
        crate::profiler::push_moe(crate::profiler::MoeStepProfile {
            router_us,
            routing_us,
            dispatch_us,
            scatter_us,
            num_active_experts: num_active,
            num_experts_total: ne,
        });

        Ok(())
    }
}

/// In-place sigmoid over a slice.
fn sigmoid_inplace(logits: &mut [f32]) {
    for v in logits.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// In-place softmax over a slice.
fn softmax_inplace(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

/// Return the top-K (index, value) pairs sorted by descending value.
/// Uses stack-allocated arrays to avoid heap allocation.
/// Returns the number of elements written to `out`.
fn top_k_indices(values: &[f32], k: usize, out: &mut [(usize, f32)]) -> usize {
    if k == 0 || values.is_empty() {
        return 0;
    }
    const MAX_EXPERTS: usize = 256;
    debug_assert!(values.len() <= MAX_EXPERTS, "too many experts for stack allocation");
    let mut indexed = [(0usize, 0.0f32); MAX_EXPERTS];
    let len = values.len();
    for (i, &v) in values.iter().enumerate() {
        indexed[i] = (i, v);
    }
    let indexed = &mut indexed[..len];
    let n = k.min(len);
    // Partial sort: we only need top-K
    indexed.select_nth_unstable_by(n - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let selected = &mut indexed[..n];
    // Sort the selected top-K by descending weight for determinism
    selected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (i, &entry) in selected.iter().enumerate() {
        out[i] = entry;
    }
    n
}
