//! Trait abstracting linear algebra operations across weight types.

use crate::autotune::{RuntimeAutoTuneDefaults, RuntimeAutoTuner};
use crate::kernels as common_kernels;
use crate::thread_pool::{global_pool, SendMutPtr, SendPtr};
use herbert_core::error::{HerbertError, Result};
use herbert_core::tensor::BF16;
use std::sync::OnceLock;

const RMS_NORM_BATCH_PAR_THRESHOLD: usize = 8;
const RMS_NORM_BATCH_TARGET_WORK_PER_WORKER: usize = RMS_NORM_BATCH_PAR_THRESHOLD / 2;
const ROPE_BATCH_PAR_THRESHOLD: usize = 8;
const ROPE_BATCH_TARGET_WORK_PER_WORKER: usize = ROPE_BATCH_PAR_THRESHOLD / 2;
const AUTOTUNE_MAX_UNITS: usize = usize::MAX / 4;

fn rms_norm_batch_autotuner() -> &'static RuntimeAutoTuner {
    static TUNER: OnceLock<RuntimeAutoTuner> = OnceLock::new();
    TUNER.get_or_init(|| {
        RuntimeAutoTuner::from_env(
            "common/rms_norm_batch",
            "HERBERT_RMS_NORM_BATCH",
            RuntimeAutoTuneDefaults {
                threshold_units: RMS_NORM_BATCH_PAR_THRESHOLD,
                threshold_min: 1,
                threshold_max: AUTOTUNE_MAX_UNITS,
                target_units_per_worker: RMS_NORM_BATCH_TARGET_WORK_PER_WORKER,
                target_min: 1,
                target_max: AUTOTUNE_MAX_UNITS,
            },
        )
    })
}

fn rope_batch_autotuner() -> &'static RuntimeAutoTuner {
    static TUNER: OnceLock<RuntimeAutoTuner> = OnceLock::new();
    TUNER.get_or_init(|| {
        RuntimeAutoTuner::from_env(
            "common/rope_batch",
            "HERBERT_ROPE_BATCH",
            RuntimeAutoTuneDefaults {
                threshold_units: ROPE_BATCH_PAR_THRESHOLD,
                threshold_min: 1,
                threshold_max: AUTOTUNE_MAX_UNITS,
                target_units_per_worker: ROPE_BATCH_TARGET_WORK_PER_WORKER,
                target_min: 1,
                target_max: AUTOTUNE_MAX_UNITS,
            },
        )
    })
}

/// Pre-quantized input for VNNI matvec operations.
/// Backends that use i8 quantization (Q4) can pre-quantize the input once
/// and reuse across multiple matvec calls with the same input vector.
pub struct PreQuantizedInput {
    pub i8_data: Vec<i8>,
    pub group_scales: Vec<f32>,
    pub col_sums: Vec<i64>,
}

/// Abstraction over weight-type-specific linear algebra kernels.
///
/// Each CPU backend implements this trait with its own weight type.
/// All methods are monomorphized at compile time — no dynamic dispatch overhead.
///
/// `matvec` and `matmul` must be provided; the remaining 6 methods have default
/// parallelized implementations that backends may override.
pub trait LinearOps: Send + Sync + 'static {
    /// The weight type for linear projections.
    type Weight: Send + Sync;

    /// Whether this backend supports parallel expert dispatch (v8).
    /// When true, `matmul_st` and `matvec_st` must be safe to call from
    /// within a `parallel_for` worker (i.e. they must not use the thread pool).
    const PARALLEL_EXPERTS: bool = false;

    /// Return contiguous byte ranges backing a weight (for NUMA first-touch).
    /// Each entry is `(ptr, len)` — the raw memory region to touch.
    /// Default: empty (no-op first-touch).
    fn weight_byte_ranges(_w: &Self::Weight) -> Vec<(*const u8, usize)> {
        Vec::new()
    }

    /// Matrix-vector product: y = W @ x (decode, M=1)
    fn matvec(x: &[f32], w: &Self::Weight, y: &mut [f32]) -> Result<()>;

    /// Matrix-matrix product: C = A @ W (prefill, M>1)
    fn matmul(a: &[f32], w: &Self::Weight, c: &mut [f32], m: usize) -> Result<()>;

    /// Single-threaded matvec — safe to call from within `parallel_for`.
    /// Default delegates to `Self::matvec`.
    fn matvec_st(x: &[f32], w: &Self::Weight, y: &mut [f32]) -> Result<()> {
        Self::matvec(x, w, y)
    }

    /// Single-threaded matmul — safe to call from within `parallel_for`.
    /// Default delegates to `Self::matmul`.
    fn matmul_st(a: &[f32], w: &Self::Weight, c: &mut [f32], m: usize) -> Result<()> {
        Self::matmul(a, w, c, m)
    }

    /// Float-dequant matvec: dequantize weights to f32 on-the-fly, no activation quantization.
    /// Used for all decode layers to avoid double-quantization error (i8 activations × Q4 weights).
    /// Default falls back to regular matvec.
    fn matvec_float_dequant(x: &[f32], w: &Self::Weight, y: &mut [f32]) -> Result<()> {
        Self::matvec(x, w, y)
    }

    /// Single-threaded float-dequant matvec — safe to call from within `parallel_for`.
    /// Default falls back to `matvec_float_dequant`.
    fn matvec_float_dequant_st(x: &[f32], w: &Self::Weight, y: &mut [f32]) -> Result<()> {
        Self::matvec_float_dequant(x, w, y)
    }

    /// Fused gate+up matmul (single-threaded, prefill): quantizes input rows once for both projections.
    /// Default falls back to two separate matmul_st calls.
    fn fused_gate_up_matmul_st(
        a: &[f32],
        w_gate: &Self::Weight,
        w_up: &Self::Weight,
        c_gate: &mut [f32],
        c_up: &mut [f32],
        m: usize,
    ) -> Result<()> {
        Self::matmul_st(a, w_gate, c_gate, m)?;
        Self::matmul_st(a, w_up, c_up, m)?;
        Ok(())
    }

    /// Fused gate+up matvec (single-threaded): quantizes x once for both projections.
    /// Default falls back to two separate matvec_st calls.
    fn fused_gate_up_matvec_st(
        x: &[f32],
        w_gate: &Self::Weight,
        w_up: &Self::Weight,
        y_gate: &mut [f32],
        y_up: &mut [f32],
    ) -> Result<()> {
        Self::matvec_st(x, w_gate, y_gate)?;
        Self::matvec_st(x, w_up, y_up)?;
        Ok(())
    }

    /// Pre-quantize input for reuse across multiple matvec calls with the same x.
    /// Returns `Some(PreQuantizedInput)` for backends that use i8 quantization (Q4).
    /// Returns `None` for backends that don't benefit from pre-quantization (BF16).
    fn prequantize_input(_x: &[f32]) -> Option<PreQuantizedInput> {
        None
    }

    /// Fused gate+up matvec with pre-quantized input (single-threaded).
    /// Uses cached quantization instead of re-quantizing x.
    /// Default falls back to `fused_gate_up_matvec_st` (ignoring pre-quantized data).
    fn fused_gate_up_matvec_pq_st(
        _pq: &PreQuantizedInput,
        x: &[f32],
        w_gate: &Self::Weight,
        w_up: &Self::Weight,
        y_gate: &mut [f32],
        y_up: &mut [f32],
    ) -> Result<()> {
        Self::fused_gate_up_matvec_st(x, w_gate, w_up, y_gate, y_up)
    }

    /// Single-threaded matvec with pre-quantized input.
    /// Uses cached quantization instead of re-quantizing x.
    /// Default falls back to `matvec_st` (ignoring pre-quantized data).
    fn matvec_pq_st(
        _pq: &PreQuantizedInput,
        x: &[f32],
        w: &Self::Weight,
        y: &mut [f32],
    ) -> Result<()> {
        Self::matvec_st(x, w, y)
    }

    /// Fused gate+up matvec with SwiGLU activation applied in the epilogue.
    /// Computes: output = silu(gate_proj @ x) * (up_proj @ x).
    /// Avoids the separate swiglu_inplace call by applying activation during dequant.
    /// Returns Ok(true) if activation was applied (caller should skip apply_activation),
    /// Ok(false) if not supported (caller applies activation normally).
    /// Default: falls back to fused_2_matvec + swiglu, returns Ok(true).
    fn fused_gate_up_swiglu_matvec(
        x: &[f32],
        w_gate: &Self::Weight, gate: &mut [f32],
        w_up: &Self::Weight, up: &mut [f32],
    ) -> Result<bool> {
        // Default: no fusion, caller must apply activation
        let _ = (x, w_gate, gate, w_up, up);
        Ok(false)
    }

    /// Fused 2-projection matvec: quantizes x once for both projections.
    /// Default falls back to two separate matvec calls.
    fn fused_2_matvec(
        x: &[f32],
        w1: &Self::Weight, y1: &mut [f32],
        w2: &Self::Weight, y2: &mut [f32],
    ) -> Result<()> {
        Self::matvec(x, w1, y1)?;
        Self::matvec(x, w2, y2)?;
        Ok(())
    }

    /// Fused 3-projection matvec: quantizes x once for all three projections.
    /// Default falls back to three separate matvec calls.
    fn fused_3_matvec(
        x: &[f32],
        w1: &Self::Weight, y1: &mut [f32],
        w2: &Self::Weight, y2: &mut [f32],
        w3: &Self::Weight, y3: &mut [f32],
    ) -> Result<()> {
        Self::matvec(x, w1, y1)?;
        Self::matvec(x, w2, y2)?;
        Self::matvec(x, w3, y3)?;
        Ok(())
    }

    /// RMS normalization with BF16 weights (single vector)
    fn rms_norm(input: &[f32], weight: &[BF16], output: &mut [f32], eps: f32) {
        common_kernels::rms_norm_bf16(input, weight, output, eps);
    }

    /// Fused RMSNorm + input pre-quantization (single vector, decode).
    /// Computes norm output AND quantizes it for subsequent VNNI matvec in one kernel,
    /// avoiding a store-reload through L2 between the two operations.
    /// Default: calls rms_norm then prequantize_input separately.
    fn fused_rmsnorm_prequantize(
        input: &[f32],
        weight: &[BF16],
        output: &mut [f32],
        eps: f32,
    ) -> Option<PreQuantizedInput> {
        Self::rms_norm(input, weight, output, eps);
        Self::prequantize_input(output)
    }

    /// Fused residual add + RMS normalization (single vector, decode).
    /// Computes: a[i] += b[i], then output[i] = rms_norm(a, weight, eps).
    /// Saves one full load+store of `a` vs separate residual add + rms_norm.
    fn fused_rms_norm_residual(
        a: &mut [f32],
        b: &[f32],
        weight: &[BF16],
        output: &mut [f32],
        eps: f32,
    ) {
        common_kernels::rms_norm_residual_bf16(a, b, weight, output, eps);
    }

    /// Fused residual add + RMSNorm + input pre-quantization (single vector, decode).
    /// Three-way fusion: residual add + norm + quantize in minimal passes.
    /// Eliminates one full load+store of `a` compared to separate residual + fused_rmsnorm_prequantize.
    /// Default: calls fused_rms_norm_residual then prequantize_input.
    fn fused_residual_rmsnorm_prequantize(
        a: &mut [f32],
        b: &[f32],
        weight: &[BF16],
        output: &mut [f32],
        eps: f32,
    ) -> Option<PreQuantizedInput> {
        Self::fused_rms_norm_residual(a, b, weight, output, eps);
        Self::prequantize_input(output)
    }

    /// Batched RMS normalization with BF16 weights (parallelized)
    fn rms_norm_batch(
        input: &[f32],
        weight: &[BF16],
        output: &mut [f32],
        batch: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }

        let pool = global_pool();
        let num_threads = pool.effective_workers();
        let tuner = rms_norm_batch_autotuner();
        let decision = tuner.decide(batch, batch, num_threads);
        let sample = tuner.begin_sample(&decision);

        if !decision.parallel {
            for b in 0..batch {
                let offset = b * dim;
                common_kernels::rms_norm_bf16(
                    &input[offset..offset + dim],
                    weight,
                    &mut output[offset..offset + dim],
                    eps,
                );
            }
            tuner.observe(sample, decision, batch);
            return Ok(());
        }

        let input_ptr = SendPtr::new(input.as_ptr());
        let output_ptr = SendMutPtr::new(output.as_mut_ptr());
        let weight_ptr = SendPtr::new(weight.as_ptr());

        pool.parallel_for_with_max_workers(batch, decision.max_workers, move |_, b_start, b_end| {
            let input_ptr = input_ptr.ptr();
            let output_ptr = output_ptr.ptr();
            let weight_ptr = weight_ptr.ptr();

            for b in b_start..b_end {
                let offset = b * dim;

                unsafe {
                    let input_row = std::slice::from_raw_parts(input_ptr.add(offset), dim);
                    let output_row = std::slice::from_raw_parts_mut(output_ptr.add(offset), dim);
                    let weight_row = std::slice::from_raw_parts(weight_ptr, dim);
                    common_kernels::rms_norm_bf16(input_row, weight_row, output_row, eps);
                }
            }
        })
        .map_err(|e| HerbertError::Backend(format!("rms_norm_batch thread-pool error: {}", e)))?;
        tuner.observe(sample, decision, batch);
        Ok(())
    }

    /// SwiGLU activation: silu(gate) * up
    fn swiglu(gate: &mut [f32], up: &[f32]) {
        common_kernels::swiglu_inplace(gate, up);
    }

    /// Softmax in-place
    fn softmax(logits: &mut [f32]) {
        common_kernels::softmax_inplace(logits);
    }

    /// Apply RoPE for single position (decode)
    fn apply_rope_single(
        qk: &mut [f32],
        cos: &[f32],
        sin: &[f32],
        num_heads: usize,
        head_dim: usize,
        rotary_ndims: usize,
    ) -> Result<()> {
        common_kernels::apply_rope_single(qk, cos, sin, num_heads, head_dim, rotary_ndims)
    }

    /// Apply RoPE for batch of positions (prefill, parallelized)
    fn apply_rope_batch(
        qk: &mut [f32],
        cos_cache: &[f32],
        sin_cache: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_ndims: usize,
        start_pos: usize,
    ) -> Result<()> {
        let half_dim = common_kernels::validate_rope_batch_window(
            cos_cache, sin_cache, seq_len, rotary_ndims, start_pos,
        )?;

        if seq_len == 0 {
            return Ok(());
        }

        let qk_per_pos = num_heads * head_dim;
        let pool = global_pool();
        let num_threads = pool.effective_workers();
        let tuner = rope_batch_autotuner();
        let decision = tuner.decide(seq_len, seq_len, num_threads);
        let sample = tuner.begin_sample(&decision);

        if !decision.parallel {
            for pos in 0..seq_len {
                let abs_pos = start_pos + pos;
                let pos_offset = pos * qk_per_pos;
                let cos_offset = abs_pos * half_dim;
                let qk_pos = &mut qk[pos_offset..pos_offset + qk_per_pos];
                let cos_pos = &cos_cache[cos_offset..cos_offset + half_dim];
                let sin_pos = &sin_cache[cos_offset..cos_offset + half_dim];
                common_kernels::apply_rope_position_unchecked(
                    qk_pos, cos_pos, sin_pos, num_heads, head_dim, rotary_ndims,
                );
            }
            tuner.observe(sample, decision, seq_len);
            return Ok(());
        }

        let qk_ptr = SendMutPtr::new(qk.as_mut_ptr());
        let cos_cache_ptr = SendPtr::new(cos_cache.as_ptr());
        let sin_cache_ptr = SendPtr::new(sin_cache.as_ptr());

        pool.parallel_for_with_max_workers(
            seq_len,
            decision.max_workers,
            move |_, pos_start, pos_end| {
                let qk_ptr = qk_ptr.ptr();
                let cos_cache_ptr = cos_cache_ptr.ptr();
                let sin_cache_ptr = sin_cache_ptr.ptr();

                for pos in pos_start..pos_end {
                    let abs_pos = start_pos + pos;

                    unsafe {
                        let pos_offset = pos * qk_per_pos;
                        let cos_offset = abs_pos * half_dim;
                        let qk_pos =
                            std::slice::from_raw_parts_mut(qk_ptr.add(pos_offset), qk_per_pos);
                        let cos_pos =
                            std::slice::from_raw_parts(cos_cache_ptr.add(cos_offset), half_dim);
                        let sin_pos =
                            std::slice::from_raw_parts(sin_cache_ptr.add(cos_offset), half_dim);
                        common_kernels::apply_rope_position_unchecked(
                            qk_pos, cos_pos, sin_pos, num_heads, head_dim, rotary_ndims,
                        );
                    }
                }
            },
        )
        .map_err(|e| HerbertError::Backend(format!("apply_rope_batch thread-pool error: {}", e)))?;
        tuner.observe(sample, decision, seq_len);
        Ok(())
    }

    /// Hint the hardware prefetcher to start pulling weight data into L3 cache.
    /// Called before a matvec to overlap DRAM fetch with prior computation.
    /// Default is a no-op; Q4 backend issues `_mm_prefetch` on the data buffer head.
    fn prefetch_weight(_w: &Self::Weight) {}

    /// Dump a weight tensor to a binary file for snapshot benchmarking.
    #[cfg(feature = "bench-decode-snapshot")]
    fn dump_weight(_w: &Self::Weight, _path: &std::path::Path) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "dump_weight not implemented for this backend",
        ))
    }
}
