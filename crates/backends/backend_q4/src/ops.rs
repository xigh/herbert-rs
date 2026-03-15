//! LinearOps implementation for unified Q4 backend.

use herbert_backend_common::linear_ops::{LinearOps, PreQuantizedInput};
use herbert_core::error::Result;

use crate::kernels;
use crate::kernels::KernelWeight;

/// Q4 linear ops -- dispatches to AVX-512 VNNI or NEON SMMLA Q4 kernels.
pub enum Q4Ops {}

impl LinearOps for Q4Ops {
    type Weight = KernelWeight;
    const PARALLEL_EXPERTS: bool = true;

    fn weight_byte_ranges(w: &Self::Weight) -> Vec<(*const u8, usize)> {
        w.byte_ranges()
    }

    fn matvec(x: &[f32], w: &Self::Weight, y: &mut [f32]) -> Result<()> {
        kernels::matvec_q4(x, w, y)
    }

    fn matmul(a: &[f32], w: &Self::Weight, c: &mut [f32], m: usize) -> Result<()> {
        kernels::matmul_q4(a, w, c, m)
    }

    fn matvec_st(x: &[f32], w: &Self::Weight, y: &mut [f32]) -> Result<()> {
        kernels::matvec_q4_st(x, w, y)
    }

    fn matvec_float_dequant(x: &[f32], w: &Self::Weight, y: &mut [f32]) -> Result<()> {
        kernels::matvec_q4_float_dequant(x, w, y)
    }

    fn matvec_float_dequant_st(x: &[f32], w: &Self::Weight, y: &mut [f32]) -> Result<()> {
        kernels::matvec_q4_float_dequant_st(x, w, y)
    }

    fn matmul_st(a: &[f32], w: &Self::Weight, c: &mut [f32], m: usize) -> Result<()> {
        kernels::matmul_q4_st(a, w, c, m)
    }

    fn fused_gate_up_matmul_st(
        a: &[f32],
        w_gate: &Self::Weight,
        w_up: &Self::Weight,
        c_gate: &mut [f32],
        c_up: &mut [f32],
        m: usize,
    ) -> Result<()> {
        kernels::fused_gate_up_matmul_q4_st(a, w_gate, w_up, c_gate, c_up, m)
    }

    fn fused_gate_up_matvec_st(
        x: &[f32],
        w_gate: &Self::Weight,
        w_up: &Self::Weight,
        y_gate: &mut [f32],
        y_up: &mut [f32],
    ) -> Result<()> {
        kernels::fused_gate_up_matvec_q4_st(x, w_gate, w_up, y_gate, y_up)
    }

    fn fused_gate_up_swiglu_matvec(
        x: &[f32],
        w_gate: &Self::Weight, gate: &mut [f32],
        w_up: &Self::Weight, up: &mut [f32],
    ) -> Result<bool> {
        kernels::fused_gate_up_swiglu_2_matvec_q4(x, w_gate, gate, w_up, up)?;
        Ok(true) // activation already applied
    }

    fn fused_2_matvec(
        x: &[f32],
        w1: &Self::Weight, y1: &mut [f32],
        w2: &Self::Weight, y2: &mut [f32],
    ) -> Result<()> {
        kernels::fused_2_matvec_q4(x, w1, y1, w2, y2)
    }

    fn fused_3_matvec(
        x: &[f32],
        w1: &Self::Weight, y1: &mut [f32],
        w2: &Self::Weight, y2: &mut [f32],
        w3: &Self::Weight, y3: &mut [f32],
    ) -> Result<()> {
        kernels::fused_3_matvec_q4(x, w1, y1, w2, y2, w3, y3)
    }

    fn prequantize_input(x: &[f32]) -> Option<PreQuantizedInput> {
        let (i8_data, group_scales, col_sums) = kernels::prequantize_input(x);
        Some(PreQuantizedInput { i8_data, group_scales, col_sums })
    }

    fn fused_rmsnorm_prequantize(
        input: &[f32],
        weight: &[herbert_core::tensor::BF16],
        output: &mut [f32],
        eps: f32,
    ) -> Option<PreQuantizedInput> {
        let (i8_data, group_scales, col_sums) =
            kernels::fused_rmsnorm_prequantize(input, weight, output, eps);
        Some(PreQuantizedInput { i8_data, group_scales, col_sums })
    }

    fn fused_residual_rmsnorm_prequantize(
        a: &mut [f32],
        b: &[f32],
        weight: &[herbert_core::tensor::BF16],
        output: &mut [f32],
        eps: f32,
    ) -> Option<PreQuantizedInput> {
        let (i8_data, group_scales, col_sums) =
            kernels::fused_residual_rmsnorm_prequantize(a, b, weight, output, eps);
        Some(PreQuantizedInput { i8_data, group_scales, col_sums })
    }

    fn prefetch_weight(w: &Self::Weight) {
        kernels::prefetch_weight(w);
    }

    fn fused_gate_up_matvec_pq_st(
        pq: &PreQuantizedInput,
        _x: &[f32],
        w_gate: &Self::Weight,
        w_up: &Self::Weight,
        y_gate: &mut [f32],
        y_up: &mut [f32],
    ) -> Result<()> {
        kernels::fused_gate_up_matvec_q4_pq_st(
            &pq.i8_data, &pq.group_scales, &pq.col_sums,
            w_gate, w_up, y_gate, y_up,
        )
    }

    fn matvec_pq_st(
        pq: &PreQuantizedInput,
        _x: &[f32],
        w: &Self::Weight,
        y: &mut [f32],
    ) -> Result<()> {
        kernels::matvec_q4_pq_st(
            &pq.i8_data, &pq.group_scales, &pq.col_sums,
            w, y,
        )
    }

    #[cfg(feature = "bench-decode-snapshot")]
    fn dump_weight(w: &Self::Weight, path: &std::path::Path) -> std::io::Result<()> {
        crate::snapshot::dump_kernel_weight(w, path)
    }
}
