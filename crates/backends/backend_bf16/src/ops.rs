//! LinearOps implementation for BF16 backend.

use herbert_backend_common::linear_ops::LinearOps;
use herbert_core::error::Result;

use crate::kernels;
use crate::weight::Bf16Weight;

/// BF16 linear ops — dispatches to scalar f32 kernels with BF16→f32 dequant.
pub enum Bf16Ops {}

impl LinearOps for Bf16Ops {
    type Weight = Bf16Weight;
    const PARALLEL_EXPERTS: bool = true;

    fn matvec(x: &[f32], w: &Self::Weight, y: &mut [f32]) -> Result<()> {
        kernels::matvec_bf16(x, w, y)
    }

    fn matmul(a: &[f32], w: &Self::Weight, c: &mut [f32], m: usize) -> Result<()> {
        kernels::matmul_bf16(a, w, c, m)
    }

    fn matvec_st(x: &[f32], w: &Self::Weight, y: &mut [f32]) -> Result<()> {
        kernels::matvec_bf16_st(x, w, y)
    }

    fn matmul_st(a: &[f32], w: &Self::Weight, c: &mut [f32], m: usize) -> Result<()> {
        kernels::matmul_bf16_st(a, w, c, m)
    }
}
