//! LinearOps implementation for BF16 AVX-512 backend.

use herbert_backend_bf16::weight::Bf16Weight;
use herbert_backend_common::linear_ops::LinearOps;
use herbert_core::error::Result;

use crate::kernels;

/// BF16 AVX-512 linear ops — dispatches to vdpbf16ps kernels when available,
/// scalar fallback otherwise.
pub enum Bf16Avx512Ops {}

impl LinearOps for Bf16Avx512Ops {
    type Weight = Bf16Weight;
    const PARALLEL_EXPERTS: bool = true;

    fn matvec(x: &[f32], w: &Self::Weight, y: &mut [f32]) -> Result<()> {
        kernels::matvec_bf16_avx512(x, w, y)
    }

    fn matmul(a: &[f32], w: &Self::Weight, c: &mut [f32], m: usize) -> Result<()> {
        kernels::matmul_bf16_avx512(a, w, c, m)
    }

    fn matvec_st(x: &[f32], w: &Self::Weight, y: &mut [f32]) -> Result<()> {
        kernels::matvec_bf16_avx512_st(x, w, y)
    }

    fn matmul_st(a: &[f32], w: &Self::Weight, c: &mut [f32], m: usize) -> Result<()> {
        kernels::matmul_bf16_avx512_st(a, w, c, m)
    }
}
