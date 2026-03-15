//! LinearOps implementation for INT8 AVX-512 backend.

use crate::kernels;
use crate::weight::Int8Weight;
use herbert_backend_common::linear_ops::LinearOps;
use herbert_core::error::Result;

/// INT8 AVX-512 linear ops — dispatches to VPDPBUSD kernels when available,
/// scalar fallback otherwise.
pub enum Int8Avx512Ops {}

impl LinearOps for Int8Avx512Ops {
    type Weight = Int8Weight;
    const PARALLEL_EXPERTS: bool = true;

    fn matvec(x: &[f32], w: &Self::Weight, y: &mut [f32]) -> Result<()> {
        kernels::matvec_int8_avx512(x, w, y)
    }

    fn matmul(a: &[f32], w: &Self::Weight, c: &mut [f32], m: usize) -> Result<()> {
        kernels::matmul_int8_avx512(a, w, c, m)
    }

    fn matvec_st(x: &[f32], w: &Self::Weight, y: &mut [f32]) -> Result<()> {
        kernels::matvec_int8_avx512_st(x, w, y)
    }

    fn matmul_st(a: &[f32], w: &Self::Weight, c: &mut [f32], m: usize) -> Result<()> {
        kernels::matmul_int8_avx512_st(a, w, c, m)
    }

    fn fused_2_matvec(
        x: &[f32],
        w1: &Self::Weight, y1: &mut [f32],
        w2: &Self::Weight, y2: &mut [f32],
    ) -> Result<()> {
        kernels::fused_2_matvec_int8_avx512(x, w1, y1, w2, y2)
    }

    fn fused_3_matvec(
        x: &[f32],
        w1: &Self::Weight, y1: &mut [f32],
        w2: &Self::Weight, y2: &mut [f32],
        w3: &Self::Weight, y3: &mut [f32],
    ) -> Result<()> {
        kernels::fused_3_matvec_int8_avx512(x, w1, y1, w2, y2, w3, y3)
    }
}
