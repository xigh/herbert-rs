#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_div_ceil
)]

//! BF16 CPU backend — no quantization, BF16→f32 dequant on the fly.
//!
//! Stores weights as Vec<BF16> (u16, 2 bytes/param). All computation in f32
//! with BF16→f32 conversion at read time. Scalar loops, no SIMD.
//!
//! Purpose: verify that full-precision BF16 weights produce correct output
//! (no rumination), confirming that Q4 quantization causes the issue.

mod backend;
mod kernels;
mod loader;
pub mod ops;
pub mod weight;

pub use backend::Bf16Backend;
pub use herbert_backend_common::thread_pool;
