#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_div_ceil
)]

//! BF16 AVX-512 CPU backend — BF16 weights with `vdpbf16ps` SIMD kernels.
//!
//! Stores weights as Vec<BF16> (u16, 2 bytes/param). Computation uses AVX-512
//! BF16 dot-product instructions (`vdpbf16ps`) for ~16× speedup over scalar.
//! Falls back to scalar BF16 backend if `avx512bf16` is not detected at runtime.

mod backend;
mod kernels;
mod loader;
pub mod ops;

pub use backend::Bf16Avx512Backend;
pub use herbert_backend_bf16::weight;
pub use herbert_backend_common::thread_pool;
