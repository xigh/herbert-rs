#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_div_ceil
)]

//! INT8 AVX-512 CPU backend — symmetric per-channel INT8 weights with `VPDPBUSD` VNNI kernels.
//!
//! Quantizes all weights to INT8 symmetric per-channel (1 scale per row) at load time.
//! Computation uses AVX-512 VNNI dot-product instructions (`VPDPBUSD`) for ~2x bandwidth
//! savings over BF16. Falls back to scalar if `avx512vnni` is not detected at runtime.

mod backend;
mod kernels;
mod loader;
pub mod ops;
pub mod weight;

pub use backend::Int8Avx512Backend;
pub use herbert_backend_common::thread_pool;
