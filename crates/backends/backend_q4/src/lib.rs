#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_div_ceil,
    clippy::never_loop
)]

//! Unified Q4 backend.
//!
//! Uses 4-bit symmetric quantization with packed nibble layout [N/32, K/4, 32, 2].
//! Dispatches to AVX-512 VNNI (x86_64) at compile time.

pub mod backend;
pub mod experimental;
pub mod kernels;
#[cfg(target_arch = "x86_64")]
mod kernels_x86;
mod loader;
pub mod ops;
#[cfg(feature = "bench-decode-snapshot")]
pub mod snapshot;
pub mod weight;

pub use backend::Q4Backend;
pub use herbert_backend_common::thread_pool;
