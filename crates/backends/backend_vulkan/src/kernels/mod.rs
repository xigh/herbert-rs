//! Kernel dispatch wrappers.
//!
//! Each function binds the correct pipeline, assembles push constants as
//! little-endian bytes, and calls `ctx.cmd_dispatch`.

pub mod norm;
pub mod activation;
pub mod rope;
pub mod matvec;
pub mod matmul;
pub mod attention;
pub mod moe;

/// Helper: round up `n / d` (ceiling division).
#[inline]
pub fn div_ceil(n: u32, d: u32) -> u32 {
    (n + d - 1) / d
}
