//! BF16 kernels — scalar f32 with BF16→f32 dequant on the fly.
//!
//! No SIMD, no architecture dispatch. Parallelized by row tiles via thread pool.
//! Single-thread variants (_st) for MoE expert dispatch.

use crate::weight::Bf16Weight;
use herbert_backend_common::thread_pool::{global_pool, SendMutPtr, SendPtr};
use herbert_core::error::{HerbertError, Result};
use herbert_core::tensor::bf16_to_f32;

/// Tile size for parallelization (number of output rows per work unit).
const TILE_ROWS: usize = 64;

/// Parallel threshold: parallelize only if N * K >= this.
const MATVEC_PAR_THRESHOLD: usize = 64 * 1024;

/// Parallel threshold for matmul.
const MATMUL_PAR_THRESHOLD: usize = 64 * 1024;

// ============================================================================
// matvec: y[n] = sum_k(bf16_to_f32(w[n,k]) * x[k]), parallelized by row tiles
// ============================================================================

pub fn matvec_bf16(x: &[f32], w: &Bf16Weight, y: &mut [f32]) -> Result<()> {
    let n = w.n;
    let k = w.k;
    assert!(x.len() >= k, "matvec_bf16: x.len()={} < k={}", x.len(), k);
    assert!(y.len() >= n, "matvec_bf16: y.len()={} < n={}", y.len(), n);

    if n * k < MATVEC_PAR_THRESHOLD {
        return matvec_bf16_st(x, w, y);
    }

    let pool = global_pool();
    let num_tiles = n.div_ceil(TILE_ROWS);

    let x_ptr = SendPtr::new(x.as_ptr());
    let w_ptr = SendPtr::new(w.data.as_ptr());
    let y_ptr = SendMutPtr::new(y.as_mut_ptr());

    pool.parallel_for(num_tiles, move |_, tile_start, tile_end| {
        let x_ptr = x_ptr.ptr();
        let w_ptr = w_ptr.ptr();
        let y_ptr = y_ptr.ptr();

        for tile in tile_start..tile_end {
            let row_start = tile * TILE_ROWS;
            let row_end = (row_start + TILE_ROWS).min(n);

            for row in row_start..row_end {
                let mut acc = 0.0f32;
                let w_offset = row * k;
                for col in 0..k {
                    let wf = bf16_to_f32(unsafe { *w_ptr.add(w_offset + col) });
                    let xf = unsafe { *x_ptr.add(col) };
                    acc += wf * xf;
                }
                unsafe {
                    *y_ptr.add(row) = acc;
                }
            }
        }
    })
    .map_err(|e| HerbertError::Backend(format!("matvec_bf16 thread-pool error: {}", e)))?;

    Ok(())
}

// ============================================================================
// matvec_st: single-threaded variant for MoE expert dispatch
// ============================================================================

pub fn matvec_bf16_st(x: &[f32], w: &Bf16Weight, y: &mut [f32]) -> Result<()> {
    let n = w.n;
    let k = w.k;
    assert!(x.len() >= k, "matvec_bf16_st: x.len()={} < k={}", x.len(), k);
    assert!(y.len() >= n, "matvec_bf16_st: y.len()={} < n={}", y.len(), n);

    for row in 0..n {
        let mut acc = 0.0f32;
        let w_offset = row * k;
        for col in 0..k {
            acc += bf16_to_f32(w.data[w_offset + col]) * x[col];
        }
        y[row] = acc;
    }

    Ok(())
}

// ============================================================================
// matmul: C[m, n] = A[m, k] @ W[n, k]^T, parallelized by row tiles of N
// ============================================================================

pub fn matmul_bf16(a: &[f32], w: &Bf16Weight, c: &mut [f32], m: usize) -> Result<()> {
    let n = w.n;
    let k = w.k;
    assert!(a.len() >= m * k, "matmul_bf16: a.len()={} < m*k={}", a.len(), m * k);
    assert!(c.len() >= m * n, "matmul_bf16: c.len()={} < m*n={}", c.len(), m * n);

    if m * n * k < MATMUL_PAR_THRESHOLD {
        return matmul_bf16_st(a, w, c, m);
    }

    let pool = global_pool();
    let num_tiles = n.div_ceil(TILE_ROWS);

    let a_ptr = SendPtr::new(a.as_ptr());
    let w_ptr = SendPtr::new(w.data.as_ptr());
    let c_ptr = SendMutPtr::new(c.as_mut_ptr());

    pool.parallel_for(num_tiles, move |_, tile_start, tile_end| {
        let a_ptr = a_ptr.ptr();
        let w_ptr = w_ptr.ptr();
        let c_ptr = c_ptr.ptr();

        for tile in tile_start..tile_end {
            let row_start = tile * TILE_ROWS;
            let row_end = (row_start + TILE_ROWS).min(n);

            for mi in 0..m {
                for row in row_start..row_end {
                    let mut acc = 0.0f32;
                    let w_offset = row * k;
                    let a_offset = mi * k;
                    for col in 0..k {
                        let wf = bf16_to_f32(unsafe { *w_ptr.add(w_offset + col) });
                        let af = unsafe { *a_ptr.add(a_offset + col) };
                        acc += wf * af;
                    }
                    // C is [m, n] row-major
                    unsafe {
                        *c_ptr.add(mi * n + row) = acc;
                    }
                }
            }
        }
    })
    .map_err(|e| HerbertError::Backend(format!("matmul_bf16 thread-pool error: {}", e)))?;

    Ok(())
}

// ============================================================================
// matmul_st: single-threaded variant
// ============================================================================

pub fn matmul_bf16_st(a: &[f32], w: &Bf16Weight, c: &mut [f32], m: usize) -> Result<()> {
    let n = w.n;
    let k = w.k;
    assert!(a.len() >= m * k, "matmul_bf16_st: a.len()={} < m*k={}", a.len(), m * k);
    assert!(c.len() >= m * n, "matmul_bf16_st: c.len()={} < m*n={}", c.len(), m * n);

    for mi in 0..m {
        for row in 0..n {
            let mut acc = 0.0f32;
            let w_offset = row * k;
            let a_offset = mi * k;
            for col in 0..k {
                acc += bf16_to_f32(w.data[w_offset + col]) * a[a_offset + col];
            }
            c[mi * n + row] = acc;
        }
    }

    Ok(())
}
