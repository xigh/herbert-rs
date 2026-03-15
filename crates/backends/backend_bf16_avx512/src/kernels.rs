//! BF16 AVX-512 kernels — dispatches to vdpbf16ps assembly when available,
//! falls back to scalar BF16 kernels otherwise.
//!
//! All loops (rows × K) are inside the .S file — one FFI call per tile.

use herbert_backend_bf16::weight::Bf16Weight;
#[cfg(target_arch = "x86_64")]
use herbert_backend_common::kernels::has_avx512bf16;
#[cfg(target_arch = "x86_64")]
use herbert_backend_common::thread_pool::{global_pool, SendMutPtr, SendPtr};
#[cfg(target_arch = "x86_64")]
use herbert_core::error::HerbertError;
use herbert_core::error::Result;
use herbert_core::tensor::bf16_to_f32;

/// Tile size for parallelization (number of output rows per work unit).
#[cfg(target_arch = "x86_64")]
const TILE_ROWS: usize = 64;

/// Parallel threshold: parallelize only if N * K >= this.
#[cfg(target_arch = "x86_64")]
const MATVEC_PAR_THRESHOLD: usize = 64 * 1024;

/// Parallel threshold for matmul.
#[cfg(target_arch = "x86_64")]
const MATMUL_PAR_THRESHOLD: usize = 64 * 1024;

// ============================================================================
// Assembly FFI declarations
// ============================================================================

#[cfg(target_arch = "x86_64")]
extern "C" {
    /// Compute y[0..num_rows] = W @ x for a tile of rows.
    /// Converts x from f32→BF16 internally, loops over rows×K in ASM.
    fn avx512_bf16_matvec_tile(
        x_f32: *const f32,
        w_bf16: *const u16,
        y_f32: *mut f32,
        num_rows: u64,
        k: u64,
    );

    /// Compute C[mi, 0..num_rows] = A[mi, :] @ W^T for all mi in 0..m.
    /// Converts each A row from f32→BF16 internally.
    fn avx512_bf16_matmul_tile(
        a_f32: *const f32,
        w_bf16: *const u16,
        c_f32: *mut f32,
        num_rows: u64,
        k: u64,
        m: u64,
        n: u64,
    );
}

// ============================================================================
// Scalar fallback (identical to backend_bf16)
// ============================================================================

fn matvec_scalar(x: &[f32], w: &Bf16Weight, y: &mut [f32]) {
    let n = w.n;
    let k = w.k;
    for row in 0..n {
        let mut acc = 0.0f32;
        let w_offset = row * k;
        for col in 0..k {
            acc += bf16_to_f32(w.data[w_offset + col]) * x[col];
        }
        y[row] = acc;
    }
}

fn matmul_scalar(a: &[f32], w: &Bf16Weight, c: &mut [f32], m: usize) {
    let n = w.n;
    let k = w.k;
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
}

// ============================================================================
// matvec — parallel, dispatches to ASM tile kernel
// ============================================================================

pub fn matvec_bf16_avx512(x: &[f32], w: &Bf16Weight, y: &mut [f32]) -> Result<()> {
    let n = w.n;
    let k = w.k;
    assert!(x.len() >= k, "matvec: x.len()={} < k={}", x.len(), k);
    assert!(y.len() >= n, "matvec: y.len()={} < n={}", y.len(), n);

    #[cfg(not(target_arch = "x86_64"))]
    {
        matvec_scalar(x, w, y);
        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if !has_avx512bf16() {
            matvec_scalar(x, w, y);
            return Ok(());
        }

        if n * k < MATVEC_PAR_THRESHOLD {
            return matvec_bf16_avx512_st(x, w, y);
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
                let tile_rows = row_end - row_start;

                unsafe {
                    avx512_bf16_matvec_tile(
                        x_ptr,
                        w_ptr.add(row_start * k),
                        y_ptr.add(row_start),
                        tile_rows as u64,
                        k as u64,
                    );
                }
            }
        })
        .map_err(|e| HerbertError::Backend(format!("matvec thread-pool error: {}", e)))?;

        Ok(())
    }
}

// ============================================================================
// matvec_st — single-threaded
// ============================================================================

pub fn matvec_bf16_avx512_st(x: &[f32], w: &Bf16Weight, y: &mut [f32]) -> Result<()> {
    let n = w.n;
    let k = w.k;
    assert!(x.len() >= k, "matvec_st: x.len()={} < k={}", x.len(), k);
    assert!(y.len() >= n, "matvec_st: y.len()={} < n={}", y.len(), n);

    #[cfg(not(target_arch = "x86_64"))]
    {
        matvec_scalar(x, w, y);
        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if !has_avx512bf16() {
            matvec_scalar(x, w, y);
            return Ok(());
        }

        unsafe {
            avx512_bf16_matvec_tile(
                x.as_ptr(),
                w.data.as_ptr(),
                y.as_mut_ptr(),
                n as u64,
                k as u64,
            );
        }
        Ok(())
    }
}

// ============================================================================
// matmul — parallel by row tiles
// ============================================================================

pub fn matmul_bf16_avx512(a: &[f32], w: &Bf16Weight, c: &mut [f32], m: usize) -> Result<()> {
    let n = w.n;
    let k = w.k;
    assert!(a.len() >= m * k, "matmul: a.len()={} < m*k={}", a.len(), m * k);
    assert!(c.len() >= m * n, "matmul: c.len()={} < m*n={}", c.len(), m * n);

    #[cfg(not(target_arch = "x86_64"))]
    {
        matmul_scalar(a, w, c, m);
        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if !has_avx512bf16() {
            matmul_scalar(a, w, c, m);
            return Ok(());
        }

        if m * n * k < MATMUL_PAR_THRESHOLD {
            return matmul_bf16_avx512_st(a, w, c, m);
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
                let tile_rows = row_end - row_start;

                unsafe {
                    avx512_bf16_matmul_tile(
                        a_ptr,
                        w_ptr.add(row_start * k),
                        c_ptr.add(row_start),
                        tile_rows as u64,
                        k as u64,
                        m as u64,
                        n as u64,
                    );
                }
            }
        })
        .map_err(|e| HerbertError::Backend(format!("matmul thread-pool error: {}", e)))?;

        Ok(())
    }
}

// ============================================================================
// matmul_st — single-threaded
// ============================================================================

pub fn matmul_bf16_avx512_st(a: &[f32], w: &Bf16Weight, c: &mut [f32], m: usize) -> Result<()> {
    let n = w.n;
    let k = w.k;
    assert!(a.len() >= m * k, "matmul_st: a.len()={} < m*k={}", a.len(), m * k);
    assert!(c.len() >= m * n, "matmul_st: c.len()={} < m*n={}", c.len(), m * n);

    #[cfg(not(target_arch = "x86_64"))]
    {
        matmul_scalar(a, w, c, m);
        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if !has_avx512bf16() {
            matmul_scalar(a, w, c, m);
            return Ok(());
        }

        unsafe {
            avx512_bf16_matmul_tile(
                a.as_ptr(),
                w.data.as_ptr(),
                c.as_mut_ptr(),
                n as u64,
                k as u64,
                m as u64,
                n as u64,
            );
        }
        Ok(())
    }
}
