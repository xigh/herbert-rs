//! INT8 AVX-512 kernels — per-channel weights + per-token activations.
//!
//! ╔═══════════════════════════════════════════════════════════════════════════╗
//! ║ ALL hot-path SIMD code MUST be in .S assembly, NEVER Rust intrinsics.   ║
//! ║ See feedback_asm_not_intrinsics.md. NON-NEGOTIABLE.                     ║
//! ╚═══════════════════════════════════════════════════════════════════════════╝
//!
//! ASM kernel: avx512_int8_raw_dot_tile — 4-row unrolled VPDPBUSD loop.
//! Outputs raw i32 dot products (u8 × s8). Rust does:
//!   y[row] = (raw_dot - 128 * x_sum) * w_scale[row] * x_scale

use crate::weight::{self, Int8Weight};
#[cfg(target_arch = "x86_64")]
use herbert_backend_common::thread_pool::{global_pool, SendMutPtr, SendPtr};
#[cfg(target_arch = "x86_64")]
use herbert_core::error::HerbertError;
use herbert_core::error::Result;

#[cfg(target_arch = "x86_64")]
const TILE_ROWS: usize = 64;
#[cfg(target_arch = "x86_64")]
const MATVEC_PAR_THRESHOLD: usize = 64 * 1024;
#[cfg(target_arch = "x86_64")]
const MATMUL_PAR_THRESHOLD: usize = 64 * 1024;

const INT8_ZERO_BIAS: i64 = 128;

// ============================================================================
// Assembly FFI
// ============================================================================

#[cfg(target_arch = "x86_64")]
extern "C" {
    /// V1: Generic 4-row unrolled VPDPBUSD with stride params.
    /// 12 loads/iter in matvec mode (4 redundant x loads + 4 stack ptr loads).
    fn avx512_int8_raw_dot_tile(
        src_u8: *const u8,
        src_s8: *const i8,
        y_i32: *mut i32,
        num_rows: u64,
        k: u64,
        u8_stride: u64,
        s8_stride: u64,
    );

    /// V2: Matvec-specialized 4-row unroll. x loaded once, shared across 4 rows.
    /// 5 loads/iter (−58% LDQ pressure vs V1). See bench-int8-matvec-proj results.
    fn avx512_int8_matvec_v2(
        w_u8: *const u8,
        x_i8: *const i8,
        y_i32: *mut i32,
        num_rows: u64,
        k: u64,
    );
}

/// Dispatch matvec dot product: V2 (shared x load) with `matvec-v2` feature,
/// V1 (generic stride-based) otherwise.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn matvec_dot(w_u8: *const u8, x_i8: *const i8, y_i32: *mut i32, n: u64, k: u64) {
    #[cfg(feature = "matvec-v2")]
    {
        avx512_int8_matvec_v2(w_u8, x_i8, y_i32, n, k);
    }
    #[cfg(not(feature = "matvec-v2"))]
    {
        avx512_int8_raw_dot_tile(w_u8, x_i8, y_i32, n, k, k, 0);
    }
}

// ============================================================================
// Runtime CPU detection
// ============================================================================

#[cfg(target_arch = "x86_64")]
fn has_avx512_vnni() -> bool {
    use std::sync::OnceLock;
    static HAS: OnceLock<bool> = OnceLock::new();
    *HAS.get_or_init(|| is_x86_feature_detected!("avx512vnni"))
}

// ============================================================================
// Scalar fallback
// ============================================================================

fn matvec_scalar(x: &[f32], w: &Int8Weight, y: &mut [f32]) {
    let n = w.n;
    let k = w.k;
    let (x_i8, x_scale, x_sum) = weight::quantize_x_pertoken(x);
    let correction = INT8_ZERO_BIAS * x_sum;
    for row in 0..n {
        let mut dot = 0i64;
        for i in 0..k {
            dot += w.data[row * k + i] as i64 * x_i8[i] as i64;
        }
        y[row] = (dot - correction) as f32 * w.scales[row] * x_scale;
    }
}

fn matmul_scalar(a: &[f32], w: &Int8Weight, c: &mut [f32], m: usize) {
    let n = w.n;
    let k = w.k;
    for mi in 0..m {
        matvec_scalar(&a[mi * k..(mi + 1) * k], w, &mut c[mi * n..(mi + 1) * n]);
    }
}

// ============================================================================
// matvec — parallel
// ============================================================================

pub fn matvec_int8_avx512(x: &[f32], w: &Int8Weight, y: &mut [f32]) -> Result<()> {
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
        if !has_avx512_vnni() {
            matvec_scalar(x, w, y);
            return Ok(());
        }

        let (x_i8, x_scale, x_sum) = weight::quantize_x_pertoken(x);
        let correction = INT8_ZERO_BIAS * x_sum;

        if n * k < MATVEC_PAR_THRESHOLD {
            let mut raw_dots = vec![0i32; n];
            unsafe {
                matvec_dot(
                    w.data.as_ptr(),
                    x_i8.as_ptr(),
                    raw_dots.as_mut_ptr(),
                    n as u64,
                    k as u64,
                );
            }
            for row in 0..n {
                y[row] = (raw_dots[row] as i64 - correction) as f32 * w.scales[row] * x_scale;
            }
            return Ok(());
        }

        let pool = global_pool();
        let num_tiles = n.div_ceil(TILE_ROWS);
        let w_data_ptr = SendPtr::new(w.data.as_ptr());
        let w_scales_ptr = SendPtr::new(w.scales.as_ptr());
        let x_i8_ptr = SendPtr::new(x_i8.as_ptr());
        let y_ptr = SendMutPtr::new(y.as_mut_ptr());

        pool.parallel_for(num_tiles, move |_, tile_start, tile_end| {
            for tile in tile_start..tile_end {
                let row_start = tile * TILE_ROWS;
                let row_end = (row_start + TILE_ROWS).min(n);
                let tile_rows = row_end - row_start;
                let mut raw_dots = vec![0i32; tile_rows];

                unsafe {
                    matvec_dot(
                        w_data_ptr.ptr().add(row_start * k),
                        x_i8_ptr.ptr(),
                        raw_dots.as_mut_ptr(),
                        tile_rows as u64,
                        k as u64,
                    );
                    for i in 0..tile_rows {
                        *y_ptr.ptr().add(row_start + i) = (raw_dots[i] as i64 - correction)
                            as f32
                            * *w_scales_ptr.ptr().add(row_start + i)
                            * x_scale;
                    }
                }
            }
        })
        .map_err(|e| HerbertError::Backend(format!("matvec thread-pool error: {}", e)))?;

        Ok(())
    }
}

// ============================================================================
// fused_3_matvec — quantize x once, dispatch Q/K/V tiles in one parallel_for
// ============================================================================

pub fn fused_3_matvec_int8_avx512(
    x: &[f32],
    w1: &Int8Weight, y1: &mut [f32],
    w2: &Int8Weight, y2: &mut [f32],
    w3: &Int8Weight, y3: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(w1.k, w2.k);
    debug_assert_eq!(w1.k, w3.k);

    #[cfg(not(target_arch = "x86_64"))]
    {
        matvec_scalar(x, w1, y1);
        matvec_scalar(x, w2, y2);
        matvec_scalar(x, w3, y3);
        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if !has_avx512_vnni() {
            matvec_scalar(x, w1, y1);
            matvec_scalar(x, w2, y2);
            matvec_scalar(x, w3, y3);
            return Ok(());
        }

        let k = w1.k;
        let n1 = w1.n;
        let n2 = w2.n;
        let n3 = w3.n;

        // Quantize x ONCE for all three projections
        let (x_i8, x_scale, x_sum) = weight::quantize_x_pertoken(x);
        let correction = INT8_ZERO_BIAS * x_sum;

        let nt1 = n1.div_ceil(TILE_ROWS);
        let nt2 = n2.div_ceil(TILE_ROWS);
        let nt3 = n3.div_ceil(TILE_ROWS);
        let total_tiles = nt1 + nt2 + nt3;
        let split12 = nt1 + nt2;

        let pool = global_pool();
        let x_i8_ptr = SendPtr::new(x_i8.as_ptr());
        let w1_data = SendPtr::new(w1.data.as_ptr());
        let w1_scales = SendPtr::new(w1.scales.as_ptr());
        let y1_ptr = SendMutPtr::new(y1.as_mut_ptr());
        let w2_data = SendPtr::new(w2.data.as_ptr());
        let w2_scales = SendPtr::new(w2.scales.as_ptr());
        let y2_ptr = SendMutPtr::new(y2.as_mut_ptr());
        let w3_data = SendPtr::new(w3.data.as_ptr());
        let w3_scales = SendPtr::new(w3.scales.as_ptr());
        let y3_ptr = SendMutPtr::new(y3.as_mut_ptr());

        pool.parallel_for(total_tiles, move |_, tile_start, tile_end| {
            for global_tile in tile_start..tile_end {
                let (tile, n, wd, ws, yp) = if global_tile < nt1 {
                    (global_tile, n1, w1_data, w1_scales, y1_ptr)
                } else if global_tile < split12 {
                    (global_tile - nt1, n2, w2_data, w2_scales, y2_ptr)
                } else {
                    (global_tile - split12, n3, w3_data, w3_scales, y3_ptr)
                };

                let row_start = tile * TILE_ROWS;
                let row_end = (row_start + TILE_ROWS).min(n);
                let tile_rows = row_end - row_start;
                let mut raw_dots = vec![0i32; tile_rows];

                unsafe {
                    matvec_dot(
                        wd.ptr().add(row_start * k),
                        x_i8_ptr.ptr(),
                        raw_dots.as_mut_ptr(),
                        tile_rows as u64,
                        k as u64,
                    );
                    for i in 0..tile_rows {
                        *yp.ptr().add(row_start + i) = (raw_dots[i] as i64 - correction)
                            as f32
                            * *ws.ptr().add(row_start + i)
                            * x_scale;
                    }
                }
            }
        })
        .map_err(|e| HerbertError::Backend(format!("fused_3_matvec thread-pool error: {}", e)))?;

        Ok(())
    }
}

// ============================================================================
// fused_2_matvec — quantize x once, dispatch gate/up tiles in one parallel_for
// ============================================================================

pub fn fused_2_matvec_int8_avx512(
    x: &[f32],
    w1: &Int8Weight, y1: &mut [f32],
    w2: &Int8Weight, y2: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(w1.k, w2.k);

    #[cfg(not(target_arch = "x86_64"))]
    {
        matvec_scalar(x, w1, y1);
        matvec_scalar(x, w2, y2);
        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if !has_avx512_vnni() {
            matvec_scalar(x, w1, y1);
            matvec_scalar(x, w2, y2);
            return Ok(());
        }

        let k = w1.k;
        let n1 = w1.n;
        let n2 = w2.n;

        // Quantize x ONCE for both projections
        let (x_i8, x_scale, x_sum) = weight::quantize_x_pertoken(x);
        let correction = INT8_ZERO_BIAS * x_sum;

        let nt1 = n1.div_ceil(TILE_ROWS);
        let nt2 = n2.div_ceil(TILE_ROWS);
        let total_tiles = nt1 + nt2;

        let pool = global_pool();
        let x_i8_ptr = SendPtr::new(x_i8.as_ptr());
        let w1_data = SendPtr::new(w1.data.as_ptr());
        let w1_scales = SendPtr::new(w1.scales.as_ptr());
        let y1_ptr = SendMutPtr::new(y1.as_mut_ptr());
        let w2_data = SendPtr::new(w2.data.as_ptr());
        let w2_scales = SendPtr::new(w2.scales.as_ptr());
        let y2_ptr = SendMutPtr::new(y2.as_mut_ptr());

        pool.parallel_for(total_tiles, move |_, tile_start, tile_end| {
            for global_tile in tile_start..tile_end {
                let (tile, n, wd, ws, yp) = if global_tile < nt1 {
                    (global_tile, n1, w1_data, w1_scales, y1_ptr)
                } else {
                    (global_tile - nt1, n2, w2_data, w2_scales, y2_ptr)
                };

                let row_start = tile * TILE_ROWS;
                let row_end = (row_start + TILE_ROWS).min(n);
                let tile_rows = row_end - row_start;
                let mut raw_dots = vec![0i32; tile_rows];

                unsafe {
                    matvec_dot(
                        wd.ptr().add(row_start * k),
                        x_i8_ptr.ptr(),
                        raw_dots.as_mut_ptr(),
                        tile_rows as u64,
                        k as u64,
                    );
                    for i in 0..tile_rows {
                        *yp.ptr().add(row_start + i) = (raw_dots[i] as i64 - correction)
                            as f32
                            * *ws.ptr().add(row_start + i)
                            * x_scale;
                    }
                }
            }
        })
        .map_err(|e| HerbertError::Backend(format!("fused_2_matvec thread-pool error: {}", e)))?;

        Ok(())
    }
}

// ============================================================================
// matvec_st — single-threaded
// ============================================================================

pub fn matvec_int8_avx512_st(x: &[f32], w: &Int8Weight, y: &mut [f32]) -> Result<()> {
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
        if !has_avx512_vnni() {
            matvec_scalar(x, w, y);
            return Ok(());
        }

        let (x_i8, x_scale, x_sum) = weight::quantize_x_pertoken(x);
        let correction = INT8_ZERO_BIAS * x_sum;
        let mut raw_dots = vec![0i32; n];

        unsafe {
            matvec_dot(
                w.data.as_ptr(),
                x_i8.as_ptr(),
                raw_dots.as_mut_ptr(),
                n as u64,
                k as u64,
            );
        }
        for row in 0..n {
            y[row] = (raw_dots[row] as i64 - correction) as f32 * w.scales[row] * x_scale;
        }
        Ok(())
    }
}

// ============================================================================
// matmul — parallel
// ============================================================================

pub fn matmul_int8_avx512(a: &[f32], w: &Int8Weight, c: &mut [f32], m: usize) -> Result<()> {
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
        if !has_avx512_vnni() {
            matmul_scalar(a, w, c, m);
            return Ok(());
        }

        if m * n * k < MATMUL_PAR_THRESHOLD {
            let mut raw_dots = vec![0i32; n];
            for mi in 0..m {
                let (x_i8, x_scale, x_sum) = weight::quantize_x_pertoken(&a[mi * k..(mi + 1) * k]);
                let correction = INT8_ZERO_BIAS * x_sum;
                unsafe {
                    matvec_dot(
                        w.data.as_ptr(),
                        x_i8.as_ptr(),
                        raw_dots.as_mut_ptr(),
                        n as u64,
                        k as u64,
                    );
                }
                for row in 0..n {
                    c[mi * n + row] =
                        (raw_dots[row] as i64 - correction) as f32 * w.scales[row] * x_scale;
                }
            }
            return Ok(());
        }

        // Pre-quantize all input rows (each tile needs all m rows)
        let quantized: Vec<_> = (0..m)
            .map(|mi| weight::quantize_x_pertoken(&a[mi * k..(mi + 1) * k]))
            .collect();

        let pool = global_pool();
        let num_tiles = n.div_ceil(TILE_ROWS);
        let w_data_ptr = SendPtr::new(w.data.as_ptr());
        let w_scales_ptr = SendPtr::new(w.scales.as_ptr());
        let c_ptr = SendMutPtr::new(c.as_mut_ptr());
        let q_ptr = SendPtr::new(quantized.as_ptr());

        pool.parallel_for(num_tiles, move |_, tile_start, tile_end| {
            for tile in tile_start..tile_end {
                let row_start = tile * TILE_ROWS;
                let row_end = (row_start + TILE_ROWS).min(n);
                let tile_rows = row_end - row_start;
                let mut raw_dots = vec![0i32; tile_rows];

                for mi in 0..m {
                    let (ref x_i8, x_scale, x_sum) = unsafe { &*q_ptr.ptr().add(mi) };
                    let correction = INT8_ZERO_BIAS * x_sum;
                    unsafe {
                        matvec_dot(
                            w_data_ptr.ptr().add(row_start * k),
                            x_i8.as_ptr(),
                            raw_dots.as_mut_ptr(),
                            tile_rows as u64,
                            k as u64,
                        );
                        for i in 0..tile_rows {
                            *c_ptr.ptr().add(mi * n + row_start + i) =
                                (raw_dots[i] as i64 - correction) as f32
                                    * *w_scales_ptr.ptr().add(row_start + i)
                                    * x_scale;
                        }
                    }
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

pub fn matmul_int8_avx512_st(a: &[f32], w: &Int8Weight, c: &mut [f32], m: usize) -> Result<()> {
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
        if !has_avx512_vnni() {
            matmul_scalar(a, w, c, m);
            return Ok(());
        }

        let mut raw_dots = vec![0i32; n];
        for mi in 0..m {
            let (x_i8, x_scale, x_sum) = weight::quantize_x_pertoken(&a[mi * k..(mi + 1) * k]);
            let correction = INT8_ZERO_BIAS * x_sum;
            unsafe {
                matvec_dot(
                    w.data.as_ptr(),
                    x_i8.as_ptr(),
                    raw_dots.as_mut_ptr(),
                    n as u64,
                    k as u64,
                );
            }
            for row in 0..n {
                c[mi * n + row] =
                    (raw_dots[row] as i64 - correction) as f32 * w.scales[row] * x_scale;
            }
        }
        Ok(())
    }
}
