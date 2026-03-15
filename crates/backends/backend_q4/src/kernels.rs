//! Q4 kernel dispatch — routes to arch-specific implementations.
//!
//! KernelWeight is an enum: Q4 (quantized), INT8, or BF16 (full-precision f32).
//! MoE routers use INT8 for precision. All projections including LM head use Q4.
//! All public functions dispatch on the variant.

use crate::weight::{BF16Weight, INT8Weight, Q4_GROUP_SIZE};
use herbert_backend_common::weight_cache::CacheWeight;
use herbert_core::error::Result;

#[cfg(target_arch = "x86_64")]
use crate::kernels_x86;

/// Mixed-precision weight: Q4-quantized, INT8-quantized, or full-precision f32.
pub enum KernelWeight {
    Q4(crate::weight::Q4Weight),
    INT8(INT8Weight),
    BF16(BF16Weight),
}

impl KernelWeight {
    /// Output dimension (number of rows).
    pub fn n(&self) -> usize {
        match self {
            KernelWeight::Q4(q) => q.n,
            KernelWeight::INT8(i) => i.n,
            KernelWeight::BF16(b) => b.n,
        }
    }
    /// Input dimension (number of columns).
    pub fn k(&self) -> usize {
        match self {
            KernelWeight::Q4(q) => q.k,
            KernelWeight::INT8(i) => i.k,
            KernelWeight::BF16(b) => b.k,
        }
    }

    /// Return raw memory ranges for NUMA first-touch.
    pub fn byte_ranges(&self) -> Vec<(*const u8, usize)> {
        match self {
            KernelWeight::Q4(q) => {
                let mut v = Vec::with_capacity(2);
                v.push((q.data.as_ptr(), q.data.len()));
                let scales_bytes = q.scales.len() * std::mem::size_of::<f32>();
                v.push((q.scales.as_ptr() as *const u8, scales_bytes));
                v
            }
            KernelWeight::INT8(i) => {
                let mut v = Vec::with_capacity(2);
                v.push((i.data.as_ptr() as *const u8, i.data.len()));
                let scales_bytes = i.scales.len() * std::mem::size_of::<f32>();
                v.push((i.scales.as_ptr() as *const u8, scales_bytes));
                v
            }
            KernelWeight::BF16(b) => {
                let bytes = b.data.len() * std::mem::size_of::<f32>();
                vec![(b.data.as_ptr() as *const u8, bytes)]
            }
        }
    }
}

// ============================================================================
// Weight prefetch — seed hardware prefetcher for upcoming matvec
// ============================================================================

/// Issue prefetch hints for a KernelWeight's data buffer.
/// **Disabled**: Zen 4 HW prefetcher is more effective without SW interference.
/// matvec-bench showed 31% BW degradation with prefetchnta on DDR5-5200.
pub fn prefetch_weight(_w: &KernelWeight) {
    // No-op: SW prefetch removed — HW prefetcher alone achieves higher DRAM BW.
    // Previous implementation touched first 2 KB at _MM_HINT_T2 locality.
}

#[allow(dead_code)]
fn _prefetch_weight_original(w: &KernelWeight) {
    const PREFETCH_LINES: usize = 32;
    const CACHE_LINE: usize = 64;

    match w {
        KernelWeight::Q4(q) => {
            let len = q.data.len();
            let ptr = q.data.as_ptr();
            let n = PREFETCH_LINES.min(len / CACHE_LINE);
            for i in 0..n {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T2 }>(
                        ptr.add(i * CACHE_LINE) as *const i8,
                    );
                }
                #[cfg(not(target_arch = "x86_64"))]
                let _ = (ptr, i);
            }
        }
        KernelWeight::INT8(i) => {
            let len = i.data.len();
            let ptr = i.data.as_ptr();
            let n = PREFETCH_LINES.min(len / CACHE_LINE);
            for j in 0..n {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T2 }>(
                        ptr.add(j * CACHE_LINE) as *const i8,
                    );
                }
                #[cfg(not(target_arch = "x86_64"))]
                let _ = (ptr, j);
            }
        }
        KernelWeight::BF16(b) => {
            let len = b.data.len();
            let ptr = b.data.as_ptr();
            let n = PREFETCH_LINES.min(len * 4 / CACHE_LINE);
            for j in 0..n {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T2 }>(
                        (ptr as *const u8).add(j * CACHE_LINE) as *const i8,
                    );
                }
                #[cfg(not(target_arch = "x86_64"))]
                let _ = (ptr, j);
            }
        }
    }
}

// ============================================================================
// f32 matvec / matmul for BF16Weight (AVX512 FMA + multi-threaded)
// ============================================================================

/// f32 dot product — AVX512 FMA when available, scalar fallback otherwise.
#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    if herbert_backend_common::kernels::has_avx512f() {
        unsafe { dot_f32_avx512(a.as_ptr(), b.as_ptr(), a.len()) }
    } else {
        dot_f32_scalar(a, b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    dot_f32_scalar(a, b)
}

#[inline]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_avx512(a: *const f32, b: *const f32, k: usize) -> f32 {
    use std::arch::x86_64::*;
    let chunks = k / 16;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut i = 0;
    let chunks2 = chunks / 2;
    for _ in 0..chunks2 {
        let va0 = _mm512_loadu_ps(a.add(i));
        let vb0 = _mm512_loadu_ps(b.add(i));
        acc0 = _mm512_fmadd_ps(va0, vb0, acc0);
        let va1 = _mm512_loadu_ps(a.add(i + 16));
        let vb1 = _mm512_loadu_ps(b.add(i + 16));
        acc1 = _mm512_fmadd_ps(va1, vb1, acc1);
        i += 32;
    }
    if chunks % 2 != 0 {
        let va = _mm512_loadu_ps(a.add(i));
        let vb = _mm512_loadu_ps(b.add(i));
        acc0 = _mm512_fmadd_ps(va, vb, acc0);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    while i < k {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }
    sum
}

/// Single-row f32 matvec: y[n] = dot(w[n,:], x) for each output row.
/// Uses AVX512 FMA + thread pool for large matrices.
fn matvec_f32(x: &[f32], w: &BF16Weight, y: &mut [f32]) {
    let n = w.n;
    let k = w.k;
    debug_assert_eq!(x.len(), k);
    debug_assert!(y.len() >= n);

    // Parallelize for large matrices (e.g. LMHead [151936, 3584])
    const PAR_THRESHOLD: usize = 1024;
    if n >= PAR_THRESHOLD {
        use herbert_backend_common::thread_pool::{global_pool, SendPtr, SendMutPtr};
        let pool = global_pool();
        let x_ptr = SendPtr::new(x.as_ptr());
        let w_ptr = SendPtr::new(w.data.as_ptr());
        let y_ptr = SendMutPtr::new(y.as_mut_ptr());
        pool.parallel_for(n, move |_, row_start, row_end| {
            for row in row_start..row_end {
                let row_data = unsafe { std::slice::from_raw_parts(w_ptr.ptr().add(row * k), k) };
                let x_slice = unsafe { std::slice::from_raw_parts(x_ptr.ptr(), k) };
                let val = dot_f32(x_slice, row_data);
                unsafe { *y_ptr.ptr().add(row) = val; }
            }
        }).expect("matvec_f32 parallel_for failed");
        return;
    }

    for row in 0..n {
        let row_data = &w.data[row * k..(row + 1) * k];
        y[row] = dot_f32(x, row_data);
    }
}

/// Multi-row f32 matmul: C[i,n] = dot(A[i,:], W[n,:]) for each (i, n).
/// A is [m, k] row-major, C is [m, n] row-major.
/// Uses AVX512 FMA + thread pool, parallelized over (i, n) output elements.
fn matmul_f32(a: &[f32], w: &BF16Weight, c: &mut [f32], m: usize) {
    let n = w.n;
    let k = w.k;
    debug_assert_eq!(a.len(), m * k);
    debug_assert!(c.len() >= m * n);

    let total = m * n;
    const PAR_THRESHOLD: usize = 128;

    if total >= PAR_THRESHOLD {
        use herbert_backend_common::thread_pool::{global_pool, SendPtr, SendMutPtr};
        let pool = global_pool();
        let a_ptr = SendPtr::new(a.as_ptr());
        let w_ptr = SendPtr::new(w.data.as_ptr());
        let c_ptr = SendMutPtr::new(c.as_mut_ptr());

        pool.parallel_for(total, move |_, start, end| {
            for idx in start..end {
                let i = idx / n;
                let row = idx % n;
                let a_row = unsafe { std::slice::from_raw_parts(a_ptr.ptr().add(i * k), k) };
                let w_row = unsafe { std::slice::from_raw_parts(w_ptr.ptr().add(row * k), k) };
                let val = dot_f32(a_row, w_row);
                unsafe { *c_ptr.ptr().add(i * n + row) = val; }
            }
        }).expect("matmul_f32 parallel_for failed");
        return;
    }

    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for row in 0..n {
            let w_row = &w.data[row * k..(row + 1) * k];
            c[i * n + row] = dot_f32(a_row, w_row);
        }
    }
}

// ============================================================================
// INT8 matvec / matmul (AVX2 maddubs + thread pool)
// ============================================================================

/// INT8 zero-point offset: weights stored as (w_i8 + 128) unsigned u8.
const INT8_ZERO_BIAS: i64 = 128;

/// Per-row INT8 dot product (scalar fallback).
#[inline]
fn dot_int8_row_scalar(
    w_u8: &[u8],
    x_i8: &[i8],
    w_scales: &[f32],
    x_group_scales: &[f32],
    x_group_col_sums: &[i64],
    k: usize,
) -> f32 {
    let n_groups = k.div_ceil(Q4_GROUP_SIZE);
    let mut acc_f32 = 0.0f32;
    for g in 0..n_groups {
        let start = g * Q4_GROUP_SIZE;
        let end = (start + Q4_GROUP_SIZE).min(k);
        let mut dot = 0i64;
        for i in start..end {
            dot += w_u8[i] as i64 * x_i8[i] as i64;
        }
        let corrected = dot - INT8_ZERO_BIAS * x_group_col_sums[g];
        acc_f32 += corrected as f32 * w_scales[g] * x_group_scales[g];
    }
    acc_f32
}

/// AVX512 VNNI per-row INT8 dot product using VPDPBUSD.
/// Processes 2 groups (64 bytes) per ZMM iteration with direct i32 accumulation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dot_int8_row_avx512_vnni(
    w_u8: &[u8],
    x_i8: &[i8],
    w_scales: &[f32],
    x_group_scales: &[f32],
    x_group_col_sums: &[i64],
    k: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let n_groups = k.div_ceil(Q4_GROUP_SIZE);
    let mut acc_f32 = 0.0f32;

    // Process pairs of groups (64 bytes = 1 ZMM) using VPDPBUSD.
    // VPDPBUSD: for each 4-byte lane, computes sum(u8[i] * i8[i]) and adds to i32 accumulator.
    let mut g = 0usize;
    while g + 1 < n_groups {
        let start = g * Q4_GROUP_SIZE;

        // Load 64 bytes (2 groups) into one ZMM
        let w_zmm = _mm512_loadu_si512(w_u8.as_ptr().add(start) as *const __m512i);
        let x_zmm = _mm512_loadu_si512(x_i8.as_ptr().add(start) as *const __m512i);

        // VPDPBUSD: u8 × i8 → accumulate i32 (16 lanes, each sums 4 products)
        let dot_zmm = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w_zmm, x_zmm);

        // Split into two halves: low 256 = group g, high 256 = group g+1
        let dot_lo = _mm512_castsi512_si256(dot_zmm);
        let dot_hi = _mm512_extracti64x4_epi64::<1>(dot_zmm);

        // Horizontal sum each half (8 i32 lanes → 1 scalar)
        let sum_g0 = {
            let hi128 = _mm256_extracti128_si256::<1>(dot_lo);
            let s128 = _mm_add_epi32(_mm256_castsi256_si128(dot_lo), hi128);
            let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
            let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
            _mm_cvtsi128_si32(s32) as i64
        };
        let sum_g1 = {
            let hi128 = _mm256_extracti128_si256::<1>(dot_hi);
            let s128 = _mm_add_epi32(_mm256_castsi256_si128(dot_hi), hi128);
            let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
            let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
            _mm_cvtsi128_si32(s32) as i64
        };

        // Per-group bias correction and scaling
        let c0 = sum_g0 - INT8_ZERO_BIAS * x_group_col_sums[g];
        let c1 = sum_g1 - INT8_ZERO_BIAS * x_group_col_sums[g + 1];
        acc_f32 += c0 as f32 * w_scales[g] * x_group_scales[g];
        acc_f32 += c1 as f32 * w_scales[g + 1] * x_group_scales[g + 1];

        g += 2;
    }

    // Handle odd remaining group with AVX2 maddubs
    if g < n_groups {
        let start = g * Q4_GROUP_SIZE;
        let end = (start + Q4_GROUP_SIZE).min(k);
        let len = end - start;
        let dot_i32;
        if len == Q4_GROUP_SIZE {
            let ones = _mm256_set1_epi16(1);
            let w = _mm256_loadu_si256(w_u8.as_ptr().add(start) as *const __m256i);
            let x = _mm256_loadu_si256(x_i8.as_ptr().add(start) as *const __m256i);
            let dot16 = _mm256_maddubs_epi16(w, x);
            let dot32 = _mm256_madd_epi16(dot16, ones);
            let hi128 = _mm256_extracti128_si256::<1>(dot32);
            let s128 = _mm_add_epi32(_mm256_castsi256_si128(dot32), hi128);
            let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
            let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
            dot_i32 = _mm_cvtsi128_si32(s32) as i64;
        } else {
            let mut dot = 0i64;
            for i in start..end { dot += w_u8[i] as i64 * x_i8[i] as i64; }
            dot_i32 = dot;
        }
        let corrected = dot_i32 - INT8_ZERO_BIAS * x_group_col_sums[g];
        acc_f32 += corrected as f32 * w_scales[g] * x_group_scales[g];
    }

    acc_f32
}

/// Per-row INT8 dot product — dispatches to AVX512 VNNI, AVX2, or scalar.
#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_int8_row(
    w_u8: &[u8],
    x_i8: &[i8],
    w_scales: &[f32],
    x_group_scales: &[f32],
    x_group_col_sums: &[i64],
    k: usize,
) -> f32 {
    if std::is_x86_feature_detected!("avx512vnni") {
        unsafe {
            dot_int8_row_avx512_vnni(w_u8, x_i8, w_scales, x_group_scales, x_group_col_sums, k)
        }
    } else if std::is_x86_feature_detected!("avx2") {
        unsafe {
            dot_int8_row_avx2(w_u8, x_i8, w_scales, x_group_scales, x_group_col_sums, k)
        }
    } else {
        dot_int8_row_scalar(w_u8, x_i8, w_scales, x_group_scales, x_group_col_sums, k)
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn dot_int8_row(
    w_u8: &[u8],
    x_i8: &[i8],
    w_scales: &[f32],
    x_group_scales: &[f32],
    x_group_col_sums: &[i64],
    k: usize,
) -> f32 {
    dot_int8_row_scalar(w_u8, x_i8, w_scales, x_group_scales, x_group_col_sums, k)
}

/// AVX2 per-row INT8 dot product using vpmaddubsw + vpmaddwd.
/// Processes one scale group (32 bytes = 1 YMM) at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_int8_row_avx2(
    w_u8: &[u8],
    x_i8: &[i8],
    w_scales: &[f32],
    x_group_scales: &[f32],
    x_group_col_sums: &[i64],
    k: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let n_groups = k.div_ceil(Q4_GROUP_SIZE);
    let ones = _mm256_set1_epi16(1);
    let mut acc_f32 = 0.0f32;

    for g in 0..n_groups {
        let start = g * Q4_GROUP_SIZE;
        let end = (start + Q4_GROUP_SIZE).min(k);
        let len = end - start;

        let dot_i32;
        if len == Q4_GROUP_SIZE {
            // Full group: 32 bytes = 1 YMM register
            let w = _mm256_loadu_si256(w_u8.as_ptr().add(start) as *const __m256i);
            let x = _mm256_loadu_si256(x_i8.as_ptr().add(start) as *const __m256i);
            // u8 × i8 → pairwise i16 sums
            let dot16 = _mm256_maddubs_epi16(w, x);
            // i16 × 1 → pairwise i32 sums
            let dot32 = _mm256_madd_epi16(dot16, ones);
            // Horizontal sum of 8 i32 lanes
            let hi128 = _mm256_extracti128_si256::<1>(dot32);
            let sum128 = _mm_add_epi32(_mm256_castsi256_si128(dot32), hi128);
            let sum64 = _mm_add_epi32(sum128, _mm_srli_si128::<8>(sum128));
            let sum32 = _mm_add_epi32(sum64, _mm_srli_si128::<4>(sum64));
            dot_i32 = _mm_cvtsi128_si32(sum32) as i64;
        } else {
            // Partial group: scalar
            let mut dot = 0i64;
            for i in start..end {
                dot += w_u8[i] as i64 * x_i8[i] as i64;
            }
            dot_i32 = dot;
        }

        let corrected = dot_i32 - INT8_ZERO_BIAS * x_group_col_sums[g];
        acc_f32 += corrected as f32 * w_scales[g] * x_group_scales[g];
    }
    acc_f32
}

/// Quantize f32 activations for INT8 matvec (reuses Q4 per-group quantization).
#[cfg(target_arch = "x86_64")]
fn quantize_x_for_int8(x: &[f32]) -> (Vec<i8>, Vec<f32>, Vec<i64>) {
    crate::weight::quantize_x_and_colsums(x)
}

#[cfg(not(target_arch = "x86_64"))]
fn quantize_x_for_int8(x: &[f32]) -> (Vec<i8>, Vec<f32>, Vec<i64>) {
    let (x_i8, scales) = crate::weight::quantize_x_f32_to_i8_grouped_scalar(x);
    let n_groups = x.len().div_ceil(Q4_GROUP_SIZE);
    let mut col_sums = vec![0i64; n_groups];
    for g in 0..n_groups {
        let start = g * Q4_GROUP_SIZE;
        let end = (start + Q4_GROUP_SIZE).min(x.len());
        for i in start..end {
            col_sums[g] += x_i8[i] as i64;
        }
    }
    (x_i8, scales, col_sums)
}

/// INT8 matvec: y[n] = dot(w[n,:], x) for each output row.
/// Uses AVX2 maddubs + thread pool for large matrices.
fn matvec_int8(x: &[f32], w: &INT8Weight, y: &mut [f32]) {
    let n = w.n;
    let k = w.k;
    debug_assert_eq!(x.len(), k);
    debug_assert!(y.len() >= n);
    let n_groups = k.div_ceil(Q4_GROUP_SIZE);

    let (x_i8, x_group_scales, x_group_col_sums) = quantize_x_for_int8(x);

    const PAR_THRESHOLD: usize = 1024;
    if n >= PAR_THRESHOLD {
        use herbert_backend_common::thread_pool::{global_pool, SendPtr, SendMutPtr};
        let pool = global_pool();
        let w_data_ptr = SendPtr::new(w.data.as_ptr());
        let w_scales_ptr = SendPtr::new(w.scales.as_ptr());
        let x_i8_ptr = SendPtr::new(x_i8.as_ptr());
        let x_gs_ptr = SendPtr::new(x_group_scales.as_ptr());
        let x_gcs_ptr = SendPtr::new(x_group_col_sums.as_ptr());
        let y_ptr = SendMutPtr::new(y.as_mut_ptr());
        pool.parallel_for(n, move |_, row_start, row_end| {
            let x_slice = unsafe { std::slice::from_raw_parts(x_i8_ptr.ptr(), k) };
            let x_gs = unsafe { std::slice::from_raw_parts(x_gs_ptr.ptr(), n_groups) };
            let x_gcs = unsafe { std::slice::from_raw_parts(x_gcs_ptr.ptr(), n_groups) };
            for row in row_start..row_end {
                let w_row = unsafe { std::slice::from_raw_parts(w_data_ptr.ptr().add(row * k), k) };
                let ws = unsafe { std::slice::from_raw_parts(w_scales_ptr.ptr().add(row * n_groups), n_groups) };
                let val = dot_int8_row(w_row, x_slice, ws, x_gs, x_gcs, k);
                unsafe { *y_ptr.ptr().add(row) = val; }
            }
        }).expect("matvec_int8 parallel_for failed");
        return;
    }

    for row in 0..n {
        let w_row = &w.data[row * k..(row + 1) * k];
        let ws = &w.scales[row * n_groups..(row + 1) * n_groups];
        y[row] = dot_int8_row(w_row, &x_i8, ws, &x_group_scales, &x_group_col_sums, k);
    }
}

/// INT8 matvec with pre-quantized input (single-threaded).
/// Skips quantization — uses caller-provided i8 data, scales, and col_sums.
fn matvec_int8_pq(
    x_i8: &[i8],
    x_group_scales: &[f32],
    x_group_col_sums: &[i64],
    w: &INT8Weight,
    y: &mut [f32],
) {
    let n = w.n;
    let k = w.k;
    debug_assert_eq!(x_i8.len(), k);
    debug_assert!(y.len() >= n);
    let n_groups = k.div_ceil(Q4_GROUP_SIZE);

    // Cast i8 to u8 slice for dot_int8_row (which expects w as &[u8], x as &[i8])
    // x_i8 stays as &[i8], w.data is already u8.
    for row in 0..n {
        let w_row = &w.data[row * k..(row + 1) * k];
        let ws = &w.scales[row * n_groups..(row + 1) * n_groups];
        y[row] = dot_int8_row(w_row, x_i8, ws, x_group_scales, x_group_col_sums, k);
    }
}

/// INT8 matmul: C[i,n] = dot(W[n,:], A[i,:]) for each (i, n).
/// Pre-quantizes all M input rows once, then uses parallel_for over m*n outputs.
fn matmul_int8(a: &[f32], w: &INT8Weight, c: &mut [f32], m: usize) {
    let n = w.n;
    let k = w.k;
    debug_assert_eq!(a.len(), m * k);
    debug_assert!(c.len() >= m * n);
    let n_groups = k.div_ceil(Q4_GROUP_SIZE);

    // Pre-quantize all M input rows once.
    let mut quants: Vec<(Vec<i8>, Vec<f32>, Vec<i64>)> = Vec::with_capacity(m);
    for i in 0..m {
        quants.push(quantize_x_for_int8(&a[i * k..(i + 1) * k]));
    }

    let total = m * n;
    const PAR_THRESHOLD: usize = 128;
    if total >= PAR_THRESHOLD {
        use herbert_backend_common::thread_pool::{global_pool, SendPtr, SendMutPtr};
        let pool = global_pool();
        let w_data_ptr = SendPtr::new(w.data.as_ptr());
        let w_scales_ptr = SendPtr::new(w.scales.as_ptr());
        let c_ptr = SendMutPtr::new(c.as_mut_ptr());

        // Build flat pointers to pre-quantized data for each row.
        let q_ptrs: Vec<(*const i8, *const f32, *const i64)> = quants.iter()
            .map(|(d, s, cs)| (d.as_ptr(), s.as_ptr(), cs.as_ptr()))
            .collect();
        let q_ptrs_ptr = SendPtr::new(q_ptrs.as_ptr());

        pool.parallel_for(total, move |_, start, end| {
            for idx in start..end {
                let i = idx / n;
                let row = idx % n;
                let (x_i8_ptr, x_gs_ptr, x_gcs_ptr) = unsafe { *q_ptrs_ptr.ptr().add(i) };
                let x_slice = unsafe { std::slice::from_raw_parts(x_i8_ptr, k) };
                let x_gs = unsafe { std::slice::from_raw_parts(x_gs_ptr, n_groups) };
                let x_gcs = unsafe { std::slice::from_raw_parts(x_gcs_ptr, n_groups) };
                let w_row = unsafe { std::slice::from_raw_parts(w_data_ptr.ptr().add(row * k), k) };
                let ws = unsafe { std::slice::from_raw_parts(w_scales_ptr.ptr().add(row * n_groups), n_groups) };
                let val = dot_int8_row(w_row, x_slice, ws, x_gs, x_gcs, k);
                unsafe { *c_ptr.ptr().add(idx) = val; }
            }
        }).expect("matmul_int8 parallel_for failed");
        return;
    }

    // Small: sequential fallback.
    for i in 0..m {
        let (ref x_i8, ref x_gs, ref x_gcs) = quants[i];
        for row in 0..n {
            let w_row = &w.data[row * k..(row + 1) * k];
            let ws = &w.scales[row * n_groups..(row + 1) * n_groups];
            c[i * n + row] = dot_int8_row(w_row, x_i8, ws, x_gs, x_gcs, k);
        }
    }
}

// ============================================================================
// Helper: extract Q4 ref from KernelWeight (panics on BF16/INT8)
// ============================================================================

#[cfg(target_arch = "x86_64")]
fn as_q4(w: &KernelWeight) -> &crate::weight::Q4Weight {
    match w {
        KernelWeight::Q4(q) => q,
        _ => panic!("expected Q4 weight in fused kernel"),
    }
}

// ============================================================================
// Public dispatch functions
// ============================================================================

pub fn matvec_q4(x: &[f32], w: &KernelWeight, y: &mut [f32]) -> Result<()> {
    match w {
        KernelWeight::BF16(bf) => { matvec_f32(x, bf, y); return Ok(()); }
        KernelWeight::INT8(i8w) => { matvec_int8(x, i8w, y); return Ok(()); }
        KernelWeight::Q4(_) => {}
    }
    #[cfg(target_arch = "x86_64")]
    return kernels_x86::matvec_q4(x, as_q4(w), y);
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (x, w, y);
        Err(herbert_core::error::HerbertError::Backend(
            "Q4 backend not supported on this architecture".to_string(),
        ))
    }
}

pub fn matmul_q4(a: &[f32], w: &KernelWeight, c: &mut [f32], m: usize) -> Result<()> {
    match w {
        KernelWeight::BF16(bf) => { matmul_f32(a, bf, c, m); return Ok(()); }
        KernelWeight::INT8(i8w) => { matmul_int8(a, i8w, c, m); return Ok(()); }
        KernelWeight::Q4(_) => {}
    }
    #[cfg(target_arch = "x86_64")]
    return kernels_x86::matmul_q4(a, as_q4(w), c, m);
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (a, w, c, m);
        Err(herbert_core::error::HerbertError::Backend(
            "Q4 backend not supported on this architecture".to_string(),
        ))
    }
}

pub fn matvec_q4_st(x: &[f32], w: &KernelWeight, y: &mut [f32]) -> Result<()> {
    match w {
        KernelWeight::BF16(bf) => { matvec_f32(x, bf, y); return Ok(()); }
        KernelWeight::INT8(i8w) => { matvec_int8(x, i8w, y); return Ok(()); }
        KernelWeight::Q4(_) => {}
    }
    #[cfg(target_arch = "x86_64")]
    return kernels_x86::matvec_q4_st(x, as_q4(w), y);
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (x, w, y);
        Err(herbert_core::error::HerbertError::Backend(
            "Q4 backend not supported on this architecture".to_string(),
        ))
    }
}

pub fn matmul_q4_st(a: &[f32], w: &KernelWeight, c: &mut [f32], m: usize) -> Result<()> {
    match w {
        KernelWeight::BF16(bf) => { matmul_f32(a, bf, c, m); return Ok(()); }
        KernelWeight::INT8(i8w) => { matmul_int8(a, i8w, c, m); return Ok(()); }
        KernelWeight::Q4(_) => {}
    }
    #[cfg(target_arch = "x86_64")]
    return kernels_x86::matmul_q4_st(a, as_q4(w), c, m);
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (a, w, c, m);
        Err(herbert_core::error::HerbertError::Backend(
            "Q4 backend not supported on this architecture".to_string(),
        ))
    }
}

pub fn fused_gate_up_matmul_q4_st(
    a: &[f32],
    w_gate: &KernelWeight,
    w_up: &KernelWeight,
    c_gate: &mut [f32],
    c_up: &mut [f32],
    m: usize,
) -> Result<()> {
    // Both must be Q4 for the fused path
    match (w_gate, w_up) {
        (KernelWeight::Q4(_), KernelWeight::Q4(_)) => {}
        _ => {
            // Fallback: separate matmuls for non-Q4 weights
            matmul_q4_st(a, w_gate, c_gate, m)?;
            matmul_q4_st(a, w_up, c_up, m)?;
            return Ok(());
        }
    }
    #[cfg(target_arch = "x86_64")]
    return kernels_x86::fused_gate_up_matmul_q4_st(a, as_q4(w_gate), as_q4(w_up), c_gate, c_up, m);
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (a, w_gate, w_up, c_gate, c_up, m);
        Err(herbert_core::error::HerbertError::Backend(
            "Q4 backend not supported on this architecture".to_string(),
        ))
    }
}

pub fn fused_gate_up_matvec_q4_st(
    x: &[f32],
    w_gate: &KernelWeight,
    w_up: &KernelWeight,
    y_gate: &mut [f32],
    y_up: &mut [f32],
) -> Result<()> {
    // If either weight is non-Q4, fall back to separate calls
    if !matches!(w_gate, KernelWeight::Q4(_)) || !matches!(w_up, KernelWeight::Q4(_)) {
        matvec_q4_st(x, w_gate, y_gate)?;
        matvec_q4_st(x, w_up, y_up)?;
        return Ok(());
    }
    #[cfg(target_arch = "x86_64")]
    return kernels_x86::fused_gate_up_matvec_q4_st(x, as_q4(w_gate), as_q4(w_up), y_gate, y_up);
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (x, w_gate, w_up, y_gate, y_up);
        Err(herbert_core::error::HerbertError::Backend(
            "Q4 backend not supported on this architecture".to_string(),
        ))
    }
}

/// Pre-quantize input vector for reuse across multiple matvec calls.
#[cfg(target_arch = "x86_64")]
pub fn prequantize_input(x: &[f32]) -> (Vec<i8>, Vec<f32>, Vec<i64>) {
    crate::weight::quantize_x_and_colsums(x)
}

#[cfg(not(target_arch = "x86_64"))]
pub fn prequantize_input(_x: &[f32]) -> (Vec<i8>, Vec<f32>, Vec<i64>) {
    panic!("Q4 prequantize_input not supported on this architecture");
}

/// Fused RMSNorm + input pre-quantization (AVX512).
/// Computes norm output AND quantizes it in 2 passes (vs 4 for separate ops),
/// keeping data in registers between RMSNorm and quantization.
///
/// Pass 1: sum of squares (for inv_rms)
/// Pass 2: per-group: apply inv_rms*weight → find max_abs → quantize → col_sum
#[cfg(target_arch = "x86_64")]
pub fn fused_rmsnorm_prequantize(
    input: &[f32],
    weight: &[herbert_core::tensor::BF16],
    output: &mut [f32],
    eps: f32,
) -> (Vec<i8>, Vec<f32>, Vec<i64>) {
    if herbert_backend_common::kernels::has_avx512f() && input.len() >= 64 {
        return unsafe { fused_rmsnorm_prequantize_avx512(input, weight, output, eps) };
    }
    // Scalar fallback: separate ops
    herbert_backend_common::kernels::rms_norm_bf16(input, weight, output, eps);
    crate::weight::quantize_x_and_colsums(output)
}

#[cfg(not(target_arch = "x86_64"))]
pub fn fused_rmsnorm_prequantize(
    _input: &[f32],
    _weight: &[herbert_core::tensor::BF16],
    _output: &mut [f32],
    _eps: f32,
) -> (Vec<i8>, Vec<f32>, Vec<i64>) {
    panic!("Q4 fused_rmsnorm_prequantize not supported on this architecture");
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn fused_rmsnorm_prequantize_avx512(
    input: &[f32],
    weight: &[herbert_core::tensor::BF16],
    output: &mut [f32],
    eps: f32,
) -> (Vec<i8>, Vec<f32>, Vec<i64>) {
    use std::arch::x86_64::*;
    use crate::weight::Q4_GROUP_SIZE;

    let dim = input.len();
    let inp = input.as_ptr();
    let wt = weight.as_ptr();
    let out = output.as_mut_ptr();
    let chunks = dim / 64;

    // === Pass 1: sum of squares (identical to rms_norm_bf16_avx512) ===
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    for i in 0..chunks {
        let base = i * 64;
        let v0 = _mm512_loadu_ps(inp.add(base));
        let v1 = _mm512_loadu_ps(inp.add(base + 16));
        let v2 = _mm512_loadu_ps(inp.add(base + 32));
        let v3 = _mm512_loadu_ps(inp.add(base + 48));
        acc0 = _mm512_fmadd_ps(v0, v0, acc0);
        acc1 = _mm512_fmadd_ps(v1, v1, acc1);
        acc2 = _mm512_fmadd_ps(v2, v2, acc2);
        acc3 = _mm512_fmadd_ps(v3, v3, acc3);
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
    let mut sum_sq = _mm512_reduce_add_ps(acc0);
    let done = chunks * 64;
    for i in done..dim {
        let x = *inp.add(i);
        sum_sq += x * x;
    }
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    let inv_rms_v = _mm512_set1_ps(inv_rms);

    // === Pass 2: fused apply_norm + quantize per group of 32 ===
    let n_groups = dim.div_ceil(Q4_GROUP_SIZE);
    let mut x_i8 = vec![0i8; dim];
    let mut group_scales = vec![0.0f32; n_groups];
    let mut col_sums = vec![0i64; n_groups];
    let qi8 = x_i8.as_mut_ptr();
    let abs_mask = _mm512_set1_epi32(0x7FFF_FFFFi32);

    for g in 0..n_groups {
        let base = g * Q4_GROUP_SIZE;
        let remain = dim - base;

        if remain >= 32 {
            // Load input and BF16 weights, compute norm output
            let v0 = _mm512_loadu_ps(inp.add(base));
            let v1 = _mm512_loadu_ps(inp.add(base + 16));
            let w16_0 = _mm256_loadu_si256(wt.add(base) as *const __m256i);
            let w16_1 = _mm256_loadu_si256(wt.add(base + 16) as *const __m256i);
            let wf0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(w16_0), 16));
            let wf1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(w16_1), 16));
            let norm0 = _mm512_mul_ps(_mm512_mul_ps(v0, inv_rms_v), wf0);
            let norm1 = _mm512_mul_ps(_mm512_mul_ps(v1, inv_rms_v), wf1);

            // Store norm output
            _mm512_storeu_ps(out.add(base), norm0);
            _mm512_storeu_ps(out.add(base + 16), norm1);

            // Find max_abs across 32 norm values
            let abs0 = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(norm0), abs_mask));
            let abs1 = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(norm1), abs_mask));
            let max_abs = _mm512_reduce_max_ps(_mm512_max_ps(abs0, abs1));

            let (scale, inv_scale) = if max_abs > f32::EPSILON {
                (max_abs / 127.0, 127.0 / max_abs)
            } else {
                (1.0, 1.0)
            };
            group_scales[g] = scale;
            let inv_scale_v = _mm512_set1_ps(inv_scale);

            // Quantize norm output to i8
            let q0_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(norm0, inv_scale_v));
            let q1_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(norm1, inv_scale_v));
            let q0_i8 = _mm512_cvtsepi32_epi8(q0_i32);
            let q1_i8 = _mm512_cvtsepi32_epi8(q1_i32);
            _mm_storeu_si128(qi8.add(base) as *mut __m128i, q0_i8);
            _mm_storeu_si128(qi8.add(base + 16) as *mut __m128i, q1_i8);

            // Column sum
            let sum0 = _mm512_reduce_add_epi32(q0_i32);
            let sum1 = _mm512_reduce_add_epi32(q1_i32);
            col_sums[g] = (sum0 + sum1) as i64;
        } else {
            // Partial group: scalar fallback
            let bf16_to_f32 = herbert_core::tensor::bf16_to_f32;
            for i in base..dim {
                let norm_val = *inp.add(i) * inv_rms * bf16_to_f32(*wt.add(i));
                *out.add(i) = norm_val;
            }
            let mut abs_max = 0.0f32;
            for i in base..dim {
                let av = (*out.add(i)).abs();
                if av > abs_max { abs_max = av; }
            }
            let scale = if abs_max > f32::EPSILON { abs_max / 127.0 } else { 1.0 };
            group_scales[g] = scale;
            let is = 1.0 / scale;
            let mut cs = 0i64;
            for i in base..dim {
                let q = (*out.add(i) * is).round().clamp(-127.0, 127.0) as i8;
                *qi8.add(i) = q;
                cs += q as i64;
            }
            col_sums[g] = cs;
        }
    }

    (x_i8, group_scales, col_sums)
}

/// Fused residual add + RMSNorm + per-group i8 quantization.
/// Three-way fusion: a[i] += b[i], then norm, then quantize — all in 2 passes.
/// Eliminates one full load+store of the hidden state vs separate residual + fused_rmsnorm_prequantize.
///
/// Pass 1: residual add + sum of squares (for inv_rms)
/// Pass 2: per-group: apply inv_rms*weight → find max_abs → quantize → col_sum
#[cfg(target_arch = "x86_64")]
pub fn fused_residual_rmsnorm_prequantize(
    a: &mut [f32],
    b: &[f32],
    weight: &[herbert_core::tensor::BF16],
    output: &mut [f32],
    eps: f32,
) -> (Vec<i8>, Vec<f32>, Vec<i64>) {
    if herbert_backend_common::kernels::has_avx512f() && a.len() >= 64 {
        return unsafe { fused_residual_rmsnorm_prequantize_avx512(a, b, weight, output, eps) };
    }
    // Scalar fallback: fused residual+norm, then quantize
    herbert_backend_common::kernels::rms_norm_residual_bf16(a, b, weight, output, eps);
    crate::weight::quantize_x_and_colsums(output)
}

#[cfg(not(target_arch = "x86_64"))]
pub fn fused_residual_rmsnorm_prequantize(
    _a: &mut [f32],
    _b: &[f32],
    _weight: &[herbert_core::tensor::BF16],
    _output: &mut [f32],
    _eps: f32,
) -> (Vec<i8>, Vec<f32>, Vec<i64>) {
    panic!("Q4 fused_residual_rmsnorm_prequantize not supported on this architecture");
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn fused_residual_rmsnorm_prequantize_avx512(
    a: &mut [f32],
    b: &[f32],
    weight: &[herbert_core::tensor::BF16],
    output: &mut [f32],
    eps: f32,
) -> (Vec<i8>, Vec<f32>, Vec<i64>) {
    use std::arch::x86_64::*;
    use crate::weight::Q4_GROUP_SIZE;

    let dim = a.len();
    let a_ptr = a.as_mut_ptr();
    let b_ptr = b.as_ptr();
    let wt = weight.as_ptr();
    let out = output.as_mut_ptr();
    let chunks = dim / 64;

    // === Pass 1: residual add + sum of squares ===
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    for i in 0..chunks {
        let base = i * 64;
        macro_rules! residual_sq {
            ($acc:ident, $off:expr) => {{
                let av = _mm512_loadu_ps(a_ptr.add(base + $off));
                let bv = _mm512_loadu_ps(b_ptr.add(base + $off));
                let sum = _mm512_add_ps(av, bv);
                _mm512_storeu_ps(a_ptr.add(base + $off), sum);
                $acc = _mm512_fmadd_ps(sum, sum, $acc);
            }};
        }
        residual_sq!(acc0, 0);
        residual_sq!(acc1, 16);
        residual_sq!(acc2, 32);
        residual_sq!(acc3, 48);
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
    let mut sum_sq = _mm512_reduce_add_ps(acc0);
    let done = chunks * 64;
    for i in done..dim {
        let val = *a_ptr.add(i) + *b_ptr.add(i);
        *a_ptr.add(i) = val;
        sum_sq += val * val;
    }
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    let inv_rms_v = _mm512_set1_ps(inv_rms);

    // === Pass 2: fused apply_norm + quantize per group of 32 ===
    // (identical to fused_rmsnorm_prequantize_avx512 pass 2, but reads from a_ptr)
    let n_groups = dim.div_ceil(Q4_GROUP_SIZE);
    let mut x_i8 = vec![0i8; dim];
    let mut group_scales = vec![0.0f32; n_groups];
    let mut col_sums = vec![0i64; n_groups];
    let qi8 = x_i8.as_mut_ptr();
    let abs_mask = _mm512_set1_epi32(0x7FFF_FFFFi32);

    for g in 0..n_groups {
        let base = g * Q4_GROUP_SIZE;
        let remain = dim - base;

        if remain >= 32 {
            // Load a (already has residual applied) and BF16 weights, compute norm output
            let v0 = _mm512_loadu_ps(a_ptr.add(base));
            let v1 = _mm512_loadu_ps(a_ptr.add(base + 16));
            let w16_0 = _mm256_loadu_si256(wt.add(base) as *const __m256i);
            let w16_1 = _mm256_loadu_si256(wt.add(base + 16) as *const __m256i);
            let wf0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(w16_0), 16));
            let wf1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(w16_1), 16));
            let norm0 = _mm512_mul_ps(_mm512_mul_ps(v0, inv_rms_v), wf0);
            let norm1 = _mm512_mul_ps(_mm512_mul_ps(v1, inv_rms_v), wf1);

            // Store norm output
            _mm512_storeu_ps(out.add(base), norm0);
            _mm512_storeu_ps(out.add(base + 16), norm1);

            // Find max_abs across 32 norm values
            let abs0 = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(norm0), abs_mask));
            let abs1 = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(norm1), abs_mask));
            let max_abs = _mm512_reduce_max_ps(_mm512_max_ps(abs0, abs1));

            let (scale, inv_scale) = if max_abs > f32::EPSILON {
                (max_abs / 127.0, 127.0 / max_abs)
            } else {
                (1.0, 1.0)
            };
            group_scales[g] = scale;
            let inv_scale_v = _mm512_set1_ps(inv_scale);

            // Quantize norm output to i8
            let q0_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(norm0, inv_scale_v));
            let q1_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(norm1, inv_scale_v));
            let q0_i8 = _mm512_cvtsepi32_epi8(q0_i32);
            let q1_i8 = _mm512_cvtsepi32_epi8(q1_i32);
            _mm_storeu_si128(qi8.add(base) as *mut __m128i, q0_i8);
            _mm_storeu_si128(qi8.add(base + 16) as *mut __m128i, q1_i8);

            // Column sum
            let sum0 = _mm512_reduce_add_epi32(q0_i32);
            let sum1 = _mm512_reduce_add_epi32(q1_i32);
            col_sums[g] = (sum0 + sum1) as i64;
        } else {
            // Partial group: scalar fallback
            let bf16_to_f32 = herbert_core::tensor::bf16_to_f32;
            for i in base..dim {
                let norm_val = *a_ptr.add(i) * inv_rms * bf16_to_f32(*wt.add(i));
                *out.add(i) = norm_val;
            }
            let mut abs_max = 0.0f32;
            for i in base..dim {
                let av = (*out.add(i)).abs();
                if av > abs_max { abs_max = av; }
            }
            let scale = if abs_max > f32::EPSILON { abs_max / 127.0 } else { 1.0 };
            group_scales[g] = scale;
            let is = 1.0 / scale;
            let mut cs = 0i64;
            for i in base..dim {
                let q = (*out.add(i) * is).round().clamp(-127.0, 127.0) as i8;
                *qi8.add(i) = q;
                cs += q as i64;
            }
            col_sums[g] = cs;
        }
    }

    (x_i8, group_scales, col_sums)
}

pub fn fused_gate_up_matvec_q4_pq_st(
    x_i8: &[i8],
    x_group_scales: &[f32],
    x_group_col_sums: &[i64],
    w_gate: &KernelWeight,
    w_up: &KernelWeight,
    y_gate: &mut [f32],
    y_up: &mut [f32],
) -> Result<()> {
    if !matches!(w_gate, KernelWeight::Q4(_)) || !matches!(w_up, KernelWeight::Q4(_)) {
        // Non-Q4 weights don't use pre-quantized path; fall back but need f32 x
        // This shouldn't happen in practice — MoE experts are always Q4
        return Err(herbert_core::error::HerbertError::Backend(
            "fused_gate_up_matvec_pq_st requires Q4 weights".to_string(),
        ));
    }
    #[cfg(target_arch = "x86_64")]
    return kernels_x86::fused_gate_up_matvec_q4_pq_st(
        x_i8, x_group_scales, x_group_col_sums,
        as_q4(w_gate), as_q4(w_up), y_gate, y_up,
    );
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (x_i8, x_group_scales, x_group_col_sums, w_gate, w_up, y_gate, y_up);
        Err(herbert_core::error::HerbertError::Backend(
            "Q4 backend not supported on this architecture".to_string(),
        ))
    }
}

pub fn matvec_q4_pq_st(
    x_i8: &[i8],
    x_group_scales: &[f32],
    x_group_col_sums: &[i64],
    w: &KernelWeight,
    y: &mut [f32],
) -> Result<()> {
    match w {
        KernelWeight::INT8(i8w) => {
            matvec_int8_pq(x_i8, x_group_scales, x_group_col_sums, i8w, y);
            return Ok(());
        }
        KernelWeight::BF16(_) => {
            return Err(herbert_core::error::HerbertError::Backend(
                "matvec_pq_st not supported for BF16 weights".to_string(),
            ));
        }
        KernelWeight::Q4(_) => {}
    }
    #[cfg(target_arch = "x86_64")]
    return kernels_x86::matvec_q4_pq_st(
        x_i8, x_group_scales, x_group_col_sums, as_q4(w), y,
    );
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (x_i8, x_group_scales, x_group_col_sums, w, y);
        Err(herbert_core::error::HerbertError::Backend(
            "Q4 backend not supported on this architecture".to_string(),
        ))
    }
}

pub fn matvec_q4_float_dequant(x: &[f32], w: &KernelWeight, y: &mut [f32]) -> Result<()> {
    match w {
        KernelWeight::BF16(bf) => { matvec_f32(x, bf, y); return Ok(()); }
        KernelWeight::INT8(i8w) => { matvec_int8(x, i8w, y); return Ok(()); }
        KernelWeight::Q4(_) => {}
    }
    #[cfg(target_arch = "x86_64")]
    return kernels_x86::matvec_q4_float_dequant(x, as_q4(w), y);
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (x, w, y);
        Err(herbert_core::error::HerbertError::Backend(
            "Q4 backend not supported on this architecture".to_string(),
        ))
    }
}

pub fn matvec_q4_float_dequant_st(x: &[f32], w: &KernelWeight, y: &mut [f32]) -> Result<()> {
    match w {
        KernelWeight::BF16(bf) => { matvec_f32(x, bf, y); return Ok(()); }
        KernelWeight::INT8(i8w) => { matvec_int8(x, i8w, y); return Ok(()); }
        KernelWeight::Q4(_) => {}
    }
    #[cfg(target_arch = "x86_64")]
    return kernels_x86::matvec_q4_float_dequant_st(x, as_q4(w), y);
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (x, w, y);
        Err(herbert_core::error::HerbertError::Backend(
            "Q4 backend not supported on this architecture".to_string(),
        ))
    }
}

pub fn fused_2_matvec_q4(
    x: &[f32],
    w1: &KernelWeight, y1: &mut [f32],
    w2: &KernelWeight, y2: &mut [f32],
) -> Result<()> {
    // If either weight is non-Q4, fall back to separate calls
    if !matches!(w1, KernelWeight::Q4(_)) || !matches!(w2, KernelWeight::Q4(_)) {
        matvec_q4(x, w1, y1)?;
        matvec_q4(x, w2, y2)?;
        return Ok(());
    }
    #[cfg(target_arch = "x86_64")]
    return kernels_x86::fused_2_matvec_q4(x, as_q4(w1), y1, as_q4(w2), y2);
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (x, w1, y1, w2, y2);
        Err(herbert_core::error::HerbertError::Backend(
            "Q4 backend not supported on this architecture".to_string(),
        ))
    }
}

pub fn fused_gate_up_swiglu_2_matvec_q4(
    x: &[f32],
    w_gate: &KernelWeight, gate: &mut [f32],
    w_up: &KernelWeight, up: &mut [f32],
) -> Result<()> {
    // Both weights must be Q4 for fused path
    if !matches!(w_gate, KernelWeight::Q4(_)) || !matches!(w_up, KernelWeight::Q4(_)) {
        return Err(herbert_core::error::HerbertError::Backend(
            "fused_gate_up_swiglu requires Q4 weights".to_string(),
        ));
    }
    #[cfg(target_arch = "x86_64")]
    return kernels_x86::fused_gate_up_swiglu_2_matvec_q4(x, as_q4(w_gate), gate, as_q4(w_up), up);
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (x, w_gate, gate, w_up, up);
        Err(herbert_core::error::HerbertError::Backend(
            "Q4 backend not supported on this architecture".to_string(),
        ))
    }
}

pub fn fused_3_matvec_q4(
    x: &[f32],
    w1: &KernelWeight, y1: &mut [f32],
    w2: &KernelWeight, y2: &mut [f32],
    w3: &KernelWeight, y3: &mut [f32],
) -> Result<()> {
    // If any weight is non-Q4, fall back to separate calls
    if !matches!(w1, KernelWeight::Q4(_)) || !matches!(w2, KernelWeight::Q4(_)) || !matches!(w3, KernelWeight::Q4(_)) {
        matvec_q4(x, w1, y1)?;
        matvec_q4(x, w2, y2)?;
        matvec_q4(x, w3, y3)?;
        return Ok(());
    }
    #[cfg(target_arch = "x86_64")]
    return kernels_x86::fused_3_matvec_q4(x, as_q4(w1), y1, as_q4(w2), y2, as_q4(w3), y3);
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (x, w1, y1, w2, y2, w3, y3);
        Err(herbert_core::error::HerbertError::Backend(
            "Q4 backend not supported on this architecture".to_string(),
        ))
    }
}

// ============================================================================
// CacheWeight implementation for KernelWeight enum
// ============================================================================

/// Tag bytes for cache serialization.
const CACHE_TAG_Q4: u8 = 0;
const CACHE_TAG_BF16: u8 = 1;
const CACHE_TAG_INT8: u8 = 2;

impl CacheWeight for KernelWeight {
    // v19: pre-interleaved Q4 weight layout (eliminates vpunpck, enables dual acc)
    const CACHE_VERSION: u32 = 19;

    fn cache_write(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        match self {
            KernelWeight::Q4(q4) => {
                w.write_all(&[CACHE_TAG_Q4])?;
                q4.cache_write_inner(w)
            }
            KernelWeight::INT8(i8w) => {
                w.write_all(&[CACHE_TAG_INT8])?;
                i8w.cache_write_inner(w)
            }
            KernelWeight::BF16(bf) => {
                w.write_all(&[CACHE_TAG_BF16])?;
                bf.cache_write_inner(w)
            }
        }
    }

    fn cache_read(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut tag = [0u8; 1];
        r.read_exact(&mut tag)?;
        match tag[0] {
            CACHE_TAG_Q4 => {
                Ok(KernelWeight::Q4(crate::weight::Q4Weight::cache_read_inner(r)?))
            }
            CACHE_TAG_INT8 => {
                Ok(KernelWeight::INT8(INT8Weight::cache_read_inner(r)?))
            }
            CACHE_TAG_BF16 => {
                Ok(KernelWeight::BF16(BF16Weight::cache_read_inner(r)?))
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown KernelWeight cache tag: {}", tag[0]),
            )),
        }
    }
}

// ============================================================================
// Micro-benchmarks for INT8 vs BF16 matmul (run with: cargo test -p herbert-backend-q4 --release -- bench_int8 --nocapture)
// ============================================================================

#[cfg(test)]
mod bench_int8 {
    use super::*;
    use std::time::Instant;

    fn make_random_f32(len: usize, seed: u64) -> Vec<f32> {
        let mut v = vec![0.0f32; len];
        let mut s = seed;
        for x in v.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *x = ((s >> 33) as i32 as f32) / (i32::MAX as f32);
        }
        v
    }

    fn make_int8_weight(n: usize, k: usize, seed: u64) -> INT8Weight {
        let n_groups = k.div_ceil(Q4_GROUP_SIZE);
        let mut data = vec![0u8; n * k];
        let mut s = seed;
        for x in data.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *x = (s >> 56) as u8;
        }
        let scales: Vec<f32> = (0..n * n_groups).map(|i| 0.001 + (i as f32) * 0.0001).collect();
        INT8Weight { data, scales, n, k }
    }

    fn make_bf16_weight(n: usize, k: usize, seed: u64) -> BF16Weight {
        let data = make_random_f32(n * k, seed);
        BF16Weight { data, n, k }
    }

    fn bench_fn(name: &str, iters: usize, mut f: impl FnMut()) -> f64 {
        // warmup
        for _ in 0..3 { f(); }
        let start = Instant::now();
        for _ in 0..iters { f(); }
        let elapsed = start.elapsed();
        let us = elapsed.as_secs_f64() * 1e6 / iters as f64;
        eprintln!("  {:<40} {:>8.1} us/iter ({} iters)", name, us, iters);
        us
    }

    #[test]
    fn bench_int8_vs_bf16_router() {
        // Router dimensions: Qwen3-30B-A3B = [128 experts, 2048 hidden]
        let n = 128;
        let k = 2048;

        eprintln!("\n=== INT8 vs BF16 matmul benchmark (router [{}, {}]) ===\n", n, k);

        let w_int8 = make_int8_weight(n, k, 42);
        let w_bf16 = make_bf16_weight(n, k, 42);

        for &m in &[1, 4, 16, 64, 128, 200] {
            let a = make_random_f32(m * k, 123 + m as u64);
            let mut c_int8 = vec![0.0f32; m * n];
            let mut c_bf16 = vec![0.0f32; m * n];

            eprintln!("--- M={} ({}x{} @ {}x{}) ---", m, m, k, n, k);

            let iters = if m <= 16 { 500 } else { 100 };

            let us_int8 = bench_fn("matmul_int8", iters, || {
                matmul_int8(&a, &w_int8, &mut c_int8, m);
            });

            let us_bf16 = bench_fn("matmul_f32 (AVX512 FMA)", iters, || {
                matmul_f32(&a, &w_bf16, &mut c_bf16, m);
            });

            eprintln!("  ratio INT8/BF16: {:.2}x", us_int8 / us_bf16);
            eprintln!();
        }
    }

    #[test]
    fn bench_int8_breakdown() {
        // Breakdown: quantize vs compute for router dimensions
        let n: usize = 128;
        let k: usize = 2048;
        let m: usize = 200;
        let n_groups = k.div_ceil(Q4_GROUP_SIZE);

        eprintln!("\n=== INT8 matmul breakdown (M={}, N={}, K={}) ===\n", m, n, k);

        let a = make_random_f32(m * k, 99);
        let w = make_int8_weight(n, k, 42);

        // 1. Quantize all M rows
        let iters = 100;
        let mut quants: Vec<(Vec<i8>, Vec<f32>, Vec<i64>)> = Vec::new();
        bench_fn("quantize_x_for_int8 (M rows)", iters, || {
            quants.clear();
            for i in 0..m {
                quants.push(quantize_x_for_int8(&a[i * k..(i + 1) * k]));
            }
        });

        // 2. Compute all dot products (sequential, no parallel_for)
        let quants: Vec<_> = (0..m).map(|i| quantize_x_for_int8(&a[i * k..(i + 1) * k])).collect();
        let mut c = vec![0.0f32; m * n];
        bench_fn("dot_int8_row (M*N dots, sequential)", iters, || {
            for i in 0..m {
                let (ref x_i8, ref x_gs, ref x_gcs) = quants[i];
                for row in 0..n {
                    let w_row = &w.data[row * k..(row + 1) * k];
                    let ws = &w.scales[row * n_groups..(row + 1) * n_groups];
                    c[i * n + row] = dot_int8_row(
                        unsafe { std::slice::from_raw_parts(w_row.as_ptr(), k) },
                        x_i8, ws, x_gs, x_gcs, k,
                    );
                }
            }
        });

        // 3. Single dot_int8_row
        let (ref x_i8, ref x_gs, ref x_gcs) = quants[0];
        let w_row = &w.data[0..k];
        let ws = &w.scales[0..n_groups];
        bench_fn("single dot_int8_row (K=2048)", 10000, || {
            std::hint::black_box(dot_int8_row(
                unsafe { std::slice::from_raw_parts(w_row.as_ptr(), k) },
                x_i8, ws, x_gs, x_gcs, k,
            ));
        });

        // 4. Full matmul_int8 (with parallel_for)
        bench_fn("matmul_int8 (full, parallel)", iters, || {
            matmul_int8(&a, &w, &mut c, m);
        });

        // 5. Single quantize_x_for_int8
        let row = &a[0..k];
        bench_fn("single quantize_x_for_int8 (K=2048)", 10000, || {
            std::hint::black_box(quantize_x_for_int8(row));
        });
    }
}
