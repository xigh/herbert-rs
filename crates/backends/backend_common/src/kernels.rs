//! Shared utility kernels used by threaded BF16/Q8/Q4 backends.

use herbert_core::error::{HerbertError, Result};
use herbert_core::tensor::{bf16_to_f32, BF16};

// ── Feature detection (cached) ───────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
static AVX512F_DETECTED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(target_arch = "x86_64")]
pub fn has_avx512f() -> bool {
    let v = AVX512F_DETECTED.load(std::sync::atomic::Ordering::Relaxed);
    if v != 0 {
        return v == 1;
    }
    let detected = is_x86_feature_detected!("avx512f");
    AVX512F_DETECTED.store(if detected { 1 } else { 2 }, std::sync::atomic::Ordering::Relaxed);
    detected
}

#[cfg(target_arch = "x86_64")]
static AVX512BF16_DETECTED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(target_arch = "x86_64")]
pub fn has_avx512bf16() -> bool {
    let v = AVX512BF16_DETECTED.load(std::sync::atomic::Ordering::Relaxed);
    if v != 0 {
        return v == 1;
    }
    let detected = is_x86_feature_detected!("avx512bf16");
    AVX512BF16_DETECTED.store(if detected { 1 } else { 2 }, std::sync::atomic::Ordering::Relaxed);
    detected
}

#[cfg(target_arch = "x86_64")]
static AVX2_FMA_DETECTED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(target_arch = "x86_64")]
pub fn has_avx2_fma() -> bool {
    let v = AVX2_FMA_DETECTED.load(std::sync::atomic::Ordering::Relaxed);
    if v != 0 {
        return v == 1;
    }
    let detected = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    AVX2_FMA_DETECTED.store(if detected { 1 } else { 2 }, std::sync::atomic::Ordering::Relaxed);
    detected
}

// ── RMS Norm ────────────────────────────────────────────────────────────────

/// RMS normalization with BF16 weights (scalar fallback).
#[inline(never)]
pub fn rms_norm_bf16_scalar(input: &[f32], weight: &[BF16], output: &mut [f32], eps: f32) {
    let dim = input.len();
    let mut sum_sq = 0.0f32;
    for &x in input {
        sum_sq += x * x;
    }
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        output[i] = input[i] * inv_rms * bf16_to_f32(weight[i]);
    }
}

/// AVX-512 RMS normalization with BF16 weights.
/// 4× unrolled: processes 64 elements per iteration.
/// BF16→f32: zero-extend u16→u32, shift left 16, reinterpret as f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn rms_norm_bf16_avx512(input: &[f32], weight: &[BF16], output: &mut [f32], eps: f32) {
    use std::arch::x86_64::*;

    let dim = input.len();
    let inp = input.as_ptr();
    let wt = weight.as_ptr();
    let out = output.as_mut_ptr();
    let chunks = dim / 64;

    // Pass 1: sum of squares — 4× unrolled FMA
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
    // Remainder (dim not multiple of 64)
    let done = chunks * 64;
    for i in done..dim {
        let x = *inp.add(i);
        sum_sq += x * x;
    }

    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    let inv_rms_v = _mm512_set1_ps(inv_rms);

    // Pass 2: output = input * inv_rms * bf16_to_f32(weight)
    // BF16→f32: load 16×u16 → zero-extend to 32-bit → shift left 16 → cast
    for i in 0..chunks {
        let base = i * 64;
        macro_rules! scale {
            ($off:expr) => {{
                let v = _mm512_loadu_ps(inp.add(base + $off));
                let w16 = _mm256_loadu_si256(wt.add(base + $off) as *const __m256i);
                let w32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(w16), 16);
                let wf = _mm512_castsi512_ps(w32);
                _mm512_storeu_ps(out.add(base + $off),
                    _mm512_mul_ps(_mm512_mul_ps(v, inv_rms_v), wf));
            }};
        }
        scale!(0);
        scale!(16);
        scale!(32);
        scale!(48);
    }
    // Remainder
    for i in done..dim {
        *out.add(i) = *inp.add(i) * inv_rms * bf16_to_f32(*wt.add(i));
    }
}

/// AVX2+FMA RMS normalization with BF16 weights.
/// 4× unrolled: processes 32 elements per iteration (4 × 8 YMM).
/// BF16→f32: zero-extend u16→u32, shift left 16, reinterpret as f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rms_norm_bf16_avx2(input: &[f32], weight: &[BF16], output: &mut [f32], eps: f32) {
    use std::arch::x86_64::*;

    let dim = input.len();
    let inp = input.as_ptr();
    let wt = weight.as_ptr();
    let out = output.as_mut_ptr();
    let chunks = dim / 32;

    // Pass 1: sum of squares — 4× unrolled FMA
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    for i in 0..chunks {
        let base = i * 32;
        let v0 = _mm256_loadu_ps(inp.add(base));
        let v1 = _mm256_loadu_ps(inp.add(base + 8));
        let v2 = _mm256_loadu_ps(inp.add(base + 16));
        let v3 = _mm256_loadu_ps(inp.add(base + 24));
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        acc2 = _mm256_fmadd_ps(v2, v2, acc2);
        acc3 = _mm256_fmadd_ps(v3, v3, acc3);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    // Horizontal sum of ymm → scalar
    let hi128 = _mm256_extractf128_ps(acc0, 1);
    let lo128 = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo128, hi128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let mut sum_sq = _mm_cvtss_f32(sum128);
    // Remainder (dim not multiple of 32)
    let done = chunks * 32;
    for i in done..dim {
        let x = *inp.add(i);
        sum_sq += x * x;
    }

    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    let inv_rms_v = _mm256_set1_ps(inv_rms);

    // Pass 2: output = input * inv_rms * bf16_to_f32(weight)
    // BF16→f32: load 8×u16 → zero-extend to 32-bit → shift left 16 → cast
    for i in 0..chunks {
        let base = i * 32;
        macro_rules! scale {
            ($off:expr) => {{
                let v = _mm256_loadu_ps(inp.add(base + $off));
                let w16 = _mm_loadu_si128(wt.add(base + $off) as *const __m128i);
                let w32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(w16), 16);
                let wf = _mm256_castsi256_ps(w32);
                _mm256_storeu_ps(out.add(base + $off),
                    _mm256_mul_ps(_mm256_mul_ps(v, inv_rms_v), wf));
            }};
        }
        scale!(0);
        scale!(8);
        scale!(16);
        scale!(24);
    }
    // Remainder
    for i in done..dim {
        *out.add(i) = *inp.add(i) * inv_rms * bf16_to_f32(*wt.add(i));
    }
}

/// RMS normalization with BF16 weights — auto-dispatches to AVX-512 / AVX2 when available.
pub fn rms_norm_bf16(input: &[f32], weight: &[BF16], output: &mut [f32], eps: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512f() && input.len() >= 64 {
            unsafe { rms_norm_bf16_avx512(input, weight, output, eps); }
            return;
        }
        if has_avx2_fma() && input.len() >= 32 {
            unsafe { rms_norm_bf16_avx2(input, weight, output, eps); }
            return;
        }
    }
    rms_norm_bf16_scalar(input, weight, output, eps);
}

// ── Fused RMS Norm + i8 Quantization ─────────────────────────────────────

/// Fused RMS normalization + per-group i8 quantization (scalar fallback).
/// Computes norm output (f32), quantized i8 vector, per-group scales, and col_sums in two passes.
#[inline(never)]
pub fn rms_norm_bf16_and_quantize_scalar(
    input: &[f32],
    weight: &[BF16],
    eps: f32,
    norm_out: &mut [f32],
    x_i8: &mut [i8],
    group_scales: &mut [f32],
    col_sums: &mut [i64],
    group_size: usize,
) {
    let dim = input.len();
    // Pass 1: RMS norm
    let mut sum_sq = 0.0f32;
    for &x in input {
        sum_sq += x * x;
    }
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        norm_out[i] = input[i] * inv_rms * bf16_to_f32(weight[i]);
    }
    // Pass 2: Per-group i8 quantize
    let n_groups = dim.div_ceil(group_size);
    for g in 0..n_groups {
        let start = g * group_size;
        let end = (start + group_size).min(dim);
        let mut abs_max = 0.0f32;
        for i in start..end {
            let av = norm_out[i].abs();
            if av > abs_max { abs_max = av; }
        }
        let scale = if abs_max > f32::EPSILON { abs_max / 127.0 } else { 1.0 };
        group_scales[g] = scale;
        let inv_scale = 1.0 / scale;
        let mut cs = 0i64;
        for i in start..end {
            let q = (norm_out[i] * inv_scale).round().clamp(-127.0, 127.0) as i8;
            x_i8[i] = q;
            cs += q as i64;
        }
        col_sums[g] = cs;
    }
}

/// AVX-512 fused RMS normalization + per-group i8 quantization.
/// Pass 1: sum_sq (4× unrolled FMA). Pass 2: per-group norm + quantize + col_sum.
/// group_size must be 32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline(never)]
pub unsafe fn rms_norm_bf16_and_quantize_avx512(
    input: &[f32],
    weight: &[BF16],
    eps: f32,
    norm_out: &mut [f32],
    x_i8: &mut [i8],
    group_scales: &mut [f32],
    col_sums: &mut [i64],
    group_size: usize,
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(group_size, 32);
    let dim = input.len();
    let inp = input.as_ptr();
    let wt = weight.as_ptr();
    let out = norm_out.as_mut_ptr();
    let qi8 = x_i8.as_mut_ptr();

    // Pass 1: sum of squares — 4× unrolled FMA
    let chunks64 = dim / 64;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    for i in 0..chunks64 {
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
    let done = chunks64 * 64;
    for i in done..dim {
        let x = *inp.add(i);
        sum_sq += x * x;
    }

    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    let inv_rms_v = _mm512_set1_ps(inv_rms);
    let abs_mask = _mm512_set1_epi32(0x7FFF_FFFFi32);

    // Pass 2: per-group (32 elements = 2 ZMM) norm + quantize + col_sum
    let n_groups = dim.div_ceil(group_size);
    for g in 0..n_groups {
        let base = g * 32;
        let remain = dim - base;

        if remain >= 32 {
            // Norm: output = input * inv_rms * bf16_weight
            let v0 = _mm512_loadu_ps(inp.add(base));
            let w0_16 = _mm256_loadu_si256(wt.add(base) as *const __m256i);
            let w0_32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(w0_16), 16);
            let w0_f = _mm512_castsi512_ps(w0_32);
            let normed0 = _mm512_mul_ps(_mm512_mul_ps(v0, inv_rms_v), w0_f);

            let v1 = _mm512_loadu_ps(inp.add(base + 16));
            let w1_16 = _mm256_loadu_si256(wt.add(base + 16) as *const __m256i);
            let w1_32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(w1_16), 16);
            let w1_f = _mm512_castsi512_ps(w1_32);
            let normed1 = _mm512_mul_ps(_mm512_mul_ps(v1, inv_rms_v), w1_f);

            // Store normed f32
            _mm512_storeu_ps(out.add(base), normed0);
            _mm512_storeu_ps(out.add(base + 16), normed1);

            // max_abs across 32 normed values
            let abs0 = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(normed0), abs_mask));
            let abs1 = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(normed1), abs_mask));
            let max_abs = _mm512_reduce_max_ps(_mm512_max_ps(abs0, abs1));

            let (scale, inv_scale) = if max_abs > f32::EPSILON {
                (max_abs / 127.0, 127.0 / max_abs)
            } else {
                (1.0, 1.0)
            };
            group_scales[g] = scale;

            let inv_scale_v = _mm512_set1_ps(inv_scale);

            // Quantize from normed registers (no re-read from memory)
            let q0_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(normed0, inv_scale_v));
            let q1_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(normed1, inv_scale_v));

            // Pack i32→i8 with saturation
            let q0_i8 = _mm512_cvtsepi32_epi8(q0_i32);
            let q1_i8 = _mm512_cvtsepi32_epi8(q1_i32);
            _mm_storeu_si128(qi8.add(base) as *mut __m128i, q0_i8);
            _mm_storeu_si128(qi8.add(base + 16) as *mut __m128i, q1_i8);

            // Column sum
            let sum0 = _mm512_reduce_add_epi32(q0_i32);
            let sum1 = _mm512_reduce_add_epi32(q1_i32);
            col_sums[g] = (sum0 + sum1) as i64;
        } else {
            // Scalar remainder
            let mut abs_max = 0.0f32;
            for i in base..dim {
                let normed = *inp.add(i) * inv_rms * bf16_to_f32(*wt.add(i));
                *out.add(i) = normed;
                let av = normed.abs();
                if av > abs_max { abs_max = av; }
            }
            let scale = if abs_max > f32::EPSILON { abs_max / 127.0 } else { 1.0 };
            group_scales[g] = scale;
            let inv_scale = 1.0 / scale;
            let mut cs = 0i64;
            for i in base..dim {
                let q = (*out.add(i) * inv_scale).round().clamp(-127.0, 127.0) as i8;
                *qi8.add(i) = q;
                cs += q as i64;
            }
            col_sums[g] = cs;
        }
    }
}

/// Fused RMS normalization + i8 quantization — auto-dispatches to AVX-512.
pub fn rms_norm_bf16_and_quantize(
    input: &[f32],
    weight: &[BF16],
    eps: f32,
    norm_out: &mut [f32],
    x_i8: &mut [i8],
    group_scales: &mut [f32],
    col_sums: &mut [i64],
    group_size: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512f() && input.len() >= 64 {
            unsafe {
                rms_norm_bf16_and_quantize_avx512(
                    input, weight, eps, norm_out, x_i8, group_scales, col_sums, group_size,
                );
            }
            return;
        }
    }
    rms_norm_bf16_and_quantize_scalar(
        input, weight, eps, norm_out, x_i8, group_scales, col_sums, group_size,
    );
}

// ── Fused RMS Norm + Residual Add ─────────────────────────────────────

/// Fused residual add + RMS normalization (scalar fallback).
/// Computes: a[i] += b[i], then output[i] = a[i] * inv_rms * weight[i].
/// Saves one full load+store of `a` vs separate residual add + rms_norm.
#[inline(never)]
pub fn rms_norm_residual_bf16_scalar(
    a: &mut [f32],
    b: &[f32],
    weight: &[BF16],
    output: &mut [f32],
    eps: f32,
) {
    let dim = a.len();
    // Pass 1: residual add + sum of squares
    let mut sum_sq = 0.0f32;
    for i in 0..dim {
        let val = a[i] + b[i];
        a[i] = val;
        sum_sq += val * val;
    }
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    // Pass 2: normalize
    for i in 0..dim {
        output[i] = a[i] * inv_rms * bf16_to_f32(weight[i]);
    }
}

/// AVX-512 fused residual add + RMS normalization with BF16 weights.
/// Pass 1 (4× unrolled): a[i] += b[i], accumulate sum_sq via FMA.
/// Pass 2 (4× unrolled): output[i] = a[i] * inv_rms * bf16_to_f32(weight[i]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn rms_norm_residual_bf16_avx512(
    a: &mut [f32],
    b: &[f32],
    weight: &[BF16],
    output: &mut [f32],
    eps: f32,
) {
    use std::arch::x86_64::*;

    let dim = a.len();
    let a_ptr = a.as_mut_ptr();
    let b_ptr = b.as_ptr();
    let wt = weight.as_ptr();
    let out = output.as_mut_ptr();
    let chunks = dim / 64;

    // Pass 1: residual add + sum of squares — 4× unrolled FMA
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
    // Remainder
    let done = chunks * 64;
    for i in done..dim {
        let val = *a_ptr.add(i) + *b_ptr.add(i);
        *a_ptr.add(i) = val;
        sum_sq += val * val;
    }

    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    let inv_rms_v = _mm512_set1_ps(inv_rms);

    // Pass 2: output = a * inv_rms * bf16_to_f32(weight)
    for i in 0..chunks {
        let base = i * 64;
        macro_rules! scale {
            ($off:expr) => {{
                let v = _mm512_loadu_ps(a_ptr.add(base + $off));
                let w16 = _mm256_loadu_si256(wt.add(base + $off) as *const __m256i);
                let w32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(w16), 16);
                let wf = _mm512_castsi512_ps(w32);
                _mm512_storeu_ps(out.add(base + $off),
                    _mm512_mul_ps(_mm512_mul_ps(v, inv_rms_v), wf));
            }};
        }
        scale!(0);
        scale!(16);
        scale!(32);
        scale!(48);
    }
    // Remainder
    for i in done..dim {
        *out.add(i) = *a_ptr.add(i) * inv_rms * bf16_to_f32(*wt.add(i));
    }
}

/// AVX2+FMA fused residual add + RMS normalization with BF16 weights.
/// 4× unrolled: processes 32 elements per iteration (4 × 8 YMM).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rms_norm_residual_bf16_avx2(
    a: &mut [f32],
    b: &[f32],
    weight: &[BF16],
    output: &mut [f32],
    eps: f32,
) {
    use std::arch::x86_64::*;

    let dim = a.len();
    let a_ptr = a.as_mut_ptr();
    let b_ptr = b.as_ptr();
    let wt = weight.as_ptr();
    let out = output.as_mut_ptr();
    let chunks = dim / 32;

    // Pass 1: residual add + sum of squares — 4× unrolled FMA
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    for i in 0..chunks {
        let base = i * 32;
        macro_rules! residual_sq {
            ($acc:ident, $off:expr) => {{
                let av = _mm256_loadu_ps(a_ptr.add(base + $off));
                let bv = _mm256_loadu_ps(b_ptr.add(base + $off));
                let sum = _mm256_add_ps(av, bv);
                _mm256_storeu_ps(a_ptr.add(base + $off), sum);
                $acc = _mm256_fmadd_ps(sum, sum, $acc);
            }};
        }
        residual_sq!(acc0, 0);
        residual_sq!(acc1, 8);
        residual_sq!(acc2, 16);
        residual_sq!(acc3, 24);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    // Horizontal sum of ymm → scalar
    let hi128 = _mm256_extractf128_ps(acc0, 1);
    let lo128 = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo128, hi128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let mut sum_sq = _mm_cvtss_f32(sum128);
    // Remainder
    let done = chunks * 32;
    for i in done..dim {
        let val = *a_ptr.add(i) + *b_ptr.add(i);
        *a_ptr.add(i) = val;
        sum_sq += val * val;
    }

    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    let inv_rms_v = _mm256_set1_ps(inv_rms);

    // Pass 2: output = a * inv_rms * bf16_to_f32(weight)
    for i in 0..chunks {
        let base = i * 32;
        macro_rules! scale {
            ($off:expr) => {{
                let v = _mm256_loadu_ps(a_ptr.add(base + $off));
                let w16 = _mm_loadu_si128(wt.add(base + $off) as *const __m128i);
                let w32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(w16), 16);
                let wf = _mm256_castsi256_ps(w32);
                _mm256_storeu_ps(out.add(base + $off),
                    _mm256_mul_ps(_mm256_mul_ps(v, inv_rms_v), wf));
            }};
        }
        scale!(0);
        scale!(8);
        scale!(16);
        scale!(24);
    }
    // Remainder
    for i in done..dim {
        *out.add(i) = *a_ptr.add(i) * inv_rms * bf16_to_f32(*wt.add(i));
    }
}

/// Fused residual add + RMS normalization — auto-dispatches to AVX-512 / AVX2.
pub fn rms_norm_residual_bf16(
    a: &mut [f32],
    b: &[f32],
    weight: &[BF16],
    output: &mut [f32],
    eps: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512f() && a.len() >= 64 {
            unsafe { rms_norm_residual_bf16_avx512(a, b, weight, output, eps); }
            return;
        }
        if has_avx2_fma() && a.len() >= 32 {
            unsafe { rms_norm_residual_bf16_avx2(a, b, weight, output, eps); }
            return;
        }
    }
    rms_norm_residual_bf16_scalar(a, b, weight, output, eps);
}

/// Layer normalization with BF16 weights and f32 bias.
/// If `bias` is empty, zero bias is assumed.
pub fn layer_norm_bf16(input: &[f32], weight: &[BF16], bias: &[f32], output: &mut [f32], eps: f32) {
    let dim = input.len();
    let mut mean = 0.0f32;
    for &x in input {
        mean += x;
    }
    mean /= dim as f32;
    let mut var = 0.0f32;
    for &x in input {
        let d = x - mean;
        var += d * d;
    }
    var /= dim as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    if bias.len() >= dim {
        for i in 0..dim {
            output[i] = (input[i] - mean) * inv_std * bf16_to_f32(weight[i]) + bias[i];
        }
    } else {
        for i in 0..dim {
            output[i] = (input[i] - mean) * inv_std * bf16_to_f32(weight[i]);
        }
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU: silu(gate) * up.
pub fn swiglu_inplace(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    for i in 0..gate.len() {
        gate[i] = silu(gate[i]) * up[i];
    }
}

/// Gemma-style RMS normalization: output = (1 + weight) * (input / sqrt(mean(input²) + eps)).
pub fn rms_norm_gemma_bf16(input: &[f32], weight: &[BF16], output: &mut [f32], eps: f32) {
    let dim = input.len();
    let mut sum_sq = 0.0f32;
    for &x in input {
        sum_sq += x * x;
    }
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        output[i] = input[i] * inv_rms * (1.0 + bf16_to_f32(weight[i]));
    }
}

/// GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
/// Applied as gated activation: gate[i] = gelu_tanh(gate[i]) * up[i].
pub fn gelu_tanh_inplace(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044715;
    for i in 0..gate.len() {
        let x = gate[i];
        let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        gate[i] = 0.5 * x * (1.0 + inner.tanh()) * up[i];
    }
}

/// Softmax with numerical stability.
pub fn softmax_inplace(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for val in logits.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }
    if sum > 0.0 {
        for val in logits.iter_mut() {
            *val /= sum;
        }
    }
}

/// Apply RoPE for a single position without bounds checks.
///
/// `rotary_ndims`: number of dimensions to rotate (= head_dim for full RoPE,
/// < head_dim for partial RoPE like Phi-4). Non-rotated dims are left unchanged.
/// Callers must ensure `cos` and `sin` have at least `rotary_ndims / 2` values.
#[inline]
pub fn apply_rope_position_unchecked(
    qk: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    num_heads: usize,
    head_dim: usize,
    rotary_ndims: usize,
) {
    let half_rot = rotary_ndims / 2;
    for h in 0..num_heads {
        let offset = h * head_dim;
        for j in 0..half_rot {
            let x0 = qk[offset + j];
            let x1 = qk[offset + j + half_rot];
            let cos_val = cos[j];
            let sin_val = sin[j];
            qk[offset + j] = x0 * cos_val - x1 * sin_val;
            qk[offset + j + half_rot] = x0 * sin_val + x1 * cos_val;
        }
    }
}

/// Apply RoPE to a single position (decode).
pub fn apply_rope_single(
    qk: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    num_heads: usize,
    head_dim: usize,
    rotary_ndims: usize,
) -> Result<()> {
    let required = num_heads * head_dim;
    if qk.len() < required {
        return Err(HerbertError::Backend(format!(
            "qk buffer too small: need {} ({} heads × {} head_dim), got {}",
            required, num_heads, head_dim, qk.len()
        )));
    }
    let half_rot = rotary_ndims / 2;
    if cos.len() < half_rot || sin.len() < half_rot {
        return Err(HerbertError::Backend(format!(
            "RoPE cache slice too small: need {}, got cos={} sin={}",
            half_rot,
            cos.len(),
            sin.len()
        )));
    }

    apply_rope_position_unchecked(qk, cos, sin, num_heads, head_dim, rotary_ndims);
    Ok(())
}

/// Validate the requested RoPE batch window and return `half_rot` (rotary_ndims/2).
pub fn validate_rope_batch_window(
    cos_cache: &[f32],
    sin_cache: &[f32],
    seq_len: usize,
    rotary_ndims: usize,
    start_pos: usize,
) -> Result<usize> {
    let half_rot = rotary_ndims / 2;
    let required = start_pos
        .checked_add(seq_len)
        .and_then(|v| v.checked_mul(half_rot))
        .ok_or_else(|| HerbertError::Backend("RoPE size overflow".to_string()))?;

    if cos_cache.len() < required || sin_cache.len() < required {
        return Err(HerbertError::Backend(format!(
            "RoPE cache too small for requested window: need {}, got cos={} sin={}",
            required,
            cos_cache.len(),
            sin_cache.len()
        )));
    }

    Ok(half_rot)
}
