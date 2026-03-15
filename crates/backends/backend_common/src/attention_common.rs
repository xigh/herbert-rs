//! Shared attention helpers for CPU inference backends.

use herbert_core::tensor::bf16_to_f32;

use crate::kernels::softmax_inplace;

pub const DECODE_HEAD_PAR_MIN_HEADS: usize = 8;
pub const DECODE_HEAD_PAR_CONTEXT_THRESHOLD: usize = 256;
pub const DECODE_ATTN_SMALL_CONTEXT_THRESHOLD: usize = 64;

#[cfg(target_arch = "x86_64")]
pub const DECODE_COMMON_PF_DIST: usize = 4;

#[inline]
pub fn decode_head_parallel_workers(pool_workers: usize, num_heads: usize) -> usize {
    pool_workers.min(num_heads).max(1)
}

#[inline]
pub fn decode_head_attention(
    q_head: &[f32],
    cached_k: &[u16],
    cached_v: &[u16],
    kv_dim: usize,
    head_dim: usize,
    kv_head_idx: usize,
    scale: f32,
    scores: &mut [f32],
    out_head: &mut [f32],
) {
    let kv_head_base = kv_head_idx * head_dim;
    let _cached_len = scores.len();
    let mut k_row_base = kv_head_base;

    // Check SIMD availability once for BF16 decode
    #[cfg(target_arch = "x86_64")]
    let use_avx2_bf16 = !has_avx512f() && has_avx2_fma();

    #[allow(clippy::unused_enumerate_index)]
    for (_j, score_slot) in scores.iter_mut().enumerate() {
        #[cfg(all(target_arch = "x86_64", not(feature = "no-sw-prefetch")))]
        {
            if _j + DECODE_COMMON_PF_DIST < _cached_len {
                let pf_k = kv_head_base + (_j + DECODE_COMMON_PF_DIST) * kv_dim;
                unsafe {
                    core::arch::x86_64::_mm_prefetch(
                        cached_k.as_ptr().add(pf_k) as *const i8,
                        core::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if use_avx2_bf16 {
                let raw = unsafe {
                    avx2_attn_dot_qf32_kbf16(
                        q_head.as_ptr(),
                        cached_k.as_ptr().add(k_row_base),
                        head_dim as u64,
                    )
                };
                *score_slot = raw * scale;
                k_row_base += kv_dim;
                continue;
            }
        }
        let mut score = 0.0f32;
        for d in 0..head_dim {
            score += q_head[d] * bf16_to_f32(cached_k[k_row_base + d]);
        }
        *score_slot = score * scale;
        k_row_base += kv_dim;
    }
    softmax_inplace(scores);
    if scores.len() <= DECODE_ATTN_SMALL_CONTEXT_THRESHOLD {
        for d in 0..head_dim {
            let mut sum = 0.0f32;
            let mut v_col_offset = kv_head_base + d;
            for &score in scores.iter() {
                sum += score * bf16_to_f32(cached_v[v_col_offset]);
                v_col_offset += kv_dim;
            }
            out_head[d] = sum;
        }
    } else {
        out_head.fill(0.0);
        let mut v_row_base = kv_head_base;
        #[allow(clippy::unused_enumerate_index)]
        for (_j, &score) in scores.iter().enumerate() {
            #[cfg(all(target_arch = "x86_64", not(feature = "no-sw-prefetch")))]
            {
                if _j + DECODE_COMMON_PF_DIST < _cached_len {
                    let pf_v = kv_head_base + (_j + DECODE_COMMON_PF_DIST) * kv_dim;
                    unsafe {
                        core::arch::x86_64::_mm_prefetch(
                            cached_v.as_ptr().add(pf_v) as *const i8,
                            core::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if use_avx2_bf16 {
                    unsafe {
                        avx2_attn_sv_accum_bf16(
                            out_head.as_mut_ptr(),
                            cached_v.as_ptr().add(v_row_base),
                            score,
                            head_dim as u64,
                        );
                    }
                    v_row_base += kv_dim;
                    continue;
                }
            }
            for d in 0..head_dim {
                out_head[d] += score * bf16_to_f32(cached_v[v_row_base + d]);
            }
            v_row_base += kv_dim;
        }
    }
}

/// Helper: determines if decode heads should be parallelized.
#[inline]
pub fn should_parallelize_decode_heads(
    num_heads: usize,
    cached_len: usize,
    workers: usize,
) -> bool {
    if workers <= 1 || num_heads < DECODE_HEAD_PAR_MIN_HEADS || cached_len == 0 {
        return false;
    }
    cached_len >= DECODE_HEAD_PAR_CONTEXT_THRESHOLD
}

/// F32 decode attention kernel (no conversion, pure Rust).
#[inline]
pub fn decode_head_attention_f32(
    q_head: &[f32],
    cached_k_head: &[f32],
    cached_v_head: &[f32],
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
    out_head: &mut [f32],
) {
    let cached_len = scores.len();
    for j in 0..cached_len {
        let kv_offset = j * head_dim;
        let mut score = 0.0f32;
        for d in 0..head_dim {
            score += q_head[d] * cached_k_head[kv_offset + d];
        }
        scores[j] = score * scale;
    }
    softmax_inplace(scores);
    if cached_len <= DECODE_ATTN_SMALL_CONTEXT_THRESHOLD {
        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for j in 0..cached_len {
                sum += scores[j] * cached_v_head[j * head_dim + d];
            }
            out_head[d] = sum;
        }
    } else {
        out_head.fill(0.0);
        for j in 0..cached_len {
            let kv_offset = j * head_dim;
            let score = scores[j];
            for d in 0..head_dim {
                out_head[d] += score * cached_v_head[kv_offset + d];
            }
        }
    }
}

/// Int8 decode attention kernel (symmetric per-position per-head quantization).
/// Dequantizes on-the-fly during computation using SIMD when available.
#[inline]
pub fn decode_head_attention_int8(
    q_head: &[f32],
    cached_k_head: &[i8],
    cached_v_head: &[i8],
    k_scales: &[f32],
    v_scales: &[f32],
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
    out_head: &mut [f32],
) {
    // ========== Phase 1: Q·K^T with dequantization (SIMD optimized) ==========
    #[cfg(feature = "profile-kvcache-int8")]
    let t1 = std::time::Instant::now();

    for (j, score_slot) in scores.iter_mut().enumerate() {
        let k_scale = k_scales[j];
        let kv_offset = j * head_dim;
        let k_slice = &cached_k_head[kv_offset..kv_offset + head_dim];
        *score_slot = dot_qf32_ki8(q_head, k_slice, k_scale) * scale;
    }

    #[cfg(feature = "profile-kvcache-int8")]
    crate::profiler::push_kvcache_int8_scores(t1.elapsed().as_micros() as u64);

    // ========== Phase 2: Softmax ==========
    #[cfg(feature = "profile-kvcache-int8")]
    let t2 = std::time::Instant::now();

    softmax_inplace(scores);

    #[cfg(feature = "profile-kvcache-int8")]
    crate::profiler::push_kvcache_int8_softmax(t2.elapsed().as_micros() as u64);

    // ========== Phase 3: Output aggregation with dequantization (SIMD optimized) ==========
    #[cfg(feature = "profile-kvcache-int8")]
    let t3 = std::time::Instant::now();

    if scores.len() <= DECODE_ATTN_SMALL_CONTEXT_THRESHOLD {
        // Column reduction for small context: iterate over dimension
        for d in 0..head_dim {
            let mut sum = 0.0f32;
            let mut v_col_offset = d;
            for (j, &score) in scores.iter().enumerate() {
                let v_deq = cached_v_head[v_col_offset] as f32 * v_scales[j];
                sum += score * v_deq;
                v_col_offset += head_dim;
            }
            out_head[d] = sum;
        }
    } else {
        // Row reduction for large context: iterate over cached positions
        out_head.fill(0.0);
        for (j, &score) in scores.iter().enumerate() {
            let v_scale = v_scales[j];
            let kv_offset = j * head_dim;
            let v_slice = &cached_v_head[kv_offset..kv_offset + head_dim];
            sv_accum_i8(out_head, v_slice, score, v_scale);
        }
    }

    #[cfg(feature = "profile-kvcache-int8")]
    crate::profiler::push_kvcache_int8_output(t3.elapsed().as_micros() as u64);

}

/// Int4 decode attention kernel (Q4_0 group quantization, group_size=32).
/// Per-group scales: num_groups f32 per position (num_groups = head_dim/32).
#[inline]
pub fn decode_head_attention_int4(
    q_head: &[f32],
    cached_k_head: &[u8],
    cached_v_head: &[u8],
    k_scales: &[f32],       // num_groups * cached_len, flattened [pos0_g0, pos0_g1, ..., pos1_g0, ...]
    v_scales: &[f32],
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
    out_head: &mut [f32],
) {
    let packed_per_pos = head_dim / 2;
    let num_groups = head_dim / 32;

    // ========== Phase 1: Q·K^T with INT4 dequantization ==========
    for (j, score_slot) in scores.iter_mut().enumerate() {
        let kv_offset = j * packed_per_pos;
        let k_packed = &cached_k_head[kv_offset..kv_offset + packed_per_pos];
        let k_group_scales = &k_scales[j * num_groups..(j + 1) * num_groups];
        *score_slot = dot_qf32_ki4(q_head, k_packed, k_group_scales, head_dim) * scale;
    }

    // ========== Phase 2: Softmax ==========
    softmax_inplace(scores);

    // ========== Phase 3: Output aggregation with INT4 dequantization ==========
    out_head.fill(0.0);
    for (j, &score) in scores.iter().enumerate() {
        let kv_offset = j * packed_per_pos;
        let v_packed = &cached_v_head[kv_offset..kv_offset + packed_per_pos];
        let v_group_scales = &v_scales[j * num_groups..(j + 1) * num_groups];
        sv_accum_i4(out_head, v_packed, score, v_group_scales, head_dim);
    }
}

// ============================================================================
// SIMD Dispatch Functions for int8 KV Cache Operations
// ============================================================================

// ============================================================================
// SIMD Kernel Declarations (Module Level - for proper linkage)
// ============================================================================

#[cfg(target_arch = "x86_64")]
extern "C" {
    fn avx512_attn_dot_qf32_ki8(q_ptr: *const f32, k_ptr: *const i8, dim: u64) -> f32;
    fn avx512_attn_sv_accum_i8(out_ptr: *mut f32, v_ptr: *const i8, score: f32, dim: u64);
    fn avx2_attn_dot_qf32_ki8(q_ptr: *const f32, k_ptr: *const i8, dim: u64) -> f32;
    fn avx2_attn_sv_accum_i8(out_ptr: *mut f32, v_ptr: *const i8, score: f32, dim: u64);
    fn avx2_attn_dot_qf32_kbf16(q_ptr: *const f32, k_ptr: *const u16, dim: u64) -> f32;
    fn avx2_attn_sv_accum_bf16(out_ptr: *mut f32, v_ptr: *const u16, score: f32, dim: u64);
    fn avx512_attn_dot_qf32_ki4(q_ptr: *const f32, k_packed: *const u8, k_scales: *const f32, dim: u64) -> f32;
    fn avx512_attn_sv_accum_i4(out_ptr: *mut f32, v_packed: *const u8, v_scales: *const f32, score: f32, dim: u64);
    fn avx2_attn_dot_qf32_ki4(q_ptr: *const f32, k_packed: *const u8, k_scales: *const f32, dim: u64) -> f32;
    fn avx2_attn_sv_accum_i4(out_ptr: *mut f32, v_packed: *const u8, v_scales: *const f32, score: f32, dim: u64);
}

#[cfg(target_arch = "x86_64")]
use crate::kernels::{has_avx512f, has_avx2_fma};

/// Compute dot(q_f32, k_i8) * k_scale using SIMD if available.
#[inline(always)]
pub fn dot_qf32_ki8(q: &[f32], k: &[i8], scale: f32) -> f32 {
    debug_assert_eq!(q.len(), k.len());
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512f() {
            let raw = unsafe {
                avx512_attn_dot_qf32_ki8(q.as_ptr(), k.as_ptr(), q.len() as u64)
            };
            return raw * scale;
        }
        if has_avx2_fma() {
            let raw = unsafe {
                avx2_attn_dot_qf32_ki8(q.as_ptr(), k.as_ptr(), q.len() as u64)
            };
            return raw * scale;
        }
        // Scalar fallback
        let mut acc = 0.0f32;
        for (q_v, k_v) in q.iter().zip(k.iter()) {
            acc += q_v * (*k_v as f32);
        }
        return acc * scale;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut acc = 0.0f32;
        for (q_v, k_v) in q.iter().zip(k.iter()) {
            acc += q_v * (*k_v as f32);
        }
        acc * scale
    }
}

/// Accumulate: out += score * (v_i8 * v_scale) using SIMD if available.
#[inline(always)]
pub fn sv_accum_i8(out: &mut [f32], v: &[i8], score: f32, v_scale: f32) {
    debug_assert_eq!(out.len(), v.len());
    let weighted = score * v_scale;
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512f() {
            unsafe {
                avx512_attn_sv_accum_i8(out.as_mut_ptr(), v.as_ptr(), weighted, out.len() as u64)
            };
            return;
        }
        if has_avx2_fma() {
            unsafe {
                avx2_attn_sv_accum_i8(out.as_mut_ptr(), v.as_ptr(), weighted, out.len() as u64)
            };
            return;
        }
        // Scalar fallback
        for (o, k_v) in out.iter_mut().zip(v.iter()) {
            *o += weighted * (*k_v as f32);
        }
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        for (o, k_v) in out.iter_mut().zip(v.iter()) {
            *o += weighted * (*k_v as f32);
        }
    }
}

// ============================================================================
// INT4 SIMD Dispatch Functions
// ============================================================================

/// Compute dot(q_f32, k_i4_packed) with per-group scales.
/// k_packed is half-split layout: head_dim/2 bytes for head_dim elements.
/// group_scales has head_dim/32 entries (one per group of 32 elements).
#[inline(always)]
pub fn dot_qf32_ki4(q: &[f32], k_packed: &[u8], group_scales: &[f32], head_dim: usize) -> f32 {
    debug_assert_eq!(q.len(), head_dim);
    debug_assert_eq!(k_packed.len(), head_dim / 2);
    let num_groups = head_dim / 32;
    debug_assert_eq!(group_scales.len(), num_groups);
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512f() {
            return unsafe {
                avx512_attn_dot_qf32_ki4(q.as_ptr(), k_packed.as_ptr(), group_scales.as_ptr(), head_dim as u64)
            };
        }
        if has_avx2_fma() {
            return unsafe {
                avx2_attn_dot_qf32_ki4(q.as_ptr(), k_packed.as_ptr(), group_scales.as_ptr(), head_dim as u64)
            };
        }
    }
    // Scalar fallback: per-group dot product with per-group scale
    let mut acc = 0.0f32;
    for g in 0..num_groups {
        let packed = &k_packed[g * 16..(g + 1) * 16];
        let q_base = g * 32;
        let scale = group_scales[g];
        let mut group_dot = 0.0f32;
        for i in 0..16 {
            let byte = packed[i];
            let lo = ((byte & 0x0F) as i32 - 8) as f32;
            let hi = ((byte >> 4) as i32 - 8) as f32;
            group_dot += q[q_base + i] * lo;
            group_dot += q[q_base + i + 16] * hi;
        }
        acc += group_dot * scale;
    }
    acc
}

/// Accumulate: out += score * dequant(v_i4_packed) with per-group scales.
#[inline(always)]
pub fn sv_accum_i4(out: &mut [f32], v_packed: &[u8], score: f32, group_scales: &[f32], head_dim: usize) {
    debug_assert_eq!(out.len(), head_dim);
    debug_assert_eq!(v_packed.len(), head_dim / 2);
    let num_groups = head_dim / 32;
    debug_assert_eq!(group_scales.len(), num_groups);
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512f() {
            unsafe {
                avx512_attn_sv_accum_i4(out.as_mut_ptr(), v_packed.as_ptr(), group_scales.as_ptr(), score, head_dim as u64)
            };
            return;
        }
        if has_avx2_fma() {
            unsafe {
                avx2_attn_sv_accum_i4(out.as_mut_ptr(), v_packed.as_ptr(), group_scales.as_ptr(), score, head_dim as u64)
            };
            return;
        }
    }
    // Scalar fallback: per-group SV accumulation
    for g in 0..num_groups {
        let packed = &v_packed[g * 16..(g + 1) * 16];
        let out_base = g * 32;
        let weighted = score * group_scales[g];
        for i in 0..16 {
            let byte = packed[i];
            let lo = ((byte & 0x0F) as i32 - 8) as f32;
            let hi = ((byte >> 4) as i32 - 8) as f32;
            out[out_base + i] += weighted * lo;
            out[out_base + i + 16] += weighted * hi;
        }
    }
}

// ============================================================================
// AVX2 Online SV Intrinsics (for prefill paths)
// ============================================================================

/// AVX2 online SV for BF16 V: out[d] = correction * out[d] + alpha * bf16_to_f32(v[d])
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn avx2_online_sv_bf16(
    out: &mut [f32],
    v: *const u16,
    correction: f32,
    alpha: f32,
    dim: usize,
) {
    use std::arch::x86_64::*;

    let corr_v = _mm256_set1_ps(correction);
    let alpha_v = _mm256_set1_ps(alpha);
    let out_ptr = out.as_mut_ptr();
    let mut i = 0;
    while i + 8 <= dim {
        // Load current output
        let o = _mm256_loadu_ps(out_ptr.add(i));
        // BF16→f32: zero-extend u16→u32, shift left 16
        let v16 = _mm_loadu_si128(v.add(i) as *const __m128i);
        let v32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(v16), 16);
        let vf = _mm256_castsi256_ps(v32);
        // out = correction * out + alpha * v
        let result = _mm256_fmadd_ps(alpha_v, vf, _mm256_mul_ps(corr_v, o));
        _mm256_storeu_ps(out_ptr.add(i), result);
        i += 8;
    }
    // Scalar remainder
    while i < dim {
        let v_val = f32::from_bits((*v.add(i) as u32) << 16);
        *out_ptr.add(i) = correction * *out_ptr.add(i) + alpha * v_val;
        i += 1;
    }
}

/// AVX2 online SV for i8 V: out[d] = correction * out[d] + alpha * (v[d] as f32)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn avx2_online_sv_i8(
    out: &mut [f32],
    v: *const i8,
    correction: f32,
    alpha: f32,
    dim: usize,
) {
    use std::arch::x86_64::*;

    let corr_v = _mm256_set1_ps(correction);
    let alpha_v = _mm256_set1_ps(alpha);
    let out_ptr = out.as_mut_ptr();
    let mut i = 0;
    while i + 8 <= dim {
        // Load current output
        let o = _mm256_loadu_ps(out_ptr.add(i));
        // i8→f32: sign-extend i8→i32, convert to f32
        let v_i8 = _mm_loadl_epi64(v.add(i) as *const __m128i);
        let v_i32 = _mm256_cvtepi8_epi32(v_i8);
        let vf = _mm256_cvtepi32_ps(v_i32);
        // out = correction * out + alpha * v
        let result = _mm256_fmadd_ps(alpha_v, vf, _mm256_mul_ps(corr_v, o));
        _mm256_storeu_ps(out_ptr.add(i), result);
        i += 8;
    }
    // Scalar remainder
    while i < dim {
        *out_ptr.add(i) = correction * *out_ptr.add(i) + alpha * (*v.add(i) as f32);
        i += 1;
    }
}

/// AVX2 online SV for INT4 V (packed nibbles, half-split layout).
/// out[d] = correction * out[d] + alpha * dequant(v_packed[d])
/// Processes in groups of 32 elements (16 packed bytes).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn avx2_online_sv_i4(
    out: &mut [f32],
    v_packed: *const u8,
    correction: f32,
    alpha: f32,
    head_dim: usize,
) {
    use std::arch::x86_64::*;

    let corr_v = _mm256_set1_ps(correction);
    let alpha_v = _mm256_set1_ps(alpha);
    let mask_0f = _mm256_set1_epi32(0x0F);
    let bias_8 = _mm256_set1_epi32(8);
    let out_ptr = out.as_mut_ptr();
    let num_groups = head_dim / 32;

    for g in 0..num_groups {
        let packed = v_packed.add(g * 16);
        let out_base = g * 32;

        // Process first 16 elements (low nibbles), 8 at a time
        // Load 8 bytes, zero-extend to 32-bit, mask low nibble, subtract 8
        let v8_lo = _mm_loadl_epi64(packed as *const __m128i);
        let v32_lo = _mm256_cvtepu8_epi32(v8_lo);
        let lo_nibbles = _mm256_and_si256(v32_lo, mask_0f);
        let lo_signed = _mm256_sub_epi32(lo_nibbles, bias_8);
        let lo_f = _mm256_cvtepi32_ps(lo_signed);
        let o_lo = _mm256_loadu_ps(out_ptr.add(out_base));
        let r_lo = _mm256_fmadd_ps(alpha_v, lo_f, _mm256_mul_ps(corr_v, o_lo));
        _mm256_storeu_ps(out_ptr.add(out_base), r_lo);

        let v8_lo2 = _mm_loadl_epi64(packed.add(8) as *const __m128i);
        let v32_lo2 = _mm256_cvtepu8_epi32(v8_lo2);
        let lo_nibbles2 = _mm256_and_si256(v32_lo2, mask_0f);
        let lo_signed2 = _mm256_sub_epi32(lo_nibbles2, bias_8);
        let lo_f2 = _mm256_cvtepi32_ps(lo_signed2);
        let o_lo2 = _mm256_loadu_ps(out_ptr.add(out_base + 8));
        let r_lo2 = _mm256_fmadd_ps(alpha_v, lo_f2, _mm256_mul_ps(corr_v, o_lo2));
        _mm256_storeu_ps(out_ptr.add(out_base + 8), r_lo2);

        // Process second 16 elements (high nibbles), 8 at a time
        let v32_hi = _mm256_srli_epi32(v32_lo, 4);
        let hi_nibbles = _mm256_and_si256(v32_hi, mask_0f);
        let hi_signed = _mm256_sub_epi32(hi_nibbles, bias_8);
        let hi_f = _mm256_cvtepi32_ps(hi_signed);
        let o_hi = _mm256_loadu_ps(out_ptr.add(out_base + 16));
        let r_hi = _mm256_fmadd_ps(alpha_v, hi_f, _mm256_mul_ps(corr_v, o_hi));
        _mm256_storeu_ps(out_ptr.add(out_base + 16), r_hi);

        let v32_hi2 = _mm256_srli_epi32(v32_lo2, 4);
        let hi_nibbles2 = _mm256_and_si256(v32_hi2, mask_0f);
        let hi_signed2 = _mm256_sub_epi32(hi_nibbles2, bias_8);
        let hi_f2 = _mm256_cvtepi32_ps(hi_signed2);
        let o_hi2 = _mm256_loadu_ps(out_ptr.add(out_base + 24));
        let r_hi2 = _mm256_fmadd_ps(alpha_v, hi_f2, _mm256_mul_ps(corr_v, o_hi2));
        _mm256_storeu_ps(out_ptr.add(out_base + 24), r_hi2);
    }
}

