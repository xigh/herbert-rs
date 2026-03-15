//! F32 → BF16 conversion utilities.
//!
//! AVX-512 BF16 path uses VCVTNEPS2BF16 (16 F32 → 16 BF16 per instruction).
//! Scalar fallback uses round-to-nearest-even truncation.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Convert F32 buffer to BF16.
///
/// Uses VCVTNEPS2BF16 when avx512bf16 is available, otherwise scalar fallback.
/// Both `src` and `dst` must have the same length.
/// Length must be a multiple of 16 for the AVX512 path.
#[cfg(target_arch = "x86_64")]
pub fn convert_f32_to_bf16(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len());
    if is_x86_feature_detected!("avx512bf16") && src.len() >= 16 {
        unsafe { convert_f32_to_bf16_avx512bf16(src, dst) }
    } else {
        convert_f32_to_bf16_scalar(src, dst)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn convert_f32_to_bf16(src: &[f32], dst: &mut [u16]) {
    convert_f32_to_bf16_scalar(src, dst);
}

/// Scalar F32→BF16 with round-to-nearest-even.
pub fn convert_f32_to_bf16_scalar(src: &[f32], dst: &mut [u16]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        let bits = s.to_bits();
        // Round to nearest even: add rounding bias
        let round = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
        *d = (round >> 16) as u16;
    }
}

/// AVX-512 BF16 F32→BF16 conversion using VCVTNEPS2BF16.
/// Processes 16 F32 → 16 BF16 per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn convert_f32_to_bf16_avx512bf16(src: &[f32], dst: &mut [u16]) {
    let n = src.len();
    let chunks = n / 16;
    let remainder = n % 16;

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 16;
        let v = _mm512_loadu_ps(src_ptr.add(offset));
        let bf16 = _mm512_cvtneps_pbh(v);
        let as_si256: __m256i = std::mem::transmute(bf16);
        _mm256_storeu_si256(dst_ptr.add(offset) as *mut __m256i, as_si256);
    }

    // Handle remainder with scalar
    if remainder > 0 {
        let base = chunks * 16;
        for j in 0..remainder {
            let bits = (*src_ptr.add(base + j)).to_bits();
            let round = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
            *dst_ptr.add(base + j) = (round >> 16) as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_conversion() {
        let src = [1.0f32, -2.5, 0.0, 3.14159, f32::INFINITY, f32::NEG_INFINITY];
        let mut dst = vec![0u16; src.len()];
        convert_f32_to_bf16_scalar(&src, &mut dst);

        // BF16 of 1.0 = 0x3F80
        assert_eq!(dst[0], 0x3F80);
        // BF16 of 0.0 = 0x0000
        assert_eq!(dst[2], 0x0000);
    }

    #[test]
    fn test_roundtrip_precision() {
        let values: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
        let mut bf16 = vec![0u16; values.len()];
        convert_f32_to_bf16(&values, &mut bf16);

        // Verify each value round-trips within BF16 precision (~0.8% relative error)
        for (i, &v) in values.iter().enumerate() {
            let bits = (bf16[i] as u32) << 16;
            let back = f32::from_bits(bits);
            if v.abs() > 0.01 {
                let rel_err = ((back - v) / v).abs();
                assert!(rel_err < 0.01, "value {} roundtrip error {}", v, rel_err);
            }
        }
    }
}
