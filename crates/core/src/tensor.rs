//! Tensor operations and utilities

pub type BF16 = u16;

/// BF16 → f32: zero-extend the upper 16 bits.
#[inline(always)]
pub fn bf16_to_f32(bf16: BF16) -> f32 {
    f32::from_bits((bf16 as u32) << 16)
}

/// Convert an FP8 E4M3 byte to f32.
///
/// FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits, bias = 7.
/// Special values: 0x7F and 0xFF are NaN (no infinities in E4M3).
#[inline(always)]
pub fn fp8e4m3_to_f32(byte: u8) -> f32 {
    // NaN: exponent=0b1111, mantissa=0b111
    if byte == 0x7F || byte == 0xFF {
        return f32::NAN;
    }
    let sign = (byte >> 7) & 1;
    let exp = ((byte >> 3) & 0xF) as i32;
    let mant = (byte & 0x7) as u32;

    let val = if exp == 0 {
        // Subnormal: value = (-1)^sign * (mant/8) * 2^(-6)
        // = (-1)^sign * mant * 2^(-9)
        (mant as f32) * (1.0 / 512.0) // 2^(-9) = 1/512
    } else {
        // Normal: value = (-1)^sign * (1 + mant/8) * 2^(exp-7)
        let significand = (1u32 << 3) | mant; // (8 + mant)
        // significand/8 * 2^(exp-7) = significand * 2^(exp-10)
        let shift = exp - 10;
        if shift >= 0 {
            (significand as f32) * (1u64 << shift as u64) as f32
        } else {
            (significand as f32) / (1u64 << (-shift) as u64) as f32
        }
    };

    if sign == 1 { -val } else { val }
}

/// Convert f32 to BF16 with round-to-nearest-even (IEEE 754 RNE).
#[inline(always)]
pub fn f32_to_bf16(f32_val: f32) -> BF16 {
    let bits = f32_val.to_bits();
    let lsb = (bits >> 16) & 1;
    let round_bit = (bits >> 15) & 1;
    let sticky = bits & 0x7FFF;
    let mut result = bits >> 16;
    if round_bit == 1 && (sticky != 0 || lsb == 1) {
        result += 1;
    }
    result as u16
}
