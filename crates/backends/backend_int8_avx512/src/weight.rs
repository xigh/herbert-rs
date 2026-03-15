//! INT8 weight type with per-channel symmetric quantization.
//!
//! Per-channel = 1 scale per output row. Weights stored as unsigned u8
//! (w_i8 + 128) for VPDPBUSD compatibility (u8 × s8 → i32).
//!
//! Activations are quantized per-token (1 scale for entire input vector).
//! Bias correction: result -= 128 * sum(x_i8) per row.

use herbert_backend_common::weight_cache::CacheWeight;
use herbert_core::tensor::{bf16_to_f32, BF16};

/// INT8 weight with per-channel (per-row) symmetric scales.
#[derive(Clone)]
pub struct Int8Weight {
    /// Unsigned weight data [N, K], where w_u8 = w_i8 + 128.
    pub data: Vec<u8>,
    /// Per-channel (per-row) scales [N].
    pub scales: Vec<f32>,
    pub n: usize,
    pub k: usize,
}

/// Per-channel symmetric INT8 quantization of BF16 weights.
///
/// One scale per row: scale = abs_max(row) / 127.0.
/// Values quantized to i8 [-127, +127], stored as u8 (i8 + 128).
pub fn quantize_bf16_to_int8(src: &[BF16], n: usize, k: usize) -> Int8Weight {
    assert_eq!(src.len(), n * k);

    let mut scales = vec![0.0f32; n];
    let mut data = vec![128u8; n * k];

    for row in 0..n {
        let row_start = row * k;

        let mut abs_max = 0.0f32;
        for col in 0..k {
            let val = bf16_to_f32(src[row_start + col]).abs();
            if val > abs_max {
                abs_max = val;
            }
        }
        let scale = if abs_max > f32::EPSILON { abs_max / 127.0 } else { 1.0 };
        scales[row] = scale;
        let inv_scale = 1.0 / scale;

        for col in 0..k {
            let val = bf16_to_f32(src[row_start + col]);
            let q = (val * inv_scale).round().clamp(-127.0, 127.0) as i8;
            data[row * k + col] = (q as i16 + 128) as u8;
        }
    }

    Int8Weight { data, scales, n, k }
}

// ============================================================================
// Per-token activation quantization
// ============================================================================

/// Per-token activation quantization: one scale for the entire vector.
/// Returns (x_i8, x_scale, x_total_sum).
pub fn quantize_x_pertoken(x: &[f32]) -> (Vec<i8>, f32, i64) {
    let mut abs_max = 0.0f32;
    for &v in x {
        let av = v.abs();
        if av > abs_max {
            abs_max = av;
        }
    }
    let scale = if abs_max > f32::EPSILON { abs_max / 127.0 } else { 1.0 };
    let inv_scale = 1.0 / scale;

    let mut total_sum = 0i64;
    let x_i8: Vec<i8> = x
        .iter()
        .map(|&v| {
            let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
            total_sum += q as i64;
            q
        })
        .collect();

    (x_i8, scale, total_sum)
}

// ============================================================================
// CacheWeight implementation
// ============================================================================

impl CacheWeight for Int8Weight {
    /// v3: per-channel scales.
    const CACHE_VERSION: u32 = 3;

    fn cache_write(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_all(&(self.n as u64).to_le_bytes())?;
        w.write_all(&(self.k as u64).to_le_bytes())?;
        w.write_all(&self.data)?;
        let scale_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(self.scales.as_ptr() as *const u8, self.scales.len() * 4)
        };
        w.write_all(scale_bytes)
    }

    fn cache_read(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)?;
        let n = u64::from_le_bytes(buf8) as usize;
        r.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        let mut data = vec![0u8; n * k];
        r.read_exact(&mut data)?;
        let mut scales = vec![0.0f32; n];
        let scale_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(scales.as_mut_ptr() as *mut u8, n * 4)
        };
        r.read_exact(scale_bytes)?;
        Ok(Int8Weight { data, scales, n, k })
    }
}
