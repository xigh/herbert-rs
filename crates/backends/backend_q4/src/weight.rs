//! Q4 symmetric quantization types and conversion functions.
//!
//! Weight layout: [n_tiles, K/4, 16, 4] (pre-interleaved)
//!   Per K-group of 4, 16 half-lanes × 4 k-values:
//!     byte[4*hl + j] = nib(lane_hl, kj) | (nib(lane_hl+16, kj) << 4)
//!   Low nibble = lanes 0-15, high nibble = lanes 16-31.
//!   16 half-lanes × 4 bytes = 64 bytes per K-group.
//!
//! This layout eliminates vpunpcklbw/vpunpckhbw from the inner loop:
//!   vpandd gives lanes 0-15 ready for VPDPBUSD
//!   vpsrld+vpandd gives lanes 16-31 ready for VPDPBUSD
//!
//! Nibble values are unsigned [0,15] representing symmetric Q4 (q+8).
//! Original range: q in [-8, +7], stored as (q+8) in [0, 15].
//!
//! Per-group quantization: each row is divided into groups of Q4_GROUP_SIZE
//! K-values. Each group has its own scale (abs_max / 7.0). This provides
//! much better precision than per-channel quantization, especially for
//! weights with mixed-magnitude values (e.g. MLP gate projections where
//! outliers would otherwise dominate the per-channel scale).
//!
//! AVX-512 uses unsigned*signed VPDPBUSD (needs x_col_sum correction).

use herbert_backend_common::hugepages::HugeVec;
use herbert_core::tensor::{bf16_to_f32, BF16};

/// Output tile size (N dimension), matches other backends.
pub const TILE_N: usize = 32;

/// Group size for per-group quantization (number of K-values per scale).
/// Must be a multiple of 4.
pub const Q4_GROUP_SIZE: usize = 32;

/// Q4 weight matrix packed in nibble layout with per-group scales.
///
/// Buffers use `HugeVec<T>`: when the `hugepages` feature is enabled on Linux,
/// data is backed by `mmap(MAP_HUGETLB)` for guaranteed 2MB page allocation,
/// eliminating TLB page-walk overhead (measured: -45% DRAM latency on Zen 4).
/// Falls back to regular Vec if hugepage allocation fails.
#[derive(Clone)]
pub struct Q4Weight {
    /// Packed nibbles in pre-interleaved layout [n_tiles, K/4, 16, 4].
    /// byte[4*hl + j] = nib(lane_hl, kj) | (nib(lane_hl+16, kj) << 4).
    pub data: HugeVec<u8>,
    /// Per-group scales, length = N * n_groups where n_groups = ceil(K / Q4_GROUP_SIZE).
    /// Column-major layout: scales[group * N + row] — contiguous across tile rows
    /// for a given scale group, enabling stride-1 SIMD loads in dequant kernels.
    pub scales: HugeVec<f32>,
    /// Tile-local transposed scales for the fused AVX-512 kernel.
    /// Layout: [tile * n_sg * TILE_N + sg * TILE_N + lane]
    /// Contiguous per tile (~12KB for 96 SGs) instead of strided across N pages.
    /// Used only by avx512_q4_tile_fused; all other paths use `scales`.
    pub scales_tiled: HugeVec<f32>,
    pub n: usize,
    pub k: usize,
}

/// Per-group symmetric Q4 quantization and packing (pre-interleaved layout).
///
/// Quantizes BF16 row-major weights [N, K] to Q4 packed nibbles in
/// pre-interleaved layout [n_tiles, K/4, 16, 4].
///
/// Each row is divided into groups of Q4_GROUP_SIZE K-values. Each group
/// has its own scale = abs_max_in_group / 7.0. This provides much better
/// precision than per-channel quantization.
///
/// Each nibble = clamp(round(val / group_scale), -8, 7) + 8, giving [0, 15].
///
/// Pre-interleaved packing per K-group of 4:
///   byte[4*hl + j] = nib(lane_hl, kj) | (nib(lane_hl+16, kj) << 4)
/// This makes vpandd give lanes 0-15 and vpsrld+vpandd give lanes 16-31,
/// eliminating vpunpcklbw/vpunpckhbw from the inner loop.
pub fn quantize_and_pack_q4(
    src: &[BF16],
    n: usize,
    k: usize,
) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(
        src.len(),
        n * k,
        "quantize_and_pack_q4: shape mismatch: src.len()={} != n*k={}",
        src.len(),
        n * k
    );

    let n_tiles = n.div_ceil(TILE_N);
    let k_groups = k.div_ceil(4);
    let n_scale_groups = k.div_ceil(Q4_GROUP_SIZE);

    // Per-group scales in column-major layout: scales[g * n + row]
    // This makes scales contiguous across tile rows for a given group,
    // enabling stride-1 SIMD loads instead of expensive gather instructions.
    let mut scales = vec![0.0f32; n * n_scale_groups];
    for row in 0..n {
        let row_start = row * k;
        for g in 0..n_scale_groups {
            let g_start = g * Q4_GROUP_SIZE;
            let g_end = (g_start + Q4_GROUP_SIZE).min(k);
            let mut abs_max = 0.0f32;
            for col in g_start..g_end {
                let val = bf16_to_f32(src[row_start + col]).abs();
                if val > abs_max {
                    abs_max = val;
                }
            }
            scales[g * n + row] = if abs_max > f32::EPSILON {
                abs_max / 7.0
            } else {
                1.0
            };
        }
    }

    // Packed data: [n_tiles, k_groups, 16, 4] = 64 bytes per KG
    // Total bytes = n_tiles * k_groups * 64 (same as TILE_N * 2)
    let total = n_tiles * k_groups * TILE_N * 2;
    // Zero-bias fill: nibble 8 = (0+8), packed as 0x88 = 8 | (8<<4)
    let mut data = vec![0x88u8; total];

    for tile in 0..n_tiles {
        let n_start = tile * TILE_N;
        let n_end = (n_start + TILE_N).min(n);
        let tile_base = tile * k_groups * TILE_N * 2;

        for kg in 0..k_groups {
            let kg_base = tile_base + kg * TILE_N * 2;

            // First compute nibbles for all 32 lanes in this K-group
            let mut all_nibs = [[8u8; 4]; TILE_N]; // default to zero-bias
            for lane in 0..(n_end - n_start) {
                let row = n_start + lane;
                let col_start = kg * 4;
                let g = col_start / Q4_GROUP_SIZE;
                let inv_scale = 1.0 / scales[g * n + row];

                for ki in 0..4 {
                    let col = col_start + ki;
                    if col < k {
                        let val = bf16_to_f32(src[row * k + col]);
                        let q = (val * inv_scale).round().clamp(-8.0, 7.0) as i8;
                        all_nibs[lane][ki] = (q + 8) as u8;
                    }
                }
            }

            // Pack in pre-interleaved format:
            //   byte[4*hl + j] = nib(lane_hl, kj) | (nib(lane_hl+16, kj) << 4)
            for hl in 0..16 {
                for j in 0..4 {
                    let lo_nib = all_nibs[hl][j];
                    let hi_nib = all_nibs[hl + 16][j];
                    data[kg_base + hl * 4 + j] = lo_nib | (hi_nib << 4);
                }
            }
        }
    }

    (data, scales)
}

/// Number of scale groups for a given K dimension.
#[inline]
pub fn n_scale_groups(k: usize) -> usize {
    k.div_ceil(Q4_GROUP_SIZE)
}

// ============================================================================
// Cache helper methods (used by KernelWeight CacheWeight impl in kernels.rs)
// ============================================================================

impl Q4Weight {
    /// Transpose scales from column-major [sg * N + n] to tile-local
    /// [tile * n_sg * TILE_N + sg * TILE_N + lane] layout.
    /// This makes all scale groups for a single tile contiguous in memory,
    /// eliminating DTLB misses in the fused AVX-512 kernel.
    pub fn compute_scales_tiled(&mut self) {
        let n_sg = n_scale_groups(self.k);
        let n_tiles = self.n.div_ceil(TILE_N);
        let mut tiled = vec![0.0f32; n_tiles * n_sg * TILE_N];
        for tile in 0..n_tiles {
            let n_start = tile * TILE_N;
            for sg in 0..n_sg {
                for lane in 0..TILE_N.min(self.n - n_start) {
                    tiled[tile * n_sg * TILE_N + sg * TILE_N + lane]
                        = self.scales[sg * self.n + n_start + lane];
                }
            }
        }
        self.scales_tiled = HugeVec::from_vec_no_huge(tiled);
    }

    /// Move all weight buffers to MAP_HUGETLB-backed memory (2MB pages).
    /// This replaces the Vec-backed data/scales/scales_tiled with hugepage mmap.
    /// Falls back to MADV_HUGEPAGE hint if MAP_HUGETLB allocation fails.
    pub fn move_to_hugepages(&mut self) {
        // Re-wrap each buffer through HugeVec::from_vec which attempts MAP_HUGETLB
        let data_vec: Vec<u8> = self.data.to_vec();
        self.data = HugeVec::from_vec(data_vec);
        let scales_vec: Vec<f32> = self.scales.to_vec();
        self.scales = HugeVec::from_vec(scales_vec);
        let tiled_vec: Vec<f32> = self.scales_tiled.to_vec();
        self.scales_tiled = HugeVec::from_vec(tiled_vec);
    }

    /// Report hugepage backing status.
    pub fn is_hugepage_backed(&self) -> bool {
        self.data.is_hugepage_backed()
    }

    pub fn cache_write_inner(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_all(&(self.n as u64).to_le_bytes())?;
        w.write_all(&(self.k as u64).to_le_bytes())?;
        w.write_all(&(self.data.len() as u64).to_le_bytes())?;
        w.write_all(&self.data)?;
        w.write_all(&(self.scales.len() as u64).to_le_bytes())?;
        let scale_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(self.scales.as_ptr() as *const u8, self.scales.len() * 4)
        };
        w.write_all(scale_bytes)
    }

    pub fn cache_read_inner(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)?;
        let n = u64::from_le_bytes(buf8) as usize;
        r.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        r.read_exact(&mut buf8)?;
        let data_len = u64::from_le_bytes(buf8) as usize;
        let mut data = vec![0u8; data_len];
        r.read_exact(&mut data)?;
        r.read_exact(&mut buf8)?;
        let scales_len = u64::from_le_bytes(buf8) as usize;
        let expected_scales = n * k.div_ceil(Q4_GROUP_SIZE);
        if scales_len != expected_scales {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("scales_len ({}) != expected ({})", scales_len, expected_scales)));
        }
        let mut scales = vec![0.0f32; scales_len];
        let scale_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(scales.as_mut_ptr() as *mut u8, scales_len * 4)
        };
        r.read_exact(scale_bytes)?;
        let mut w = Q4Weight {
            data: HugeVec::from_vec_no_huge(data),
            scales: HugeVec::from_vec_no_huge(scales),
            scales_tiled: HugeVec::from_vec_no_huge(vec![]),
            n, k,
        };
        w.compute_scales_tiled();
        w.move_to_hugepages();
        Ok(w)
    }
}

// ============================================================================
// BF16Weight — full-precision weight kept as f32 (for lm_head, MoE routers)
// ============================================================================

/// Weight kept at full precision (loaded from BF16, stored as f32 row-major).
///
/// Used for critical tensors where Q4 quantization causes too much precision
/// loss: lm_head (output projection) and MoE router gates.
#[derive(Clone)]
pub struct BF16Weight {
    /// Row-major f32 data [N, K].
    pub data: Vec<f32>,
    pub n: usize,
    pub k: usize,
}

/// Convert BF16 row-major weights [N, K] to f32 BF16Weight (no quantization).
pub fn bf16_to_f32_weight(src: &[BF16], n: usize, k: usize) -> BF16Weight {
    assert_eq!(
        src.len(),
        n * k,
        "bf16_to_f32_weight: shape mismatch: src.len()={} != n*k={}",
        src.len(),
        n * k
    );
    let data: Vec<f32> = src.iter().map(|&b| bf16_to_f32(b)).collect();
    BF16Weight { data, n, k }
}

impl BF16Weight {
    pub fn cache_write_inner(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_all(&(self.n as u64).to_le_bytes())?;
        w.write_all(&(self.k as u64).to_le_bytes())?;
        let data_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len() * 4)
        };
        w.write_all(data_bytes)
    }

    pub fn cache_read_inner(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)?;
        let n = u64::from_le_bytes(buf8) as usize;
        r.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        let len = n * k;
        let mut data = vec![0.0f32; len];
        let data_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, len * 4)
        };
        r.read_exact(data_bytes)?;
        Ok(BF16Weight { data, n, k })
    }
}

// ============================================================================
// INT8Weight — symmetric per-group INT8 quantization (for lm_head)
// ============================================================================

/// INT8 weight with per-group symmetric scales.
/// Weights stored as unsigned u8 (w_i8 + 128) in row-major [N, K].
/// This unsigned representation allows direct use with VPDPBUSD (u8 × i8 → i32).
/// Correction: acc -= 128 * sum(x_i8_in_group) to undo the offset.
#[derive(Clone)]
pub struct INT8Weight {
    /// Unsigned weight data [N, K], where w_u8 = w_i8 + 128.
    pub data: Vec<u8>,
    /// Per-group scales, length = N * n_groups where n_groups = ceil(K / Q4_GROUP_SIZE).
    /// Row-major layout: scales[row * n_groups + group] — contiguous across groups
    /// for a given row, enabling per-row slice access in dot_int8_row.
    pub scales: Vec<f32>,
    pub n: usize,
    pub k: usize,
}

/// Per-group symmetric INT8 quantization of BF16 weights.
///
/// Each row is divided into groups of Q4_GROUP_SIZE K-values. Each group gets
/// its own scale = abs_max / 127.0. Values are quantized to i8 [-127, +127]
/// then stored as u8 (i8 + 128) for unsigned VPDPBUSD compatibility.
pub fn quantize_bf16_to_int8(src: &[BF16], n: usize, k: usize) -> INT8Weight {
    assert_eq!(src.len(), n * k);
    let n_groups = k.div_ceil(Q4_GROUP_SIZE);

    let mut scales = vec![0.0f32; n * n_groups];
    let mut data = vec![128u8; n * k]; // zero bias default

    for row in 0..n {
        let row_start = row * k;
        for g in 0..n_groups {
            let g_start = g * Q4_GROUP_SIZE;
            let g_end = (g_start + Q4_GROUP_SIZE).min(k);

            let mut abs_max = 0.0f32;
            for col in g_start..g_end {
                let val = bf16_to_f32(src[row_start + col]).abs();
                if val > abs_max {
                    abs_max = val;
                }
            }
            let scale = if abs_max > f32::EPSILON {
                abs_max / 127.0
            } else {
                1.0
            };
            scales[row * n_groups + g] = scale;
            let inv_scale = 1.0 / scale;

            for col in g_start..g_end {
                let val = bf16_to_f32(src[row_start + col]);
                let q = (val * inv_scale).round().clamp(-127.0, 127.0) as i8;
                data[row * k + col] = (q as i16 + 128) as u8;
            }
        }
    }

    INT8Weight { data, scales, n, k }
}

impl INT8Weight {
    pub fn cache_write_inner(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_all(&(self.n as u64).to_le_bytes())?;
        w.write_all(&(self.k as u64).to_le_bytes())?;
        w.write_all(&self.data)?;
        let scale_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(self.scales.as_ptr() as *const u8, self.scales.len() * 4)
        };
        w.write_all(scale_bytes)
    }

    pub fn cache_read_inner(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)?;
        let n = u64::from_le_bytes(buf8) as usize;
        r.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        let mut data = vec![0u8; n * k];
        r.read_exact(&mut data)?;
        let n_groups = k.div_ceil(Q4_GROUP_SIZE);
        let scales_len = n * n_groups;
        let mut scales = vec![0.0f32; scales_len];
        let scale_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(scales.as_mut_ptr() as *mut u8, scales_len * 4)
        };
        r.read_exact(scale_bytes)?;
        Ok(INT8Weight { data, scales, n, k })
    }
}

/// Dynamic input quantization: f32 -> i8 (per-row scale).
///
/// Returns (quantized i8 vector, scale) where scale = abs_max / 127.0.
pub fn quantize_x_f32_to_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let mut abs_max = 0.0f32;
    for &v in x {
        let av = v.abs();
        if av > abs_max {
            abs_max = av;
        }
    }
    let x_scale = if abs_max > f32::EPSILON {
        abs_max / 127.0
    } else {
        1.0
    };
    let inv_scale = 1.0 / x_scale;

    let x_i8: Vec<i8> = x
        .iter()
        .map(|&v| (v * inv_scale).round().clamp(-127.0, 127.0) as i8)
        .collect();

    (x_i8, x_scale)
}

/// Dynamic input quantization: f32 -> i8 with per-group scales (scalar fallback).
///
/// Each group of Q4_GROUP_SIZE elements gets its own scale, preventing
/// activation outliers in one group from destroying precision in others.
///
/// Returns (quantized i8 vector, per-group scales).
#[inline(never)]
pub fn quantize_x_f32_to_i8_grouped_scalar(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let k = x.len();
    let n_groups = k.div_ceil(Q4_GROUP_SIZE);
    let mut group_scales = vec![0.0f32; n_groups];
    let mut x_i8 = vec![0i8; k];

    for g in 0..n_groups {
        let start = g * Q4_GROUP_SIZE;
        let end = (start + Q4_GROUP_SIZE).min(k);

        let mut abs_max = 0.0f32;
        for i in start..end {
            let av = x[i].abs();
            if av > abs_max {
                abs_max = av;
            }
        }
        let scale = if abs_max > f32::EPSILON {
            abs_max / 127.0
        } else {
            1.0
        };
        group_scales[g] = scale;
        let inv_scale = 1.0 / scale;

        for i in start..end {
            x_i8[i] = (x[i] * inv_scale).round().clamp(-127.0, 127.0) as i8;
        }
    }

    (x_i8, group_scales)
}

/// Compute per-group column sums (scalar fallback).
/// Returns a vector of length ceil(k / Q4_GROUP_SIZE), where each element
/// is the sum of x_i8 values in that group.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub fn compute_x_group_col_sums_scalar(x_i8: &[i8], k: usize) -> Vec<i64> {
    let n_groups = k.div_ceil(Q4_GROUP_SIZE);
    let mut sums = vec![0i64; n_groups];
    for g in 0..n_groups {
        let start = g * Q4_GROUP_SIZE;
        let end = (start + Q4_GROUP_SIZE).min(k).min(x_i8.len());
        for i in start..end {
            sums[g] += x_i8[i] as i64;
        }
    }
    sums
}

/// AVX-512 combined x-quantization + column sums in one pass per group.
/// Processes 32 f32 values (2 ZMM registers) per group.
/// Returns (quantized i8 vector, per-group scales, per-group col_sums).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline(never)]
pub unsafe fn quantize_x_and_colsums_avx512(x: &[f32]) -> (Vec<i8>, Vec<f32>, Vec<i64>) {
    use std::arch::x86_64::*;

    let k = x.len();
    let n_groups = k.div_ceil(Q4_GROUP_SIZE);
    let mut x_i8 = vec![0i8; k];
    let mut group_scales = vec![0.0f32; n_groups];
    let mut col_sums = vec![0i64; n_groups];

    let inp = x.as_ptr();
    let qi8 = x_i8.as_mut_ptr();
    let abs_mask = _mm512_set1_epi32(0x7FFF_FFFFi32);

    for g in 0..n_groups {
        let base = g * Q4_GROUP_SIZE;
        let remain = k - base;

        if remain >= 32 {
            // Full group: 2 ZMMs of 16 f32 each
            let v0 = _mm512_loadu_ps(inp.add(base));
            let v1 = _mm512_loadu_ps(inp.add(base + 16));

            // Find max_abs across 32 values
            let abs0 = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(v0), abs_mask));
            let abs1 = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(v1), abs_mask));
            let max_abs = _mm512_reduce_max_ps(_mm512_max_ps(abs0, abs1));

            let (scale, inv_scale) = if max_abs > f32::EPSILON {
                (max_abs / 127.0, 127.0 / max_abs)
            } else {
                (1.0, 1.0)
            };
            group_scales[g] = scale;

            let inv_scale_v = _mm512_set1_ps(inv_scale);

            // Quantize: multiply by inv_scale, round to nearest, saturate i32→i8
            let q0_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(v0, inv_scale_v));
            let q1_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(v1, inv_scale_v));

            // Pack i32→i8 with saturation (vpmovdb)
            let q0_i8 = _mm512_cvtsepi32_epi8(q0_i32); // __m128i: 16 i8
            let q1_i8 = _mm512_cvtsepi32_epi8(q1_i32); // __m128i: 16 i8
            _mm_storeu_si128(qi8.add(base) as *mut __m128i, q0_i8);
            _mm_storeu_si128(qi8.add(base + 16) as *mut __m128i, q1_i8);

            // Column sum: horizontal add of i32 values (before saturation to preserve precision)
            let sum0 = _mm512_reduce_add_epi32(q0_i32);
            let sum1 = _mm512_reduce_add_epi32(q1_i32);
            col_sums[g] = (sum0 + sum1) as i64;
        } else {
            // Partial group: scalar fallback
            let mut abs_max = 0.0f32;
            for i in base..k {
                let av = (*inp.add(i)).abs();
                if av > abs_max { abs_max = av; }
            }
            let scale = if abs_max > f32::EPSILON { abs_max / 127.0 } else { 1.0 };
            group_scales[g] = scale;
            let inv_scale = 1.0 / scale;
            let mut cs = 0i64;
            for i in base..k {
                let q = (*inp.add(i) * inv_scale).round().clamp(-127.0, 127.0) as i8;
                *qi8.add(i) = q;
                cs += q as i64;
            }
            col_sums[g] = cs;
        }
    }

    (x_i8, group_scales, col_sums)
}

/// Dynamic input quantization: f32 -> i8 with per-group scales.
/// Auto-dispatches to AVX-512 when available.
pub fn quantize_x_f32_to_i8_grouped(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    quantize_x_f32_to_i8_grouped_scalar(x)
}

/// Compute per-group column sums for per-group quantization.
/// Returns a vector of length ceil(k / Q4_GROUP_SIZE), where each element
/// is the sum of x_i8 values in that group.
#[cfg(target_arch = "x86_64")]
pub fn compute_x_group_col_sums(x_i8: &[i8], k: usize) -> Vec<i64> {
    compute_x_group_col_sums_scalar(x_i8, k)
}

/// Combined quantize + col_sums with AVX-512 dispatch.
/// On AVX-512: single pass per group doing quant + col_sums together.
/// On scalar: falls back to separate scalar quant + col_sums.
#[cfg(target_arch = "x86_64")]
pub fn quantize_x_and_colsums(x: &[f32]) -> (Vec<i8>, Vec<f32>, Vec<i64>) {
    if herbert_backend_common::kernels::has_avx512f() && x.len() >= Q4_GROUP_SIZE {
        unsafe { quantize_x_and_colsums_avx512(x) }
    } else {
        let (x_i8, scales) = quantize_x_f32_to_i8_grouped_scalar(x);
        let col_sums = compute_x_group_col_sums_scalar(&x_i8, x.len());
        (x_i8, scales, col_sums)
    }
}

