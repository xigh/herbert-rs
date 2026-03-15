//! x86 Q4 kernels for decode (matvec) and prefill (matmul).
//!
//! Q4 weights are stored in pre-interleaved nibble layout [ceil(N/TILE_N), K/4, 16, 4].
//! Runtime dispatch: AVX-512 VNNI (dual acc) → AVX2 → scalar fallback.
//!
//! AVX-512 uses VPDPBUSD (1 instruction per dot product) with dual accumulators
//! to hide 5-cycle latency on Zen 4. Pre-interleaved layout eliminates vpunpck.
//! AVX2 emulates with vpmaddubsw + vpmaddwd + vpaddd (3 instructions).
//!
//! Dequant: y[n] = (acc_i32[n] - Q4_ZERO_BIAS * x_col_sum) * w_scale[n] * x_scale
//! (offset is 8 because nibbles are stored as (q+8) with q in [-8, +7])

use crate::weight::*;
use herbert_backend_common::autotune::{RuntimeAutoTuneDefaults, RuntimeAutoTuner};
use herbert_backend_common::thread_pool::{global_pool, SendMutPtr, SendPtr};
use herbert_core::error::{HerbertError, Result};
use std::sync::OnceLock;

/// Q4 zero-point offset: nibbles stored as (q+8) with q in [-8, +7].
const Q4_ZERO_BIAS: i64 = 8;

/// Lane reordering table for AVX-512 ASM accumulator output.
///
/// With pre-interleaved layout, the ASM kernel produces identity mapping:
///   zmm0 (acc[0..15]):  output lanes [0..15] (lo nibbles)
///   zmm1 (acc[16..31]): output lanes [16..31] (hi nibbles)
const ASM_LANE_TO_ACC: [usize; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];

/// Lane reordering table for AVX2 ASM accumulator output.
///
/// With pre-interleaved layout, the AVX2 kernel produces:
///   ymm0 (acc[0..7]):   lanes [0..7]
///   ymm1 (acc[8..15]):  lanes [16..23]
///   ymm2 (acc[16..23]): lanes [8..15]
///   ymm3 (acc[24..31]): lanes [24..31]
const AVX2_LANE_TO_ACC: [usize; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7,             // lanes 0-7   -> ymm0
    16, 17, 18, 19, 20, 21, 22, 23,      // lanes 8-15  -> ymm2 (offset 16 in acc)
    8, 9, 10, 11, 12, 13, 14, 15,        // lanes 16-23 -> ymm1 (offset 8 in acc)
    24, 25, 26, 27, 28, 29, 30, 31,      // lanes 24-31 -> ymm3
];

/// Fused AVX512 dequantization for one tile (32 lanes) and one scale group.
/// Delegates to hand-written assembly (avx512_q4_dequant_tile_fma) which uses
/// FMA (vfmadd231ps) to fuse the final multiply+accumulate into a single instruction.
///
/// SAFETY: Caller must ensure AVX-512F + FMA are available and all pointers are valid.
#[inline(always)]
unsafe fn avx512_fused_dequant_tile(
    acc_i32: *const i32,
    col_sum: i64,
    w_scales_base: *const f32,
    w_scales_stride: i32,
    x_scale: f32,
    f32_acc: *mut f32,
) {
    avx512_q4_dequant_tile_fma(acc_i32, col_sum, w_scales_base, w_scales_stride, x_scale, f32_acc);
}

/// Fused AVX2 dequantization for one full tile (32 lanes) and one scale group.
/// Delegates to hand-written assembly (avx2_q4_dequant_tile_fma) which uses
/// FMA (vfmadd231ps) and handles the v1/v2 lane swap internally.
///
/// SAFETY: Caller must ensure AVX2 + FMA are available and all pointers are valid.
#[inline(always)]
unsafe fn avx2_fused_dequant_tile(
    acc_i32: *const i32,
    col_sum: i64,
    w_scales_base: *const f32,
    w_scales_stride: i32,
    x_scale: f32,
    f32_acc: *mut f32,
) {
    avx2_q4_dequant_tile_fma(acc_i32, col_sum, w_scales_base, w_scales_stride, x_scale, f32_acc);
}

const AUTOTUNE_MAX_UNITS: usize = usize::MAX / 4;

const MATVEC_PAR_THRESHOLD_OPS: usize = 64 * 1024;
const MATVEC_TARGET_OPS_PER_WORKER: usize = MATVEC_PAR_THRESHOLD_OPS / 2;
const MATMUL_PAR_THRESHOLD_OPS: usize = 512 * 1024;
const MATMUL_TARGET_OPS_PER_WORKER: usize = MATMUL_PAR_THRESHOLD_OPS / 2;

fn matvec_autotuner() -> &'static RuntimeAutoTuner {
    static TUNER: OnceLock<RuntimeAutoTuner> = OnceLock::new();
    TUNER.get_or_init(|| {
        RuntimeAutoTuner::from_env(
            "avx512_q4/matvec",
            "HERBERT_AVX512_Q4_MATVEC",
            RuntimeAutoTuneDefaults {
                threshold_units: MATVEC_PAR_THRESHOLD_OPS,
                threshold_min: 1,
                threshold_max: AUTOTUNE_MAX_UNITS,
                target_units_per_worker: MATVEC_TARGET_OPS_PER_WORKER,
                target_min: 1,
                target_max: AUTOTUNE_MAX_UNITS,
            },
        )
    })
}

fn matmul_autotuner() -> &'static RuntimeAutoTuner {
    static TUNER: OnceLock<RuntimeAutoTuner> = OnceLock::new();
    TUNER.get_or_init(|| {
        RuntimeAutoTuner::from_env(
            "avx512_q4/matmul",
            "HERBERT_AVX512_Q4_MATMUL",
            RuntimeAutoTuneDefaults {
                threshold_units: MATMUL_PAR_THRESHOLD_OPS,
                threshold_min: 1,
                threshold_max: AUTOTUNE_MAX_UNITS,
                target_units_per_worker: MATMUL_TARGET_OPS_PER_WORKER,
                target_min: 1,
                target_max: AUTOTUNE_MAX_UNITS,
            },
        )
    })
}

// ============================================================================
// CPUID runtime check
// ============================================================================

fn has_avx512_vnni() -> bool {
    use std::sync::OnceLock;
    static HAS: OnceLock<bool> = OnceLock::new();
    *HAS.get_or_init(|| is_x86_feature_detected!("avx512vnni"))
}

fn has_avx2() -> bool {
    use std::sync::OnceLock;
    static HAS: OnceLock<bool> = OnceLock::new();
    *HAS.get_or_init(|| is_x86_feature_detected!("avx2"))
}

fn use_fused3_3way_avx512() -> bool {
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        !std::env::var("HERBERT_AVX512_Q4_DISABLE_FUSED3_3WAY")
            .ok()
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false)
    })
}

// ============================================================================
// Extern C declarations for ASM kernels
// ============================================================================

extern "C" {
    fn avx512_q4_matvec_tile(
        x_i8: *const i8,
        w_packed: *const u8,
        k_groups: u64,
        acc_out: *mut i32,
    );

    fn avx512_q4_matvec_tile_3way(
        x_i8: *const i8,
        w1_packed: *const u8,
        w2_packed: *const u8,
        w3_packed: *const u8,
        k_groups: u64,
        acc_out_all: *mut i32,
    );

    #[allow(dead_code)]
    fn avx512_q4_matmul_tile(
        x_i8_base: *const i8,
        w_packed: *const u8,
        k_groups: u64,
        acc_out: *mut i32,
        m_rows: u64,
        x_stride: u64,
    );

    fn avx512_q4_dequant_tile_fma(
        acc_i32: *const i32,
        col_sum: i64,
        w_scales_base: *const f32,
        w_scales_stride: i32,
        x_scale: f32,
        f32_acc: *mut f32,
    );

    fn avx512_q4_tile_fused(
        x_i8: *const i8,
        w_packed: *const u8,
        kgs_per_sg: u64,
        n_sg: u64,
        col_sums: *const i64,
        w_scales: *const f32,
        x_scales: *const f32,
        f32_acc: *mut f32,
        w_scales_sg_stride: u64,
    );
}

extern "C" {
    fn avx2_q4_matvec_tile(
        x_i8: *const i8,
        w_packed: *const u8,
        k_groups: u64,
        acc_out: *mut i32,
    );

    fn avx2_q4_dequant_tile_fma(
        acc_i32: *const i32,
        col_sum: i64,
        w_scales_base: *const f32,
        w_scales_stride: i32,
        x_scale: f32,
        f32_acc: *mut f32,
    );
}

extern "C" {
    /// Fused SiLU×Gate for one AVX-512 tile (32 f32 lanes).
    /// gate[i] = silu(gate[i]) * up[i]
    fn avx512_silu_gate_tile(gate: *mut f32, up: *const f32);

    /// Fused SiLU×Gate for one AVX2 tile (32 f32 lanes, 4×ymm loop).
    /// gate[i] = silu(gate[i]) * up[i]
    fn avx2_silu_gate_tile(gate: *mut f32, up: *const f32);
}

// ============================================================================
// Pre-interleaved nibble extraction helper
// ============================================================================

/// Extract 4 nibbles for a given lane from the pre-interleaved weight layout.
/// Layout: byte[4*hl + j] = nib(lane_hl, kj) | (nib(lane_hl+16, kj) << 4)
#[inline(always)]
fn unpack_pre_interleaved_lane(data: &[u8], kg_base: usize, lane: usize) -> [u8; 4] {
    let (hl, shift) = if lane < 16 { (lane, 0) } else { (lane - 16, 4) };
    let base = kg_base + hl * 4;
    [
        (data[base] >> shift) & 0xF,
        (data[base + 1] >> shift) & 0xF,
        (data[base + 2] >> shift) & 0xF,
        (data[base + 3] >> shift) & 0xF,
    ]
}

/// Unsafe pointer variant for use inside parallel_for closures.
#[inline(always)]
unsafe fn unpack_pre_interleaved_lane_ptr(data_ptr: *const u8, kg_base: usize, lane: usize) -> [u8; 4] {
    let (hl, shift) = if lane < 16 { (lane, 0) } else { (lane - 16, 4) };
    let base = kg_base + hl * 4;
    [
        (*data_ptr.add(base) >> shift) & 0xF,
        (*data_ptr.add(base + 1) >> shift) & 0xF,
        (*data_ptr.add(base + 2) >> shift) & 0xF,
        (*data_ptr.add(base + 3) >> shift) & 0xF,
    ]
}

// ============================================================================
// Scalar fallback
// ============================================================================

fn scalar_q4_matvec(x: &[f32], w: &Q4Weight, y: &mut [f32]) {
    let k = w.k;
    let n = w.n;
    let n_tiles = n.div_ceil(TILE_N);
    let k_groups = k.div_ceil(4);
    let n_sg = n_scale_groups(k);
    let kgs_per_sg = Q4_GROUP_SIZE / 4;

    let (x_i8, x_group_scales, x_group_col_sums) = quantize_x_and_colsums(x);

    for tile in 0..n_tiles {
        let n_start = tile * TILE_N;
        let n_end = (n_start + TILE_N).min(n);
        let tile_base = tile * k_groups * TILE_N * 2;

        for lane in 0..(n_end - n_start) {
            let row = n_start + lane;
            let mut f32_acc = 0.0f32;

            for sg in 0..n_sg {
                let kg_start = sg * kgs_per_sg;
                let kg_end = ((sg + 1) * kgs_per_sg).min(k_groups);
                let mut acc_i32 = 0i64;

                for kg in kg_start..kg_end {
                    let kg_base = tile_base + kg * TILE_N * 2;
                    let nibs = unpack_pre_interleaved_lane(&w.data, kg_base, lane);

                    for ki in 0..4 {
                        let col = kg * 4 + ki;
                        if col < k {
                            let w_val = nibs[ki] as i32;
                            let x_val = x_i8[col] as i32;
                            acc_i32 += (w_val * x_val) as i64;
                        }
                    }
                }

                let corrected = acc_i32 - Q4_ZERO_BIAS * x_group_col_sums[sg];
                let group_scale = w.scales[sg * n + row];
                f32_acc += corrected as f32 * group_scale * x_group_scales[sg];
            }

            y[row] = f32_acc;
        }
    }
}

fn scalar_q4_matmul(a: &[f32], w: &Q4Weight, c: &mut [f32], m: usize) {
    let k = w.k;
    let n = w.n;
    let n_tiles = n.div_ceil(TILE_N);
    let k_groups = k.div_ceil(4);
    let n_sg = n_scale_groups(k);
    let kgs_per_sg = Q4_GROUP_SIZE / 4;
    c.fill(0.0);

    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let (x_i8, x_group_scales, x_group_col_sums) = quantize_x_and_colsums(a_row);

        for tile in 0..n_tiles {
            let n_start = tile * TILE_N;
            let n_end = (n_start + TILE_N).min(n);
            let tile_base = tile * k_groups * TILE_N * 2;

            for lane in 0..(n_end - n_start) {
                let row = n_start + lane;
                let mut f32_acc = 0.0f32;

                for sg in 0..n_sg {
                    let kg_start = sg * kgs_per_sg;
                    let kg_end = ((sg + 1) * kgs_per_sg).min(k_groups);
                    let mut acc_i32 = 0i64;

                    for kg in kg_start..kg_end {
                        let kg_base = tile_base + kg * TILE_N * 2;
                        let nibs = unpack_pre_interleaved_lane(&w.data, kg_base, lane);

                        for ki in 0..4 {
                            let col = kg * 4 + ki;
                            if col < k {
                                let w_val = nibs[ki] as i32;
                                let x_val = x_i8[col] as i32;
                                acc_i32 += (w_val * x_val) as i64;
                            }
                        }
                    }

                    let corrected = acc_i32 - Q4_ZERO_BIAS * x_group_col_sums[sg];
                    let group_scale = w.scales[sg * n + row];
                    f32_acc += corrected as f32 * group_scale * x_group_scales[sg];
                }

                c[i * n + row] = f32_acc;
            }
        }
    }
}

// ============================================================================
// ASM tile processing helper (single-thread, pre-quantized x)
// ============================================================================

/// Process all tiles for a weight matrix using pre-quantized x (ASM path).
/// Caller must ensure AVX-512 VNNI is available.
#[inline]
fn matvec_q4_tiles_asm(
    x_i8: &[i8],
    x_group_scales: &[f32],
    x_group_col_sums: &[i64],
    w: &Q4Weight,
    y: &mut [f32],
) {
    let n = w.n;
    let k = w.k;
    let n_tiles = n.div_ceil(TILE_N);
    let k_groups = k.div_ceil(4);
    let n_sg = n_scale_groups(k);
    let kgs_per_sg = Q4_GROUP_SIZE / 4;


    for tile in 0..n_tiles {
        let n_start = tile * TILE_N;
        let n_end = (n_start + TILE_N).min(n);
        let tile_base = tile * k_groups * TILE_N * 2;

        if n_end - n_start == TILE_N {
            // Full tile: single fused ASM call for all SGs
            let mut f32_acc = [0.0f32; TILE_N];
            unsafe {
                avx512_q4_tile_fused(
                    x_i8.as_ptr(),
                    w.data.as_ptr().add(tile_base),
                    kgs_per_sg as u64,
                    n_sg as u64,
                    x_group_col_sums.as_ptr(),
                    w.scales_tiled.as_ptr().add(tile * n_sg * TILE_N),
                    x_group_scales.as_ptr(),
                    f32_acc.as_mut_ptr(),
                    TILE_N as u64,
                );
            }
            for lane in 0..TILE_N {
                y[n_start + lane] = f32_acc[lane];
            }
        } else {
            // Partial tile: per-SG tile kernel + scalar dequant
            let mut f32_acc = [0.0f32; TILE_N];
            for sg in 0..n_sg {
                let kg_start = sg * kgs_per_sg;
                let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
                let w_offset = tile_base + kg_start * TILE_N * 2;
                let x_offset = sg * Q4_GROUP_SIZE;

                let mut acc_i32 = [0i32; TILE_N];
                unsafe {
                    avx512_q4_matvec_tile(
                        x_i8.as_ptr().add(x_offset),
                        w.data.as_ptr().add(w_offset),
                        sg_kgs as u64,
                        acc_i32.as_mut_ptr(),
                    );
                }

                for lane in 0..(n_end - n_start) {
                    let row = n_start + lane;
                    let corrected = acc_i32[ASM_LANE_TO_ACC[lane]] as i64
                        - Q4_ZERO_BIAS * x_group_col_sums[sg];
                    let group_scale = w.scales[sg * n + row];
                    f32_acc[lane] += corrected as f32 * group_scale * x_group_scales[sg];
                }
            }
            for lane in 0..(n_end - n_start) {
                y[n_start + lane] = f32_acc[lane];
            }
        }
    }
}

// ============================================================================
// ASM tile processing — DEFERRED accumulation (single-thread, pre-quantized x)
// ============================================================================

/// Deferred integer accumulation: all tile kernels (INT) first, then all dequant (FP).
/// Reduces FP scheduler contention on Zen 4 where VPDPBUSD shares the FP pipe.
#[inline]
fn matvec_q4_tiles_asm_deferred(
    x_i8: &[i8],
    x_group_scales: &[f32],
    x_group_col_sums: &[i64],
    w: &Q4Weight,
    y: &mut [f32],
) {
    let n = w.n;
    let k = w.k;
    let n_tiles = n.div_ceil(TILE_N);
    let k_groups = k.div_ceil(4);
    let n_sg = n_scale_groups(k);
    let kgs_per_sg = Q4_GROUP_SIZE / 4;

    // Per-SG i32 buffers, reused across tiles.
    // Worst case K=9216: 288 SGs × 32 × 4 = 36 KB — fits comfortably on stack via Vec.
    let mut sg_accs = vec![0i32; n_sg * TILE_N];

    for tile in 0..n_tiles {
        let n_start = tile * TILE_N;
        let n_end = (n_start + TILE_N).min(n);
        let tile_base = tile * k_groups * TILE_N * 2;

        // Phase 1: ALL tile kernels (integer work)
        for sg in 0..n_sg {
            let kg_start = sg * kgs_per_sg;
            let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
            let w_offset = tile_base + kg_start * TILE_N * 2;
            let x_offset = sg * Q4_GROUP_SIZE;

            let acc_base = sg * TILE_N;
            for j in 0..TILE_N {
                sg_accs[acc_base + j] = 0;
            }
            unsafe {
                avx512_q4_matvec_tile(
                    x_i8.as_ptr().add(x_offset),
                    w.data.as_ptr().add(w_offset),
                    sg_kgs as u64,
                    sg_accs[acc_base..].as_mut_ptr(),
                );
            }
        }

        // Phase 2: ALL dequant (FP work)
        let mut f32_acc = [0.0f32; TILE_N];
        for sg in 0..n_sg {
            let acc_base = sg * TILE_N;
            if n_end - n_start == TILE_N {
                unsafe {
                    avx512_fused_dequant_tile(
                        sg_accs[acc_base..].as_ptr(),
                        x_group_col_sums[sg],
                        w.scales.as_ptr().add(sg * n + n_start),
                        1,
                        x_group_scales[sg],
                        f32_acc.as_mut_ptr(),
                    );
                }
            } else {
                for lane in 0..(n_end - n_start) {
                    let row = n_start + lane;
                    let corrected = sg_accs[acc_base + ASM_LANE_TO_ACC[lane]] as i64
                        - Q4_ZERO_BIAS * x_group_col_sums[sg];
                    let group_scale = w.scales[sg * n + row];
                    f32_acc[lane] += corrected as f32 * group_scale * x_group_scales[sg];
                }
            }
        }

        for lane in 0..(n_end - n_start) {
            y[n_start + lane] = f32_acc[lane];
        }
    }
}

// ============================================================================
// AVX2 tile processing — DEFERRED accumulation (single-thread, pre-quantized x)
// ============================================================================

/// Deferred integer accumulation for AVX2 path.
#[inline]
fn matvec_q4_tiles_avx2_deferred(
    x_i8: &[i8],
    x_group_scales: &[f32],
    x_group_col_sums: &[i64],
    w: &Q4Weight,
    y: &mut [f32],
) {
    let n = w.n;
    let k = w.k;
    let n_tiles = n.div_ceil(TILE_N);
    let k_groups = k.div_ceil(4);
    let n_sg = n_scale_groups(k);
    let kgs_per_sg = Q4_GROUP_SIZE / 4;

    let mut sg_accs = vec![0i32; n_sg * TILE_N];

    for tile in 0..n_tiles {
        let n_start = tile * TILE_N;
        let n_end = (n_start + TILE_N).min(n);
        let tile_base = tile * k_groups * TILE_N * 2;

        // Phase 1: ALL tile kernels (integer)
        for sg in 0..n_sg {
            let kg_start = sg * kgs_per_sg;
            let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
            let w_offset = tile_base + kg_start * TILE_N * 2;
            let x_offset = sg * Q4_GROUP_SIZE;

            let acc_base = sg * TILE_N;
            for j in 0..TILE_N {
                sg_accs[acc_base + j] = 0;
            }
            unsafe {
                avx2_q4_matvec_tile(
                    x_i8.as_ptr().add(x_offset),
                    w.data.as_ptr().add(w_offset),
                    sg_kgs as u64,
                    sg_accs[acc_base..].as_mut_ptr(),
                );
            }
        }

        // Phase 2: ALL dequant (FP)
        let mut f32_acc = [0.0f32; TILE_N];
        for sg in 0..n_sg {
            let acc_base = sg * TILE_N;
            if n_end - n_start == TILE_N {
                unsafe {
                    avx2_fused_dequant_tile(
                        sg_accs[acc_base..].as_ptr(),
                        x_group_col_sums[sg],
                        w.scales.as_ptr().add(sg * n + n_start),
                        1,
                        x_group_scales[sg],
                        f32_acc.as_mut_ptr(),
                    );
                }
            } else {
                for lane in 0..(n_end - n_start) {
                    let row = n_start + lane;
                    let corrected = sg_accs[acc_base + AVX2_LANE_TO_ACC[lane]] as i64
                        - Q4_ZERO_BIAS * x_group_col_sums[sg];
                    let group_scale = w.scales[sg * n + row];
                    f32_acc[lane] += corrected as f32 * group_scale * x_group_scales[sg];
                }
            }
        }

        for lane in 0..(n_end - n_start) {
            y[n_start + lane] = f32_acc[lane];
        }
    }
}

// ============================================================================
// AVX2 tile processing helper (single-thread, pre-quantized x)
// ============================================================================

/// Process all tiles using AVX2 ASM kernel.
/// Identical structure to matvec_q4_tiles_asm but uses AVX2 kernel + lane table.
#[inline]
fn matvec_q4_tiles_avx2(
    x_i8: &[i8],
    x_group_scales: &[f32],
    x_group_col_sums: &[i64],
    w: &Q4Weight,
    y: &mut [f32],
) {
    let n = w.n;
    let k = w.k;
    let n_tiles = n.div_ceil(TILE_N);
    let k_groups = k.div_ceil(4);
    let n_sg = n_scale_groups(k);
    let kgs_per_sg = Q4_GROUP_SIZE / 4;

    for tile in 0..n_tiles {
        let n_start = tile * TILE_N;
        let n_end = (n_start + TILE_N).min(n);
        let tile_base = tile * k_groups * TILE_N * 2;

        let mut f32_acc = [0.0f32; TILE_N];

        for sg in 0..n_sg {
            let kg_start = sg * kgs_per_sg;
            let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
            let w_offset = tile_base + kg_start * TILE_N * 2;
            let x_offset = sg * Q4_GROUP_SIZE;

            let mut acc_i32 = [0i32; TILE_N];
            unsafe {
                avx2_q4_matvec_tile(
                    x_i8.as_ptr().add(x_offset),
                    w.data.as_ptr().add(w_offset),
                    sg_kgs as u64,
                    acc_i32.as_mut_ptr(),
                );
            }

            if n_end - n_start == TILE_N {
                unsafe {
                    avx2_fused_dequant_tile(
                        acc_i32.as_ptr(),
                        x_group_col_sums[sg],
                        w.scales.as_ptr().add(sg * n + n_start),
                        1, // stride=1: column-major scales are contiguous per tile
                        x_group_scales[sg],
                        f32_acc.as_mut_ptr(),
                    );
                }
            } else {
                for lane in 0..(n_end - n_start) {
                    let row = n_start + lane;
                    let corrected = acc_i32[AVX2_LANE_TO_ACC[lane]] as i64
                        - Q4_ZERO_BIAS * x_group_col_sums[sg];
                    let group_scale = w.scales[sg * n + row];
                    f32_acc[lane] += corrected as f32 * group_scale * x_group_scales[sg];
                }
            }
        }

        for lane in 0..(n_end - n_start) {
            y[n_start + lane] = f32_acc[lane];
        }
    }
}

// ============================================================================
// ASM dispatch (single-thread)
// ============================================================================

fn matvec_q4_single(x: &[f32], w: &Q4Weight, y: &mut [f32]) {
    let deferred = crate::experimental::deferred_accum_enabled();
    if has_avx512_vnni() {
        let (x_i8, x_group_scales, x_group_col_sums) = quantize_x_and_colsums(x);
        if deferred {
            matvec_q4_tiles_asm_deferred(&x_i8, &x_group_scales, &x_group_col_sums, w, y);
        } else {
            matvec_q4_tiles_asm(&x_i8, &x_group_scales, &x_group_col_sums, w, y);
        }
        return;
    }

    if has_avx2() {
        let (x_i8, x_group_scales, x_group_col_sums) = quantize_x_and_colsums(x);
        if deferred {
            matvec_q4_tiles_avx2_deferred(&x_i8, &x_group_scales, &x_group_col_sums, w, y);
        } else {
            matvec_q4_tiles_avx2(&x_i8, &x_group_scales, &x_group_col_sums, w, y);
        }
        return;
    }

    scalar_q4_matvec(x, w, y);
}

// ============================================================================
// Public dispatch: matvec (decode, M=1)
// ============================================================================

/// Parallel matvec dispatch with pre-quantized x. Extracted from matvec_q4 for reuse in fused variants.
fn matvec_q4_prequant(
    x_i8: &[i8],
    x_group_scales: &[f32],
    x_group_col_sums: &[i64],
    w: &Q4Weight,
    y: &mut [f32],
) -> Result<()> {
    let k = w.k;
    let n = w.n;

    let n_tiles = n.div_ceil(TILE_N);
    let pool = global_pool();
    let num_threads = pool.effective_workers();
    let work_ops = n.saturating_mul(k);
    let tuner = matvec_autotuner();
    let decision = tuner.decide(work_ops, n_tiles, num_threads);
    let sample = tuner.begin_sample(&decision);

    let deferred = crate::experimental::deferred_accum_enabled();
    if !decision.parallel {
        if has_avx512_vnni() {
            if deferred {
                matvec_q4_tiles_asm_deferred(x_i8, x_group_scales, x_group_col_sums, w, y);
            } else {
                matvec_q4_tiles_asm(x_i8, x_group_scales, x_group_col_sums, w, y);
            }
        } else if has_avx2() {
            if deferred {
                matvec_q4_tiles_avx2_deferred(x_i8, x_group_scales, x_group_col_sums, w, y);
            } else {
                matvec_q4_tiles_avx2(x_i8, x_group_scales, x_group_col_sums, w, y);
            }
        } else {
            matvec_q4_tiles_asm(x_i8, x_group_scales, x_group_col_sums, w, y);
        }
        tuner.observe(sample, decision, work_ops);
        return Ok(());
    }

    let k_groups = k.div_ceil(4);
    let n_sg = n_scale_groups(k);
    let kgs_per_sg = Q4_GROUP_SIZE / 4;

    let use_avx512 = has_avx512_vnni();
    let use_avx2 = !use_avx512 && has_avx2();


    let x_i8_ptr = SendPtr::new(x_i8.as_ptr());
    let w_data_ptr = SendPtr::new(w.data.as_ptr());
    let w_scales_ptr = SendPtr::new(w.scales.as_ptr());
    let w_scales_tiled_ptr = SendPtr::new(w.scales_tiled.as_ptr());
    let x_gcs_ptr = SendPtr::new(x_group_col_sums.as_ptr());
    let x_gs_ptr = SendPtr::new(x_group_scales.as_ptr());
    let y_ptr = SendMutPtr::new(y.as_mut_ptr());

    pool.parallel_for_with_max_workers(
        n_tiles,
        decision.max_workers,
        move |_, t_start, t_end| {
            if use_avx512 || use_avx2 {
                let lane_to_acc = if use_avx512 { &ASM_LANE_TO_ACC } else { &AVX2_LANE_TO_ACC };

                if deferred {
                    // Deferred accumulation: per-worker sg_accs buffer
                    let mut sg_accs = vec![0i32; n_sg * TILE_N];
                    for tile in t_start..t_end {
                        let n_start = tile * TILE_N;
                        let n_end = (n_start + TILE_N).min(n);
                        let tile_base = tile * k_groups * TILE_N * 2;

                        // Phase 1: ALL tile kernels (integer)
                        for sg in 0..n_sg {
                            let kg_start = sg * kgs_per_sg;
                            let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
                            let w_offset = tile_base + kg_start * TILE_N * 2;
                            let x_offset = sg * Q4_GROUP_SIZE;

                            let acc_base = sg * TILE_N;
                            for j in 0..TILE_N {
                                sg_accs[acc_base + j] = 0;
                            }
                            unsafe {
                                if use_avx512 {
                                    avx512_q4_matvec_tile(
                                        x_i8_ptr.ptr().add(x_offset),
                                        w_data_ptr.ptr().add(w_offset),
                                        sg_kgs as u64,
                                        sg_accs[acc_base..].as_mut_ptr(),
                                    );
                                } else {
                                    avx2_q4_matvec_tile(
                                        x_i8_ptr.ptr().add(x_offset),
                                        w_data_ptr.ptr().add(w_offset),
                                        sg_kgs as u64,
                                        sg_accs[acc_base..].as_mut_ptr(),
                                    );
                                }
                            }
                        }

                        // Phase 2: ALL dequant (FP)
                        let mut f32_acc = [0.0f32; TILE_N];
                        for sg in 0..n_sg {
                            let acc_base = sg * TILE_N;
                            if n_end - n_start == TILE_N {
                                unsafe {
                                    let scales_base = w_scales_ptr.ptr().add(sg * n + n_start);
                                    if use_avx512 {
                                        avx512_fused_dequant_tile(
                                            sg_accs[acc_base..].as_ptr(),
                                            *x_gcs_ptr.ptr().add(sg),
                                            scales_base,
                                            1,
                                            *x_gs_ptr.ptr().add(sg),
                                            f32_acc.as_mut_ptr(),
                                        );
                                    } else {
                                        avx2_fused_dequant_tile(
                                            sg_accs[acc_base..].as_ptr(),
                                            *x_gcs_ptr.ptr().add(sg),
                                            scales_base,
                                            1,
                                            *x_gs_ptr.ptr().add(sg),
                                            f32_acc.as_mut_ptr(),
                                        );
                                    }
                                }
                            } else {
                                for lane in 0..(n_end - n_start) {
                                    let row = n_start + lane;
                                    let x_gcs = unsafe { *x_gcs_ptr.ptr().add(sg) };
                                    let x_gs = unsafe { *x_gs_ptr.ptr().add(sg) };
                                    let corrected = sg_accs[acc_base + lane_to_acc[lane]] as i64
                                        - Q4_ZERO_BIAS * x_gcs;
                                    let group_scale = unsafe {
                                        *w_scales_ptr.ptr().add(sg * n + row)
                                    };
                                    f32_acc[lane] += corrected as f32 * group_scale * x_gs;
                                }
                            }
                        }

                        for lane in 0..(n_end - n_start) {
                            unsafe {
                                *y_ptr.ptr().add(n_start + lane) = f32_acc[lane];
                            }
                        }
                    }
                    return;
                }

                for tile in t_start..t_end {
                    let n_start = tile * TILE_N;
                    let n_end = (n_start + TILE_N).min(n);
                    let tile_base = tile * k_groups * TILE_N * 2;

                    if use_avx512 && n_end - n_start == TILE_N {
                        // Full tile + AVX-512: single fused ASM call for all SGs
                        let mut f32_acc = [0.0f32; TILE_N];
                        unsafe {
                            avx512_q4_tile_fused(
                                x_i8_ptr.ptr(),
                                w_data_ptr.ptr().add(tile_base),
                                kgs_per_sg as u64,
                                n_sg as u64,
                                x_gcs_ptr.ptr(),
                                w_scales_tiled_ptr.ptr().add(tile * n_sg * TILE_N),
                                x_gs_ptr.ptr(),
                                f32_acc.as_mut_ptr(),
                                TILE_N as u64,
                            );
                        }
                        for lane in 0..TILE_N {
                            unsafe {
                                *y_ptr.ptr().add(n_start + lane) = f32_acc[lane];
                            }
                        }
                        continue;
                    }

                    let mut f32_acc = [0.0f32; TILE_N];

                    for sg in 0..n_sg {
                        let kg_start = sg * kgs_per_sg;
                        let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
                        let w_offset = tile_base + kg_start * TILE_N * 2;
                        let x_offset = sg * Q4_GROUP_SIZE;

                        let mut acc_i32 = [0i32; TILE_N];
                        unsafe {
                            if use_avx512 {
                                avx512_q4_matvec_tile(
                                    x_i8_ptr.ptr().add(x_offset),
                                    w_data_ptr.ptr().add(w_offset),
                                    sg_kgs as u64,
                                    acc_i32.as_mut_ptr(),
                                );
                            } else {
                                avx2_q4_matvec_tile(
                                    x_i8_ptr.ptr().add(x_offset),
                                    w_data_ptr.ptr().add(w_offset),
                                    sg_kgs as u64,
                                    acc_i32.as_mut_ptr(),
                                );
                            }
                        }

                        if n_end - n_start == TILE_N {
                            unsafe {
                                let scales_base = w_scales_ptr.ptr().add(sg * n + n_start);
                                // AVX2 full tile (AVX-512 full tiles handled above)
                                avx2_fused_dequant_tile(
                                    acc_i32.as_ptr(),
                                    *x_gcs_ptr.ptr().add(sg),
                                    scales_base,
                                    1,
                                    *x_gs_ptr.ptr().add(sg),
                                    f32_acc.as_mut_ptr(),
                                );
                            }
                        } else {
                            for lane in 0..(n_end - n_start) {
                                let row = n_start + lane;
                                let x_gcs = unsafe { *x_gcs_ptr.ptr().add(sg) };
                                let x_gs = unsafe { *x_gs_ptr.ptr().add(sg) };
                                let corrected = acc_i32[lane_to_acc[lane]] as i64
                                    - Q4_ZERO_BIAS * x_gcs;
                                let group_scale = unsafe {
                                    *w_scales_ptr.ptr().add(sg * n + row)
                                };
                                f32_acc[lane] += corrected as f32 * group_scale * x_gs;
                            }
                        }
                    }

                    for lane in 0..(n_end - n_start) {
                        unsafe {
                            *y_ptr.ptr().add(n_start + lane) = f32_acc[lane];
                        }
                    }
                }
                return;
            }

            // Scalar fallback in parallel
            for tile in t_start..t_end {
                let n_start = tile * TILE_N;
                let n_end = (n_start + TILE_N).min(n);
                let tile_base = tile * k_groups * TILE_N * 2;

                for lane in 0..(n_end - n_start) {
                    let row = n_start + lane;
                    let mut f32_acc = 0.0f32;

                    for sg in 0..n_sg {
                        let kg_start = sg * kgs_per_sg;
                        let kg_end = ((sg + 1) * kgs_per_sg).min(k_groups);
                        let x_gcs = unsafe { *x_gcs_ptr.ptr().add(sg) };
                        let x_gs = unsafe { *x_gs_ptr.ptr().add(sg) };
                        let mut acc_i32 = 0i64;

                        for kg in kg_start..kg_end {
                            let kg_base = tile_base + kg * TILE_N * 2;
                            unsafe {
                                let nibs = unpack_pre_interleaved_lane_ptr(w_data_ptr.ptr(), kg_base, lane);

                                for ki in 0..4 {
                                    let col = kg * 4 + ki;
                                    if col < k {
                                        let w_val = nibs[ki] as i32;
                                        let x_val = *x_i8_ptr.ptr().add(col) as i32;
                                        acc_i32 += (w_val * x_val) as i64;
                                    }
                                }
                            }
                        }

                        let corrected = acc_i32 - Q4_ZERO_BIAS * x_gcs;
                        let group_scale = unsafe {
                            *w_scales_ptr.ptr().add(sg * n + row)
                        };
                        f32_acc += corrected as f32 * group_scale * x_gs;
                    }

                    unsafe {
                        *y_ptr.ptr().add(row) = f32_acc;
                    }
                }
            }
        },
    )
    .map_err(|e| HerbertError::Backend(format!("matvec_q4 thread-pool error: {}", e)))?;
    tuner.observe(sample, decision, work_ops);

    Ok(())
}

pub fn matvec_q4(x: &[f32], w: &Q4Weight, y: &mut [f32]) -> Result<()> {
    let k = w.k;
    debug_assert!(x.len() >= k, "x.len()={} < k={}", x.len(), k);
    debug_assert_eq!(y.len(), w.n);
    let (x_i8, x_group_scales, x_group_col_sums) = quantize_x_and_colsums(x);
    matvec_q4_prequant(&x_i8, &x_group_scales, &x_group_col_sums, w, y)
}

// ============================================================================
// VNNI matmul: per-group i8 input quantization + VPDPBUSD per scale group.
// Same precision as matvec (decode) path — per-group quantization prevents
// activation outliers from destroying precision.
// ============================================================================

/// Inner matmul loop using pre-quantized input rows.
/// Extracted to enable sharing quantized rows between gate and up projections.
fn matmul_q4_with_prequant(
    quants: &[(Vec<i8>, Vec<f32>, Vec<i64>)],
    w: &Q4Weight,
    c: &mut [f32],
    m: usize,
    use_avx512: bool,
) {
    let k = w.k;
    let n = w.n;
    let n_tiles = n.div_ceil(TILE_N);
    let k_groups = k.div_ceil(4);
    let n_sg = n_scale_groups(k);
    let kgs_per_sg = Q4_GROUP_SIZE / 4;
    let lane_to_acc = if use_avx512 { &ASM_LANE_TO_ACC } else { &AVX2_LANE_TO_ACC };


    const M_BLOCK: usize = 32;

    for tile in 0..n_tiles {
        let n_start = tile * TILE_N;
        let n_end = (n_start + TILE_N).min(n);
        let tile_base = tile * k_groups * TILE_N * 2;
        let full_tile = n_end - n_start == TILE_N;

        let mut m_off = 0;
        while m_off < m {
            let chunk_end = (m_off + M_BLOCK).min(m);
            let chunk_m = chunk_end - m_off;
            let mut f32_accs = [[0.0f32; TILE_N]; M_BLOCK];

            for sg in 0..n_sg {
                let kg_start = sg * kgs_per_sg;
                let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
                let w_offset = tile_base + kg_start * TILE_N * 2;
                let x_offset = sg * Q4_GROUP_SIZE;

                for bi in 0..chunk_m {
                    let (ref x_i8, ref x_scales, ref x_colsums) = quants[m_off + bi];
                    let mut acc_i32 = [0i32; TILE_N];
                    unsafe {
                        if use_avx512 {
                            avx512_q4_matvec_tile(
                                x_i8.as_ptr().add(x_offset),
                                w.data.as_ptr().add(w_offset),
                                sg_kgs as u64,
                                acc_i32.as_mut_ptr(),
                            );
                        } else {
                            avx2_q4_matvec_tile(
                                x_i8.as_ptr().add(x_offset),
                                w.data.as_ptr().add(w_offset),
                                sg_kgs as u64,
                                acc_i32.as_mut_ptr(),
                            );
                        }
                    }

                    if full_tile {
                        unsafe {
                            if use_avx512 {
                                avx512_fused_dequant_tile(
                                    acc_i32.as_ptr(),
                                    x_colsums[sg],
                                    w.scales.as_ptr().add(sg * n + n_start),
                                    1,
                                    x_scales[sg],
                                    f32_accs[bi].as_mut_ptr(),
                                );
                            } else {
                                avx2_fused_dequant_tile(
                                    acc_i32.as_ptr(),
                                    x_colsums[sg],
                                    w.scales.as_ptr().add(sg * n + n_start),
                                    1,
                                    x_scales[sg],
                                    f32_accs[bi].as_mut_ptr(),
                                );
                            }
                        }
                    } else {
                        for lane in 0..(n_end - n_start) {
                            let row = n_start + lane;
                            let corrected = acc_i32[lane_to_acc[lane]] as i64
                                - Q4_ZERO_BIAS * x_colsums[sg];
                            let group_scale = w.scales[sg * w.n + row];
                            f32_accs[bi][lane] += corrected as f32 * group_scale * x_scales[sg];
                        }
                    }
                }
            }

            for bi in 0..chunk_m {
                let i = m_off + bi;
                for lane in 0..(n_end - n_start) {
                    c[i * n + n_start + lane] = f32_accs[bi][lane];
                }
            }

            m_off = chunk_end;
        }
    }
}

fn matmul_q4_sequential(a: &[f32], w: &Q4Weight, c: &mut [f32], m: usize) {
    let k = w.k;
    let n = w.n;
    let n_tiles = n.div_ceil(TILE_N);
    let k_groups = k.div_ceil(4);
    let n_sg = n_scale_groups(k);
    let kgs_per_sg = Q4_GROUP_SIZE / 4;

    let use_avx512 = has_avx512_vnni();
    let use_avx2 = !use_avx512 && has_avx2();


    if use_avx512 || use_avx2 {
        let lane_to_acc = if use_avx512 { &ASM_LANE_TO_ACC } else { &AVX2_LANE_TO_ACC };

        // Pre-quantize all input rows (amortized cost, enables tile-outer ordering).
        let mut quants: Vec<(Vec<i8>, Vec<f32>, Vec<i64>)> = Vec::with_capacity(m);
        for i in 0..m {
            quants.push(quantize_x_and_colsums(&a[i * k..(i + 1) * k]));
        }

        // Tile-outer, M-chunked loop ordering for L2 weight reuse.
        // Each tile's weight data (~56 KB for K=3584) stays in L2 across all rows.
        // Within each M chunk, per-row accumulators fit in L1 (M_BLOCK * 128 B = 4 KB).
        // Weight per (tile, sg) = 512 B stays L1-hot across all rows in the chunk.
        const M_BLOCK: usize = 32;

        for tile in 0..n_tiles {
            let n_start = tile * TILE_N;
            let n_end = (n_start + TILE_N).min(n);
            let tile_base = tile * k_groups * TILE_N * 2;
            let full_tile = n_end - n_start == TILE_N;

            let mut m_off = 0;
            while m_off < m {
                let chunk_end = (m_off + M_BLOCK).min(m);
                let chunk_m = chunk_end - m_off;

                let mut f32_accs = [[0.0f32; TILE_N]; M_BLOCK];

                for sg in 0..n_sg {
                    let kg_start = sg * kgs_per_sg;
                    let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
                    let w_offset = tile_base + kg_start * TILE_N * 2;
                    let x_offset = sg * Q4_GROUP_SIZE;

                    // Weight data for (tile, sg) = sg_kgs * TILE_N * 2 bytes (~512 B).
                    // Stays L1-hot while processing all rows in this chunk.
                    for bi in 0..chunk_m {
                        let (ref x_i8, ref x_scales, ref x_colsums) = quants[m_off + bi];

                        let mut acc_i32 = [0i32; TILE_N];
                        unsafe {
                            if use_avx512 {
                                avx512_q4_matvec_tile(
                                    x_i8.as_ptr().add(x_offset),
                                    w.data.as_ptr().add(w_offset),
                                    sg_kgs as u64,
                                    acc_i32.as_mut_ptr(),
                                );
                            } else {
                                avx2_q4_matvec_tile(
                                    x_i8.as_ptr().add(x_offset),
                                    w.data.as_ptr().add(w_offset),
                                    sg_kgs as u64,
                                    acc_i32.as_mut_ptr(),
                                );
                            }
                        }

                        if full_tile {
                            unsafe {
                                if use_avx512 {
                                    avx512_fused_dequant_tile(
                                        acc_i32.as_ptr(),
                                        x_colsums[sg],
                                        w.scales.as_ptr().add(sg * n + n_start),
                                        1,
                                        x_scales[sg],
                                        f32_accs[bi].as_mut_ptr(),
                                    );
                                } else {
                                    avx2_fused_dequant_tile(
                                        acc_i32.as_ptr(),
                                        x_colsums[sg],
                                        w.scales.as_ptr().add(sg * n + n_start),
                                        1,
                                        x_scales[sg],
                                        f32_accs[bi].as_mut_ptr(),
                                    );
                                }
                            }
                        } else {
                            for lane in 0..(n_end - n_start) {
                                let row = n_start + lane;
                                let corrected = acc_i32[lane_to_acc[lane]] as i64
                                    - Q4_ZERO_BIAS * x_colsums[sg];
                                let group_scale = w.scales[sg * n + row];
                                f32_accs[bi][lane] += corrected as f32 * group_scale * x_scales[sg];
                            }
                        }
                    }
                }

                for bi in 0..chunk_m {
                    let i = m_off + bi;
                    for lane in 0..(n_end - n_start) {
                        c[i * n + n_start + lane] = f32_accs[bi][lane];
                    }
                }

                m_off = chunk_end;
            }
        }
        return;
    }

    // Scalar fallback
    scalar_q4_matmul(a, w, c, m);
}

// ============================================================================
// Public dispatch: matmul (prefill, M>1)
// ============================================================================

pub fn matmul_q4(
    a: &[f32],
    w: &Q4Weight,
    c: &mut [f32],
    m: usize,
) -> Result<()> {
    let k = w.k;
    let n = w.n;
    debug_assert!(a.len() >= m * k, "a.len()={} < m*k={}", a.len(), m * k);
    debug_assert!(c.len() >= m * n, "c.len()={} < m*n={}", c.len(), m * n);

    let n_tiles = n.div_ceil(TILE_N);
    let k_groups = k.div_ceil(4);
    let n_sg = n_scale_groups(k);
    let kgs_per_sg = Q4_GROUP_SIZE / 4;

    let pool = global_pool();
    let num_threads = pool.num_workers();
    let work_ops = m.saturating_mul(n).saturating_mul(k);
    let tuner = matmul_autotuner();
    let decision = tuner.decide(work_ops, m, num_threads);
    let sample = tuner.begin_sample(&decision);

    if !decision.parallel {
        matmul_q4_sequential(a, w, c, m);
        tuner.observe(sample, decision, work_ops);
        return Ok(());
    }

    // Parallel matmul: per-group i8 quantization + ASM kernel per group
    let use_avx512 = has_avx512_vnni();
    let use_avx2 = !use_avx512 && has_avx2();


    let a_ptr = SendPtr::new(a.as_ptr());
    let w_data_ptr = SendPtr::new(w.data.as_ptr());
    let w_scales_ptr = SendPtr::new(w.scales.as_ptr());
    let c_ptr = SendMutPtr::new(c.as_mut_ptr());

    pool.parallel_for_with_max_workers(m, decision.max_workers, move |_, m_start, m_end| {
        if use_avx512 || use_avx2 {
            let lane_to_acc = if use_avx512 { &ASM_LANE_TO_ACC } else { &AVX2_LANE_TO_ACC };
            let local_m = m_end - m_start;

            // Pre-quantize this thread's rows
            let mut quants: Vec<(Vec<i8>, Vec<f32>, Vec<i64>)> = Vec::with_capacity(local_m);
            for i in m_start..m_end {
                let a_row = unsafe { std::slice::from_raw_parts(a_ptr.ptr().add(i * k), k) };
                quants.push(quantize_x_and_colsums(a_row));
            }

            // Tile-outer ordering: weight data stays L2-hot across all rows
            const M_BLOCK: usize = 32;

            for tile in 0..n_tiles {
                let n_start = tile * TILE_N;
                let n_end = (n_start + TILE_N).min(n);
                let tile_base = tile * k_groups * TILE_N * 2;
                let full_tile = n_end - n_start == TILE_N;

                let mut m_off = 0;
                while m_off < local_m {
                    let chunk_end = (m_off + M_BLOCK).min(local_m);
                    let chunk_m = chunk_end - m_off;

                    let mut f32_accs = [[0.0f32; TILE_N]; M_BLOCK];

                    for sg in 0..n_sg {
                        let kg_start = sg * kgs_per_sg;
                        let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
                        let w_offset = tile_base + kg_start * TILE_N * 2;
                        let x_offset = sg * Q4_GROUP_SIZE;

                        for bi in 0..chunk_m {
                            let (ref x_i8, ref x_scales, ref x_colsums) = quants[m_off + bi];

                            let mut acc_i32 = [0i32; TILE_N];
                            unsafe {
                                if use_avx512 {
                                    avx512_q4_matvec_tile(
                                        x_i8.as_ptr().add(x_offset),
                                        w_data_ptr.ptr().add(w_offset),
                                        sg_kgs as u64,
                                        acc_i32.as_mut_ptr(),
                                    );
                                } else {
                                    avx2_q4_matvec_tile(
                                        x_i8.as_ptr().add(x_offset),
                                        w_data_ptr.ptr().add(w_offset),
                                        sg_kgs as u64,
                                        acc_i32.as_mut_ptr(),
                                    );
                                }
                            }

                            if full_tile {
                                unsafe {
                                    let scales_base = w_scales_ptr.ptr().add(sg * n + n_start);
                                    if use_avx512 {
                                        avx512_fused_dequant_tile(
                                            acc_i32.as_ptr(),
                                            x_colsums[sg],
                                            scales_base,
                                            1,
                                            x_scales[sg],
                                            f32_accs[bi].as_mut_ptr(),
                                        );
                                    } else {
                                        avx2_fused_dequant_tile(
                                            acc_i32.as_ptr(),
                                            x_colsums[sg],
                                            scales_base,
                                            1,
                                            x_scales[sg],
                                            f32_accs[bi].as_mut_ptr(),
                                        );
                                    }
                                }
                            } else {
                                for lane in 0..(n_end - n_start) {
                                    let row = n_start + lane;
                                    let corrected = acc_i32[lane_to_acc[lane]] as i64
                                        - Q4_ZERO_BIAS * x_colsums[sg];
                                    let group_scale = unsafe {
                                        *w_scales_ptr.ptr().add(sg * n + row)
                                    };
                                    f32_accs[bi][lane] += corrected as f32 * group_scale * x_scales[sg];
                                }
                            }
                        }
                    }

                    for bi in 0..chunk_m {
                        let i = m_start + m_off + bi;
                        for lane in 0..(n_end - n_start) {
                            unsafe {
                                *c_ptr.ptr().add(i * n + n_start + lane) = f32_accs[bi][lane];
                            }
                        }
                    }

                    m_off = chunk_end;
                }
            }
            return;
        }

        // Scalar float dequant fallback
        for i in m_start..m_end {
            for tile in 0..n_tiles {
                let n_start = tile * TILE_N;
                let n_end = (n_start + TILE_N).min(n);
                let tile_base = tile * k_groups * TILE_N * 2;

                for lane in 0..(n_end - n_start) {
                    let out_col = n_start + lane;
                    let mut acc = 0.0f32;
                    let kgs_per_wg = Q4_GROUP_SIZE / 4;

                    for kg in 0..k_groups {
                        let kg_base = tile_base + kg * TILE_N * 2;
                        unsafe {
                            let nibs = unpack_pre_interleaved_lane_ptr(w_data_ptr.ptr(), kg_base, lane);

                            let wg = kg / kgs_per_wg;
                            let group_scale = *w_scales_ptr.ptr().add(wg * n + out_col);

                            for ki in 0..4 {
                                let col = kg * 4 + ki;
                                if col < k {
                                    let w_dequant = (nibs[ki] as f32 - 8.0) * group_scale;
                                    acc += *a_ptr.ptr().add(i * k + col) * w_dequant;
                                }
                            }
                        }
                    }

                    unsafe {
                        *c_ptr.ptr().add(i * n + out_col) = acc;
                    }
                }
            }
        }
    })
    .map_err(|e| HerbertError::Backend(format!("matmul_q4 thread-pool error: {}", e)))?;
    tuner.observe(sample, decision, work_ops);

    Ok(())
}

// ============================================================================
// Single-threaded variants for parallel expert dispatch (v8)
// ============================================================================

pub fn matmul_q4_st(
    a: &[f32],
    w: &Q4Weight,
    c: &mut [f32],
    m: usize,
) -> Result<()> {
    let k = w.k;
    let n = w.n;
    debug_assert!(a.len() >= m * k, "a.len()={} < m*k={}", a.len(), m * k);
    debug_assert!(c.len() >= m * n, "c.len()={} < m*n={}", c.len(), m * n);
    matmul_q4_sequential(a, w, c, m);
    Ok(())
}

/// Fused gate+up matmul (single-threaded): quantizes each input row once for both projections.
/// Saves M redundant quantizations compared to two separate matmul_q4_st calls.
pub fn fused_gate_up_matmul_q4_st(
    a: &[f32],
    w_gate: &Q4Weight,
    w_up: &Q4Weight,
    c_gate: &mut [f32],
    c_up: &mut [f32],
    m: usize,
) -> Result<()> {
    let k = w_gate.k;
    debug_assert_eq!(k, w_up.k, "gate and up must have same input dim");
    let n_gate = w_gate.n;
    let n_up = w_up.n;
    debug_assert!(a.len() >= m * k);
    debug_assert!(c_gate.len() >= m * n_gate);
    debug_assert!(c_up.len() >= m * n_up);

    let use_avx512 = has_avx512_vnni();
    let use_avx2 = !use_avx512 && has_avx2();

    if use_avx512 || use_avx2 {
        // Pre-quantize all M rows ONCE (shared between gate and up).
        let mut quants: Vec<(Vec<i8>, Vec<f32>, Vec<i64>)> = Vec::with_capacity(m);
        for i in 0..m {
            quants.push(quantize_x_and_colsums(&a[i * k..(i + 1) * k]));
        }

        // Process gate with pre-quantized rows
        matmul_q4_with_prequant(&quants, w_gate, c_gate, m, use_avx512);
        // Process up with the same pre-quantized rows
        matmul_q4_with_prequant(&quants, w_up, c_up, m, use_avx512);
        return Ok(());
    }

    // Scalar fallback
    scalar_q4_matmul(a, w_gate, c_gate, m);
    scalar_q4_matmul(a, w_up, c_up, m);
    Ok(())
}

pub fn matvec_q4_st(x: &[f32], w: &Q4Weight, y: &mut [f32]) -> Result<()> {
    let k = w.k;
    let n = w.n;
    debug_assert!(x.len() >= k, "x.len()={} < k={}", x.len(), k);
    debug_assert_eq!(y.len(), n);
    matvec_q4_single(x, w, y);
    Ok(())
}

/// Fused gate+up matvec: quantizes x once, then processes both weight matrices.
/// Saves ~50% of x quantization + col_sums compute, and x_i8 stays L1-hot for up.
pub fn fused_gate_up_matvec_q4_st(
    x: &[f32],
    w_gate: &Q4Weight,
    w_up: &Q4Weight,
    y_gate: &mut [f32],
    y_up: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(w_gate.k, w_up.k, "gate and up must have same input dim");
    let k = w_gate.k;
    debug_assert!(x.len() >= k, "x.len()={} < k={}", x.len(), k);
    debug_assert_eq!(y_gate.len(), w_gate.n);
    debug_assert_eq!(y_up.len(), w_up.n);

    if has_avx512_vnni() {
        let (x_i8, x_group_scales, x_group_col_sums) = quantize_x_and_colsums(x);
        matvec_q4_tiles_asm(&x_i8, &x_group_scales, &x_group_col_sums, w_gate, y_gate);
        matvec_q4_tiles_asm(&x_i8, &x_group_scales, &x_group_col_sums, w_up, y_up);
        return Ok(());
    }

    if has_avx2() {
        let (x_i8, x_group_scales, x_group_col_sums) = quantize_x_and_colsums(x);
        matvec_q4_tiles_avx2(&x_i8, &x_group_scales, &x_group_col_sums, w_gate, y_gate);
        matvec_q4_tiles_avx2(&x_i8, &x_group_scales, &x_group_col_sums, w_up, y_up);
        return Ok(());
    }

    scalar_q4_matvec(x, w_gate, y_gate);
    scalar_q4_matvec(x, w_up, y_up);
    Ok(())
}

/// Fused gate+up matvec with pre-quantized input (single-threaded).
/// Skips quantization of x — uses the provided i8/scales/col_sums directly.
pub fn fused_gate_up_matvec_q4_pq_st(
    x_i8: &[i8],
    x_group_scales: &[f32],
    x_group_col_sums: &[i64],
    w_gate: &Q4Weight,
    w_up: &Q4Weight,
    y_gate: &mut [f32],
    y_up: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(w_gate.k, w_up.k, "gate and up must have same input dim");
    debug_assert_eq!(y_gate.len(), w_gate.n);
    debug_assert_eq!(y_up.len(), w_up.n);

    if has_avx512_vnni() {
        matvec_q4_tiles_asm(x_i8, x_group_scales, x_group_col_sums, w_gate, y_gate);
        matvec_q4_tiles_asm(x_i8, x_group_scales, x_group_col_sums, w_up, y_up);
        return Ok(());
    }

    if has_avx2() {
        matvec_q4_tiles_avx2(x_i8, x_group_scales, x_group_col_sums, w_gate, y_gate);
        matvec_q4_tiles_avx2(x_i8, x_group_scales, x_group_col_sums, w_up, y_up);
        return Ok(());
    }

    // Scalar fallback — need f32 x, reconstruct approximately
    // (this path shouldn't happen on real hardware)
    let k = w_gate.k;
    let mut x_f32 = vec![0.0f32; k];
    for g in 0..x_group_scales.len() {
        let scale = x_group_scales[g];
        let start = g * Q4_GROUP_SIZE;
        let end = (start + Q4_GROUP_SIZE).min(k);
        for i in start..end {
            x_f32[i] = x_i8[i] as f32 * scale;
        }
    }
    scalar_q4_matvec(&x_f32, w_gate, y_gate);
    scalar_q4_matvec(&x_f32, w_up, y_up);
    Ok(())
}

/// Single-threaded matvec with pre-quantized input.
/// Skips quantization — uses provided i8/scales/col_sums directly.
pub fn matvec_q4_pq_st(
    x_i8: &[i8],
    x_group_scales: &[f32],
    x_group_col_sums: &[i64],
    w: &Q4Weight,
    y: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(y.len(), w.n);

    if has_avx512_vnni() {
        matvec_q4_tiles_asm(x_i8, x_group_scales, x_group_col_sums, w, y);
        return Ok(());
    }

    if has_avx2() {
        matvec_q4_tiles_avx2(x_i8, x_group_scales, x_group_col_sums, w, y);
        return Ok(());
    }

    let k = w.k;
    let mut x_f32 = vec![0.0f32; k];
    for g in 0..x_group_scales.len() {
        let scale = x_group_scales[g];
        let start = g * Q4_GROUP_SIZE;
        let end = (start + Q4_GROUP_SIZE).min(k);
        for i in start..end {
            x_f32[i] = x_i8[i] as f32 * scale;
        }
    }
    scalar_q4_matvec(&x_f32, w, y);
    Ok(())
}

// ============================================================================
// Float dequant matvec (no activation quantization — for DeltaNet)
// ============================================================================

/// Matvec with float dequantization: dequantize Q4 weights to f32 on-the-fly,
/// multiply with f32 activations directly. No i8 activation quantization.
///
/// This avoids the double quantization error (i8 activations × Q4 weights) that
/// causes rumination in recurrent layers like DeltaNet where errors accumulate.
/// Slower than VPDPBUSD but more precise.
pub fn matvec_q4_float_dequant(x: &[f32], w: &Q4Weight, y: &mut [f32]) -> Result<()> {
    let k = w.k;
    let n = w.n;
    debug_assert!(x.len() >= k, "x.len()={} < k={}", x.len(), k);
    debug_assert_eq!(y.len(), n);

    let n_tiles = n.div_ceil(TILE_N);
    let k_groups = k.div_ceil(4);
    let kgs_per_wg = Q4_GROUP_SIZE / 4;

    let pool = global_pool();
    let num_threads = pool.num_workers();
    let work_ops = n.saturating_mul(k);
    let tuner = matvec_autotuner();
    let decision = tuner.decide(work_ops, n_tiles, num_threads);
    let sample = tuner.begin_sample(&decision);

    if !decision.parallel {
        // Single-threaded scalar float dequant
        for tile in 0..n_tiles {
            let n_start = tile * TILE_N;
            let n_end = (n_start + TILE_N).min(n);
            let tile_base = tile * k_groups * TILE_N * 2;

            for lane in 0..(n_end - n_start) {
                let out_col = n_start + lane;
                let mut acc = 0.0f32;

                for kg in 0..k_groups {
                    let kg_base = tile_base + kg * TILE_N * 2;
                    let nibs = unpack_pre_interleaved_lane(&w.data, kg_base, lane);

                    let wg = kg / kgs_per_wg;
                    let group_scale = w.scales[wg * n + out_col];

                    for ki in 0..4 {
                        let col = kg * 4 + ki;
                        if col < k {
                            let w_dequant = (nibs[ki] as f32 - 8.0) * group_scale;
                            acc += x[col] * w_dequant;
                        }
                    }
                }

                y[out_col] = acc;
            }
        }
        tuner.observe(sample, decision, work_ops);
        return Ok(());
    }

    // Parallel float dequant
    let x_ptr = SendPtr::new(x.as_ptr());
    let w_data_ptr = SendPtr::new(w.data.as_ptr());
    let w_scales_ptr = SendPtr::new(w.scales.as_ptr());
    let y_ptr = SendMutPtr::new(y.as_mut_ptr());

    pool.parallel_for_with_max_workers(
        n_tiles,
        decision.max_workers,
        move |_, t_start, t_end| {
            for tile in t_start..t_end {
                let n_start = tile * TILE_N;
                let n_end = (n_start + TILE_N).min(n);
                let tile_base = tile * k_groups * TILE_N * 2;

                for lane in 0..(n_end - n_start) {
                    let out_col = n_start + lane;
                    let mut acc = 0.0f32;

                    for kg in 0..k_groups {
                        let kg_base = tile_base + kg * TILE_N * 2;
                        unsafe {
                            let nibs = unpack_pre_interleaved_lane_ptr(w_data_ptr.ptr(), kg_base, lane);

                            let wg = kg / kgs_per_wg;
                            let group_scale = *w_scales_ptr.ptr().add(wg * n + out_col);

                            for ki in 0..4 {
                                let col = kg * 4 + ki;
                                if col < k {
                                    let w_dequant = (nibs[ki] as f32 - 8.0) * group_scale;
                                    acc += *x_ptr.ptr().add(col) * w_dequant;
                                }
                            }
                        }
                    }

                    unsafe {
                        *y_ptr.ptr().add(out_col) = acc;
                    }
                }
            }
        },
    )
    .map_err(|e| HerbertError::Backend(format!("matvec_q4_float_dequant thread-pool error: {}", e)))?;
    tuner.observe(sample, decision, work_ops);

    Ok(())
}

/// Single-threaded float dequant matvec — safe to call from within `parallel_for`.
/// No autotuner, no thread pool. Used for MoE expert dispatch.
pub fn matvec_q4_float_dequant_st(x: &[f32], w: &Q4Weight, y: &mut [f32]) -> Result<()> {
    let k = w.k;
    let n = w.n;
    debug_assert!(x.len() >= k, "x.len()={} < k={}", x.len(), k);
    debug_assert_eq!(y.len(), n);

    let n_tiles = n.div_ceil(TILE_N);
    let k_groups = k.div_ceil(4);
    let kgs_per_wg = Q4_GROUP_SIZE / 4;

    for tile in 0..n_tiles {
        let n_start = tile * TILE_N;
        let n_end = (n_start + TILE_N).min(n);
        let tile_base = tile * k_groups * TILE_N * 2;

        for lane in 0..(n_end - n_start) {
            let out_col = n_start + lane;
            let mut acc = 0.0f32;

            for kg in 0..k_groups {
                let kg_base = tile_base + kg * TILE_N * 2;
                let nibs = unpack_pre_interleaved_lane(&w.data, kg_base, lane);

                let wg = kg / kgs_per_wg;
                let group_scale = w.scales[wg * n + out_col];

                for ki in 0..4 {
                    let col = kg * 4 + ki;
                    if col < k {
                        let w_dequant = (nibs[ki] as f32 - 8.0) * group_scale;
                        acc += x[col] * w_dequant;
                    }
                }
            }

            y[out_col] = acc;
        }
    }

    Ok(())
}

/// Fused 2-projection matvec: quantizes x once, SINGLE thread pool dispatch for both projections.
/// Both weights must share the same K dimension (same input).
/// Tiles from w1 and w2 are combined into one parallel_for over (n_tiles_1 + n_tiles_2) tiles.
pub fn fused_2_matvec_q4(
    x: &[f32],
    w1: &Q4Weight, y1: &mut [f32],
    w2: &Q4Weight, y2: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(w1.k, w2.k, "fused_2_matvec: w1.k={} != w2.k={}", w1.k, w2.k);
    let k = w1.k;
    debug_assert!(x.len() >= k);

    let (x_i8, x_group_scales, x_group_col_sums) = quantize_x_and_colsums(x);

    let n1 = w1.n;
    let n2 = w2.n;
    let n_tiles_1 = n1.div_ceil(TILE_N);
    let n_tiles_2 = n2.div_ceil(TILE_N);
    let total_tiles = n_tiles_1 + n_tiles_2;

    let pool = global_pool();
    let k_groups = k.div_ceil(4);
    let n_sg = n_scale_groups(k);
    let kgs_per_sg = Q4_GROUP_SIZE / 4;
    let use_avx512 = has_avx512_vnni();
    let use_avx2 = !use_avx512 && has_avx2();
    let deferred = crate::experimental::deferred_accum_enabled();


    let x_i8_ptr = SendPtr::new(x_i8.as_ptr());
    let x_gcs_ptr = SendPtr::new(x_group_col_sums.as_ptr());
    let x_gs_ptr = SendPtr::new(x_group_scales.as_ptr());
    let w1_data = SendPtr::new(w1.data.as_ptr());
    let w1_scales = SendPtr::new(w1.scales.as_ptr());
    let y1_ptr = SendMutPtr::new(y1.as_mut_ptr());
    let w2_data = SendPtr::new(w2.data.as_ptr());
    let w2_scales = SendPtr::new(w2.scales.as_ptr());
    let y2_ptr = SendMutPtr::new(y2.as_mut_ptr());

    pool.parallel_for_phase_aware(total_tiles, move |_, t_start, t_end| {
        if !(use_avx512 || use_avx2) { return; }
        let lane_to_acc = if use_avx512 { &ASM_LANE_TO_ACC } else { &AVX2_LANE_TO_ACC };

        if deferred {
            let mut sg_accs = vec![0i32; n_sg * TILE_N];
            for global_tile in t_start..t_end {
                let (tile, n, wd, ws, yp) = if global_tile < n_tiles_1 {
                    (global_tile, n1, w1_data, w1_scales, y1_ptr)
                } else {
                    (global_tile - n_tiles_1, n2, w2_data, w2_scales, y2_ptr)
                };

                let n_start = tile * TILE_N;
                let n_end = (n_start + TILE_N).min(n);
                let tile_base = tile * k_groups * TILE_N * 2;

                // Phase 1: ALL tile kernels (integer)
                for sg in 0..n_sg {
                    let kg_start = sg * kgs_per_sg;
                    let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
                    let w_offset = tile_base + kg_start * TILE_N * 2;
                    let x_offset = sg * Q4_GROUP_SIZE;

                    let acc_base = sg * TILE_N;
                    for j in 0..TILE_N { sg_accs[acc_base + j] = 0; }
                    unsafe {
                        if use_avx512 {
                            avx512_q4_matvec_tile(
                                x_i8_ptr.ptr().add(x_offset),
                                wd.ptr().add(w_offset),
                                sg_kgs as u64,
                                sg_accs[acc_base..].as_mut_ptr(),
                            );
                        } else {
                            avx2_q4_matvec_tile(
                                x_i8_ptr.ptr().add(x_offset),
                                wd.ptr().add(w_offset),
                                sg_kgs as u64,
                                sg_accs[acc_base..].as_mut_ptr(),
                            );
                        }
                    }
                }

                // Phase 2: ALL dequant (FP)
                let mut f32_acc = [0.0f32; TILE_N];
                for sg in 0..n_sg {
                    let acc_base = sg * TILE_N;
                    if n_end - n_start == TILE_N {
                        unsafe {
                            let scales_base = ws.ptr().add(sg * n + n_start);
                            if use_avx512 {
                                avx512_fused_dequant_tile(
                                    sg_accs[acc_base..].as_ptr(),
                                    *x_gcs_ptr.ptr().add(sg),
                                    scales_base,
                                    1,
                                    *x_gs_ptr.ptr().add(sg),
                                    f32_acc.as_mut_ptr(),
                                );
                            } else {
                                avx2_fused_dequant_tile(
                                    sg_accs[acc_base..].as_ptr(),
                                    *x_gcs_ptr.ptr().add(sg),
                                    scales_base,
                                    1,
                                    *x_gs_ptr.ptr().add(sg),
                                    f32_acc.as_mut_ptr(),
                                );
                            }
                        }
                    } else {
                        for lane in 0..(n_end - n_start) {
                            let row = n_start + lane;
                            let x_gcs = unsafe { *x_gcs_ptr.ptr().add(sg) };
                            let x_gs = unsafe { *x_gs_ptr.ptr().add(sg) };
                            let corrected = sg_accs[acc_base + lane_to_acc[lane]] as i64
                                - Q4_ZERO_BIAS * x_gcs;
                            let group_scale = unsafe { *ws.ptr().add(sg * n + row) };
                            f32_acc[lane] += corrected as f32 * group_scale * x_gs;
                        }
                    }
                }

                for lane in 0..(n_end - n_start) {
                    unsafe { *yp.ptr().add(n_start + lane) = f32_acc[lane]; }
                }
            }
            return;
        }

        for global_tile in t_start..t_end {
            let (tile, n, wd, ws, yp) = if global_tile < n_tiles_1 {
                (global_tile, n1, w1_data, w1_scales, y1_ptr)
            } else {
                (global_tile - n_tiles_1, n2, w2_data, w2_scales, y2_ptr)
            };

            let n_start = tile * TILE_N;
            let n_end = (n_start + TILE_N).min(n);
            let tile_base = tile * k_groups * TILE_N * 2;

            let mut f32_acc = [0.0f32; TILE_N];
            for sg in 0..n_sg {
                let kg_start = sg * kgs_per_sg;
                let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
                let w_offset = tile_base + kg_start * TILE_N * 2;
                let x_offset = sg * Q4_GROUP_SIZE;

                let mut acc_i32 = [0i32; TILE_N];
                unsafe {
                    if use_avx512 {
                        avx512_q4_matvec_tile(
                            x_i8_ptr.ptr().add(x_offset),
                            wd.ptr().add(w_offset),
                            sg_kgs as u64,
                            acc_i32.as_mut_ptr(),
                        );
                    } else {
                        avx2_q4_matvec_tile(
                            x_i8_ptr.ptr().add(x_offset),
                            wd.ptr().add(w_offset),
                            sg_kgs as u64,
                            acc_i32.as_mut_ptr(),
                        );
                    }
                }

                if n_end - n_start == TILE_N {
                    unsafe {
                        let scales_base = ws.ptr().add(sg * n + n_start);
                        if use_avx512 {
                            avx512_fused_dequant_tile(
                                acc_i32.as_ptr(),
                                *x_gcs_ptr.ptr().add(sg),
                                scales_base,
                                1,
                                *x_gs_ptr.ptr().add(sg),
                                f32_acc.as_mut_ptr(),
                            );
                        } else {
                            avx2_fused_dequant_tile(
                                acc_i32.as_ptr(),
                                *x_gcs_ptr.ptr().add(sg),
                                scales_base,
                                1,
                                *x_gs_ptr.ptr().add(sg),
                                f32_acc.as_mut_ptr(),
                            );
                        }
                    }
                } else {
                    for lane in 0..(n_end - n_start) {
                        let row = n_start + lane;
                        let x_gcs = unsafe { *x_gcs_ptr.ptr().add(sg) };
                        let x_gs = unsafe { *x_gs_ptr.ptr().add(sg) };
                        let corrected = acc_i32[lane_to_acc[lane]] as i64
                            - Q4_ZERO_BIAS * x_gcs;
                        let group_scale = unsafe { *ws.ptr().add(sg * n + row) };
                        f32_acc[lane] += corrected as f32 * group_scale * x_gs;
                    }
                }
            }

            for lane in 0..(n_end - n_start) {
                unsafe { *yp.ptr().add(n_start + lane) = f32_acc[lane]; }
            }
        }
    }).map_err(|e| HerbertError::Backend(format!("fused_2_matvec pool error: {}", e)))?;
    Ok(())
}

/// Fused gate+up matvec with SiLU×Gate applied in the tile epilogue.
/// Processes gate and up tiles paired: for each tile index, computes both projections,
/// applies silu(gate) * up in-register, then stores.
/// Eliminates the separate swiglu_inplace pass (saves 1 load of up + 1 load+store of gate).
pub fn fused_gate_up_swiglu_2_matvec_q4(
    x: &[f32],
    w_gate: &Q4Weight, gate: &mut [f32],
    w_up: &Q4Weight, up: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(w_gate.k, w_up.k, "fused_swiglu: w_gate.k={} != w_up.k={}", w_gate.k, w_up.k);
    debug_assert_eq!(w_gate.n, w_up.n, "fused_swiglu: w_gate.n={} != w_up.n={}", w_gate.n, w_up.n);
    let k = w_gate.k;
    let n = w_gate.n;
    debug_assert!(x.len() >= k);

    let (x_i8, x_group_scales, x_group_col_sums) = quantize_x_and_colsums(x);

    let n_tiles = n.div_ceil(TILE_N);

    let pool = global_pool();
    let k_groups = k.div_ceil(4);
    let n_sg = n_scale_groups(k);
    let kgs_per_sg = Q4_GROUP_SIZE / 4;
    let use_avx512 = has_avx512_vnni();
    let use_avx2 = !use_avx512 && has_avx2();
    let deferred = crate::experimental::deferred_accum_enabled();

    let x_i8_ptr = SendPtr::new(x_i8.as_ptr());
    let x_gcs_ptr = SendPtr::new(x_group_col_sums.as_ptr());
    let x_gs_ptr = SendPtr::new(x_group_scales.as_ptr());
    let wg_data = SendPtr::new(w_gate.data.as_ptr());
    let wg_scales = SendPtr::new(w_gate.scales.as_ptr());
    let gate_ptr = SendMutPtr::new(gate.as_mut_ptr());
    let wu_data = SendPtr::new(w_up.data.as_ptr());
    let wu_scales = SendPtr::new(w_up.scales.as_ptr());
    let _up_buf = up; // up buffer is scratch; only gate receives the activated output

    pool.parallel_for_phase_aware(n_tiles, move |_, t_start, t_end| {
        if !(use_avx512 || use_avx2) { return; }
        let lane_to_acc = if use_avx512 { &ASM_LANE_TO_ACC } else { &AVX2_LANE_TO_ACC };

        // Helper: compute one full tile for a given weight, returning f32_acc
        macro_rules! compute_tile {
            ($tile:expr, $wd:expr, $ws:expr, $f32_acc:expr) => {{
                let n_start = $tile * TILE_N;
                let n_end = (n_start + TILE_N).min(n);
                let tile_base = $tile * k_groups * TILE_N * 2;

                if deferred {
                    // Phase 1: ALL tile kernels (integer) across scale groups
                    let mut sg_accs = vec![0i32; n_sg * TILE_N];
                    for sg in 0..n_sg {
                        let kg_start = sg * kgs_per_sg;
                        let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
                        let w_offset = tile_base + kg_start * TILE_N * 2;
                        let x_offset = sg * Q4_GROUP_SIZE;
                        let acc_base = sg * TILE_N;
                        for j in 0..TILE_N { sg_accs[acc_base + j] = 0; }
                        unsafe {
                            if use_avx512 {
                                avx512_q4_matvec_tile(
                                    x_i8_ptr.ptr().add(x_offset),
                                    $wd.ptr().add(w_offset),
                                    sg_kgs as u64,
                                    sg_accs[acc_base..].as_mut_ptr(),
                                );
                            } else {
                                avx2_q4_matvec_tile(
                                    x_i8_ptr.ptr().add(x_offset),
                                    $wd.ptr().add(w_offset),
                                    sg_kgs as u64,
                                    sg_accs[acc_base..].as_mut_ptr(),
                                );
                            }
                        }
                    }
                    // Phase 2: dequant
                    for sg in 0..n_sg {
                        let acc_base = sg * TILE_N;
                        if n_end - n_start == TILE_N {
                            unsafe {
                                let scales_base = $ws.ptr().add(sg * n + n_start);
                                if use_avx512 {
                                    avx512_fused_dequant_tile(
                                        sg_accs[acc_base..].as_ptr(),
                                        *x_gcs_ptr.ptr().add(sg),
                                        scales_base, 1,
                                        *x_gs_ptr.ptr().add(sg),
                                        $f32_acc.as_mut_ptr(),
                                    );
                                } else {
                                    avx2_fused_dequant_tile(
                                        sg_accs[acc_base..].as_ptr(),
                                        *x_gcs_ptr.ptr().add(sg),
                                        scales_base, 1,
                                        *x_gs_ptr.ptr().add(sg),
                                        $f32_acc.as_mut_ptr(),
                                    );
                                }
                            }
                        } else {
                            for lane in 0..(n_end - n_start) {
                                let row = n_start + lane;
                                let x_gcs = unsafe { *x_gcs_ptr.ptr().add(sg) };
                                let x_gs = unsafe { *x_gs_ptr.ptr().add(sg) };
                                let corrected = sg_accs[acc_base + lane_to_acc[lane]] as i64
                                    - Q4_ZERO_BIAS * x_gcs;
                                let group_scale = unsafe { *$ws.ptr().add(sg * n + row) };
                                $f32_acc[lane] += corrected as f32 * group_scale * x_gs;
                            }
                        }
                    }
                } else {
                    // Standard path: interleaved tile+dequant per SG
                    let tile_base_s = tile_base;
                    for sg in 0..n_sg {
                        let kg_start = sg * kgs_per_sg;
                        let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
                        let w_offset = tile_base_s + kg_start * TILE_N * 2;
                        let x_offset = sg * Q4_GROUP_SIZE;
                        let mut acc_i32 = [0i32; TILE_N];
                        unsafe {
                            if use_avx512 {
                                avx512_q4_matvec_tile(
                                    x_i8_ptr.ptr().add(x_offset),
                                    $wd.ptr().add(w_offset),
                                    sg_kgs as u64,
                                    acc_i32.as_mut_ptr(),
                                );
                            } else {
                                avx2_q4_matvec_tile(
                                    x_i8_ptr.ptr().add(x_offset),
                                    $wd.ptr().add(w_offset),
                                    sg_kgs as u64,
                                    acc_i32.as_mut_ptr(),
                                );
                            }
                        }
                        if n_end - n_start == TILE_N {
                            unsafe {
                                let scales_base = $ws.ptr().add(sg * n + n_start);
                                if use_avx512 {
                                    avx512_fused_dequant_tile(
                                        acc_i32.as_ptr(),
                                        *x_gcs_ptr.ptr().add(sg),
                                        scales_base, 1,
                                        *x_gs_ptr.ptr().add(sg),
                                        $f32_acc.as_mut_ptr(),
                                    );
                                } else {
                                    avx2_fused_dequant_tile(
                                        acc_i32.as_ptr(),
                                        *x_gcs_ptr.ptr().add(sg),
                                        scales_base, 1,
                                        *x_gs_ptr.ptr().add(sg),
                                        $f32_acc.as_mut_ptr(),
                                    );
                                }
                            }
                        } else {
                            for lane in 0..(n_end - n_start) {
                                let row = n_start + lane;
                                let x_gcs = unsafe { *x_gcs_ptr.ptr().add(sg) };
                                let x_gs = unsafe { *x_gs_ptr.ptr().add(sg) };
                                let corrected = acc_i32[lane_to_acc[lane]] as i64
                                    - Q4_ZERO_BIAS * x_gcs;
                                let group_scale = unsafe { *$ws.ptr().add(sg * n + row) };
                                $f32_acc[lane] += corrected as f32 * group_scale * x_gs;
                            }
                        }
                    }
                }
                (n_start, n_end)
            }}
        }

        for tile in t_start..t_end {
            // Compute gate tile
            let mut gate_acc = [0.0f32; TILE_N];
            let (n_start, n_end) = compute_tile!(tile, wg_data, wg_scales, gate_acc);

            // Compute up tile
            let mut up_acc = [0.0f32; TILE_N];
            let _ = compute_tile!(tile, wu_data, wu_scales, up_acc);

            // Apply SiLU×Gate in-register: gate_acc = silu(gate_acc) * up_acc
            if n_end - n_start == TILE_N {
                unsafe {
                    if use_avx512 {
                        avx512_silu_gate_tile(gate_acc.as_mut_ptr(), up_acc.as_ptr());
                    } else {
                        avx2_silu_gate_tile(gate_acc.as_mut_ptr(), up_acc.as_ptr());
                    }
                }
            } else {
                // Scalar fallback for partial tile
                for lane in 0..(n_end - n_start) {
                    let g = gate_acc[lane];
                    let sig = 1.0 / (1.0 + (-g).exp());
                    gate_acc[lane] = g * sig * up_acc[lane];
                }
            }

            // Store results
            for lane in 0..(n_end - n_start) {
                unsafe {
                    *gate_ptr.ptr().add(n_start + lane) = gate_acc[lane];
                    // up buffer can be left dirty — caller only uses gate after this
                }
            }
        }
    }).map_err(|e| HerbertError::Backend(format!("fused_swiglu_2_matvec pool error: {}", e)))?;
    Ok(())
}

/// Fused 3-projection matvec: quantizes x once, SINGLE thread pool dispatch for all three projections.
/// All weights must share the same K dimension (same input).
/// Tiles from w1, w2, w3 are combined into one parallel_for.
pub fn fused_3_matvec_q4(
    x: &[f32],
    w1: &Q4Weight, y1: &mut [f32],
    w2: &Q4Weight, y2: &mut [f32],
    w3: &Q4Weight, y3: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(w1.k, w2.k);
    debug_assert_eq!(w1.k, w3.k);
    let k = w1.k;
    debug_assert!(x.len() >= k);

    let (x_i8, x_group_scales, x_group_col_sums) = quantize_x_and_colsums(x);

    let n1 = w1.n;
    let n2 = w2.n;
    let n3 = w3.n;
    let nt1 = n1.div_ceil(TILE_N);
    let nt2 = n2.div_ceil(TILE_N);
    let nt3 = n3.div_ceil(TILE_N);
    let total_tiles = nt1 + nt2 + nt3;
    let split12 = nt1 + nt2;

    let pool = global_pool();
    let k_groups = k.div_ceil(4);
    let n_sg = n_scale_groups(k);
    let kgs_per_sg = Q4_GROUP_SIZE / 4;
    let use_avx512 = has_avx512_vnni();
    let use_avx2 = !use_avx512 && has_avx2();
    let deferred = crate::experimental::deferred_accum_enabled();


    let x_i8_ptr = SendPtr::new(x_i8.as_ptr());
    let x_gcs_ptr = SendPtr::new(x_group_col_sums.as_ptr());
    let x_gs_ptr = SendPtr::new(x_group_scales.as_ptr());
    let w1_data = SendPtr::new(w1.data.as_ptr());
    let w1_scales = SendPtr::new(w1.scales.as_ptr());
    let y1_ptr = SendMutPtr::new(y1.as_mut_ptr());
    let w2_data = SendPtr::new(w2.data.as_ptr());
    let w2_scales = SendPtr::new(w2.scales.as_ptr());
    let y2_ptr = SendMutPtr::new(y2.as_mut_ptr());
    let w3_data = SendPtr::new(w3.data.as_ptr());
    let w3_scales = SendPtr::new(w3.scales.as_ptr());
    let y3_ptr = SendMutPtr::new(y3.as_mut_ptr());

    if use_avx512
        && crate::experimental::cpu_experimental_enabled()
        && use_fused3_3way_avx512()
        && nt1 >= nt2
        && nt2 == nt3
        && nt2 > 0
    {
        let shared_nt = nt2;
        pool.parallel_for_phase_aware(shared_nt, move |_, t_start, t_end| {
            let lane_to_acc = &ASM_LANE_TO_ACC;

            for tile in t_start..t_end {
                let n_start = tile * TILE_N;
                let q_end = (n_start + TILE_N).min(n1);
                let k_end = (n_start + TILE_N).min(n2);
                let v_end = (n_start + TILE_N).min(n3);
                let tile_base = tile * k_groups * TILE_N * 2;

                let mut q_f32_acc = [0.0f32; TILE_N];
                let mut k_f32_acc = [0.0f32; TILE_N];
                let mut v_f32_acc = [0.0f32; TILE_N];

                for sg in 0..n_sg {
                    let kg_start = sg * kgs_per_sg;
                    let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
                    let w_offset = tile_base + kg_start * TILE_N * 2;
                    let x_offset = sg * Q4_GROUP_SIZE;

                    unsafe {
                        let x_ptr = x_i8_ptr.ptr().add(x_offset);
                        let x_gcs = *x_gcs_ptr.ptr().add(sg);
                        let x_gs = *x_gs_ptr.ptr().add(sg);

                        if q_end - n_start == TILE_N && k_end - n_start == TILE_N && v_end - n_start == TILE_N {
                            let mut acc_i32 = [0i32; TILE_N * 3];
                            avx512_q4_matvec_tile_3way(
                                x_ptr,
                                w1_data.ptr().add(w_offset),
                                w2_data.ptr().add(w_offset),
                                w3_data.ptr().add(w_offset),
                                sg_kgs as u64,
                                acc_i32.as_mut_ptr(),
                            );
                            avx512_fused_dequant_tile(
                                acc_i32.as_ptr(),
                                x_gcs,
                                w1_scales.ptr().add(sg * n1 + n_start),
                                1,
                                x_gs,
                                q_f32_acc.as_mut_ptr(),
                            );
                            avx512_fused_dequant_tile(
                                acc_i32.as_ptr().add(TILE_N),
                                x_gcs,
                                w2_scales.ptr().add(sg * n2 + n_start),
                                1,
                                x_gs,
                                k_f32_acc.as_mut_ptr(),
                            );
                            avx512_fused_dequant_tile(
                                acc_i32.as_ptr().add(TILE_N * 2),
                                x_gcs,
                                w3_scales.ptr().add(sg * n3 + n_start),
                                1,
                                x_gs,
                                v_f32_acc.as_mut_ptr(),
                            );
                        } else {
                            let mut q_acc_i32 = [0i32; TILE_N];
                            let mut k_acc_i32 = [0i32; TILE_N];
                            let mut v_acc_i32 = [0i32; TILE_N];

                            avx512_q4_matvec_tile(
                                x_ptr,
                                w1_data.ptr().add(w_offset),
                                sg_kgs as u64,
                                q_acc_i32.as_mut_ptr(),
                            );
                            avx512_q4_matvec_tile(
                                x_ptr,
                                w2_data.ptr().add(w_offset),
                                sg_kgs as u64,
                                k_acc_i32.as_mut_ptr(),
                            );
                            avx512_q4_matvec_tile(
                                x_ptr,
                                w3_data.ptr().add(w_offset),
                                sg_kgs as u64,
                                v_acc_i32.as_mut_ptr(),
                            );

                            if q_end - n_start == TILE_N {
                                avx512_fused_dequant_tile(
                                    q_acc_i32.as_ptr(),
                                    x_gcs,
                                    w1_scales.ptr().add(sg * n1 + n_start),
                                    1,
                                    x_gs,
                                    q_f32_acc.as_mut_ptr(),
                                );
                            } else {
                                for lane in 0..(q_end - n_start) {
                                    let row = n_start + lane;
                                    let corrected =
                                        q_acc_i32[lane_to_acc[lane]] as i64 - Q4_ZERO_BIAS * x_gcs;
                                    let group_scale = *w1_scales.ptr().add(sg * n1 + row);
                                    q_f32_acc[lane] += corrected as f32 * group_scale * x_gs;
                                }
                            }

                            if k_end - n_start == TILE_N {
                                avx512_fused_dequant_tile(
                                    k_acc_i32.as_ptr(),
                                    x_gcs,
                                    w2_scales.ptr().add(sg * n2 + n_start),
                                    1,
                                    x_gs,
                                    k_f32_acc.as_mut_ptr(),
                                );
                            } else {
                                for lane in 0..(k_end - n_start) {
                                    let row = n_start + lane;
                                    let corrected =
                                        k_acc_i32[lane_to_acc[lane]] as i64 - Q4_ZERO_BIAS * x_gcs;
                                    let group_scale = *w2_scales.ptr().add(sg * n2 + row);
                                    k_f32_acc[lane] += corrected as f32 * group_scale * x_gs;
                                }
                            }

                            if v_end - n_start == TILE_N {
                                avx512_fused_dequant_tile(
                                    v_acc_i32.as_ptr(),
                                    x_gcs,
                                    w3_scales.ptr().add(sg * n3 + n_start),
                                    1,
                                    x_gs,
                                    v_f32_acc.as_mut_ptr(),
                                );
                            } else {
                                for lane in 0..(v_end - n_start) {
                                    let row = n_start + lane;
                                    let corrected =
                                        v_acc_i32[lane_to_acc[lane]] as i64 - Q4_ZERO_BIAS * x_gcs;
                                    let group_scale = *w3_scales.ptr().add(sg * n3 + row);
                                    v_f32_acc[lane] += corrected as f32 * group_scale * x_gs;
                                }
                            }
                        }
                    }
                }

                for lane in 0..(q_end - n_start) {
                    unsafe { *y1_ptr.ptr().add(n_start + lane) = q_f32_acc[lane]; }
                }
                for lane in 0..(k_end - n_start) {
                    unsafe { *y2_ptr.ptr().add(n_start + lane) = k_f32_acc[lane]; }
                }
                for lane in 0..(v_end - n_start) {
                    unsafe { *y3_ptr.ptr().add(n_start + lane) = v_f32_acc[lane]; }
                }
            }
        }).map_err(|e| HerbertError::Backend(format!("fused_3_matvec shared pool error: {}", e)))?;

        let q_tail_tiles = nt1 - shared_nt;
        if q_tail_tiles > 0 {
            pool.parallel_for_phase_aware(q_tail_tiles, move |_, t_start, t_end| {
                let lane_to_acc = &ASM_LANE_TO_ACC;

                for tile_offset in t_start..t_end {
                    let tile = shared_nt + tile_offset;
                    let n_start = tile * TILE_N;
                    let n_end = (n_start + TILE_N).min(n1);
                    let tile_base = tile * k_groups * TILE_N * 2;

                    let mut f32_acc = [0.0f32; TILE_N];
                    for sg in 0..n_sg {
                        let kg_start = sg * kgs_per_sg;
                        let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
                        let w_offset = tile_base + kg_start * TILE_N * 2;
                        let x_offset = sg * Q4_GROUP_SIZE;

                        let mut acc_i32 = [0i32; TILE_N];
                        unsafe {
                            avx512_q4_matvec_tile(
                                x_i8_ptr.ptr().add(x_offset),
                                w1_data.ptr().add(w_offset),
                                sg_kgs as u64,
                                acc_i32.as_mut_ptr(),
                            );

                            let x_gcs = *x_gcs_ptr.ptr().add(sg);
                            let x_gs = *x_gs_ptr.ptr().add(sg);
                            if n_end - n_start == TILE_N {
                                avx512_fused_dequant_tile(
                                    acc_i32.as_ptr(),
                                    x_gcs,
                                    w1_scales.ptr().add(sg * n1 + n_start),
                                    1,
                                    x_gs,
                                    f32_acc.as_mut_ptr(),
                                );
                            } else {
                                for lane in 0..(n_end - n_start) {
                                    let row = n_start + lane;
                                    let corrected =
                                        acc_i32[lane_to_acc[lane]] as i64 - Q4_ZERO_BIAS * x_gcs;
                                    let group_scale = *w1_scales.ptr().add(sg * n1 + row);
                                    f32_acc[lane] += corrected as f32 * group_scale * x_gs;
                                }
                            }
                        }
                    }

                    for lane in 0..(n_end - n_start) {
                        unsafe { *y1_ptr.ptr().add(n_start + lane) = f32_acc[lane]; }
                    }
                }
            }).map_err(|e| HerbertError::Backend(format!("fused_3_matvec q-tail pool error: {}", e)))?;
        }

        return Ok(());
    }

    pool.parallel_for_phase_aware(total_tiles, move |_, t_start, t_end| {
        if !(use_avx512 || use_avx2) { return; }
        let lane_to_acc = if use_avx512 { &ASM_LANE_TO_ACC } else { &AVX2_LANE_TO_ACC };

        if deferred {
            let mut sg_accs = vec![0i32; n_sg * TILE_N];
            for global_tile in t_start..t_end {
                let (tile, n, wd, ws, yp) = if global_tile < nt1 {
                    (global_tile, n1, w1_data, w1_scales, y1_ptr)
                } else if global_tile < split12 {
                    (global_tile - nt1, n2, w2_data, w2_scales, y2_ptr)
                } else {
                    (global_tile - split12, n3, w3_data, w3_scales, y3_ptr)
                };

                let n_start = tile * TILE_N;
                let n_end = (n_start + TILE_N).min(n);
                let tile_base = tile * k_groups * TILE_N * 2;

                // Phase 1: ALL tile kernels (integer)
                for sg in 0..n_sg {
                    let kg_start = sg * kgs_per_sg;
                    let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
                    let w_offset = tile_base + kg_start * TILE_N * 2;
                    let x_offset = sg * Q4_GROUP_SIZE;

                    let acc_base = sg * TILE_N;
                    for j in 0..TILE_N { sg_accs[acc_base + j] = 0; }
                    unsafe {
                        if use_avx512 {
                            avx512_q4_matvec_tile(
                                x_i8_ptr.ptr().add(x_offset),
                                wd.ptr().add(w_offset),
                                sg_kgs as u64,
                                sg_accs[acc_base..].as_mut_ptr(),
                            );
                        } else {
                            avx2_q4_matvec_tile(
                                x_i8_ptr.ptr().add(x_offset),
                                wd.ptr().add(w_offset),
                                sg_kgs as u64,
                                sg_accs[acc_base..].as_mut_ptr(),
                            );
                        }
                    }
                }

                // Phase 2: ALL dequant (FP)
                let mut f32_acc = [0.0f32; TILE_N];
                for sg in 0..n_sg {
                    let acc_base = sg * TILE_N;
                    if n_end - n_start == TILE_N {
                        unsafe {
                            let scales_base = ws.ptr().add(sg * n + n_start);
                            if use_avx512 {
                                avx512_fused_dequant_tile(
                                    sg_accs[acc_base..].as_ptr(),
                                    *x_gcs_ptr.ptr().add(sg),
                                    scales_base,
                                    1,
                                    *x_gs_ptr.ptr().add(sg),
                                    f32_acc.as_mut_ptr(),
                                );
                            } else {
                                avx2_fused_dequant_tile(
                                    sg_accs[acc_base..].as_ptr(),
                                    *x_gcs_ptr.ptr().add(sg),
                                    scales_base,
                                    1,
                                    *x_gs_ptr.ptr().add(sg),
                                    f32_acc.as_mut_ptr(),
                                );
                            }
                        }
                    } else {
                        for lane in 0..(n_end - n_start) {
                            let row = n_start + lane;
                            let x_gcs = unsafe { *x_gcs_ptr.ptr().add(sg) };
                            let x_gs = unsafe { *x_gs_ptr.ptr().add(sg) };
                            let corrected = sg_accs[acc_base + lane_to_acc[lane]] as i64
                                - Q4_ZERO_BIAS * x_gcs;
                            let group_scale = unsafe { *ws.ptr().add(sg * n + row) };
                            f32_acc[lane] += corrected as f32 * group_scale * x_gs;
                        }
                    }
                }

                for lane in 0..(n_end - n_start) {
                    unsafe { *yp.ptr().add(n_start + lane) = f32_acc[lane]; }
                }
            }
            return;
        }

        for global_tile in t_start..t_end {
            let (tile, n, wd, ws, yp) = if global_tile < nt1 {
                (global_tile, n1, w1_data, w1_scales, y1_ptr)
            } else if global_tile < split12 {
                (global_tile - nt1, n2, w2_data, w2_scales, y2_ptr)
            } else {
                (global_tile - split12, n3, w3_data, w3_scales, y3_ptr)
            };

            let n_start = tile * TILE_N;
            let n_end = (n_start + TILE_N).min(n);
            let tile_base = tile * k_groups * TILE_N * 2;

            let mut f32_acc = [0.0f32; TILE_N];
            for sg in 0..n_sg {
                let kg_start = sg * kgs_per_sg;
                let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
                let w_offset = tile_base + kg_start * TILE_N * 2;
                let x_offset = sg * Q4_GROUP_SIZE;

                let mut acc_i32 = [0i32; TILE_N];
                unsafe {
                    if use_avx512 {
                        avx512_q4_matvec_tile(
                            x_i8_ptr.ptr().add(x_offset),
                            wd.ptr().add(w_offset),
                            sg_kgs as u64,
                            acc_i32.as_mut_ptr(),
                        );
                    } else {
                        avx2_q4_matvec_tile(
                            x_i8_ptr.ptr().add(x_offset),
                            wd.ptr().add(w_offset),
                            sg_kgs as u64,
                            acc_i32.as_mut_ptr(),
                        );
                    }
                }

                if n_end - n_start == TILE_N {
                    unsafe {
                        let scales_base = ws.ptr().add(sg * n + n_start);
                        if use_avx512 {
                            avx512_fused_dequant_tile(
                                acc_i32.as_ptr(),
                                *x_gcs_ptr.ptr().add(sg),
                                scales_base,
                                1,
                                *x_gs_ptr.ptr().add(sg),
                                f32_acc.as_mut_ptr(),
                            );
                        } else {
                            avx2_fused_dequant_tile(
                                acc_i32.as_ptr(),
                                *x_gcs_ptr.ptr().add(sg),
                                scales_base,
                                1,
                                *x_gs_ptr.ptr().add(sg),
                                f32_acc.as_mut_ptr(),
                            );
                        }
                    }
                } else {
                    for lane in 0..(n_end - n_start) {
                        let row = n_start + lane;
                        let x_gcs = unsafe { *x_gcs_ptr.ptr().add(sg) };
                        let x_gs = unsafe { *x_gs_ptr.ptr().add(sg) };
                        let corrected = acc_i32[lane_to_acc[lane]] as i64
                            - Q4_ZERO_BIAS * x_gcs;
                        let group_scale = unsafe { *ws.ptr().add(sg * n + row) };
                        f32_acc[lane] += corrected as f32 * group_scale * x_gs;
                    }
                }
            }

            for lane in 0..(n_end - n_start) {
                unsafe { *yp.ptr().add(n_start + lane) = f32_acc[lane]; }
            }
        }
    }).map_err(|e| HerbertError::Backend(format!("fused_3_matvec pool error: {}", e)))?;
    Ok(())
}

// ============================================================================
// Microbenchmarks: rdtsc dequant loop Q4
// ============================================================================

#[cfg(all(test, target_arch = "x86_64"))]
mod bench_dequant {
    use super::*;
    use std::arch::x86_64::{__rdtscp, _mm_lfence, _rdtsc};
    use std::time::Instant;

    // ── xorshift64 RNG (deterministic, no deps) ─────────────────────

    struct Xorshift64(u64);

    impl Xorshift64 {
        fn new(seed: u64) -> Self {
            Self(if seed == 0 { 0xdeadbeef } else { seed })
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
            let bits = (self.next_u64() & 0xFFFFFFFF) as u32;
            lo + (bits as f32 / u32::MAX as f32) * (hi - lo)
        }
        fn next_i32(&mut self, lo: i32, hi: i32) -> i32 {
            let range = (hi - lo + 1) as u64;
            lo + (self.next_u64() % range) as i32
        }
    }

    // ── rdtsc helpers ───────────────────────────────────────────────

    #[inline(always)]
    unsafe fn tsc_start() -> u64 {
        _mm_lfence();
        _rdtsc()
    }

    #[inline(always)]
    unsafe fn tsc_stop() -> u64 {
        let mut aux: u32 = 0;
        let t = __rdtscp(&mut aux);
        _mm_lfence();
        t
    }

    fn calibrate_tsc() -> (f64, u64) {
        // Measure TSC frequency via sleep
        let n_cal = 5;
        let mut freq_samples = Vec::with_capacity(n_cal);
        for _ in 0..n_cal {
            let t0_inst = Instant::now();
            let t0_tsc = unsafe { tsc_start() };
            std::thread::sleep(std::time::Duration::from_millis(50));
            let t1_tsc = unsafe { tsc_stop() };
            let elapsed_ns = t0_inst.elapsed().as_nanos() as f64;
            let elapsed_cycles = t1_tsc.wrapping_sub(t0_tsc) as f64;
            freq_samples.push(elapsed_cycles / elapsed_ns);
        }
        freq_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let tsc_ghz = freq_samples[n_cal / 2];

        // Measure rdtsc overhead
        let mut overhead_samples = Vec::with_capacity(1000);
        for _ in 0..1000 {
            unsafe {
                let t0 = tsc_start();
                let t1 = tsc_stop();
                overhead_samples.push(t1.wrapping_sub(t0));
            }
        }
        overhead_samples.sort();
        let overhead = overhead_samples[overhead_samples.len() / 2];

        (tsc_ghz, overhead)
    }

    fn median(v: &mut [u64]) -> u64 {
        v.sort();
        v[v.len() / 2]
    }

    fn percentile(v: &mut [u64], p: f64) -> u64 {
        v.sort();
        let idx = ((v.len() as f64 - 1.0) * p / 100.0).round() as usize;
        v[idx.min(v.len() - 1)]
    }

    // ── Synthetic data generators ───────────────────────────────────

    fn make_synthetic_weight(n: usize, k: usize, rng: &mut Xorshift64) -> Q4Weight {
        let n_tiles = n.div_ceil(TILE_N);
        let k_groups = k.div_ceil(4);
        let n_sg = n_scale_groups(k);

        // Random packed nibbles
        let data_len = n_tiles * k_groups * TILE_N * 2;
        let data: Vec<u8> = (0..data_len).map(|_| rng.next_u64() as u8).collect();

        // Realistic scales [0.001, 0.1]
        let scales: Vec<f32> = (0..n * n_sg)
            .map(|_| rng.next_f32(0.001, 0.1))
            .collect();

        use herbert_backend_common::hugepages::HugeVec;
        let mut w = Q4Weight {
            data: HugeVec::from_vec_no_huge(data),
            scales: HugeVec::from_vec_no_huge(scales),
            scales_tiled: HugeVec::from_vec_no_huge(vec![]),
            n, k,
        };
        w.compute_scales_tiled();
        w
    }

    fn make_synthetic_x(k: usize, rng: &mut Xorshift64) -> Vec<f32> {
        (0..k).map(|_| rng.next_f32(-2.0, 2.0)).collect()
    }

    // ── Bench 1: dequant loop only ──────────────────────────────────

    #[test]
    fn bench_dequant_loop_only() {
        let k: usize = 2048;
        let n: usize = 768;
        let n_sg = n_scale_groups(k);
        let n_tiles = n.div_ceil(TILE_N);

        let (tsc_ghz, overhead) = calibrate_tsc();
        println!("\n=== rdtsc calibration ===");
        println!("  TSC freq: {:.3} GHz    overhead: {} cycles", tsc_ghz, overhead);

        // Generate synthetic data
        let mut rng = Xorshift64::new(42);

        // acc_i32 for each tile iteration (we cycle through)
        let acc_data: Vec<[i32; TILE_N]> = (0..n_tiles * n_sg)
            .map(|_| {
                let mut acc = [0i32; TILE_N];
                for a in acc.iter_mut() {
                    *a = rng.next_i32(-50000, 50000);
                }
                acc
            })
            .collect();

        let w = make_synthetic_weight(n, k, &mut rng);
        let x = make_synthetic_x(k, &mut rng);
        let (x_i8, x_group_scales, x_group_col_sums) = quantize_x_and_colsums(&x);

        let warmup = 10_000;
        let batches = 1_000;
        let iters_per_batch = 1_000;

        // Warmup: run the dequant loop to warm caches
        let mut sink = 0.0f32;
        for w_iter in 0..warmup {
            let tile = w_iter % n_tiles;
            let sg = w_iter % n_sg;
            let n_start = tile * TILE_N;
            let n_end = (n_start + TILE_N).min(n);
            let acc_idx = tile * n_sg + sg;
            let acc_i32 = &acc_data[acc_idx % acc_data.len()];
            let mut f32_acc = [0.0f32; TILE_N];

            for lane in 0..(n_end - n_start) {
                let row = n_start + lane;
                let corrected = acc_i32[ASM_LANE_TO_ACC[lane]] as i64
                    - Q4_ZERO_BIAS * x_group_col_sums[sg];
                let group_scale = w.scales[sg * w.n + row];
                f32_acc[lane] += corrected as f32 * group_scale * x_group_scales[sg];
            }
            sink += f32_acc[0];
        }
        std::hint::black_box(sink);

        // Timed batches
        let mut batch_medians = Vec::with_capacity(batches);
        let mut f32_acc = [0.0f32; TILE_N]; // persistent across iters: += is a real dependency
        for batch in 0..batches {
            let mut samples = Vec::with_capacity(iters_per_batch);
            for i in 0..iters_per_batch {
                let iter_idx = batch * iters_per_batch + i;
                let tile = iter_idx % n_tiles;
                let sg = iter_idx % n_sg;
                let n_start = tile * TILE_N;
                let n_end = (n_start + TILE_N).min(n);
                let acc_idx = tile * n_sg + sg;
                let acc_i32 = &acc_data[acc_idx % acc_data.len()];

                let t0 = unsafe { tsc_start() };
                for lane in 0..(n_end - n_start) {
                    let row = n_start + lane;
                    let corrected = acc_i32[ASM_LANE_TO_ACC[lane]] as i64
                        - Q4_ZERO_BIAS * x_group_col_sums[sg];
                    let group_scale = w.scales[sg * w.n + row];
                    f32_acc[lane] += corrected as f32 * group_scale * x_group_scales[sg];
                }
                let t1 = unsafe { tsc_stop() };

                let elapsed = t1.wrapping_sub(t0).saturating_sub(overhead);
                samples.push(elapsed);
            }
            batch_medians.push(median(&mut samples));
        }
        std::hint::black_box(&f32_acc);

        let med = median(&mut batch_medians.clone());
        let p5 = percentile(&mut batch_medians.clone(), 5.0);
        let p95 = percentile(&mut batch_medians, 95.0);
        let ns_per_iter = med as f64 / tsc_ghz;

        println!("\n=== Dequant loop only (32 lanes, K={}, N={}, GS={}) ===", k, n, Q4_GROUP_SIZE);
        println!("  {} batches x {} iters (+ {} warmup)", batches, iters_per_batch, warmup);
        println!("  median: {:.1} c/iter   p5: {:.1}   p95: {:.1}", med as f64, p5 as f64, p95 as f64);
        println!("  -> {:.1} ns/iter @ {:.3} GHz", ns_per_iter, tsc_ghz);
    }

    // ── Bench 2: scale-group iteration (ASM + dequant) ──────────────

    #[test]
    fn bench_scale_group_iter() {
        if !has_avx512_vnni() {
            println!("\n=== Scale-group iteration: SKIPPED (no AVX-512 VNNI) ===");
            return;
        }

        let k: usize = 2048;
        let n: usize = 768;
        let n_sg = n_scale_groups(k);
        let n_tiles = n.div_ceil(TILE_N);
        let k_groups = k.div_ceil(4);
        let kgs_per_sg = Q4_GROUP_SIZE / 4;

        let (tsc_ghz, overhead) = calibrate_tsc();

        let mut rng = Xorshift64::new(123);
        let w = make_synthetic_weight(n, k, &mut rng);
        let x = make_synthetic_x(k, &mut rng);
        let (x_i8, x_group_scales, x_group_col_sums) = quantize_x_and_colsums(&x);

        let warmup = 5_000;
        let batches = 1_000;
        let iters_per_batch = 100;

        // Warmup
        let mut sink = 0.0f32;
        for w_iter in 0..warmup {
            let tile = w_iter % n_tiles;
            let sg = w_iter % n_sg;
            let n_start = tile * TILE_N;
            let n_end = (n_start + TILE_N).min(n);
            let tile_base = tile * k_groups * TILE_N * 2;
            let kg_start = sg * kgs_per_sg;
            let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
            let w_offset = tile_base + kg_start * TILE_N * 2;
            let x_offset = sg * Q4_GROUP_SIZE;

            let mut acc_i32 = [0i32; TILE_N];
            unsafe {
                avx512_q4_matvec_tile(
                    x_i8.as_ptr().add(x_offset),
                    w.data.as_ptr().add(w_offset),
                    sg_kgs as u64,
                    acc_i32.as_mut_ptr(),
                );
            }

            let mut f32_acc = [0.0f32; TILE_N];
            for lane in 0..(n_end - n_start) {
                let row = n_start + lane;
                let corrected = acc_i32[ASM_LANE_TO_ACC[lane]] as i64
                    - Q4_ZERO_BIAS * x_group_col_sums[sg];
                let group_scale = w.scales[sg * w.n + row];
                f32_acc[lane] += corrected as f32 * group_scale * x_group_scales[sg];
            }
            sink += f32_acc[0];
        }
        std::hint::black_box(sink);

        // Timed batches
        let mut batch_medians = Vec::with_capacity(batches);
        let mut f32_acc = [0.0f32; TILE_N]; // persistent: += is a real dependency
        for batch in 0..batches {
            let mut samples = Vec::with_capacity(iters_per_batch);
            for i in 0..iters_per_batch {
                let iter_idx = batch * iters_per_batch + i;
                let tile = iter_idx % n_tiles;
                let sg = iter_idx % n_sg;
                let n_start = tile * TILE_N;
                let n_end = (n_start + TILE_N).min(n);
                let tile_base = tile * k_groups * TILE_N * 2;
                let kg_start = sg * kgs_per_sg;
                let sg_kgs = ((sg + 1) * kgs_per_sg).min(k_groups) - kg_start;
                let w_offset = tile_base + kg_start * TILE_N * 2;
                let x_offset = sg * Q4_GROUP_SIZE;

                let t0 = unsafe { tsc_start() };

                let mut acc_i32 = [0i32; TILE_N];
                unsafe {
                    avx512_q4_matvec_tile(
                        x_i8.as_ptr().add(x_offset),
                        w.data.as_ptr().add(w_offset),
                        sg_kgs as u64,
                        acc_i32.as_mut_ptr(),
                    );
                }

                for lane in 0..(n_end - n_start) {
                    let row = n_start + lane;
                    let corrected = acc_i32[ASM_LANE_TO_ACC[lane]] as i64
                        - Q4_ZERO_BIAS * x_group_col_sums[sg];
                    let group_scale = w.scales[sg * w.n + row];
                    f32_acc[lane] += corrected as f32 * group_scale * x_group_scales[sg];
                }

                let t1 = unsafe { tsc_stop() };

                let elapsed = t1.wrapping_sub(t0).saturating_sub(overhead);
                samples.push(elapsed);
            }
            batch_medians.push(median(&mut samples));
        }
        std::hint::black_box(&f32_acc);

        let med = median(&mut batch_medians.clone());
        let p5 = percentile(&mut batch_medians.clone(), 5.0);
        let p95 = percentile(&mut batch_medians, 95.0);

        println!("\n=== Scale-group iteration (ASM + dequant, K={}, N={}, GS={}) ===", k, n, Q4_GROUP_SIZE);
        println!("  {} batches x {} iters (+ {} warmup)", batches, iters_per_batch, warmup);
        println!("  median: {:.1} c/iter   p5: {:.1}   p95: {:.1}", med as f64, p5 as f64, p95 as f64);
        println!("  -> {:.1} ns/iter @ {:.3} GHz", med as f64 / tsc_ghz, tsc_ghz);
    }

    // ── Bench 3: full matvec_q4_single e2e ──────────────────────────

    #[test]
    fn bench_matvec_e2e() {
        if !has_avx512_vnni() {
            println!("\n=== Full matvec_q4_single: SKIPPED (no AVX-512 VNNI) ===");
            return;
        }

        let k: usize = 2048;
        let n: usize = 768;
        let n_sg = n_scale_groups(k);

        let (tsc_ghz, overhead) = calibrate_tsc();

        let mut rng = Xorshift64::new(777);
        let w = make_synthetic_weight(n, k, &mut rng);
        let x = make_synthetic_x(k, &mut rng);
        let mut y = vec![0.0f32; n];

        let warmup = 1_000;
        let batches = 100;
        let iters_per_batch = 100;

        // Warmup
        for _ in 0..warmup {
            matvec_q4_single(&x, &w, &mut y);
            std::hint::black_box(&y);
        }

        // Timed batches (rdtsc)
        let mut batch_medians = Vec::with_capacity(batches);
        // Also collect Instant timings for cross-validation
        let mut instant_total = std::time::Duration::ZERO;
        let mut instant_count: u64 = 0;

        for _ in 0..batches {
            let mut samples = Vec::with_capacity(iters_per_batch);
            let inst_start = Instant::now();
            for _ in 0..iters_per_batch {
                let t0 = unsafe { tsc_start() };
                matvec_q4_single(&x, &w, &mut y);
                let t1 = unsafe { tsc_stop() };
                std::hint::black_box(&y);
                let elapsed = t1.wrapping_sub(t0).saturating_sub(overhead);
                samples.push(elapsed);
            }
            instant_total += inst_start.elapsed();
            instant_count += iters_per_batch as u64;
            batch_medians.push(median(&mut samples));
        }

        let med = median(&mut batch_medians.clone());
        let p5 = percentile(&mut batch_medians.clone(), 5.0);
        let p95 = percentile(&mut batch_medians, 95.0);
        let us_per_iter = med as f64 / tsc_ghz / 1000.0;
        let instant_us = instant_total.as_nanos() as f64 / instant_count as f64 / 1000.0;

        let n_tiles = n.div_ceil(TILE_N);
        let total_sg_iters = n_tiles * n_sg;
        let per_sg = med as f64 / total_sg_iters as f64;

        println!("\n=== Full matvec_q4_single [{}->{}] ===", k, n);
        println!("  {} batches x {} iters (+ {} warmup)", batches, iters_per_batch, warmup);
        println!("  median: {:.0} c/iter = {:.1} us   p5: {:.0}   p95: {:.0}",
                 med as f64, us_per_iter, p5 as f64, p95 as f64);
        println!("  Instant cross-check: {:.1} us", instant_us);
        println!("  per-sg-iter: {:.0} cycles (total / {} sg-iters)", per_sg, total_sg_iters);

        // Cross-validation: rdtsc vs Instant should agree within 10%
        let rdtsc_us = us_per_iter;
        let ratio = rdtsc_us / instant_us;
        println!("  rdtsc/Instant ratio: {:.3} (expect ~1.0)", ratio);
    }

    // ── Test: deferred accumulation numerical equivalence ────────────

    #[test]
    fn test_deferred_matches_interleaved() {
        if !has_avx512_vnni() && !has_avx2() {
            println!("SKIPPED: no AVX-512 VNNI or AVX2");
            return;
        }

        let configs = &[
            (768, 2048),   // small
            (3072, 4096),  // q_proj dims
            (9216, 3072),  // down_proj dims
            (3072, 100),   // partial last tile (100 % 32 = 4)
        ];

        for &(k, n) in configs {
            let mut rng = Xorshift64::new(42 + k as u64);
            let w = make_synthetic_weight(n, k, &mut rng);
            let x = make_synthetic_x(k, &mut rng);
            let (x_i8, x_group_scales, x_group_col_sums) = quantize_x_and_colsums(&x);

            let mut y_interleaved = vec![0.0f32; n];
            let mut y_deferred = vec![0.0f32; n];

            if has_avx512_vnni() {
                matvec_q4_tiles_asm(&x_i8, &x_group_scales, &x_group_col_sums, &w, &mut y_interleaved);
                matvec_q4_tiles_asm_deferred(&x_i8, &x_group_scales, &x_group_col_sums, &w, &mut y_deferred);
            } else {
                matvec_q4_tiles_avx2(&x_i8, &x_group_scales, &x_group_col_sums, &w, &mut y_interleaved);
                matvec_q4_tiles_avx2_deferred(&x_i8, &x_group_scales, &x_group_col_sums, &w, &mut y_deferred);
            }

            for i in 0..n {
                assert!(
                    (y_interleaved[i] - y_deferred[i]).abs() < 1e-6 * y_interleaved[i].abs().max(1.0),
                    "K={k} N={n} mismatch at [{i}]: interleaved={} deferred={}",
                    y_interleaved[i], y_deferred[i]
                );
            }
            println!("  K={k} N={n}: deferred matches interleaved (max_diff=0)");
        }
    }
}
