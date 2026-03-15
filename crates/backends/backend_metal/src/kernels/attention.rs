//! Attention and KV cache kernel dispatch wrappers.

use std::sync::OnceLock;

use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::context::MetalContext;
use crate::memory::MetalBuffer;
use super::{dispatch_kernel, dispatch_kernel_with_tgmem, div_ceil};

/// FlashDecoding tile size (positions per threadgroup).
const FLASH_TILE_SIZE: u32 = 256;

fn gqa_decode_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("METAL_DISABLE_GQA_DECODE")
            .map(|v| v != "1")
            .unwrap_or(true)
    })
}

/// Selected prefill attention variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefillVariant {
    V1,  // baseline: 1 threadgroup per (head, position), 32 threads
    V2,  // GQA: share K/V across query heads via threadgroup memory
    V3,  // K-tiled GQA: BK=16 K-token tiles for better coalescing
    V4,  // simdgroup_matrix: BQ=8 query tiles + hardware matrix Q·K^T
    V5,  // half KV: v4 with half-precision KV cache (now identical to v4)
    V6,  // Metal 4: cooperative_tensor 32×32 for Q·K^T (Apple11+)
    V7,  // Metal 4: MPP matmul2d 32×32 for Q·K^T (per-head dispatch, BQ=BK=32)
}

fn prefill_variant() -> PrefillVariant {
    static VARIANT: OnceLock<PrefillVariant> = OnceLock::new();
    *VARIANT.get_or_init(|| {
        match std::env::var("METAL_ATTN_PREFILL").as_deref() {
            Ok("v1") => PrefillVariant::V1,
            Ok("v2") => PrefillVariant::V2,
            Ok("v3") => PrefillVariant::V3,
            Ok("v4") => PrefillVariant::V4,
            Ok("v5") => PrefillVariant::V5,
            Ok("v6") => PrefillVariant::V6,
            Ok("v7") => PrefillVariant::V7,
            _ => PrefillVariant::V4, // auto: best general variant (v6/v7 auto-selected when available)
        }
    })
}

/// Decode attention: single query token against cached K/V.
///
/// Uses FlashDecoding (tile-parallel) when cached_len > FLASH_TILE_SIZE,
/// otherwise falls back to single-pass GQA/non-GQA decoder.
///
/// KV cache is half-precision; decode Q/output are f32.
pub fn attention_decode(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q: &MetalBuffer,
    k_cache: &MetalBuffer,
    v_cache: &MetalBuffer,
    output: &MetalBuffer,
    flash_partials: &MetalBuffer,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    cached_len: u32,
    scale: f32,
) {
    let heads_per_kv = if num_kv_heads > 0 { num_heads / num_kv_heads } else { 1 };
    let num_tiles = div_ceil(cached_len, FLASH_TILE_SIZE);

    if num_tiles > 1 {
        // FlashDecoding: tile-parallel over sequence dimension.
        attention_decode_flash(
            ctx, encoder, q, k_cache, v_cache, output, flash_partials,
            num_heads, num_kv_heads, head_dim, kv_dim, cached_len, scale,
            heads_per_kv, num_tiles,
        );
    } else {
        // Single-pass: short sequence, no tiling overhead.
        attention_decode_single_pass(
            ctx, encoder, q, k_cache, v_cache, output,
            num_heads, num_kv_heads, head_dim, kv_dim, cached_len, scale,
            heads_per_kv,
        );
    }
}

/// Single-pass decode attention (original path, now with half KV cache).
fn attention_decode_single_pass(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q: &MetalBuffer,
    k_cache: &MetalBuffer,
    v_cache: &MetalBuffer,
    output: &MetalBuffer,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    cached_len: u32,
    scale: f32,
    heads_per_kv: u32,
) {
    let mut push = [0u8; 24];
    push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    push[4..8].copy_from_slice(&num_kv_heads.to_le_bytes());
    push[8..12].copy_from_slice(&head_dim.to_le_bytes());
    push[12..16].copy_from_slice(&kv_dim.to_le_bytes());
    push[16..20].copy_from_slice(&cached_len.to_le_bytes());
    push[20..24].copy_from_slice(&scale.to_le_bytes());

    let use_gqa = gqa_decode_enabled()
        && num_kv_heads > 0
        && num_heads.is_multiple_of(num_kv_heads)
        && heads_per_kv > 1
        && heads_per_kv <= 8
        && head_dim <= 256;

    if use_gqa {
        dispatch_kernel_with_tgmem(
            encoder,
            &ctx.pipelines.attention_decode_gqa,
            &[q, k_cache, v_cache, output],
            &push,
            (head_dim as usize * 2) * std::mem::size_of::<f32>(),
            MTLSize { width: num_kv_heads as usize, height: 1, depth: 1 },
            MTLSize { width: (heads_per_kv * 32) as usize, height: 1, depth: 1 },
        );
    } else {
        dispatch_kernel(
            encoder,
            &ctx.pipelines.attention_decode,
            &[q, k_cache, v_cache, output],
            &push,
            MTLSize { width: num_heads as usize, height: 1, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );
    }
}

/// FlashDecoding: tile kernel + reduce kernel.
fn attention_decode_flash(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q: &MetalBuffer,
    k_cache: &MetalBuffer,
    v_cache: &MetalBuffer,
    output: &MetalBuffer,
    flash_partials: &MetalBuffer,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    cached_len: u32,
    scale: f32,
    heads_per_kv: u32,
    num_tiles: u32,
) {
    // Phase 1: Tile kernel — one TG per (kv_head, tile).
    let mut tile_push = [0u8; 24];
    tile_push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    tile_push[4..8].copy_from_slice(&num_kv_heads.to_le_bytes());
    tile_push[8..12].copy_from_slice(&head_dim.to_le_bytes());
    tile_push[12..16].copy_from_slice(&kv_dim.to_le_bytes());
    tile_push[16..20].copy_from_slice(&cached_len.to_le_bytes());
    tile_push[20..24].copy_from_slice(&scale.to_le_bytes());

    // Prefer v2 (double-buffered) when available, unless METAL_DECODE_V1=1
    let use_v2 = !std::env::var("METAL_DECODE_V1").is_ok_and(|v| v == "1")
        && ctx.pipelines.attention_decode_flash_tile_v2.is_some();

    if use_v2 {
        // v2: half TG memory, double-buffered: 4 × head_dim × sizeof(half)
        let tgmem = (head_dim as usize * 4) * std::mem::size_of::<u16>();
        dispatch_kernel_with_tgmem(
            encoder,
            ctx.pipelines.attention_decode_flash_tile_v2.as_ref().unwrap(),
            &[q, k_cache, v_cache, flash_partials],
            &tile_push,
            tgmem,
            MTLSize { width: num_kv_heads as usize, height: num_tiles as usize, depth: 1 },
            MTLSize { width: (heads_per_kv * 32) as usize, height: 1, depth: 1 },
        );
    } else {
        let tgmem = (head_dim as usize * 2) * std::mem::size_of::<f32>();
        dispatch_kernel_with_tgmem(
            encoder,
            &ctx.pipelines.attention_decode_flash_tile,
            &[q, k_cache, v_cache, flash_partials],
            &tile_push,
            tgmem,
            MTLSize { width: num_kv_heads as usize, height: num_tiles as usize, depth: 1 },
            MTLSize { width: (heads_per_kv * 32) as usize, height: 1, depth: 1 },
        );
    }

    // Phase 2: Reduce kernel — one TG per query head.
    let mut reduce_push = [0u8; 12];
    reduce_push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    reduce_push[4..8].copy_from_slice(&head_dim.to_le_bytes());
    reduce_push[8..12].copy_from_slice(&num_tiles.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.attention_decode_flash_reduce,
        &[flash_partials, output],
        &reduce_push,
        MTLSize { width: num_heads as usize, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// Prefill attention: batched Q*K^T with causal mask (flash / online softmax).
pub fn attention_prefill(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q: &MetalBuffer,
    k_cache: &MetalBuffer,
    v_cache: &MetalBuffer,
    output: &MetalBuffer,
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    q_dim: u32,
    cached_len: u32,
    start_pos: u32,
    scale: f32,
) {
    let mut push = [0u8; 36];
    push[0..4].copy_from_slice(&seq_len.to_le_bytes());
    push[4..8].copy_from_slice(&num_heads.to_le_bytes());
    push[8..12].copy_from_slice(&num_kv_heads.to_le_bytes());
    push[12..16].copy_from_slice(&head_dim.to_le_bytes());
    push[16..20].copy_from_slice(&kv_dim.to_le_bytes());
    push[20..24].copy_from_slice(&q_dim.to_le_bytes());
    push[24..28].copy_from_slice(&cached_len.to_le_bytes());
    push[28..32].copy_from_slice(&start_pos.to_le_bytes());
    push[32..36].copy_from_slice(&scale.to_le_bytes());

    let heads_per_kv = if num_kv_heads > 0 { num_heads / num_kv_heads } else { 1 };

    // Select variant, with fallback to v1 when GQA is 1:1 (vision encoder, 2B model).
    let mut variant = if heads_per_kv <= 1 {
        PrefillVariant::V1
    } else {
        prefill_variant()
    };

    // Auto-upgrade to V7 (MPP) or V6 (coop32) on Metal 4
    if variant == PrefillVariant::V4 {
        // Prefer V7 (MPP matmul2d) when available and head_dim is divisible by 32
        if ctx.pipelines.attention_prefill_v7_mpp.is_some() && head_dim % 32 == 0 {
            // V7: 1 TG per query head, BQ=BK=32
            // TG mem: shared_q(8192) + shared_kv(8192) + shared_s(4096) = 20480 B
            // Always fits in 32KB regardless of heads_per_kv
            variant = PrefillVariant::V7;
        } else if ctx.pipelines.attention_prefill_v6_coop32.is_some() {
            // V6: 1 TG per KV head, BQ=32 BK=8
            let bq: u32 = 32;
            let bk: u32 = 8;
            let q_bytes = (heads_per_kv * bq * head_dim) as usize * 2;
            let kv_bytes = (2 * bk * head_dim) as usize * 2;
            let s_bytes = (heads_per_kv * bq * bk) as usize * 4;
            let total = q_bytes + kv_bytes + s_bytes;
            if total <= 32768 {
                variant = PrefillVariant::V6;
            }
        }
    }

    match variant {
        PrefillVariant::V1 => {
            dispatch_kernel(
                encoder,
                &ctx.pipelines.attention_prefill,
                &[q, k_cache, v_cache, output],
                &push,
                MTLSize { width: num_heads as usize, height: seq_len as usize, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
        }
        PrefillVariant::V2 => {
            let tgmem = (head_dim as usize * 2) * std::mem::size_of::<f32>();
            dispatch_kernel_with_tgmem(
                encoder,
                &ctx.pipelines.attention_prefill_v2_gqa,
                &[q, k_cache, v_cache, output],
                &push,
                tgmem,
                MTLSize { width: num_kv_heads as usize, height: seq_len as usize, depth: 1 },
                MTLSize { width: (heads_per_kv * 32) as usize, height: 1, depth: 1 },
            );
        }
        PrefillVariant::V3 => {
            let bk: u32 = 16;
            let tgmem = (2 * bk as usize * head_dim as usize) * std::mem::size_of::<f32>();
            dispatch_kernel_with_tgmem(
                encoder,
                &ctx.pipelines.attention_prefill_v3_tiled,
                &[q, k_cache, v_cache, output],
                &push,
                tgmem,
                MTLSize { width: num_kv_heads as usize, height: seq_len as usize, depth: 1 },
                MTLSize { width: (heads_per_kv * 32) as usize, height: 1, depth: 1 },
            );
        }
        PrefillVariant::V4 | PrefillVariant::V5 => {
            let bq: u32 = 8;
            let bk: u32 = 8;
            // shared_q: heads_per_kv * BQ * head_dim * sizeof(half)
            let q_bytes = (heads_per_kv * bq * head_dim) as usize * 2;
            // shared_k + shared_v: 2 * BK * head_dim * sizeof(half)
            let kv_bytes = (2 * bk * head_dim) as usize * 2;
            // shared_s: heads_per_kv * BQ * BK * sizeof(float)
            let s_bytes = (heads_per_kv * bq * bk) as usize * 4;
            let tgmem = q_bytes + kv_bytes + s_bytes;

            // V4 and V5 are now identical (both read half KV cache).
            // Keep V5 pipeline for backwards compatibility with env var.
            let pipeline = if variant == PrefillVariant::V5 {
                &ctx.pipelines.attention_prefill_v5_half_kv
            } else {
                &ctx.pipelines.attention_prefill_v4_simdgroup
            };

            dispatch_kernel_with_tgmem(
                encoder,
                pipeline,
                &[q, k_cache, v_cache, output],
                &push,
                tgmem,
                MTLSize {
                    width: num_kv_heads as usize,
                    height: div_ceil(seq_len, bq) as usize,
                    depth: 1,
                },
                MTLSize { width: (heads_per_kv * 32) as usize, height: 1, depth: 1 },
            );
        }
        PrefillVariant::V6 => {
            let bq: u32 = 32;
            let bk: u32 = 8;
            let q_bytes = (heads_per_kv * bq * head_dim) as usize * 2;
            let kv_bytes = (2 * bk * head_dim) as usize * 2;
            let s_bytes = (heads_per_kv * bq * bk) as usize * 4;
            let tgmem = q_bytes + kv_bytes + s_bytes;

            let pipeline = ctx.pipelines.attention_prefill_v6_coop32.as_ref()
                .expect("V6 coop32 pipeline should be available when variant is V6");

            dispatch_kernel_with_tgmem(
                encoder,
                pipeline,
                &[q, k_cache, v_cache, output],
                &push,
                tgmem,
                MTLSize {
                    width: num_kv_heads as usize,
                    height: div_ceil(seq_len, bq) as usize,
                    depth: 1,
                },
                MTLSize { width: (heads_per_kv * 32) as usize, height: 1, depth: 1 },
            );
        }
        PrefillVariant::V7 => {
            // MPP matmul2d: 1 TG per query head, BQ=BK=32, 128 threads (4 simdgroups)
            // TG mem: shared_q(8192) + shared_kv(8192) + shared_s(4096) = 20480 B
            let tgmem: usize = 20480;

            let pipeline = ctx.pipelines.attention_prefill_v7_mpp.as_ref()
                .expect("V7 MPP pipeline should be available when variant is V7");

            dispatch_kernel_with_tgmem(
                encoder,
                pipeline,
                &[q, k_cache, v_cache, output],
                &push,
                tgmem,
                MTLSize {
                    width: num_heads as usize,          // 1 TG per query head
                    height: div_ceil(seq_len, 32) as usize,
                    depth: 1,
                },
                MTLSize { width: 128, height: 1, depth: 1 },  // 4 simdgroups for matmul2d
            );
        }
    }
}

/// Append a single K or V vector to the cache at position `seq_len`.
pub fn kv_cache_append(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    cache: &MetalBuffer,
    new_kv: &MetalBuffer,
    kv_dim: u32,
    seq_len: u32,
) {
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&kv_dim.to_le_bytes());
    push[4..8].copy_from_slice(&seq_len.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.kv_cache_append,
        &[cache, new_kv],
        &push,
        MTLSize { width: div_ceil(kv_dim, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Decode attention for INT8 KV cache: single query token against cached K/V.
///
/// Always uses FlashDecoding tile kernel (i8 variant) + standard reduce kernel.
/// Falls back to half-precision path for very short sequences (< FLASH_TILE_SIZE).
pub fn attention_decode_i8(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q: &MetalBuffer,
    k_cache: &MetalBuffer,
    v_cache: &MetalBuffer,
    k_scales: &MetalBuffer,
    v_scales: &MetalBuffer,
    output: &MetalBuffer,
    flash_partials: &MetalBuffer,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    cached_len: u32,
    scale: f32,
) {
    let heads_per_kv = if num_kv_heads > 0 { num_heads / num_kv_heads } else { 1 };
    let num_tiles = div_ceil(cached_len, FLASH_TILE_SIZE);

    // Phase 1: INT8 tile kernel
    let mut tile_push = [0u8; 24];
    tile_push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    tile_push[4..8].copy_from_slice(&num_kv_heads.to_le_bytes());
    tile_push[8..12].copy_from_slice(&head_dim.to_le_bytes());
    tile_push[12..16].copy_from_slice(&kv_dim.to_le_bytes());
    tile_push[16..20].copy_from_slice(&cached_len.to_le_bytes());
    tile_push[20..24].copy_from_slice(&scale.to_le_bytes());

    // Prefer v2 (double-buffered) when available, unless METAL_DECODE_V1=1
    let use_v2_i8 = !std::env::var("METAL_DECODE_V1").is_ok_and(|v| v == "1")
        && ctx.pipelines.attention_decode_flash_tile_i8_v2.is_some();

    if use_v2_i8 {
        // v2: double-buffered float TG: 4 × head_dim × sizeof(float)
        let tgmem = (head_dim as usize * 4) * std::mem::size_of::<f32>();
        dispatch_kernel_with_tgmem(
            encoder,
            ctx.pipelines.attention_decode_flash_tile_i8_v2.as_ref().unwrap(),
            &[q, k_cache, v_cache, flash_partials, k_scales, v_scales],
            &tile_push,
            tgmem,
            MTLSize { width: num_kv_heads as usize, height: num_tiles.max(1) as usize, depth: 1 },
            MTLSize { width: (heads_per_kv * 32) as usize, height: 1, depth: 1 },
        );
    } else {
        let tgmem = (head_dim as usize * 2) * std::mem::size_of::<f32>();
        dispatch_kernel_with_tgmem(
            encoder,
            &ctx.pipelines.attention_decode_flash_tile_i8,
            &[q, k_cache, v_cache, flash_partials, k_scales, v_scales],
            &tile_push,
            tgmem,
            MTLSize { width: num_kv_heads as usize, height: num_tiles.max(1) as usize, depth: 1 },
            MTLSize { width: (heads_per_kv * 32) as usize, height: 1, depth: 1 },
        );
    }

    // Phase 2: standard reduce kernel (partials are always float)
    let actual_tiles = num_tiles.max(1);
    let mut reduce_push = [0u8; 12];
    reduce_push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    reduce_push[4..8].copy_from_slice(&head_dim.to_le_bytes());
    reduce_push[8..12].copy_from_slice(&actual_tiles.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.attention_decode_flash_reduce,
        &[flash_partials, output],
        &reduce_push,
        MTLSize { width: num_heads as usize, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// Append a batch of K or V vectors to the cache starting at position `start`.
pub fn kv_cache_append_batch(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    cache: &MetalBuffer,
    new_kv: &MetalBuffer,
    kv_dim: u32,
    start: u32,
    count: u32,
) {
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&kv_dim.to_le_bytes());
    push[4..8].copy_from_slice(&start.to_le_bytes());
    push[8..12].copy_from_slice(&count.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.kv_cache_append_batch,
        &[cache, new_kv],
        &push,
        MTLSize { width: div_ceil(count * kv_dim, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Append a batch of K or V vectors to INT8 cache with scale computation.
///
/// Quantizes f32 → int8 with symmetric per-position per-head scales.
pub fn kv_cache_append_batch_i8(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    cache: &MetalBuffer,
    scales: &MetalBuffer,
    new_kv: &MetalBuffer,
    kv_dim: u32,
    start: u32,
    count: u32,
    head_dim: u32,
    num_kv_heads: u32,
) {
    let mut push = [0u8; 20];
    push[0..4].copy_from_slice(&kv_dim.to_le_bytes());
    push[4..8].copy_from_slice(&start.to_le_bytes());
    push[8..12].copy_from_slice(&count.to_le_bytes());
    push[12..16].copy_from_slice(&head_dim.to_le_bytes());
    push[16..20].copy_from_slice(&num_kv_heads.to_le_bytes());

    // One threadgroup per (kv_head, position) — 32 threads so simd_max covers all threads
    dispatch_kernel(
        encoder,
        &ctx.pipelines.kv_cache_append_batch_i8,
        &[cache, new_kv, scales],
        &push,
        MTLSize { width: num_kv_heads as usize, height: count as usize, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// Bulk quantize half-precision KV cache → INT8 with per-position per-head scales.
///
/// Used after prefill to convert the half KV cache to INT8 for decode.
pub fn kv_cache_quantize_half_to_i8(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    cache_i8: &MetalBuffer,
    cache_half: &MetalBuffer,
    scales: &MetalBuffer,
    kv_dim: u32,
    start: u32,
    count: u32,
    head_dim: u32,
    num_kv_heads: u32,
) {
    let mut push = [0u8; 20];
    push[0..4].copy_from_slice(&kv_dim.to_le_bytes());
    push[4..8].copy_from_slice(&start.to_le_bytes());
    push[8..12].copy_from_slice(&count.to_le_bytes());
    push[12..16].copy_from_slice(&head_dim.to_le_bytes());
    push[16..20].copy_from_slice(&num_kv_heads.to_le_bytes());

    // 32 threads = one SIMD group so simd_max covers all threads
    dispatch_kernel(
        encoder,
        &ctx.pipelines.kv_cache_quantize_half_to_i8,
        &[cache_i8, cache_half, scales],
        &push,
        MTLSize { width: num_kv_heads as usize, height: count as usize, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

// ============================================================================
// H2O eviction kernels
// ============================================================================

const H2O_TILE_SIZE: u32 = 256;

/// H2O score probe: compute Q.K dot products for eviction scoring (half KV).
///
/// Dispatch: grid=(num_kv_heads, num_tiles), threads=(heads_per_kv * 32)
/// Threadgroup memory: head_dim * sizeof(float) for shared K loading.
/// scores_out must be zeroed before calling (accumulates via +=).
pub fn h2o_score_probe_half(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q: &MetalBuffer,
    k_cache: &MetalBuffer,
    scores_out: &MetalBuffer,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    cached_len: u32,
    scale: f32,
) {
    let mut push = [0u8; 24];
    push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    push[4..8].copy_from_slice(&num_kv_heads.to_le_bytes());
    push[8..12].copy_from_slice(&head_dim.to_le_bytes());
    push[12..16].copy_from_slice(&kv_dim.to_le_bytes());
    push[16..20].copy_from_slice(&cached_len.to_le_bytes());
    push[20..24].copy_from_slice(&scale.to_le_bytes());

    let heads_per_kv = if num_kv_heads > 0 { num_heads / num_kv_heads } else { 1 };
    let num_tiles = div_ceil(cached_len, H2O_TILE_SIZE);

    let tgmem = head_dim as usize * std::mem::size_of::<f32>();

    dispatch_kernel_with_tgmem(
        encoder,
        &ctx.pipelines.h2o_score_probe_half,
        &[q, k_cache, scores_out],
        &push,
        tgmem,
        MTLSize { width: num_kv_heads as usize, height: num_tiles.max(1) as usize, depth: 1 },
        MTLSize { width: (heads_per_kv * 32) as usize, height: 1, depth: 1 },
    );
}

/// H2O score probe: compute Q.K dot products for eviction scoring (INT8 KV).
pub fn h2o_score_probe_i8(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    q: &MetalBuffer,
    k_cache: &MetalBuffer,
    k_scales: &MetalBuffer,
    scores_out: &MetalBuffer,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    cached_len: u32,
    scale: f32,
) {
    let mut push = [0u8; 24];
    push[0..4].copy_from_slice(&num_heads.to_le_bytes());
    push[4..8].copy_from_slice(&num_kv_heads.to_le_bytes());
    push[8..12].copy_from_slice(&head_dim.to_le_bytes());
    push[12..16].copy_from_slice(&kv_dim.to_le_bytes());
    push[16..20].copy_from_slice(&cached_len.to_le_bytes());
    push[20..24].copy_from_slice(&scale.to_le_bytes());

    let heads_per_kv = if num_kv_heads > 0 { num_heads / num_kv_heads } else { 1 };
    let num_tiles = div_ceil(cached_len, H2O_TILE_SIZE);

    let tgmem = head_dim as usize * std::mem::size_of::<f32>();

    dispatch_kernel_with_tgmem(
        encoder,
        &ctx.pipelines.h2o_score_probe_i8,
        &[q, k_cache, k_scales, scores_out],
        &push,
        tgmem,
        MTLSize { width: num_kv_heads as usize, height: num_tiles.max(1) as usize, depth: 1 },
        MTLSize { width: (heads_per_kv * 32) as usize, height: 1, depth: 1 },
    );
}

/// KV cache compaction: gather kept positions into contiguous layout (half).
///
/// index_map: [num_kept] u32, sorted ascending (guarantees new_pos <= old_pos).
pub fn kv_cache_compact_half(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    cache: &MetalBuffer,
    index_map: &MetalBuffer,
    stride: u32,
    num_kept: u32,
) {
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&stride.to_le_bytes());
    push[4..8].copy_from_slice(&num_kept.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.kv_cache_compact_half,
        &[cache, index_map],
        &push,
        MTLSize { width: num_kept as usize, height: 1, depth: 1 },
        MTLSize { width: stride.min(256) as usize, height: 1, depth: 1 },
    );
}

/// KV cache compaction: gather kept positions into contiguous layout (INT8).
pub fn kv_cache_compact_i8(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    cache: &MetalBuffer,
    index_map: &MetalBuffer,
    stride: u32,
    num_kept: u32,
) {
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&stride.to_le_bytes());
    push[4..8].copy_from_slice(&num_kept.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.kv_cache_compact_i8,
        &[cache, index_map],
        &push,
        MTLSize { width: num_kept as usize, height: 1, depth: 1 },
        MTLSize { width: stride.min(256) as usize, height: 1, depth: 1 },
    );
}

/// KV cache compaction: gather kept scale values (f32) into contiguous layout.
pub fn kv_cache_compact_scales(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    scales: &MetalBuffer,
    index_map: &MetalBuffer,
    stride: u32,
    num_kept: u32,
) {
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&stride.to_le_bytes());
    push[4..8].copy_from_slice(&num_kept.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.kv_cache_compact_scales,
        &[scales, index_map],
        &push,
        MTLSize { width: num_kept as usize, height: 1, depth: 1 },
        MTLSize { width: stride.min(256) as usize, height: 1, depth: 1 },
    );
}
