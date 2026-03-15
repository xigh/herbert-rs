# CPU Backends (AVX-512)

herbert-rs ships four CPU backends sharing a common thread pool and attention infrastructure. All critical inner loops are hand-written x86-64 assembly (`.S` files) with runtime feature detection and scalar fallbacks.

## Backend Overview

| Backend | Crate | Weights | SIMD | Instruction |
|---------|-------|---------|------|-------------|
| BF16 scalar | `backend_bf16` | BF16 (2 B/param) | None | Scalar f32 loops |
| BF16 AVX-512 | `backend_bf16_avx512` | BF16 (2 B/param) | AVX-512 BF16 | `vdpbf16ps` |
| INT8 AVX-512 | `backend_int8_avx512` | INT8 (1 B/param) | AVX-512 VNNI | `VPDPBUSD` |
| Q4 | `backend_q4` | Q4 (0.5 B/param) | AVX-512 VNNI / AVX2 | `VPDPBUSD` / `vpmaddubsw` |

---

## Weight Formats

### BF16

Row-major `[N, K]` array of `u16` (BrainFloat16). No quantization — used as a reference baseline for correctness verification. Cache format version 2.

### INT8

Per-channel symmetric quantization. Weights stored as unsigned `u8` (value + 128) in row-major `[N, K]` layout. One `f32` scale per output row.

- **Quantization:** `scale = max(|row|) / 127.0`, then `q = clamp(round(val / scale), -127, +127) + 128`
- **Dequantization:** `y[row] = (raw_dot - 128 * sum(x_i8)) * w_scale[row] * x_scale`
- **Cache format:** Version 3 — `[n:u64][k:u64][data:u8*n*k][scales:f32*n]`

### Q4

Per-group symmetric 4-bit quantization with pre-interleaved nibble packing. Group size = 32 K-values.

**Data layout:** `[n_tiles, K/4, 16, 4]` where `n_tiles = ceil(N / 32)`

Each byte packs two lanes:
```
byte[4*hl + j] = nib(lane_hl, kj) | (nib(lane_hl+16, kj) << 4)
```
- Low nibble = lanes 0–15, high nibble = lanes 16–31
- Eliminates `vpunpcklbw` / `vpunpckhbw` from the inner loop
- Direct extraction: `vpandd` gives lanes 0–15, `vpsrld + vpandd` gives 16–31

**Scales:** Column-major `scales[group * N + row]` — contiguous rows per group for stride-1 SIMD loads. An alternative tile-local layout (`scales_tiled`) is computed at load time for fused kernels, reducing DTLB misses by 10–20%.

**Nibble encoding:** Unsigned `[0, 15]` representing symmetric Q4 values `[-8, +7]` stored as `q + 8`.

---

## SIMD Instructions

### `vdpbf16ps` (AVX-512 BF16)

Paired BF16 dot-product: 32 BF16 inputs → 16 f32 partial sums per instruction. Throughput: 1/cycle on Zen4.

```
For each i in 0..15:
  acc[i] += a[2*i] * b[2*i] + a[2*i+1] * b[2*i+1]
```

Input conversion uses `vcvtne2ps2bf16` (32 f32 → 16 BF16).

### `VPDPBUSD` (AVX-512 VNNI)

4-element unsigned×signed dot-product: 64 u8×i8 → 16 i32 per ZMM. Throughput: 1/cycle on Zen4, 5-cycle latency. Dual-accumulator strategy hides latency.

Two kernel variants for matvec:
- **V1 (generic):** 4-row unroll, x loaded per row → 12 loads/iter
- **V2 (specialized):** x loaded once, shared across 4 rows → 5 loads/iter (−58% LDQ pressure)

Controlled by the `matvec-v2` Cargo feature.

### AVX2 Fallback (Q4)

For CPUs without AVX-512 VNNI:
```
vpmaddubsw ymm, ymm_u8, ymm_i8   → u8×i8 → i16 pairs
vpmaddwd   ymm, ymm_i16, ymm_ones → i16 pairs → i32
vpaddd     ymm_acc, ymm_acc, ymm  → accumulate
```
3 instructions per `VPDPBUSD` equivalent.

---

## Kernel Architecture

### Matvec (y = W @ x, M=1)

Tile-based parallelization: `TILE_ROWS = 64` output rows per work unit.

**BF16-AVX512:**
1. Convert input x (f32) → BF16 via `vcvtne2ps2bf16` (stack buffer)
2. 4-row unrolled `vdpbf16ps` loop over K dimension
3. Horizontal sum each ZMM → scalar f32 (`vextractf32x8`, `vhaddps`)

**INT8-AVX512:**
1. Quantize input x (f32) → i8 + per-token scale
2. 4-row `VPDPBUSD` unroll (V2: x loaded once per K-chunk)
3. Dequant: `y[row] = (raw_dot - 128*x_sum) * w_scale[row] * x_scale`

**Q4:**
1. Pre-interleaved nibble extraction: `vpandd` for low lanes, `vpsrld + vpandd` for high lanes
2. 4-lane `VPDPBUSD` unroll per iteration
3. Per-group dequant with column-major scale loads

### Matmul (C = A @ W^T, M>1)

Two-level tiling: outer over output row tiles (TILE_ROWS=64), inner over all M input rows. Pre-quantizes all M input rows at load time (INT8) to avoid redundant quantization.

### Fused Operations

| Operation | What it fuses | Bandwidth savings |
|-----------|--------------|-------------------|
| **Fused 3-matvec** (INT8) | Q, K, V projections | Quantizes x once instead of 3× |
| **Fused 2-matvec** (INT8) | Gate + up (FFN) | Quantizes x once instead of 2× |
| **Fused attention 8-head** | DOT + EXP + SV | No memory round-trip between phases |
| **Q4 dequant-in-ASM** | Unpack nibbles + dequant | No separate dequant pass |

The fused 8-head attention kernel (`avx512_fused_attn_bf16.S`) keeps scores in registers between DOT → HSUM → EXP → SV phases, eliminating stack round-trips. RDTSC profiling counters are available via the `PROFILE_RDTSC` macro.

---

## Attention Kernels

### Decode Attention

Per-head, per-position dot products with online softmax. Multiple KV quantization variants:

| KV Format | Kernel | Notes |
|-----------|--------|-------|
| BF16 | `decode_head_attention` | BF16→f32 on-the-fly |
| F32 | Same function | Pure f32 path |
| INT8 | `decode_head_attention` (quantized path) | Per-position per-head dequant |

**Software prefetch:** `_mm_prefetch(_MM_HINT_T0)` with distance = 4 positions ahead. Disabled via `no-sw-prefetch` feature.

**Context-dependent inner loop:**
- Small context (≤ 64 positions): inner loop per-dimension, outer per-position
- Large context: outer loop per-position, inner per-dimension

### Prefill Attention

2-pass softmax with batch processing. Separate kernel files per KV quantization:
- `kernel_bf16.rs`, `kernel_int8.rs`, `kernel_int4.rs`, `kernel_f32.rs`

### Fused 8-Head Attention (AVX-512)

Processes 8 heads simultaneously for one KV position:
1. **DOT8:** 8 parallel dot products (zmm4–11 accumulators)
2. **HSUM8:** Horizontal sum → 8 scalar scores
3. **EXP:** Online softmax with running max/sum, inline exp approximation
4. **SV:** `out[h] = correction[h] * out[h] + alpha[h] * V[j]`

Exp approximation uses piecewise polynomial (`avx512_exp_inline.inc`).

---

## MoE Support

- Single-threaded matvec variants (`matvec_st`) for sequential expert dispatch
- Parallel expert execution when `PARALLEL_EXPERTS = true`
- Router gates kept at INT8 precision (better than Q4 for routing decisions)
- Expert weights quantized same as main projections

---

## Memory Optimizations

### Huge Pages (Q4 backend)

`HugeVec<T>` attempts `mmap(MAP_HUGETLB)` for 2MB page allocation on Linux, falling back to `Vec + MADV_HUGEPAGE`. Reduces DTLB misses for multi-MB weight buffers (measured: −45% DRAM latency on Zen4). Enabled via `hugepages` feature.

### NUMA First-Touch

`first_touch_weights()` forces page placement near pinned worker threads after weight cache load. No-op on non-NUMA systems.

### Scale Layout for DTLB

The tile-local scale variant (`scales_tiled`) keeps all scale groups for one tile contiguous (~12 KB for 96 groups), eliminating DTLB misses within tile processing.

---

## Thread Parallelization

### Global Thread Pool

Persistent worker threads with shared job queue and generation counter (ABA avoidance). Work-stealing dispatch via `parallel_for`:

```rust
pool.parallel_for(num_tiles, |ctx, start, end| {
    kernel_func(ctx, start, end)
})
```

### Dispatch Priority

Thread-local priority (0–9) for concurrent inference request scheduling. Lower values preempt.

### Autotune

`RuntimeAutoTuner` learns optimal parallelization thresholds across runs via environment variables (`HERBERT_AVX512_Q4_MATVEC`, etc.).

---

## Weight Caching

| Backend | Cache Version | Contents |
|---------|--------------|----------|
| BF16 | 2 | `[n:u64][k:u64][data:u16*n*k]` |
| INT8 | 3 | `[n:u64][k:u64][data:u8*n*k][scales:f32*n]` |
| Q4 | 5 | `[n:u64][k:u64][data_len:u64][data:u8*..][scales_len:u64][scales:f32*..]` |

Zero-copy mmap read. Model hash (SHA-256 of config.json + weight names) invalidates stale caches.

---

## Runtime Feature Detection

Cached detection via `AtomicU8` + `is_x86_feature_detected!()`:
- `avx512f` — 512-bit integer/float
- `avx512bf16` — BF16 dot-product
- `avx512vnni` — VNNI dot-product
- `avx2`, `fma` — fallback

**Dispatch chain:**
1. BF16-AVX512: try `avx512bf16` → else scalar fallback
2. INT8-AVX512: try `avx512vnni` → else scalar
3. Q4: try `avx512vnni` → else try `avx2` → else scalar

All backends ship scalar Rust fallbacks — correctness is always preserved, only performance degrades.

**Environment overrides:**
- `HERBERT_AVX512_Q4_DISABLE_FUSED3_3WAY=1` — disable fused3_3way variant
- `HERBERT_AVX512_Q4_MATVEC` / `_MATMUL` — autotuner configuration
