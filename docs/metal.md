# Metal GPU Backend

Apple Metal compute backend for inference on Apple Silicon (macOS only). Supports BF16, INT8, and Q4 quantization with 100+ compute shaders, including Metal 4 cooperative tensor and MetalPerformancePrimitives paths.

## GPU Family Support

| Family | Chips | Key Features |
|--------|-------|-------------|
| Apple7 | M1 / M1 Pro / M1 Max | `simdgroup_matrix` 8×8 |
| Apple8 | M2 | Baseline compute |
| Apple9 | M3 | Dynamic caching |
| Apple10 | M4 | `cooperative_tensor` 32×32 |
| Apple11 | M5 | Neural Accelerators (`matmul2d` via MetalPerformancePrimitives) |

---

## Memory Management

**Unified Memory:** `StorageModeShared` provides zero-copy CPU-GPU access on Apple Silicon. Model weights can be loaded via `from_ptr_nocopy()` for mmap-backed zero-copy binding.

Key buffer operations: `new()` (zeroed), `new_uninit()` (output-only), `from_data()` (copy), `slice()` (view with byte offset, shared GPU reference).

---

## KV Cache

**Dual-format storage:**
- **Half-precision** (always allocated): `[max_tokens, kv_dim]` — used by prefill attention
- **INT8** (optional): `[max_tokens, kv_dim]` + per-position per-head scales — used by decode

**INT8 quantization:** Per-position per-head symmetric: `scale = max(|x|) / 127.0`. Applied after prefill via `kv_cache_quantize_i8` shader. During decode, fused norm+rope+i8 shaders append directly to the INT8 cache.

**H2O eviction buffers** (optional): `h2o_scores` for attention probe scores, `h2o_index_map` for position remapping after compaction.

---

## Shader Catalog

### Normalization

| Shader | Purpose | Threads | Dispatch |
|--------|---------|---------|----------|
| `rms_norm` | Single vector (decode) | 256 | `(1, 1, 1)` |
| `rms_norm_batch` | Batch (prefill) | 32/pos | `(batch, 1, 1)` |
| `rms_norm_residual` | Fused residual + norm | 256 | `(1, 1, 1)` |
| `rms_norm_half_mpp` | Metal 4 half accumulation (2× FP16 throughput) | 256 | `(1, 1, 1)` |
| `head_rms_norm` | Per-head QK norms | 32/head | `(num_heads, 1, 1)` |
| `layer_norm_batch` | Vision encoder (mean-centered) | 32/pos | `(batch, 1, 1)` |

### Matvec (Decode, M=1)

All dispatch: `grid=(ceil(N/4), 1, 1)`, `threads=(128, 1, 1)` (4 rows per threadgroup).

| Shader | Quantization | Notes |
|--------|-------------|-------|
| `bf16_matvec` | BF16 | — |
| `int8_matvec` | INT8 | Per-channel scales |
| `q4_matvec` | Q4 | Baseline |
| `q4_matvec_v2` | Q4 | 8-row, `uint4` loads, arithmetic dequant |
| `q4_matvec_residual_add` | Q4 | Fused matvec + residual |
| `q4_matvec_residual_add_v2` | Q4 | V2 variant |
| `q4_matvec_residual_add_8row` | Q4 | 8-row fused |
| `q4_matvec_qkv` | Q4 | Fused Q/K/V projection (3 outputs) |
| `q4_matvec_qkv_normed` | Q4 | Fused RMS norm + Q/K/V |

### Matmul (Prefill, M>1)

| Shader | Quantization | Tile | Threads | Target |
|--------|-------------|------|---------|--------|
| `q4_matmul` / `bf16_matmul` / `int8_matmul` | Various | None | 32 | Small M,N |
| `q4_matmul_tiled` / `bf16_matmul_tiled` | Q4, BF16 | 4×4 | 128 | General prefill |
| `q4_matmul_tiled_simdgroup` | Q4 | 16×32 | 256 | Apple7+ (`simdgroup_matrix` 8×8) |
| `q4_matmul_tiled_coop32` / `bf16_matmul_tiled_coop32` | Q4, BF16 | 32×32 | 128 | Metal 4 (`cooperative_tensor`) |
| `q4_matmul_tiled_mpp` / `bf16_matmul_tiled_mpp` | Q4, BF16 | 32×32×32 | 128 | Metal 4 (Neural Accelerators, **1.7–1.9× faster** than coop32) |

**MPP matmul details:** 4 simdgroups × 32 threads, `matmul2d` descriptor `(32, 32, 32, false, true, false, multiply_accumulate)`. Double-buffered dequant 4-by-4. Threadgroup memory: 12 KB (2× input tiles as half + output as float).

### Attention — Decode

| Shader | KV Format | Threads | Use Case |
|--------|----------|---------|----------|
| `attention_decode` | F32 | 32 per head | Short sequences |
| `attention_decode_gqa` | F32 | `heads_per_kv × 32` | GQA with shared K/V |
| `attention_decode_flash_tile` | Half | `heads_per_kv × 32` | FlashDecoding (>256 tokens) |
| `attention_decode_flash_tile_v2` | Half | `heads_per_kv × 32` | Double-buffered (overlaps K/V load with compute) |
| `attention_decode_flash_tile_i8` | INT8 | `heads_per_kv × 32` | INT8 KV dequant |
| `attention_decode_flash_tile_i8_v2` | INT8 | `heads_per_kv × 32` | Double-buffered INT8 |

FlashDecoding uses tile size = 256 positions. Partials layout: `[tile_idx, num_heads, 2 + head_dim]` (max_val, sum_exp, accumulated output). Online softmax for numerical stability.

### Attention — Prefill

| Variant | Tile | Threads | Target |
|---------|------|---------|--------|
| v1 | 1 query | 32 | Baseline |
| v2 (GQA) | 1 query | `heads_per_kv × 32` | Share K/V via threadgroup memory |
| v3 (K-tiled) | BK=16 | `heads_per_kv × 32` | Better coalescing |
| v4 (simdgroup) | BQ=8 | `heads_per_kv × 32` | Apple7+ (`simdgroup_matrix` 8×8) |
| v6 (coop32) | BQ=32 | `heads_per_kv × 32` | Metal 4 (`cooperative_tensor` 32×32) |
| v7 (MPP) | BQ=32 | 128 | Metal 4 (Neural Accelerator `matmul2d`) |

Auto-selection: v7 if available, else v6, else v4.

### RoPE

| Shader | Purpose | Threads |
|--------|---------|---------|
| `rope_single` | Single position (decode) | 256 |
| `rope_batch` | Batch (prefill) | 256 |
| `rope_kv_append` | Fused RoPE + KV cache append | 256 |
| `head_norm_rope` | Fused per-head norm + RoPE | per-head |
| `head_norm_rope_kv_append` | Fused norm + RoPE + KV append (half) | per-head |
| `head_norm_rope_kv_append_i8` | Same, INT8 KV variant | per-head |

### KV Cache Operations

| Shader | Purpose |
|--------|---------|
| `kv_cache_append` / `kv_cache_append_i8` | Single-token append (half / INT8) |
| `kv_cache_append_batch` / `kv_cache_append_batch_i8` | Batch append (prefill) |
| `kv_cache_quantize_i8` | Bulk half → INT8 (post-prefill) |
| `kv_cache_compact_half` / `kv_cache_compact_i8` / `kv_cache_compact_scales` | H2O compaction |

### MoE Kernels (18+)

**Routing:**
- `moe_softmax_topk` — single token, GPU-side softmax + top-k
- `moe_softmax_topk_batch` — per-token batch routing

**Decode (single token, contiguous expert weights):**
- `moe_fused_gate_up_swiglu_{q4,int8,bf16}` — fused gate + up + SwiGLU per expert
- `matvec_scaled_add_{q4,int8,bf16}` — down projection + weighted residual

**Decode (batched, zero CPU-GPU sync):**
- `moe_batched_gate_up_swiglu_{q4,int8,bf16}` — contiguous expert dispatch
- `moe_batched_down_{q4,int8,bf16}` — down projection
- `moe_reduce` / `moe_reduce_residual` — sum expert outputs

**Prefill (zero-sync with counting sort):**
- `moe_prefill_sort` — counting sort (group tokens by expert on GPU)
- `moe_prefill_tiled_gate_up_swiglu_{q4,int8,bf16}` — tiled (16 tokens × 32 cols)
- `moe_prefill_tiled_down_{q4,int8,bf16}` / `moe_prefill_tiled_down_residual_{q4,int8,bf16}` — down + atomic residual
- `moe_prefill_reduce_residual` — final residual

Expert weights stored contiguously: `[expert_0 || expert_1 || ... || expert_N]`. No per-layer GPU→CPU sync — all routing happens on GPU.

### Activation & Utilities

`swiglu`, `fused_gate_up_swiglu`, `gelu_batch` (vision), `softmax`, `argmax` (two-stage), `embedding`, `residual_add`, `scaled_add`, `bias_add_batch`, `copy_buffer`

### Vision Encoder

- `layer_norm_batch` — LayerNorm for vision (not RMS norm)
- `gelu_batch` — GELU activation
- `vision_split_qkv` — split fused QKV → separate Q, K, V
- `vision_spatial_merge` — merge 2×2 consecutive patches
- `deepstack_add` — add vision features at image token positions

GPU-accelerated for Qwen3-VL. Pixtral (Mistral3) runs on CPU.

---

## Pipeline Architecture

### Decode Pipeline (per token)

1. Embedding lookup
2. For each layer:
   - Norm1 → Q/K/V projections → QK norms + RoPE (fused) → KV append
   - Attention decode (FlashDecoding if seq_len > 256)
   - O projection → residual + norm2
   - MLP: dense (gate+up+SwiGLU → down + residual) or MoE (router → dispatch)
3. Final norm → LM head → argmax

### Prefill Pipeline

1. Embedding lookup (batch)
2. For each layer:
   - Norm1 (batch) → Q/K/V matmul → QK norms + RoPE (batch) → KV batch append
   - Attention prefill (v4/v6/v7 auto-selected)
   - O matmul → residual + norm2 (batch)
   - MLP matmul or MoE (softmax_topk_batch → sort → tiled dispatch)
3. Final norm → LM head matmul → argmax

### Command Buffer Strategy

Default: single command buffer per inference step. With profiling, multiple CBs for timing granularity. Timeout: 30s per CB with async completion handler (avoids uninterruptible kernel sleep).

---

## Profiling

| Level | `METAL_PROFILE` | Granularity |
|-------|----------------|-------------|
| 0 | Disabled | — |
| 1 | Coarse | Attention vs MLP per layer |
| 2 | Detailed | Per-kernel (norm, qkv, rope, attn, oproj, gateup, down) |
| 3 | GPU counters | Timestamp sampling at dispatch boundaries (zero overhead) |

**Environment variables:**
- `METAL_ATTN_PREFILL=v1|v2|v3|v4|v6|v7` — force prefill variant
- `METAL_DISABLE_GQA_DECODE=1` — disable GQA decode
- `METAL_DISABLE_DENSE_Q4_GATEUP=1` — disable fused Q4 gate+up
- `METAL_DECODE_V1=1` — use FlashDecoding v1
- `METAL_LAYERS_PER_CB=N` — group N layers per command buffer

---

## Weight Formats on GPU

| Format | Storage | Dequant |
|--------|---------|---------|
| BF16 | `uint32` (2 BF16 packed) | Unpack during compute |
| INT8 | `uint32` (4 i8 packed) + `[N]` f32 scales | `val * scale[row]` |
| Q4 | Nibble pairs (2 per byte) + `[N, n_groups]` f32 scales | `(float(nibble) - 8.0) * scale[row * n_groups + col/32]` |

---

## Quantization Support Matrix

| Category | BF16 | INT8 | Q4 |
|----------|------|------|-----|
| Matvec (decode) | ✓ | ✓ | ✓ |
| Matmul (prefill) | ✓ | ✓ | ✓ |
| Matmul tiled | ✓ | — | ✓ |
| Matmul coop32 / MPP | ✓ | — | ✓ |
| Attention decode | ✓ (KV) | ✓ (KV) | — |
| MoE gate+up / down | ✓ | ✓ | ✓ |
| KV cache | Half, INT8 | — | — |
| Head norm + RoPE | ✓ | ✓ | — |
