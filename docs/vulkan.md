# Vulkan GPU Backend

Vulkan 1.3 compute backend for GPU inference on Linux. Portable across AMD, NVIDIA, and Intel GPUs. Supports BF16, INT8, and Q4 quantization with optional cooperative matrix acceleration.

## GPU Support

| Vendor | Subgroup Size | Cooperative Matrix | Notes |
|--------|--------------|-------------------|-------|
| NVIDIA (RTX) | 32 | FP16 16×16×16 + INT8 16×16×32 | INT8 WMMA for 2× matmul throughput |
| AMD (RDNA3) | 64 (wave64) or 32 | FP16 16×16×16 | Wave64 for wider subgroup reductions |
| Intel | 32 | — | Baseline compute |

Subgroup size is auto-detected. Shaders are compiled as both w32 and w64 SPIR-V variants.

---

## Shader Catalog

31 GLSL compute shaders covering all inference operations. All use `SET 0` with push descriptor extension (descriptors injected per-dispatch, no pre-allocated descriptor sets).

### Linear Algebra

| Shader | Quantization | Dispatch | Push Constants |
|--------|-------------|----------|----------------|
| `f32_matvec` | F32 | `(N, 1, 1)` | K |
| `f32_matmul` | F32 | `(N, M, 1)` | M, K |
| `bf16_matvec` | BF16 | `(N, 1, 1)` | K |
| `bf16_matmul` | BF16 | `(N, M, 1)` | M, K |
| `int8_matvec` | INT8 | `(N, 1, 1)` | K |
| `int8_matmul` | INT8 | `(N, M, 1)` | M, K |
| `q4_matvec` | Q4 | `(N, 1, 1)` | N, K |
| `q4_matvec_v5` | Q4 | `(N, 1, 1)` | N, K |
| `q4_matmul` | Q4 | `(N, M, 1)` | M, N, K |
| `q4_matmul_v5` | Q4 | `(N, M, 1)` | M, N, K |
| `q4_matmul_coopmat` | Q4 (FP16 WMMA) | `(ceil(N/16), ceil(M/16), 1)` | M, N, K |
| `q4_matmul_coopmat_i8` | Q4 (INT8 WMMA) | `(ceil(N/16), ceil(M/16), 1)` | M, N, K |

### Attention

| Shader | Purpose | Dispatch | Push Constants |
|--------|---------|----------|----------------|
| `attention_decode` | Decode (Q vs cached KV) | `(num_heads, 1, 1)` | 6 params (24 B) |
| `attention_prefill` | Prefill (causal mask) | `(num_heads, seq_len, 1)` | 9 params (36 B) |

### Normalization & Activation

| Shader | Purpose | Dispatch |
|--------|---------|----------|
| `rms_norm` | Single vector (decode) | `(1, 1, 1)` |
| `rms_norm_batch` | Batch (prefill) | `(batch_size, 1, 1)` |
| `head_rms_norm` | Per-head QK norms | `(num_heads, 1, 1)` |
| `softmax` | In-place softmax | `(1, 1, 1)` |
| `swiglu` | SiLU(gate) × up | `(ceil(n/256), 1, 1)` |

### Position Encoding

| Shader | Purpose | Dispatch |
|--------|---------|----------|
| `rope_single` | Single position (decode) | `(ceil(total/256), 1, 1)` |
| `rope_batch` | Batch (prefill) | `(ceil(total/256), 1, 1)` |

### KV Cache

| Shader | Purpose | Dispatch |
|--------|---------|----------|
| `kv_cache_append` | Single-token append | `(ceil(kv_dim/256), 1, 1)` |
| `kv_cache_append_batch` | Batch append | `(ceil(count*kv_dim/256), 1, 1)` |

### MoE

| Shader | Purpose | Dispatch |
|--------|---------|----------|
| `moe_gather` | Extract expert rows by index | `(ceil(count*dim/256), 1, 1)` |
| `moe_scatter_add` | Weighted scatter-add of expert outputs | `(ceil(count*dim/256), 1, 1)` |

### Utilities

`embedding`, `argmax`, `residual_add`, `scaled_add`, `bias_add_batch`, `copy_buffer`

---

## Q4 Matvec — Baseline vs V5

### Baseline (`q4_matvec.comp`)

Per-thread processes packed words (8 nibbles per `uint32`):
```glsl
for (uint i = lane; i < K/8; i += SUBGROUP_SIZE) {
    uint packed = w_packed[row * K/8 + i];
    for (uint nib = 0; nib < 8; nib++) {
        uint nibble = (packed >> (nib*4)) & 0xF;
        acc += x[i*8 + nib] * (float(nibble) - 8.0) * scale;
    }
}
acc = subgroupAdd(acc);
```

### V5 — Vectorized Per-Group Loads (`q4_matvec_v5.comp`)

Processes full K-groups (32 nibbles = 4 packed words) per iteration:
```glsl
for (uint g = lane; g < n_groups; g += SUBGROUP_SIZE) {
    float scale = scales[row * n_groups + g];  // 1 scale read per 32 values
    uint p0 = w[g*4], p1 = w[g*4+1], p2 = w[g*4+2], p3 = w[g*4+3];
    // Unpack all 32 nibbles with shared scale
}
```

**Benefits:** 4× fewer scale fetches, better instruction-level parallelism, aligned with group_size=32.

---

## Cooperative Matrix Matmul

### FP16 WMMA (`q4_matmul_coopmat.comp`)

256 threads, shared memory tiles of 16×16 (2048 bytes).

```glsl
coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> MatA;
coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> MatB;
coopmat<float32_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> Acc;

// Tiling loop (K in steps of 16):
// 1. Load A tile into shared memory as FP16
// 2. Dequant W tile from Q4 → FP16, transpose into shared memory
// 3. Acc = coopMatMulAdd(MatA, MatB, Acc)
```

### INT8 WMMA (`q4_matmul_coopmat_i8.comp`) — NVIDIA Only

16×16×32 tiles for 2× throughput vs FP16:
- A dynamically quantized to INT8 per (row, K-group)
- W Q4 nibbles converted to INT8: `nibble - 8 ∈ [-8, +7]`
- Accumulation as `int32`, then dequantized: `a_scale * w_scale * int32_result`
- K_tile = 32 matches Q4 group size exactly

**Dispatch strategy:**
```
if M >= 16 && N < K*2:
    try INT8 WMMA (NVIDIA)
    fallback to FP16 WMMA
else:
    use v5 vectorized or baseline
```

---

## Attention Implementation

### Decode (`attention_decode.comp`)

Three synchronous phases within one workgroup:

1. **Q·K^T:** Each lane processes `cached_len / SUBGROUP_SIZE` positions. GQA: `kv_head = query_head / (num_heads / num_kv_heads)`.
2. **Softmax:** `subgroupMax()` → `exp(score - max)` → `subgroupAdd()` → normalize. Barrier between phases.
3. **V accumulation:** `output[d] = sum_t(softmax[t] * v_cache[t, d])`

### Prefill (`attention_prefill.comp`)

Same structure but with causal masking: positions `t > query_pos` receive `-INF` before softmax.

---

## Subgroup Operations

Used throughout for efficient reductions:

```glsl
subgroupAdd(value)    // Sum across all lanes
subgroupMax(value)    // Max across all lanes
subgroupShuffleDown(value, offset)  // Shift for tree reduction
```

**Pattern (matvec):** Each lane accumulates partial sums over K/SUBGROUP_SIZE elements, then `subgroupAdd()` reduces to a single result. Lane 0 writes the output.

---

## Vulkan Pipeline Architecture

### Push Descriptors

All bindings use `VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR`. Descriptors are pushed per-dispatch — no pre-allocated descriptor pools.

### Command Buffer Lifecycle

```
cmd_begin() → record dispatches with barriers → cmd_end_submit_wait()
```

Pipeline barriers between compute kernels: `SHADER_WRITE → SHADER_READ`, `COMPUTE_SHADER → COMPUTE_SHADER`.

### Memory Management

Two allocation modes:
- **Device-local:** `DEVICE_LOCAL` — GPU-only, highest bandwidth
- **Host-visible:** `HOST_VISIBLE | HOST_COHERENT` — CPU-accessible staging

`device_local_zeroed()` allocates and clears via staging copy.

### Feature Detection

```rust
// Subgroup size selection
if is_amd && max_subgroup_size >= 64 { 64 } else { 32 }

// Cooperative matrix (optional)
has_coopmat = extensions.contains("VK_KHR_cooperative_matrix")

// Required features
storageBuffer8BitAccess: true
subgroupSizeControl: true
computeFullSubgroups: true
```

---

## Weight Formats

| Format | Buffer Layout | Dequantization |
|--------|--------------|----------------|
| F32 | `float[N * K]` | None |
| BF16 | `uint32[N * K/2]` (2 BF16 per word) | Bit-shift to f32 |
| INT8 | `uint32[N * K/4]` (4 i8 per word) + `float[N]` scales | `val * scale[row]` |
| Q4 | `uint32[N * K/8]` (8 nibbles per word) + `float[N * ceil(K/32)]` scales | `(float(nibble) - 8.0) * scale[row * n_groups + col/32]` |
