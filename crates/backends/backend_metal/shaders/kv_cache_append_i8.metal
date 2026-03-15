#include <metal_stdlib>
using namespace metal;

// Symmetric per-position per-head INT8 quantization for KV cache writes.
//
// Quantization scheme:
//   scale = max(|head_vector|) / 127.0
//   quantized = round(x / scale), clamped to [-127, 127]
//
// Cache layout: cache_data[(pos * num_kv_heads + head_idx) * head_dim + d]
// Scale layout: scales[pos * num_kv_heads + head_idx]

// ---------------------------------------------------------------------------
// Kernel 1: Single token append
// ---------------------------------------------------------------------------

struct KvCacheAppendI8Params {
    uint kv_dim;       // num_kv_heads * head_dim
    uint seq_len;      // position to write to
    uint head_dim;
    uint num_kv_heads;
};

// Dispatch: grid = (num_kv_heads, 1, 1), threads = (min(head_dim, 256), 1, 1)
[[kernel]]
void kv_cache_append_i8(
    device char*                    cache_data [[buffer(0)]],
    device const float*             new_kv     [[buffer(1)]],
    device float*                   scales     [[buffer(2)]],
    constant KvCacheAppendI8Params& p          [[buffer(3)]],
    uint tid                        [[thread_position_in_threadgroup]],
    uint head_idx                   [[threadgroup_position_in_grid]]
) {
    if (head_idx >= p.num_kv_heads) return;

    uint head_dim  = p.head_dim;
    uint src_off   = head_idx * head_dim;

    // --- Pass 1: find max(|x|) across this head using simd reduction ---
    float local_max = 0.0f;
    for (uint d = tid; d < head_dim; d += 256) {
        float val = new_kv[src_off + d];
        local_max = max(local_max, fabs(val));
    }
    float max_abs = simd_max(local_max);

    // Compute scale (avoid div-by-zero)
    float scale = (max_abs == 0.0f) ? 1.0f : (max_abs / 127.0f);
    float inv_scale = 1.0f / scale;

    // Store scale
    if (tid == 0) {
        scales[p.seq_len * p.num_kv_heads + head_idx] = scale;
    }

    // --- Pass 2: quantize and write ---
    uint dst_off = (p.seq_len * p.num_kv_heads + head_idx) * head_dim;
    for (uint d = tid; d < head_dim; d += 256) {
        float val = new_kv[src_off + d];
        float q   = clamp(rint(val * inv_scale), -127.0f, 127.0f);
        cache_data[dst_off + d] = char(int(q));
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Batch append (prefill)
// ---------------------------------------------------------------------------

struct KvCacheAppendBatchI8Params {
    uint kv_dim;       // num_kv_heads * head_dim
    uint start;        // start position in cache
    uint count;        // number of positions to append
    uint head_dim;
    uint num_kv_heads;
};

// Dispatch: grid = (num_kv_heads, count, 1), threads = (min(head_dim, 256), 1, 1)
[[kernel]]
void kv_cache_append_batch_i8(
    device char*                         cache_data [[buffer(0)]],
    device const float*                  new_kv     [[buffer(1)]],
    device float*                        scales     [[buffer(2)]],
    constant KvCacheAppendBatchI8Params& p          [[buffer(3)]],
    uint3 tid3                           [[thread_position_in_threadgroup]],
    uint3 gid                            [[threadgroup_position_in_grid]]
) {
    uint tid      = tid3.x;
    uint head_idx = gid.x;
    uint pos_idx  = gid.y;

    if (head_idx >= p.num_kv_heads) return;
    if (pos_idx >= p.count) return;

    uint head_dim  = p.head_dim;
    // Input: new_kv is [count][kv_dim], each row is num_kv_heads * head_dim
    uint src_off   = pos_idx * p.kv_dim + head_idx * head_dim;

    // --- Pass 1: find max(|x|) across this head ---
    float local_max = 0.0f;
    for (uint d = tid; d < head_dim; d += 256) {
        float val = new_kv[src_off + d];
        local_max = max(local_max, fabs(val));
    }
    float max_abs = simd_max(local_max);

    // Compute scale
    float scale = (max_abs == 0.0f) ? 1.0f : (max_abs / 127.0f);
    float inv_scale = 1.0f / scale;

    // Store scale
    uint cache_pos = p.start + pos_idx;
    if (tid == 0) {
        scales[cache_pos * p.num_kv_heads + head_idx] = scale;
    }

    // --- Pass 2: quantize and write ---
    uint dst_off = (cache_pos * p.num_kv_heads + head_idx) * head_dim;
    for (uint d = tid; d < head_dim; d += 256) {
        float val = new_kv[src_off + d];
        float q   = clamp(rint(val * inv_scale), -127.0f, 127.0f);
        cache_data[dst_off + d] = char(int(q));
    }
}
