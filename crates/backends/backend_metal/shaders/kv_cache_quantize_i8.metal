#include <metal_stdlib>
using namespace metal;

// Bulk quantize half-precision KV cache → INT8 with per-position per-head scales.
//
// Used after prefill to convert the half KV cache (used by prefill attention) to
// INT8 (used by decode attention for 2× bandwidth reduction).
//
// Dispatch: grid = (num_kv_heads, count, 1), threads = (32, 1, 1)

struct KvQuantizeI8Params {
    uint kv_dim;       // num_kv_heads * head_dim
    uint start;        // starting position
    uint count;        // number of positions
    uint head_dim;
    uint num_kv_heads;
};

[[kernel]]
void kv_cache_quantize_half_to_i8(
    device char*         cache_i8   [[buffer(0)]],
    device const half*   cache_half [[buffer(1)]],
    device float*        scales     [[buffer(2)]],
    constant KvQuantizeI8Params& p  [[buffer(3)]],
    uint3                tid3       [[thread_position_in_threadgroup]],
    uint3                gid3       [[threadgroup_position_in_grid]],
    uint3                tg_size    [[threads_per_threadgroup]]
) {
    uint head_idx = gid3.x;
    uint pos_idx  = gid3.y;
    uint tid      = tid3.x;
    uint tg_w     = tg_size.x;

    if (head_idx >= p.num_kv_heads || pos_idx >= p.count) return;

    uint pos = p.start + pos_idx;
    uint base = (pos * p.num_kv_heads + head_idx) * p.head_dim;

    // Pass 1: find abs-max across head_dim elements
    float max_abs = 0.0f;
    for (uint d = tid; d < p.head_dim; d += tg_w) {
        max_abs = max(max_abs, fabs(float(cache_half[base + d])));
    }
    max_abs = simd_max(max_abs);

    float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    // Store scale
    if (tid == 0) {
        scales[pos * p.num_kv_heads + head_idx] = scale;
    }

    // Pass 2: quantize and store
    for (uint d = tid; d < p.head_dim; d += tg_w) {
        float v = float(cache_half[base + d]) * inv_scale;
        v = clamp(rint(v), -127.0f, 127.0f);
        cache_i8[base + d] = char(int(v));
    }
}
