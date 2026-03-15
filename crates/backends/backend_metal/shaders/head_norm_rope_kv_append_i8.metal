#include <metal_stdlib>
using namespace metal;

#define SUBGROUP_SIZE 32u

struct HeadNormRopeKvAppendI8Params {
    uint  num_heads;
    uint  head_dim;
    uint  half_dim;
    uint  kv_dim;
    uint  cache_pos;
    uint  rope_pos;
    float eps;
    uint  skip_norm; // 1 = skip RMS norm (Mistral3), 0 = apply norm (Qwen3)
};

// Fused per-head RMS norm + RoPE + INT8 KV cache append for K and V vectors (decode only).
//
// Same as head_norm_rope_kv_append but quantizes to INT8 with per-head scales.
//
// Per-head:
//   1. RMS norm the K head vector
//   2. Apply RoPE to first half_dim pairs
//   3. Compute abs-max scale, quantize K → int8, write to K cache + K scales
//   4. Compute abs-max scale for V, quantize V → int8, write to V cache + V scales
//
// Dispatch: (num_heads, 1, 1) threadgroups of 32 threads.
//
// Buffers:
//   k_data    : [num_heads, head_dim] - K vector (modified in-place with normed+RoPE result)
//   weight    : [head_dim]            - per-head norm weight
//   cos_cache : [max_pos, half_dim]   - precomputed cosine
//   sin_cache : [max_pos, half_dim]   - precomputed sine
//   k_cache   : [max_tokens, kv_dim]  - INT8 K cache
//   k_scales  : [max_tokens, num_heads] - K scales (f32 per position per head)
//   v_data    : [num_heads, head_dim] - V vector (read-only)
//   v_cache   : [max_tokens, kv_dim]  - INT8 V cache
//   v_scales  : [max_tokens, num_heads] - V scales
[[kernel]]
void head_norm_rope_kv_append_i8(
    device float*        k_data    [[buffer(0)]],
    device const float*  weight    [[buffer(1)]],
    device const float*  cos_cache [[buffer(2)]],
    device const float*  sin_cache [[buffer(3)]],
    device char*         k_cache   [[buffer(4)]],
    device float*        k_scales  [[buffer(5)]],
    device const float*  v_data    [[buffer(6)]],
    device char*         v_cache   [[buffer(7)]],
    device float*        v_scales  [[buffer(8)]],
    constant HeadNormRopeKvAppendI8Params& p [[buffer(9)]],
    uint gid                       [[threadgroup_position_in_grid]],
    uint lane                      [[thread_position_in_threadgroup]]
) {
    uint h = gid;
    if (h >= p.num_heads) return;

    uint head_offset = h * p.head_dim;
    uint cache_row = p.cache_pos * p.kv_dim;

    threadgroup float normed[256];

    if (p.skip_norm) {
        // No QK norm (Mistral3): copy data directly
        for (uint i = lane; i < p.head_dim; i += SUBGROUP_SIZE) {
            normed[i] = k_data[head_offset + i];
        }
    } else {
        // Step 1: Compute sum of squares for RMS norm
        float sum_sq = 0.0f;
        for (uint i = lane; i < p.head_dim; i += SUBGROUP_SIZE) {
            float v = k_data[head_offset + i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        float inv_rms = rsqrt(sum_sq / float(p.head_dim) + p.eps);

        // Step 2: Normalize into threadgroup memory
        for (uint i = lane; i < p.head_dim; i += SUBGROUP_SIZE) {
            normed[i] = k_data[head_offset + i] * inv_rms * weight[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Apply RoPE and store results back to normed[]
    uint cos_sin_offset = p.rope_pos * p.half_dim;
    for (uint j = lane; j < p.half_dim; j += SUBGROUP_SIZE) {
        float c = cos_cache[cos_sin_offset + j];
        float s = sin_cache[cos_sin_offset + j];
        float x0 = normed[j];
        float x1 = normed[p.half_dim + j];
        float r0 = x0 * c - x1 * s;
        float r1 = x0 * s + x1 * c;

        // Write back to k_data (in-place) and normed
        k_data[head_offset + j]              = r0;
        k_data[head_offset + p.half_dim + j] = r1;
        normed[j]              = r0;
        normed[p.half_dim + j] = r1;
    }
    // Non-RoPE dimensions
    for (uint i = lane + p.half_dim * 2; i < p.head_dim; i += SUBGROUP_SIZE) {
        k_data[head_offset + i] = normed[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Quantize K → int8 with per-head abs-max scale
    float k_max_abs = 0.0f;
    for (uint i = lane; i < p.head_dim; i += SUBGROUP_SIZE) {
        k_max_abs = max(k_max_abs, fabs(normed[i]));
    }
    k_max_abs = simd_max(k_max_abs);
    float k_scale = (k_max_abs > 0.0f) ? (k_max_abs / 127.0f) : 1.0f;
    float k_inv_scale = 1.0f / k_scale;

    if (lane == 0) {
        k_scales[p.cache_pos * p.num_heads + h] = k_scale;
    }

    for (uint i = lane; i < p.head_dim; i += SUBGROUP_SIZE) {
        float q = rint(normed[i] * k_inv_scale);
        q = clamp(q, -127.0f, 127.0f);
        k_cache[cache_row + head_offset + i] = char(int(q));
    }

    // Step 5: Quantize V → int8
    float v_max_abs = 0.0f;
    for (uint i = lane; i < p.head_dim; i += SUBGROUP_SIZE) {
        v_max_abs = max(v_max_abs, fabs(v_data[head_offset + i]));
    }
    v_max_abs = simd_max(v_max_abs);
    float v_scale = (v_max_abs > 0.0f) ? (v_max_abs / 127.0f) : 1.0f;
    float v_inv_scale = 1.0f / v_scale;

    if (lane == 0) {
        v_scales[p.cache_pos * p.num_heads + h] = v_scale;
    }

    for (uint i = lane; i < p.head_dim; i += SUBGROUP_SIZE) {
        float q = rint(v_data[head_offset + i] * v_inv_scale);
        q = clamp(q, -127.0f, 127.0f);
        v_cache[cache_row + head_offset + i] = char(int(q));
    }
}
