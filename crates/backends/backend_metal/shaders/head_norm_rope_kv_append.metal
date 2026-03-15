#include <metal_stdlib>
using namespace metal;

#define SUBGROUP_SIZE 32u

struct HeadNormRopeKvAppendParams {
    uint  num_heads;
    uint  head_dim;
    uint  half_dim;
    uint  kv_dim;
    uint  cache_pos;
    uint  rope_pos;   // RoPE cos/sin lookup position (== cache_pos for text-only)
    float eps;
    uint  skip_norm;  // 1 = skip RMS norm (Mistral3), 0 = apply norm (Qwen3)
};

// Fused per-head RMS norm + RoPE + KV cache append for K and V vectors (decode only).
//
// Per-head:
//   1. RMS norm the K head vector
//   2. Apply RoPE to first half_dim pairs
//   3. Write result to K cache at cache_pos
//   4. Copy V head vector to V cache at cache_pos
//
// Eliminates 4 separate dispatches (head_rms_norm + rope_kv_append + kv_cache_append).
//
// Dispatch: (num_heads, 1, 1) threadgroups of 32 threads.
//
// Buffers:
//   k_data    : [num_heads, head_dim] - K vector (modified in-place)
//   weight    : [head_dim]            - per-head norm weight
//   cos_cache : [max_pos, half_dim]   - precomputed cosine
//   sin_cache : [max_pos, half_dim]   - precomputed sine
//   k_cache   : [max_tokens, kv_dim]  - K cache to append to
//   v_data    : [num_heads, head_dim] - V vector (read-only)
//   v_cache   : [max_tokens, kv_dim]  - V cache to append to
[[kernel]]
void head_norm_rope_kv_append(
    device float*        k_data    [[buffer(0)]],
    device const float*  weight    [[buffer(1)]],
    device const float*  cos_cache [[buffer(2)]],
    device const float*  sin_cache [[buffer(3)]],
    device half*         k_cache   [[buffer(4)]],
    device const float*  v_data    [[buffer(5)]],
    device half*         v_cache   [[buffer(6)]],
    constant HeadNormRopeKvAppendParams& p [[buffer(7)]],
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

    // Step 3: Apply RoPE + write to K cache + write back to k_data
    uint cos_sin_offset = p.rope_pos * p.half_dim;
    for (uint j = lane; j < p.half_dim; j += SUBGROUP_SIZE) {
        float c = cos_cache[cos_sin_offset + j];
        float s = sin_cache[cos_sin_offset + j];
        float x0 = normed[j];
        float x1 = normed[p.half_dim + j];
        float r0 = x0 * c - x1 * s;
        float r1 = x0 * s + x1 * c;

        // Write back to k_data (in-place)
        k_data[head_offset + j]              = r0;
        k_data[head_offset + p.half_dim + j] = r1;

        // Write to K cache (f32 → half)
        k_cache[cache_row + head_offset + j]              = half(r0);
        k_cache[cache_row + head_offset + p.half_dim + j] = half(r1);
    }

    // Copy non-RoPE dimensions (if any)
    for (uint i = lane + p.half_dim * 2; i < p.head_dim; i += SUBGROUP_SIZE) {
        float v = normed[i];
        k_data[head_offset + i] = v;
        k_cache[cache_row + head_offset + i] = half(v);
    }

    // Step 4: Copy V head vector to V cache (f32 → half)
    for (uint i = lane; i < p.head_dim; i += SUBGROUP_SIZE) {
        v_cache[cache_row + head_offset + i] = half(v_data[head_offset + i]);
    }
}
