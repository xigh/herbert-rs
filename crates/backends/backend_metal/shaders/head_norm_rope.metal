#include <metal_stdlib>
using namespace metal;

#define SUBGROUP_SIZE 32u

struct HeadNormRopeParams {
    uint  num_heads;
    uint  head_dim;
    uint  half_dim;
    uint  pos;
    float eps;
    uint  skip_norm; // 1 = skip RMS norm (Mistral3), 0 = apply norm (Qwen3)
};

// Fused per-head RMS norm + RoPE for Q vectors (decode only).
//
// Per-head:
//   1. Compute RMS norm of head vector
//   2. Normalize: data[i] = data[i] * inv_rms * weight[i]
//   3. Apply RoPE to first half_dim pairs
//
// This eliminates 2 separate dispatches (head_rms_norm + rope_batch).
//
// Dispatch: (num_heads, 1, 1) threadgroups of 32 threads.
//
// Buffers:
//   data      : [num_heads, head_dim] - Q vector (modified in-place)
//   weight    : [head_dim]            - per-head norm weight
//   cos_cache : [max_pos, half_dim]   - precomputed cosine
//   sin_cache : [max_pos, half_dim]   - precomputed sine
[[kernel]]
void head_norm_rope(
    device float*        data      [[buffer(0)]],
    device const float*  weight    [[buffer(1)]],
    device const float*  cos_cache [[buffer(2)]],
    device const float*  sin_cache [[buffer(3)]],
    constant HeadNormRopeParams& p [[buffer(4)]],
    uint gid                       [[threadgroup_position_in_grid]],
    uint lane                      [[thread_position_in_threadgroup]]
) {
    uint h = gid;
    if (h >= p.num_heads) return;
    uint offset = h * p.head_dim;

    threadgroup float normed[256]; // head_dim up to 256

    if (p.skip_norm) {
        // No QK norm (Mistral3): copy data directly
        for (uint i = lane; i < p.head_dim; i += SUBGROUP_SIZE) {
            normed[i] = data[offset + i];
        }
    } else {
        // Step 1: Compute sum of squares for RMS norm
        float sum_sq = 0.0f;
        for (uint i = lane; i < p.head_dim; i += SUBGROUP_SIZE) {
            float v = data[offset + i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        float inv_rms = rsqrt(sum_sq / float(p.head_dim) + p.eps);

        // Step 2: Normalize in-place (store to threadgroup memory for RoPE)
        for (uint i = lane; i < p.head_dim; i += SUBGROUP_SIZE) {
            normed[i] = data[offset + i] * inv_rms * weight[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Apply RoPE to first half_dim pairs
    uint cos_sin_offset = p.pos * p.half_dim;
    for (uint j = lane; j < p.half_dim; j += SUBGROUP_SIZE) {
        float c = cos_cache[cos_sin_offset + j];
        float s = sin_cache[cos_sin_offset + j];
        float x0 = normed[j];
        float x1 = normed[p.half_dim + j];
        data[offset + j]              = x0 * c - x1 * s;
        data[offset + p.half_dim + j] = x0 * s + x1 * c;
    }

    // Copy non-RoPE dimensions (if head_dim > 2*half_dim)
    for (uint i = lane + p.half_dim * 2; i < p.head_dim; i += SUBGROUP_SIZE) {
        data[offset + i] = normed[i];
    }
}
