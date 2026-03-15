#include <metal_stdlib>
using namespace metal;

#define SUBGROUP_SIZE 32u

struct HeadNormRopeBatchParams {
    uint  num_heads;
    uint  head_dim;
    uint  half_dim;
    uint  seq_len;
    uint  start_pos;
    float eps;
    uint  skip_norm; // 1 = skip RMS norm (Mistral3), 0 = apply norm (Qwen3)
};

// Fused per-head RMS norm + RoPE for prefill (batch of seq_len tokens).
//
// Combines head_rms_norm_batch + rope_batch into a single kernel:
//   1. Compute RMS norm of each head vector
//   2. Normalize in-place with weight
//   3. Apply RoPE rotation to first half_dim pairs
//
// Dispatch: (seq_len * num_heads, 1, 1) threadgroups of 32 threads.
// Each threadgroup processes one head of one token.
//
// Buffers:
//   data      : [seq_len, num_heads, head_dim] - Q or K vectors (modified in-place)
//   weight    : [head_dim]                     - per-head norm weight
//   cos_cache : [max_pos, half_dim]            - precomputed cosine
//   sin_cache : [max_pos, half_dim]            - precomputed sine
[[kernel]]
void head_norm_rope_batch(
    device float*        data      [[buffer(0)]],
    device const float*  weight    [[buffer(1)]],
    device const float*  cos_cache [[buffer(2)]],
    device const float*  sin_cache [[buffer(3)]],
    constant HeadNormRopeBatchParams& p [[buffer(4)]],
    uint gid                       [[threadgroup_position_in_grid]],
    uint lane                      [[thread_position_in_threadgroup]]
) {
    uint total_heads = p.seq_len * p.num_heads;
    if (gid >= total_heads) return;

    uint token = gid / p.num_heads;
    uint h     = gid % p.num_heads;
    uint offset = (token * p.num_heads + h) * p.head_dim;

    threadgroup float normed[256]; // head_dim up to 256 (Qwen3 = 128)

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

        // Step 2: Normalize in-place and store to threadgroup memory for RoPE
        for (uint i = lane; i < p.head_dim; i += SUBGROUP_SIZE) {
            normed[i] = data[offset + i] * inv_rms * weight[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Apply RoPE to first half_dim pairs
    uint abs_pos = p.start_pos + token;
    uint cos_sin_offset = abs_pos * p.half_dim;
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
