#include <metal_stdlib>
using namespace metal;

struct AttentionDecodeParams {
    uint  num_heads;
    uint  num_kv_heads;
    uint  head_dim;
    uint  kv_dim;
    uint  cached_len;
    float scale;
};

// Decode attention specialized for GQA.
//
// One threadgroup handles all query heads that share a single KV head.
// K/V are loaded once per position into threadgroup memory and reused by each
// warp's query head, which avoids rereading the same KV cache for every head.
//
// Dispatch:
//   grid.x    = num_kv_heads
//   threads.x = 32 * heads_per_kv
// Threadgroup memory:
//   2 * head_dim * sizeof(float) bytes for shared K and V.
[[kernel]]
void attention_decode_gqa(
    device const float*         q         [[buffer(0)]],
    device const half*          k_cache   [[buffer(1)]],
    device const half*          v_cache   [[buffer(2)]],
    device float*               output    [[buffer(3)]],
    constant AttentionDecodeParams& p     [[buffer(4)]],
    threadgroup float*          shared_kv [[threadgroup(0)]],
    uint                        tid       [[thread_position_in_threadgroup]],
    uint                        gid       [[threadgroup_position_in_grid]]
) {
    uint num_heads = p.num_heads;
    uint num_kv_heads = p.num_kv_heads;
    uint head_dim = p.head_dim;
    uint cached_len = p.cached_len;
    float scale = p.scale;

    if (gid >= num_kv_heads) return;

    uint heads_per_kv = num_heads / num_kv_heads;
    uint warp_id = tid / 32;
    uint lane = tid % 32;
    uint tg_threads = heads_per_kv * 32;
    uint q_head = gid * heads_per_kv + warp_id;
    uint q_offset = q_head * head_dim;

    threadgroup float* shared_k = shared_kv;
    threadgroup float* shared_v = shared_kv + head_dim;

    float q_reg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    uint q_idx = 0;
    for (uint d = lane; d < head_dim; d += 32) {
        q_reg[q_idx++] = q[q_offset + d];
    }

    for (uint t = 0; t < cached_len; t++) {
        uint kv_offset = (t * num_kv_heads + gid) * head_dim;

        for (uint d = tid; d < head_dim; d += tg_threads) {
            shared_k[d] = float(k_cache[kv_offset + d]);
            shared_v[d] = float(v_cache[kv_offset + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float dot_val = 0.0f;
        uint acc_idx = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            dot_val += q_reg[acc_idx++] * shared_k[d];
        }
        float score = simd_sum(dot_val) * scale;

        float new_max = max(max_val, score);
        float correction = exp(max_val - new_max);
        float weight = exp(score - new_max);
        sum_exp = sum_exp * correction + weight;

        acc_idx = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            acc[acc_idx] = acc[acc_idx] * correction + weight * shared_v[d];
            acc_idx++;
        }

        max_val = new_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    uint out_offset = q_head * head_dim;
    uint out_idx = 0;
    for (uint d = lane; d < head_dim; d += 32) {
        output[out_offset + d] = acc[out_idx++] * inv_sum;
    }
}
