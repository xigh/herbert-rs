#include <metal_stdlib>
using namespace metal;

struct AttentionPrefillV2Params {
    uint  seq_len;
    uint  num_heads;
    uint  num_kv_heads;
    uint  head_dim;
    uint  kv_dim;
    uint  q_dim;
    uint  cached_len;
    uint  start_pos;
    float scale;
};

// GQA Prefill Attention (v2): share K/V loads across query heads.
//
// One threadgroup handles all query heads sharing a single KV head for one
// query position. K/V are loaded once into threadgroup memory and reused by
// each warp (one warp per query head), avoiding redundant DRAM reads.
//
// Dispatch:
//   grid    = (num_kv_heads, seq_len, 1)
//   threads = (heads_per_kv * 32, 1, 1)
// Threadgroup memory:
//   2 * head_dim * sizeof(float) bytes for shared K and V.
[[kernel]]
void attention_prefill_v2_gqa(
    device const float*                q         [[buffer(0)]],
    device const half*                 k_cache   [[buffer(1)]],
    device const half*                 v_cache   [[buffer(2)]],
    device float*                      output    [[buffer(3)]],
    constant AttentionPrefillV2Params&  p         [[buffer(4)]],
    threadgroup float*                 shared_kv [[threadgroup(0)]],
    uint3                              tid3      [[thread_position_in_threadgroup]],
    uint3                              gid3      [[threadgroup_position_in_grid]]
) {
    uint tid          = tid3.x;
    uint seq_len      = p.seq_len;
    uint num_heads    = p.num_heads;
    uint num_kv_heads = p.num_kv_heads;
    uint head_dim     = p.head_dim;
    uint cached_len   = p.cached_len;
    uint start_pos    = p.start_pos;
    float scale       = p.scale;

    uint kv_head = gid3.x;
    uint pos     = gid3.y;

    if (kv_head >= num_kv_heads || pos >= seq_len) return;

    uint heads_per_kv = num_heads / num_kv_heads;
    uint warp_id      = tid / 32;
    uint lane         = tid % 32;
    uint tg_threads   = heads_per_kv * 32;

    uint q_head    = kv_head * heads_per_kv + warp_id;
    uint q_offset  = (pos * num_heads + q_head) * head_dim;
    uint query_pos = start_pos + pos;
    uint valid_len = min(query_pos + 1, cached_len);

    threadgroup float* shared_k = shared_kv;
    threadgroup float* shared_v = shared_kv + head_dim;

    // Load Q into registers (per warp).
    float q_reg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint qi = 0;
    for (uint d = lane; d < head_dim; d += 32) {
        q_reg[qi++] = q[q_offset + d];
    }

    // Online softmax accumulators.
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    for (uint t = 0; t < valid_len; t++) {
        uint kv_offset = (t * num_kv_heads + kv_head) * head_dim;

        // Cooperative load K + V into shared memory (half → float).
        for (uint d = tid; d < head_dim; d += tg_threads) {
            shared_k[d] = float(k_cache[kv_offset + d]);
            shared_v[d] = float(v_cache[kv_offset + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-warp dot product Q·K.
        float dot_val = 0.0f;
        uint ai = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            dot_val += q_reg[ai++] * shared_k[d];
        }
        float score = simd_sum(dot_val) * scale;

        // Online softmax update.
        float new_max = max(max_val, score);
        float correction = exp(max_val - new_max);
        float weight = exp(score - new_max);
        sum_exp = sum_exp * correction + weight;

        // Accumulate weighted V.
        ai = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            acc[ai] = acc[ai] * correction + weight * shared_v[d];
            ai++;
        }

        max_val = new_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output.
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    uint out_offset = (pos * num_heads + q_head) * head_dim;
    uint oi = 0;
    for (uint d = lane; d < head_dim; d += 32) {
        output[out_offset + d] = acc[oi++] * inv_sum;
    }
}
