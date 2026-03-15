#include <metal_stdlib>
using namespace metal;

struct AttentionPrefillV3Params {
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

// K-Tiled GQA Prefill Attention (v3): load K/V in blocks of BK tokens.
//
// Same GQA structure as v2, but loads BK=16 K/V tokens at a time into
// threadgroup memory for better coalescing and L1/L2 cache utilisation.
//
// Dispatch:
//   grid    = (num_kv_heads, seq_len, 1)
//   threads = (heads_per_kv * 32, 1, 1)
// Threadgroup memory:
//   2 * BK * head_dim * sizeof(float) bytes.

constant constexpr uint BK_V3 = 16;

[[kernel]]
void attention_prefill_v3_tiled(
    device const float*                q         [[buffer(0)]],
    device const half*                 k_cache   [[buffer(1)]],
    device const half*                 v_cache   [[buffer(2)]],
    device float*                      output    [[buffer(3)]],
    constant AttentionPrefillV3Params&  p         [[buffer(4)]],
    threadgroup float*                 shared_mem [[threadgroup(0)]],
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

    threadgroup float* shared_k = shared_mem;
    threadgroup float* shared_v = shared_mem + BK_V3 * head_dim;

    // Load Q into registers.
    float q_reg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint qi = 0;
    for (uint d = lane; d < head_dim; d += 32) {
        q_reg[qi++] = q[q_offset + d];
    }

    // Online softmax accumulators.
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    for (uint t_base = 0; t_base < valid_len; t_base += BK_V3) {
        uint tile_len = min(BK_V3, valid_len - t_base);

        // Phase 1: cooperative load K + V tile into shared memory.
        uint total_elems = tile_len * head_dim;
        for (uint i = tid; i < total_elems; i += tg_threads) {
            uint tt = i / head_dim;
            uint dd = i % head_dim;
            uint kv_offset = ((t_base + tt) * num_kv_heads + kv_head) * head_dim + dd;
            shared_k[tt * head_dim + dd] = float(k_cache[kv_offset]);
            shared_v[tt * head_dim + dd] = float(v_cache[kv_offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: per-warp dot products + online softmax + V accumulation.
        for (uint ti = 0; ti < tile_len; ti++) {
            float dot_val = 0.0f;
            uint ai = 0;
            for (uint d = lane; d < head_dim; d += 32) {
                dot_val += q_reg[ai++] * shared_k[ti * head_dim + d];
            }
            float score = simd_sum(dot_val) * scale;

            float new_max = max(max_val, score);
            float correction = exp(max_val - new_max);
            float weight = exp(score - new_max);
            sum_exp = sum_exp * correction + weight;

            ai = 0;
            for (uint d = lane; d < head_dim; d += 32) {
                acc[ai] = acc[ai] * correction + weight * shared_v[ti * head_dim + d];
                ai++;
            }

            max_val = new_max;
        }
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
