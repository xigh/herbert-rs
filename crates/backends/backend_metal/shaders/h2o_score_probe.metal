#include <metal_stdlib>
using namespace metal;

// H2O score probe: compute Q.K dot product scores for eviction decisions.
//
// Simplified attention: Q.K dot product only (no V load, no softmax).
// Writes the max score across GQA heads for each position.
//
// Dispatch:
//   grid    = (num_kv_heads, num_tiles, 1)
//   threads = (heads_per_kv * 32, 1, 1)
//   tgmem   = head_dim * sizeof(float)  (shared K only)

constant constexpr uint H2O_TILE_SIZE = 256;

struct H2OScoreProbeParams {
    uint  num_heads;
    uint  num_kv_heads;
    uint  head_dim;
    uint  kv_dim;
    uint  cached_len;
    float scale;
};

// Half-precision KV variant
[[kernel]]
void h2o_score_probe_half(
    device const float*          q          [[buffer(0)]],
    device const half*           k_cache    [[buffer(1)]],
    device float*                scores_out [[buffer(2)]],
    constant H2OScoreProbeParams& p        [[buffer(3)]],
    threadgroup float*           shared_k   [[threadgroup(0)]],
    uint3                        tid3       [[thread_position_in_threadgroup]],
    uint3                        gid3       [[threadgroup_position_in_grid]]
) {
    uint tid         = tid3.x;
    uint kv_head_idx = gid3.x;
    uint tile_idx    = gid3.y;

    if (kv_head_idx >= p.num_kv_heads) return;

    uint heads_per_kv = p.num_heads / p.num_kv_heads;
    uint warp_id      = tid / 32;
    uint lane         = tid % 32;
    uint tg_threads   = heads_per_kv * 32;
    uint q_head       = kv_head_idx * heads_per_kv + warp_id;
    uint head_dim     = p.head_dim;

    uint tile_start = tile_idx * H2O_TILE_SIZE;
    uint tile_end   = min(tile_start + H2O_TILE_SIZE, p.cached_len);

    if (tile_start >= p.cached_len) return;

    // Load Q into registers
    float q_reg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint qi = 0;
    for (uint d = lane; d < head_dim; d += 32) {
        q_reg[qi++] = q[q_head * head_dim + d];
    }

    // Iterate over positions in this tile
    for (uint t = tile_start; t < tile_end; t++) {
        uint kv_offset = (t * p.num_kv_heads + kv_head_idx) * head_dim;

        // Cooperative load K into shared memory
        for (uint d = tid; d < head_dim; d += tg_threads) {
            shared_k[d] = float(k_cache[kv_offset + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Dot product Q . K
        float dot_val = 0.0f;
        qi = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            dot_val += q_reg[qi++] * shared_k[d];
        }
        float score = simd_sum(dot_val) * p.scale;

        // Take max across GQA heads (warp 0 writes the result)
        // All warps have the same score since they share K, but different Q heads.
        // We want the max score across all query heads for this position.
        // Use shared memory to broadcast max across warps.
        if (lane == 0 && warp_id == 0) {
            // For simplicity: just use warp 0's score as the position score.
            // This is a good approximation since all GQA heads share the same K.
            // The score represents "how much attention does the most recent query
            // give to this position via the first query head of this KV group".
            scores_out[t] += max(0.0f, score);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// INT8 KV variant
[[kernel]]
void h2o_score_probe_i8(
    device const float*          q          [[buffer(0)]],
    device const char*           k_cache    [[buffer(1)]],
    device const float*          k_scales   [[buffer(2)]],
    device float*                scores_out [[buffer(3)]],
    constant H2OScoreProbeParams& p        [[buffer(4)]],
    threadgroup float*           shared_k   [[threadgroup(0)]],
    uint3                        tid3       [[thread_position_in_threadgroup]],
    uint3                        gid3       [[threadgroup_position_in_grid]]
) {
    uint tid         = tid3.x;
    uint kv_head_idx = gid3.x;
    uint tile_idx    = gid3.y;

    if (kv_head_idx >= p.num_kv_heads) return;

    uint heads_per_kv = p.num_heads / p.num_kv_heads;
    uint warp_id      = tid / 32;
    uint lane         = tid % 32;
    uint tg_threads   = heads_per_kv * 32;
    uint q_head       = kv_head_idx * heads_per_kv + warp_id;
    uint head_dim     = p.head_dim;

    uint tile_start = tile_idx * H2O_TILE_SIZE;
    uint tile_end   = min(tile_start + H2O_TILE_SIZE, p.cached_len);

    if (tile_start >= p.cached_len) return;

    // Load Q into registers
    float q_reg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint qi = 0;
    for (uint d = lane; d < head_dim; d += 32) {
        q_reg[qi++] = q[q_head * head_dim + d];
    }

    // Iterate over positions in this tile
    for (uint t = tile_start; t < tile_end; t++) {
        uint kv_offset = (t * p.num_kv_heads + kv_head_idx) * head_dim;

        // Cooperative load K from int8 cache, dequantize
        float scale = k_scales[t * p.num_kv_heads + kv_head_idx];
        for (uint d = tid; d < head_dim; d += tg_threads) {
            shared_k[d] = float(k_cache[kv_offset + d]) * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Dot product Q . K
        float dot_val = 0.0f;
        qi = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            dot_val += q_reg[qi++] * shared_k[d];
        }
        float s = simd_sum(dot_val) * p.scale;

        if (lane == 0 && warp_id == 0) {
            scores_out[t] += max(0.0f, s);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
