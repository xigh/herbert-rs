#include <metal_stdlib>
using namespace metal;

struct AttentionPrefillV5Params {
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

// Half-precision KV cache attention (v5).
//
// Identical algorithm to v4 (simdgroup_matrix Q·K^T + scalar V accumulation)
// but reads K/V cache stored as float16 instead of float32. This halves DRAM
// bandwidth for KV reads and eliminates the float→half conversion in shared
// memory loads.
//
// REQUIRES: KV cache allocated and written as half (kv_cache.rs changes).
//
// Dispatch: same as v4.
//   grid    = (num_kv_heads, ceil(seq_len / 8), 1)
//   threads = (heads_per_kv * 32, 1, 1)
// Threadgroup memory: same as v4 (~13 KB for head_dim=128, heads_per_kv=4).

constant constexpr uint BQ_V5 = 8;
constant constexpr uint BK_V5 = 8;

[[kernel]]
void attention_prefill_v5_half_kv(
    device const float*                q          [[buffer(0)]],
    device const half*                 k_cache    [[buffer(1)]],
    device const half*                 v_cache    [[buffer(2)]],
    device float*                      output     [[buffer(3)]],
    constant AttentionPrefillV5Params&  p          [[buffer(4)]],
    threadgroup float*                 shared_raw [[threadgroup(0)]],
    uint3                              tid3       [[thread_position_in_threadgroup]],
    uint3                              gid3       [[threadgroup_position_in_grid]]
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
    uint q_tile  = gid3.y;
    uint q_base  = q_tile * BQ_V5;

    if (kv_head >= num_kv_heads || q_base >= seq_len) return;

    uint heads_per_kv  = num_heads / num_kv_heads;
    uint warp_id       = tid / 32;
    uint lane          = tid % 32;
    uint tg_threads    = heads_per_kv * 32;
    uint q_head        = kv_head * heads_per_kv + warp_id;
    uint actual_bq     = min(BQ_V5, seq_len - q_base);
    uint dims_per_lane = head_dim / 32;

    // Shared memory layout (same as v4).
    threadgroup half*  shared_q = (threadgroup half*)shared_raw;
    threadgroup half*  shared_k = shared_q + heads_per_kv * BQ_V5 * head_dim;
    threadgroup half*  shared_v = shared_k + BK_V5 * head_dim;
    threadgroup float* shared_s = (threadgroup float*)(shared_v + BK_V5 * head_dim);

    // Load Q into shared as half (once).
    uint total_q_slots = heads_per_kv * BQ_V5 * head_dim;
    for (uint i = tid; i < total_q_slots; i += tg_threads) {
        uint w   = i / (BQ_V5 * head_dim);
        uint rem = i % (BQ_V5 * head_dim);
        uint pi  = rem / head_dim;
        uint d   = rem % head_dim;
        if (pi < actual_bq) {
            uint qh = kv_head * heads_per_kv + w;
            shared_q[i] = half(q[((q_base + pi) * num_heads + qh) * head_dim + d]);
        } else {
            shared_q[i] = half(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-warp online softmax state.
    float acc[BQ_V5][8];
    float row_max[BQ_V5];
    float row_sum_v[BQ_V5];
    for (uint i = 0; i < BQ_V5; i++) {
        for (uint j = 0; j < 8; j++) acc[i][j] = 0.0f;
        row_max[i] = -INFINITY;
        row_sum_v[i] = 0.0f;
    }

    uint max_valid = min(start_pos + q_base + actual_bq, cached_len);

    for (uint t_base = 0; t_base < max_valid; t_base += BK_V5) {
        uint tile_len = min(BK_V5, max_valid - t_base);

        // Load K + V directly as half (no float→half conversion!).
        uint total_kv = BK_V5 * head_dim;
        for (uint i = tid; i < total_kv; i += tg_threads) {
            uint tt = i / head_dim;
            uint dd = i % head_dim;
            if (tt < tile_len) {
                uint kv_off = ((t_base + tt) * num_kv_heads + kv_head) * head_dim + dd;
                shared_k[i] = k_cache[kv_off];
                shared_v[i] = v_cache[kv_off];
            } else {
                shared_k[i] = half(0.0f);
                shared_v[i] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Q · K^T via simdgroup_matrix.
        simdgroup_matrix<float, 8, 8> S_acc;
        S_acc = simdgroup_matrix<float, 8, 8>(0);

        threadgroup half* q_ptr = shared_q + warp_id * BQ_V5 * head_dim;
        for (uint sub_k = 0; sub_k < head_dim / 8; sub_k++) {
            simdgroup_matrix<half, 8, 8> A, B;
            simdgroup_load(A, q_ptr, head_dim, ulong2(sub_k * 8, 0));
            simdgroup_load(B, shared_k, head_dim, ulong2(sub_k * 8, 0), true);
            simdgroup_multiply_accumulate(S_acc, A, B, S_acc);
        }

        uint warp_s = warp_id * BQ_V5 * BK_V5;
        simdgroup_store(S_acc, shared_s + warp_s, BK_V5);

        // Per-row softmax + V accumulation.
        for (uint row = 0; row < actual_bq; row++) {
            uint abs_pos   = start_pos + q_base + row;
            uint row_valid = min(abs_pos + 1, cached_len);

            float tile_max = -INFINITY;
            float scores[BK_V5];
            for (uint j = 0; j < BK_V5; j++) {
                float s;
                if (j < tile_len && t_base + j < row_valid) {
                    s = shared_s[warp_s + row * BK_V5 + j] * scale;
                } else {
                    s = -INFINITY;
                }
                scores[j] = s;
                tile_max = max(tile_max, s);
            }

            float new_max    = max(row_max[row], tile_max);
            float correction = exp(row_max[row] - new_max);

            for (uint di = 0; di < dims_per_lane; di++) {
                acc[row][di] *= correction;
            }

            float tile_sum = 0.0f;
            for (uint j = 0; j < tile_len; j++) {
                float w = (t_base + j < row_valid)
                          ? exp(scores[j] - new_max) : 0.0f;
                tile_sum += w;

                for (uint di = 0; di < dims_per_lane; di++) {
                    uint d = lane + di * 32;
                    acc[row][di] += w * float(shared_v[j * head_dim + d]);
                }
            }

            row_sum_v[row] = row_sum_v[row] * correction + tile_sum;
            row_max[row] = new_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output.
    for (uint row = 0; row < actual_bq; row++) {
        float inv = (row_sum_v[row] > 0.0f) ? (1.0f / row_sum_v[row]) : 0.0f;
        uint out_off = ((q_base + row) * num_heads + q_head) * head_dim;
        for (uint di = 0; di < dims_per_lane; di++) {
            uint d = lane + di * 32;
            output[out_off + d] = acc[row][di] * inv;
        }
    }
}
