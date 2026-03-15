// Attention prefill v6: BQ=32 query tile with simdgroup_matrix Q·K^T.
//
// Same algorithm as v4 but BQ=32 (vs 8): 4× better K/V reuse.
// Q·K^T computed via simdgroup_matrix 8×8 (same as v4).
// V accumulation uses scalar online softmax.
//
// Requires: MSL 4.0
//
// Dispatch:
//   grid:    (num_kv_heads, ceil(seq_len / 32), 1)
//   threads: (heads_per_kv * 32, 1, 1)
// Threadgroup memory (head_dim=128, heads_per_kv=2):
//   shared_q : 2 * 32 * 128 * 2 = 16384 B
//   shared_k : 8 * 128 * 2      = 2048 B
//   shared_v : 8 * 128 * 2      = 2048 B
//   shared_s : 2 * 32 * 8 * 4   = 2048 B
//   Total ≈ 22 KB (fits in 32 KB tg mem)

#if __METAL_VERSION__ >= 400

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct AttentionPrefillV6Params {
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

constant constexpr uint BQ_V6 = 32;
constant constexpr uint BK_V6 = 8;  // K tile same as v4 for simdgroup_matrix 8×8

[[kernel]]
void attention_prefill_v6_coop32(
    device const float*                q          [[buffer(0)]],
    device const half*                 k_cache    [[buffer(1)]],
    device const half*                 v_cache    [[buffer(2)]],
    device float*                      output     [[buffer(3)]],
    constant AttentionPrefillV6Params&  p          [[buffer(4)]],
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
    uint q_base  = q_tile * BQ_V6;

    if (kv_head >= num_kv_heads || q_base >= seq_len) return;

    uint heads_per_kv  = num_heads / num_kv_heads;
    uint warp_id       = tid / 32;
    uint lane          = tid % 32;
    uint tg_threads    = heads_per_kv * 32;
    uint q_head        = kv_head * heads_per_kv + warp_id;
    uint actual_bq     = min(BQ_V6, seq_len - q_base);
    uint dims_per_lane = head_dim / 32;

    // ---- Shared memory layout ----
    threadgroup half*  shared_q = (threadgroup half*)shared_raw;
    threadgroup half*  shared_k = shared_q + heads_per_kv * BQ_V6 * head_dim;
    threadgroup half*  shared_v = shared_k + BK_V6 * head_dim;
    threadgroup float* shared_s = (threadgroup float*)(shared_v + BK_V6 * head_dim);

    // ---- Load Q into shared as half (once) ----
    uint total_q_slots = heads_per_kv * BQ_V6 * head_dim;
    for (uint i = tid; i < total_q_slots; i += tg_threads) {
        uint w   = i / (BQ_V6 * head_dim);
        uint rem = i % (BQ_V6 * head_dim);
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

    // ---- Per-warp online softmax state for BQ_V6 positions ----
    // We process query positions in 4 sub-blocks of 8 (matching simdgroup_matrix 8×8).
    float acc[BQ_V6][8];  // max head_dim=256 → 256/32=8 slots per lane
    float row_max[BQ_V6];
    float row_sum[BQ_V6];
    for (uint i = 0; i < BQ_V6; i++) {
        for (uint j = 0; j < 8; j++) acc[i][j] = 0.0f;
        row_max[i] = -INFINITY;
        row_sum[i] = 0.0f;
    }

    uint max_valid = min(start_pos + q_base + actual_bq, cached_len);

    // ---- K-tile loop ----
    for (uint t_base = 0; t_base < max_valid; t_base += BK_V6) {
        uint tile_len = min(BK_V6, max_valid - t_base);

        // Phase 1: cooperative load K + V tiles as half
        uint total_kv = BK_V6 * head_dim;
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

        // Phase 2: Q · K^T via simdgroup_matrix (per warp).
        // Process 4 sub-blocks of 8 query positions each.
        for (uint qsub = 0; qsub < 4; qsub++) {
            uint q_sub_base = qsub * 8;

            simdgroup_matrix<float, 8, 8> S_acc;
            S_acc = simdgroup_matrix<float, 8, 8>(0);

            threadgroup half* q_ptr = shared_q + warp_id * BQ_V6 * head_dim + q_sub_base * head_dim;
            for (uint sub_k = 0; sub_k < head_dim / 8; sub_k++) {
                simdgroup_matrix<half, 8, 8> A, B;
                simdgroup_load(A, q_ptr, head_dim, ulong2(sub_k * 8, 0));
                simdgroup_load(B, shared_k, head_dim, ulong2(sub_k * 8, 0), true);
                simdgroup_multiply_accumulate(S_acc, A, B, S_acc);
            }

            // Store S[8×8] to shared_s
            uint warp_s = warp_id * BQ_V6 * BK_V6 + q_sub_base * BK_V6;
            simdgroup_store(S_acc, shared_s + warp_s, BK_V6);
        }

        // Phase 3: per-row softmax + scalar V accumulation
        uint warp_s_base = warp_id * BQ_V6 * BK_V6;
        for (uint row = 0; row < actual_bq; row++) {
            uint abs_pos   = start_pos + q_base + row;
            uint row_valid = min(abs_pos + 1, cached_len);

            float tile_max = -INFINITY;
            float scores[BK_V6];
            for (uint j = 0; j < BK_V6; j++) {
                float s;
                if (j < tile_len && t_base + j < row_valid) {
                    s = shared_s[warp_s_base + row * BK_V6 + j] * scale;
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

            row_sum[row] = row_sum[row] * correction + tile_sum;
            row_max[row] = new_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Normalize and write output ----
    for (uint row = 0; row < actual_bq; row++) {
        float inv = (row_sum[row] > 0.0f) ? (1.0f / row_sum[row]) : 0.0f;
        uint out_off = ((q_base + row) * num_heads + q_head) * head_dim;
        for (uint di = 0; di < dims_per_lane; di++) {
            uint d = lane + di * 32;
            output[out_off + d] = acc[row][di] * inv;
        }
    }
}

#endif // __METAL_VERSION__ >= 400
