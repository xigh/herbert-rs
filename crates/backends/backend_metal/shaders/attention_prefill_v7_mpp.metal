// Attention prefill v7: MPP matmul2d for Q·K^T, BQ=32 BK=32.
//
// Uses matmul2d(32, 32, 32, false, true) for Q·K^T computation.
// 1 threadgroup per query head (not per KV head) — no GQA sharing.
// This trades K/V bandwidth for faster compute and simpler memory layout.
// The L2 cache serves redundant K/V reads across GQA heads.
//
// BQ=32: 4× more query positions than V4 → better K/V reuse per tile.
// BK=32: 4× more K positions per tile → 4× fewer K-tile iterations.
//
// Requires: MSL 4.0, head_dim divisible by 32
//
// Dispatch:
//   grid:    (num_heads, ceil(seq_len / 32), 1)
//   threads: (128, 1, 1)  — 4 simdgroups for matmul2d
//
// Threadgroup memory (head_dim=128):
//   shared_q:  4 × [32×32] half  = 8192 B  (chunked for matmul2d)
//   shared_kv: 4 × [32×32] half  = 8192 B  (K chunked → V chunked)
//   shared_s:  [32×32] float     = 4096 B  (scores from matmul2d)
//   Total: 20480 B (well within 32 KB)

#if __METAL_VERSION__ >= 400

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct AttentionPrefillV7Params {
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

constant constexpr uint BQ = 32;
constant constexpr uint BK = 32;

[[kernel]]
void attention_prefill_v7_mpp(
    device const float*                q          [[buffer(0)]],
    device const half*                 k_cache    [[buffer(1)]],
    device const half*                 v_cache    [[buffer(2)]],
    device float*                      output     [[buffer(3)]],
    constant AttentionPrefillV7Params& p          [[buffer(4)]],
    threadgroup char*                  tg_raw     [[threadgroup(0)]],
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

    uint q_head  = gid3.x;                        // which query head
    uint q_tile  = gid3.y;                         // which BQ-block of query positions
    uint q_base  = q_tile * BQ;
    uint kv_head = q_head / (num_heads / num_kv_heads);  // GQA mapping

    if (q_head >= num_heads || q_base >= seq_len) return;

    uint actual_bq  = min(BQ, seq_len - q_base);
    uint hd_tiles   = head_dim / 32;  // number of 32-wide head_dim chunks

    // ---- Threadgroup memory layout ----
    // Q and K/V stored in chunked format: hd_tiles × [32 × 32] half blocks
    // This makes each [32 × 32] block contiguous for tensor_inline.
    threadgroup half*  shared_q  = (threadgroup half*)(tg_raw);           // 8192 B
    threadgroup half*  shared_kv = (threadgroup half*)(tg_raw + 8192);    // 8192 B
    threadgroup float* shared_s  = (threadgroup float*)(tg_raw + 16384); // 4096 B

    // ---- Load Q once (chunked: hd_tiles × [32 × 32]) ----
    for (uint i = tid; i < BQ * head_dim; i += 128) {
        uint row = i / head_dim;       // query position within tile (0..31)
        uint col = i % head_dim;       // head dimension
        uint chunk = col / 32;
        uint local_col = col % 32;
        half val = 0.0h;
        if (q_base + row < seq_len) {
            val = half(q[((q_base + row) * num_heads + q_head) * head_dim + col]);
        }
        shared_q[chunk * 1024 + row * 32 + local_col] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- matmul2d setup ----
    constexpr auto desc = matmul2d_descriptor(32, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    using LeftT  = tensor<threadgroup half, extents<int32_t, 32, 32>, tensor_inline>;
    using RightT = tensor<threadgroup half, extents<int32_t, 32, 32>, tensor_inline>;

    // ---- Per-query online softmax state ----
    // Each of 128 threads handles 1 V dimension (head_dim=128)
    // Register arrays: acc[BQ], row_max[BQ], row_sum[BQ]
    float acc[BQ];
    float row_max[BQ];
    float row_sum[BQ];
    for (uint i = 0; i < BQ; i++) {
        acc[i] = 0.0f;
        row_max[i] = -INFINITY;
        row_sum[i] = 0.0f;
    }

    uint d = tid;  // V dimension this thread handles (0..127 for head_dim=128)
    uint max_valid = min(start_pos + q_base + actual_bq, cached_len);

    // ---- K-tile loop ----
    for (uint t_base = 0; t_base < max_valid; t_base += BK) {
        uint tile_len = min(BK, max_valid - t_base);

        // Phase 1: Load K tile (chunked: hd_tiles × [32 × 32])
        for (uint i = tid; i < BK * head_dim; i += 128) {
            uint row = i / head_dim;       // K position within tile
            uint col = i % head_dim;       // head dimension
            uint chunk = col / 32;
            uint local_col = col % 32;
            half val = 0.0h;
            if (row < tile_len) {
                uint kv_off = ((t_base + row) * num_kv_heads + kv_head) * head_dim + col;
                val = k_cache[kv_off];
            }
            shared_kv[chunk * 1024 + row * 32 + local_col] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: Q·K^T via matmul2d — S[32×32] = Q[32×hd] × K[32×hd]^T
        auto cT = op.get_destination_cooperative_tensor<LeftT, RightT, float>();
        #pragma clang loop unroll(full)
        for (unsigned short ci = 0; ci < cT.get_capacity(); ++ci) {
            if (cT.is_valid_element(ci)) cT[ci] = 0;
        }

        for (uint hd = 0; hd < hd_tiles; hd++) {
            auto Q_tile = LeftT(shared_q + hd * 1024, extents<int32_t, 32, 32>());
            auto K_tile = RightT(shared_kv + hd * 1024, extents<int32_t, 32, 32>());
            op.run(Q_tile, K_tile, cT);
        }

        // Store S[32×32] to shared_s
        auto s_tensor = tensor<threadgroup float, extents<int32_t, 32, 32>, tensor_inline>(
            shared_s, extents<int32_t, 32, 32>());
        cT.store(s_tensor);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3: Load V tile (chunked, overwriting K)
        for (uint i = tid; i < BK * head_dim; i += 128) {
            uint row = i / head_dim;
            uint col = i % head_dim;
            uint chunk = col / 32;
            uint local_col = col % 32;
            half val = 0.0h;
            if (row < tile_len) {
                uint kv_off = ((t_base + row) * num_kv_heads + kv_head) * head_dim + col;
                val = v_cache[kv_off];
            }
            shared_kv[chunk * 1024 + row * 32 + local_col] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 4: Online softmax + V accumulation
        // Each thread handles V dimension d = tid
        uint v_chunk = d / 32;
        uint v_local = d % 32;

        for (uint row = 0; row < actual_bq; row++) {
            uint abs_pos = start_pos + q_base + row;
            uint row_valid = min(abs_pos + 1, cached_len);

            // Find tile max
            float tile_max = -INFINITY;
            for (uint j = 0; j < BK; j++) {
                float s;
                if (j < tile_len && t_base + j < row_valid) {
                    s = shared_s[row * 32 + j] * scale;
                } else {
                    s = -INFINITY;
                }
                tile_max = max(tile_max, s);
            }

            // Online softmax correction
            float new_max = max(row_max[row], tile_max);
            float correction = exp(row_max[row] - new_max);
            acc[row] *= correction;

            // Accumulate weighted V
            float tile_sum = 0.0f;
            for (uint j = 0; j < tile_len; j++) {
                float s;
                if (t_base + j < row_valid) {
                    s = shared_s[row * 32 + j] * scale;
                } else {
                    s = -INFINITY;
                }
                float w = exp(s - new_max);
                tile_sum += w;

                if (d < head_dim) {
                    acc[row] += w * float(shared_kv[v_chunk * 1024 + j * 32 + v_local]);
                }
            }

            row_sum[row] = row_sum[row] * correction + tile_sum;
            row_max[row] = new_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Normalize and write output ----
    if (d < head_dim) {
        for (uint row = 0; row < actual_bq; row++) {
            if (q_base + row < seq_len) {
                float inv = (row_sum[row] > 0.0f) ? (1.0f / row_sum[row]) : 0.0f;
                uint out_off = ((q_base + row) * num_heads + q_head) * head_dim + d;
                output[out_off] = acc[row] * inv;
            }
        }
    }
}

#endif // __METAL_VERSION__ >= 400
