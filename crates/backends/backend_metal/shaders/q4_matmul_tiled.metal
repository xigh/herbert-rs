#include <metal_stdlib>
using namespace metal;

// Tiled Q4 matmul for prefill: Y[m,n] = sum_k X[m,k] * W[n,k]
//
// Uses shared memory tiling to improve data reuse.
// TILE_M=4 tokens, TILE_N=4 output rows, TILE_K=32 (matches Q4 group_size).
// 128 threads = 4 warps; each warp handles one output row.
//
// Each threadgroup produces a TILE_M × TILE_N tile of the output matrix.
// The inner loop iterates over K in steps of TILE_K=32.
//
// Dispatch:
//   grid:  (ceil(N/4), ceil(M/4), 1)
//   threads: (128, 1, 1)

struct Q4MatmulTiledParams {
    uint M;
    uint N;
    uint K;
};

#define TILE_M 4u
#define TILE_K 32u

[[kernel]]
void q4_matmul_tiled(
    device const float*    x        [[buffer(0)]],
    device const uchar*    w_packed [[buffer(1)]],
    device const float*    scales   [[buffer(2)]],
    device float*          y        [[buffer(3)]],
    constant Q4MatmulTiledParams& p [[buffer(4)]],
    uint2                  tid2     [[thread_position_in_threadgroup]],
    uint2                  gid      [[threadgroup_position_in_grid]]
) {
    uint tid = tid2.x;
    uint M = p.M;
    uint N = p.N;
    uint K = p.K;
    uint n_groups = (K + 31) / 32;
    uint num_bytes = K / 2;

    uint warp_id = tid / 32;
    uint lane = tid % 32;

    // This threadgroup handles output rows [n_base..n_base+4), tokens [m_base..m_base+4)
    uint n_base = gid.x * 4;
    uint m_base = gid.y * TILE_M;

    // Each warp handles one output row n
    uint n = n_base + warp_id;

    // Accumulate for TILE_M tokens
    float acc[TILE_M];
    for (uint mi = 0; mi < TILE_M; mi++) {
        acc[mi] = 0.0f;
    }

    // Shared memory for input tile: TILE_M × TILE_K
    threadgroup float shared_x[TILE_M][TILE_K];

    // Iterate over K dimension in steps of TILE_K
    for (uint kk = 0; kk < K; kk += TILE_K) {
        // Cooperative load of X tile into shared memory
        // 128 threads load TILE_M * TILE_K = 128 elements (1 per thread)
        uint load_idx = tid;
        if (load_idx < TILE_M * TILE_K) {
            uint mi = load_idx / TILE_K;
            uint ki = load_idx % TILE_K;
            uint m_idx = m_base + mi;
            uint k_idx = kk + ki;
            shared_x[mi][ki] = (m_idx < M && k_idx < K) ? x[m_idx * K + k_idx] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (n < N) {
            // Scale for this group
            float s = scales[n * n_groups + kk / 32];

            // Each lane processes one byte (2 Q4 values) within the TILE_K=32 range
            // 32 lanes × 1 byte = 16 bytes = 32 nibbles = TILE_K elements
            uint byte_idx = n * num_bytes + kk / 2 + lane;
            uchar b = w_packed[byte_idx];
            float w0 = (float(b & 0xFu) - 8.0f) * s;
            float w1 = (float(b >> 4) - 8.0f) * s;

            uint ki0 = lane * 2;
            uint ki1 = lane * 2 + 1;

            // We need only 16 lanes for 32 elements (2 per lane)
            if (lane < 16) {
                for (uint mi = 0; mi < TILE_M; mi++) {
                    acc[mi] += w0 * shared_x[mi][ki0] + w1 * shared_x[mi][ki1];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reduce across lanes
    for (uint mi = 0; mi < TILE_M; mi++) {
        acc[mi] = simd_sum(acc[mi]);
    }

    // Lane 0 writes results
    if (lane == 0 && n < N) {
        for (uint mi = 0; mi < TILE_M; mi++) {
            uint m_idx = m_base + mi;
            if (m_idx < M) {
                y[m_idx * N + n] = acc[mi];
            }
        }
    }
}
