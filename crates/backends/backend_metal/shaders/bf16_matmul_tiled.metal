#include <metal_stdlib>
using namespace metal;

// Tiled BF16 matmul for prefill: Y[m,n] = sum_k X[m,k] * W[n,k]
//
// Uses shared memory tiling for better data reuse.
// TILE_M=4 tokens, TILE_N=4 output rows, TILE_K=64.
// 128 threads = 4 warps; each warp handles one output row.
//
// Dispatch:
//   grid:  (ceil(N/4), ceil(M/4), 1)
//   threads: (128, 1, 1)
//
// Buffers:
//   x        : [M, K]      - f32 input matrix (row-major)
//   w_packed : [N, K/2]    - BF16 weights packed as uint32 (row-major)
//   y        : [M, N]      - f32 output (row-major)

struct Bf16MatmulTiledParams {
    uint M;
    uint N;
    uint K;
};

#define TILE_M 4u
#define TILE_K 64u

[[kernel]]
void bf16_matmul_tiled(
    device const float*    x        [[buffer(0)]],
    device const uint*     w_packed [[buffer(1)]],
    device float*          y        [[buffer(2)]],
    constant Bf16MatmulTiledParams& p [[buffer(3)]],
    uint2                  tid2     [[thread_position_in_threadgroup]],
    uint2                  gid      [[threadgroup_position_in_grid]]
) {
    uint tid = tid2.x;
    uint M = p.M;
    uint N = p.N;
    uint K = p.K;
    uint num_packed = K / 2;

    uint warp_id = tid / 32;
    uint lane = tid % 32;

    uint n_base = gid.x * 4;
    uint m_base = gid.y * TILE_M;

    uint n = n_base + warp_id;

    float acc[TILE_M];
    for (uint mi = 0; mi < TILE_M; mi++) {
        acc[mi] = 0.0f;
    }

    // Shared memory for input tile
    threadgroup float shared_x[TILE_M][TILE_K];

    for (uint kk = 0; kk < K; kk += TILE_K) {
        // Cooperative load of X tile: 128 threads load TILE_M * TILE_K = 256 elements
        for (uint load_idx = tid; load_idx < TILE_M * TILE_K; load_idx += 128) {
            uint mi = load_idx / TILE_K;
            uint ki = load_idx % TILE_K;
            uint m_idx = m_base + mi;
            uint k_idx = kk + ki;
            shared_x[mi][ki] = (m_idx < M && k_idx < K) ? x[m_idx * K + k_idx] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (n < N) {
            // Each lane processes 2 packed BF16 values (= 2 elements)
            // 32 lanes × 2 = 64 elements per iteration = TILE_K
            uint packed_offset = n * num_packed + kk / 2;

            for (uint j = lane; j < TILE_K / 2; j += 32) {
                uint packed_val = w_packed[packed_offset + j];
                float w0 = as_type<float>((packed_val & 0xFFFFu) << 16);
                float w1 = as_type<float>(packed_val & 0xFFFF0000u);

                uint ki0 = j * 2;
                uint ki1 = j * 2 + 1;

                for (uint mi = 0; mi < TILE_M; mi++) {
                    acc[mi] += w0 * shared_x[mi][ki0] + w1 * shared_x[mi][ki1];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint mi = 0; mi < TILE_M; mi++) {
        acc[mi] = simd_sum(acc[mi]);
    }

    if (lane == 0 && n < N) {
        for (uint mi = 0; mi < TILE_M; mi++) {
            uint m_idx = m_base + mi;
            if (m_idx < M) {
                y[m_idx * N + n] = acc[mi];
            }
        }
    }
}
