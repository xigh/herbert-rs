// Tiled Q4 matmul for prefill using simdgroup_matrix: Y[m,n] = sum_k X[m,k] * W[n,k]
//
// 256 threads = 8 simdgroups, arranged 2x4 (2 along M, 4 along N).
// TG output: TILE_M=16 tokens x TILE_N=32 output rows.
// Each simdgroup: 8x8 sub-tile via simdgroup_multiply_accumulate.
// W is loaded once and reused by 2 simdgroup pairs along M.
//
// Dispatch:
//   grid:    (ceil(N/32), ceil(M/16), 1)
//   threads: (256, 1, 1)

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct Q4MatmulTiledParams {
    uint M;
    uint N;
    uint K;
};

[[kernel]]
void q4_matmul_tiled(
    device const float*  x        [[buffer(0)]],
    device const uchar*  w_packed [[buffer(1)]],
    device const float*  scales   [[buffer(2)]],
    device float*        y        [[buffer(3)]],
    constant Q4MatmulTiledParams& p [[buffer(4)]],
    uint2                tid2     [[thread_position_in_threadgroup]],
    uint2                gid      [[threadgroup_position_in_grid]]
) {
    const uint TILE_M = 16;
    const uint TILE_K = 32;
    const uint SG_N = 8;
    const uint SG_M = 8;
    const uint N_SG_M = 2;   // simdgroups along M
    const uint N_SG_N = 4;   // simdgroups along N
    const uint TILE_N = N_SG_N * SG_N;  // 32

    uint tid = tid2.x;
    uint M = p.M, N = p.N, K = p.K;
    uint n_groups = (K + 31) / 32;
    uint num_bytes = K / 2;

    uint sg_id = tid / 32;
    uint lane = tid % 32;
    uint sg_m = sg_id / N_SG_N;   // 0 or 1
    uint sg_n = sg_id % N_SG_N;   // 0..3

    uint m_base = gid.y * TILE_M;
    uint n_base = gid.x * TILE_N;
    uint m_sg = m_base + sg_m * SG_M;
    uint n_sg = n_base + sg_n * SG_N;

    // Shared memory
    threadgroup half sh_x[TILE_M * TILE_K];              // 16x32 half = 1024 B
    threadgroup half sh_w[N_SG_N * SG_N * TILE_K];       // 4x8x32 half = 2048 B
    threadgroup float sh_out[8 * SG_M * SG_N];           // 8x64 float = 2048 B

    // Accumulator
    simdgroup_matrix<float, 8, 8> C;
    C.thread_elements()[0] = 0.0f;
    C.thread_elements()[1] = 0.0f;

    for (uint kk = 0; kk < K; kk += TILE_K) {
        // --- Load X tile [16 x 32] as half ---
        for (uint i = tid; i < TILE_M * TILE_K; i += 256) {
            uint mi = i / TILE_K;
            uint ki = i % TILE_K;
            uint m_idx = m_base + mi;
            uint k_idx = kk + ki;
            sh_x[i] = (m_idx < M && k_idx < K) ? half(x[m_idx * K + k_idx]) : 0.0h;
        }

        // --- Dequant W for 4 unique N groups ---
        for (uint i = tid; i < N_SG_N * SG_N * (TILE_K / 2); i += 256) {
            uint n_group_idx = i / (SG_N * (TILE_K / 2));
            uint within = i % (SG_N * (TILE_K / 2));
            uint ni = within / (TILE_K / 2);
            uint bi = within % (TILE_K / 2);

            uint n_row = n_base + n_group_idx * SG_N + ni;
            uint k_ofs = bi * 2;
            uint dst = n_group_idx * (SG_N * TILE_K) + ni * TILE_K + k_ofs;

            if (n_row < N) {
                float s = scales[n_row * n_groups + kk / 32];
                uchar packed = w_packed[n_row * num_bytes + (kk + k_ofs) / 2];
                sh_w[dst]     = half((float(packed & 0xFu) - 8.0f) * s);
                sh_w[dst + 1] = half((float(packed >> 4) - 8.0f) * s);
            } else {
                sh_w[dst]     = 0.0h;
                sh_w[dst + 1] = 0.0h;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- simdgroup multiply-accumulate ---
        simdgroup_matrix<half, 8, 8> A, B;
        uint x_row_ofs = sg_m * SG_M * TILE_K;
        uint w_sg_base = sg_n * (SG_N * TILE_K);

        for (uint sk = 0; sk < 4; sk++) {
            simdgroup_load(A, &sh_x[x_row_ofs], TILE_K, ulong2(sk * 8, 0));
            simdgroup_load(B, &sh_w[w_sg_base], TILE_K, ulong2(sk * 8, 0), true);
            simdgroup_multiply_accumulate(C, A, B, C);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Store output ---
    if (m_sg + SG_M <= M && n_sg + SG_N <= N) {
        simdgroup_store(C, y + m_sg * N + n_sg, N);
    } else {
        uint out_base = sg_id * (SG_M * SG_N);
        simdgroup_store(C, &sh_out[out_base], SG_N);
        for (uint i = lane; i < SG_M * SG_N; i += 32) {
            uint mi = i / SG_N;
            uint ni = i % SG_N;
            uint m_idx = m_sg + mi;
            uint n_idx = n_sg + ni;
            if (m_idx < M && n_idx < N) {
                y[m_idx * N + n_idx] = sh_out[out_base + i];
            }
        }
    }
}
