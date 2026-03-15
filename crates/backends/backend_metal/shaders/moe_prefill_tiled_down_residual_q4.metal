// Fused tiled prefill down projection + reduce + residual add for Q4 MoE.
//
// Simdgroup matrix variant: 256 threads = 8 simdgroups arranged 2x4.
// TILE_M=16 tokens, TILE_N=32 output columns, TILE_K=32 inner dim.
// K-tiled loop with simdgroup_multiply_accumulate on half-precision tiles.
// Weights dequantized from Q4 to half, X loaded as half, accumulated in float.
//
// Grid:  (ceil(N/32), ceil(max_count/16), num_experts) — indirect dispatch
// Threads: (256, 1, 1)

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct MoePrefillTiledDownResidualQ4Params {
    uint N;       // hidden_size (output dim)
    uint K;       // moe_intermediate_size (input dim)
    uint top_k;
};

#define TILE_M 16u
#define TILE_K 32u
#define SG_M   8u
#define SG_N   8u
#define N_SG_M 2u
#define N_SG_N 4u
#define TILE_N (N_SG_N * SG_N)  // 32

[[kernel]]
void moe_prefill_tiled_down_residual_q4(
    device const float*    inputs           [[buffer(0)]],
    device const uchar*    down_packed_all  [[buffer(1)]],
    device const float*    down_scales_all  [[buffer(2)]],
    device const float*    expert_weights   [[buffer(3)]],
    device const uint*     expert_counts    [[buffer(4)]],
    device const uint*     expert_offsets   [[buffer(5)]],
    device const uint*     sorted_src_idx   [[buffer(6)]],
    device float*          down_out         [[buffer(7)]],
    device atomic_uint*    counters         [[buffer(8)]],
    device float*          residual         [[buffer(9)]],
    constant MoePrefillTiledDownResidualQ4Params& p [[buffer(10)]],
    uint                   tid              [[thread_index_in_threadgroup]],
    uint3                  gid              [[threadgroup_position_in_grid]]
) {
    uint N = p.N;
    uint K = p.K;
    uint top_k = p.top_k;
    uint num_bytes = K / 2;
    uint n_groups = (K + 31) / 32;

    uint expert = gid.z;
    uint count = expert_counts[expert];
    uint tile_start = gid.y * TILE_M;
    if (tile_start >= count) return;

    uint start = expert_offsets[expert];
    uint tile_count = min(TILE_M, count - tile_start);

    uint sg_id = tid / 32;
    uint sg_m = sg_id / N_SG_N;   // 0 or 1
    uint sg_n = sg_id % N_SG_N;   // 0..3
    uint n_base = gid.x * TILE_N;

    // Shared memory
    threadgroup uint  sh_flat_idx[TILE_M];
    threadgroup float sh_rw[TILE_M];
    threadgroup half  sh_x[TILE_M * TILE_K];              // 16x32 half = 1024 B
    threadgroup half  sh_w[N_SG_N * SG_N * TILE_K];       // 32x32 half = 2048 B
    threadgroup float sh_out[TILE_M * TILE_N];             // 16x32 float = 2048 B

    // Load sorted indices and routing weights (TILE_M <= 256 threads)
    if (tid < TILE_M) {
        if (tid < tile_count) {
            uint idx = sorted_src_idx[start + tile_start + tid];
            sh_flat_idx[tid] = idx;
            sh_rw[tid] = expert_weights[idx];
        } else {
            sh_flat_idx[tid] = 0;
            sh_rw[tid] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulator
    simdgroup_matrix<float, 8, 8> C;
    C.thread_elements()[0] = 0.0f;
    C.thread_elements()[1] = 0.0f;

    for (uint kk = 0; kk < K; kk += TILE_K) {
        // --- Load X tile [TILE_M x TILE_K] as half, scattered via sh_flat_idx ---
        for (uint i = tid; i < TILE_M * TILE_K; i += 256) {
            uint mi = i / TILE_K;
            uint ki = i % TILE_K;
            uint k_idx = kk + ki;
            sh_x[i] = (mi < tile_count && k_idx < K)
                ? half(inputs[sh_flat_idx[mi] * K + k_idx]) : 0.0h;
        }

        // --- Dequant W for TILE_N=32 output rows (4 groups of 8, expert-offset) ---
        for (uint i = tid; i < N_SG_N * SG_N * (TILE_K / 2); i += 256) {
            uint n_group_idx = i / (SG_N * (TILE_K / 2));
            uint within = i % (SG_N * (TILE_K / 2));
            uint ni = within / (TILE_K / 2);
            uint bi = within % (TILE_K / 2);

            uint n_row = n_base + n_group_idx * SG_N + ni;
            uint k_ofs = bi * 2;
            uint dst = n_group_idx * (SG_N * TILE_K) + ni * TILE_K + k_ofs;

            if (n_row < N) {
                uint global_n_row = expert * N + n_row;
                float s = down_scales_all[global_n_row * n_groups + (kk + k_ofs) / 32];
                uchar packed = down_packed_all[global_n_row * num_bytes + (kk + k_ofs) / 2];
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

    // --- Store C to shared output ---
    simdgroup_store(C, &sh_out[sg_m * SG_M * TILE_N + sg_n * SG_N], TILE_N);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Scatter-write: routing weight * result, then atomic counter reduce ---
    for (uint i = tid; i < tile_count * TILE_N; i += 256) {
        uint mi = i / TILE_N;
        uint ni = i % TILE_N;
        uint n_idx = n_base + ni;
        if (n_idx >= N) continue;

        uint fidx = sh_flat_idx[mi];
        down_out[fidx * N + n_idx] = sh_rw[mi] * sh_out[mi * TILE_N + ni];

        uint token = fidx / top_k;
        uint old = atomic_fetch_add_explicit(
            &counters[token * N + n_idx], 1u, memory_order_relaxed);

        if (old == top_k - 1) {
            float sum = 0.0f;
            for (uint e = 0; e < top_k; e++) {
                sum += down_out[(token * top_k + e) * N + n_idx];
            }
            residual[token * N + n_idx] += sum;
            atomic_store_explicit(&counters[token * N + n_idx], 0u, memory_order_relaxed);
        }
    }
}
