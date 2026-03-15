// Tiled prefill gate+up+SwiGLU for Q4 MoE with counting-sort grouping.
//
// Simdgroup matrix variant: 256 threads = 8 simdgroups arranged 2x4.
// TILE_M=16 tokens, TILE_N=32 output columns, TILE_K=32 inner dim.
// Dual weight tiles (gate + up) loaded in parallel, SwiGLU fused at output.
// Weights dequantized from Q4 to half, X loaded as half, accumulated in float.
//
// Grid:  (ceil(N/32), ceil(max_count/16), num_experts) — indirect dispatch
// Threads: (256, 1, 1)

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct MoePrefillTiledGateUpSwigluQ4Params {
    uint N;       // moe_intermediate_size (output dim)
    uint K;       // hidden_size (input dim)
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
void moe_prefill_tiled_gate_up_swiglu_q4(
    device const float*    x                [[buffer(0)]],
    device const uchar*    gate_packed_all  [[buffer(1)]],
    device const float*    gate_scales_all  [[buffer(2)]],
    device const uchar*    up_packed_all    [[buffer(3)]],
    device const float*    up_scales_all    [[buffer(4)]],
    device const uint*     expert_counts    [[buffer(5)]],
    device const uint*     expert_offsets   [[buffer(6)]],
    device const uint*     sorted_src_idx   [[buffer(7)]],
    device float*          output           [[buffer(8)]],
    constant MoePrefillTiledGateUpSwigluQ4Params& p [[buffer(9)]],
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

    // Shared memory (~9 KB total)
    threadgroup uint  sh_flat_idx[TILE_M];
    threadgroup half  sh_x[TILE_M * TILE_K];              // 16x32 half = 1024 B
    threadgroup half  sh_w_gate[N_SG_N * SG_N * TILE_K];  // 32x32 half = 2048 B
    threadgroup half  sh_w_up[N_SG_N * SG_N * TILE_K];    // 32x32 half = 2048 B
    threadgroup float sh_out_gate[TILE_M * TILE_N];        // 16x32 float = 2048 B
    threadgroup float sh_out_up[TILE_M * TILE_N];          // 16x32 float = 2048 B

    // Load sorted indices
    if (tid < TILE_M) {
        if (tid < tile_count) {
            sh_flat_idx[tid] = sorted_src_idx[start + tile_start + tid];
        } else {
            sh_flat_idx[tid] = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Dual accumulators
    simdgroup_matrix<float, 8, 8> C_gate, C_up;
    C_gate.thread_elements()[0] = 0.0f;
    C_gate.thread_elements()[1] = 0.0f;
    C_up.thread_elements()[0] = 0.0f;
    C_up.thread_elements()[1] = 0.0f;

    for (uint kk = 0; kk < K; kk += TILE_K) {
        // --- Load X tile [TILE_M x TILE_K] as half ---
        // Input indexed by token: flat_idx / top_k
        for (uint i = tid; i < TILE_M * TILE_K; i += 256) {
            uint mi = i / TILE_K;
            uint ki = i % TILE_K;
            uint k_idx = kk + ki;
            sh_x[i] = (mi < tile_count && k_idx < K)
                ? half(x[(sh_flat_idx[mi] / top_k) * K + k_idx]) : 0.0h;
        }

        // --- Dequant gate weights for TILE_N=32 rows ---
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
                float sg = gate_scales_all[global_n_row * n_groups + (kk + k_ofs) / 32];
                uchar gp = gate_packed_all[global_n_row * num_bytes + (kk + k_ofs) / 2];
                sh_w_gate[dst]     = half((float(gp & 0xFu) - 8.0f) * sg);
                sh_w_gate[dst + 1] = half((float(gp >> 4) - 8.0f) * sg);

                float su = up_scales_all[global_n_row * n_groups + (kk + k_ofs) / 32];
                uchar up = up_packed_all[global_n_row * num_bytes + (kk + k_ofs) / 2];
                sh_w_up[dst]     = half((float(up & 0xFu) - 8.0f) * su);
                sh_w_up[dst + 1] = half((float(up >> 4) - 8.0f) * su);
            } else {
                sh_w_gate[dst]     = 0.0h;
                sh_w_gate[dst + 1] = 0.0h;
                sh_w_up[dst]     = 0.0h;
                sh_w_up[dst + 1] = 0.0h;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Dual simdgroup multiply-accumulate ---
        simdgroup_matrix<half, 8, 8> A, B;
        uint x_row_ofs = sg_m * SG_M * TILE_K;
        uint w_sg_base = sg_n * (SG_N * TILE_K);

        for (uint sk = 0; sk < 4; sk++) {
            simdgroup_load(A, &sh_x[x_row_ofs], TILE_K, ulong2(sk * 8, 0));

            simdgroup_load(B, &sh_w_gate[w_sg_base], TILE_K, ulong2(sk * 8, 0), true);
            simdgroup_multiply_accumulate(C_gate, A, B, C_gate);

            simdgroup_load(B, &sh_w_up[w_sg_base], TILE_K, ulong2(sk * 8, 0), true);
            simdgroup_multiply_accumulate(C_up, A, B, C_up);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Store results to shared memory ---
    simdgroup_store(C_gate, &sh_out_gate[sg_m * SG_M * TILE_N + sg_n * SG_N], TILE_N);
    simdgroup_store(C_up, &sh_out_up[sg_m * SG_M * TILE_N + sg_n * SG_N], TILE_N);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Scatter-write with SwiGLU fusion ---
    for (uint i = tid; i < tile_count * TILE_N; i += 256) {
        uint mi = i / TILE_N;
        uint ni = i % TILE_N;
        uint n_idx = n_base + ni;
        if (n_idx >= N) continue;

        float g = sh_out_gate[mi * TILE_N + ni];
        float u = sh_out_up[mi * TILE_N + ni];
        output[sh_flat_idx[mi] * N + n_idx] = (g / (1.0f + exp(-g))) * u;
    }
}
