#include <metal_stdlib>
using namespace metal;

struct MoePrefillTiledGateUpSwigluInt8Params {
    uint N;       // moe_intermediate_size
    uint K;       // hidden_size
    uint top_k;
};

// Tiled prefill gate+up+SwiGLU for Int8 MoE with counting-sort grouping.
// TILE_M=4 tokens share weight loads. Int8 per-channel scaling.
// Shared memory: flat shared_x[8192], constraint: TILE_M * K <= 8192.
//
// Grid:  (ceil(N/4), ceil(max_count/TILE_M), num_experts)
// Threads: (128, 1, 1)

#define TILE_M 4u

[[kernel]]
void moe_prefill_tiled_gate_up_swiglu_int8(
    device const float*    x                [[buffer(0)]],
    device const uint*     gate_packed_all  [[buffer(1)]],
    device const float*    gate_scales_all  [[buffer(2)]],
    device const uint*     up_packed_all    [[buffer(3)]],
    device const float*    up_scales_all    [[buffer(4)]],
    device const uint*     expert_counts    [[buffer(5)]],
    device const uint*     expert_offsets   [[buffer(6)]],
    device const uint*     sorted_src_idx   [[buffer(7)]],
    device float*          output           [[buffer(8)]],
    constant MoePrefillTiledGateUpSwigluInt8Params& p [[buffer(9)]],
    uint3                  tid_3d           [[thread_position_in_threadgroup]],
    uint3                  gid              [[threadgroup_position_in_grid]]
) {
    uint tid = tid_3d.x;
    uint N = p.N;
    uint K = p.K;
    uint top_k = p.top_k;
    uint num_packed = K / 4;

    uint expert = gid.z;
    uint count = expert_counts[expert];
    uint tile_start = gid.y * TILE_M;
    if (tile_start >= count) return;

    uint start = expert_offsets[expert];
    uint tile_count = min(TILE_M, count - tile_start);

    uint warp_id = tid / 32;
    uint lane = tid % 32;
    uint n = gid.x * 4 + warp_id;

    uint flat_idx[TILE_M];
    uint token_idx[TILE_M];
    for (uint mi = 0; mi < TILE_M; mi++) {
        if (mi < tile_count) {
            flat_idx[mi] = sorted_src_idx[start + tile_start + mi];
            token_idx[mi] = flat_idx[mi] / top_k;
        } else {
            flat_idx[mi] = 0;
            token_idx[mi] = 0;
        }
    }

    threadgroup float shared_x[8192];
    for (uint mi = 0; mi < tile_count; mi++) {
        device const float* token_x = x + token_idx[mi] * K;
        for (uint i = tid; i < K; i += 128) {
            shared_x[mi * K + i] = token_x[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n >= N) return;

    uint expert_row = expert * N + n;
    uint row_offset = expert_row * num_packed;

    float gate_acc[TILE_M];
    float up_acc[TILE_M];
    for (uint mi = 0; mi < TILE_M; mi++) {
        gate_acc[mi] = 0.0f;
        up_acc[mi] = 0.0f;
    }

    // 4x unrolled int8 loop
    uint i = lane;
    for (; i + 96 < num_packed; i += 128) {
        char4 gb0 = as_type<char4>(gate_packed_all[row_offset + i]);
        char4 ub0 = as_type<char4>(up_packed_all[row_offset + i]);
        for (uint mi = 0; mi < TILE_M; mi++) {
            uint base = mi * K + i * 4;
            gate_acc[mi] += float(gb0[0]) * shared_x[base + 0] + float(gb0[1]) * shared_x[base + 1]
                          + float(gb0[2]) * shared_x[base + 2] + float(gb0[3]) * shared_x[base + 3];
            up_acc[mi]   += float(ub0[0]) * shared_x[base + 0] + float(ub0[1]) * shared_x[base + 1]
                          + float(ub0[2]) * shared_x[base + 2] + float(ub0[3]) * shared_x[base + 3];
        }

        char4 gb1 = as_type<char4>(gate_packed_all[row_offset + i + 32]);
        char4 ub1 = as_type<char4>(up_packed_all[row_offset + i + 32]);
        for (uint mi = 0; mi < TILE_M; mi++) {
            uint base = mi * K + (i + 32) * 4;
            gate_acc[mi] += float(gb1[0]) * shared_x[base + 0] + float(gb1[1]) * shared_x[base + 1]
                          + float(gb1[2]) * shared_x[base + 2] + float(gb1[3]) * shared_x[base + 3];
            up_acc[mi]   += float(ub1[0]) * shared_x[base + 0] + float(ub1[1]) * shared_x[base + 1]
                          + float(ub1[2]) * shared_x[base + 2] + float(ub1[3]) * shared_x[base + 3];
        }

        char4 gb2 = as_type<char4>(gate_packed_all[row_offset + i + 64]);
        char4 ub2 = as_type<char4>(up_packed_all[row_offset + i + 64]);
        for (uint mi = 0; mi < TILE_M; mi++) {
            uint base = mi * K + (i + 64) * 4;
            gate_acc[mi] += float(gb2[0]) * shared_x[base + 0] + float(gb2[1]) * shared_x[base + 1]
                          + float(gb2[2]) * shared_x[base + 2] + float(gb2[3]) * shared_x[base + 3];
            up_acc[mi]   += float(ub2[0]) * shared_x[base + 0] + float(ub2[1]) * shared_x[base + 1]
                          + float(ub2[2]) * shared_x[base + 2] + float(ub2[3]) * shared_x[base + 3];
        }

        char4 gb3 = as_type<char4>(gate_packed_all[row_offset + i + 96]);
        char4 ub3 = as_type<char4>(up_packed_all[row_offset + i + 96]);
        for (uint mi = 0; mi < TILE_M; mi++) {
            uint base = mi * K + (i + 96) * 4;
            gate_acc[mi] += float(gb3[0]) * shared_x[base + 0] + float(gb3[1]) * shared_x[base + 1]
                          + float(gb3[2]) * shared_x[base + 2] + float(gb3[3]) * shared_x[base + 3];
            up_acc[mi]   += float(ub3[0]) * shared_x[base + 0] + float(ub3[1]) * shared_x[base + 1]
                          + float(ub3[2]) * shared_x[base + 2] + float(ub3[3]) * shared_x[base + 3];
        }
    }

    for (; i < num_packed; i += 32) {
        char4 gb = as_type<char4>(gate_packed_all[row_offset + i]);
        char4 ub = as_type<char4>(up_packed_all[row_offset + i]);
        for (uint mi = 0; mi < TILE_M; mi++) {
            uint base = mi * K + i * 4;
            gate_acc[mi] += float(gb[0]) * shared_x[base + 0] + float(gb[1]) * shared_x[base + 1]
                          + float(gb[2]) * shared_x[base + 2] + float(gb[3]) * shared_x[base + 3];
            up_acc[mi]   += float(ub[0]) * shared_x[base + 0] + float(ub[1]) * shared_x[base + 1]
                          + float(ub[2]) * shared_x[base + 2] + float(ub[3]) * shared_x[base + 3];
        }
    }

    for (uint mi = 0; mi < TILE_M; mi++) {
        gate_acc[mi] = simd_sum(gate_acc[mi]);
        up_acc[mi] = simd_sum(up_acc[mi]);
    }

    if (lane == 0) {
        for (uint mi = 0; mi < tile_count; mi++) {
            float gate_val = gate_acc[mi] * gate_scales_all[expert * N + n];
            float up_val = up_acc[mi] * up_scales_all[expert * N + n];
            float silu_gate = gate_val / (1.0f + exp(-gate_val));
            output[flat_idx[mi] * N + n] = silu_gate * up_val;
        }
    }
}
