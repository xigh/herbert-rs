#include <metal_stdlib>
using namespace metal;

struct MoePrefillTiledDownInt8Params {
    uint N;       // hidden_size
    uint K;       // moe_intermediate_size
    uint top_k;
};

// Tiled prefill down projection for Int8 MoE with counting-sort grouping.
// TILE_M=4. Int8 per-channel: result *= scale[row].
// Shared memory: flat shared_x[8192], constraint: TILE_M * K <= 8192.
//
// Grid:  (ceil(N/4), ceil(max_count/TILE_M), num_experts)
// Threads: (128, 1, 1)

#define TILE_M 4u

[[kernel]]
void moe_prefill_tiled_down_int8(
    device const float*    inputs           [[buffer(0)]],
    device const uint*     down_packed_all  [[buffer(1)]],
    device const float*    down_scales_all  [[buffer(2)]],
    device const float*    expert_weights   [[buffer(3)]],
    device const uint*     expert_counts    [[buffer(4)]],
    device const uint*     expert_offsets   [[buffer(5)]],
    device const uint*     sorted_src_idx   [[buffer(6)]],
    device float*          output           [[buffer(7)]],
    constant MoePrefillTiledDownInt8Params& p [[buffer(8)]],
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
    float routing_weight[TILE_M];
    for (uint mi = 0; mi < TILE_M; mi++) {
        if (mi < tile_count) {
            flat_idx[mi] = sorted_src_idx[start + tile_start + mi];
            routing_weight[mi] = expert_weights[flat_idx[mi]];
        } else {
            flat_idx[mi] = 0;
            routing_weight[mi] = 0.0f;
        }
    }

    threadgroup float shared_x[8192];
    for (uint mi = 0; mi < tile_count; mi++) {
        device const float* expert_input = inputs + flat_idx[mi] * K;
        for (uint i = tid; i < K; i += 128) {
            shared_x[mi * K + i] = expert_input[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n >= N) return;

    uint expert_row = expert * N + n;
    uint row_offset = expert_row * num_packed;

    float acc[TILE_M];
    for (uint mi = 0; mi < TILE_M; mi++) {
        acc[mi] = 0.0f;
    }

    // 4x unrolled int8 loop
    uint i = lane;
    for (; i + 96 < num_packed; i += 128) {
        char4 b0 = as_type<char4>(down_packed_all[row_offset + i]);
        for (uint mi = 0; mi < TILE_M; mi++) {
            uint base = mi * K + i * 4;
            acc[mi] += float(b0[0]) * shared_x[base + 0] + float(b0[1]) * shared_x[base + 1]
                     + float(b0[2]) * shared_x[base + 2] + float(b0[3]) * shared_x[base + 3];
        }

        char4 b1 = as_type<char4>(down_packed_all[row_offset + i + 32]);
        for (uint mi = 0; mi < TILE_M; mi++) {
            uint base = mi * K + (i + 32) * 4;
            acc[mi] += float(b1[0]) * shared_x[base + 0] + float(b1[1]) * shared_x[base + 1]
                     + float(b1[2]) * shared_x[base + 2] + float(b1[3]) * shared_x[base + 3];
        }

        char4 b2 = as_type<char4>(down_packed_all[row_offset + i + 64]);
        for (uint mi = 0; mi < TILE_M; mi++) {
            uint base = mi * K + (i + 64) * 4;
            acc[mi] += float(b2[0]) * shared_x[base + 0] + float(b2[1]) * shared_x[base + 1]
                     + float(b2[2]) * shared_x[base + 2] + float(b2[3]) * shared_x[base + 3];
        }

        char4 b3 = as_type<char4>(down_packed_all[row_offset + i + 96]);
        for (uint mi = 0; mi < TILE_M; mi++) {
            uint base = mi * K + (i + 96) * 4;
            acc[mi] += float(b3[0]) * shared_x[base + 0] + float(b3[1]) * shared_x[base + 1]
                     + float(b3[2]) * shared_x[base + 2] + float(b3[3]) * shared_x[base + 3];
        }
    }

    for (; i < num_packed; i += 32) {
        char4 bytes = as_type<char4>(down_packed_all[row_offset + i]);
        for (uint mi = 0; mi < TILE_M; mi++) {
            uint base = mi * K + i * 4;
            acc[mi] += float(bytes[0]) * shared_x[base + 0] + float(bytes[1]) * shared_x[base + 1]
                     + float(bytes[2]) * shared_x[base + 2] + float(bytes[3]) * shared_x[base + 3];
        }
    }

    for (uint mi = 0; mi < TILE_M; mi++) {
        acc[mi] = simd_sum(acc[mi]);
    }

    if (lane == 0) {
        float row_scale = down_scales_all[expert_row];
        for (uint mi = 0; mi < tile_count; mi++) {
            output[flat_idx[mi] * N + n] = routing_weight[mi] * acc[mi] * row_scale;
        }
    }
}
