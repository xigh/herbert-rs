#include <metal_stdlib>
using namespace metal;

struct MoePrefillTiledDownBf16Params {
    uint N;       // hidden_size
    uint K;       // moe_intermediate_size
    uint top_k;
};

// Tiled prefill down projection for BF16 MoE with counting-sort grouping.
// TILE_M=4. BF16 packed 2 per uint32, no scales.
// Shared memory: flat shared_x[8192], constraint: TILE_M * K <= 8192.
//
// Grid:  (ceil(N/4), ceil(max_count/TILE_M), num_experts)
// Threads: (128, 1, 1)

#define TILE_M 4u

[[kernel]]
void moe_prefill_tiled_down_bf16(
    device const float*    inputs           [[buffer(0)]],
    device const uint*     down_packed_all  [[buffer(1)]],
    device const float*    expert_weights   [[buffer(2)]],
    device const uint*     expert_counts    [[buffer(3)]],
    device const uint*     expert_offsets   [[buffer(4)]],
    device const uint*     sorted_src_idx   [[buffer(5)]],
    device float*          output           [[buffer(6)]],
    constant MoePrefillTiledDownBf16Params& p [[buffer(7)]],
    uint3                  tid_3d           [[thread_position_in_threadgroup]],
    uint3                  gid              [[threadgroup_position_in_grid]]
) {
    uint tid = tid_3d.x;
    uint N = p.N;
    uint K = p.K;
    uint top_k = p.top_k;
    uint num_packed = K / 2;

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

    // 4x unrolled BF16 loop
    uint i = lane;
    for (; i + 96 < num_packed; i += 128) {
        uint packed0 = down_packed_all[row_offset + i];
        float w0_0 = as_type<float>((packed0 & 0xFFFFu) << 16);
        float w0_1 = as_type<float>(packed0 & 0xFFFF0000u);
        for (uint mi = 0; mi < TILE_M; mi++) {
            acc[mi] += w0_0 * shared_x[mi * K + i * 2 + 0] + w0_1 * shared_x[mi * K + i * 2 + 1];
        }

        uint packed1 = down_packed_all[row_offset + i + 32];
        float w1_0 = as_type<float>((packed1 & 0xFFFFu) << 16);
        float w1_1 = as_type<float>(packed1 & 0xFFFF0000u);
        for (uint mi = 0; mi < TILE_M; mi++) {
            acc[mi] += w1_0 * shared_x[mi * K + (i + 32) * 2 + 0] + w1_1 * shared_x[mi * K + (i + 32) * 2 + 1];
        }

        uint packed2 = down_packed_all[row_offset + i + 64];
        float w2_0 = as_type<float>((packed2 & 0xFFFFu) << 16);
        float w2_1 = as_type<float>(packed2 & 0xFFFF0000u);
        for (uint mi = 0; mi < TILE_M; mi++) {
            acc[mi] += w2_0 * shared_x[mi * K + (i + 64) * 2 + 0] + w2_1 * shared_x[mi * K + (i + 64) * 2 + 1];
        }

        uint packed3 = down_packed_all[row_offset + i + 96];
        float w3_0 = as_type<float>((packed3 & 0xFFFFu) << 16);
        float w3_1 = as_type<float>(packed3 & 0xFFFF0000u);
        for (uint mi = 0; mi < TILE_M; mi++) {
            acc[mi] += w3_0 * shared_x[mi * K + (i + 96) * 2 + 0] + w3_1 * shared_x[mi * K + (i + 96) * 2 + 1];
        }
    }

    for (; i < num_packed; i += 32) {
        uint packed = down_packed_all[row_offset + i];
        float w0 = as_type<float>((packed & 0xFFFFu) << 16);
        float w1 = as_type<float>(packed & 0xFFFF0000u);
        for (uint mi = 0; mi < TILE_M; mi++) {
            acc[mi] += w0 * shared_x[mi * K + i * 2 + 0] + w1 * shared_x[mi * K + i * 2 + 1];
        }
    }

    for (uint mi = 0; mi < TILE_M; mi++) {
        acc[mi] = simd_sum(acc[mi]);
    }

    if (lane == 0) {
        for (uint mi = 0; mi < tile_count; mi++) {
            output[flat_idx[mi] * N + n] = routing_weight[mi] * acc[mi];
        }
    }
}
