#include <metal_stdlib>
using namespace metal;

struct MoePrefillTiledDownQ4Params {
    uint N;       // hidden_size (output dim)
    uint K;       // moe_intermediate_size (input dim)
    uint top_k;
};

// Tiled prefill down projection for Q4 MoE with counting-sort grouping.
// TILE_M=4. Shared memory: flat shared_x[8192], constraint: TILE_M * K <= 8192.
//
// Input is gate_up_out indexed by original flat_idx (not sorted order).
// Routing weight applied to output.
//
// Grid:  (ceil(N/4), ceil(max_count/TILE_M), num_experts)
// Threads: (128, 1, 1)

#define TILE_M 4u

[[kernel]]
void moe_prefill_tiled_down_q4(
    device const float*    inputs           [[buffer(0)]],
    device const uchar*    down_packed_all  [[buffer(1)]],
    device const float*    down_scales_all  [[buffer(2)]],
    device const float*    expert_weights   [[buffer(3)]],
    device const uint*     expert_counts    [[buffer(4)]],
    device const uint*     expert_offsets   [[buffer(5)]],
    device const uint*     sorted_src_idx   [[buffer(6)]],
    device float*          output           [[buffer(7)]],
    constant MoePrefillTiledDownQ4Params& p [[buffer(8)]],
    uint3                  tid_3d           [[thread_position_in_threadgroup]],
    uint3                  gid              [[threadgroup_position_in_grid]]
) {
    uint tid = tid_3d.x;
    uint N = p.N;   // hidden_size
    uint K = p.K;   // moe_inter
    uint top_k = p.top_k;
    uint num_bytes = K / 2;
    uint n_groups = (K + 31) / 32;

    uint expert = gid.z;
    uint count = expert_counts[expert];
    uint tile_start = gid.y * TILE_M;
    if (tile_start >= count) return;

    uint start = expert_offsets[expert];
    uint tile_count = min(TILE_M, count - tile_start);

    uint warp_id = tid / 32;
    uint lane = tid % 32;
    uint n = gid.x * 4 + warp_id;

    // Load sorted indices and routing weights
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

    // Load TILE_M intermediate activations into shared memory
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
    uint row_byte_offset = expert_row * num_bytes;
    uint scale_row_offset = expert_row * n_groups;

    float acc[TILE_M];
    for (uint mi = 0; mi < TILE_M; mi++) {
        acc[mi] = 0.0f;
    }

    // 4x unrolled Q4 dot product
    uint i = lane;
    for (; i + 96 < num_bytes; i += 128) {
        uchar b0 = down_packed_all[row_byte_offset + i];
        uint k0 = i * 2;
        float s0 = down_scales_all[scale_row_offset + k0 / 32];
        float s0h = down_scales_all[scale_row_offset + (k0 + 1) / 32];
        float w0_lo = (float(b0 & 0xFu) - 8.0f) * s0;
        float w0_hi = (float(b0 >> 4) - 8.0f) * s0h;
        for (uint mi = 0; mi < TILE_M; mi++) {
            acc[mi] += w0_lo * shared_x[mi * K + k0] + w0_hi * shared_x[mi * K + k0 + 1];
        }

        uchar b1 = down_packed_all[row_byte_offset + i + 32];
        uint k1 = (i + 32) * 2;
        float s1 = down_scales_all[scale_row_offset + k1 / 32];
        float s1h = down_scales_all[scale_row_offset + (k1 + 1) / 32];
        float w1_lo = (float(b1 & 0xFu) - 8.0f) * s1;
        float w1_hi = (float(b1 >> 4) - 8.0f) * s1h;
        for (uint mi = 0; mi < TILE_M; mi++) {
            acc[mi] += w1_lo * shared_x[mi * K + k1] + w1_hi * shared_x[mi * K + k1 + 1];
        }

        uchar b2 = down_packed_all[row_byte_offset + i + 64];
        uint k2 = (i + 64) * 2;
        float s2 = down_scales_all[scale_row_offset + k2 / 32];
        float s2h = down_scales_all[scale_row_offset + (k2 + 1) / 32];
        float w2_lo = (float(b2 & 0xFu) - 8.0f) * s2;
        float w2_hi = (float(b2 >> 4) - 8.0f) * s2h;
        for (uint mi = 0; mi < TILE_M; mi++) {
            acc[mi] += w2_lo * shared_x[mi * K + k2] + w2_hi * shared_x[mi * K + k2 + 1];
        }

        uchar b3 = down_packed_all[row_byte_offset + i + 96];
        uint k3 = (i + 96) * 2;
        float s3 = down_scales_all[scale_row_offset + k3 / 32];
        float s3h = down_scales_all[scale_row_offset + (k3 + 1) / 32];
        float w3_lo = (float(b3 & 0xFu) - 8.0f) * s3;
        float w3_hi = (float(b3 >> 4) - 8.0f) * s3h;
        for (uint mi = 0; mi < TILE_M; mi++) {
            acc[mi] += w3_lo * shared_x[mi * K + k3] + w3_hi * shared_x[mi * K + k3 + 1];
        }
    }

    for (; i < num_bytes; i += 32) {
        uchar byte_val = down_packed_all[row_byte_offset + i];
        uint k = i * 2;
        float s_lo = down_scales_all[scale_row_offset + k / 32];
        float s_hi = down_scales_all[scale_row_offset + (k + 1) / 32];
        float w_lo = (float(byte_val & 0xFu) - 8.0f) * s_lo;
        float w_hi = (float(byte_val >> 4) - 8.0f) * s_hi;
        for (uint mi = 0; mi < TILE_M; mi++) {
            acc[mi] += w_lo * shared_x[mi * K + k] + w_hi * shared_x[mi * K + k + 1];
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
