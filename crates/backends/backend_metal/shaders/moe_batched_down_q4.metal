#include <metal_stdlib>
using namespace metal;

#ifndef Q4_LUT_DEFINED
#define Q4_LUT_DEFINED
constant half q4_lut[16] = {
    -8.0h, -7.0h, -6.0h, -5.0h, -4.0h, -3.0h, -2.0h, -1.0h,
     0.0h,  1.0h,  2.0h,  3.0h,  4.0h,  5.0h,  6.0h,  7.0h
};
#endif

struct MoeBatchedDownQ4Params {
    uint hidden;      // N (output dim = hidden_size)
    uint moe_inter;   // K (input dim = moe_intermediate_size)
    uint top_k;       // number of active experts
};

// Batched Down projection for Q4 MoE experts with LUT dequantization.
//
// Uses q4_lut[] for nibble dequantization. Half shared_x for occupancy,
// float scales and float multiplies for precision.
//
// Dispatch: grid = (ceil(hidden/4), top_k, 1), threads = (128, 1, 1)

[[kernel]]
void moe_batched_down_q4(
    device const float*    inputs           [[buffer(0)]],
    device const uchar*    down_packed_all  [[buffer(1)]],
    device const float*    down_scales_all  [[buffer(2)]],
    device const uint*     expert_ids       [[buffer(3)]],
    device const float*    expert_weights   [[buffer(4)]],
    device float*          output           [[buffer(5)]],
    constant MoeBatchedDownQ4Params& p      [[buffer(6)]],
    threadgroup half*      shared_x         [[threadgroup(0)]],
    uint2                  tid_2d           [[thread_position_in_threadgroup]],
    uint2                  gid              [[threadgroup_position_in_grid]]
) {
    uint tid = tid_2d.x;
    uint hidden = p.hidden;
    uint K = p.moe_inter;
    uint num_bytes = K / 2;
    uint n_groups = (K + 31) / 32;

    uint expert_idx = gid.y;
    uint expert_id = expert_ids[expert_idx];
    float routing_weight = expert_weights[expert_idx];

    uint warp_id = tid / 32;
    uint lane = tid % 32;
    uint n = gid.x * 4 + warp_id;

    if (n >= hidden) return;

    // Load expert input as half (K * 2 bytes).
    device const float* expert_input = inputs + expert_idx * K;
    for (uint i = tid; i < K; i += 128) {
        shared_x[i] = half(expert_input[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint expert_row = expert_id * hidden + n;
    uint row_byte_offset = expert_row * num_bytes;
    uint scale_row_offset = expert_row * n_groups;

    float acc = 0.0f;

    // 4x unrolled loop (byte loads)
    uint i = lane;
    for (; i + 96 < num_bytes; i += 128) {
        uchar b0 = down_packed_all[row_byte_offset + i];
        uint k0 = i * 2;
        float s0 = down_scales_all[scale_row_offset + k0 / 32];
        acc += float(q4_lut[b0 & 0xF]) * s0 * float(shared_x[k0])
             + float(q4_lut[b0 >> 4])  * s0 * float(shared_x[k0 + 1]);

        uchar b1 = down_packed_all[row_byte_offset + i + 32];
        uint k1 = (i + 32) * 2;
        float s1 = down_scales_all[scale_row_offset + k1 / 32];
        acc += float(q4_lut[b1 & 0xF]) * s1 * float(shared_x[k1])
             + float(q4_lut[b1 >> 4])  * s1 * float(shared_x[k1 + 1]);

        uchar b2 = down_packed_all[row_byte_offset + i + 64];
        uint k2 = (i + 64) * 2;
        float s2 = down_scales_all[scale_row_offset + k2 / 32];
        acc += float(q4_lut[b2 & 0xF]) * s2 * float(shared_x[k2])
             + float(q4_lut[b2 >> 4])  * s2 * float(shared_x[k2 + 1]);

        uchar b3 = down_packed_all[row_byte_offset + i + 96];
        uint k3 = (i + 96) * 2;
        float s3 = down_scales_all[scale_row_offset + k3 / 32];
        acc += float(q4_lut[b3 & 0xF]) * s3 * float(shared_x[k3])
             + float(q4_lut[b3 >> 4])  * s3 * float(shared_x[k3 + 1]);
    }

    for (; i < num_bytes; i += 32) {
        uchar byte_val = down_packed_all[row_byte_offset + i];
        uint k = i * 2;
        float s = down_scales_all[scale_row_offset + k / 32];
        acc += float(q4_lut[byte_val & 0xF]) * s * float(shared_x[k])
             + float(q4_lut[byte_val >> 4])  * s * float(shared_x[k + 1]);
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        output[expert_idx * hidden + n] = routing_weight * acc;
    }
}
