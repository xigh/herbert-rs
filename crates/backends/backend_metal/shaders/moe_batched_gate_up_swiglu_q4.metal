#include <metal_stdlib>
using namespace metal;

#ifndef Q4_LUT_DEFINED
#define Q4_LUT_DEFINED
constant half q4_lut[16] = {
    -8.0h, -7.0h, -6.0h, -5.0h, -4.0h, -3.0h, -2.0h, -1.0h,
     0.0h,  1.0h,  2.0h,  3.0h,  4.0h,  5.0h,  6.0h,  7.0h
};
#endif

struct MoeBatchedGateUpSwigluQ4Params {
    uint moe_inter;   // N (output dim per expert, = moe_intermediate_size)
    uint K;           // input dim (= hidden_size)
    uint top_k;       // number of active experts
};

// Batched Gate + Up + SwiGLU for Q4 MoE experts with LUT dequantization.
//
// Uses q4_lut[] for nibble dequantization. Half shared_x for occupancy,
// float scales and float multiplies for precision.
//
// Dispatch: grid = (ceil(moe_inter/4), top_k, 1), threads = (128, 1, 1)

[[kernel]]
void moe_batched_gate_up_swiglu_q4(
    device const float*    x                [[buffer(0)]],
    device const uchar*    gate_packed_all  [[buffer(1)]],
    device const float*    gate_scales_all  [[buffer(2)]],
    device const uchar*    up_packed_all    [[buffer(3)]],
    device const float*    up_scales_all    [[buffer(4)]],
    device const uint*     expert_ids       [[buffer(5)]],
    device float*          output           [[buffer(6)]],
    constant MoeBatchedGateUpSwigluQ4Params& p [[buffer(7)]],
    threadgroup half*      shared_x         [[threadgroup(0)]],
    uint2                  tid_2d           [[thread_position_in_threadgroup]],
    uint2                  gid              [[threadgroup_position_in_grid]]
) {
    uint tid = tid_2d.x;
    uint N = p.moe_inter;
    uint K = p.K;
    uint num_bytes = K / 2;
    uint n_groups = (K + 31) / 32;

    uint expert_idx = gid.y;   // which of the top_k active experts
    uint expert_id = expert_ids[expert_idx];  // actual expert ID

    uint warp_id = tid / 32;
    uint lane = tid % 32;
    uint n = gid.x * 4 + warp_id;

    if (n >= N) return;

    // Load x as half (K * 2 bytes).
    for (uint i = tid; i < K; i += 128) {
        shared_x[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint expert_row = expert_id * N + n;
    uint row_byte_offset = expert_row * num_bytes;
    uint scale_row_offset = expert_row * n_groups;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    // 4x unrolled loop (byte loads)
    uint i = lane;
    for (; i + 96 < num_bytes; i += 128) {
        uchar gb0 = gate_packed_all[row_byte_offset + i];
        uchar ub0 = up_packed_all[row_byte_offset + i];
        uint k0 = i * 2;
        float sg0 = gate_scales_all[scale_row_offset + k0 / 32];
        float su0 = up_scales_all[scale_row_offset + k0 / 32];
        float x0_0 = float(shared_x[k0]);
        float x0_1 = float(shared_x[k0 + 1]);
        gate_acc += float(q4_lut[gb0 & 0xF]) * sg0 * x0_0
                  + float(q4_lut[gb0 >> 4])  * sg0 * x0_1;
        up_acc   += float(q4_lut[ub0 & 0xF]) * su0 * x0_0
                  + float(q4_lut[ub0 >> 4])  * su0 * x0_1;

        uchar gb1 = gate_packed_all[row_byte_offset + i + 32];
        uchar ub1 = up_packed_all[row_byte_offset + i + 32];
        uint k1 = (i + 32) * 2;
        float sg1 = gate_scales_all[scale_row_offset + k1 / 32];
        float su1 = up_scales_all[scale_row_offset + k1 / 32];
        float x1_0 = float(shared_x[k1]);
        float x1_1 = float(shared_x[k1 + 1]);
        gate_acc += float(q4_lut[gb1 & 0xF]) * sg1 * x1_0
                  + float(q4_lut[gb1 >> 4])  * sg1 * x1_1;
        up_acc   += float(q4_lut[ub1 & 0xF]) * su1 * x1_0
                  + float(q4_lut[ub1 >> 4])  * su1 * x1_1;

        uchar gb2 = gate_packed_all[row_byte_offset + i + 64];
        uchar ub2 = up_packed_all[row_byte_offset + i + 64];
        uint k2 = (i + 64) * 2;
        float sg2 = gate_scales_all[scale_row_offset + k2 / 32];
        float su2 = up_scales_all[scale_row_offset + k2 / 32];
        float x2_0 = float(shared_x[k2]);
        float x2_1 = float(shared_x[k2 + 1]);
        gate_acc += float(q4_lut[gb2 & 0xF]) * sg2 * x2_0
                  + float(q4_lut[gb2 >> 4])  * sg2 * x2_1;
        up_acc   += float(q4_lut[ub2 & 0xF]) * su2 * x2_0
                  + float(q4_lut[ub2 >> 4])  * su2 * x2_1;

        uchar gb3 = gate_packed_all[row_byte_offset + i + 96];
        uchar ub3 = up_packed_all[row_byte_offset + i + 96];
        uint k3 = (i + 96) * 2;
        float sg3 = gate_scales_all[scale_row_offset + k3 / 32];
        float su3 = up_scales_all[scale_row_offset + k3 / 32];
        float x3_0 = float(shared_x[k3]);
        float x3_1 = float(shared_x[k3 + 1]);
        gate_acc += float(q4_lut[gb3 & 0xF]) * sg3 * x3_0
                  + float(q4_lut[gb3 >> 4])  * sg3 * x3_1;
        up_acc   += float(q4_lut[ub3 & 0xF]) * su3 * x3_0
                  + float(q4_lut[ub3 >> 4])  * su3 * x3_1;
    }

    // Handle remainder
    for (; i < num_bytes; i += 32) {
        uint k = i * 2;
        float s_g = gate_scales_all[scale_row_offset + k / 32];
        float s_u = up_scales_all[scale_row_offset + k / 32];

        float x0 = float(shared_x[k]);
        float x1 = float(shared_x[k + 1]);

        uchar gb = gate_packed_all[row_byte_offset + i];
        gate_acc += float(q4_lut[gb & 0xF]) * s_g * x0
                  + float(q4_lut[gb >> 4])  * s_g * x1;

        uchar ub = up_packed_all[row_byte_offset + i];
        up_acc += float(q4_lut[ub & 0xF]) * s_u * x0
                + float(q4_lut[ub >> 4])  * s_u * x1;
    }

    gate_acc = simd_sum(gate_acc);
    up_acc = simd_sum(up_acc);

    if (lane == 0) {
        float silu_gate = gate_acc / (1.0f + exp(-gate_acc));
        output[expert_idx * N + n] = silu_gate * up_acc;
    }
}
