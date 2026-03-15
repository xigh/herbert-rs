#include <metal_stdlib>
using namespace metal;

#ifndef Q4_LUT_DEFINED
#define Q4_LUT_DEFINED
constant half q4_lut[16] = {
    -8.0h, -7.0h, -6.0h, -5.0h, -4.0h, -3.0h, -2.0h, -1.0h,
     0.0h,  1.0h,  2.0h,  3.0h,  4.0h,  5.0h,  6.0h,  7.0h
};
#endif

struct MoeFusedGateUpSwigluQ4Params {
    uint K;
};

// Fused Gate + Up projection + SwiGLU for Q4 weights (MoE single-expert decode).
//
// Uses q4_lut[] for nibble dequantization. Half shared_x for occupancy,
// float scales and float multiplies for precision.
//
// Dispatch: (ceil(N/4), 1, 1) threadgroups of 128 threads (4 rows per TG).

[[kernel]]
void moe_fused_gate_up_swiglu_q4(
    device const float*    x              [[buffer(0)]],
    device const uchar*    gate_w_packed  [[buffer(1)]],
    device const float*    gate_scales    [[buffer(2)]],
    device const uchar*    up_w_packed    [[buffer(3)]],
    device const float*    up_scales      [[buffer(4)]],
    device float*          output         [[buffer(5)]],
    constant MoeFusedGateUpSwigluQ4Params& p [[buffer(6)]],
    threadgroup half*      shared_x       [[threadgroup(0)]],
    uint                   tid            [[thread_position_in_threadgroup]],
    uint                   gid            [[threadgroup_position_in_grid]]
) {
    uint K = p.K;
    uint num_bytes = K / 2;
    uint n_groups = (K + 31) / 32;

    uint warp_id = tid / 32;
    uint lane = tid % 32;
    uint n = gid * 4 + warp_id;

    // Load x as half (K * 2 bytes).
    for (uint i = tid; i < K; i += 128) {
        shared_x[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row_byte_offset = n * num_bytes;
    uint scale_row_offset = n * n_groups;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (uint i = lane; i < num_bytes; i += 32) {
        uint k = i * 2;
        float s_g = gate_scales[scale_row_offset + k / 32];
        float s_u = up_scales[scale_row_offset + k / 32];

        float x0 = float(shared_x[k]);
        float x1 = float(shared_x[k + 1]);

        // Gate weights
        uchar gb = gate_w_packed[row_byte_offset + i];
        gate_acc += float(q4_lut[gb & 0xF]) * s_g * x0
                  + float(q4_lut[gb >> 4])  * s_g * x1;

        // Up weights
        uchar ub = up_w_packed[row_byte_offset + i];
        up_acc += float(q4_lut[ub & 0xF]) * s_u * x0
                + float(q4_lut[ub >> 4])  * s_u * x1;
    }

    gate_acc = simd_sum(gate_acc);
    up_acc = simd_sum(up_acc);

    if (lane == 0) {
        float silu_gate = gate_acc / (1.0f + exp(-gate_acc));
        output[n] = silu_gate * up_acc;
    }
}
