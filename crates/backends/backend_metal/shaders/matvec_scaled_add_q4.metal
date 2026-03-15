#include <metal_stdlib>
using namespace metal;

#ifndef Q4_LUT_DEFINED
#define Q4_LUT_DEFINED
constant half q4_lut[16] = {
    -8.0h, -7.0h, -6.0h, -5.0h, -4.0h, -3.0h, -2.0h, -1.0h,
     0.0h,  1.0h,  2.0h,  3.0h,  4.0h,  5.0h,  6.0h,  7.0h
};
#endif

struct MatvecScaledAddQ4Params {
    uint K;
    float scale;
};

// Fused Q4 matvec + scaled accumulation with LUT dequantization.
//
// Directly accumulates: output[n] += scale * dot(w[n], x)
// Uses q4_lut[] for nibble dequantization. Half shared_x for occupancy,
// float scales and float multiplies for precision.
//
// Dispatch: (ceil(N/4), 1, 1) threadgroups of 128 threads (4 rows per TG).

[[kernel]]
void matvec_scaled_add_q4(
    device const float*    x        [[buffer(0)]],
    device const uchar*    w_packed [[buffer(1)]],
    device const float*    scales   [[buffer(2)]],
    device float*          output   [[buffer(3)]],
    constant MatvecScaledAddQ4Params& p [[buffer(4)]],
    threadgroup half*      shared_x [[threadgroup(0)]],
    uint                   tid      [[thread_position_in_threadgroup]],
    uint                   gid      [[threadgroup_position_in_grid]]
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
    float acc = 0.0f;

    // 4x unrolled loop (byte loads, 2 weights per byte)
    uint i = lane;
    for (; i + 96 < num_bytes; i += 128) {
        uchar b0 = w_packed[row_byte_offset + i];
        uint k0 = i * 2;
        float s0 = scales[scale_row_offset + k0 / 32];
        acc += float(q4_lut[b0 & 0xF]) * s0 * float(shared_x[k0])
             + float(q4_lut[b0 >> 4])  * s0 * float(shared_x[k0 + 1]);

        uchar b1 = w_packed[row_byte_offset + i + 32];
        uint k1 = (i + 32) * 2;
        float s1 = scales[scale_row_offset + k1 / 32];
        acc += float(q4_lut[b1 & 0xF]) * s1 * float(shared_x[k1])
             + float(q4_lut[b1 >> 4])  * s1 * float(shared_x[k1 + 1]);

        uchar b2 = w_packed[row_byte_offset + i + 64];
        uint k2 = (i + 64) * 2;
        float s2 = scales[scale_row_offset + k2 / 32];
        acc += float(q4_lut[b2 & 0xF]) * s2 * float(shared_x[k2])
             + float(q4_lut[b2 >> 4])  * s2 * float(shared_x[k2 + 1]);

        uchar b3 = w_packed[row_byte_offset + i + 96];
        uint k3 = (i + 96) * 2;
        float s3 = scales[scale_row_offset + k3 / 32];
        acc += float(q4_lut[b3 & 0xF]) * s3 * float(shared_x[k3])
             + float(q4_lut[b3 >> 4])  * s3 * float(shared_x[k3 + 1]);
    }

    for (; i < num_bytes; i += 32) {
        uchar byte_val = w_packed[row_byte_offset + i];
        uint k = i * 2;
        float s = scales[scale_row_offset + k / 32];
        acc += float(q4_lut[byte_val & 0xF]) * s * float(shared_x[k])
             + float(q4_lut[byte_val >> 4])  * s * float(shared_x[k + 1]);
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        output[n] += p.scale * acc;
    }
}
