#include <metal_stdlib>
using namespace metal;

#ifndef Q4_LUT_DEFINED
#define Q4_LUT_DEFINED
constant half q4_lut[16] = {
    -8.0h, -7.0h, -6.0h, -5.0h, -4.0h, -3.0h, -2.0h, -1.0h,
     0.0h,  1.0h,  2.0h,  3.0h,  4.0h,  5.0h,  6.0h,  7.0h
};
#endif

struct Q4MatvecParams {
    uint K;
};

// Q4 matrix-vector multiply with LUT dequantization and half shared_x.
//
// 4 rows per threadgroup (128 threads = 4 SIMD groups of 32).
// Input vector x stored in threadgroup memory as half (K*2 bytes) for
// improved occupancy. All multiplies and accumulation in float32 for precision.
//
// Weight dequant uses q4_lut[] (cast to float) instead of float(nibble) - 8.0.
//
// Dispatch: (ceil(N/4), 1, 1) threadgroups of 128 threads.

[[kernel]]
void q4_matvec(
    device const float*    x        [[buffer(0)]],
    device const uchar*    w_packed [[buffer(1)]],
    device const float*    scales   [[buffer(2)]],
    device float*          y        [[buffer(3)]],
    constant Q4MatvecParams&       p        [[buffer(4)]],
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

    // Load x into threadgroup memory as half (K * 2 bytes).
    for (uint i = tid; i < K; i += 128) {
        shared_x[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Q4 dot product with uint loads (8 nibbles per uint).
    // LUT gives half → cast to float. Scales and accumulation in float32.
    uint row_byte_offset = n * num_bytes;
    uint scale_row_offset = n * n_groups;
    uint num_uint = num_bytes / 4;
    device const uint* w_uint = (device const uint*)(w_packed + row_byte_offset);
    float acc = 0.0f;

    // 4x unrolled loop: process 4 uints (32 Q4 values) per iteration.
    uint i = lane;
    for (; i + 96 < num_uint; i += 128) {
        uint pack0 = w_uint[i];
        uint k0 = i * 8;
        float s0 = scales[scale_row_offset + k0 / 32];
        acc += float(q4_lut[(pack0      ) & 0xF]) * s0 * float(shared_x[k0])
             + float(q4_lut[(pack0 >>  4) & 0xF]) * s0 * float(shared_x[k0 + 1])
             + float(q4_lut[(pack0 >>  8) & 0xF]) * s0 * float(shared_x[k0 + 2])
             + float(q4_lut[(pack0 >> 12) & 0xF]) * s0 * float(shared_x[k0 + 3])
             + float(q4_lut[(pack0 >> 16) & 0xF]) * s0 * float(shared_x[k0 + 4])
             + float(q4_lut[(pack0 >> 20) & 0xF]) * s0 * float(shared_x[k0 + 5])
             + float(q4_lut[(pack0 >> 24) & 0xF]) * s0 * float(shared_x[k0 + 6])
             + float(q4_lut[(pack0 >> 28)       ]) * s0 * float(shared_x[k0 + 7]);

        uint pack1 = w_uint[i + 32];
        uint k1 = (i + 32) * 8;
        float s1 = scales[scale_row_offset + k1 / 32];
        acc += float(q4_lut[(pack1      ) & 0xF]) * s1 * float(shared_x[k1])
             + float(q4_lut[(pack1 >>  4) & 0xF]) * s1 * float(shared_x[k1 + 1])
             + float(q4_lut[(pack1 >>  8) & 0xF]) * s1 * float(shared_x[k1 + 2])
             + float(q4_lut[(pack1 >> 12) & 0xF]) * s1 * float(shared_x[k1 + 3])
             + float(q4_lut[(pack1 >> 16) & 0xF]) * s1 * float(shared_x[k1 + 4])
             + float(q4_lut[(pack1 >> 20) & 0xF]) * s1 * float(shared_x[k1 + 5])
             + float(q4_lut[(pack1 >> 24) & 0xF]) * s1 * float(shared_x[k1 + 6])
             + float(q4_lut[(pack1 >> 28)       ]) * s1 * float(shared_x[k1 + 7]);

        uint pack2 = w_uint[i + 64];
        uint k2 = (i + 64) * 8;
        float s2 = scales[scale_row_offset + k2 / 32];
        acc += float(q4_lut[(pack2      ) & 0xF]) * s2 * float(shared_x[k2])
             + float(q4_lut[(pack2 >>  4) & 0xF]) * s2 * float(shared_x[k2 + 1])
             + float(q4_lut[(pack2 >>  8) & 0xF]) * s2 * float(shared_x[k2 + 2])
             + float(q4_lut[(pack2 >> 12) & 0xF]) * s2 * float(shared_x[k2 + 3])
             + float(q4_lut[(pack2 >> 16) & 0xF]) * s2 * float(shared_x[k2 + 4])
             + float(q4_lut[(pack2 >> 20) & 0xF]) * s2 * float(shared_x[k2 + 5])
             + float(q4_lut[(pack2 >> 24) & 0xF]) * s2 * float(shared_x[k2 + 6])
             + float(q4_lut[(pack2 >> 28)       ]) * s2 * float(shared_x[k2 + 7]);

        uint pack3 = w_uint[i + 96];
        uint k3 = (i + 96) * 8;
        float s3 = scales[scale_row_offset + k3 / 32];
        acc += float(q4_lut[(pack3      ) & 0xF]) * s3 * float(shared_x[k3])
             + float(q4_lut[(pack3 >>  4) & 0xF]) * s3 * float(shared_x[k3 + 1])
             + float(q4_lut[(pack3 >>  8) & 0xF]) * s3 * float(shared_x[k3 + 2])
             + float(q4_lut[(pack3 >> 12) & 0xF]) * s3 * float(shared_x[k3 + 3])
             + float(q4_lut[(pack3 >> 16) & 0xF]) * s3 * float(shared_x[k3 + 4])
             + float(q4_lut[(pack3 >> 20) & 0xF]) * s3 * float(shared_x[k3 + 5])
             + float(q4_lut[(pack3 >> 24) & 0xF]) * s3 * float(shared_x[k3 + 6])
             + float(q4_lut[(pack3 >> 28)       ]) * s3 * float(shared_x[k3 + 7]);
    }

    // Handle remainder (non-unrolled).
    for (; i < num_uint; i += 32) {
        uint pack = w_uint[i];
        uint k = i * 8;
        float s = scales[scale_row_offset + k / 32];
        acc += float(q4_lut[(pack      ) & 0xF]) * s * float(shared_x[k])
             + float(q4_lut[(pack >>  4) & 0xF]) * s * float(shared_x[k + 1])
             + float(q4_lut[(pack >>  8) & 0xF]) * s * float(shared_x[k + 2])
             + float(q4_lut[(pack >> 12) & 0xF]) * s * float(shared_x[k + 3])
             + float(q4_lut[(pack >> 16) & 0xF]) * s * float(shared_x[k + 4])
             + float(q4_lut[(pack >> 20) & 0xF]) * s * float(shared_x[k + 5])
             + float(q4_lut[(pack >> 24) & 0xF]) * s * float(shared_x[k + 6])
             + float(q4_lut[(pack >> 28)       ]) * s * float(shared_x[k + 7]);
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        y[n] = acc;
    }
}
