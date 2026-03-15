#include <metal_stdlib>
using namespace metal;

struct Q4MatvecParams {
    uint K;
};

// Q4 matvec v2 with fused residual add — optimised decode kernel.
//
// Same 3 optimisations as q4_matvec_v2:
//   1) 8 rows per threadgroup (256 threads = 8 SIMD groups)
//   2) Arithmetic dequant: float(nibble) - 8.0f
//   3) Vectorised uint4 loads (32 nibbles = 1 quant group per load)
//
// Output: y[n] = residual[n] + dot(w[n], x)
//
// Dispatch: (ceil(N/8), 1, 1) threadgroups of 256 threads.
// TG memory: K × 2 bytes (shared_x as half).

inline float q4_dot32_ra(uint4 pack, float s, threadgroup const half* sx) {
    return (float((pack.x      ) & 0xF) - 8.0f) * s * float(sx[0])
         + (float((pack.x >>  4) & 0xF) - 8.0f) * s * float(sx[1])
         + (float((pack.x >>  8) & 0xF) - 8.0f) * s * float(sx[2])
         + (float((pack.x >> 12) & 0xF) - 8.0f) * s * float(sx[3])
         + (float((pack.x >> 16) & 0xF) - 8.0f) * s * float(sx[4])
         + (float((pack.x >> 20) & 0xF) - 8.0f) * s * float(sx[5])
         + (float((pack.x >> 24) & 0xF) - 8.0f) * s * float(sx[6])
         + (float((pack.x >> 28)       ) - 8.0f) * s * float(sx[7])
         + (float((pack.y      ) & 0xF) - 8.0f) * s * float(sx[8])
         + (float((pack.y >>  4) & 0xF) - 8.0f) * s * float(sx[9])
         + (float((pack.y >>  8) & 0xF) - 8.0f) * s * float(sx[10])
         + (float((pack.y >> 12) & 0xF) - 8.0f) * s * float(sx[11])
         + (float((pack.y >> 16) & 0xF) - 8.0f) * s * float(sx[12])
         + (float((pack.y >> 20) & 0xF) - 8.0f) * s * float(sx[13])
         + (float((pack.y >> 24) & 0xF) - 8.0f) * s * float(sx[14])
         + (float((pack.y >> 28)       ) - 8.0f) * s * float(sx[15])
         + (float((pack.z      ) & 0xF) - 8.0f) * s * float(sx[16])
         + (float((pack.z >>  4) & 0xF) - 8.0f) * s * float(sx[17])
         + (float((pack.z >>  8) & 0xF) - 8.0f) * s * float(sx[18])
         + (float((pack.z >> 12) & 0xF) - 8.0f) * s * float(sx[19])
         + (float((pack.z >> 16) & 0xF) - 8.0f) * s * float(sx[20])
         + (float((pack.z >> 20) & 0xF) - 8.0f) * s * float(sx[21])
         + (float((pack.z >> 24) & 0xF) - 8.0f) * s * float(sx[22])
         + (float((pack.z >> 28)       ) - 8.0f) * s * float(sx[23])
         + (float((pack.w      ) & 0xF) - 8.0f) * s * float(sx[24])
         + (float((pack.w >>  4) & 0xF) - 8.0f) * s * float(sx[25])
         + (float((pack.w >>  8) & 0xF) - 8.0f) * s * float(sx[26])
         + (float((pack.w >> 12) & 0xF) - 8.0f) * s * float(sx[27])
         + (float((pack.w >> 16) & 0xF) - 8.0f) * s * float(sx[28])
         + (float((pack.w >> 20) & 0xF) - 8.0f) * s * float(sx[29])
         + (float((pack.w >> 24) & 0xF) - 8.0f) * s * float(sx[30])
         + (float((pack.w >> 28)       ) - 8.0f) * s * float(sx[31]);
}

[[kernel]]
void q4_matvec_residual_add_v2(
    device const float*    x         [[buffer(0)]],
    device const uchar*    w_packed  [[buffer(1)]],
    device const float*    scales    [[buffer(2)]],
    device const float*    residual  [[buffer(3)]],
    device float*          y         [[buffer(4)]],
    constant Q4MatvecParams&       p [[buffer(5)]],
    threadgroup half*      shared_x  [[threadgroup(0)]],
    uint                   tid       [[thread_position_in_threadgroup]],
    uint                   gid       [[threadgroup_position_in_grid]]
) {
    uint K = p.K;
    uint num_bytes = K / 2;
    uint n_groups = (K + 31) / 32;

    uint warp_id = tid / 32;
    uint lane = tid % 32;
    uint n = gid * 8 + warp_id;

    for (uint i = tid; i < K; i += 256) {
        shared_x[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row_byte_offset = n * num_bytes;
    uint scale_row_offset = n * n_groups;
    uint num_uint = num_bytes / 4;
    device const uint* w_uint = (device const uint*)(w_packed + row_byte_offset);
    float acc = 0.0f;

    // Main loop: uint4 loads, 2× unrolled.
    uint base = lane * 4;
    uint i = base;
    for (; i + 128 < num_uint; i += 256) {
        uint4 p0 = *((device const uint4*)(w_uint + i));
        uint k0 = i * 8;
        float s0 = scales[scale_row_offset + k0 / 32];
        acc += q4_dot32_ra(p0, s0, shared_x + k0);

        uint4 p1 = *((device const uint4*)(w_uint + i + 128));
        uint k1 = (i + 128) * 8;
        float s1 = scales[scale_row_offset + k1 / 32];
        acc += q4_dot32_ra(p1, s1, shared_x + k1);
    }

    // Remaining full uint4 groups
    for (; i < num_uint; i += 128) {
        uint4 p0 = *((device const uint4*)(w_uint + i));
        uint k0 = i * 8;
        float s0 = scales[scale_row_offset + k0 / 32];
        acc += q4_dot32_ra(p0, s0, shared_x + k0);
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        y[n] = residual[n] + acc;
    }
}
