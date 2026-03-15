#include <metal_stdlib>
using namespace metal;

struct Int8MatvecParams {
    uint K;
};

// Int8 per-channel matrix-vector multiply with multi-row processing and shared input.
//
// 4 rows per threadgroup (128 threads = 4 SIMD groups of 32).
// The input vector x is loaded once into threadgroup memory and shared
// across all 4 rows, reducing global memory traffic.
//
// W is stored as packed uint32 where each uint holds 4 signed int8 bytes.
// y[n] = scales[n] * sum_k W[n, k] * x[k]
//
// Dispatch: (ceil(N/4), 1, 1) threadgroups of 128 threads.
// Each threadgroup handles 4 consecutive output rows.
//
// Buffers:
//   x        : [K]      - float input vector
//   w_packed : [N, K/4] - int8 weights packed as uint32 (4 bytes per uint)
//   scales   : [N]      - per-output-row dequantization scales
//   y        : [N]      - float output
[[kernel]]
void int8_matvec(
    device const float*    x        [[buffer(0)]],
    device const uint*     w_packed [[buffer(1)]],
    device const float*    scales   [[buffer(2)]],
    device float*          y        [[buffer(3)]],
    constant Int8MatvecParams&       p        [[buffer(4)]],
    threadgroup float*     shared_x [[threadgroup(0)]],
    uint                   tid      [[thread_position_in_threadgroup]],
    uint                   gid      [[threadgroup_position_in_grid]]
) {
    uint K = p.K;
    uint num_packed = K / 4;

    // Identify which SIMD group (warp) and lane within it.
    uint warp_id = tid / 32;
    uint lane = tid % 32;

    // Which output row this warp handles.
    uint n = gid * 4 + warp_id;

    // Load x into threadgroup memory (cooperative load across all 128 threads).
    // shared_x size set dynamically via setThreadgroupMemoryLength (K * 4 bytes).
    for (uint i = tid; i < K; i += 128) {
        shared_x[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute dot product for this row using shared x.
    uint row_offset = n * num_packed;
    float acc = 0.0f;

    // 4x unrolled loop: process 4 packed words (16 int8 values) per iteration.
    uint i = lane;
    for (; i + 96 < num_packed; i += 128) {
        // Iteration 0
        char4 b0 = as_type<char4>(w_packed[row_offset + i]);
        acc += float(b0[0]) * shared_x[(i) * 4 + 0] + float(b0[1]) * shared_x[(i) * 4 + 1]
             + float(b0[2]) * shared_x[(i) * 4 + 2] + float(b0[3]) * shared_x[(i) * 4 + 3];

        // Iteration 1
        char4 b1 = as_type<char4>(w_packed[row_offset + i + 32]);
        acc += float(b1[0]) * shared_x[(i + 32) * 4 + 0] + float(b1[1]) * shared_x[(i + 32) * 4 + 1]
             + float(b1[2]) * shared_x[(i + 32) * 4 + 2] + float(b1[3]) * shared_x[(i + 32) * 4 + 3];

        // Iteration 2
        char4 b2 = as_type<char4>(w_packed[row_offset + i + 64]);
        acc += float(b2[0]) * shared_x[(i + 64) * 4 + 0] + float(b2[1]) * shared_x[(i + 64) * 4 + 1]
             + float(b2[2]) * shared_x[(i + 64) * 4 + 2] + float(b2[3]) * shared_x[(i + 64) * 4 + 3];

        // Iteration 3
        char4 b3 = as_type<char4>(w_packed[row_offset + i + 96]);
        acc += float(b3[0]) * shared_x[(i + 96) * 4 + 0] + float(b3[1]) * shared_x[(i + 96) * 4 + 1]
             + float(b3[2]) * shared_x[(i + 96) * 4 + 2] + float(b3[3]) * shared_x[(i + 96) * 4 + 3];
    }

    // Handle remainder (non-unrolled).
    for (; i < num_packed; i += 32) {
        char4 bytes = as_type<char4>(w_packed[row_offset + i]);
        acc += float(bytes[0]) * shared_x[i * 4 + 0] + float(bytes[1]) * shared_x[i * 4 + 1]
             + float(bytes[2]) * shared_x[i * 4 + 2] + float(bytes[3]) * shared_x[i * 4 + 3];
    }

    // Reduce across the 32 lanes using SIMD.
    acc = simd_sum(acc);

    // Lane 0 writes the scaled result.
    if (lane == 0) {
        y[n] = acc * scales[n];
    }
}
