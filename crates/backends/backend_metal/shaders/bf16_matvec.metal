#include <metal_stdlib>
using namespace metal;

struct Bf16MatvecParams {
    uint K;
};

// BF16 matrix-vector multiply with multi-row processing and shared input.
//
// 4 rows per threadgroup (128 threads = 4 SIMD groups of 32).
// The input vector x is loaded once into threadgroup memory and shared
// across all 4 rows, reducing global memory traffic.
//
// W is stored as packed uint32 where each uint holds two BF16 values:
//   low  16 bits -> w0 (BF16)
//   high 16 bits -> w1 (BF16)
//
// Dispatch: (ceil(N/4), 1, 1) threadgroups of 128 threads.
// Each threadgroup handles 4 consecutive output rows.

[[kernel]]
void bf16_matvec(
    device const float*    x        [[buffer(0)]],
    device const uint*     w_packed [[buffer(1)]],
    device float*          y        [[buffer(2)]],
    constant Bf16MatvecParams&       p        [[buffer(3)]],
    threadgroup float*     shared_x [[threadgroup(0)]],
    uint                   tid      [[thread_position_in_threadgroup]],
    uint                   gid      [[threadgroup_position_in_grid]]
) {
    uint K = p.K;
    uint num_packed = K / 2;

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

    // 4x unrolled loop: process 4 packed pairs (8 BF16 values) per iteration.
    uint i = lane;
    for (; i + 96 < num_packed; i += 128) {
        // Iteration 0
        uint packed0 = w_packed[row_offset + i];
        float w0_0 = as_type<float>((packed0 & 0xFFFFu) << 16);
        float w0_1 = as_type<float>(packed0 & 0xFFFF0000u);
        acc += w0_0 * shared_x[(i) * 2 + 0] + w0_1 * shared_x[(i) * 2 + 1];

        // Iteration 1
        uint packed1 = w_packed[row_offset + i + 32];
        float w1_0 = as_type<float>((packed1 & 0xFFFFu) << 16);
        float w1_1 = as_type<float>(packed1 & 0xFFFF0000u);
        acc += w1_0 * shared_x[(i + 32) * 2 + 0] + w1_1 * shared_x[(i + 32) * 2 + 1];

        // Iteration 2
        uint packed2 = w_packed[row_offset + i + 64];
        float w2_0 = as_type<float>((packed2 & 0xFFFFu) << 16);
        float w2_1 = as_type<float>(packed2 & 0xFFFF0000u);
        acc += w2_0 * shared_x[(i + 64) * 2 + 0] + w2_1 * shared_x[(i + 64) * 2 + 1];

        // Iteration 3
        uint packed3 = w_packed[row_offset + i + 96];
        float w3_0 = as_type<float>((packed3 & 0xFFFFu) << 16);
        float w3_1 = as_type<float>(packed3 & 0xFFFF0000u);
        acc += w3_0 * shared_x[(i + 96) * 2 + 0] + w3_1 * shared_x[(i + 96) * 2 + 1];
    }

    // Handle remainder (non-unrolled).
    for (; i < num_packed; i += 32) {
        uint packed = w_packed[row_offset + i];
        float w0 = as_type<float>((packed & 0xFFFFu) << 16);
        float w1 = as_type<float>(packed & 0xFFFF0000u);
        acc += w0 * shared_x[i * 2 + 0] + w1 * shared_x[i * 2 + 1];
    }

    // Reduce across the 32 lanes using SIMD.
    acc = simd_sum(acc);

    // Lane 0 writes the result.
    if (lane == 0) {
        y[n] = acc;
    }
}
