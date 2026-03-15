#include <metal_stdlib>
using namespace metal;

struct F32MatvecParams {
    uint K;
};

// F32 matrix-vector multiply with multi-row processing and shared input.
//
// 4 rows per threadgroup (128 threads = 4 SIMD groups of 32).
// The input vector x is loaded once into threadgroup memory and shared
// across all 4 rows, reducing global memory traffic.
//
// y[n] = sum_k W[n, k] * x[k]
//
// Dispatch: (ceil(N/4), 1, 1) threadgroups of 128 threads.
// Each threadgroup handles 4 consecutive output rows.
[[kernel]]
void f32_matvec(
    device const float*    x    [[buffer(0)]],
    device const float*    W    [[buffer(1)]],
    device float*          y    [[buffer(2)]],
    constant F32MatvecParams&       p    [[buffer(3)]],
    threadgroup float*     shared_x [[threadgroup(0)]],
    uint                   tid  [[thread_position_in_threadgroup]],
    uint                   gid  [[threadgroup_position_in_grid]]
) {
    uint K = p.K;

    // Identify which SIMD group (warp) and lane within it.
    uint warp_id = tid / 32;
    uint lane = tid % 32;

    // Which output row this warp handles.
    uint n = gid * 4 + warp_id;
    uint row_offset = n * K;

    // Load x into threadgroup memory (cooperative load across all 128 threads).
    // shared_x size set dynamically via setThreadgroupMemoryLength (K * 4 bytes).
    for (uint i = tid; i < K; i += 128) {
        shared_x[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;

    // 4-element unrolled inner loop; each lane strides by 32*4 = 128.
    uint i = lane * 4;
    for (; i + 3 < K; i += 32 * 4) {
        acc += W[row_offset + i + 0] * shared_x[i + 0];
        acc += W[row_offset + i + 1] * shared_x[i + 1];
        acc += W[row_offset + i + 2] * shared_x[i + 2];
        acc += W[row_offset + i + 3] * shared_x[i + 3];
    }

    // Handle remainder elements (when K is not a multiple of 128).
    uint base = (K / (32 * 4)) * (32 * 4);
    for (uint r = base + lane; r < K; r += 32) {
        acc += W[row_offset + r] * shared_x[r];
    }

    // Reduce across the 32 lanes using SIMD.
    acc = simd_sum(acc);

    // Lane 0 writes the result.
    if (lane == 0) {
        y[n] = acc;
    }
}
