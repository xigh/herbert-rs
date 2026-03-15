#include <metal_stdlib>
using namespace metal;

struct F32MatmulParams {
    uint M;
    uint K;
    uint N;
};

// F32 matrix-matrix multiply for MoE router.
// C[m, n] = sum_k A[m, k] * W[n, k]
// Dispatch: (N, M, 1) threadgroups of 32 threads.
// n = gid.x, m = gid.y
[[kernel]]
void f32_matmul(
    device const float*    A    [[buffer(0)]],
    device const float*    W    [[buffer(1)]],
    device float*          C    [[buffer(2)]],
    constant F32MatmulParams&       p    [[buffer(3)]],
    uint3                  tid3 [[thread_position_in_threadgroup]],
    uint3                  gid3 [[threadgroup_position_in_grid]]
) {
    uint lane = tid3.x;
    uint n = gid3.x;
    uint m = gid3.y;
    uint M = p.M;
    uint K = p.K;
    uint N = p.N;

    if (n >= N || m >= M) return;

    uint w_row_offset = n * K;
    uint a_row_offset = m * K;

    float acc = 0.0f;

    // 4-element unrolled inner loop; each lane strides by 32*4 = 128.
    uint i = lane * 4;
    for (; i + 3 < K; i += 32 * 4) {
        acc += W[w_row_offset + i + 0] * A[a_row_offset + i + 0];
        acc += W[w_row_offset + i + 1] * A[a_row_offset + i + 1];
        acc += W[w_row_offset + i + 2] * A[a_row_offset + i + 2];
        acc += W[w_row_offset + i + 3] * A[a_row_offset + i + 3];
    }

    // Handle remainder elements (when K is not a multiple of 128).
    uint base = (K / (32 * 4)) * (32 * 4);
    for (uint r = base + lane; r < K; r += 32) {
        acc += W[w_row_offset + r] * A[a_row_offset + r];
    }

    // Reduce across the 32 lanes using SIMD.
    acc = simd_sum(acc);

    // Lane 0 writes the result to C[m, n].
    if (lane == 0) {
        C[m * N + n] = acc;
    }
}
