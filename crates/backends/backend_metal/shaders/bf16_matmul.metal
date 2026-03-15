#include <metal_stdlib>
using namespace metal;

struct Bf16MatmulParams {
    uint M;
    uint K;
    uint N;
};

// BF16 matrix-matrix multiply for M>1 prefill.
// W is stored as packed uint32 where each uint holds two BF16 values:
//   low  16 bits -> w0 (BF16)
//   high 16 bits -> w1 (BF16)
// C[m, n] = sum_k A[m, k] * W[n, k]
// Dispatch: (N, M, 1) threadgroups of 32 threads.
// n = gid.x, m = gid.y
[[kernel]]
void bf16_matmul(
    device const float*    A        [[buffer(0)]],
    device const uint*     W_packed [[buffer(1)]],
    device float*          C        [[buffer(2)]],
    constant Bf16MatmulParams&       p        [[buffer(3)]],
    uint3                  tid3     [[thread_position_in_threadgroup]],
    uint3                  gid3     [[threadgroup_position_in_grid]]
) {
    uint lane = tid3.x;
    uint n = gid3.x;
    uint m = gid3.y;
    uint M = p.M;
    uint K = p.K;
    uint N = p.N;

    if (n >= N || m >= M) return;

    // Number of packed uint32 words per row of W: K/2.
    uint num_packed = K / 2;
    uint w_row_offset = n * num_packed;
    uint a_row_offset = m * K;

    float acc = 0.0f;

    // Each lane strides over the packed words.
    for (uint i = lane; i < num_packed; i += 32) {
        uint packed = W_packed[w_row_offset + i];

        // Decode two BF16 values.
        float w0 = as_type<float>((packed & 0xFFFFu) << 16);
        float w1 = as_type<float>(packed & 0xFFFF0000u);

        // Corresponding input elements from row m of A.
        float a0 = A[a_row_offset + i * 2 + 0];
        float a1 = A[a_row_offset + i * 2 + 1];

        acc += w0 * a0 + w1 * a1;
    }

    // Reduce across the 32 lanes using SIMD.
    acc = simd_sum(acc);

    // Lane 0 writes the result to C[m, n].
    if (lane == 0) {
        C[m * N + n] = acc;
    }
}
