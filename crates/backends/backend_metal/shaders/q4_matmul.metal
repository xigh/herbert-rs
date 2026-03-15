#include <metal_stdlib>
using namespace metal;

struct Q4MatmulParams {
    uint M;
    uint K;
    uint N;
};

// Q4 matrix-matrix multiply for M>1 prefill.
//
// W is stored as packed bytes where each byte holds two 4-bit nibbles:
//   low  4 bits -> nibble for k+0
//   high 4 bits -> nibble for k+1
// Layout: [N, K/2] bytes
//
// Scales are per-group (group_size=32):
//   scales[n * n_groups + g] where g = k / 32, n_groups = ceil(K / 32)
//
// Dequant: value = (float(nibble) - 8.0) * scales[n * n_groups + k/32]
//
// C[m, n] = sum_k A[m, k] * W[n, k]
//
// Dispatch: (N, M, 1) threadgroups of 32 threads.
// n = gid.x, m = gid.y
//
// Buffers:
//   A        : [M, K]           - float input matrix
//   W_packed : [N, K/2]         - Q4 weights packed as bytes (2 nibbles per byte)
//   scales   : [N, n_groups]    - per-group dequantization scales (f32)
//   C        : [M, N]           - float output matrix

[[kernel]]
void q4_matmul(
    device const float*    A        [[buffer(0)]],
    device const uchar*    W_packed [[buffer(1)]],
    device const float*    scales   [[buffer(2)]],
    device float*          C        [[buffer(3)]],
    constant Q4MatmulParams&       p        [[buffer(4)]],
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

    // Number of packed bytes per row of W: K/2.
    uint num_bytes = K / 2;
    uint n_groups = (K + 31) / 32;  // number of scale groups per row
    uint w_row_offset = n * num_bytes;
    uint scale_row_offset = n * n_groups;
    uint a_row_offset = m * K;

    float acc = 0.0f;

    // Each lane strides over the packed bytes.
    for (uint i = lane; i < num_bytes; i += 32) {
        uchar byte_val = W_packed[w_row_offset + i];

        // Each byte contains two Q4 nibbles.
        uint k = i * 2;

        // Dequantize: value = (nibble - 8) * scale[group]
        float s_lo = scales[scale_row_offset + k / 32];
        float s_hi = scales[scale_row_offset + (k + 1) / 32];

        float w0 = (float(byte_val & 0xFu) - 8.0f) * s_lo;
        float w1 = (float(byte_val >> 4) - 8.0f) * s_hi;

        // Corresponding input elements from row m of A.
        float a0 = A[a_row_offset + k];
        float a1 = A[a_row_offset + k + 1];

        acc += w0 * a0 + w1 * a1;
    }

    // Reduce across the 32 lanes using SIMD.
    acc = simd_sum(acc);

    // Lane 0 writes the result to C[m, n].
    if (lane == 0) {
        C[m * N + n] = acc;
    }
}
