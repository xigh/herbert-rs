#include <metal_stdlib>
using namespace metal;

struct Int8MatmulParams {
    uint M;
    uint K;
    uint N;
};

// Int8 per-channel matrix-matrix multiply for M>1.
// W is stored as packed uint32 where each uint holds 4 signed int8 bytes.
// C[m, n] = scales[n] * sum_k W[n, k] * A[m, k]
//
// Dispatch: (N, M, 1) threadgroups of 32 threads.
// n = gid.x, m = gid.y
//
// Buffers:
//   A        : [M, K]      - float input matrix
//   W_packed : [N, K/4]    - int8 weights packed as uint32 (4 bytes per uint)
//   scales   : [N]         - per-output-row dequantization scales
//   C        : [M, N]      - float output matrix
[[kernel]]
void int8_matmul(
    device const float*    A        [[buffer(0)]],
    device const uint*     W_packed [[buffer(1)]],
    device const float*    scales   [[buffer(2)]],
    device float*          C        [[buffer(3)]],
    constant Int8MatmulParams&       p        [[buffer(4)]],
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

    // Each uint32 packs 4 int8 values.
    uint num_packed    = K / 4;
    uint w_row_offset  = n * num_packed;
    uint a_row_offset  = m * K;

    float acc = 0.0f;

    // Each lane strides over the packed words.
    for (uint i = lane; i < num_packed; i += 32) {
        uint packed = W_packed[w_row_offset + i];

        // Extract 4 signed int8 values using char4 reinterpretation.
        char4 bytes = as_type<char4>(packed);

        float w0 = float(bytes[0]);
        float w1 = float(bytes[1]);
        float w2 = float(bytes[2]);
        float w3 = float(bytes[3]);

        float a0 = A[a_row_offset + i * 4 + 0];
        float a1 = A[a_row_offset + i * 4 + 1];
        float a2 = A[a_row_offset + i * 4 + 2];
        float a3 = A[a_row_offset + i * 4 + 3];

        acc += w0 * a0 + w1 * a1 + w2 * a2 + w3 * a3;
    }

    // Handle remainder (when K is not a multiple of 4).
    uint remainder_start = num_packed * 4;
    for (uint r = remainder_start + lane; r < K; r += 32) {
        uint word_idx  = r / 4;
        uint byte_idx  = r % 4;
        uint packed    = W_packed[w_row_offset + word_idx];
        char4 bytes    = as_type<char4>(packed);
        float w_val    = float(bytes[byte_idx]);
        acc += w_val * A[a_row_offset + r];
    }

    // Reduce across the 32 lanes using SIMD.
    acc = simd_sum(acc);

    // Lane 0 writes the scaled result to C[m, n].
    if (lane == 0) {
        C[m * N + n] = acc * scales[n];
    }
}
