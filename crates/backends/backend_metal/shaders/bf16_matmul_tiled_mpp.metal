// BF16 matrix multiply using MetalPerformancePrimitives tensor_ops::matmul2d.
//
// Computes: C[m][n] = Σ_k A[m][k] × W[n][k]
//
// Same pattern as q4_matmul_tiled_mpp.metal but without dequant:
//   - 32×32×32 square tiles
//   - Double-buffered: overlap load K+1 with compute K
//   - BF16 unpack: uint32 → 2 halfs (shift + as_type)
//
// matmul2d(32, 32, 32, false, true, false, multiply_accumulate):
//   C[i][j] += Σ_k left[i][k] × right[j][k]   (right is transposed)
//
// Dispatch:
//   grid:    (ceil(N/32), ceil(M/32), 1)
//   threads: (128, 1, 1)   — 4 cooperative simdgroups
//   tg_mem:  12288 bytes
//
// Threadgroup memory layout (double-buffered, 12288 B total):
//   tg_a0:  [32×32] half  = 2048 B  (offset 0)
//   tg_a1:  [32×32] half  = 2048 B  (offset 2048)
//   tg_w0:  [32×32] half  = 2048 B  (offset 4096)
//   tg_w1:  [32×32] half  = 2048 B  (offset 6144)
//   tg_c:   [32×32] float = 4096 B  (offset 8192)
//
// Requires: MSL 4.0, MetalPerformancePrimitives

#if __METAL_VERSION__ >= 400

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct Bf16MatmulMppParams {
    uint M;   // rows of A (seq_len for prefill)
    uint N;   // rows of W / output columns
    uint K;   // cols of A = cols of W (hidden_dim)
};

[[kernel]]
void bf16_matmul_tiled_mpp(
    device const float*           a        [[buffer(0)]],  // [M, K] row-major float32
    device const uint*            w_packed [[buffer(1)]],  // [N, K/2] BF16 packed as uint32
    device float*                 c        [[buffer(2)]],  // [M, N] output float32
    constant Bf16MatmulMppParams& p        [[buffer(3)]],
    threadgroup char*             tg_raw   [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]
) {
    uint M = p.M;
    uint N = p.N;
    uint K = p.K;

    uint m_base = tgid.y * 32;
    uint n_base = tgid.x * 32;
    uint num_packed = K / 2;  // uint32s per weight row

    // Double-buffered threadgroup memory
    threadgroup half*  tg_a0 = (threadgroup half*)(tg_raw);            // 2048 B
    threadgroup half*  tg_a1 = (threadgroup half*)(tg_raw + 2048);     // 2048 B
    threadgroup half*  tg_w0 = (threadgroup half*)(tg_raw + 4096);     // 2048 B
    threadgroup half*  tg_w1 = (threadgroup half*)(tg_raw + 6144);     // 2048 B
    threadgroup float* tg_c  = (threadgroup float*)(tg_raw + 8192);    // 4096 B

    // matmul2d 32×32×32, right transposed, multiply-accumulate
    constexpr auto desc = matmul2d_descriptor(32, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    using LeftT  = tensor<threadgroup half, extents<int32_t, 32, 32>, tensor_inline>;
    using RightT = tensor<threadgroup half, extents<int32_t, 32, 32>, tensor_inline>;

    auto cT = op.get_destination_cooperative_tensor<LeftT, RightT, float>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0;
    }

    uint k_tiles = K / 32;

    // Pre-load first tile into slot 0
    {
        // A load f32→half
        for (uint i = tid; i < 32 * 32; i += 128) {
            uint m_local = i / 32;
            uint k_local = i % 32;
            uint m_idx = m_base + m_local;
            tg_a0[i] = (m_idx < M && k_local < K) ? half(a[m_idx * K + k_local]) : half(0);
        }
        // W unpack BF16: 2 elements per uint32, 16 uints per 32-wide K-tile
        for (uint i = tid; i < 32 * 16; i += 128) {
            uint n_local = i / 16;
            uint pack_idx = i % 16;   // which uint32 within this K-tile
            uint n_idx = n_base + n_local;
            uint k_idx = pack_idx * 2;
            uint dst = n_local * 32 + k_idx;
            if (n_idx < N && k_idx < K) {
                uint packed_val = w_packed[n_idx * num_packed + k_idx / 2];
                float w0 = as_type<float>((packed_val & 0xFFFFu) << 16);
                float w1 = as_type<float>(packed_val & 0xFFFF0000u);
                tg_w0[dst]     = half(w0);
                tg_w0[dst + 1] = half(w1);
            } else {
                tg_w0[dst]     = 0.0h;
                tg_w0[dst + 1] = 0.0h;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kt = 0; kt < k_tiles; kt++) {
        // Select current and next buffer slots
        threadgroup half* cur_a = (kt & 1) ? tg_a1 : tg_a0;
        threadgroup half* cur_w = (kt & 1) ? tg_w1 : tg_w0;
        threadgroup half* nxt_a = (kt & 1) ? tg_a0 : tg_a1;
        threadgroup half* nxt_w = (kt & 1) ? tg_w0 : tg_w1;

        // Compute current tile
        auto A_tile = LeftT(cur_a, extents<int32_t, 32, 32>());
        auto W_tile = RightT(cur_w, extents<int32_t, 32, 32>());
        op.run(A_tile, W_tile, cT);

        // Load next tile (if exists) into alternate buffer
        if (kt + 1 < k_tiles) {
            uint k_base = (kt + 1) * 32;
            // A load f32→half
            for (uint i = tid; i < 32 * 32; i += 128) {
                uint m_local = i / 32;
                uint k_local = i % 32;
                uint m_idx = m_base + m_local;
                uint k_idx = k_base + k_local;
                nxt_a[i] = (m_idx < M && k_idx < K) ? half(a[m_idx * K + k_idx]) : half(0);
            }
            // W unpack BF16
            for (uint i = tid; i < 32 * 16; i += 128) {
                uint n_local = i / 16;
                uint pack_idx = i % 16;
                uint n_idx = n_base + n_local;
                uint k_idx = k_base + pack_idx * 2;
                uint dst = n_local * 32 + pack_idx * 2;
                if (n_idx < N && k_idx < K) {
                    uint packed_val = w_packed[n_idx * num_packed + k_idx / 2];
                    float w0 = as_type<float>((packed_val & 0xFFFFu) << 16);
                    float w1 = as_type<float>(packed_val & 0xFFFF0000u);
                    nxt_w[dst]     = half(w0);
                    nxt_w[dst + 1] = half(w1);
                } else {
                    nxt_w[dst]     = 0.0h;
                    nxt_w[dst + 1] = 0.0h;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store cooperative tensor → threadgroup
    auto out_tensor = tensor<threadgroup float, extents<int32_t, 32, 32>, tensor_inline>(
        tg_c, extents<int32_t, 32, 32>());
    cT.store(out_tensor);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Copy TG → device with bounds checking
    for (uint i = tid; i < 32 * 32; i += 128) {
        uint m_idx = m_base + i / 32;
        uint n_idx = n_base + i % 32;
        if (m_idx < M && n_idx < N) {
            c[m_idx * N + n_idx] = tg_c[i];
        }
    }
}

#endif // __METAL_VERSION__ >= 400
