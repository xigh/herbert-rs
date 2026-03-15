#include <metal_stdlib>
using namespace metal;

struct MatvecScaledAddBf16Params {
    uint K;
    float scale;
};

// Fused BF16 matvec + scaled accumulation for MoE expert down projection.
//
// output[n] += scale * dot(w[n], x)
//
// Dispatch: (ceil(N/4), 1, 1) threadgroups of 128 threads (4 rows per TG).
//
// Buffers:
//   x        : [K]      - float input vector
//   w_packed : [N, K/2] - BF16 weights packed as uint32
//   output   : [N]      - float accumulator (read-modify-write)
[[kernel]]
void matvec_scaled_add_bf16(
    device const float*    x        [[buffer(0)]],
    device const uint*     w_packed [[buffer(1)]],
    device float*          output   [[buffer(2)]],
    constant MatvecScaledAddBf16Params& p [[buffer(3)]],
    threadgroup float*     shared_x [[threadgroup(0)]],
    uint                   tid      [[thread_position_in_threadgroup]],
    uint                   gid      [[threadgroup_position_in_grid]]
) {
    uint K = p.K;
    uint num_packed = K / 2;

    uint warp_id = tid / 32;
    uint lane = tid % 32;
    uint n = gid * 4 + warp_id;

    // shared_x size set dynamically via setThreadgroupMemoryLength (K * 4 bytes).
    for (uint i = tid; i < K; i += 128) {
        shared_x[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row_offset = n * num_packed;
    float acc = 0.0f;

    // 4x unrolled loop
    uint i = lane;
    for (; i + 96 < num_packed; i += 128) {
        uint packed0 = w_packed[row_offset + i];
        float w0_0 = as_type<float>((packed0 & 0xFFFFu) << 16);
        float w0_1 = as_type<float>(packed0 & 0xFFFF0000u);
        acc += w0_0 * shared_x[(i) * 2 + 0] + w0_1 * shared_x[(i) * 2 + 1];

        uint packed1 = w_packed[row_offset + i + 32];
        float w1_0 = as_type<float>((packed1 & 0xFFFFu) << 16);
        float w1_1 = as_type<float>(packed1 & 0xFFFF0000u);
        acc += w1_0 * shared_x[(i + 32) * 2 + 0] + w1_1 * shared_x[(i + 32) * 2 + 1];

        uint packed2 = w_packed[row_offset + i + 64];
        float w2_0 = as_type<float>((packed2 & 0xFFFFu) << 16);
        float w2_1 = as_type<float>(packed2 & 0xFFFF0000u);
        acc += w2_0 * shared_x[(i + 64) * 2 + 0] + w2_1 * shared_x[(i + 64) * 2 + 1];

        uint packed3 = w_packed[row_offset + i + 96];
        float w3_0 = as_type<float>((packed3 & 0xFFFFu) << 16);
        float w3_1 = as_type<float>(packed3 & 0xFFFF0000u);
        acc += w3_0 * shared_x[(i + 96) * 2 + 0] + w3_1 * shared_x[(i + 96) * 2 + 1];
    }

    for (; i < num_packed; i += 32) {
        uint packed = w_packed[row_offset + i];
        float w0 = as_type<float>((packed & 0xFFFFu) << 16);
        float w1 = as_type<float>(packed & 0xFFFF0000u);
        acc += w0 * shared_x[i * 2 + 0] + w1 * shared_x[i * 2 + 1];
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        output[n] += p.scale * acc;
    }
}
