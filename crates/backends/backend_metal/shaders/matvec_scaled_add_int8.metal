#include <metal_stdlib>
using namespace metal;

struct MatvecScaledAddInt8Params {
    uint K;
    float scale;
};

// Fused Int8 matvec + scaled accumulation for MoE expert down projection.
//
// output[n] += scale * (per_channel_scale[n] * dot(w[n], x))
//
// Dispatch: (ceil(N/4), 1, 1) threadgroups of 128 threads (4 rows per TG).
//
// Buffers:
//   x              : [K]    - float input vector
//   w_packed       : [N, K/4] - int8 weights packed as uint32
//   per_ch_scales  : [N]    - f32 per-channel dequant scales
//   output         : [N]    - float accumulator (read-modify-write)
[[kernel]]
void matvec_scaled_add_int8(
    device const float*    x              [[buffer(0)]],
    device const uint*     w_packed       [[buffer(1)]],
    device const float*    per_ch_scales  [[buffer(2)]],
    device float*          output         [[buffer(3)]],
    constant MatvecScaledAddInt8Params& p [[buffer(4)]],
    threadgroup float*     shared_x       [[threadgroup(0)]],
    uint                   tid            [[thread_position_in_threadgroup]],
    uint                   gid            [[threadgroup_position_in_grid]]
) {
    uint K = p.K;
    uint num_packed = K / 4;

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
        char4 b0 = as_type<char4>(w_packed[row_offset + i]);
        acc += float(b0[0]) * shared_x[(i) * 4 + 0] + float(b0[1]) * shared_x[(i) * 4 + 1]
             + float(b0[2]) * shared_x[(i) * 4 + 2] + float(b0[3]) * shared_x[(i) * 4 + 3];

        char4 b1 = as_type<char4>(w_packed[row_offset + i + 32]);
        acc += float(b1[0]) * shared_x[(i + 32) * 4 + 0] + float(b1[1]) * shared_x[(i + 32) * 4 + 1]
             + float(b1[2]) * shared_x[(i + 32) * 4 + 2] + float(b1[3]) * shared_x[(i + 32) * 4 + 3];

        char4 b2 = as_type<char4>(w_packed[row_offset + i + 64]);
        acc += float(b2[0]) * shared_x[(i + 64) * 4 + 0] + float(b2[1]) * shared_x[(i + 64) * 4 + 1]
             + float(b2[2]) * shared_x[(i + 64) * 4 + 2] + float(b2[3]) * shared_x[(i + 64) * 4 + 3];

        char4 b3 = as_type<char4>(w_packed[row_offset + i + 96]);
        acc += float(b3[0]) * shared_x[(i + 96) * 4 + 0] + float(b3[1]) * shared_x[(i + 96) * 4 + 1]
             + float(b3[2]) * shared_x[(i + 96) * 4 + 2] + float(b3[3]) * shared_x[(i + 96) * 4 + 3];
    }

    for (; i < num_packed; i += 32) {
        char4 bytes = as_type<char4>(w_packed[row_offset + i]);
        acc += float(bytes[0]) * shared_x[i * 4 + 0] + float(bytes[1]) * shared_x[i * 4 + 1]
             + float(bytes[2]) * shared_x[i * 4 + 2] + float(bytes[3]) * shared_x[i * 4 + 3];
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        output[n] += p.scale * acc * per_ch_scales[n];
    }
}
