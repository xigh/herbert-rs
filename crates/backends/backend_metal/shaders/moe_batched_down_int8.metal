#include <metal_stdlib>
using namespace metal;

struct MoeBatchedDownInt8Params {
    uint hidden;
    uint moe_inter;
    uint top_k;
};

// Batched Down projection for Int8 MoE experts with contiguous weight storage.
//
// Dispatch: grid = (ceil(hidden/4), top_k, 1), threads = (128, 1, 1)
[[kernel]]
void moe_batched_down_int8(
    device const float*    inputs           [[buffer(0)]],
    device const uint*     down_packed_all  [[buffer(1)]],
    device const float*    down_scales_all  [[buffer(2)]],
    device const uint*     expert_ids       [[buffer(3)]],
    device const float*    expert_weights   [[buffer(4)]],
    device float*          output           [[buffer(5)]],
    constant MoeBatchedDownInt8Params& p    [[buffer(6)]],
    threadgroup float*     shared_x         [[threadgroup(0)]],
    uint2                  tid_2d           [[thread_position_in_threadgroup]],
    uint2                  gid              [[threadgroup_position_in_grid]]
) {
    uint tid = tid_2d.x;
    uint hidden = p.hidden;
    uint K = p.moe_inter;
    uint num_packed = K / 4;

    uint expert_idx = gid.y;
    uint expert_id = expert_ids[expert_idx];
    float routing_weight = expert_weights[expert_idx];

    uint warp_id = tid / 32;
    uint lane = tid % 32;
    uint n = gid.x * 4 + warp_id;

    if (n >= hidden) return;

    // shared_x size set dynamically via setThreadgroupMemoryLength (K * 4 bytes).
    device const float* expert_input = inputs + expert_idx * K;
    for (uint i = tid; i < K; i += 128) {
        shared_x[i] = expert_input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint expert_row = expert_id * hidden + n;
    uint row_offset = expert_row * num_packed;
    float acc = 0.0f;

    // 4x unrolled loop
    uint i = lane;
    for (; i + 96 < num_packed; i += 128) {
        char4 b0 = as_type<char4>(down_packed_all[row_offset + i]);
        acc += float(b0[0]) * shared_x[(i) * 4 + 0] + float(b0[1]) * shared_x[(i) * 4 + 1]
             + float(b0[2]) * shared_x[(i) * 4 + 2] + float(b0[3]) * shared_x[(i) * 4 + 3];

        char4 b1 = as_type<char4>(down_packed_all[row_offset + i + 32]);
        acc += float(b1[0]) * shared_x[(i + 32) * 4 + 0] + float(b1[1]) * shared_x[(i + 32) * 4 + 1]
             + float(b1[2]) * shared_x[(i + 32) * 4 + 2] + float(b1[3]) * shared_x[(i + 32) * 4 + 3];

        char4 b2 = as_type<char4>(down_packed_all[row_offset + i + 64]);
        acc += float(b2[0]) * shared_x[(i + 64) * 4 + 0] + float(b2[1]) * shared_x[(i + 64) * 4 + 1]
             + float(b2[2]) * shared_x[(i + 64) * 4 + 2] + float(b2[3]) * shared_x[(i + 64) * 4 + 3];

        char4 b3 = as_type<char4>(down_packed_all[row_offset + i + 96]);
        acc += float(b3[0]) * shared_x[(i + 96) * 4 + 0] + float(b3[1]) * shared_x[(i + 96) * 4 + 1]
             + float(b3[2]) * shared_x[(i + 96) * 4 + 2] + float(b3[3]) * shared_x[(i + 96) * 4 + 3];
    }

    for (; i < num_packed; i += 32) {
        char4 bytes = as_type<char4>(down_packed_all[row_offset + i]);
        acc += float(bytes[0]) * shared_x[i * 4 + 0] + float(bytes[1]) * shared_x[i * 4 + 1]
             + float(bytes[2]) * shared_x[i * 4 + 2] + float(bytes[3]) * shared_x[i * 4 + 3];
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        output[expert_idx * hidden + n] = routing_weight * acc * down_scales_all[expert_row];
    }
}
