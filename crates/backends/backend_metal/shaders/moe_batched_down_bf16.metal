#include <metal_stdlib>
using namespace metal;

struct MoeBatchedDownBf16Params {
    uint hidden;
    uint moe_inter;
    uint top_k;
};

// Batched Down projection for BF16 MoE experts with contiguous weight storage.
//
// Dispatch: grid = (ceil(hidden/4), top_k, 1), threads = (128, 1, 1)
[[kernel]]
void moe_batched_down_bf16(
    device const float*    inputs           [[buffer(0)]],
    device const uint*     down_packed_all  [[buffer(1)]],
    device const uint*     expert_ids       [[buffer(2)]],
    device const float*    expert_weights   [[buffer(3)]],
    device float*          output           [[buffer(4)]],
    constant MoeBatchedDownBf16Params& p    [[buffer(5)]],
    threadgroup float*     shared_x         [[threadgroup(0)]],
    uint2                  tid_2d           [[thread_position_in_threadgroup]],
    uint2                  gid              [[threadgroup_position_in_grid]]
) {
    uint tid = tid_2d.x;
    uint hidden = p.hidden;
    uint K = p.moe_inter;
    uint num_packed = K / 2;

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
        uint packed0 = down_packed_all[row_offset + i];
        float w0_0 = as_type<float>((packed0 & 0xFFFFu) << 16);
        float w0_1 = as_type<float>(packed0 & 0xFFFF0000u);
        acc += w0_0 * shared_x[(i) * 2 + 0] + w0_1 * shared_x[(i) * 2 + 1];

        uint packed1 = down_packed_all[row_offset + i + 32];
        float w1_0 = as_type<float>((packed1 & 0xFFFFu) << 16);
        float w1_1 = as_type<float>(packed1 & 0xFFFF0000u);
        acc += w1_0 * shared_x[(i + 32) * 2 + 0] + w1_1 * shared_x[(i + 32) * 2 + 1];

        uint packed2 = down_packed_all[row_offset + i + 64];
        float w2_0 = as_type<float>((packed2 & 0xFFFFu) << 16);
        float w2_1 = as_type<float>(packed2 & 0xFFFF0000u);
        acc += w2_0 * shared_x[(i + 64) * 2 + 0] + w2_1 * shared_x[(i + 64) * 2 + 1];

        uint packed3 = down_packed_all[row_offset + i + 96];
        float w3_0 = as_type<float>((packed3 & 0xFFFFu) << 16);
        float w3_1 = as_type<float>(packed3 & 0xFFFF0000u);
        acc += w3_0 * shared_x[(i + 96) * 2 + 0] + w3_1 * shared_x[(i + 96) * 2 + 1];
    }

    for (; i < num_packed; i += 32) {
        uint packed = down_packed_all[row_offset + i];
        float w0 = as_type<float>((packed & 0xFFFFu) << 16);
        float w1 = as_type<float>(packed & 0xFFFF0000u);
        acc += w0 * shared_x[i * 2 + 0] + w1 * shared_x[i * 2 + 1];
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        output[expert_idx * hidden + n] = routing_weight * acc;
    }
}
