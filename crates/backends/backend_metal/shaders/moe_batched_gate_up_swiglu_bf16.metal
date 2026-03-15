#include <metal_stdlib>
using namespace metal;

struct MoeBatchedGateUpSwigluBf16Params {
    uint moe_inter;
    uint K;
    uint top_k;
};

// Batched Gate + Up + SwiGLU for BF16 MoE experts with contiguous weight storage.
//
// BF16 decode: packed is uint32 with two BF16 values.
// Low BF16: as_type<float>((packed & 0xFFFF) << 16)
// High BF16: as_type<float>(packed & 0xFFFF0000)
//
// Dispatch: grid = (ceil(moe_inter/4), top_k, 1), threads = (128, 1, 1)
[[kernel]]
void moe_batched_gate_up_swiglu_bf16(
    device const float*    x                [[buffer(0)]],
    device const uint*     gate_packed_all  [[buffer(1)]],
    device const uint*     up_packed_all    [[buffer(2)]],
    device const uint*     expert_ids       [[buffer(3)]],
    device float*          output           [[buffer(4)]],
    constant MoeBatchedGateUpSwigluBf16Params& p [[buffer(5)]],
    threadgroup float*     shared_x         [[threadgroup(0)]],
    uint2                  tid_2d           [[thread_position_in_threadgroup]],
    uint2                  gid              [[threadgroup_position_in_grid]]
) {
    uint tid = tid_2d.x;
    uint N = p.moe_inter;
    uint K = p.K;
    uint num_packed = K / 2;  // uint32 pairs

    uint expert_idx = gid.y;
    uint expert_id = expert_ids[expert_idx];

    uint warp_id = tid / 32;
    uint lane = tid % 32;
    uint n = gid.x * 4 + warp_id;

    if (n >= N) return;

    // shared_x size set dynamically via setThreadgroupMemoryLength (K * 4 bytes).
    for (uint i = tid; i < K; i += 128) {
        shared_x[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint expert_row = expert_id * N + n;
    uint row_offset = expert_row * num_packed;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (uint i = lane; i < num_packed; i += 32) {
        float x0 = shared_x[i * 2 + 0];
        float x1 = shared_x[i * 2 + 1];

        uint g_packed = gate_packed_all[row_offset + i];
        float gw0 = as_type<float>((g_packed & 0xFFFFu) << 16);
        float gw1 = as_type<float>(g_packed & 0xFFFF0000u);
        gate_acc += gw0 * x0 + gw1 * x1;

        uint u_packed = up_packed_all[row_offset + i];
        float uw0 = as_type<float>((u_packed & 0xFFFFu) << 16);
        float uw1 = as_type<float>(u_packed & 0xFFFF0000u);
        up_acc += uw0 * x0 + uw1 * x1;
    }

    gate_acc = simd_sum(gate_acc);
    up_acc = simd_sum(up_acc);

    if (lane == 0) {
        float silu_gate = gate_acc / (1.0f + exp(-gate_acc));
        output[expert_idx * N + n] = silu_gate * up_acc;
    }
}
