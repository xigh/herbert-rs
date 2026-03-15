#include <metal_stdlib>
using namespace metal;

struct MoeBatchedGateUpSwigluInt8Params {
    uint moe_inter;
    uint K;
    uint top_k;
};

// Batched Gate + Up + SwiGLU for Int8 MoE experts with contiguous weight storage.
//
// Int8 per-channel: y[n] = scales[n] * sum_k(w[n,k] * x[k])
//
// Dispatch: grid = (ceil(moe_inter/4), top_k, 1), threads = (128, 1, 1)
[[kernel]]
void moe_batched_gate_up_swiglu_int8(
    device const float*    x                [[buffer(0)]],
    device const uint*     gate_packed_all  [[buffer(1)]],
    device const float*    gate_scales_all  [[buffer(2)]],
    device const uint*     up_packed_all    [[buffer(3)]],
    device const float*    up_scales_all    [[buffer(4)]],
    device const uint*     expert_ids       [[buffer(5)]],
    device float*          output           [[buffer(6)]],
    constant MoeBatchedGateUpSwigluInt8Params& p [[buffer(7)]],
    threadgroup float*     shared_x         [[threadgroup(0)]],
    uint2                  tid_2d           [[thread_position_in_threadgroup]],
    uint2                  gid              [[threadgroup_position_in_grid]]
) {
    uint tid = tid_2d.x;
    uint N = p.moe_inter;
    uint K = p.K;
    uint num_packed = K / 4;

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

    // 4x unrolled loop
    uint i = lane;
    for (; i + 96 < num_packed; i += 128) {
        char4 gb0 = as_type<char4>(gate_packed_all[row_offset + i]);
        char4 ub0 = as_type<char4>(up_packed_all[row_offset + i]);
        gate_acc += float(gb0[0]) * shared_x[(i) * 4 + 0] + float(gb0[1]) * shared_x[(i) * 4 + 1]
                  + float(gb0[2]) * shared_x[(i) * 4 + 2] + float(gb0[3]) * shared_x[(i) * 4 + 3];
        up_acc   += float(ub0[0]) * shared_x[(i) * 4 + 0] + float(ub0[1]) * shared_x[(i) * 4 + 1]
                  + float(ub0[2]) * shared_x[(i) * 4 + 2] + float(ub0[3]) * shared_x[(i) * 4 + 3];

        char4 gb1 = as_type<char4>(gate_packed_all[row_offset + i + 32]);
        char4 ub1 = as_type<char4>(up_packed_all[row_offset + i + 32]);
        gate_acc += float(gb1[0]) * shared_x[(i + 32) * 4 + 0] + float(gb1[1]) * shared_x[(i + 32) * 4 + 1]
                  + float(gb1[2]) * shared_x[(i + 32) * 4 + 2] + float(gb1[3]) * shared_x[(i + 32) * 4 + 3];
        up_acc   += float(ub1[0]) * shared_x[(i + 32) * 4 + 0] + float(ub1[1]) * shared_x[(i + 32) * 4 + 1]
                  + float(ub1[2]) * shared_x[(i + 32) * 4 + 2] + float(ub1[3]) * shared_x[(i + 32) * 4 + 3];

        char4 gb2 = as_type<char4>(gate_packed_all[row_offset + i + 64]);
        char4 ub2 = as_type<char4>(up_packed_all[row_offset + i + 64]);
        gate_acc += float(gb2[0]) * shared_x[(i + 64) * 4 + 0] + float(gb2[1]) * shared_x[(i + 64) * 4 + 1]
                  + float(gb2[2]) * shared_x[(i + 64) * 4 + 2] + float(gb2[3]) * shared_x[(i + 64) * 4 + 3];
        up_acc   += float(ub2[0]) * shared_x[(i + 64) * 4 + 0] + float(ub2[1]) * shared_x[(i + 64) * 4 + 1]
                  + float(ub2[2]) * shared_x[(i + 64) * 4 + 2] + float(ub2[3]) * shared_x[(i + 64) * 4 + 3];

        char4 gb3 = as_type<char4>(gate_packed_all[row_offset + i + 96]);
        char4 ub3 = as_type<char4>(up_packed_all[row_offset + i + 96]);
        gate_acc += float(gb3[0]) * shared_x[(i + 96) * 4 + 0] + float(gb3[1]) * shared_x[(i + 96) * 4 + 1]
                  + float(gb3[2]) * shared_x[(i + 96) * 4 + 2] + float(gb3[3]) * shared_x[(i + 96) * 4 + 3];
        up_acc   += float(ub3[0]) * shared_x[(i + 96) * 4 + 0] + float(ub3[1]) * shared_x[(i + 96) * 4 + 1]
                  + float(ub3[2]) * shared_x[(i + 96) * 4 + 2] + float(ub3[3]) * shared_x[(i + 96) * 4 + 3];
    }

    // Remainder loop
    for (; i < num_packed; i += 32) {
        char4 gb = as_type<char4>(gate_packed_all[row_offset + i]);
        gate_acc += float(gb[0]) * shared_x[i * 4 + 0] + float(gb[1]) * shared_x[i * 4 + 1]
                  + float(gb[2]) * shared_x[i * 4 + 2] + float(gb[3]) * shared_x[i * 4 + 3];

        char4 ub = as_type<char4>(up_packed_all[row_offset + i]);
        up_acc += float(ub[0]) * shared_x[i * 4 + 0] + float(ub[1]) * shared_x[i * 4 + 1]
                + float(ub[2]) * shared_x[i * 4 + 2] + float(ub[3]) * shared_x[i * 4 + 3];
    }

    gate_acc = simd_sum(gate_acc);
    up_acc = simd_sum(up_acc);

    if (lane == 0) {
        float gate_val = gate_acc * gate_scales_all[expert_row];
        float up_val = up_acc * up_scales_all[expert_row];
        float silu_gate = gate_val / (1.0f + exp(-gate_val));
        output[expert_idx * N + n] = silu_gate * up_val;
    }
}
