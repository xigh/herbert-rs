#include <metal_stdlib>
using namespace metal;

struct MoeFusedGateUpSwigluInt8Params {
    uint K;
};

// Fused Gate + Up projection + SwiGLU activation for Int8 weights (MoE experts).
//
// For each output index n, computes:
//   gate_val = gate_scales[n] * dot(gate_w[n], x)
//   up_val   = up_scales[n] * dot(up_w[n], x)
//   output[n] = silu(gate_val) * up_val
//
// Int8 per-channel: y[n] = scales[n] * sum_k(w[n,k] * x[k])
// Both gate and up weights share the same input x (loaded once into shared memory).
//
// Dispatch: (ceil(N/4), 1, 1) threadgroups of 128 threads (4 rows per TG).
//
// Buffers:
//   x              : [K]    - float input vector
//   gate_w_packed  : [N, K/4] - int8 gate weights packed as uint32
//   gate_scales    : [N]    - f32 gate per-channel scales
//   up_w_packed    : [N, K/4] - int8 up weights packed as uint32
//   up_scales      : [N]    - f32 up per-channel scales
//   output         : [N]    - float output
[[kernel]]
void moe_fused_gate_up_swiglu_int8(
    device const float*    x              [[buffer(0)]],
    device const uint*     gate_w_packed  [[buffer(1)]],
    device const float*    gate_scales    [[buffer(2)]],
    device const uint*     up_w_packed    [[buffer(3)]],
    device const float*    up_scales      [[buffer(4)]],
    device float*          output         [[buffer(5)]],
    constant MoeFusedGateUpSwigluInt8Params& p [[buffer(6)]],
    threadgroup float*     shared_x       [[threadgroup(0)]],
    uint                   tid            [[thread_position_in_threadgroup]],
    uint                   gid            [[threadgroup_position_in_grid]]
) {
    uint K = p.K;
    uint num_packed = K / 4;

    uint warp_id = tid / 32;
    uint lane = tid % 32;
    uint n = gid * 4 + warp_id;

    // Load x into threadgroup memory (shared across all 4 rows).
    // shared_x size set dynamically via setThreadgroupMemoryLength (K * 4 bytes).
    for (uint i = tid; i < K; i += 128) {
        shared_x[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row_offset = n * num_packed;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (uint i = lane; i < num_packed; i += 32) {
        float x0 = shared_x[i * 4 + 0];
        float x1 = shared_x[i * 4 + 1];
        float x2 = shared_x[i * 4 + 2];
        float x3 = shared_x[i * 4 + 3];

        // Gate weights
        char4 gb = as_type<char4>(gate_w_packed[row_offset + i]);
        gate_acc += float(gb[0]) * x0 + float(gb[1]) * x1
                  + float(gb[2]) * x2 + float(gb[3]) * x3;

        // Up weights
        char4 ub = as_type<char4>(up_w_packed[row_offset + i]);
        up_acc += float(ub[0]) * x0 + float(ub[1]) * x1
                + float(ub[2]) * x2 + float(ub[3]) * x3;
    }

    gate_acc = simd_sum(gate_acc);
    up_acc = simd_sum(up_acc);

    if (lane == 0) {
        float gate_val = gate_acc * gate_scales[n];
        float up_val = up_acc * up_scales[n];
        float silu_gate = gate_val / (1.0f + exp(-gate_val));
        output[n] = silu_gate * up_val;
    }
}
