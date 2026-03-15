#include <metal_stdlib>
using namespace metal;

struct FusedGateUpSwigluParams {
    uint K;
};

// Fused Gate + Up projection + SwiGLU activation for BF16 weights.
//
// For each output index n, computes:
//   gate_val = dot(gate_w[n], x)
//   up_val   = dot(up_w[n], x)
//   output[n] = silu(gate_val) * up_val
//
// This eliminates 2 dispatches per MLP layer (gate matvec + up matvec + swiglu
// → single fused dispatch), and avoids writing/reading intermediate gate and
// up buffers from global memory.
//
// Dispatch: (ceil(N/4), 1, 1) threadgroups of 128 threads (4 rows per TG).
//
// Buffers:
//   x              : [K]      - float input vector
//   gate_w_packed  : [N, K/2] - BF16 gate weights packed as uint32
//   up_w_packed    : [N, K/2] - BF16 up weights packed as uint32
//   output         : [N]      - float output
[[kernel]]
void fused_gate_up_swiglu(
    device const float*    x              [[buffer(0)]],
    device const uint*     gate_w_packed  [[buffer(1)]],
    device const uint*     up_w_packed    [[buffer(2)]],
    device float*          output         [[buffer(3)]],
    constant FusedGateUpSwigluParams& p   [[buffer(4)]],
    threadgroup float*     shared_x       [[threadgroup(0)]],
    uint                   tid            [[thread_position_in_threadgroup]],
    uint                   gid            [[threadgroup_position_in_grid]]
) {
    uint K = p.K;
    uint num_packed = K / 2;

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

    // Accumulate gate and up dot products simultaneously.
    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (uint i = lane; i < num_packed; i += 32) {
        float x0 = shared_x[i * 2 + 0];
        float x1 = shared_x[i * 2 + 1];

        // Gate weights
        uint g_packed = gate_w_packed[row_offset + i];
        float gw0 = as_type<float>((g_packed & 0xFFFFu) << 16);
        float gw1 = as_type<float>(g_packed & 0xFFFF0000u);
        gate_acc += gw0 * x0 + gw1 * x1;

        // Up weights
        uint u_packed = up_w_packed[row_offset + i];
        float uw0 = as_type<float>((u_packed & 0xFFFFu) << 16);
        float uw1 = as_type<float>(u_packed & 0xFFFF0000u);
        up_acc += uw0 * x0 + uw1 * x1;
    }

    // Reduce across SIMD lanes.
    gate_acc = simd_sum(gate_acc);
    up_acc = simd_sum(up_acc);

    // Apply SwiGLU: silu(gate) * up
    if (lane == 0) {
        float silu_gate = gate_acc / (1.0f + exp(-gate_acc));
        output[n] = silu_gate * up_acc;
    }
}
