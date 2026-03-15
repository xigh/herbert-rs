#include <metal_stdlib>
using namespace metal;

struct MoeReduceParams {
    uint hidden;   // output dimension
    uint top_k;    // number of experts to sum
};

// Reduce MoE expert outputs: output[n] = sum over top_k experts of per_expert_output[e * hidden + n].
//
// No atomics needed — each output element is written by exactly one thread.
//
// Dispatch: grid = (ceil(hidden/256), 1, 1), threads = (256, 1, 1)
//
// Buffers:
//   per_expert_output : [top_k, hidden] - f32 weighted expert outputs
//   output            : [hidden]        - f32 reduced output
[[kernel]]
void moe_reduce(
    device const float*    per_expert_output [[buffer(0)]],
    device float*          output            [[buffer(1)]],
    constant MoeReduceParams& p             [[buffer(2)]],
    uint                   tid              [[thread_position_in_grid]]
) {
    if (tid >= p.hidden) return;

    float sum = 0.0f;
    for (uint e = 0; e < p.top_k; e++) {
        sum += per_expert_output[e * p.hidden + tid];
    }
    output[tid] = sum;
}
