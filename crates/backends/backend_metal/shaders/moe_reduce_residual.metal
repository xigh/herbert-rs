#include <metal_stdlib>
using namespace metal;

struct MoeReduceResidualParams {
    uint hidden;   // output dimension
    uint top_k;    // number of experts to sum
};

// Fused MoE reduce + residual add.
//
// Combines moe_reduce and residual_add into a single dispatch:
//   residual[n] += sum over top_k experts of per_expert_output[e * hidden + n]
//
// Dispatch: grid = (ceil(hidden/256), 1, 1), threads = (256, 1, 1)
//
// Buffers:
//   per_expert_output : [top_k, hidden] - f32 weighted expert outputs
//   residual          : [hidden]        - f32 residual (accumulated in-place)
[[kernel]]
void moe_reduce_residual(
    device const float*    per_expert_output [[buffer(0)]],
    device float*          residual          [[buffer(1)]],
    constant MoeReduceResidualParams& p     [[buffer(2)]],
    uint                   tid              [[thread_position_in_grid]]
) {
    if (tid >= p.hidden) return;

    float sum = 0.0f;
    for (uint e = 0; e < p.top_k; e++) {
        sum += per_expert_output[e * p.hidden + tid];
    }
    residual[tid] += sum;
}
