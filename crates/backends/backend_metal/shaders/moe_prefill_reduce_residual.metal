#include <metal_stdlib>
using namespace metal;

struct MoePrefillReduceResidualParams {
    uint hidden;      // output dimension
    uint top_k;       // number of experts to sum per token
    uint num_tokens;  // M (number of tokens)
};

// Fused batched MoE reduce + residual add for prefill.
//
// For each token, sums the weighted outputs from all top_k experts and adds
// to the residual stream.
//
//   residual[t * hidden + n] += sum_{e=0..top_k-1} down_out[(t * top_k + e) * hidden + n]
//
// Dispatch: grid = (ceil(hidden/256), num_tokens, 1), threads = (256, 1, 1)
// gid.y = token index
//
// Buffers:
//   per_expert_output : [num_tokens * top_k, hidden] - f32 weighted expert outputs
//   residual          : [num_tokens, hidden]          - f32 residual (accumulated in-place)
[[kernel]]
void moe_prefill_reduce_residual(
    device const float*    per_expert_output [[buffer(0)]],
    device float*          residual          [[buffer(1)]],
    constant MoePrefillReduceResidualParams& p [[buffer(2)]],
    uint2                  tid_2d           [[thread_position_in_threadgroup]],
    uint2                  gid              [[threadgroup_position_in_grid]]
) {
    uint token = gid.y;
    uint n = gid.x * 256 + tid_2d.x;

    if (n >= p.hidden) return;

    float sum = 0.0f;
    for (uint e = 0; e < p.top_k; e++) {
        sum += per_expert_output[(token * p.top_k + e) * p.hidden + n];
    }
    residual[token * p.hidden + n] += sum;
}
