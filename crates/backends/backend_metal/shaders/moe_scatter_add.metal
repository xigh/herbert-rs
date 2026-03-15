#include <metal_stdlib>
using namespace metal;

struct MoeScatterAddParams {
    uint dim;
    uint count;
};

kernel void moe_scatter_add(
    device float* out_data          [[buffer(0)]],
    device const float* in_data     [[buffer(1)]],
    device const uint* indices      [[buffer(2)]],
    device const float* weights     [[buffer(3)]],
    constant MoeScatterAddParams& p              [[buffer(4)]],
    uint tid                        [[thread_position_in_grid]]
) {
    uint total = p.count * p.dim;
    if (tid >= total) return;
    uint row = tid / p.dim;
    uint col = tid % p.dim;
    uint dst_row = indices[row];
    float w = weights[row];
    out_data[dst_row * p.dim + col] += w * in_data[tid];
}
