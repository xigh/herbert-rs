#include <metal_stdlib>
using namespace metal;

struct BiasAddBatchParams {
    uint dim;
    uint total;
};

kernel void bias_add_batch(
    device float* data         [[buffer(0)]],
    device const float* bias   [[buffer(1)]],
    constant BiasAddBatchParams& p         [[buffer(2)]],
    uint tid                   [[thread_position_in_grid]]
) {
    if (tid >= p.total) return;
    data[tid] += bias[tid % p.dim];
}
