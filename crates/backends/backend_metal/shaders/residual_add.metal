#include <metal_stdlib>
using namespace metal;

struct ResidualAddParams {
    uint len;
};

kernel void residual_add(
    device float* a        [[buffer(0)]],
    device const float* b  [[buffer(1)]],
    constant ResidualAddParams& p     [[buffer(2)]],
    uint tid               [[thread_position_in_grid]]
) {
    if (tid >= p.len) return;
    a[tid] += b[tid];
}
