#include <metal_stdlib>
using namespace metal;

struct ScaledAddParams {
    uint len;
    float scale;
};

kernel void scaled_add(
    device float* a        [[buffer(0)]],
    device const float* b  [[buffer(1)]],
    constant ScaledAddParams& p     [[buffer(2)]],
    uint tid               [[thread_position_in_grid]]
) {
    if (tid >= p.len) return;
    a[tid] += p.scale * b[tid];
}
