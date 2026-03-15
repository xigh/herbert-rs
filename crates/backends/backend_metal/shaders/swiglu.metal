#include <metal_stdlib>
using namespace metal;

struct SwigluParams {
    uint n;
};

kernel void swiglu(
    device float* gate        [[buffer(0)]],
    device const float* up    [[buffer(1)]],
    constant SwigluParams& p        [[buffer(2)]],
    uint tid                  [[thread_position_in_grid]]
) {
    if (tid >= p.n) return;
    float x = gate[tid];
    float silu_x = x / (1.0f + exp(-x));
    gate[tid] = silu_x * up[tid];
}
