#include <metal_stdlib>
using namespace metal;

struct CopyBufferParams {
    uint n;
};

kernel void copy_buffer(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant CopyBufferParams& p      [[buffer(2)]],
    uint tid                [[thread_position_in_grid]]
) {
    if (tid >= p.n) return;
    dst[tid] = src[tid];
}
