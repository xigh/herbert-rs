#include <metal_stdlib>
using namespace metal;

// Fixed SIMD width for Apple Silicon
#define SUBGROUP_SIZE 32u

struct HeadRmsNormParams {
    uint  num_heads;
    uint  head_dim;
    float eps;
};

kernel void head_rms_norm(
    device float*        data    [[buffer(0)]],
    device const float*  weight  [[buffer(1)]],
    constant HeadRmsNormParams&     p       [[buffer(2)]],
    uint gid                     [[threadgroup_position_in_grid]],
    uint lane                    [[thread_position_in_threadgroup]]
) {
    uint h = gid;
    if (h >= p.num_heads) return;
    uint offset = h * p.head_dim;

    float sum_sq = 0.0f;
    for (uint i = lane; i < p.head_dim; i += SUBGROUP_SIZE) {
        float v = data[offset + i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);

    float inv_rms = rsqrt(sum_sq / float(p.head_dim) + p.eps);

    for (uint i = lane; i < p.head_dim; i += SUBGROUP_SIZE) {
        data[offset + i] = data[offset + i] * inv_rms * weight[i];
    }
}
