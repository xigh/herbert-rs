#include <metal_stdlib>
using namespace metal;

// Fixed SIMD width for Apple Silicon
#define SUBGROUP_SIZE 32u

struct SoftmaxParams {
    uint n;
};

kernel void softmax(
    device float*    data  [[buffer(0)]],
    constant SoftmaxParams& p     [[buffer(1)]],
    uint lane              [[thread_position_in_threadgroup]]
) {
    // Pass 1: find max value
    float max_val = -INFINITY;
    for (uint i = lane; i < p.n; i += SUBGROUP_SIZE) {
        max_val = max(max_val, data[i]);
    }
    max_val = simd_max(max_val);

    // Pass 2: compute exp(x - max) and accumulate sum
    float sum_exp = 0.0f;
    for (uint i = lane; i < p.n; i += SUBGROUP_SIZE) {
        float v = exp(data[i] - max_val);
        data[i] = v;
        sum_exp += v;
    }
    sum_exp = simd_sum(sum_exp);

    // Pass 3: normalize
    if (sum_exp > 0.0f) {
        float inv = 1.0f / sum_exp;
        for (uint i = lane; i < p.n; i += SUBGROUP_SIZE) {
            data[i] *= inv;
        }
    }
}
