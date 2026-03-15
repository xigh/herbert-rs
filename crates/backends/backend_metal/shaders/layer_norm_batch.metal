#include <metal_stdlib>
using namespace metal;

#define SUBGROUP_SIZE 32u

struct LayerNormBatchParams {
    uint  dim;
    float eps;
    uint  batch_size;
};

// LayerNorm with mean subtraction and bias (differs from RMS norm):
//   y[i] = (x[i] - mean) * rsqrt(var + eps) * weight[i] + bias[i]
//
// Dispatch: (batch_size, 1, 1) threadgroups of 32 threads.
kernel void layer_norm_batch(
    device const float* input_data   [[buffer(0)]],
    device const float* weight       [[buffer(1)]],
    device const float* bias         [[buffer(2)]],
    device float*       output_data  [[buffer(3)]],
    constant LayerNormBatchParams& p [[buffer(4)]],
    uint gid                         [[threadgroup_position_in_grid]],
    uint lane                        [[thread_position_in_threadgroup]]
) {
    uint offset = gid * p.dim;

    // Pass 1: compute mean
    float sum_val = 0.0f;
    for (uint i = lane; i < p.dim; i += SUBGROUP_SIZE) {
        sum_val += input_data[offset + i];
    }
    float mean = simd_sum(sum_val) / float(p.dim);

    // Pass 2: compute variance
    float sum_sq = 0.0f;
    for (uint i = lane; i < p.dim; i += SUBGROUP_SIZE) {
        float diff = input_data[offset + i] - mean;
        sum_sq += diff * diff;
    }
    float inv_std = rsqrt(simd_sum(sum_sq) / float(p.dim) + p.eps);

    // Pass 3: normalize, scale, bias
    for (uint i = lane; i < p.dim; i += SUBGROUP_SIZE) {
        output_data[offset + i] = (input_data[offset + i] - mean) * inv_std * weight[i] + bias[i];
    }
}
