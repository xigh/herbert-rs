#include <metal_stdlib>
using namespace metal;

// Fixed SIMD width for Apple Silicon
#define SUBGROUP_SIZE 32u

struct RmsNormBatchParams {
    uint  dim;
    float eps;
    uint  batch_size;
};

kernel void rms_norm_batch(
    device const float*  input_data   [[buffer(0)]],
    device const float*  weight       [[buffer(1)]],
    device float*        output_data  [[buffer(2)]],
    constant RmsNormBatchParams&     p            [[buffer(3)]],
    uint gid                          [[threadgroup_position_in_grid]],
    uint lane                         [[thread_position_in_threadgroup]]
) {
    uint batch_idx = gid;
    uint offset    = batch_idx * p.dim;

    float sum_sq = 0.0f;
    for (uint i = lane; i < p.dim; i += SUBGROUP_SIZE) {
        float v = input_data[offset + i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);

    float inv_rms = rsqrt(sum_sq / float(p.dim) + p.eps);

    for (uint i = lane; i < p.dim; i += SUBGROUP_SIZE) {
        output_data[offset + i] = input_data[offset + i] * inv_rms * weight[i];
    }
}
