#include <metal_stdlib>
using namespace metal;

#define SUBGROUP_SIZE 32u
#define TG_SIZE 256u
#define NUM_WARPS (TG_SIZE / SUBGROUP_SIZE)  // 8

struct RmsNormParams {
    uint  dim;
    float eps;
};

kernel void rms_norm(
    device const float*  input_data   [[buffer(0)]],
    device const float*  weight       [[buffer(1)]],
    device float*        output_data  [[buffer(2)]],
    constant RmsNormParams&     p            [[buffer(3)]],
    uint tid                          [[thread_index_in_threadgroup]]
) {
    // Pass 1: each thread accumulates partial sum of squares
    float sum_sq = 0.0f;
    for (uint i = tid; i < p.dim; i += TG_SIZE) {
        float v = input_data[i];
        sum_sq += v * v;
    }

    // SIMD reduction within warp
    sum_sq = simd_sum(sum_sq);

    // Threadgroup reduction across 8 warps
    threadgroup float shared_sums[NUM_WARPS];
    uint warp_id = tid / SUBGROUP_SIZE;
    uint lane = tid % SUBGROUP_SIZE;

    if (lane == 0) {
        shared_sums[warp_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Warp 0 reduces the 8 partial sums
    if (warp_id == 0) {
        sum_sq = (lane < NUM_WARPS) ? shared_sums[lane] : 0.0f;
        sum_sq = simd_sum(sum_sq);
    }

    // Broadcast inv_rms to all threads
    threadgroup float shared_inv_rms;
    if (tid == 0) {
        shared_inv_rms = rsqrt(sum_sq / float(p.dim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = shared_inv_rms;

    // Pass 2: normalize with weights
    for (uint i = tid; i < p.dim; i += TG_SIZE) {
        output_data[i] = input_data[i] * inv_rms * weight[i];
    }
}
