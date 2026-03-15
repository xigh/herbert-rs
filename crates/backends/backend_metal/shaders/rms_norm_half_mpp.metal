// RMS normalization with half-precision accumulation (Metal 4).
//
// Same algorithm as rms_norm.metal / rms_norm_batch.metal but accumulates
// sum-of-squares in half, exploiting Metal 4's 2× FP16 throughput.
// The rsqrt and final multiply stay in float for numerical stability.
//
// Requires: MSL 4.0

#if __METAL_VERSION__ >= 400

#include <metal_stdlib>
using namespace metal;

#define SUBGROUP_SIZE 32u
#define TG_SIZE 256u
#define NUM_WARPS (TG_SIZE / SUBGROUP_SIZE)

struct RmsNormParams {
    uint  dim;
    float eps;
};

struct RmsNormBatchParams {
    uint  dim;
    float eps;
    uint  batch_size;
};

// Single-vector RMS norm with half accumulation (decode path, 256 threads)
[[kernel]]
void rms_norm_half(
    device const float*    input_data   [[buffer(0)]],
    device const float*    weight       [[buffer(1)]],
    device float*          output_data  [[buffer(2)]],
    constant RmsNormParams& p           [[buffer(3)]],
    uint tid                            [[thread_index_in_threadgroup]]
) {
    // Pass 1: accumulate sum of squares in half (2× FP16 throughput)
    half sum_sq_h = 0.0h;
    for (uint i = tid; i < p.dim; i += TG_SIZE) {
        half v = half(input_data[i]);
        sum_sq_h += v * v;
    }

    // Convert to float for cross-warp reduction
    float sum_sq = float(sum_sq_h);
    sum_sq = simd_sum(sum_sq);

    threadgroup float shared_sums[NUM_WARPS];
    uint warp_id = tid / SUBGROUP_SIZE;
    uint lane = tid % SUBGROUP_SIZE;

    if (lane == 0) {
        shared_sums[warp_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_id == 0) {
        sum_sq = (lane < NUM_WARPS) ? shared_sums[lane] : 0.0f;
        sum_sq = simd_sum(sum_sq);
    }

    threadgroup float shared_inv_rms;
    if (tid == 0) {
        shared_inv_rms = rsqrt(sum_sq / float(p.dim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = shared_inv_rms;

    // Pass 2: normalize (stays in float for output precision)
    for (uint i = tid; i < p.dim; i += TG_SIZE) {
        output_data[i] = input_data[i] * inv_rms * weight[i];
    }
}

// Batch RMS norm with half accumulation (prefill path, 32 threads = 1 simdgroup)
// Matches rms_norm_batch.metal dispatch: grid=(batch_size), threads=(32)
[[kernel]]
void rms_norm_batch_half(
    device const float*        input_data   [[buffer(0)]],
    device const float*        weight       [[buffer(1)]],
    device float*              output_data  [[buffer(2)]],
    constant RmsNormBatchParams& p          [[buffer(3)]],
    uint gid                               [[threadgroup_position_in_grid]],
    uint lane                              [[thread_position_in_threadgroup]]
) {
    uint offset = gid * p.dim;

    // Accumulate sum of squares in half (1 simdgroup)
    half sum_sq_h = 0.0h;
    for (uint i = lane; i < p.dim; i += SUBGROUP_SIZE) {
        half v = half(input_data[offset + i]);
        sum_sq_h += v * v;
    }

    float sum_sq = float(sum_sq_h);
    sum_sq = simd_sum(sum_sq);
    float inv_rms = rsqrt(sum_sq / float(p.dim) + p.eps);

    for (uint i = lane; i < p.dim; i += SUBGROUP_SIZE) {
        output_data[offset + i] = input_data[offset + i] * inv_rms * weight[i];
    }
}

#endif // __METAL_VERSION__ >= 400
