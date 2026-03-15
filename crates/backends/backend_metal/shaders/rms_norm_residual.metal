#include <metal_stdlib>
using namespace metal;

#define SUBGROUP_SIZE 32u
#define TG_SIZE 256u
#define NUM_WARPS (TG_SIZE / SUBGROUP_SIZE)  // 8

struct RmsNormResidualParams {
    uint  dim;
    float eps;
};

// Fused residual add + RMS normalization.
//
// Combines two operations in a single dispatch:
//   1. a[i] += b[i]  (in-place residual add)
//   2. output[i] = (a[i] / rms(a)) * weight[i]  (RMS norm)
//
// This saves one dispatch per fusion point in the decode loop.
// There are 2 fusion points per transformer layer (post-attention and
// post-MLP), saving ~32 dispatches total for a 16-layer model.
//
// Dispatch: 1 threadgroup of 256 threads.
//
// Buffers:
//   a      : [dim]  - residual (modified in-place: a += b)
//   b      : [dim]  - value to add
//   weight : [dim]  - RMS norm weights
//   output : [dim]  - normalized output
[[kernel]]
void rms_norm_residual(
    device float*        a      [[buffer(0)]],
    device const float*  b      [[buffer(1)]],
    device const float*  weight [[buffer(2)]],
    device float*        output [[buffer(3)]],
    constant RmsNormResidualParams& p  [[buffer(4)]],
    uint tid                           [[thread_index_in_threadgroup]]
) {
    // Pass 1: add residual and compute sum of squares
    float sum_sq = 0.0f;
    for (uint i = tid; i < p.dim; i += TG_SIZE) {
        float v = a[i] + b[i];
        a[i] = v;
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
        output[i] = a[i] * inv_rms * weight[i];
    }
}
