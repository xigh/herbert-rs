#include <metal_stdlib>
using namespace metal;

// Fixed SIMD width for Apple Silicon
#define SUBGROUP_SIZE 32u

struct ArgmaxParams {
    uint n;
};

// Stage 1: Each threadgroup reduces a chunk of the input to a single (index, value) pair.
// Dispatch: N_TG threadgroups × 256 threads.
// Output: intermediate buffer of N_TG × 2 uint (index + as_type<uint>(value)).
kernel void argmax_stage1(
    device const float*  data          [[buffer(0)]],
    device uint*         intermediate  [[buffer(1)]],
    constant ArgmaxParams&     p             [[buffer(2)]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint tg_id                         [[threadgroup_position_in_grid]],
    uint num_tgs                       [[threadgroups_per_grid]]
) {
    // Each threadgroup handles a chunk of the input
    uint chunk_size = (p.n + num_tgs - 1) / num_tgs;
    uint start = tg_id * chunk_size;
    uint end = min(start + chunk_size, p.n);

    float best_val = -INFINITY;
    uint  best_idx = 0u;

    // Each of 256 threads scans its portion of the chunk
    for (uint i = start + tid; i < end; i += 256) {
        float v = data[i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    // SIMD reduction within each warp (32 threads)
    for (uint offset = SUBGROUP_SIZE / 2u; offset > 0u; offset >>= 1u) {
        float other_val = simd_shuffle_down(best_val, offset);
        uint  other_idx = simd_shuffle_down(best_idx, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    // Threadgroup reduction across 8 warps via shared memory
    threadgroup float shared_vals[8];
    threadgroup uint  shared_idxs[8];
    uint warp_id = tid / SUBGROUP_SIZE;
    uint lane = tid % SUBGROUP_SIZE;

    if (lane == 0) {
        shared_vals[warp_id] = best_val;
        shared_idxs[warp_id] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Warp 0 reduces the 8 partial results
    if (warp_id == 0 && lane < 8) {
        best_val = shared_vals[lane];
        best_idx = shared_idxs[lane];

        // SIMD reduction across 8 active lanes
        for (uint offset = 4u; offset > 0u; offset >>= 1u) {
            float other_val = simd_shuffle_down(best_val, offset);
            uint  other_idx = simd_shuffle_down(best_idx, offset);
            if (other_val > best_val) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }

        if (lane == 0) {
            intermediate[tg_id * 2] = best_idx;
            intermediate[tg_id * 2 + 1] = as_type<uint>(best_val);
        }
    }
}

// Stage 2: Reduce N_TG intermediate results to find global max.
// Dispatch: 1 threadgroup × 256 threads.
// Input: intermediate buffer from stage 1 (N_TG × 2 uint).
// Output: result[0] = best_idx, result[1] = as_type<uint>(best_val).
struct ArgmaxStage2Params {
    uint num_chunks;
};

kernel void argmax_stage2(
    device const uint*   intermediate  [[buffer(0)]],
    device uint*         result        [[buffer(1)]],
    constant ArgmaxStage2Params& p     [[buffer(2)]],
    uint tid                           [[thread_index_in_threadgroup]]
) {
    float best_val = -INFINITY;
    uint  best_idx = 0u;

    for (uint i = tid; i < p.num_chunks; i += 256) {
        uint  idx = intermediate[i * 2];
        float val = as_type<float>(intermediate[i * 2 + 1]);
        if (val > best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    // SIMD reduction within warp
    for (uint offset = SUBGROUP_SIZE / 2u; offset > 0u; offset >>= 1u) {
        float other_val = simd_shuffle_down(best_val, offset);
        uint  other_idx = simd_shuffle_down(best_idx, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    // Threadgroup reduction across 8 warps
    threadgroup float shared_vals[8];
    threadgroup uint  shared_idxs[8];
    uint warp_id = tid / SUBGROUP_SIZE;
    uint lane = tid % SUBGROUP_SIZE;

    if (lane == 0) {
        shared_vals[warp_id] = best_val;
        shared_idxs[warp_id] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_id == 0 && lane < 8) {
        best_val = shared_vals[lane];
        best_idx = shared_idxs[lane];

        for (uint offset = 4u; offset > 0u; offset >>= 1u) {
            float other_val = simd_shuffle_down(best_val, offset);
            uint  other_idx = simd_shuffle_down(best_idx, offset);
            if (other_val > best_val) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }

        if (lane == 0) {
            result[0] = best_idx;
            result[1] = as_type<uint>(best_val);
        }
    }
}
