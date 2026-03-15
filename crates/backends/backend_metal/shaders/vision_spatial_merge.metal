#include <metal_stdlib>
using namespace metal;

struct VisionSpatialMergeParams {
    uint num_out_tokens; // = num_tokens / (merge_size * merge_size)
    uint in_dim;         // hidden_size per input token
    uint merge_sq;       // merge_size * merge_size (typically 4)
};

// Spatial merge: concatenate merge_size^2 consecutive input tokens into one output token.
//
// Input:  [num_tokens, in_dim]  where num_tokens = num_out_tokens * merge_sq
// Output: [num_out_tokens, merged_dim]  where merged_dim = in_dim * merge_sq
//
// For each output token i, sub-patch j (0..merge_sq):
//   output[i * merged_dim + j * in_dim + d] = input[(i * merge_sq + j) * in_dim + d]
//
// Dispatch: (ceil(num_out_tokens * merged_dim / 256), 1, 1) threadgroups of 256.
kernel void vision_spatial_merge(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant VisionSpatialMergeParams& p [[buffer(2)]],
    uint tid                   [[thread_position_in_grid]]
) {
    uint merged_dim = p.in_dim * p.merge_sq;
    uint total = p.num_out_tokens * merged_dim;
    if (tid >= total) return;

    uint out_token = tid / merged_dim;
    uint rem       = tid % merged_dim;
    uint sub_patch = rem / p.in_dim;
    uint d         = rem % p.in_dim;

    uint src_token = out_token * p.merge_sq + sub_patch;
    output[tid] = input[src_token * p.in_dim + d];
}
