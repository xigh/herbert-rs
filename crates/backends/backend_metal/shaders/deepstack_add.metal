#include <metal_stdlib>
using namespace metal;

struct DeepStackParams {
    uint hidden_size;
    uint num_image_tokens;
};

// Add DeepStack vision features to image token positions in the hidden state.
//
// For each image token i at position token_indices[i]:
//   embed[token_indices[i] * hidden_size + d] += features[i * hidden_size + d]
//
// Dispatch: ceil(num_image_tokens * hidden_size / 256) threadgroups of 256.
kernel void deepstack_add(
    device float*       embed         [[buffer(0)]],
    device const float* features      [[buffer(1)]],
    device const uint*  token_indices [[buffer(2)]],
    constant DeepStackParams& p       [[buffer(3)]],
    uint gid                          [[thread_position_in_grid]]
) {
    uint total = p.num_image_tokens * p.hidden_size;
    if (gid >= total) return;

    uint tok = gid / p.hidden_size;
    uint dim = gid % p.hidden_size;
    uint embed_idx = token_indices[tok] * p.hidden_size + dim;
    embed[embed_idx] += features[tok * p.hidden_size + dim];
}
