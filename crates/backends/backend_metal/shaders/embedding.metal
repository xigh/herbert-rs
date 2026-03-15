#include <metal_stdlib>
using namespace metal;

struct EmbeddingParams {
    uint hidden_size;
    uint total_elements;
};

kernel void embedding(
    device const float* embed_table  [[buffer(0)]],
    device const uint* tokens        [[buffer(1)]],
    device float* output_data        [[buffer(2)]],
    constant EmbeddingParams& p               [[buffer(3)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= p.total_elements) return;
    uint seq_idx = tid / p.hidden_size;
    uint dim_idx = tid % p.hidden_size;
    uint token_id = tokens[seq_idx];
    output_data[tid] = embed_table[token_id * p.hidden_size + dim_idx];
}
