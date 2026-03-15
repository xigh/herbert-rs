#include <metal_stdlib>
using namespace metal;

struct MoeGatherParams {
    uint dim;
    uint count;
};

kernel void moe_gather(
    device float* out_data         [[buffer(0)]],
    device const float* in_data    [[buffer(1)]],
    device const uint* indices     [[buffer(2)]],
    constant MoeGatherParams& p             [[buffer(3)]],
    uint tid                       [[thread_position_in_grid]]
) {
    uint total = p.count * p.dim;
    if (tid >= total) return;
    uint row = tid / p.dim;
    uint col = tid % p.dim;
    uint src_row = indices[row];
    out_data[tid] = in_data[src_row * p.dim + col];
}
