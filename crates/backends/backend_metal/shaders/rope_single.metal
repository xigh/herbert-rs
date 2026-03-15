#include <metal_stdlib>
using namespace metal;

struct RopeSingleParams {
    uint num_heads;
    uint half_dim;
};

kernel void rope_single(
    device float*        qk        [[buffer(0)]],
    device const float*  cos_vals  [[buffer(1)]],
    device const float*  sin_vals  [[buffer(2)]],
    constant RopeSingleParams&     p         [[buffer(3)]],
    uint tid                       [[thread_position_in_grid]]
) {
    uint h = tid / p.half_dim;
    uint j = tid % p.half_dim;
    if (h >= p.num_heads) return;
    uint head_dim = p.half_dim * 2u;
    uint base = h * head_dim;
    float x0 = qk[base + j];
    float x1 = qk[base + p.half_dim + j];
    float c  = cos_vals[j];
    float s  = sin_vals[j];
    qk[base + j]              = x0 * c - x1 * s;
    qk[base + p.half_dim + j] = x0 * s + x1 * c;
}
