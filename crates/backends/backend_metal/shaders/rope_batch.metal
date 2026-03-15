#include <metal_stdlib>
using namespace metal;

struct RopeBatchParams {
    uint seq_len;
    uint num_heads;
    uint half_dim;
    uint start_pos;
};

kernel void rope_batch(
    device float*        qk         [[buffer(0)]],
    device const float*  cos_cache  [[buffer(1)]],
    device const float*  sin_cache  [[buffer(2)]],
    constant RopeBatchParams&     p          [[buffer(3)]],
    uint tid                        [[thread_position_in_grid]]
) {
    uint head_dim    = p.half_dim * 2u;
    uint qk_per_pos  = p.num_heads * head_dim;
    uint work_per_pos = p.num_heads * p.half_dim;
    uint pos = tid / work_per_pos;
    uint rem = tid % work_per_pos;
    uint h   = rem / p.half_dim;
    uint j   = rem % p.half_dim;
    if (pos >= p.seq_len) return;
    uint abs_pos = p.start_pos + pos;
    uint base    = pos * qk_per_pos + h * head_dim;
    float x0 = qk[base + j];
    float x1 = qk[base + j + p.half_dim];
    float c  = cos_cache[abs_pos * p.half_dim + j];
    float s  = sin_cache[abs_pos * p.half_dim + j];
    qk[base + j]              = x0 * c - x1 * s;
    qk[base + j + p.half_dim] = x0 * s + x1 * c;
}
