#include <metal_stdlib>
using namespace metal;

struct KvCacheAppendParams {
    uint kv_dim;
    uint seq_len;
};

kernel void kv_cache_append(
    device half* cache_data       [[buffer(0)]],
    device const float* new_kv    [[buffer(1)]],
    constant KvCacheAppendParams& p            [[buffer(2)]],
    uint tid                      [[thread_position_in_grid]]
) {
    if (tid >= p.kv_dim) return;
    cache_data[p.seq_len * p.kv_dim + tid] = half(new_kv[tid]);
}
