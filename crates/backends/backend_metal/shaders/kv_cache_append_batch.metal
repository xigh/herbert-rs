#include <metal_stdlib>
using namespace metal;

struct KvCacheAppendBatchParams {
    uint kv_dim;
    uint start;
    uint count;
};

kernel void kv_cache_append_batch(
    device half* cache_data       [[buffer(0)]],
    device const float* new_kv    [[buffer(1)]],
    constant KvCacheAppendBatchParams& p            [[buffer(2)]],
    uint tid                      [[thread_position_in_grid]]
) {
    if (tid >= p.count * p.kv_dim) return;
    uint pos = tid / p.kv_dim;
    uint dim = tid % p.kv_dim;
    cache_data[(p.start + pos) * p.kv_dim + dim] = half(new_kv[pos * p.kv_dim + dim]);
}
