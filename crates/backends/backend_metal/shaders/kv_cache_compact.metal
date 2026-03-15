#include <metal_stdlib>
using namespace metal;

// KV cache compaction: gather kept positions into contiguous layout.
//
// After H2O eviction, the kept positions are non-contiguous. This kernel
// copies data from old positions to new contiguous positions.
//
// Safety: index_map must be sorted by original position (ascending) so that
// new_pos <= old_pos for all entries, making in-place compaction safe.
//
// Three variants: half (KV), int8 (KV), and scales (f32 per head).

struct KvCompactParams {
    uint stride;     // elements per position (kv_dim for K/V, num_kv_heads for scales)
    uint num_kept;   // number of positions to keep
};

// Dispatch: grid = (num_kept, 1, 1), threads = (min(stride, 256), 1, 1)

// Half-precision KV compaction
[[kernel]]
void kv_cache_compact_half(
    device half*               cache     [[buffer(0)]],
    device const uint*         index_map [[buffer(1)]],
    constant KvCompactParams&  p         [[buffer(2)]],
    uint                       tid       [[thread_position_in_threadgroup]],
    uint                       gid       [[threadgroup_position_in_grid]]
) {
    if (gid >= p.num_kept) return;
    uint old_pos = index_map[gid];
    uint new_pos = gid;
    if (old_pos == new_pos) return;  // no-op

    for (uint d = tid; d < p.stride; d += 256) {
        cache[new_pos * p.stride + d] = cache[old_pos * p.stride + d];
    }
}

// INT8 KV compaction
[[kernel]]
void kv_cache_compact_i8(
    device char*               cache     [[buffer(0)]],
    device const uint*         index_map [[buffer(1)]],
    constant KvCompactParams&  p         [[buffer(2)]],
    uint                       tid       [[thread_position_in_threadgroup]],
    uint                       gid       [[threadgroup_position_in_grid]]
) {
    if (gid >= p.num_kept) return;
    uint old_pos = index_map[gid];
    uint new_pos = gid;
    if (old_pos == new_pos) return;

    for (uint d = tid; d < p.stride; d += 256) {
        cache[new_pos * p.stride + d] = cache[old_pos * p.stride + d];
    }
}

// Scales compaction (f32, stride = num_kv_heads)
[[kernel]]
void kv_cache_compact_scales(
    device float*              scales    [[buffer(0)]],
    device const uint*         index_map [[buffer(1)]],
    constant KvCompactParams&  p         [[buffer(2)]],
    uint                       tid       [[thread_position_in_threadgroup]],
    uint                       gid       [[threadgroup_position_in_grid]]
) {
    if (gid >= p.num_kept) return;
    uint old_pos = index_map[gid];
    uint new_pos = gid;
    if (old_pos == new_pos) return;

    for (uint d = tid; d < p.stride; d += 256) {
        scales[new_pos * p.stride + d] = scales[old_pos * p.stride + d];
    }
}
