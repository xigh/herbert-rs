#include <metal_stdlib>
using namespace metal;

struct RopeKvAppendParams {
    uint num_heads;
    uint half_dim;
    uint kv_dim;
    uint cache_pos;
};

// Fused RoPE + KV cache append for decode (single token).
//
// Applies RoPE to the input K vector in-place AND writes it to the KV cache
// at the specified position, all in a single dispatch. This saves one
// dispatch per attention layer compared to separate rope_batch + kv_cache_append.
//
// cos/sin values are indexed at position cache_pos.
//
// Dispatch: ceil(num_heads * half_dim / 256) threadgroups of 256 threads.
//
// Buffers:
//   qk       : [num_heads, head_dim]  - input K vector (modified in-place with RoPE)
//   cos_cache: [max_pos, half_dim]    - precomputed cosine values
//   sin_cache: [max_pos, half_dim]    - precomputed sine values
//   cache    : [max_tokens, kv_dim]   - KV cache to append to
[[kernel]]
void rope_kv_append(
    device float*        qk        [[buffer(0)]],
    device const float*  cos_cache [[buffer(1)]],
    device const float*  sin_cache [[buffer(2)]],
    device half*         cache     [[buffer(3)]],
    constant RopeKvAppendParams& p [[buffer(4)]],
    uint tid                       [[thread_position_in_grid]]
) {
    uint h = tid / p.half_dim;
    uint j = tid % p.half_dim;
    if (h >= p.num_heads) return;

    uint head_dim = p.half_dim * 2u;
    uint base = h * head_dim;

    // Load cos/sin at the position
    uint cos_sin_offset = p.cache_pos * p.half_dim;
    float c = cos_cache[cos_sin_offset + j];
    float s = sin_cache[cos_sin_offset + j];

    // Apply RoPE
    float x0 = qk[base + j];
    float x1 = qk[base + p.half_dim + j];
    float r0 = x0 * c - x1 * s;
    float r1 = x0 * s + x1 * c;

    // Write RoPE'd values back to input (in-place)
    qk[base + j]              = r0;
    qk[base + p.half_dim + j] = r1;

    // Write to KV cache at cache_pos (f32 → half)
    uint cache_offset = p.cache_pos * p.kv_dim;
    cache[cache_offset + base + j]              = half(r0);
    cache[cache_offset + base + p.half_dim + j] = half(r1);
}
