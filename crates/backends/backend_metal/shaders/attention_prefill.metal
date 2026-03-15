#include <metal_stdlib>
using namespace metal;

struct AttentionPrefillParams {
    uint  seq_len;
    uint  num_heads;
    uint  num_kv_heads;
    uint  head_dim;
    uint  kv_dim;
    uint  q_dim;
    uint  cached_len;
    uint  start_pos;
    float scale;
};

// Flash Attention Prefill: online softmax, single-pass, no scratch buffer.
//
// Uses the online softmax algorithm to compute attention in a single pass
// over the KV cache for each query position, accumulating weighted V values
// in registers. This eliminates the O(num_heads * seq_len * cached_len)
// scores buffer, enabling 10K+ token prefill without OOM.
//
// Dispatch: (num_heads, seq_len, 1) threadgroups of 32 threads.
// h = gid.x, pos = gid.y
//
// query_pos = start_pos + pos  (absolute position in the sequence)
// Causal mask: token t is masked out if t > query_pos.
//
// Buffers:
//   q       : [seq_len, num_heads, head_dim]
//   k_cache : [cached_len, num_kv_heads, head_dim]
//   v_cache : [cached_len, num_kv_heads, head_dim]
//   output  : [seq_len, num_heads, head_dim]
[[kernel]]
void attention_prefill(
    device const float*    q       [[buffer(0)]],
    device const half*     k_cache [[buffer(1)]],
    device const half*     v_cache [[buffer(2)]],
    device float*          output  [[buffer(3)]],
    constant AttentionPrefillParams&       p       [[buffer(4)]],
    uint3                  tid3    [[thread_position_in_threadgroup]],
    uint3                  gid3    [[threadgroup_position_in_grid]]
) {
    uint lane         = tid3.x;
    uint h            = gid3.x;
    uint pos          = gid3.y;
    uint seq_len      = p.seq_len;
    uint num_heads    = p.num_heads;
    uint num_kv_heads = p.num_kv_heads;
    uint head_dim     = p.head_dim;
    uint cached_len   = p.cached_len;
    uint start_pos    = p.start_pos;
    float scale       = p.scale;

    if (h >= num_heads || pos >= seq_len) return;

    uint query_pos = start_pos + pos;

    // GQA: map query head to KV head.
    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_head      = h / heads_per_kv;

    // Pointer into the query tensor.
    uint q_offset = (pos * num_heads + h) * head_dim;

    // Online softmax accumulators (per lane).
    // Each lane handles dimensions d = lane, lane+32, lane+64, ...
    // Max head_dim supported: 256 (8 values per lane with 32 threads).
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    // Causal attention: only attend to positions t <= query_pos.
    uint valid_len = min(query_pos + 1, cached_len);

    for (uint t = 0; t < valid_len; t++) {
        // --- Cooperative dot product Q·K^T across all 32 lanes ---
        float dot_val = 0.0f;
        uint k_offset = (t * num_kv_heads + kv_head) * head_dim;

        for (uint d = lane; d < head_dim; d += 32) {
            dot_val += q[q_offset + d] * float(k_cache[k_offset + d]);
        }
        float score = simd_sum(dot_val) * scale;

        // --- Online softmax update (uniform across all lanes) ---
        float new_max = max(max_val, score);
        float correction = exp(max_val - new_max);
        float weight = exp(score - new_max);
        sum_exp = sum_exp * correction + weight;

        // --- Accumulate weighted V values (each lane handles its dims) ---
        uint v_offset = (t * num_kv_heads + kv_head) * head_dim;
        uint idx = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            acc[idx] = acc[idx] * correction + weight * float(v_cache[v_offset + d]);
            idx++;
        }

        max_val = new_max;
    }

    // --- Normalize and write output ---
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    uint out_offset = (pos * num_heads + h) * head_dim;
    uint idx = 0;
    for (uint d = lane; d < head_dim; d += 32) {
        output[out_offset + d] = acc[idx] * inv_sum;
        idx++;
    }
}
