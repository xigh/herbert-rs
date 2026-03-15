// FlashDecoding v2: double-buffered K/V loads for decode attention.
//
// Two variants:
//   1. Half KV cache: half-precision TG memory + double-buffered cooperative loads
//   2. INT8 KV cache: double-buffered cooperative load + dequant
//
// Both overlap next-position load with current-position compute.
// The key optimization: compute on buffer A while loading into buffer B,
// eliminating the serial load→barrier→compute→barrier cycle.
//
// Same dispatch grid as v1: (num_kv_heads, num_tiles, 1), threads = heads_per_kv * 32.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint FLASH_TILE_SIZE = 256;

struct FlashDecodeTileParams {
    uint  num_heads;
    uint  num_kv_heads;
    uint  head_dim;
    uint  kv_dim;
    uint  cached_len;
    float scale;
};

// ============================================================================
// Half KV variant: half-precision TG memory + double-buffered loads
//
// TG memory: 4 × head_dim × sizeof(half) = 1024 B (head_dim=128)
//   [k_buf0: 256B] [v_buf0: 256B] [k_buf1: 256B] [v_buf1: 256B]
// Same total as v1's 2 × head_dim × sizeof(float) = 1024 B.
//
// Benefits:
//   - Half TG memory avoids half→float conversion during store
//   - Double-buffer overlaps next-position load with current compute
//   - Reads half from TG (2× bandwidth vs float reads in v1)
// ============================================================================

[[kernel]]
void attention_decode_flash_tile_v2(
    device const float*            q         [[buffer(0)]],
    device const half*             k_cache   [[buffer(1)]],
    device const half*             v_cache   [[buffer(2)]],
    device float*                  partials  [[buffer(3)]],
    constant FlashDecodeTileParams& p        [[buffer(4)]],
    threadgroup char*              tg_raw    [[threadgroup(0)]],
    uint3                          tid3      [[thread_position_in_threadgroup]],
    uint3                          gid3      [[threadgroup_position_in_grid]]
) {
    uint tid         = tid3.x;
    uint kv_head_idx = gid3.x;
    uint tile_idx    = gid3.y;

    if (kv_head_idx >= p.num_kv_heads) return;

    uint heads_per_kv = p.num_heads / p.num_kv_heads;
    uint warp_id      = tid / 32;
    uint lane         = tid % 32;
    uint tg_threads   = heads_per_kv * 32;
    uint q_head       = kv_head_idx * heads_per_kv + warp_id;
    uint head_dim     = p.head_dim;

    uint tile_start = tile_idx * FLASH_TILE_SIZE;
    uint tile_end   = min(tile_start + FLASH_TILE_SIZE, p.cached_len);

    if (tile_start >= p.cached_len) return;

    // Load Q into registers.
    float q_reg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint qi = 0;
    for (uint d = lane; d < head_dim; d += 32) {
        q_reg[qi++] = q[q_head * head_dim + d];
    }

    // Double-buffer layout: 4 half arrays of head_dim each
    threadgroup half* k_buf0 = (threadgroup half*)(tg_raw);
    threadgroup half* v_buf0 = (threadgroup half*)(tg_raw + head_dim * 2);
    threadgroup half* k_buf1 = (threadgroup half*)(tg_raw + head_dim * 4);
    threadgroup half* v_buf1 = (threadgroup half*)(tg_raw + head_dim * 6);

    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    // Pre-load first position into buffer 0
    {
        uint kv_offset = (tile_start * p.num_kv_heads + kv_head_idx) * head_dim;
        for (uint d = tid; d < head_dim; d += tg_threads) {
            k_buf0[d] = k_cache[kv_offset + d];
            v_buf0[d] = v_cache[kv_offset + d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = tile_start; t < tile_end; t++) {
        uint cur = (t - tile_start) & 1;
        threadgroup half* cur_k = cur ? k_buf1 : k_buf0;
        threadgroup half* cur_v = cur ? v_buf1 : v_buf0;
        threadgroup half* nxt_k = cur ? k_buf0 : k_buf1;
        threadgroup half* nxt_v = cur ? v_buf0 : v_buf1;

        // Compute: Q · K^T with half→float on read
        float dot_val = 0.0f;
        uint ai = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            dot_val += q_reg[ai++] * float(cur_k[d]);
        }
        float score = simd_sum(dot_val) * p.scale;

        // Online softmax update
        float new_max = max(max_val, score);
        float correction = exp(max_val - new_max);
        float weight = exp(score - new_max);
        sum_exp = sum_exp * correction + weight;

        ai = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            acc[ai] = acc[ai] * correction + weight * float(cur_v[d]);
            ai++;
        }
        max_val = new_max;

        // Pre-load next position into alternate buffer (overlaps with next iter's compute)
        if (t + 1 < tile_end) {
            uint next_offset = ((t + 1) * p.num_kv_heads + kv_head_idx) * head_dim;
            for (uint d = tid; d < head_dim; d += tg_threads) {
                nxt_k[d] = k_cache[next_offset + d];
                nxt_v[d] = v_cache[next_offset + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partials: [num_tiles][num_heads][2 + head_dim]
    uint stride = 2 + head_dim;
    uint partial_off = (tile_idx * p.num_heads + q_head) * stride;

    if (lane == 0) {
        partials[partial_off + 0] = max_val;
        partials[partial_off + 1] = sum_exp;
    }
    uint wi = 0;
    for (uint d = lane; d < head_dim; d += 32) {
        partials[partial_off + 2 + d] = acc[wi++];
    }
}


// ============================================================================
// INT8 KV variant: double-buffered cooperative load + dequant
//
// TG memory: 4 × head_dim × sizeof(float) = 2048 B (head_dim=128)
//   [k_float0: 512B] [v_float0: 512B] [k_float1: 512B] [v_float1: 512B]
// 2× current v1 size (1024 B) — still very small.
// ============================================================================

[[kernel]]
void attention_decode_flash_tile_i8_v2(
    device const float*            q         [[buffer(0)]],
    device const char*             k_cache   [[buffer(1)]],
    device const char*             v_cache   [[buffer(2)]],
    device float*                  partials  [[buffer(3)]],
    device const float*            k_scales  [[buffer(4)]],
    device const float*            v_scales  [[buffer(5)]],
    constant FlashDecodeTileParams& p        [[buffer(6)]],
    threadgroup char*              tg_raw    [[threadgroup(0)]],
    uint3                          tid3      [[thread_position_in_threadgroup]],
    uint3                          gid3      [[threadgroup_position_in_grid]]
) {
    uint tid         = tid3.x;
    uint kv_head_idx = gid3.x;
    uint tile_idx    = gid3.y;

    if (kv_head_idx >= p.num_kv_heads) return;

    uint heads_per_kv = p.num_heads / p.num_kv_heads;
    uint warp_id      = tid / 32;
    uint lane         = tid % 32;
    uint tg_threads   = heads_per_kv * 32;
    uint q_head       = kv_head_idx * heads_per_kv + warp_id;
    uint head_dim     = p.head_dim;

    uint tile_start = tile_idx * FLASH_TILE_SIZE;
    uint tile_end   = min(tile_start + FLASH_TILE_SIZE, p.cached_len);

    if (tile_start >= p.cached_len) return;

    // Load Q into registers.
    float q_reg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint qi = 0;
    for (uint d = lane; d < head_dim; d += 32) {
        q_reg[qi++] = q[q_head * head_dim + d];
    }

    // Double-buffer layout: 4 float arrays (K0, V0, K1, V1)
    threadgroup float* k_float0 = (threadgroup float*)(tg_raw);
    threadgroup float* v_float0 = (threadgroup float*)(tg_raw + head_dim * 4);
    threadgroup float* k_float1 = (threadgroup float*)(tg_raw + head_dim * 8);
    threadgroup float* v_float1 = (threadgroup float*)(tg_raw + head_dim * 12);

    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    // Pre-load first position (cooperative load + dequant)
    {
        uint kv_offset = (tile_start * p.num_kv_heads + kv_head_idx) * head_dim;
        uint scale_idx = tile_start * p.num_kv_heads + kv_head_idx;
        float ks = k_scales[scale_idx];
        float vs = v_scales[scale_idx];
        for (uint d = tid; d < head_dim; d += tg_threads) {
            k_float0[d] = float(k_cache[kv_offset + d]) * ks;
            v_float0[d] = float(v_cache[kv_offset + d]) * vs;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = tile_start; t < tile_end; t++) {
        uint cur = (t - tile_start) & 1;
        threadgroup float* cur_k = cur ? k_float1 : k_float0;
        threadgroup float* cur_v = cur ? v_float1 : v_float0;
        threadgroup float* nxt_k = cur ? k_float0 : k_float1;
        threadgroup float* nxt_v = cur ? v_float0 : v_float1;

        // Dot product Q · K^T
        float dot_val = 0.0f;
        uint ai = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            dot_val += q_reg[ai++] * cur_k[d];
        }
        float score = simd_sum(dot_val) * p.scale;

        // Online softmax update
        float new_max = max(max_val, score);
        float correction = exp(max_val - new_max);
        float weight = exp(score - new_max);
        sum_exp = sum_exp * correction + weight;

        ai = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            acc[ai] = acc[ai] * correction + weight * cur_v[d];
            ai++;
        }
        max_val = new_max;

        // Pre-load + dequant next position into alternate buffer
        if (t + 1 < tile_end) {
            uint next_offset = ((t + 1) * p.num_kv_heads + kv_head_idx) * head_dim;
            uint next_scale = (t + 1) * p.num_kv_heads + kv_head_idx;
            float nks = k_scales[next_scale];
            float nvs = v_scales[next_scale];
            for (uint d = tid; d < head_dim; d += tg_threads) {
                nxt_k[d] = float(k_cache[next_offset + d]) * nks;
                nxt_v[d] = float(v_cache[next_offset + d]) * nvs;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partials
    uint stride = 2 + head_dim;
    uint partial_off = (tile_idx * p.num_heads + q_head) * stride;

    if (lane == 0) {
        partials[partial_off + 0] = max_val;
        partials[partial_off + 1] = sum_exp;
    }
    uint wi = 0;
    for (uint d = lane; d < head_dim; d += 32) {
        partials[partial_off + 2 + d] = acc[wi++];
    }
}
