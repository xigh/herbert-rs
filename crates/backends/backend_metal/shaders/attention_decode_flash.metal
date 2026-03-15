#include <metal_stdlib>
using namespace metal;

// FlashDecoding: tile-parallel decode attention over the sequence dimension.
//
// Splits the KV cache into tiles of FLASH_TILE_SIZE positions. Each tile is
// processed by an independent threadgroup, producing partial online softmax
// results. A reduce kernel merges the partials into the final output.
//
// Combined with half-precision KV cache for 2x bandwidth reduction.
//
// Two kernels:
//   1. attention_decode_flash_tile  — per-tile online softmax (GQA shared K/V)
//   2. attention_decode_flash_reduce — merge tile partials
//
// Expected occupancy at 20K cached_len, TILE_SIZE=256:
//   80 tiles × 8 KV heads = 640 threadgroups (vs 8 in old shader)

constant constexpr uint FLASH_TILE_SIZE = 256;

struct FlashDecodeTileParams {
    uint  num_heads;
    uint  num_kv_heads;
    uint  head_dim;
    uint  kv_dim;
    uint  cached_len;
    float scale;
};

// Phase 1: Tile kernel — one threadgroup per (kv_head, tile).
//
// GQA pattern: shared K/V in threadgroup memory, one warp per query head.
//
// Dispatch:
//   grid    = (num_kv_heads, num_tiles, 1)
//   threads = (heads_per_kv * 32, 1, 1)
// Threadgroup memory:
//   2 * head_dim * sizeof(float) for shared K and V vectors.
//
// Output partials: [num_tiles, num_heads, 2 + head_dim] float
//   [0] = max_val, [1] = sum_exp, [2..] = weighted V accumulator
[[kernel]]
void attention_decode_flash_tile(
    device const float*            q         [[buffer(0)]],
    device const half*             k_cache   [[buffer(1)]],
    device const half*             v_cache   [[buffer(2)]],
    device float*                  partials  [[buffer(3)]],
    constant FlashDecodeTileParams& p        [[buffer(4)]],
    threadgroup float*             shared_kv [[threadgroup(0)]],
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

    threadgroup float* shared_k = shared_kv;
    threadgroup float* shared_v = shared_kv + head_dim;

    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    // Iterate over positions in this tile.
    for (uint t = tile_start; t < tile_end; t++) {
        uint kv_offset = (t * p.num_kv_heads + kv_head_idx) * head_dim;

        // Cooperative load K + V from half cache into shared float memory.
        for (uint d = tid; d < head_dim; d += tg_threads) {
            shared_k[d] = float(k_cache[kv_offset + d]);
            shared_v[d] = float(v_cache[kv_offset + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Dot product Q · K^T.
        float dot_val = 0.0f;
        uint ai = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            dot_val += q_reg[ai++] * shared_k[d];
        }
        float score = simd_sum(dot_val) * p.scale;

        // Online softmax update.
        float new_max = max(max_val, score);
        float correction = exp(max_val - new_max);
        float weight = exp(score - new_max);
        sum_exp = sum_exp * correction + weight;

        ai = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            acc[ai] = acc[ai] * correction + weight * shared_v[d];
            ai++;
        }
        max_val = new_max;

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


struct FlashDecodeReduceParams {
    uint num_heads;
    uint head_dim;
    uint num_tiles;
};

// Phase 2: Reduce kernel — merge tile partials into final output.
//
// Each threadgroup handles one query head, reading partials from all tiles
// and merging them using online softmax correction.
//
// Dispatch:
//   grid    = (num_heads, 1, 1)
//   threads = (32, 1, 1)
[[kernel]]
void attention_decode_flash_reduce(
    device const float*                partials  [[buffer(0)]],
    device float*                      output    [[buffer(1)]],
    constant FlashDecodeReduceParams&  p         [[buffer(2)]],
    uint                               lane      [[thread_position_in_threadgroup]],
    uint                               gid       [[threadgroup_position_in_grid]]
) {
    uint head = gid;
    if (head >= p.num_heads) return;

    uint stride = 2 + p.head_dim;

    float global_max = -INFINITY;
    float global_sum = 0.0f;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (uint tile = 0; tile < p.num_tiles; tile++) {
        uint off = (tile * p.num_heads + head) * stride;

        // Read tile partials (lane 0 wrote max/sum, all lanes wrote acc).
        float tile_max = partials[off + 0];
        float tile_sum = partials[off + 1];

        if (tile_sum <= 0.0f) continue;  // empty tile (past cached_len)

        // Online softmax merge.
        float new_max = max(global_max, tile_max);
        float old_correction  = exp(global_max - new_max);
        float tile_correction = exp(tile_max - new_max);

        global_sum = global_sum * old_correction + tile_sum * tile_correction;

        uint ai = 0;
        for (uint d = lane; d < p.head_dim; d += 32) {
            acc[ai] = acc[ai] * old_correction
                    + partials[off + 2 + d] * tile_correction;
            ai++;
        }

        global_max = new_max;
    }

    // Normalize and write output.
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    uint ai = 0;
    for (uint d = lane; d < p.head_dim; d += 32) {
        output[head * p.head_dim + d] = acc[ai++] * inv_sum;
    }
}


// Phase 1 (INT8 variant): Tile kernel for INT8 KV cache with per-position per-head scales.
//
// Same structure as the half variant but reads signed int8 K/V and dequantizes
// with per-position per-head float scales. Uses shared memory for GQA K/V sharing.
//
// Buffer layout: q(0), k_cache_i8(1), v_cache_i8(2), partials(3),
//                k_scales(4), v_scales(5), params(6)
//
// Scale layout: [cached_len * num_kv_heads] float, indexed as t * num_kv_heads + kv_head_idx.
//
// Dispatch:
//   grid    = (num_kv_heads, num_tiles, 1)
//   threads = (heads_per_kv * 32, 1, 1)
//   tgmem   = 2 * head_dim * sizeof(float)
[[kernel]]
void attention_decode_flash_tile_i8(
    device const float*            q         [[buffer(0)]],
    device const char*             k_cache   [[buffer(1)]],
    device const char*             v_cache   [[buffer(2)]],
    device float*                  partials  [[buffer(3)]],
    device const float*            k_scales  [[buffer(4)]],
    device const float*            v_scales  [[buffer(5)]],
    constant FlashDecodeTileParams& p        [[buffer(6)]],
    threadgroup float*             shared_kv [[threadgroup(0)]],
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

    threadgroup float* shared_k = shared_kv;
    threadgroup float* shared_v = shared_kv + head_dim;

    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    // Iterate over positions in this tile.
    for (uint t = tile_start; t < tile_end; t++) {
        uint kv_offset = (t * p.num_kv_heads + kv_head_idx) * head_dim;
        uint scale_idx = t * p.num_kv_heads + kv_head_idx;

        // Cooperative load K + V from int8 cache, dequantize to float in shared memory.
        float k_scale = k_scales[scale_idx];
        float v_scale = v_scales[scale_idx];
        for (uint d = tid; d < head_dim; d += tg_threads) {
            shared_k[d] = float(k_cache[kv_offset + d]) * k_scale;
            shared_v[d] = float(v_cache[kv_offset + d]) * v_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Dot product Q · K^T.
        float dot_val = 0.0f;
        uint ai = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            dot_val += q_reg[ai++] * shared_k[d];
        }
        float score = simd_sum(dot_val) * p.scale;

        // Online softmax update.
        float new_max = max(max_val, score);
        float correction = exp(max_val - new_max);
        float weight = exp(score - new_max);
        sum_exp = sum_exp * correction + weight;

        ai = 0;
        for (uint d = lane; d < head_dim; d += 32) {
            acc[ai] = acc[ai] * correction + weight * shared_v[d];
            ai++;
        }
        max_val = new_max;

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
