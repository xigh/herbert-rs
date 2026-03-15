#include <metal_stdlib>
using namespace metal;

struct MoePrefillTiledGateUpSwigluBf16Params {
    uint N;       // moe_intermediate_size
    uint K;       // hidden_size
    uint top_k;
};

// Tiled prefill gate+up+SwiGLU for BF16 MoE with counting-sort grouping.
// TILE_M=4 tokens share weight loads. BF16 packed 2 per uint32, no scales.
// Shared memory: flat shared_x[8192], constraint: TILE_M * K <= 8192.
//
// Grid:  (ceil(N/4), ceil(max_count/TILE_M), num_experts)
// Threads: (128, 1, 1)

#define TILE_M 4u

[[kernel]]
void moe_prefill_tiled_gate_up_swiglu_bf16(
    device const float*    x                [[buffer(0)]],
    device const uint*     gate_packed_all  [[buffer(1)]],
    device const uint*     up_packed_all    [[buffer(2)]],
    device const uint*     expert_counts    [[buffer(3)]],
    device const uint*     expert_offsets   [[buffer(4)]],
    device const uint*     sorted_src_idx   [[buffer(5)]],
    device float*          output           [[buffer(6)]],
    constant MoePrefillTiledGateUpSwigluBf16Params& p [[buffer(7)]],
    uint3                  tid_3d           [[thread_position_in_threadgroup]],
    uint3                  gid              [[threadgroup_position_in_grid]]
) {
    uint tid = tid_3d.x;
    uint N = p.N;
    uint K = p.K;
    uint top_k = p.top_k;
    uint num_packed = K / 2;

    uint expert = gid.z;
    uint count = expert_counts[expert];
    uint tile_start = gid.y * TILE_M;
    if (tile_start >= count) return;

    uint start = expert_offsets[expert];
    uint tile_count = min(TILE_M, count - tile_start);

    uint warp_id = tid / 32;
    uint lane = tid % 32;
    uint n = gid.x * 4 + warp_id;

    uint flat_idx[TILE_M];
    uint token_idx[TILE_M];
    for (uint mi = 0; mi < TILE_M; mi++) {
        if (mi < tile_count) {
            flat_idx[mi] = sorted_src_idx[start + tile_start + mi];
            token_idx[mi] = flat_idx[mi] / top_k;
        } else {
            flat_idx[mi] = 0;
            token_idx[mi] = 0;
        }
    }

    threadgroup float shared_x[8192];
    for (uint mi = 0; mi < tile_count; mi++) {
        device const float* token_x = x + token_idx[mi] * K;
        for (uint i = tid; i < K; i += 128) {
            shared_x[mi * K + i] = token_x[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n >= N) return;

    uint expert_row = expert * N + n;
    uint row_offset = expert_row * num_packed;

    float gate_acc[TILE_M];
    float up_acc[TILE_M];
    for (uint mi = 0; mi < TILE_M; mi++) {
        gate_acc[mi] = 0.0f;
        up_acc[mi] = 0.0f;
    }

    for (uint i = lane; i < num_packed; i += 32) {
        uint g_packed = gate_packed_all[row_offset + i];
        float gw0 = as_type<float>((g_packed & 0xFFFFu) << 16);
        float gw1 = as_type<float>(g_packed & 0xFFFF0000u);

        uint u_packed = up_packed_all[row_offset + i];
        float uw0 = as_type<float>((u_packed & 0xFFFFu) << 16);
        float uw1 = as_type<float>(u_packed & 0xFFFF0000u);

        for (uint mi = 0; mi < TILE_M; mi++) {
            float x0 = shared_x[mi * K + i * 2 + 0];
            float x1 = shared_x[mi * K + i * 2 + 1];
            gate_acc[mi] += gw0 * x0 + gw1 * x1;
            up_acc[mi]   += uw0 * x0 + uw1 * x1;
        }
    }

    for (uint mi = 0; mi < TILE_M; mi++) {
        gate_acc[mi] = simd_sum(gate_acc[mi]);
        up_acc[mi] = simd_sum(up_acc[mi]);
    }

    if (lane == 0) {
        for (uint mi = 0; mi < tile_count; mi++) {
            float g = gate_acc[mi];
            float silu_gate = g / (1.0f + exp(-g));
            output[flat_idx[mi] * N + n] = silu_gate * up_acc[mi];
        }
    }
}
