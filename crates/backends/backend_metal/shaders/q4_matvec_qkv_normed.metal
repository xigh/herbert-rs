#include <metal_stdlib>
using namespace metal;

#ifndef Q4_LUT_DEFINED
#define Q4_LUT_DEFINED
constant half q4_lut[16] = {
    -8.0h, -7.0h, -6.0h, -5.0h, -4.0h, -3.0h, -2.0h, -1.0h,
     0.0h,  1.0h,  2.0h,  3.0h,  4.0h,  5.0h,  6.0h,  7.0h
};
#endif

struct Q4MatvecQKVNormedParams {
    uint N_q;    // Q output dim (num_heads * head_dim)
    uint N_k;    // K output dim (num_kv_heads * head_dim)
    uint N_v;    // V output dim (num_kv_heads * head_dim)
    uint K;      // shared input dim (hidden_size)
    float eps;   // RMS norm epsilon
};

// Fused RMS norm + Q/K/V projection with LUT dequantization.
//
// Phase 1 (RMS norm) uses float shared_x for precision (sum-of-squares, rsqrt).
// Phase 2 (Q4 matvec) uses q4_lut[] cast to float + float multiplies for precision.
// shared_x stays as float (K*4 bytes) since the norm phase needs it.
//
// gid.y selects the projection: 0=Q, 1=K, 2=V.
// Dispatch: grid = (ceil(max(N_q,N_k,N_v)/4), 3, 1), threads = (128, 1, 1)

[[kernel]]
void q4_matvec_qkv_normed(
    device const float*    x            [[buffer(0)]],
    device const float*    norm_weight  [[buffer(1)]],
    device const uchar*    w_q_packed   [[buffer(2)]],
    device const float*    s_q          [[buffer(3)]],
    device const uchar*    w_k_packed   [[buffer(4)]],
    device const float*    s_k          [[buffer(5)]],
    device const uchar*    w_v_packed   [[buffer(6)]],
    device const float*    s_v          [[buffer(7)]],
    device float*          out_q        [[buffer(8)]],
    device float*          out_k        [[buffer(9)]],
    device float*          out_v        [[buffer(10)]],
    constant Q4MatvecQKVNormedParams& p [[buffer(11)]],
    threadgroup float*     shared_x     [[threadgroup(0)]],
    uint                   tid          [[thread_index_in_threadgroup]],
    uint2                  gid          [[threadgroup_position_in_grid]]
) {
    uint proj = gid.y;  // 0=Q, 1=K, 2=V
    uint K = p.K;

    // Select projection parameters
    uint N;
    device const uchar* w_packed;
    device const float* scales;
    device float* out;

    if (proj == 0) {
        N = p.N_q; w_packed = w_q_packed; scales = s_q; out = out_q;
    } else if (proj == 1) {
        N = p.N_k; w_packed = w_k_packed; scales = s_k; out = out_k;
    } else {
        N = p.N_v; w_packed = w_v_packed; scales = s_v; out = out_v;
    }

    uint warp_id = tid / 32;
    uint lane = tid % 32;
    uint n = gid.x * 4 + warp_id;

    // Phase 1: Compute RMS norm of x into shared_x (all 128 threads cooperate)

    // 1a: Compute sum of squares
    float sum_sq = 0.0f;
    for (uint i = tid; i < K; i += 128) {
        float v = x[i];
        sum_sq += v * v;
    }

    // SIMD reduction within warp
    sum_sq = simd_sum(sum_sq);

    // Cross-warp reduction: reuse first 4 slots of shared_x
    if (lane == 0) {
        shared_x[warp_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_id == 0) {
        sum_sq = (lane < 4) ? shared_x[lane] : 0.0f;
        sum_sq = simd_sum(sum_sq);
    }

    // Broadcast inv_rms via shared_x[0]
    if (tid == 0) {
        shared_x[0] = rsqrt(sum_sq / float(K) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = shared_x[0];

    // 1b: Normalize x into shared_x
    for (uint i = tid; i < K; i += 128) {
        shared_x[i] = x[i] * inv_rms * norm_weight[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n >= N) return;

    // Phase 2: Q4 dot product using q4_lut (cast to float) + float multiplies.
    // shared_x is float (needed for norm phase).
    uint num_bytes = K / 2;
    uint n_groups = (K + 31) / 32;
    uint num_uint = num_bytes / 4;
    uint row_byte_offset = n * num_bytes;
    uint scale_row_offset = n * n_groups;
    device const uint* w_uint = (device const uint*)(w_packed + row_byte_offset);
    float acc = 0.0f;

    // 4x unrolled loop
    uint i = lane;
    for (; i + 96 < num_uint; i += 128) {
        uint pack0 = w_uint[i];
        uint k0 = i * 8;
        float sc0 = scales[scale_row_offset + k0 / 32];
        acc += float(q4_lut[(pack0      ) & 0xF]) * sc0 * shared_x[k0]
             + float(q4_lut[(pack0 >>  4) & 0xF]) * sc0 * shared_x[k0 + 1]
             + float(q4_lut[(pack0 >>  8) & 0xF]) * sc0 * shared_x[k0 + 2]
             + float(q4_lut[(pack0 >> 12) & 0xF]) * sc0 * shared_x[k0 + 3]
             + float(q4_lut[(pack0 >> 16) & 0xF]) * sc0 * shared_x[k0 + 4]
             + float(q4_lut[(pack0 >> 20) & 0xF]) * sc0 * shared_x[k0 + 5]
             + float(q4_lut[(pack0 >> 24) & 0xF]) * sc0 * shared_x[k0 + 6]
             + float(q4_lut[(pack0 >> 28)       ]) * sc0 * shared_x[k0 + 7];

        uint pack1 = w_uint[i + 32];
        uint k1 = (i + 32) * 8;
        float sc1 = scales[scale_row_offset + k1 / 32];
        acc += float(q4_lut[(pack1      ) & 0xF]) * sc1 * shared_x[k1]
             + float(q4_lut[(pack1 >>  4) & 0xF]) * sc1 * shared_x[k1 + 1]
             + float(q4_lut[(pack1 >>  8) & 0xF]) * sc1 * shared_x[k1 + 2]
             + float(q4_lut[(pack1 >> 12) & 0xF]) * sc1 * shared_x[k1 + 3]
             + float(q4_lut[(pack1 >> 16) & 0xF]) * sc1 * shared_x[k1 + 4]
             + float(q4_lut[(pack1 >> 20) & 0xF]) * sc1 * shared_x[k1 + 5]
             + float(q4_lut[(pack1 >> 24) & 0xF]) * sc1 * shared_x[k1 + 6]
             + float(q4_lut[(pack1 >> 28)       ]) * sc1 * shared_x[k1 + 7];

        uint pack2 = w_uint[i + 64];
        uint k2 = (i + 64) * 8;
        float sc2 = scales[scale_row_offset + k2 / 32];
        acc += float(q4_lut[(pack2      ) & 0xF]) * sc2 * shared_x[k2]
             + float(q4_lut[(pack2 >>  4) & 0xF]) * sc2 * shared_x[k2 + 1]
             + float(q4_lut[(pack2 >>  8) & 0xF]) * sc2 * shared_x[k2 + 2]
             + float(q4_lut[(pack2 >> 12) & 0xF]) * sc2 * shared_x[k2 + 3]
             + float(q4_lut[(pack2 >> 16) & 0xF]) * sc2 * shared_x[k2 + 4]
             + float(q4_lut[(pack2 >> 20) & 0xF]) * sc2 * shared_x[k2 + 5]
             + float(q4_lut[(pack2 >> 24) & 0xF]) * sc2 * shared_x[k2 + 6]
             + float(q4_lut[(pack2 >> 28)       ]) * sc2 * shared_x[k2 + 7];

        uint pack3 = w_uint[i + 96];
        uint k3 = (i + 96) * 8;
        float sc3 = scales[scale_row_offset + k3 / 32];
        acc += float(q4_lut[(pack3      ) & 0xF]) * sc3 * shared_x[k3]
             + float(q4_lut[(pack3 >>  4) & 0xF]) * sc3 * shared_x[k3 + 1]
             + float(q4_lut[(pack3 >>  8) & 0xF]) * sc3 * shared_x[k3 + 2]
             + float(q4_lut[(pack3 >> 12) & 0xF]) * sc3 * shared_x[k3 + 3]
             + float(q4_lut[(pack3 >> 16) & 0xF]) * sc3 * shared_x[k3 + 4]
             + float(q4_lut[(pack3 >> 20) & 0xF]) * sc3 * shared_x[k3 + 5]
             + float(q4_lut[(pack3 >> 24) & 0xF]) * sc3 * shared_x[k3 + 6]
             + float(q4_lut[(pack3 >> 28)       ]) * sc3 * shared_x[k3 + 7];
    }

    // Remainder (non-unrolled)
    for (; i < num_uint; i += 32) {
        uint pack = w_uint[i];
        uint k = i * 8;
        float s = scales[scale_row_offset + k / 32];
        acc += float(q4_lut[(pack      ) & 0xF]) * s * shared_x[k]
             + float(q4_lut[(pack >>  4) & 0xF]) * s * shared_x[k + 1]
             + float(q4_lut[(pack >>  8) & 0xF]) * s * shared_x[k + 2]
             + float(q4_lut[(pack >> 12) & 0xF]) * s * shared_x[k + 3]
             + float(q4_lut[(pack >> 16) & 0xF]) * s * shared_x[k + 4]
             + float(q4_lut[(pack >> 20) & 0xF]) * s * shared_x[k + 5]
             + float(q4_lut[(pack >> 24) & 0xF]) * s * shared_x[k + 6]
             + float(q4_lut[(pack >> 28)       ]) * s * shared_x[k + 7];
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        out[n] = acc;
    }
}
