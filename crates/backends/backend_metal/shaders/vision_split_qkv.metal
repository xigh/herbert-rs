#include <metal_stdlib>
using namespace metal;

struct VisionSplitQkvParams {
    uint num_tokens;
    uint dim;   // = num_heads * head_dim
};

// Split fused QKV [num_tokens, 3*dim] into separate Q, K, V [num_tokens, dim].
// Source layout per token: [Q_h0..Q_hN, K_h0..K_hN, V_h0..V_hN]
// Each element maps: tid -> (token, offset_within_dim)
//   Q: qkv[token * 3*dim + offset]
//   K: qkv[token * 3*dim + dim + offset]
//   V: qkv[token * 3*dim + 2*dim + offset]
//
// Dispatch: (ceil(num_tokens * dim / 256), 1, 1) threadgroups of 256.
kernel void vision_split_qkv(
    device const float* qkv [[buffer(0)]],
    device float*       q   [[buffer(1)]],
    device float*       k   [[buffer(2)]],
    device float*       v   [[buffer(3)]],
    constant VisionSplitQkvParams& p [[buffer(4)]],
    uint tid                [[thread_position_in_grid]]
) {
    uint total = p.num_tokens * p.dim;
    if (tid >= total) return;

    uint token  = tid / p.dim;
    uint offset = tid % p.dim;
    uint src    = token * 3u * p.dim;

    q[tid] = qkv[src + offset];
    k[tid] = qkv[src + p.dim + offset];
    v[tid] = qkv[src + 2u * p.dim + offset];
}
