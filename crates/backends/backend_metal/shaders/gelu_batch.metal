#include <metal_stdlib>
using namespace metal;

struct GeluBatchParams {
    uint n;
};

// GELU activation (tanh approximation), element-wise in-place:
//   y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Dispatch: (ceil(n/256), 1, 1) threadgroups of 256 threads.
kernel void gelu_batch(
    device float* data          [[buffer(0)]],
    constant GeluBatchParams& p [[buffer(1)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= p.n) return;

    float x = data[tid];
    // sqrt(2/pi) = 0.7978845608
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    // Metal's tanh() can return NaN for |inner| > ~44 due to exp(2*inner) overflow.
    // Clamp to ±10 where tanh saturates to ±1 within float32 precision.
    float t = (inner > 10.0f) ? 1.0f : ((inner < -10.0f) ? -1.0f : tanh(inner));
    data[tid] = 0.5f * x * (1.0f + t);
}
