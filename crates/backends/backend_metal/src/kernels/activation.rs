//! Activation, embedding, and utility kernel dispatch wrappers.

use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::context::MetalContext;
use crate::memory::MetalBuffer;
use crate::model::MetalWeight;
use super::{dispatch_kernel, dispatch_kernel_with_tgmem, div_ceil};

/// SwiGLU activation: gate = silu(gate) * up (in-place on gate buffer).
pub fn swiglu(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    gate: &MetalBuffer,
    up: &MetalBuffer,
    n: u32,
) {
    let push = n.to_le_bytes();

    dispatch_kernel(
        encoder,
        &ctx.pipelines.swiglu,
        &[gate, up],
        &push,
        MTLSize { width: div_ceil(n, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Softmax (in-place, single row).
pub fn softmax(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    data: &MetalBuffer,
    n: u32,
) {
    let push = n.to_le_bytes();

    dispatch_kernel(
        encoder,
        &ctx.pipelines.softmax,
        &[data],
        &push,
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// Argmax: find the index of the maximum value (two-stage parallel reduction).
///
/// Stage 1: N_TG threadgroups of 256 threads each scan a chunk, output per-TG max.
/// Stage 2: 1 threadgroup of 256 threads reduces the N_TG intermediate results.
pub fn argmax(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    data: &MetalBuffer,
    intermediate: &MetalBuffer,
    result: &MetalBuffer,
    n: u32,
) {
    const NUM_TG: u32 = 256;

    // Stage 1: parallel scan
    let push1 = n.to_le_bytes();
    dispatch_kernel(
        encoder,
        &ctx.pipelines.argmax_stage1,
        &[data, intermediate],
        &push1,
        MTLSize { width: NUM_TG as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );

    // Stage 2: reduce intermediate results
    let push2 = NUM_TG.to_le_bytes();
    dispatch_kernel(
        encoder,
        &ctx.pipelines.argmax_stage2,
        &[intermediate, result],
        &push2,
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Token embedding lookup.
pub fn embedding(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    embed_table: &MetalBuffer,
    tokens: &MetalBuffer,
    output: &MetalBuffer,
    seq_len: u32,
    hidden_size: u32,
) {
    let total = seq_len * hidden_size;
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&hidden_size.to_le_bytes());
    push[4..8].copy_from_slice(&total.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.embedding,
        &[embed_table, tokens, output],
        &push,
        MTLSize { width: div_ceil(total, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Element-wise residual add: a[i] += b[i] (in-place on a).
pub fn residual_add(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    a: &MetalBuffer,
    b: &MetalBuffer,
    len: u32,
) {
    let push = len.to_le_bytes();

    dispatch_kernel(
        encoder,
        &ctx.pipelines.residual_add,
        &[a, b],
        &push,
        MTLSize { width: div_ceil(len, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Broadcast bias add: data[t * dim + d] += bias[d] for batch_size rows.
pub fn bias_add_batch(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    data: &MetalBuffer,
    bias: &MetalBuffer,
    dim: u32,
    batch_size: u32,
) {
    let total = dim * batch_size;
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&dim.to_le_bytes());
    push[4..8].copy_from_slice(&total.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.bias_add_batch,
        &[data, bias],
        &push,
        MTLSize { width: div_ceil(total, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Weighted accumulation: a[i] += scale * b[i] (in-place on a).
pub fn scaled_add(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    a: &MetalBuffer,
    b: &MetalBuffer,
    len: u32,
    scale: f32,
) {
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&len.to_le_bytes());
    push[4..8].copy_from_slice(&scale.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.scaled_add,
        &[a, b],
        &push,
        MTLSize { width: div_ceil(len, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Fused Gate + Up + SwiGLU for BF16 weights.
///
/// Returns true if the fused path was used, false if weights are not BF16.
/// Note: Q4 fused gate+up was tested but causes L1 thrashing (2x weight reads
/// per inner loop iteration) which hurts on bandwidth-limited M1. Separate
/// gate/up dispatches are faster because they stream one weight matrix at a time.
pub fn fused_gate_up_swiglu(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &MetalBuffer,
    gate_proj: &MetalWeight,
    up_proj: &MetalWeight,
    output: &MetalBuffer,
) -> bool {
    match (gate_proj, up_proj) {
        (MetalWeight::BF16(gate), MetalWeight::BF16(up)) => {
            let n = gate.n as u32;
            let k = gate.k as u32;
            let push = k.to_le_bytes();

            dispatch_kernel_with_tgmem(
                encoder,
                &ctx.pipelines.fused_gate_up_swiglu,
                &[input, &gate.packed, &up.packed, output],
                &push,
                k as usize * 4,
                MTLSize { width: div_ceil(n, 4) as usize, height: 1, depth: 1 },
                MTLSize { width: 128, height: 1, depth: 1 },
            );
            true
        }
        _ => false,
    }
}

/// DeepStack injection: add vision features to image token positions.
///
/// For each image token i at token_indices[i]:
///   embed[token_indices[i] * hidden_size + d] += features[i * hidden_size + d]
pub fn deepstack_add(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    embed: &MetalBuffer,
    features: &MetalBuffer,
    token_indices: &MetalBuffer,
    hidden_size: u32,
    num_image_tokens: u32,
) {
    let total = num_image_tokens * hidden_size;
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&hidden_size.to_le_bytes());
    push[4..8].copy_from_slice(&num_image_tokens.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.deepstack_add,
        &[embed, features, token_indices],
        &push,
        MTLSize { width: div_ceil(total, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Copy float buffer: dst[i] = src[i].
pub fn copy_buffer(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    src: &MetalBuffer,
    dst: &MetalBuffer,
    n: u32,
) {
    let push = n.to_le_bytes();

    dispatch_kernel(
        encoder,
        &ctx.pipelines.copy_buffer,
        &[src, dst],
        &push,
        MTLSize { width: div_ceil(n, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}
