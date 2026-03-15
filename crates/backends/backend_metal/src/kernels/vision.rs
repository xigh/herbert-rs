//! Vision encoder kernel dispatch wrappers.

use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::context::MetalContext;
use crate::memory::MetalBuffer;
use super::{dispatch_kernel, div_ceil};

/// LayerNorm with mean subtraction + bias (vision encoder uses LayerNorm, not RMS norm).
pub fn layer_norm_batch(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &MetalBuffer,
    weight: &MetalBuffer,
    bias: &MetalBuffer,
    output: &MetalBuffer,
    dim: u32,
    eps: f32,
    batch_size: u32,
) {
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&dim.to_le_bytes());
    push[4..8].copy_from_slice(&eps.to_le_bytes());
    push[8..12].copy_from_slice(&batch_size.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.layer_norm_batch,
        &[input, weight, bias, output],
        &push,
        MTLSize { width: batch_size as usize, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// GELU activation (tanh approximation), element-wise in-place.
pub fn gelu_batch(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    data: &MetalBuffer,
    n: u32,
) {
    let push = n.to_le_bytes();

    dispatch_kernel(
        encoder,
        &ctx.pipelines.gelu_batch,
        &[data],
        &push,
        MTLSize { width: div_ceil(n, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Split fused QKV [num_tokens, 3*dim] into separate Q, K, V [num_tokens, dim].
pub fn vision_split_qkv(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    qkv: &MetalBuffer,
    q: &MetalBuffer,
    k: &MetalBuffer,
    v: &MetalBuffer,
    num_tokens: u32,
    dim: u32,
) {
    let total = num_tokens * dim;
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&num_tokens.to_le_bytes());
    push[4..8].copy_from_slice(&dim.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.vision_split_qkv,
        &[qkv, q, k, v],
        &push,
        MTLSize { width: div_ceil(total, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Spatial merge: concatenate merge_size^2 consecutive tokens into one.
pub fn vision_spatial_merge(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &MetalBuffer,
    output: &MetalBuffer,
    num_out_tokens: u32,
    in_dim: u32,
    merge_sq: u32,
) {
    let merged_dim = in_dim * merge_sq;
    let total = num_out_tokens * merged_dim;
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&num_out_tokens.to_le_bytes());
    push[4..8].copy_from_slice(&in_dim.to_le_bytes());
    push[8..12].copy_from_slice(&merge_sq.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.vision_spatial_merge,
        &[input, output],
        &push,
        MTLSize { width: div_ceil(total, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}
