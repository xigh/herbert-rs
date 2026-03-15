//! MoE (Mixture of Experts) kernel dispatch wrappers.

use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::context::MetalContext;
use crate::memory::MetalBuffer;
use crate::model::MetalWeight;
use super::{dispatch_kernel, dispatch_kernel_indirect, dispatch_kernel_with_tgmem, div_ceil};

/// Gather rows by index: output[i*dim + d] = input[indices[i]*dim + d].
pub fn gather(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    output: &MetalBuffer,
    input: &MetalBuffer,
    indices: &MetalBuffer,
    dim: u32,
    count: u32,
) {
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&dim.to_le_bytes());
    push[4..8].copy_from_slice(&count.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.moe_gather,
        &[output, input, indices],
        &push,
        MTLSize { width: div_ceil(count * dim, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Weighted scatter-add: output[indices[i]*dim + d] += weights[i] * input[i*dim + d].
pub fn scatter_weighted_add(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    output: &MetalBuffer,
    input: &MetalBuffer,
    indices: &MetalBuffer,
    weights: &MetalBuffer,
    dim: u32,
    count: u32,
) {
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&dim.to_le_bytes());
    push[4..8].copy_from_slice(&count.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.moe_scatter_add,
        &[output, input, indices, weights],
        &push,
        MTLSize { width: div_ceil(count * dim, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// GPU-side softmax + top-k routing for MoE decode (single token).
pub fn softmax_topk(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    router_logits: &MetalBuffer,
    expert_ids: &MetalBuffer,
    expert_weights: &MetalBuffer,
    num_experts: u32,
    top_k: u32,
    norm_topk_prob: bool,
) {
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&num_experts.to_le_bytes());
    push[4..8].copy_from_slice(&top_k.to_le_bytes());
    push[8..12].copy_from_slice(&(norm_topk_prob as u32).to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.moe_softmax_topk,
        &[router_logits, expert_ids, expert_weights],
        &push,
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// Fused gate+up+SwiGLU for Q4 MoE expert weights.
///
/// Returns true if the Q4 fused path was used.
pub fn fused_gate_up_swiglu_weight(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &MetalBuffer,
    gate_proj: &MetalWeight,
    up_proj: &MetalWeight,
    output: &MetalBuffer,
) -> bool {
    match (gate_proj, up_proj) {
        (MetalWeight::Q4(gate), MetalWeight::Q4(up)) => {
            let n = gate.n as u32;
            let k = gate.k as u32;
            let push = k.to_le_bytes();

            dispatch_kernel_with_tgmem(
                encoder,
                &ctx.pipelines.moe_fused_gate_up_swiglu_q4,
                &[input, &gate.packed, &gate.scales, &up.packed, &up.scales, output],
                &push,
                k as usize * 2,
                MTLSize { width: div_ceil(n, 4) as usize, height: 1, depth: 1 },
                MTLSize { width: 128, height: 1, depth: 1 },
            );
            true
        }
        (MetalWeight::Int8(gate), MetalWeight::Int8(up)) => {
            let n = gate.n as u32;
            let k = gate.k as u32;
            let push = k.to_le_bytes();

            dispatch_kernel_with_tgmem(
                encoder,
                &ctx.pipelines.moe_fused_gate_up_swiglu_int8,
                &[input, &gate.packed, &gate.scales, &up.packed, &up.scales, output],
                &push,
                k as usize * 4,
                MTLSize { width: div_ceil(n, 4) as usize, height: 1, depth: 1 },
                MTLSize { width: 128, height: 1, depth: 1 },
            );
            true
        }
        (MetalWeight::BF16(_gate), MetalWeight::BF16(_up)) => {
            // BF16 fused gate+up+swiglu already exists in activation.rs
            false
        }
        _ => false,
    }
}

// ============================================================================
// Batched MoE kernels (contiguous expert weights, no CPU sync)
// ============================================================================

/// Batched gate+up+SwiGLU for all top-k experts using contiguous weight storage.
///
/// Reads expert_ids from GPU buffer. Each expert's activation is written to
/// output[expert_idx * moe_inter ... (expert_idx+1) * moe_inter].
pub fn batched_gate_up_swiglu(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &MetalBuffer,
    gate_packed: &MetalBuffer,
    gate_scales: Option<&MetalBuffer>,
    up_packed: &MetalBuffer,
    up_scales: Option<&MetalBuffer>,
    expert_ids: &MetalBuffer,
    output: &MetalBuffer,
    moe_inter: u32,
    hidden: u32,
    top_k: u32,
    is_q4: bool,
    is_int8: bool,
) {
    // Shaders load x[hidden] into dynamically-sized threadgroup shared memory
    // Q4 uses half shared_x (hidden*2), Int8/BF16 use float shared_x (hidden*4)
    debug_assert!(hidden * 4 <= 32768, "batched_gate_up_swiglu: hidden={} exceeds max threadgroup memory (32KB)", hidden);
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&moe_inter.to_le_bytes());
    push[4..8].copy_from_slice(&hidden.to_le_bytes());
    push[8..12].copy_from_slice(&top_k.to_le_bytes());

    let grid = MTLSize {
        width: div_ceil(moe_inter, 4) as usize,
        height: top_k as usize,
        depth: 1,
    };
    let threads = MTLSize { width: 128, height: 1, depth: 1 };

    if is_q4 {
        let tgmem = hidden as usize * 2;
        dispatch_kernel_with_tgmem(
            encoder,
            &ctx.pipelines.moe_batched_gate_up_swiglu_q4,
            &[x, gate_packed, gate_scales.unwrap(), up_packed, up_scales.unwrap(), expert_ids, output],
            &push,
            tgmem,
            grid,
            threads,
        );
    } else if is_int8 {
        let tgmem = hidden as usize * 4;
        dispatch_kernel_with_tgmem(
            encoder,
            &ctx.pipelines.moe_batched_gate_up_swiglu_int8,
            &[x, gate_packed, gate_scales.unwrap(), up_packed, up_scales.unwrap(), expert_ids, output],
            &push,
            tgmem,
            grid,
            threads,
        );
    } else {
        // BF16: no scales buffers
        let tgmem = hidden as usize * 4;
        dispatch_kernel_with_tgmem(
            encoder,
            &ctx.pipelines.moe_batched_gate_up_swiglu_bf16,
            &[x, gate_packed, up_packed, expert_ids, output],
            &push,
            tgmem,
            grid,
            threads,
        );
    }
}

/// Batched down projection for all top-k experts using contiguous weight storage.
///
/// Each expert's output is scaled by its routing weight.
/// Results in output[expert_idx * hidden ... (expert_idx+1) * hidden].
pub fn batched_down(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    inputs: &MetalBuffer,
    down_packed: &MetalBuffer,
    down_scales: Option<&MetalBuffer>,
    expert_ids: &MetalBuffer,
    expert_weights: &MetalBuffer,
    output: &MetalBuffer,
    hidden: u32,
    moe_inter: u32,
    top_k: u32,
    is_q4: bool,
    is_int8: bool,
) {
    // Shaders load input[moe_inter] into dynamically-sized threadgroup shared memory
    // Q4 uses half shared_x (moe_inter*2), Int8/BF16 use float shared_x (moe_inter*4)
    debug_assert!(moe_inter * 4 <= 32768, "batched_down: moe_inter={} exceeds max threadgroup memory (32KB)", moe_inter);
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&hidden.to_le_bytes());
    push[4..8].copy_from_slice(&moe_inter.to_le_bytes());
    push[8..12].copy_from_slice(&top_k.to_le_bytes());

    let grid = MTLSize {
        width: div_ceil(hidden, 4) as usize,
        height: top_k as usize,
        depth: 1,
    };
    let threads = MTLSize { width: 128, height: 1, depth: 1 };

    if is_q4 {
        let tgmem = moe_inter as usize * 2;
        dispatch_kernel_with_tgmem(
            encoder,
            &ctx.pipelines.moe_batched_down_q4,
            &[inputs, down_packed, down_scales.unwrap(), expert_ids, expert_weights, output],
            &push,
            tgmem,
            grid,
            threads,
        );
    } else if is_int8 {
        let tgmem = moe_inter as usize * 4;
        dispatch_kernel_with_tgmem(
            encoder,
            &ctx.pipelines.moe_batched_down_int8,
            &[inputs, down_packed, down_scales.unwrap(), expert_ids, expert_weights, output],
            &push,
            tgmem,
            grid,
            threads,
        );
    } else {
        // BF16: no scales
        let tgmem = moe_inter as usize * 4;
        dispatch_kernel_with_tgmem(
            encoder,
            &ctx.pipelines.moe_batched_down_bf16,
            &[inputs, down_packed, expert_ids, expert_weights, output],
            &push,
            tgmem,
            grid,
            threads,
        );
    }
}

/// Reduce MoE expert outputs: output[n] = sum over top_k of per_expert_output[e * hidden + n].
pub fn moe_reduce(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    per_expert_output: &MetalBuffer,
    output: &MetalBuffer,
    hidden: u32,
    top_k: u32,
) {
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&hidden.to_le_bytes());
    push[4..8].copy_from_slice(&top_k.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.moe_reduce,
        &[per_expert_output, output],
        &push,
        MTLSize { width: div_ceil(hidden, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Fused MoE reduce + residual add: residual[n] += sum of per_expert_output[e * hidden + n].
pub fn moe_reduce_residual(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    per_expert_output: &MetalBuffer,
    residual: &MetalBuffer,
    hidden: u32,
    top_k: u32,
) {
    let mut push = [0u8; 8];
    push[0..4].copy_from_slice(&hidden.to_le_bytes());
    push[4..8].copy_from_slice(&top_k.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.moe_reduce_residual,
        &[per_expert_output, residual],
        &push,
        MTLSize { width: div_ceil(hidden, 256) as usize, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

// ============================================================================
// Prefill MoE kernels (zero-sync, batched over tokens)
// ============================================================================

/// Batched softmax + top-k routing for prefill (M tokens in parallel).
pub fn softmax_topk_batch(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    router_logits: &MetalBuffer,
    expert_ids: &MetalBuffer,
    expert_weights: &MetalBuffer,
    num_experts: u32,
    top_k: u32,
    norm_topk_prob: bool,
    num_tokens: u32,
) {
    let mut push = [0u8; 16];
    push[0..4].copy_from_slice(&num_experts.to_le_bytes());
    push[4..8].copy_from_slice(&top_k.to_le_bytes());
    push[8..12].copy_from_slice(&(norm_topk_prob as u32).to_le_bytes());
    push[12..16].copy_from_slice(&num_tokens.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.moe_softmax_topk_batch,
        &[router_logits, expert_ids, expert_weights],
        &push,
        MTLSize { width: 1, height: num_tokens as usize, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
}

/// GPU counting sort: group M*top_k assignments by expert_id.
/// Also computes max_count and writes indirect dispatch grids for tiled kernels.
/// `is_q4`: Q4 uses simdgroup matmul tiles (TILE_N=32, TILE_M=16),
///          Int8/BF16 use scalar tiles (TILE_N=4, TILE_M=4).
pub fn prefill_sort(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    expert_ids: &MetalBuffer,
    expert_counts: &MetalBuffer,
    expert_offsets: &MetalBuffer,
    sorted_src_idx: &MetalBuffer,
    indirect_args: &MetalBuffer,
    total: u32,
    num_experts: u32,
    top_k: u32,
    moe_inter: u32,
    hidden: u32,
    is_q4: bool,
) {
    // Q4 simdgroup matmul: TILE_N=32, TILE_M=16
    // Int8/BF16 scalar: TILE_N=4, TILE_M=4
    let (tile_n_gate, tile_m_gate, tile_n_down, tile_m_down) = if is_q4 {
        (32u32, 16u32, 32u32, 16u32)
    } else {
        (4u32, 4u32, 4u32, 4u32)
    };

    let mut push = [0u8; 36];
    push[0..4].copy_from_slice(&total.to_le_bytes());
    push[4..8].copy_from_slice(&num_experts.to_le_bytes());
    push[8..12].copy_from_slice(&top_k.to_le_bytes());
    push[12..16].copy_from_slice(&moe_inter.to_le_bytes());
    push[16..20].copy_from_slice(&hidden.to_le_bytes());
    push[20..24].copy_from_slice(&tile_n_gate.to_le_bytes());
    push[24..28].copy_from_slice(&tile_m_gate.to_le_bytes());
    push[28..32].copy_from_slice(&tile_n_down.to_le_bytes());
    push[32..36].copy_from_slice(&tile_m_down.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.moe_prefill_sort,
        &[expert_ids, expert_counts, expert_offsets, sorted_src_idx, indirect_args],
        &push,
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Tiled prefill gate+up+SwiGLU with counting-sort grouping.
/// Q4: simdgroup_matrix (256 threads, TILE_M=16, TILE_N=32).
/// Int8/BF16: scalar (128 threads, TILE_M=4, TILE_N=4).
/// Grid dimensions come from `indirect_args` buffer (offset 0) written by `prefill_sort`.
#[allow(clippy::too_many_arguments)]
pub fn prefill_tiled_gate_up_swiglu(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &MetalBuffer,
    gate_packed: &MetalBuffer,
    gate_scales: Option<&MetalBuffer>,
    up_packed: &MetalBuffer,
    up_scales: Option<&MetalBuffer>,
    expert_counts: &MetalBuffer,
    expert_offsets: &MetalBuffer,
    sorted_src_idx: &MetalBuffer,
    output: &MetalBuffer,
    indirect_args: &MetalBuffer,
    moe_inter: u32,
    hidden: u32,
    top_k: u32,
    is_q4: bool,
    is_int8: bool,
) {
    if !is_q4 {
        debug_assert!(hidden * 4 <= 8192, "prefill_tiled_gate_up_swiglu: TILE_M*hidden={} exceeds shared memory (8192 floats)", hidden * 4);
    }
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&moe_inter.to_le_bytes());
    push[4..8].copy_from_slice(&hidden.to_le_bytes());
    push[8..12].copy_from_slice(&top_k.to_le_bytes());

    let threads = if is_q4 {
        MTLSize { width: 256, height: 1, depth: 1 }
    } else {
        MTLSize { width: 128, height: 1, depth: 1 }
    };

    if is_q4 {
        dispatch_kernel_indirect(
            encoder,
            &ctx.pipelines.moe_prefill_tiled_gate_up_swiglu_q4,
            &[x, gate_packed, gate_scales.unwrap(), up_packed, up_scales.unwrap(),
              expert_counts, expert_offsets, sorted_src_idx, output],
            &push,
            indirect_args,
            0,
            threads,
        );
    } else if is_int8 {
        dispatch_kernel_indirect(
            encoder,
            &ctx.pipelines.moe_prefill_tiled_gate_up_swiglu_int8,
            &[x, gate_packed, gate_scales.unwrap(), up_packed, up_scales.unwrap(),
              expert_counts, expert_offsets, sorted_src_idx, output],
            &push,
            indirect_args,
            0,
            threads,
        );
    } else {
        dispatch_kernel_indirect(
            encoder,
            &ctx.pipelines.moe_prefill_tiled_gate_up_swiglu_bf16,
            &[x, gate_packed, up_packed,
              expert_counts, expert_offsets, sorted_src_idx, output],
            &push,
            indirect_args,
            0,
            threads,
        );
    }
}

/// Tiled prefill down projection with counting-sort grouping (TILE_M=4).
/// Grid dimensions come from `indirect_args` buffer (offset 12) written by `prefill_sort`.
#[allow(clippy::too_many_arguments)]
pub fn prefill_tiled_down(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    inputs: &MetalBuffer,
    down_packed: &MetalBuffer,
    down_scales: Option<&MetalBuffer>,
    expert_weights: &MetalBuffer,
    expert_counts: &MetalBuffer,
    expert_offsets: &MetalBuffer,
    sorted_src_idx: &MetalBuffer,
    output: &MetalBuffer,
    indirect_args: &MetalBuffer,
    hidden: u32,
    moe_inter: u32,
    top_k: u32,
    is_q4: bool,
    is_int8: bool,
) {
    debug_assert!(moe_inter * 4 <= 8192, "prefill_tiled_down: TILE_M*moe_inter={} exceeds shared memory (8192 floats)", moe_inter * 4);
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&hidden.to_le_bytes());
    push[4..8].copy_from_slice(&moe_inter.to_le_bytes());
    push[8..12].copy_from_slice(&top_k.to_le_bytes());

    let threads = MTLSize { width: 128, height: 1, depth: 1 };

    if is_q4 {
        dispatch_kernel_indirect(
            encoder,
            &ctx.pipelines.moe_prefill_tiled_down_q4,
            &[inputs, down_packed, down_scales.unwrap(), expert_weights,
              expert_counts, expert_offsets, sorted_src_idx, output],
            &push,
            indirect_args,
            12,
            threads,
        );
    } else if is_int8 {
        dispatch_kernel_indirect(
            encoder,
            &ctx.pipelines.moe_prefill_tiled_down_int8,
            &[inputs, down_packed, down_scales.unwrap(), expert_weights,
              expert_counts, expert_offsets, sorted_src_idx, output],
            &push,
            indirect_args,
            12,
            threads,
        );
    } else {
        dispatch_kernel_indirect(
            encoder,
            &ctx.pipelines.moe_prefill_tiled_down_bf16,
            &[inputs, down_packed, expert_weights,
              expert_counts, expert_offsets, sorted_src_idx, output],
            &push,
            indirect_args,
            12,
            threads,
        );
    }
}

/// Fused tiled prefill down projection + reduce + residual add (v2).
/// Uses per-(token, column) uint atomic counters instead of float CAS.
/// Q4: simdgroup_matrix (256 threads, TILE_M=16, TILE_N=32).
/// Int8/BF16: scalar (128 threads, TILE_M=4, TILE_N=4).
/// Grid dimensions come from `indirect_args` buffer (offset 12) written by `prefill_sort`.
#[allow(clippy::too_many_arguments)]
pub fn prefill_tiled_down_residual(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    inputs: &MetalBuffer,
    down_packed: &MetalBuffer,
    down_scales: Option<&MetalBuffer>,
    expert_weights: &MetalBuffer,
    expert_counts: &MetalBuffer,
    expert_offsets: &MetalBuffer,
    sorted_src_idx: &MetalBuffer,
    down_out: &MetalBuffer,
    counters: &MetalBuffer,
    residual: &MetalBuffer,
    indirect_args: &MetalBuffer,
    hidden: u32,
    moe_inter: u32,
    top_k: u32,
    is_q4: bool,
    is_int8: bool,
) {
    if !is_q4 {
        debug_assert!(moe_inter * 4 <= 8192, "prefill_tiled_down_residual: TILE_M*moe_inter={} exceeds shared memory (8192 floats)", moe_inter * 4);
    }
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&hidden.to_le_bytes());
    push[4..8].copy_from_slice(&moe_inter.to_le_bytes());
    push[8..12].copy_from_slice(&top_k.to_le_bytes());

    let threads = if is_q4 {
        MTLSize { width: 256, height: 1, depth: 1 }
    } else {
        MTLSize { width: 128, height: 1, depth: 1 }
    };

    if is_q4 {
        dispatch_kernel_indirect(
            encoder,
            &ctx.pipelines.moe_prefill_tiled_down_residual_q4,
            &[inputs, down_packed, down_scales.unwrap(), expert_weights,
              expert_counts, expert_offsets, sorted_src_idx, down_out, counters, residual],
            &push,
            indirect_args,
            12,
            threads,
        );
    } else if is_int8 {
        dispatch_kernel_indirect(
            encoder,
            &ctx.pipelines.moe_prefill_tiled_down_residual_int8,
            &[inputs, down_packed, down_scales.unwrap(), expert_weights,
              expert_counts, expert_offsets, sorted_src_idx, down_out, counters, residual],
            &push,
            indirect_args,
            12,
            threads,
        );
    } else {
        dispatch_kernel_indirect(
            encoder,
            &ctx.pipelines.moe_prefill_tiled_down_residual_bf16,
            &[inputs, down_packed, expert_weights,
              expert_counts, expert_offsets, sorted_src_idx, down_out, counters, residual],
            &push,
            indirect_args,
            12,
            threads,
        );
    }
}

/// Prefill fused MoE reduce + residual add (batched over tokens).
pub fn prefill_reduce_residual(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    per_expert_output: &MetalBuffer,
    residual: &MetalBuffer,
    hidden: u32,
    top_k: u32,
    num_tokens: u32,
) {
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&hidden.to_le_bytes());
    push[4..8].copy_from_slice(&top_k.to_le_bytes());
    push[8..12].copy_from_slice(&num_tokens.to_le_bytes());

    dispatch_kernel(
        encoder,
        &ctx.pipelines.moe_prefill_reduce_residual,
        &[per_expert_output, residual],
        &push,
        MTLSize { width: div_ceil(hidden, 256) as usize, height: num_tokens as usize, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
}

/// Fused matvec + scaled accumulation: output[n] += scale * matvec(w, x)[n].
///
/// Used for MoE expert down projection. Returns true if fused path was used.
pub fn matvec_scaled_add_weight(
    ctx: &MetalContext,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    input: &MetalBuffer,
    weight: &MetalWeight,
    output: &MetalBuffer,
    scale: f32,
) -> bool {
    match weight {
        MetalWeight::Q4(w) => {
            let n = w.n as u32;
            let k = w.k as u32;
            let mut push = [0u8; 8];
            push[0..4].copy_from_slice(&k.to_le_bytes());
            push[4..8].copy_from_slice(&scale.to_le_bytes());

            dispatch_kernel_with_tgmem(
                encoder,
                &ctx.pipelines.matvec_scaled_add_q4,
                &[input, &w.packed, &w.scales, output],
                &push,
                k as usize * 2,
                MTLSize { width: div_ceil(n, 4) as usize, height: 1, depth: 1 },
                MTLSize { width: 128, height: 1, depth: 1 },
            );
            true
        }
        MetalWeight::Int8(w) => {
            let n = w.n as u32;
            let k = w.k as u32;
            let mut push = [0u8; 8];
            push[0..4].copy_from_slice(&k.to_le_bytes());
            push[4..8].copy_from_slice(&scale.to_le_bytes());

            dispatch_kernel_with_tgmem(
                encoder,
                &ctx.pipelines.matvec_scaled_add_int8,
                &[input, &w.packed, &w.scales, output],
                &push,
                k as usize * 4,
                MTLSize { width: div_ceil(n, 4) as usize, height: 1, depth: 1 },
                MTLSize { width: 128, height: 1, depth: 1 },
            );
            true
        }
        MetalWeight::BF16(w) => {
            let n = w.n as u32;
            let k = w.k as u32;
            let mut push = [0u8; 8];
            push[0..4].copy_from_slice(&k.to_le_bytes());
            push[4..8].copy_from_slice(&scale.to_le_bytes());

            dispatch_kernel_with_tgmem(
                encoder,
                &ctx.pipelines.matvec_scaled_add_bf16,
                &[input, &w.packed, output],
                &push,
                k as usize * 4,
                MTLSize { width: div_ceil(n, 4) as usize, height: 1, depth: 1 },
                MTLSize { width: 128, height: 1, depth: 1 },
            );
            true
        }
    }
}
