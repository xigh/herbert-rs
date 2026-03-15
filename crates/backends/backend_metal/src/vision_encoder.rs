//! Metal GPU vision encoder forward pass.
//!
//! Orchestrates the ViT pipeline: patch_embed → pos_embed → blocks → merger.
//!
//! When a progress callback is provided, uses MTLSharedEvent with per-group
//! command buffers for async progress notifications (no CPU blocking until
//! the final CB).

use std::sync::Arc;
use core::ptr::NonNull;

use block2::RcBlock;
use herbert_core::error::{HerbertError, Result};
use herbert_vision::encoder::VisionOutput;
use objc2::AnyThread;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLDevice,
    MTLEvent, MTLSharedEvent, MTLSharedEventListener,
};
use tracing::debug;

use crate::context::MetalContext;
use crate::memory::MetalBuffer;
use crate::kernels::{activation, attention, matmul, vision as vk};
use crate::vision_model::*;

/// Scratch GPU buffers reused across blocks.
struct VisionScratch {
    normed: MetalBuffer,
    qkv_buf: MetalBuffer,
    q: MetalBuffer,
    k: MetalBuffer,
    v: MetalBuffer,
    attn_out: MetalBuffer,
    proj_out: MetalBuffer,
    mlp_buf: MetalBuffer,
    mlp_out: MetalBuffer,
}

/// Compute 2D RoPE cos/sin for vision tokens (on CPU, upload to GPU).
///
/// Returns (cos_buf, sin_buf) each [num_tokens, half_dim] f32.
fn compute_vision_rope(
    model: &MetalVisionModel,
    ctx: &MetalContext,
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
) -> Result<(MetalBuffer, MetalBuffer)> {
    let head_dim = model.config.head_dim;
    let half_dim = head_dim / 2;
    let merge_size = model.config.spatial_merge_size;
    let num_tokens = grid_t * grid_h * grid_w;
    let freq_dim = half_dim / 2; // head_dim / 4

    let inv_freq = &model.rot_inv_freq;

    // Build 2D position IDs in merge order
    let merged_h = grid_h / merge_size;
    let merged_w = grid_w / merge_size;

    let mut cos = vec![0.0f32; num_tokens * half_dim];
    let mut sin = vec![0.0f32; num_tokens * half_dim];

    let mut t_idx = 0;
    for _frame in 0..grid_t {
        for bh in 0..merged_h {
            for bw in 0..merged_w {
                for mh in 0..merge_size {
                    for mw in 0..merge_size {
                        let row = bh * merge_size + mh;
                        let col = bw * merge_size + mw;
                        let offset = t_idx * half_dim;
                        for j in 0..freq_dim {
                            let row_angle = row as f32 * inv_freq[j];
                            cos[offset + j] = row_angle.cos();
                            sin[offset + j] = row_angle.sin();
                        }
                        for j in 0..freq_dim {
                            let col_angle = col as f32 * inv_freq[j];
                            cos[offset + freq_dim + j] = col_angle.cos();
                            sin[offset + freq_dim + j] = col_angle.sin();
                        }
                        t_idx += 1;
                    }
                }
            }
        }
    }

    let cos_buf = MetalBuffer::from_f32(&ctx.device, &cos)?;
    let sin_buf = MetalBuffer::from_f32(&ctx.device, &sin)?;
    Ok((cos_buf, sin_buf))
}

/// Bilinear interpolation of position embeddings (CPU-side, trivial cost).
///
/// Returns [num_tokens * hidden_size] f32 in merge order.
fn interpolate_pos_embed(
    pos_embed_data: &[f32],
    config: &herbert_vision::config::VisionConfig,
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
) -> Vec<f32> {
    let hidden_size = config.hidden_size;
    let num_grid = config.num_grid_per_side;
    let merge_size = config.spatial_merge_size;
    let num_tokens = grid_t * grid_h * grid_w;

    let mut result = vec![0.0f32; num_tokens * hidden_size];

    let h_idxs: Vec<f64> = (0..grid_h)
        .map(|i| i as f64 * (num_grid - 1) as f64 / (grid_h.max(1) - 1).max(1) as f64)
        .collect();
    let w_idxs: Vec<f64> = (0..grid_w)
        .map(|i| i as f64 * (num_grid - 1) as f64 / (grid_w.max(1) - 1).max(1) as f64)
        .collect();

    let mut hw_embeds = vec![0.0f32; grid_h * grid_w * hidden_size];
    for (r, &h_idx) in h_idxs.iter().enumerate() {
        let h_floor = (h_idx.floor() as usize).min(num_grid - 1);
        let h_ceil = (h_floor + 1).min(num_grid - 1);
        let dh = h_idx - h_floor as f64;
        for (c, &w_idx) in w_idxs.iter().enumerate() {
            let w_floor = (w_idx.floor() as usize).min(num_grid - 1);
            let w_ceil = (w_floor + 1).min(num_grid - 1);
            let dw = w_idx - w_floor as f64;
            let dst = (r * grid_w + c) * hidden_size;
            let i00 = (h_floor * num_grid + w_floor) * hidden_size;
            let i01 = (h_floor * num_grid + w_ceil) * hidden_size;
            let i10 = (h_ceil * num_grid + w_floor) * hidden_size;
            let i11 = (h_ceil * num_grid + w_ceil) * hidden_size;
            let w00 = ((1.0 - dh) * (1.0 - dw)) as f32;
            let w01 = ((1.0 - dh) * dw) as f32;
            let w10 = (dh * (1.0 - dw)) as f32;
            let w11 = (dh * dw) as f32;
            for d in 0..hidden_size {
                hw_embeds[dst + d] = pos_embed_data[i00 + d] * w00
                    + pos_embed_data[i01 + d] * w01
                    + pos_embed_data[i10 + d] * w10
                    + pos_embed_data[i11 + d] * w11;
            }
        }
    }

    let merged_h = grid_h / merge_size;
    let merged_w = grid_w / merge_size;
    for frame in 0..grid_t {
        for bh in 0..merged_h {
            for bw in 0..merged_w {
                for mh in 0..merge_size {
                    for mw in 0..merge_size {
                        let src_row = bh * merge_size + mh;
                        let src_col = bw * merge_size + mw;
                        let src_idx = (src_row * grid_w + src_col) * hidden_size;
                        let patch_idx = frame * grid_h * grid_w
                            + (bh * merged_w + bw) * merge_size * merge_size
                            + mh * merge_size + mw;
                        let dst_idx = patch_idx * hidden_size;
                        result[dst_idx..dst_idx + hidden_size]
                            .copy_from_slice(&hw_embeds[src_idx..src_idx + hidden_size]);
                    }
                }
            }
        }
    }
    result
}

/// Run merger forward pass on GPU: spatial_merge → (layer_norm) → fc1 → gelu → fc2.
fn merger_forward_gpu(
    ctx: &MetalContext,
    encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    merger: &MetalVisionMerger,
    hidden: &MetalBuffer,
    num_tokens: usize,
    in_dim: usize,
    merge_size: usize,
) -> Result<MetalBuffer> {
    let merge_sq = merge_size * merge_size;
    let merged_dim = in_dim * merge_sq;
    let out_tokens = num_tokens / merge_sq;

    // Spatial merge
    let merged = MetalBuffer::new(&ctx.device, (out_tokens * merged_dim * 4) as u64)?;
    if merger.use_postshuffle_norm {
        // Postshuffle: merge first, then norm
        vk::vision_spatial_merge(ctx, encoder, hidden, &merged, out_tokens as u32, in_dim as u32, merge_sq as u32);
        let normed = MetalBuffer::new(&ctx.device, (out_tokens * merged_dim * 4) as u64)?;
        vk::layer_norm_batch(ctx, encoder, &merged, &merger.norm.weight, &merger.norm.bias, &normed, merged_dim as u32, 1e-6, out_tokens as u32);
        // fc1 → gelu → fc2
        let fc1_out = MetalBuffer::new(&ctx.device, (out_tokens * merger.fc1.out_features * 4) as u64)?;
        matmul::f32_matmul(ctx, encoder, &normed, &merger.fc1.weight, &fc1_out, out_tokens as u32, merger.fc1.out_features as u32, merged_dim as u32);
        activation::bias_add_batch(ctx, encoder, &fc1_out, &merger.fc1.bias, merger.fc1.out_features as u32, out_tokens as u32);
        vk::gelu_batch(ctx, encoder, &fc1_out, (out_tokens * merger.fc1.out_features) as u32);
        let fc2_out = MetalBuffer::new(&ctx.device, (out_tokens * merger.fc2.out_features * 4) as u64)?;
        matmul::f32_matmul(ctx, encoder, &fc1_out, &merger.fc2.weight, &fc2_out, out_tokens as u32, merger.fc2.out_features as u32, merger.fc2.in_features as u32);
        activation::bias_add_batch(ctx, encoder, &fc2_out, &merger.fc2.bias, merger.fc2.out_features as u32, out_tokens as u32);
        Ok(fc2_out)
    } else {
        // Pre-norm: norm at in_dim, then spatial merge
        let normed = MetalBuffer::new(&ctx.device, (num_tokens * in_dim * 4) as u64)?;
        vk::layer_norm_batch(ctx, encoder, hidden, &merger.norm.weight, &merger.norm.bias, &normed, in_dim as u32, 1e-6, num_tokens as u32);
        vk::vision_spatial_merge(ctx, encoder, &normed, &merged, out_tokens as u32, in_dim as u32, merge_sq as u32);
        // fc1 → gelu → fc2
        let fc1_out = MetalBuffer::new(&ctx.device, (out_tokens * merger.fc1.out_features * 4) as u64)?;
        matmul::f32_matmul(ctx, encoder, &merged, &merger.fc1.weight, &fc1_out, out_tokens as u32, merger.fc1.out_features as u32, merged_dim as u32);
        activation::bias_add_batch(ctx, encoder, &fc1_out, &merger.fc1.bias, merger.fc1.out_features as u32, out_tokens as u32);
        vk::gelu_batch(ctx, encoder, &fc1_out, (out_tokens * merger.fc1.out_features) as u32);
        let fc2_out = MetalBuffer::new(&ctx.device, (out_tokens * merger.fc2.out_features * 4) as u64)?;
        matmul::f32_matmul(ctx, encoder, &fc1_out, &merger.fc2.weight, &fc2_out, out_tokens as u32, merger.fc2.out_features as u32, merger.fc2.in_features as u32);
        activation::bias_add_batch(ctx, encoder, &fc2_out, &merger.fc2.bias, merger.fc2.out_features as u32, out_tokens as u32);
        Ok(fc2_out)
    }
}

/// Encode one vision block into the given compute encoder.
fn encode_block(
    ctx: &MetalContext,
    enc: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    block: &MetalVisionBlock,
    hidden: &MetalBuffer,
    scratch: &VisionScratch,
    cos_buf: &MetalBuffer,
    sin_buf: &MetalBuffer,
    num_tokens: usize,
    dim: usize,
    intermediate_size: usize,
    num_heads: usize,
    head_dim: usize,
    half_dim: usize,
    tokens_per_frame: usize,
    grid_t: usize,
    scale: f32,
) {
    // norm1
    vk::layer_norm_batch(ctx, enc, hidden, &block.norm1.weight, &block.norm1.bias,
        &scratch.normed, dim as u32, 1e-6, num_tokens as u32);

    // QKV projection
    matmul::f32_matmul(ctx, enc, &scratch.normed, &block.qkv.weight, &scratch.qkv_buf,
        num_tokens as u32, (3 * dim) as u32, dim as u32);
    activation::bias_add_batch(ctx, enc, &scratch.qkv_buf, &block.qkv.bias,
        (3 * dim) as u32, num_tokens as u32);

    // Split QKV
    vk::vision_split_qkv(ctx, enc, &scratch.qkv_buf, &scratch.q, &scratch.k, &scratch.v,
        num_tokens as u32, dim as u32);

    // Apply RoPE to Q and K
    crate::kernels::rope::rope_batch(ctx, enc, &scratch.q, cos_buf, sin_buf,
        num_tokens as u32, num_heads as u32, half_dim as u32, 0);
    crate::kernels::rope::rope_batch(ctx, enc, &scratch.k, cos_buf, sin_buf,
        num_tokens as u32, num_heads as u32, half_dim as u32, 0);

    // Attention (non-causal via start_pos trick): per-frame
    for frame in 0..grid_t {
        let offset_bytes = frame * tokens_per_frame * dim * 4;
        let q_frame = scratch.q.slice(offset_bytes, tokens_per_frame * dim * 4);
        let k_frame = scratch.k.slice(offset_bytes, tokens_per_frame * dim * 4);
        let v_frame = scratch.v.slice(offset_bytes, tokens_per_frame * dim * 4);
        let attn_frame = scratch.attn_out.slice(offset_bytes, tokens_per_frame * dim * 4);

        // start_pos = tokens_per_frame, cached_len = tokens_per_frame → full attention
        attention::attention_prefill(ctx, enc,
            &q_frame, &k_frame, &v_frame, &attn_frame,
            tokens_per_frame as u32,
            num_heads as u32,
            num_heads as u32, // num_kv_heads = num_heads (no GQA in vision)
            head_dim as u32,
            (num_heads * head_dim) as u32, // kv_dim
            (num_heads * head_dim) as u32, // q_dim
            tokens_per_frame as u32,       // cached_len
            tokens_per_frame as u32,       // start_pos (trick for non-causal)
            scale,
        );
    }

    // Output projection
    matmul::f32_matmul(ctx, enc, &scratch.attn_out, &block.proj.weight, &scratch.proj_out,
        num_tokens as u32, dim as u32, dim as u32);
    activation::bias_add_batch(ctx, enc, &scratch.proj_out, &block.proj.bias,
        dim as u32, num_tokens as u32);

    // Residual add
    activation::residual_add(ctx, enc, hidden, &scratch.proj_out, (num_tokens * dim) as u32);

    // norm2
    vk::layer_norm_batch(ctx, enc, hidden, &block.norm2.weight, &block.norm2.bias,
        &scratch.normed, dim as u32, 1e-6, num_tokens as u32);

    // MLP: fc1 → gelu → fc2
    matmul::f32_matmul(ctx, enc, &scratch.normed, &block.fc1.weight, &scratch.mlp_buf,
        num_tokens as u32, intermediate_size as u32, dim as u32);
    activation::bias_add_batch(ctx, enc, &scratch.mlp_buf, &block.fc1.bias,
        intermediate_size as u32, num_tokens as u32);
    vk::gelu_batch(ctx, enc, &scratch.mlp_buf, (num_tokens * intermediate_size) as u32);

    matmul::f32_matmul(ctx, enc, &scratch.mlp_buf, &block.fc2.weight, &scratch.mlp_out,
        num_tokens as u32, dim as u32, intermediate_size as u32);
    activation::bias_add_batch(ctx, enc, &scratch.mlp_out, &block.fc2.bias,
        dim as u32, num_tokens as u32);

    // Residual add
    activation::residual_add(ctx, enc, hidden, &scratch.mlp_out, (num_tokens * dim) as u32);
}

/// Run the vision encoder forward pass on Metal GPU.
pub fn vision_encode_gpu(
    model: &MetalVisionModel,
    ctx: &MetalContext,
    patches: &[f32],
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
) -> Result<VisionOutput> {
    vision_encode_gpu_inner(model, ctx, patches, grid_t, grid_h, grid_w, None)
}

pub fn vision_encode_gpu_with_progress(
    model: &MetalVisionModel,
    ctx: &MetalContext,
    patches: &[f32],
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    progress: Box<dyn Fn(usize, usize) + Send + Sync>,
) -> Result<VisionOutput> {
    vision_encode_gpu_inner(model, ctx, patches, grid_t, grid_h, grid_w, Some(progress))
}

fn vision_encode_gpu_inner(
    model: &MetalVisionModel,
    ctx: &MetalContext,
    patches: &[f32],
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    progress: Option<Box<dyn Fn(usize, usize) + Send + Sync>>,
) -> Result<VisionOutput> {
    let config = &model.config;
    let dim = config.hidden_size;
    let num_tokens = grid_t * grid_h * grid_w;
    let merge_size = config.spatial_merge_size;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    let half_dim = head_dim / 2;
    let intermediate_size = config.intermediate_size;
    #[cfg(feature = "vision-debug")]
    let profile = std::env::var("PROFILE_VISION").is_ok();
    #[cfg(not(feature = "vision-debug"))]
    let profile: bool = false;
    let _ = profile; // used only when vision-debug feature is enabled

    if patches.len() != num_tokens * config.patch_dim {
        return Err(HerbertError::Backend(format!(
            "patches length {} != num_tokens({}) * patch_dim({})",
            patches.len(), num_tokens, config.patch_dim
        )));
    }

    let t_total = std::time::Instant::now();

    // Upload patches to GPU
    let patches_buf = MetalBuffer::from_f32(&ctx.device, patches)?;

    // 1. Patch embedding: [num_tokens, patch_dim] → [num_tokens, dim]
    let hidden = MetalBuffer::new(&ctx.device, (num_tokens * dim * 4) as u64)?;

    // 2. Position embedding (CPU interpolation)
    let pos_data = model.pos_embed.read_f32(config.num_position_embeddings * dim);
    let pos_embeds = interpolate_pos_embed(&pos_data, config, grid_t, grid_h, grid_w);
    let pos_buf = MetalBuffer::from_f32(&ctx.device, &pos_embeds)?;

    // 3. Compute 2D RoPE cos/sin
    let (cos_buf, sin_buf) = compute_vision_rope(model, ctx, grid_t, grid_h, grid_w)?;

    let tokens_per_frame = grid_h * grid_w;
    let scale = (head_dim as f32).powf(-0.5);

    // 4. Allocate scratch buffers
    let scratch = VisionScratch {
        normed: MetalBuffer::new(&ctx.device, (num_tokens * dim * 4) as u64)?,
        qkv_buf: MetalBuffer::new(&ctx.device, (num_tokens * 3 * dim * 4) as u64)?,
        q: MetalBuffer::new(&ctx.device, (num_tokens * dim * 4) as u64)?,
        k: MetalBuffer::new(&ctx.device, (num_tokens * dim * 4) as u64)?,
        v: MetalBuffer::new(&ctx.device, (num_tokens * dim * 4) as u64)?,
        attn_out: MetalBuffer::new(&ctx.device, (num_tokens * dim * 4) as u64)?,
        proj_out: MetalBuffer::new(&ctx.device, (num_tokens * dim * 4) as u64)?,
        mlp_buf: MetalBuffer::new(&ctx.device, (num_tokens * intermediate_size * 4) as u64)?,
        mlp_out: MetalBuffer::new(&ctx.device, (num_tokens * dim * 4) as u64)?,
    };

    // 5. Forward pass — single CB for all blocks + merger (non-profile), per-block CB (profile)
    let mut deepstack_features = Vec::new();

    #[cfg(feature = "vision-debug")]
    if profile {
        // Profile mode: per-block command buffers with readback and NaN diagnostics
        return vision_encode_profile(
            model, ctx, &hidden, &patches_buf, &pos_buf,
            &cos_buf, &sin_buf, &scratch,
            &mut deepstack_features,
            num_tokens, dim, intermediate_size, num_heads, head_dim, half_dim,
            tokens_per_frame, grid_t, merge_size, scale, t_total,
        );
    }

    // Normal mode
    // If progress callback provided: async per-group CBs with MTLSharedEvent notifications.
    // Otherwise: single CB for maximum throughput.
    let num_blocks = model.blocks.len();
    let has_progress = progress.is_some();
    let group_size = if has_progress {
        std::env::var("VISION_GROUP_SIZE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4)
    } else {
        num_blocks + 1
    };

    // Compute group boundary points: blocks_done values where we split CBs
    let boundaries: Vec<usize> = (1..=num_blocks)
        .filter(|&i| i % group_size == 0 && i < num_blocks)
        .collect();

    // Setup async MTLSharedEvent progress notification
    // We hold shared_event, listener, and RcBlocks alive until after the final wait.
    type NotifyBlock = RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTLSharedEvent>>, u64)>;
    let async_state: Option<(
        objc2::rc::Retained<ProtocolObject<dyn MTLSharedEvent>>,
        objc2::rc::Retained<MTLSharedEventListener>,
        Vec<NotifyBlock>,
    )> = if let Some(progress_fn) = progress {
        if boundaries.is_empty() {
            // No boundaries — report completion synchronously after final wait
            // (e.g., group_size >= num_blocks)
            None
        } else {
            let shared_event = ctx.device.newSharedEvent()
                .ok_or_else(|| HerbertError::Backend("Failed to create MTLSharedEvent".into()))?;

            let queue = dispatch2::DispatchQueue::new("herbert.vision.progress", None);
            let listener = unsafe {
                MTLSharedEventListener::initWithDispatchQueue(
                    MTLSharedEventListener::alloc(),
                    &queue,
                )
            };

            let progress_arc: Arc<dyn Fn(usize, usize) + Send + Sync> = Arc::from(progress_fn);
            let total = num_blocks;
            let mut blocks_to_keep: Vec<NotifyBlock> = Vec::with_capacity(boundaries.len());

            for (idx, &blocks_done) in boundaries.iter().enumerate() {
                let signal_val = (idx + 1) as u64;
                let p = progress_arc.clone();
                let rc_block: NotifyBlock = RcBlock::new(
                    move |_event: NonNull<ProtocolObject<dyn MTLSharedEvent>>, _value: u64| {
                        p(blocks_done, total);
                    },
                );
                unsafe {
                    shared_event.notifyListener_atValue_block(
                        &listener,
                        signal_val,
                        &*rc_block as *const _ as *mut _,
                    );
                }
                blocks_to_keep.push(rc_block);
            }

            Some((shared_event, listener, blocks_to_keep))
        }
    } else {
        None
    };

    let mut cb = ctx.begin_command_buffer()?;
    let mut enc = MetalContext::new_compute_encoder(&cb)?;

    // Patch embed + pos embed
    matmul::f32_matmul(ctx, &enc, &patches_buf, &model.patch_embed.weight, &hidden,
        num_tokens as u32, dim as u32, config.patch_dim as u32);
    activation::bias_add_batch(ctx, &enc, &hidden, &model.patch_embed.bias, dim as u32, num_tokens as u32);
    activation::residual_add(ctx, &enc, &hidden, &pos_buf, (num_tokens * dim) as u32);

    // All blocks — split CBs at group boundaries
    let mut signal_idx = 0usize;
    for (layer_idx, block) in model.blocks.iter().enumerate() {
        encode_block(
            ctx, &enc, block, &hidden, &scratch,
            &cos_buf, &sin_buf,
            num_tokens, dim, intermediate_size, num_heads, head_dim, half_dim,
            tokens_per_frame, grid_t, scale,
        );

        // DeepStack: extract and merge at specified layers
        if let Some(ds_idx) = config.deepstack_visual_indexes.iter().position(|&idx| idx == layer_idx) {
            let ds_out = merger_forward_gpu(ctx, &enc, &model.deepstack_mergers[ds_idx],
                &hidden, num_tokens, dim, merge_size)?;
            let out_tokens = num_tokens / (merge_size * merge_size);
            let out_dim = model.deepstack_mergers[ds_idx].fc2.out_features;
            deepstack_features.push((ds_out, out_tokens, out_dim));
        }

        // At group boundary: commit this CB, start a new one
        if (layer_idx + 1) % group_size == 0 && layer_idx + 1 < num_blocks {
            enc.endEncoding();

            if let Some((ref shared_event, _, _)) = async_state {
                // Async: signal event and commit without blocking
                signal_idx += 1;
                let event_ref: &ProtocolObject<dyn MTLEvent> =
                    ProtocolObject::from_ref(&**shared_event);
                cb.encodeSignalEvent_value(event_ref, signal_idx as u64);
                cb.commit();
            } else {
                // Sync fallback (shouldn't happen when progress is None since
                // group_size = num_blocks + 1 prevents boundaries)
                MetalContext::submit_and_wait(&cb)?;
            }

            cb = ctx.begin_command_buffer()?;
            enc = MetalContext::new_compute_encoder(&cb)?;
        }
    }

    // Final merger
    let final_out = merger_forward_gpu(ctx, &enc, &model.merger,
        &hidden, num_tokens, dim, merge_size)?;
    let out_dim = model.merger.fc2.out_features;

    enc.endEncoding();
    MetalContext::submit_and_wait(&cb)?; // Only blocking wait — all prior CBs complete first (serial queue)

    // Drop async state after final wait (listener, RcBlocks, shared event)
    drop(async_state);

    let num_merged = num_tokens / (merge_size * merge_size);
    let hidden_states = final_out.read_f32(num_merged * out_dim);
    let ds_features: Vec<Vec<f32>> = deepstack_features.iter()
        .map(|(buf, tokens, dim)| buf.read_f32(tokens * dim))
        .collect();

    eprintln!("[metal-vision] Encode: {:.1}ms ({} tokens, {} blocks)",
        t_total.elapsed().as_secs_f64() * 1000.0, num_tokens, model.blocks.len());

    debug!(num_merged, out_dim, deepstack_count = ds_features.len(), "Vision GPU encode complete");

    Ok(VisionOutput {
        hidden_states,
        num_tokens: num_merged,
        deepstack_features: ds_features,
    })
}

/// Profile mode: per-block command buffers with detailed NaN diagnostics.
/// Activated via `PROFILE_VISION=1` env var when built with `vision-debug` feature.
#[cfg(feature = "vision-debug")]
#[allow(clippy::too_many_arguments)]
fn vision_encode_profile(
    model: &MetalVisionModel,
    ctx: &MetalContext,
    hidden: &MetalBuffer,
    patches_buf: &MetalBuffer,
    pos_buf: &MetalBuffer,
    cos_buf: &MetalBuffer,
    sin_buf: &MetalBuffer,
    scratch: &VisionScratch,
    deepstack_features: &mut Vec<(MetalBuffer, usize, usize)>,
    num_tokens: usize,
    dim: usize,
    intermediate_size: usize,
    num_heads: usize,
    head_dim: usize,
    half_dim: usize,
    tokens_per_frame: usize,
    grid_t: usize,
    merge_size: usize,
    scale: f32,
    t_total: std::time::Instant,
) -> Result<VisionOutput> {
    let config = &model.config;

    // Patch embed + pos embed
    {
        let cb = ctx.begin_command_buffer()?;
        let enc = MetalContext::new_compute_encoder(&cb)?;
        matmul::f32_matmul(ctx, &enc, patches_buf, &model.patch_embed.weight, hidden,
            num_tokens as u32, dim as u32, config.patch_dim as u32);
        activation::bias_add_batch(ctx, &enc, hidden, &model.patch_embed.bias, dim as u32, num_tokens as u32);
        activation::residual_add(ctx, &enc, hidden, pos_buf, (num_tokens * dim) as u32);
        enc.endEncoding();
        MetalContext::submit_and_wait(&cb)?;
        let s = hidden.read_f32(4);
        eprintln!("[VISION-GPU] patch+pos: {:?}", s);
    }

    // Per-block forward
    for (layer_idx, block) in model.blocks.iter().enumerate() {
        let cb = ctx.begin_command_buffer()?;
        let enc = MetalContext::new_compute_encoder(&cb)?;

        encode_block(
            ctx, &enc, block, hidden, scratch,
            cos_buf, sin_buf,
            num_tokens, dim, intermediate_size, num_heads, head_dim, half_dim,
            tokens_per_frame, grid_t, scale,
        );

        // DeepStack
        if let Some(ds_idx) = config.deepstack_visual_indexes.iter().position(|&idx| idx == layer_idx) {
            let ds_out = merger_forward_gpu(ctx, &enc, &model.deepstack_mergers[ds_idx],
                hidden, num_tokens, dim, merge_size)?;
            let out_tokens = num_tokens / (merge_size * merge_size);
            let out_dim = model.deepstack_mergers[ds_idx].fc2.out_features;
            deepstack_features.push((ds_out, out_tokens, out_dim));
        }

        enc.endEncoding();
        MetalContext::submit_and_wait(&cb)?;

        // NaN check for early blocks
        if layer_idx < 2 {
            let h = hidden.read_f32(num_tokens * dim);
            let nan_count = h.iter().filter(|v| v.is_nan()).count();
            if nan_count > 0 {
                eprintln!("[VISION-GPU] block {} hidden nan={}/{}", layer_idx, nan_count, h.len());
                let mut nan_tokens: Vec<usize> = Vec::new();
                for t in 0..num_tokens {
                    if h[t * dim..(t + 1) * dim].iter().any(|v| v.is_nan()) {
                        nan_tokens.push(t);
                    }
                }
                eprintln!("[VISION-GPU]   NaN tokens: {:?}", &nan_tokens[..nan_tokens.len().min(20)]);
            } else {
                eprintln!("[VISION-GPU] block {} OK", layer_idx);
            }
        }
    }

    // Final merger
    let num_merged = num_tokens / (merge_size * merge_size);
    let final_out;
    let out_dim;
    {
        let cb = ctx.begin_command_buffer()?;
        let enc = MetalContext::new_compute_encoder(&cb)?;
        final_out = merger_forward_gpu(ctx, &enc, &model.merger,
            hidden, num_tokens, dim, merge_size)?;
        out_dim = model.merger.fc2.out_features;
        enc.endEncoding();
        MetalContext::submit_and_wait(&cb)?;
    }

    eprintln!("[VISION-GPU] Total: {:.1}ms", t_total.elapsed().as_secs_f64() * 1000.0);

    let hidden_states = final_out.read_f32(num_merged * out_dim);
    let ds_features: Vec<Vec<f32>> = deepstack_features.iter()
        .map(|(buf, tokens, d)| buf.read_f32(tokens * d))
        .collect();

    let nan_count = hidden_states.iter().filter(|v| v.is_nan()).count();
    let inf_count = hidden_states.iter().filter(|v| v.is_infinite()).count();
    eprintln!("[VISION-GPU] output: {} tokens, dim={}, NaN={}, Inf={}", num_merged, out_dim, nan_count, inf_count);
    eprintln!("[VISION-GPU] hidden_states[0..8]: {:?}", &hidden_states[..hidden_states.len().min(8)]);

    Ok(VisionOutput {
        hidden_states,
        num_tokens: num_merged,
        deepstack_features: ds_features,
    })
}
