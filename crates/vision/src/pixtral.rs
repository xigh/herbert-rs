//! Pixtral vision encoder for Mistral3/Devstral models.
//!
//! Architecturally different from Qwen3-VL:
//! - Conv2D patch embedding (no bias)
//! - RMSNorm (not LayerNorm)
//! - Separate Q/K/V/O projections (not fused QKV)
//! - SwiGLU MLP (not GELU)
//! - 2D RoPE (not learned position embeddings)
//! - No DeepStack

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use memmap2::Mmap;
use herbert_core::error::{HerbertError, Result};
use safetensors::SafeTensors;
use tracing::{debug, info};

use crate::image_process::bilinear_resize_rgb;
use crate::pixtral_config::PixtralVisionConfig;

// --- Weight Structures (no bias) ---

pub struct RmsNorm {
    pub weight: Vec<f32>,
    pub dim: usize,
    pub eps: f32,
}

pub struct LinearNoBias {
    pub weight: Vec<f32>, // [out_f, in_f] row-major
    pub in_f: usize,
    pub out_f: usize,
}

pub struct PixtralAttention {
    pub q_proj: LinearNoBias,
    pub k_proj: LinearNoBias,
    pub v_proj: LinearNoBias,
    pub o_proj: LinearNoBias,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scaling: f32,
}

pub struct PixtralMLP {
    pub gate_proj: LinearNoBias, // [inter, dim]
    pub up_proj: LinearNoBias,   // [inter, dim]
    pub down_proj: LinearNoBias, // [dim, inter]
}

pub struct PixtralBlock {
    pub attention_norm: RmsNorm,
    pub attn: PixtralAttention,
    pub ffn_norm: RmsNorm,
    pub ffn: PixtralMLP,
}

pub struct MultiModalProjector {
    pub norm: RmsNorm,
    pub merging_layer: LinearNoBias, // [hidden, hidden * merge^2]
    pub linear_1: LinearNoBias,      // [text_hidden, hidden]
    pub linear_2: LinearNoBias,      // [text_hidden, text_hidden]
    pub spatial_merge_size: usize,
}

pub struct PixtralEncoder {
    pub config: PixtralVisionConfig,
    pub patch_conv: LinearNoBias, // Conv2D reshaped: [hidden, 3*ps*ps]
    pub ln_pre: RmsNorm,
    pub blocks: Vec<PixtralBlock>,
    pub projector: MultiModalProjector,
    /// Pre-computed 2D RoPE cos/sin: [max_patches * max_patches, head_dim]
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
    pub max_patches_per_side: usize,
}

// --- Thread helpers ---

fn num_threads() -> usize {
    use std::sync::OnceLock;
    static NT: OnceLock<usize> = OnceLock::new();
    *NT.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    })
}

#[derive(Clone, Copy)]
struct SendMutPtr(*mut f32);
unsafe impl Send for SendMutPtr {}
unsafe impl Sync for SendMutPtr {}

impl SendMutPtr {
    #[inline]
    unsafe fn write(self, offset: usize, val: f32) {
        *self.0.add(offset) = val;
    }
}

// --- BF16 Precision ---

/// Truncate f32 values to BF16 precision in-place.
///
/// The Pixtral vision encoder was trained in BF16. Running in FP32 causes
/// hidden state values to diverge because FP32 preserves tiny accumulation
/// errors that BF16 truncation would discard.
#[inline]
fn truncate_to_bf16_inplace(data: &mut [f32]) {
    for v in data.iter_mut() {
        let bits = v.to_bits();
        let rounding = 0x7FFFu32 + ((bits >> 16) & 1);
        let rounded = bits.wrapping_add(rounding) & 0xFFFF0000;
        *v = f32::from_bits(rounded);
    }
}

// --- Kernels ---

fn rms_norm(input: &[f32], norm: &RmsNorm, output: &mut [f32]) {
    let dim = norm.dim;
    debug_assert_eq!(input.len() % dim, 0);
    let n = input.len() / dim;
    let nt = num_threads().min(n);

    if nt <= 1 {
        rms_norm_chunk(input, norm, output);
        return;
    }

    let chunk = n.div_ceil(nt);
    std::thread::scope(|s| {
        let mut inp = input;
        let mut out = output;
        for _ in 0..nt {
            let rows = chunk.min(inp.len() / dim);
            if rows == 0 { break; }
            let (inp_c, inp_r) = inp.split_at(rows * dim);
            let (out_c, out_r) = out.split_at_mut(rows * dim);
            inp = inp_r;
            out = out_r;
            s.spawn(move || rms_norm_chunk(inp_c, norm, out_c));
        }
    });
}

fn rms_norm_chunk(input: &[f32], norm: &RmsNorm, output: &mut [f32]) {
    let dim = norm.dim;
    let n = input.len() / dim;
    for i in 0..n {
        let x = &input[i * dim..(i + 1) * dim];
        let out = &mut output[i * dim..(i + 1) * dim];
        let sq_sum: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv_rms = 1.0 / (sq_sum + norm.eps).sqrt();
        for j in 0..dim {
            out[j] = x[j] * inv_rms * norm.weight[j];
        }
    }
}

fn linear_no_bias_forward(input: &[f32], linear: &LinearNoBias, output: &mut [f32]) {
    debug_assert_eq!(input.len() % linear.in_f, 0);
    let m = input.len() / linear.in_f;
    let nt = num_threads().min(m);

    if nt <= 1 {
        linear_no_bias_chunk(input, linear, output);
        return;
    }

    let in_f = linear.in_f;
    let out_f = linear.out_f;
    let chunk = m.div_ceil(nt);

    std::thread::scope(|s| {
        let mut inp = input;
        let mut out = output;
        for _ in 0..nt {
            let rows = chunk.min(inp.len() / in_f);
            if rows == 0 { break; }
            let (inp_c, inp_r) = inp.split_at(rows * in_f);
            let (out_c, out_r) = out.split_at_mut(rows * out_f);
            inp = inp_r;
            out = out_r;
            s.spawn(move || linear_no_bias_chunk(inp_c, linear, out_c));
        }
    });
}

fn linear_no_bias_chunk(input: &[f32], linear: &LinearNoBias, output: &mut [f32]) {
    let in_f = linear.in_f;
    let out_f = linear.out_f;
    let m = input.len() / in_f;
    for i in 0..m {
        let x = &input[i * in_f..(i + 1) * in_f];
        let out = &mut output[i * out_f..(i + 1) * out_f];
        for (j, out_j) in out.iter_mut().enumerate() {
            let w = &linear.weight[j * in_f..(j + 1) * in_f];
            let mut sum = 0.0f32;
            for k in 0..in_f {
                sum += x[k] * w[k];
            }
            *out_j = sum;
        }
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn gelu_quick(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh())
}

fn gelu_inplace(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = gelu_quick(*v);
    }
}

// --- 2D RoPE ---

/// Pre-compute 2D RoPE cos/sin tables for all possible grid positions.
///
/// For Pixtral: position (h, w) maps to angles:
///   first half of head_dim/2 uses h * freq[i] (height frequencies)
///   second half of head_dim/2 uses w * freq[i] (width frequencies)
/// Then duplicate for rotate_half pattern.
fn precompute_2d_rope(
    max_patches: usize,
    head_dim: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let quarter = half / 2;
    let total = max_patches * max_patches;

    // Compute base frequencies: freq[i] = 1/theta^(2i/head_dim)
    let mut freqs = vec![0.0f32; half];
    for i in 0..half {
        freqs[i] = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
    }

    // Split: even-indexed freqs for height, odd-indexed for width
    let freqs_h: Vec<f32> = (0..quarter).map(|i| freqs[i * 2]).collect();
    let freqs_w: Vec<f32> = (0..quarter).map(|i| freqs[i * 2 + 1]).collect();

    let mut cos = vec![0.0f32; total * head_dim];
    let mut sin = vec![0.0f32; total * head_dim];

    for h in 0..max_patches {
        for w in 0..max_patches {
            let idx = h * max_patches + w;
            let offset = idx * head_dim;

            // First quarter: h * freqs_h
            for j in 0..quarter {
                let angle = h as f32 * freqs_h[j];
                cos[offset + j] = angle.cos();
                sin[offset + j] = angle.sin();
            }
            // Second quarter: w * freqs_w
            for j in 0..quarter {
                let angle = w as f32 * freqs_w[j];
                cos[offset + quarter + j] = angle.cos();
                sin[offset + quarter + j] = angle.sin();
            }
            // Second half: duplicate first half
            for j in 0..half {
                cos[offset + half + j] = cos[offset + j];
                sin[offset + half + j] = sin[offset + j];
            }
        }
    }

    (cos, sin)
}

/// Apply rotate_half RoPE in-place to a single head vector.
fn rotate_half_inplace(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let head_dim = x.len();
    let half = head_dim / 2;
    for j in 0..half {
        let x1 = x[j];
        let x2 = x[j + half];
        x[j] = x1 * cos[j] + (-x2) * sin[j];
        x[j + half] = x2 * cos[j + half] + x1 * sin[j + half];
    }
}

// --- Attention ---

fn pixtral_attention_forward(
    attn: &PixtralAttention,
    hidden_states: &[f32],
    rope_cos: &[f32],
    rope_sin: &[f32],
    position_ids: &[usize],
    num_tokens: usize,
    output: &mut [f32],
) {
    let dim = attn.num_heads * attn.head_dim;
    let head_dim = attn.head_dim;
    let num_heads = attn.num_heads;

    // Q, K, V projections
    let mut q = vec![0.0f32; num_tokens * dim];
    let mut k = vec![0.0f32; num_tokens * dim];
    let mut v = vec![0.0f32; num_tokens * dim];
    linear_no_bias_forward(hidden_states, &attn.q_proj, &mut q);
    linear_no_bias_forward(hidden_states, &attn.k_proj, &mut k);
    linear_no_bias_forward(hidden_states, &attn.v_proj, &mut v);

    // Apply 2D RoPE to Q and K
    for t in 0..num_tokens {
        let pos_id = position_ids[t];
        let cos_t = &rope_cos[pos_id * head_dim..(pos_id + 1) * head_dim];
        let sin_t = &rope_sin[pos_id * head_dim..(pos_id + 1) * head_dim];
        for h in 0..num_heads {
            let offset = t * dim + h * head_dim;
            rotate_half_inplace(&mut q[offset..offset + head_dim], cos_t, sin_t);
            rotate_half_inplace(&mut k[offset..offset + head_dim], cos_t, sin_t);
        }
    }

    // Full bidirectional attention (no causal mask)
    let mut attn_output = vec![0.0f32; num_tokens * dim];
    let scaling = attn.scaling;

    // Parallel over heads
    let work: Vec<usize> = (0..num_heads).collect();
    let nt = num_threads().min(work.len());
    let attn_out_ptr = SendMutPtr(attn_output.as_mut_ptr());

    std::thread::scope(|s| {
        let chunk = work.len().div_ceil(nt);
        for items in work.chunks(chunk) {
            let q_ref = &q;
            let k_ref = &k;
            let v_ref = &v;
            s.spawn(move || {
                let mut scores = vec![0.0f32; num_tokens * num_tokens];
                for &h in items {
                    // Compute scores
                    for qi in 0..num_tokens {
                        for ki in 0..num_tokens {
                            let q_off = qi * dim + h * head_dim;
                            let k_off = ki * dim + h * head_dim;
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += q_ref[q_off + d] * k_ref[k_off + d];
                            }
                            scores[qi * num_tokens + ki] = dot * scaling;
                        }
                    }
                    // Softmax per row
                    for qi in 0..num_tokens {
                        let row = &mut scores[qi * num_tokens..(qi + 1) * num_tokens];
                        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0f32;
                        for sv in row.iter_mut() {
                            *sv = (*sv - max_val).exp();
                            sum += *sv;
                        }
                        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                        for sv in row.iter_mut() {
                            *sv *= inv_sum;
                        }
                    }
                    // Weighted sum
                    for qi in 0..num_tokens {
                        let out_off = qi * dim + h * head_dim;
                        for d in 0..head_dim {
                            let mut sum = 0.0f32;
                            for vi in 0..num_tokens {
                                let v_off = vi * dim + h * head_dim;
                                sum += scores[qi * num_tokens + vi] * v_ref[v_off + d];
                            }
                            unsafe { attn_out_ptr.write(out_off + d, sum); }
                        }
                    }
                }
            });
        }
    });

    // Output projection
    linear_no_bias_forward(&attn_output, &attn.o_proj, output);
}

// --- Block Forward ---

fn pixtral_block_forward(
    block: &PixtralBlock,
    hidden_states: &mut [f32],
    rope_cos: &[f32],
    rope_sin: &[f32],
    position_ids: &[usize],
    num_tokens: usize,
    dim: usize,
) {
    let inter = block.ffn.gate_proj.out_f;

    // Attention: residual + attn(norm(x))
    let mut normed = vec![0.0f32; num_tokens * dim];
    rms_norm(hidden_states, &block.attention_norm, &mut normed);

    let mut attn_out = vec![0.0f32; num_tokens * dim];
    pixtral_attention_forward(
        &block.attn, &normed, rope_cos, rope_sin, position_ids, num_tokens, &mut attn_out,
    );

    for i in 0..hidden_states.len() {
        hidden_states[i] += attn_out[i];
    }

    // FFN: residual + swiglu(norm(x))
    rms_norm(hidden_states, &block.ffn_norm, &mut normed);

    // SwiGLU: down(silu(gate(x)) * up(x))
    let mut gate_out = vec![0.0f32; num_tokens * inter];
    let mut up_out = vec![0.0f32; num_tokens * inter];
    linear_no_bias_forward(&normed, &block.ffn.gate_proj, &mut gate_out);
    linear_no_bias_forward(&normed, &block.ffn.up_proj, &mut up_out);

    for i in 0..gate_out.len() {
        gate_out[i] = silu(gate_out[i]) * up_out[i];
    }

    let mut ffn_out = vec![0.0f32; num_tokens * dim];
    linear_no_bias_forward(&gate_out, &block.ffn.down_proj, &mut ffn_out);

    for i in 0..hidden_states.len() {
        hidden_states[i] += ffn_out[i];
    }
}

// --- Projector Forward ---

fn projector_forward(
    proj: &MultiModalProjector,
    hidden: &[f32],
    num_tokens: usize,
    dim: usize,
    grid_h: usize,
    grid_w: usize,
) -> Vec<f32> {
    let merge = proj.spatial_merge_size;
    let merged_h = grid_h / merge;
    let merged_w = grid_w / merge;
    let out_tokens = merged_h * merged_w;
    let merged_dim = dim * merge * merge;

    // 1. RMSNorm
    let mut normed = vec![0.0f32; num_tokens * dim];
    rms_norm(hidden, &proj.norm, &mut normed);

    // 2. Spatial merge: reshape [grid_h, grid_w, dim] -> unfold 2x2 -> [out_tokens, merged_dim]
    //
    // Must match PyTorch's F.unfold ordering (channel-major):
    //   For each channel c, for each kernel position (kh, kw):
    //     merged[c * merge^2 + kh * merge + kw] = hidden[row, col, c]
    let merge_sq = merge * merge;
    let mut merged = vec![0.0f32; out_tokens * merged_dim];
    for bh in 0..merged_h {
        for bw in 0..merged_w {
            let out_idx = bh * merged_w + bw;
            let out_off = out_idx * merged_dim;
            for c in 0..dim {
                for mh in 0..merge {
                    for mw in 0..merge {
                        let row = bh * merge + mh;
                        let col = bw * merge + mw;
                        let src_token = row * grid_w + col;
                        let src_val = normed[src_token * dim + c];
                        merged[out_off + c * merge_sq + mh * merge + mw] = src_val;
                    }
                }
            }
        }
    }

    // 3. merging_layer: [out_tokens, merged_dim] -> [out_tokens, dim]
    let mut proj_hidden = vec![0.0f32; out_tokens * proj.merging_layer.out_f];
    linear_no_bias_forward(&merged, &proj.merging_layer, &mut proj_hidden);

    // 4. linear_1 -> GELU -> linear_2
    let text_dim = proj.linear_1.out_f;
    let mut lin1_out = vec![0.0f32; out_tokens * text_dim];
    linear_no_bias_forward(&proj_hidden, &proj.linear_1, &mut lin1_out);
    gelu_inplace(&mut lin1_out);

    let mut lin2_out = vec![0.0f32; out_tokens * proj.linear_2.out_f];
    linear_no_bias_forward(&lin1_out, &proj.linear_2, &mut lin2_out);

    lin2_out
}

// --- Public API ---

/// Output from the Pixtral vision encoder.
pub struct PixtralOutput {
    /// Projected hidden states: [num_tokens, text_hidden_size].
    pub hidden_states: Vec<f32>,
    /// Number of output tokens (after spatial merge).
    pub num_tokens: usize,
}

impl PixtralEncoder {
    /// Run the Pixtral vision encoder forward pass.
    pub fn forward(
        &self,
        patches: &[f32],
        grid_h: usize,
        grid_w: usize,
    ) -> Result<PixtralOutput> {
        self.forward_inner(patches, grid_h, grid_w, None)
    }

    /// Run with a progress label.
    pub fn forward_with_label(
        &self,
        patches: &[f32],
        grid_h: usize,
        grid_w: usize,
        label: &str,
    ) -> Result<PixtralOutput> {
        self.forward_inner(patches, grid_h, grid_w, Some(label.to_string()))
    }

    fn forward_inner(
        &self,
        patches: &[f32],
        grid_h: usize,
        grid_w: usize,
        label: Option<String>,
    ) -> Result<PixtralOutput> {
        let dim = self.config.hidden_size;
        let num_tokens = grid_h * grid_w;
        let patch_dim = 3 * self.config.patch_size * self.config.patch_size;

        if patches.len() != num_tokens * patch_dim {
            return Err(HerbertError::Backend(format!(
                "patches length {} != num_tokens({}) * patch_dim({})",
                patches.len(), num_tokens, patch_dim
            )));
        }

        // 1. Patch embedding (Conv2D equivalent)
        debug!(num_tokens, dim, "Pixtral: patch embedding");
        let mut hidden_states = vec![0.0f32; num_tokens * dim];
        linear_no_bias_forward(patches, &self.patch_conv, &mut hidden_states);
        truncate_to_bf16_inplace(&mut hidden_states);

        // 2. ln_pre (RMSNorm)
        debug!("Pixtral: ln_pre");
        let mut normed = vec![0.0f32; num_tokens * dim];
        rms_norm(&hidden_states, &self.ln_pre, &mut normed);
        truncate_to_bf16_inplace(&mut normed);
        hidden_states.copy_from_slice(&normed);

        // 3. Build position IDs for 2D RoPE
        let mut position_ids = Vec::with_capacity(num_tokens);
        for h in 0..grid_h {
            for w in 0..grid_w {
                position_ids.push(h * self.max_patches_per_side + w);
            }
        }

        // 4. Forward through blocks
        let num_blocks = self.blocks.len();
        let pb = if let Some(ref lbl) = label {
            use indicatif::{ProgressBar, ProgressStyle};
            let pb = ProgressBar::new(num_blocks as u64);
            pb.set_style(
                ProgressStyle::with_template("  Encoding {msg} [{bar:30}] {pos}/{len} blocks")
                    .unwrap()
                    .progress_chars("##-"),
            );
            pb.set_message(lbl.clone());
            Some(pb)
        } else {
            None
        };

        debug!(num_blocks, num_tokens, "Pixtral: starting block forward");
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            pixtral_block_forward(
                block,
                &mut hidden_states,
                &self.rope_cos,
                &self.rope_sin,
                &position_ids,
                num_tokens,
                dim,
            );
            truncate_to_bf16_inplace(&mut hidden_states);
            if let Some(ref pb) = pb {
                pb.set_position((layer_idx + 1) as u64);
            }
        }
        if let Some(pb) = pb {
            pb.finish_and_clear();
        }
        debug!("Pixtral: blocks done");

        // 5. Projector
        debug!("Pixtral: projector");
        let projected = projector_forward(
            &self.projector,
            &hidden_states,
            num_tokens,
            dim,
            grid_h,
            grid_w,
        );
        let merge = self.config.spatial_merge_size;
        let out_tokens = (grid_h / merge) * (grid_w / merge);
        debug!(out_tokens, "Pixtral: done");

        Ok(PixtralOutput {
            hidden_states: projected,
            num_tokens: out_tokens,
        })
    }
}

// --- Image Preprocessing ---

/// Preprocess an RGB image for Pixtral.
///
/// Returns `(patches, grid_h, grid_w)` where patches is `[grid_h * grid_w, 3 * ps * ps]`.
pub fn pixtral_preprocess(
    rgb_bytes: &[u8],
    height: usize,
    width: usize,
    config: &PixtralVisionConfig,
) -> Result<(Vec<f32>, usize, usize)> {
    if rgb_bytes.len() != height * width * 3 {
        return Err(HerbertError::Config(format!(
            "expected {} RGB bytes ({}x{}x3), got {}",
            height * width * 3, height, width, rgb_bytes.len()
        )));
    }

    let ps = config.patch_size;
    let merge = config.spatial_merge_size;
    let align = ps * merge; // 28

    // Align dimensions to patch_size * spatial_merge_size
    let resized_h = ((height + align - 1) / align) * align;
    let resized_w = ((width + align - 1) / align) * align;

    // Clamp to image_size
    let resized_h = resized_h.min(config.image_size);
    let resized_w = resized_w.min(config.image_size);

    // Ensure alignment after clamping
    let resized_h = (resized_h / align) * align;
    let resized_w = (resized_w / align) * align;

    if resized_h == 0 || resized_w == 0 {
        return Err(HerbertError::Config("Image too small after alignment".into()));
    }

    // Resize
    let mut resized = vec![0u8; resized_h * resized_w * 3];
    bilinear_resize_rgb(rgb_bytes, height, width, &mut resized, resized_h, resized_w);

    // Convert to CHW float [0, 1] and normalize with CLIP constants
    let mean = [0.48145466f32, 0.4578275, 0.40821073];
    let std_dev = [0.26862954f32, 0.26130258, 0.27577711];
    let hw = resized_h * resized_w;
    let mut chw = vec![0.0f32; 3 * hw];
    for y in 0..resized_h {
        for x in 0..resized_w {
            let src = (y * resized_w + x) * 3;
            for c in 0..3 {
                chw[c * hw + y * resized_w + x] =
                    (resized[src + c] as f32 / 255.0 - mean[c]) / std_dev[c];
            }
        }
    }

    // Extract patches: for each ps x ps block in CHW -> flatten to [3 * ps * ps]
    let grid_h = resized_h / ps;
    let grid_w = resized_w / ps;
    let patch_dim = 3 * ps * ps;
    let num_patches = grid_h * grid_w;
    let mut patches = vec![0.0f32; num_patches * patch_dim];

    for ph in 0..grid_h {
        for pw in 0..grid_w {
            let patch_idx = ph * grid_w + pw;
            let out_off = patch_idx * patch_dim;
            let mut idx = 0;
            for c in 0..3usize {
                for pr in 0..ps {
                    for pc in 0..ps {
                        let r = ph * ps + pr;
                        let col = pw * ps + pc;
                        patches[out_off + idx] = chw[c * hw + r * resized_w + col];
                        idx += 1;
                    }
                }
            }
        }
    }

    Ok((patches, grid_h, grid_w))
}

// --- Weight Loading ---

#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

struct PixtralTensorStore {
    shard_data: Vec<Mmap>,
    weight_to_shard: HashMap<String, usize>,
}

impl PixtralTensorStore {
    fn load(model_dir: &Path, show_progress: bool) -> Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let index_data = fs::read(&index_path)?;
            let index: SafetensorsIndex = serde_json::from_slice(&index_data)?;

            // Find shards containing vision/projector weights
            let mut shard_set = std::collections::HashSet::new();
            for (weight, shard) in &index.weight_map {
                if weight.starts_with("vision_tower.") || weight.starts_with("multi_modal_projector.") {
                    shard_set.insert(shard.clone());
                }
            }

            let mut shard_map = HashMap::new();
            let mut shards = Vec::new();
            let mut shard_names: Vec<String> = shard_set.into_iter().collect();
            shard_names.sort();

            let pb = if show_progress {
                use indicatif::{ProgressBar, ProgressStyle};
                let pb = ProgressBar::new(shard_names.len() as u64);
                pb.set_style(
                    ProgressStyle::with_template("  Loading Pixtral shards [{bar:30}] {pos}/{len}")
                        .unwrap()
                        .progress_chars("##-"),
                );
                Some(pb)
            } else {
                None
            };

            info!(num_shards = shard_names.len(), "Loading Pixtral vision shards");
            for shard_name in &shard_names {
                let shard_path = model_dir.join(shard_name);
                let file_size = fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
                let size_mb = file_size as f64 / (1024.0 * 1024.0);
                info!(shard = %shard_name, size_mb = format!("{:.0}", size_mb), "Mapping Pixtral shard");
                let file = fs::File::open(&shard_path)?;
                let mmap = unsafe { Mmap::map(&file)? };
                let shard_id = shards.len();
                shards.push(mmap);
                shard_map.insert(shard_name.clone(), shard_id);
                if let Some(ref pb) = pb {
                    pb.inc(1);
                }
            }
            if let Some(pb) = pb {
                pb.finish_and_clear();
            }

            let mut weight_to_shard = HashMap::new();
            for (weight, shard_file) in &index.weight_map {
                if let Some(&shard_id) = shard_map.get(shard_file.as_str()) {
                    weight_to_shard.insert(weight.clone(), shard_id);
                }
            }

            Ok(Self { shard_data: shards, weight_to_shard })
        } else {
            let file = fs::File::open(model_dir.join("model.safetensors"))?;
            let mmap = unsafe { Mmap::map(&file)? };
            Ok(Self {
                shard_data: vec![mmap],
                weight_to_shard: HashMap::new(),
            })
        }
    }

    fn parse_shards(&self) -> Result<Vec<SafeTensors<'_>>> {
        let mut parsed = Vec::with_capacity(self.shard_data.len());
        for data in &self.shard_data {
            let tensors = SafeTensors::deserialize(&data[..])
                .map_err(|e| HerbertError::ModelLoad(e.to_string()))?;
            parsed.push(tensors);
        }
        Ok(parsed)
    }

    fn load_f32_tensor(&self, parsed: &[SafeTensors<'_>], name: &str) -> Result<Vec<f32>> {
        let shard_idx = if self.shard_data.len() == 1 {
            0
        } else {
            *self.weight_to_shard.get(name).ok_or_else(|| {
                HerbertError::ModelLoad(format!("Pixtral weight not found: {}", name))
            })?
        };

        let tensors = parsed.get(shard_idx).ok_or_else(|| {
            HerbertError::ModelLoad(format!("Invalid shard index for {}", name))
        })?;
        let view = tensors.tensor(name)
            .map_err(|e| HerbertError::ModelLoad(format!("Tensor {} not found: {}", name, e)))?;

        let dtype = view.dtype();
        let data = view.data();

        match dtype {
            safetensors::Dtype::F32 => {
                Ok(data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect())
            }
            safetensors::Dtype::BF16 => {
                Ok(data.chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect())
            }
            safetensors::Dtype::F16 => {
                Ok(data.chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        half_to_f32(bits)
                    })
                    .collect())
            }
            other => Err(HerbertError::ModelLoad(format!(
                "Unsupported dtype {:?} for Pixtral tensor {}", other, name
            ))),
        }
    }
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;

    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mantissa;
        let mut e = 0i32;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3ff;
        let f32_exp = ((127 - 15 + 1 + e) as u32) & 0xff;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }

    if exp == 31 {
        if mantissa == 0 {
            return f32::from_bits((sign << 31) | (0xff << 23));
        }
        return f32::from_bits((sign << 31) | (0xff << 23) | (mantissa << 13));
    }

    let f32_exp = (exp + 127 - 15) & 0xff;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}

// --- Load Helpers ---

fn load_rms_norm(
    store: &PixtralTensorStore,
    parsed: &[SafeTensors<'_>],
    name: &str,
    dim: usize,
) -> Result<RmsNorm> {
    let weight = store.load_f32_tensor(parsed, name)?;
    if weight.len() != dim {
        return Err(HerbertError::ModelLoad(format!(
            "RmsNorm {} expected {} elements, got {}", name, dim, weight.len()
        )));
    }
    Ok(RmsNorm { weight, dim, eps: 1e-5 })
}

fn load_linear_no_bias(
    store: &PixtralTensorStore,
    parsed: &[SafeTensors<'_>],
    name: &str,
    in_f: usize,
    out_f: usize,
) -> Result<LinearNoBias> {
    let weight = store.load_f32_tensor(parsed, name)?;
    if weight.len() != out_f * in_f {
        return Err(HerbertError::ModelLoad(format!(
            "{}: expected {}x{}={} elements, got {}",
            name, out_f, in_f, out_f * in_f, weight.len()
        )));
    }
    Ok(LinearNoBias { weight, in_f, out_f })
}

// --- Main Load Function ---

/// Load a Pixtral vision encoder from safetensors files.
pub fn load_pixtral_encoder(model_dir: &Path, config: &PixtralVisionConfig) -> Result<PixtralEncoder> {
    load_pixtral_encoder_inner(model_dir, config, false)
}

/// Load with progress indicators.
pub fn load_pixtral_encoder_with_progress(model_dir: &Path, config: &PixtralVisionConfig) -> Result<PixtralEncoder> {
    load_pixtral_encoder_inner(model_dir, config, true)
}

fn load_pixtral_encoder_inner(
    model_dir: &Path,
    config: &PixtralVisionConfig,
    show_progress: bool,
) -> Result<PixtralEncoder> {
    let store = PixtralTensorStore::load(model_dir, show_progress)?;
    let parsed = store.parse_shards()?;

    let dim = config.hidden_size;
    let ps = config.patch_size;
    let patch_dim = 3 * ps * ps;

    // Patch conv: Conv2D [dim, 3, ps, ps] -> reshaped to [dim, patch_dim]
    debug!("Loading Pixtral patch conv");
    let patch_conv = load_linear_no_bias(&store, &parsed, "vision_tower.patch_conv.weight", patch_dim, dim)?;

    // ln_pre
    debug!("Loading Pixtral ln_pre");
    let ln_pre = load_rms_norm(&store, &parsed, "vision_tower.ln_pre.weight", dim)?;

    // Transformer blocks
    let pb = if show_progress {
        use indicatif::{ProgressBar, ProgressStyle};
        let pb = ProgressBar::new(config.num_layers as u64);
        pb.set_style(
            ProgressStyle::with_template("  Loading Pixtral [{bar:30}] {pos}/{len} blocks")
                .unwrap()
                .progress_chars("##-"),
        );
        Some(pb)
    } else {
        None
    };

    info!(num_blocks = config.num_layers, "Loading Pixtral blocks");
    let mut blocks = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("vision_tower.transformer.layers.{}", i);
        let inter = config.intermediate_size;

        let attention_norm = load_rms_norm(
            &store, &parsed, &format!("{}.attention_norm.weight", prefix), dim,
        )?;
        let ffn_norm = load_rms_norm(
            &store, &parsed, &format!("{}.ffn_norm.weight", prefix), dim,
        )?;

        let attn = PixtralAttention {
            q_proj: load_linear_no_bias(&store, &parsed, &format!("{}.attention.q_proj.weight", prefix), dim, dim)?,
            k_proj: load_linear_no_bias(&store, &parsed, &format!("{}.attention.k_proj.weight", prefix), dim, dim)?,
            v_proj: load_linear_no_bias(&store, &parsed, &format!("{}.attention.v_proj.weight", prefix), dim, dim)?,
            o_proj: load_linear_no_bias(&store, &parsed, &format!("{}.attention.o_proj.weight", prefix), dim, dim)?,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            scaling: (config.head_dim as f32).powf(-0.5),
        };

        let ffn = PixtralMLP {
            gate_proj: load_linear_no_bias(&store, &parsed, &format!("{}.feed_forward.gate_proj.weight", prefix), dim, inter)?,
            up_proj: load_linear_no_bias(&store, &parsed, &format!("{}.feed_forward.up_proj.weight", prefix), dim, inter)?,
            down_proj: load_linear_no_bias(&store, &parsed, &format!("{}.feed_forward.down_proj.weight", prefix), inter, dim)?,
        };

        blocks.push(PixtralBlock { attention_norm, attn, ffn_norm, ffn });

        if let Some(ref pb) = pb {
            pb.set_position((i + 1) as u64);
        }
    }
    if let Some(pb) = pb {
        pb.finish_and_clear();
    }
    info!("Pixtral blocks loaded");

    // Multi-modal projector
    debug!("Loading multi-modal projector");
    let merge = config.spatial_merge_size;
    let merged_dim = dim * merge * merge;
    let text_dim = config.text_hidden_size;

    let projector = MultiModalProjector {
        norm: load_rms_norm(&store, &parsed, "multi_modal_projector.norm.weight", dim)?,
        merging_layer: load_linear_no_bias(
            &store, &parsed, "multi_modal_projector.patch_merger.merging_layer.weight",
            merged_dim, dim,
        )?,
        linear_1: load_linear_no_bias(
            &store, &parsed, "multi_modal_projector.linear_1.weight",
            dim, text_dim,
        )?,
        linear_2: load_linear_no_bias(
            &store, &parsed, "multi_modal_projector.linear_2.weight",
            text_dim, text_dim,
        )?,
        spatial_merge_size: merge,
    };
    info!("Pixtral multi-modal projector loaded");

    // Pre-compute 2D RoPE
    let max_patches = config.image_size / config.patch_size;
    debug!(max_patches, head_dim = config.head_dim, theta = config.rope_theta, "Pre-computing 2D RoPE");
    let (rope_cos, rope_sin) = precompute_2d_rope(max_patches, config.head_dim, config.rope_theta);

    info!("Pixtral encoder ready");

    Ok(PixtralEncoder {
        config: config.clone(),
        patch_conv,
        ln_pre,
        blocks,
        projector,
        rope_cos,
        rope_sin,
        max_patches_per_side: max_patches,
    })
}
