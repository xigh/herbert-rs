//! Vision encoder: Conv3D patch embed, ViT blocks, spatial merge, DeepStack.
//!
//! All computations are in F32 — no quantization for the vision encoder.

use herbert_core::error::{HerbertError, Result};
use tracing::{debug, trace};

use crate::config::VisionConfig;

// ─── Weight Structures ───────────────────────────────────────────────

/// Linear layer with bias (used throughout the vision encoder).
pub struct Linear {
    pub weight: Vec<f32>, // [out_features, in_features] row-major
    pub bias: Vec<f32>,   // [out_features]
    pub in_features: usize,
    pub out_features: usize,
}

/// LayerNorm parameters.
pub struct LayerNorm {
    pub weight: Vec<f32>, // [dim]
    pub bias: Vec<f32>,   // [dim]
    pub dim: usize,
    pub eps: f32,
}

/// Patch merger (2x2 pooling + FC→GELU→FC).
pub struct PatchMerger {
    pub norm: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
    /// True for DeepStack mergers (norm applied after shuffle).
    pub use_postshuffle_norm: bool,
}

/// Vision attention block.
pub struct VisionAttention {
    /// Fused QKV projection: [dim, dim*3].
    pub qkv: Linear,
    /// Output projection: [dim, dim].
    pub proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scaling: f32,
}

/// Vision MLP block (GELU activation).
pub struct VisionMLP {
    pub fc1: Linear,
    pub fc2: Linear,
}

/// Vision transformer block.
pub struct VisionBlock {
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub attn: VisionAttention,
    pub mlp: VisionMLP,
}

// ─── Scratch Buffers (Phase 2: eliminate hot-path allocations) ────────

/// Pre-allocated buffers reused across all 24 ViT blocks.
/// Eliminates ~216 heap allocations per forward pass.
struct VisionScratchBuffers {
    normed: Vec<f32>,             // [num_tokens * dim]           — norm1/norm2
    attn_out: Vec<f32>,           // [num_tokens * dim]           — attention output
    mlp_hidden: Vec<f32>,         // [num_tokens * intermediate]  — fc1+gelu
    mlp_out: Vec<f32>,            // [num_tokens * dim]           — fc2
    qkv: Vec<f32>,                // [num_tokens * 3 * dim]       — fused QKV
    q: Vec<f32>,                  // [num_tokens * dim]
    k: Vec<f32>,                  // [num_tokens * dim]
    v: Vec<f32>,                  // [num_tokens * dim]
    attn_output: Vec<f32>,        // [num_tokens * dim]           — pre-proj output
    thread_scores: Vec<Vec<f32>>, // [num_threads][max_seq_len²]  — per-thread scores
}

impl VisionScratchBuffers {
    fn new() -> Self {
        Self {
            normed: Vec::new(),
            attn_out: Vec::new(),
            mlp_hidden: Vec::new(),
            mlp_out: Vec::new(),
            qkv: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            attn_output: Vec::new(),
            thread_scores: Vec::new(),
        }
    }

    fn ensure_capacity(
        &mut self,
        num_tokens: usize,
        dim: usize,
        intermediate_size: usize,
        max_seq_len: usize,
        num_threads: usize,
    ) {
        let td = num_tokens * dim;
        self.normed.resize(td, 0.0);
        self.attn_out.resize(td, 0.0);
        self.mlp_hidden.resize(num_tokens * intermediate_size, 0.0);
        self.mlp_out.resize(td, 0.0);
        self.qkv.resize(num_tokens * 3 * dim, 0.0);
        self.q.resize(td, 0.0);
        self.k.resize(td, 0.0);
        self.v.resize(td, 0.0);
        self.attn_output.resize(td, 0.0);
        let scores_size = max_seq_len * max_seq_len;
        self.thread_scores.resize_with(num_threads, Vec::new);
        for ts in &mut self.thread_scores {
            ts.resize(scores_size, 0.0);
        }
    }
}

/// Complete vision encoder.
pub struct VisionEncoder {
    pub config: VisionConfig,
    /// Patch embedding: Conv3D with kernel=stride, equivalent to Linear.
    pub patch_embed: Linear,
    /// Positional embedding: [num_position_embeddings, hidden_size].
    pub pos_embed: Vec<f32>,
    /// 2D rotary embedding inverse frequencies: [head_dim/4].
    pub rot_inv_freq: Vec<f32>,
    /// Transformer blocks.
    pub blocks: Vec<VisionBlock>,
    /// Final merger (applied after all blocks).
    pub merger: PatchMerger,
    /// DeepStack mergers (one per deepstack_visual_indexes entry).
    pub deepstack_mergers: Vec<PatchMerger>,
}

// ─── Thread helpers ──────────────────────────────────────────────────

fn num_threads() -> usize {
    use std::sync::OnceLock;
    static NT: OnceLock<usize> = OnceLock::new();
    *NT.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    })
}

/// Wrapper to send a raw mutable pointer across threads.
/// Safety: caller must ensure non-overlapping writes per thread.
#[derive(Clone, Copy)]
struct SendMutPtr(*mut f32);
unsafe impl Send for SendMutPtr {}
unsafe impl Sync for SendMutPtr {}

impl SendMutPtr {
    /// Returns a mutable slice starting at offset with given length.
    /// Safety: caller must ensure non-overlapping access and valid bounds.
    #[inline]
    unsafe fn as_mut_slice<'a>(self, offset: usize, len: usize) -> &'a mut [f32] {
        std::slice::from_raw_parts_mut(self.0.add(offset), len)
    }
}

// ─── SIMD F32 Kernels ───────────────────────────────────────────────

/// Check (once) whether SIMD is enabled. Set VISION_NO_SIMD=1 to force scalar.
fn use_simd() -> bool {
    use std::sync::OnceLock;
    static USE_SIMD: OnceLock<bool> = OnceLock::new();
    *USE_SIMD.get_or_init(|| {
        let enabled = std::env::var("VISION_NO_SIMD").is_err();
        if !enabled {
            tracing::warn!("vision SIMD disabled by VISION_NO_SIMD");
        }
        enabled
    })
}

/// Scalar f32 dot product.
#[inline(always)]
fn scalar_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Scalar f32 scale-accumulate.
#[inline(always)]
fn scalar_sv_accum_f32(out: &mut [f32], v: &[f32], scale: f32) {
    for i in 0..out.len() {
        out[i] += scale * v[i];
    }
}

/// SIMD-accelerated f32 dot product.
#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    if !use_simd() {
        return scalar_dot_f32(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { avx512_vision_dot_f32(a.as_ptr(), b.as_ptr(), a.len() as u64) };
        }
    }

    #[allow(unreachable_code)]
    scalar_dot_f32(a, b)
}

/// SIMD-accelerated f32 scale-accumulate: out[i] += scale * v[i].
#[inline(always)]
fn sv_accum_f32(out: &mut [f32], v: &[f32], scale: f32) {
    debug_assert_eq!(out.len(), v.len());

    if !use_simd() {
        scalar_sv_accum_f32(out, v, scale);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { avx512_vision_sv_accum_f32(out.as_mut_ptr(), v.as_ptr(), scale, out.len() as u64) };
            return;
        }
    }

    #[allow(unreachable_code)]
    scalar_sv_accum_f32(out, v, scale);
}

// ── SIMD implementations (extern .S) ────────────────────────────────

#[cfg(target_arch = "x86_64")]
extern "C" {
    fn avx512_vision_dot_f32(a: *const f32, b: *const f32, n: u64) -> f32;
    fn avx512_vision_sv_accum_f32(out: *mut f32, v: *const f32, scale: f32, n: u64);
    fn avx512_vision_layer_norm_row(
        x: *const f32, weight: *const f32, bias: *const f32,
        eps: f32, out: *mut f32, dim: u64,
    );
    fn avx512_vision_gelu_f32(data: *mut f32, n: u64);
}

// ─── SIMD LayerNorm ──────────────────────────────────────────────────

/// Scalar LayerNorm for a single row.
#[inline]
pub fn scalar_layer_norm_row(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, out: &mut [f32]) {
    let dim = x.len();
    let mean: f32 = x.iter().sum::<f32>() / dim as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    for j in 0..dim {
        out[j] = (x[j] - mean) * inv_std * weight[j] + bias[j];
    }
}

/// SIMD-dispatched LayerNorm for a single row.
#[inline]
pub fn layer_norm_row(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, out: &mut [f32]) {
    if !use_simd() {
        scalar_layer_norm_row(x, weight, bias, eps, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        let dim = x.len();
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                avx512_vision_layer_norm_row(
                    x.as_ptr(), weight.as_ptr(), bias.as_ptr(),
                    eps, out.as_mut_ptr(), dim as u64,
                );
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    scalar_layer_norm_row(x, weight, bias, eps, out);
}

// ─── SIMD GELU ──────────────────────────────────────────────────────

/// Scalar GELU with tanh approximation (reference implementation).
#[inline]
pub fn scalar_gelu_chunk(data: &mut [f32]) {
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    for v in data.iter_mut() {
        let x = *v;
        *v = 0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh());
    }
}

/// Padé [3,3] rational approximation of tanh(z).
/// Error < 5e-8 for |z| < 5.0. Clamped to ±4.97 for saturation.
#[inline(always)]
pub fn tanh_pade(z: f32) -> f32 {
    let z = z.clamp(-4.97, 4.97);
    let z2 = z * z;
    let num = z * (135135.0 + z2 * (17325.0 + z2 * (378.0 + z2)));
    let den = 135135.0 + z2 * (62370.0 + z2 * (3150.0 + z2 * 28.0));
    num / den
}

/// SIMD-dispatched GELU chunk (in-place).
pub fn gelu_chunk(data: &mut [f32]) {
    if !use_simd() {
        scalar_gelu_chunk(data);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { avx512_vision_gelu_f32(data.as_mut_ptr(), data.len() as u64) };
            return;
        }
    }

    #[allow(unreachable_code)]
    scalar_gelu_chunk(data);
}

// ─── Kernels (multithreaded) ─────────────────────────────────────────

fn layer_norm(input: &[f32], norm: &LayerNorm, output: &mut [f32]) {
    let dim = norm.dim;
    debug_assert_eq!(input.len() % dim, 0, "layer_norm: input length not a multiple of dim");
    let n = input.len() / dim;
    let nt = num_threads().min(n);

    if nt <= 1 {
        layer_norm_chunk(input, norm, output);
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
            s.spawn(move || layer_norm_chunk(inp_c, norm, out_c));
        }
    });
}

fn layer_norm_chunk(input: &[f32], norm: &LayerNorm, output: &mut [f32]) {
    let dim = norm.dim;
    let n = input.len() / dim;
    for i in 0..n {
        let x = &input[i * dim..(i + 1) * dim];
        let out = &mut output[i * dim..(i + 1) * dim];
        layer_norm_row(x, &norm.weight, &norm.bias, norm.eps, out);
    }
}

fn linear_forward(input: &[f32], linear: &Linear, output: &mut [f32]) {
    debug_assert_eq!(input.len() % linear.in_features, 0, "linear_forward: input length not a multiple of in_features");
    let m = input.len() / linear.in_features;
    let nt = num_threads().min(m);

    if nt <= 1 {
        linear_forward_chunk(input, linear, output);
        return;
    }

    let in_f = linear.in_features;
    let out_f = linear.out_features;
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
            s.spawn(move || linear_forward_chunk(inp_c, linear, out_c));
        }
    });
}

fn linear_forward_chunk(input: &[f32], linear: &Linear, output: &mut [f32]) {
    let in_f = linear.in_features;
    let out_f = linear.out_features;
    let m = input.len() / in_f;

    #[cfg(target_arch = "x86_64")]
    if use_simd() && m >= 4 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { linear_forward_chunk_tiled_avx2(input, linear, output, m, in_f, out_f) };
        return;
    }

    for i in 0..m {
        let x = &input[i * in_f..(i + 1) * in_f];
        let out = &mut output[i * out_f..(i + 1) * out_f];
        for (j, out_j) in out.iter_mut().enumerate() {
            let w = &linear.weight[j * in_f..(j + 1) * in_f];
            *out_j = linear.bias[j] + dot_f32(x, w);
        }
    }
}

/// Tiled F32 matmul: compute TILE_M=4 dot products per weight column load.
/// Reduces L3 cache traffic by ~4× vs per-row dot product approach.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn linear_forward_chunk_tiled_avx2(
    input: &[f32],
    linear: &Linear,
    output: &mut [f32],
    m: usize,
    in_f: usize,
    out_f: usize,
) {
    use std::arch::x86_64::*;

    const TILE_M: usize = 4;
    let m_full = m / TILE_M * TILE_M;
    let k_full = in_f / 8 * 8;

    // Tiled path: process TILE_M=4 rows at once per output column
    for i_base in (0..m_full).step_by(TILE_M) {
        for j in 0..out_f {
            let w_ptr = linear.weight.as_ptr().add(j * in_f);

            // Initialize TILE_M accumulators
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            // K loop: load weight once, FMA against TILE_M input rows
            let a0 = input.as_ptr().add(i_base * in_f);
            let a1 = input.as_ptr().add((i_base + 1) * in_f);
            let a2 = input.as_ptr().add((i_base + 2) * in_f);
            let a3 = input.as_ptr().add((i_base + 3) * in_f);

            let mut k = 0usize;
            while k < k_full {
                let wv = _mm256_loadu_ps(w_ptr.add(k));
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0.add(k)), wv, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1.add(k)), wv, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2.add(k)), wv, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3.add(k)), wv, acc3);
                k += 8;
            }

            // Horizontal reduce each accumulator
            let bias = linear.bias[j];
            *output.as_mut_ptr().add(i_base * out_f + j) = hsum_avx2(acc0) + bias;
            *output.as_mut_ptr().add((i_base + 1) * out_f + j) = hsum_avx2(acc1) + bias;
            *output.as_mut_ptr().add((i_base + 2) * out_f + j) = hsum_avx2(acc2) + bias;
            *output.as_mut_ptr().add((i_base + 3) * out_f + j) = hsum_avx2(acc3) + bias;

            // K tail (if in_f not divisible by 8)
            if k_full < in_f {
                for ti in 0..TILE_M {
                    let row = i_base + ti;
                    let mut sum = 0.0f32;
                    for kk in k_full..in_f {
                        sum += *input.as_ptr().add(row * in_f + kk) * *w_ptr.add(kk);
                    }
                    *output.as_mut_ptr().add(row * out_f + j) += sum;
                }
            }
        }
    }

    // Remainder rows (m % TILE_M)
    for i in m_full..m {
        let x = &input[i * in_f..(i + 1) * in_f];
        let out_row = &mut output[i * out_f..(i + 1) * out_f];
        for (j, out_j) in out_row.iter_mut().enumerate() {
            let w = &linear.weight[j * in_f..(j + 1) * in_f];
            *out_j = linear.bias[j] + dot_f32(x, w);
        }
    }
}

/// Horizontal sum of 8 f32 lanes in a YMM register.
/// Must be called from a function with target_feature "avx2" enabled.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // [a0+a4, a1+a5, a2+a6, a3+a7] (128-bit)
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    // [s0+s2, s1+s3, ...]
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    // [s01+s23, ...]
    let shuf2 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
}

fn gelu_inplace(data: &mut [f32]) {
    let nt = num_threads().min(data.len() / 256);
    if nt <= 1 {
        gelu_chunk(data);
        return;
    }
    let chunk_size = data.len().div_ceil(nt);
    std::thread::scope(|s| {
        for c in data.chunks_mut(chunk_size) {
            s.spawn(move || {
                gelu_chunk(c);
            });
        }
    });
}

// ─── 2D Rotary Position Embedding ────────────────────────────────────

/// Compute inverse frequencies for vision 2D RoPE.
///
/// `dim` is `head_dim / 2` (half the head dimension, since 2D RoPE uses half for each axis).
/// Default theta = 10000.0 for vision encoder.
pub fn compute_vision_rot_inv_freq(dim: usize, theta: f32) -> Vec<f32> {
    let half = dim / 2;
    let mut inv_freq = vec![0.0f32; half];
    for (i, freq) in inv_freq.iter_mut().enumerate() {
        *freq = 1.0 / theta.powf(2.0 * i as f32 / dim as f32);
    }
    inv_freq
}

/// Compute 2D rotary position embedding cos/sin for vision attention.
///
/// For each token, we have 2D coordinates (row, col).
/// The cos/sin are computed as: `freq_table[row_coord]` concat `freq_table[col_coord]`,
/// then duplicated for rotate_half style: `(cos, cos)` and `(sin, sin)`.
///
/// `pos_ids`: `[num_tokens, 2]` — (row, col) for each token.
/// `inv_freq`: `[half_head_dim / 2]`
///
/// Returns `(cos, sin)` each of shape `[num_tokens, head_dim]`.
fn compute_vision_rope_cos_sin(
    pos_ids: &[(u32, u32)],
    inv_freq: &[f32],
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let num_tokens = pos_ids.len();
    let freq_dim = inv_freq.len(); // head_dim / 4
    let half_dim = head_dim / 2; // = freq_dim * 2

    // For each token: embeddings = [freq_table[row], freq_table[col]]
    // Then emb = concat(embeddings, embeddings) → head_dim
    let mut cos = vec![0.0f32; num_tokens * head_dim];
    let mut sin = vec![0.0f32; num_tokens * head_dim];

    for (t, &(row, col)) in pos_ids.iter().enumerate() {
        let offset = t * head_dim;
        // First half_dim: [row_freqs, col_freqs]
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
        // Second half_dim: duplicate
        for j in 0..half_dim {
            cos[offset + half_dim + j] = cos[offset + j];
            sin[offset + half_dim + j] = sin[offset + j];
        }
    }

    (cos, sin)
}

/// Build 2D position IDs for vision tokens with spatial merge ordering.
///
/// For each image grid (grid_t, grid_h, grid_w), produces position IDs
/// in the merge-aware order matching HF's `rot_pos_emb`.
fn build_vision_pos_ids(
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    merge_size: usize,
) -> Vec<(u32, u32)> {
    let merged_h = grid_h / merge_size;
    let merged_w = grid_w / merge_size;
    let num_tokens = grid_t * grid_h * grid_w;
    let mut pos_ids = Vec::with_capacity(num_tokens);

    for _frame in 0..grid_t {
        for bh in 0..merged_h {
            for bw in 0..merged_w {
                for mh in 0..merge_size {
                    for mw in 0..merge_size {
                        let row = bh * merge_size + mh;
                        let col = bw * merge_size + mw;
                        pos_ids.push((row as u32, col as u32));
                    }
                }
            }
        }
    }

    pos_ids
}

/// Apply rotate_half + RoPE to Q and K tensors.
///
/// `q`, `k`: `[num_tokens, dim]` (dim = num_heads * head_dim for q/k).
/// `cos`, `sin`: `[num_tokens, head_dim]`.
fn apply_rotary_emb(
    q: &mut [f32],
    k: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
) {
    let half = head_dim / 2;
    for t in 0..num_tokens {
        let cos_t = &cos[t * head_dim..(t + 1) * head_dim];
        let sin_t = &sin[t * head_dim..(t + 1) * head_dim];

        for h in 0..num_heads {
            // Apply to Q
            let q_offset = t * num_heads * head_dim + h * head_dim;
            rotate_half_inplace(&mut q[q_offset..q_offset + head_dim], cos_t, sin_t, half);
            // Apply to K
            let k_offset = t * num_heads * head_dim + h * head_dim;
            rotate_half_inplace(&mut k[k_offset..k_offset + head_dim], cos_t, sin_t, half);
        }
    }
}

/// In-place rotate_half + apply cos/sin.
/// `x` has length `head_dim`, cos/sin have length `head_dim`.
/// rotate_half: [-x2, x1] where x1 = x[..half], x2 = x[half..]
/// result: x * cos + rotate_half(x) * sin
fn rotate_half_inplace(x: &mut [f32], cos: &[f32], sin: &[f32], half: usize) {
    // Temporarily store rotated version
    let head_dim = x.len();
    debug_assert_eq!(half * 2, head_dim);

    // rotate_half: first half = -x[half..], second half = x[..half]
    // result[j] = x[j] * cos[j] + rotate_half(x)[j] * sin[j]
    for j in 0..half {
        let x1 = x[j];
        let x2 = x[j + half];
        x[j] = x1 * cos[j] + (-x2) * sin[j];
        x[j + half] = x2 * cos[j + half] + x1 * sin[j + half];
    }
}

// ─── Attention ───────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn attention_forward(
    attn: &VisionAttention,
    hidden_states: &[f32],
    cu_seqlens: &[usize],
    cos: &[f32],
    sin: &[f32],
    num_tokens: usize,
    output: &mut [f32],
    qkv: &mut [f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    attn_output: &mut [f32],
    thread_scores: &mut [Vec<f32>],
) {
    let dim = attn.num_heads * attn.head_dim;

    // QKV projection: [num_tokens, dim] → [num_tokens, 3*dim]
    linear_forward(hidden_states, &attn.qkv, qkv);

    // Split into Q, K, V
    for t in 0..num_tokens {
        let src = &qkv[t * 3 * dim..];
        for h in 0..attn.num_heads {
            let dst = t * dim + h * attn.head_dim;
            q[dst..dst + attn.head_dim]
                .copy_from_slice(&src[h * attn.head_dim..h * attn.head_dim + attn.head_dim]);
            k[dst..dst + attn.head_dim]
                .copy_from_slice(&src[dim + h * attn.head_dim..dim + h * attn.head_dim + attn.head_dim]);
            v[dst..dst + attn.head_dim]
                .copy_from_slice(&src[2 * dim + h * attn.head_dim..2 * dim + h * attn.head_dim + attn.head_dim]);
        }
    }

    // Apply rotary embeddings to Q and K
    apply_rotary_emb(q, k, cos, sin, num_tokens, attn.num_heads, attn.head_dim);

    // Attention per sub-sequence (cu_seqlens defines boundaries)
    let num_seqs = cu_seqlens.len() - 1;

    // Build flat list of (seq_start, seq_len, head) work items
    let mut work: Vec<(usize, usize, usize)> = Vec::new();
    for seq_idx in 0..num_seqs {
        let start = cu_seqlens[seq_idx];
        let seq_len = cu_seqlens[seq_idx + 1] - start;
        for h in 0..attn.num_heads {
            work.push((start, seq_len, h));
        }
    }

    let nt = num_threads().min(work.len());
    let head_dim = attn.head_dim;
    let scaling = attn.scaling;

    // Safety: each (token, head) pair writes to a unique non-overlapping offset.
    let attn_out_ptr = SendMutPtr(attn_output.as_mut_ptr());

    // Distribute pre-allocated per-thread score buffers
    let chunk = work.len().div_ceil(nt);
    let work_chunks: Vec<&[(usize, usize, usize)]> = work.chunks(chunk).collect();
    let score_chunks: Vec<&mut Vec<f32>> = thread_scores.iter_mut().take(work_chunks.len()).collect();

    std::thread::scope(|s| {
        for (items, scores_buf) in work_chunks.into_iter().zip(score_chunks) {
            let q_ref = q as &[f32];
            let k_ref = k as &[f32];
            let v_ref = v as &[f32];
            s.spawn(move || {
                for &(start, seq_len, h) in items {
                    let scores = &mut scores_buf[..seq_len * seq_len];
                    for qi in 0..seq_len {
                        let q_off = (start + qi) * dim + h * head_dim;
                        for ki in 0..seq_len {
                            let k_off = (start + ki) * dim + h * head_dim;
                            scores[qi * seq_len + ki] = dot_f32(
                                &q_ref[q_off..q_off + head_dim],
                                &k_ref[k_off..k_off + head_dim],
                            ) * scaling;
                        }
                    }
                    // Softmax per row
                    for qi in 0..seq_len {
                        let row = &mut scores[qi * seq_len..(qi + 1) * seq_len];
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
                    // Attention output: scores @ V (row-oriented for SIMD)
                    for qi in 0..seq_len {
                        let out_off = (start + qi) * dim + h * head_dim;
                        let out_slice = unsafe { attn_out_ptr.as_mut_slice(out_off, head_dim) };
                        out_slice.fill(0.0);
                        for vi in 0..seq_len {
                            let score = scores[qi * seq_len + vi];
                            let v_off = (start + vi) * dim + h * head_dim;
                            sv_accum_f32(out_slice, &v_ref[v_off..v_off + head_dim], score);
                        }
                    }
                }
            });
        }
    });

    // Output projection
    linear_forward(attn_output, &attn.proj, output);
}

// ─── Block Forward ───────────────────────────────────────────────────

fn block_forward(
    block: &VisionBlock,
    hidden_states: &mut [f32],
    cu_seqlens: &[usize],
    cos: &[f32],
    sin: &[f32],
    num_tokens: usize,
    scratch: &mut VisionScratchBuffers,
) {
    // hidden_states += attn(norm1(hidden_states))
    layer_norm(hidden_states, &block.norm1, &mut scratch.normed);

    // Borrow split: pass individual fields to attention_forward
    attention_forward(
        &block.attn,
        &scratch.normed,
        cu_seqlens,
        cos,
        sin,
        num_tokens,
        &mut scratch.attn_out,
        &mut scratch.qkv,
        &mut scratch.q,
        &mut scratch.k,
        &mut scratch.v,
        &mut scratch.attn_output,
        &mut scratch.thread_scores,
    );

    for (hs, &attn) in hidden_states.iter_mut().zip(scratch.attn_out.iter()) {
        *hs += attn;
    }

    // hidden_states += mlp(norm2(hidden_states))
    layer_norm(hidden_states, &block.norm2, &mut scratch.normed);

    linear_forward(&scratch.normed, &block.mlp.fc1, &mut scratch.mlp_hidden);
    gelu_inplace(&mut scratch.mlp_hidden);

    linear_forward(&scratch.mlp_hidden, &block.mlp.fc2, &mut scratch.mlp_out);

    for (hs, &mlp) in hidden_states.iter_mut().zip(scratch.mlp_out.iter()) {
        *hs += mlp;
    }
}

// ─── Merger Forward ──────────────────────────────────────────────────

fn merger_forward(
    merger: &PatchMerger,
    hidden_states: &[f32],
    num_tokens: usize,
    in_dim: usize,
    merge_size: usize,
) -> Vec<f32> {
    let merged_dim = in_dim * merge_size * merge_size;
    let out_tokens = num_tokens / (merge_size * merge_size);

    // Apply norm then reshape (or reshape then norm for postshuffle)
    let normed;
    let to_merge;

    if merger.use_postshuffle_norm {
        // Postshuffle: first reshape to merged_dim, then norm
        let mut reshaped = vec![0.0f32; out_tokens * merged_dim];
        // The spatial_merge groups `merge_size^2` consecutive tokens
        for i in 0..out_tokens {
            for j in 0..(merge_size * merge_size) {
                let src_token = i * merge_size * merge_size + j;
                let dst_offset = i * merged_dim + j * in_dim;
                let src_offset = src_token * in_dim;
                reshaped[dst_offset..dst_offset + in_dim]
                    .copy_from_slice(&hidden_states[src_offset..src_offset + in_dim]);
            }
        }
        normed = vec![0.0f32; out_tokens * merged_dim];
        let mut normed_mut = normed;
        layer_norm(&reshaped, &merger.norm, &mut normed_mut);
        to_merge = normed_mut;
    } else {
        // Pre-norm: norm at in_dim, then reshape to merged_dim
        let mut normed_buf = vec![0.0f32; num_tokens * in_dim];
        layer_norm(hidden_states, &merger.norm, &mut normed_buf);

        let mut reshaped = vec![0.0f32; out_tokens * merged_dim];
        for i in 0..out_tokens {
            for j in 0..(merge_size * merge_size) {
                let src_token = i * merge_size * merge_size + j;
                let dst_offset = i * merged_dim + j * in_dim;
                let src_offset = src_token * in_dim;
                reshaped[dst_offset..dst_offset + in_dim]
                    .copy_from_slice(&normed_buf[src_offset..src_offset + in_dim]);
            }
        }
        to_merge = reshaped;
    }

    // FC1 → GELU → FC2
    let mut fc1_out = vec![0.0f32; out_tokens * merger.fc1.out_features];
    linear_forward(&to_merge, &merger.fc1, &mut fc1_out);
    gelu_inplace(&mut fc1_out);

    let mut fc2_out = vec![0.0f32; out_tokens * merger.fc2.out_features];
    linear_forward(&fc1_out, &merger.fc2, &mut fc2_out);

    fc2_out
}

// ─── Position Embedding Interpolation ────────────────────────────────

/// Bilinear interpolation of position embeddings for arbitrary grid sizes.
///
/// `pos_embed`: `[num_position_embeddings, hidden_size]` (e.g., [2304, 1024]).
/// `grid_h, grid_w`: spatial grid dimensions for the image.
/// Returns: `[grid_t * grid_h * grid_w, hidden_size]` position embeddings in merge order.
fn interpolate_pos_embed(
    pos_embed: &[f32],
    num_grid_per_side: usize,
    hidden_size: usize,
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    merge_size: usize,
) -> Vec<f32> {
    let num_tokens = grid_t * grid_h * grid_w;
    let mut result = vec![0.0f32; num_tokens * hidden_size];

    // Compute bilinear interpolation indices and weights for (grid_h, grid_w)
    let h_idxs: Vec<f64> = (0..grid_h)
        .map(|i| i as f64 * (num_grid_per_side - 1) as f64 / (grid_h.max(1) - 1).max(1) as f64)
        .collect();
    let w_idxs: Vec<f64> = (0..grid_w)
        .map(|i| i as f64 * (num_grid_per_side - 1) as f64 / (grid_w.max(1) - 1).max(1) as f64)
        .collect();

    // Interpolate for each (row, col) position
    let mut hw_embeds = vec![0.0f32; grid_h * grid_w * hidden_size];
    for (r, &h_idx) in h_idxs.iter().enumerate() {
        let h_floor = (h_idx.floor() as usize).min(num_grid_per_side - 1);
        let h_ceil = (h_floor + 1).min(num_grid_per_side - 1);
        let dh = h_idx - h_floor as f64;

        for (c, &w_idx) in w_idxs.iter().enumerate() {
            let w_floor = (w_idx.floor() as usize).min(num_grid_per_side - 1);
            let w_ceil = (w_floor + 1).min(num_grid_per_side - 1);
            let dw = w_idx - w_floor as f64;

            let dst = (r * grid_w + c) * hidden_size;

            // Four corners
            let i00 = (h_floor * num_grid_per_side + w_floor) * hidden_size;
            let i01 = (h_floor * num_grid_per_side + w_ceil) * hidden_size;
            let i10 = (h_ceil * num_grid_per_side + w_floor) * hidden_size;
            let i11 = (h_ceil * num_grid_per_side + w_ceil) * hidden_size;

            let w00 = ((1.0 - dh) * (1.0 - dw)) as f32;
            let w01 = ((1.0 - dh) * dw) as f32;
            let w10 = (dh * (1.0 - dw)) as f32;
            let w11 = (dh * dw) as f32;

            for d in 0..hidden_size {
                hw_embeds[dst + d] = pos_embed[i00 + d] * w00
                    + pos_embed[i01 + d] * w01
                    + pos_embed[i10 + d] * w10
                    + pos_embed[i11 + d] * w11;
            }
        }
    }

    // Tile across temporal frames and permute to merge order
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

// ─── Public API ──────────────────────────────────────────────────────

/// Output from the vision encoder.
pub struct VisionOutput {
    /// Final merged hidden states: `[num_merged_tokens, out_hidden_size]`.
    pub hidden_states: Vec<f32>,
    /// Number of merged tokens.
    pub num_tokens: usize,
    /// DeepStack features: one `Vec<f32>` per deepstack layer index.
    pub deepstack_features: Vec<Vec<f32>>,
}

impl VisionEncoder {
    /// Run the vision encoder forward pass.
    ///
    /// `patches`: `[num_patches, patch_dim]` — output of `patchify()`.
    /// `grid_t, grid_h, grid_w`: image grid dimensions.
    ///
    /// Returns merged hidden states and deepstack features.
    pub fn forward(
        &self,
        patches: &[f32],
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<VisionOutput> {
        self.forward_inner(patches, grid_t, grid_h, grid_w, None)
    }

    /// Run the vision encoder forward pass with a progress bar showing the label.
    pub fn forward_with_label(
        &self,
        patches: &[f32],
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
        label: &str,
    ) -> Result<VisionOutput> {
        self.forward_inner(patches, grid_t, grid_h, grid_w, Some(label.to_string()))
    }

    fn forward_inner(
        &self,
        patches: &[f32],
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
        label: Option<String>,
    ) -> Result<VisionOutput> {
        let dim = self.config.hidden_size;
        let num_tokens = grid_t * grid_h * grid_w;
        let merge_size = self.config.spatial_merge_size;

        if patches.len() != num_tokens * self.config.patch_dim {
            return Err(HerbertError::Backend(format!(
                "patches length {} != num_tokens({}) * patch_dim({})",
                patches.len(), num_tokens, self.config.patch_dim
            )));
        }

        // 1. Patch embedding (Conv3D equivalent = linear projection)
        debug!(num_tokens, dim, "Patch embedding");
        let profile_vision = std::env::var("PROFILE_VISION").is_ok();
        let t_start = std::time::Instant::now();
        let mut hidden_states = vec![0.0f32; num_tokens * dim];
        linear_forward(patches, &self.patch_embed, &mut hidden_states);
        if profile_vision { eprintln!("[VISION] Patch embed: {:.1}ms", t_start.elapsed().as_secs_f64() * 1000.0); }
        debug!("Patch embedding done");

        // 2. Position embedding (bilinear interpolation)
        debug!("Computing position embeddings");
        let pos_embeds = interpolate_pos_embed(
            &self.pos_embed,
            self.config.num_grid_per_side,
            dim,
            grid_t,
            grid_h,
            grid_w,
            merge_size,
        );

        for i in 0..hidden_states.len() {
            hidden_states[i] += pos_embeds[i];
        }

        // 3. Compute 2D rotary position embeddings
        debug!("Computing 2D RoPE");
        let pos_ids = build_vision_pos_ids(grid_t, grid_h, grid_w, merge_size);
        let (cos, sin) = compute_vision_rope_cos_sin(&pos_ids, &self.rot_inv_freq, self.config.head_dim);
        debug!("RoPE computed");

        // 4. Build cu_seqlens for attention (each frame's tokens form a sequence)
        let tokens_per_frame = grid_h * grid_w;
        let mut cu_seqlens = vec![0usize];
        let mut offset = 0;
        for _frame in 0..grid_t {
            offset += tokens_per_frame;
            cu_seqlens.push(offset);
        }

        // 5. Forward through blocks with DeepStack
        let mut deepstack_features = Vec::new();
        let num_blocks = self.blocks.len();

        let pb = if let Some(ref lbl) = label {
            use indicatif::{ProgressBar, ProgressStyle};
            let pb = ProgressBar::new(num_blocks as u64);
            pb.set_style(
                ProgressStyle::with_template("  Encoding {msg} [{bar:30}] {pos}/{len} blocks")
                    .expect("valid progress template")
                    .progress_chars("█░░"),
            );
            pb.set_message(lbl.clone());
            Some(pb)
        } else {
            None
        };

        // Pre-allocate scratch buffers for all blocks (Phase 2 optimization)
        let intermediate_size = self.blocks.first()
            .map(|b| b.mlp.fc1.out_features)
            .unwrap_or(0);
        let mut scratch = VisionScratchBuffers::new();
        scratch.ensure_capacity(num_tokens, dim, intermediate_size, tokens_per_frame, num_threads());

        if profile_vision { eprintln!("[VISION] Pos embed + RoPE: {:.1}ms", t_start.elapsed().as_secs_f64() * 1000.0); }
        let t_blocks = std::time::Instant::now();
        // Accumulators for per-component profiling across all blocks
        let mut prof_norm = 0.0f64;
        let mut prof_attn = 0.0f64;
        let mut prof_mlp = 0.0f64;
        debug!(num_blocks, num_tokens, "Starting vision block forward");
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            trace!(layer_idx, "Vision block forward");
            if profile_vision {
                // Profiled block: measure norm, attn, mlp separately
                let t0 = std::time::Instant::now();
                layer_norm(&hidden_states, &block.norm1, &mut scratch.normed);
                let t1 = std::time::Instant::now();
                attention_forward(
                    &block.attn, &scratch.normed, &cu_seqlens, &cos, &sin,
                    num_tokens, &mut scratch.attn_out, &mut scratch.qkv,
                    &mut scratch.q, &mut scratch.k, &mut scratch.v,
                    &mut scratch.attn_output, &mut scratch.thread_scores,
                );
                let t2 = std::time::Instant::now();
                for (hs, &a) in hidden_states.iter_mut().zip(scratch.attn_out.iter()) { *hs += a; }
                layer_norm(&hidden_states, &block.norm2, &mut scratch.normed);
                let t3 = std::time::Instant::now();
                linear_forward(&scratch.normed, &block.mlp.fc1, &mut scratch.mlp_hidden);
                gelu_inplace(&mut scratch.mlp_hidden);
                linear_forward(&scratch.mlp_hidden, &block.mlp.fc2, &mut scratch.mlp_out);
                for (hs, &m) in hidden_states.iter_mut().zip(scratch.mlp_out.iter()) { *hs += m; }
                let t4 = std::time::Instant::now();
                prof_norm += (t1 - t0).as_secs_f64() + (t3 - t2).as_secs_f64();
                prof_attn += (t2 - t1).as_secs_f64();
                prof_mlp += (t4 - t3).as_secs_f64();
            } else {
                block_forward(
                    block, &mut hidden_states, &cu_seqlens, &cos, &sin,
                    num_tokens, &mut scratch,
                );
            }

            // DeepStack: extract and merge at specified layers
            if let Some(ds_idx) = self
                .config
                .deepstack_visual_indexes
                .iter()
                .position(|&idx| idx == layer_idx)
            {
                trace!(layer_idx, ds_idx, "DeepStack merge");
                let ds_merged = merger_forward(
                    &self.deepstack_mergers[ds_idx],
                    &hidden_states,
                    num_tokens,
                    dim,
                    merge_size,
                );
                deepstack_features.push(ds_merged);
            }

            if let Some(ref pb) = pb {
                pb.set_position((layer_idx + 1) as u64);
            }
        }
        if profile_vision {
            let blocks_ms = t_blocks.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[VISION] Blocks ({}): {:.1}ms", num_blocks, blocks_ms);
            eprintln!("[VISION]   Norm:  {:.1}ms ({:.1}%)", prof_norm * 1000.0, prof_norm * 1000.0 / blocks_ms * 100.0);
            eprintln!("[VISION]   Attn:  {:.1}ms ({:.1}%)", prof_attn * 1000.0, prof_attn * 1000.0 / blocks_ms * 100.0);
            eprintln!("[VISION]   MLP:   {:.1}ms ({:.1}%)", prof_mlp * 1000.0, prof_mlp * 1000.0 / blocks_ms * 100.0);
        }
        debug!("Vision blocks done");

        if let Some(pb) = pb {
            pb.finish_and_clear();
        }

        // 6. Final merger
        let t_merger = std::time::Instant::now();
        debug!("Final merger");
        let merged = merger_forward(&self.merger, &hidden_states, num_tokens, dim, merge_size);
        if profile_vision { eprintln!("[VISION] Final merger: {:.1}ms", t_merger.elapsed().as_secs_f64() * 1000.0); }
        debug!("Final merger done");
        let num_merged = num_tokens / (merge_size * merge_size);

        Ok(VisionOutput {
            hidden_states: merged,
            num_tokens: num_merged,
            deepstack_features,
        })
    }
}
