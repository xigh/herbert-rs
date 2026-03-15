//! EAGLE-3 speculative decoding head.
//!
//! Loads a pre-trained EAGLE-3 head (e.g. `AngelSlim/Qwen3-a3B_eagle3`) and
//! runs draft token generation using the target model's hidden states.
//!
//! Architecture:
//!   - fc: fuses hidden states from 3 target layers [N/4, N/2, 3*N/4] → H
//!   - 1 decoder layer with 2H input (concat embed + fused hidden)
//!   - lm_head [draft_vocab, H] + d2t mapping → target token IDs
//!
//! All weights are BF16 regardless of the target model's quantization.

use herbert_core::config::Config;
use herbert_core::error::{HerbertError, Result};
use herbert_core::tensor::{bf16_to_f32, BF16};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::path::Path;

use crate::loader_common::load_bf16_from_view;
use crate::thread_pool::{global_pool, SendMutPtr, SendPtr};

/// EAGLE-3 speculative decoding head.
pub struct Eagle3Head {
    // ── Weights (all BF16 stored as Vec<u16>) ──
    fc_weight: Vec<BF16>,            // [H, 3*H]
    hidden_norm: Vec<BF16>,          // [H]
    input_layernorm: Vec<BF16>,      // [H]
    q_proj: Vec<BF16>,               // [q_dim, 2*H]
    k_proj: Vec<BF16>,               // [kv_dim, 2*H]
    v_proj: Vec<BF16>,               // [kv_dim, 2*H]
    o_proj: Vec<BF16>,               // [H, q_dim]
    post_attn_norm: Vec<BF16>,       // [H]
    mlp_gate: Vec<BF16>,             // [intermediate, H]
    mlp_up: Vec<BF16>,               // [intermediate, H]
    mlp_down: Vec<BF16>,             // [H, intermediate]
    final_norm: Vec<BF16>,           // [H]
    lm_head: Vec<BF16>,              // [draft_vocab, H]
    d2t: Vec<i64>,                   // [draft_vocab] draft→target token mapping
    t2d: Vec<i32>,                   // [target_vocab] target→draft (-1 if unmapped)

    // ── Config ──
    hidden_size: usize,              // H (2048 for 30B-A3B)
    num_heads: usize,                // 32
    num_kv_heads: usize,             // 4
    head_dim: usize,                 // 128
    intermediate_size: usize,        // 6144
    draft_vocab_size: usize,         // 32000
    target_vocab_size: usize,        // 151936
    rms_norm_eps: f32,               // 1e-6
    mrope_interleaved: bool,         // true for Qwen3-VL MRoPE
    pub extract_layers: [usize; 3],  // [N/4, N/2, 3*N/4]

    // ── KV cache (1 layer, f32) ──
    kv_keys: Vec<Vec<f32>>,          // [num_kv_heads][seq * head_dim]
    kv_values: Vec<Vec<f32>>,        // [num_kv_heads][seq * head_dim]
    kv_seq_len: usize,

    // ── RoPE cache ──
    cos_cache: Vec<f32>,             // [max_pos * half_dim]
    sin_cache: Vec<f32>,

    // ── Scratch buffers ──
    scratch_fc_out: Vec<f32>,        // [H]
    scratch_embed: Vec<f32>,         // [H]
    scratch_norm_e: Vec<f32>,        // [H]
    scratch_norm_h: Vec<f32>,        // [H]
    scratch_concat: Vec<f32>,        // [2*H]
    scratch_q: Vec<f32>,             // [q_dim]
    scratch_k: Vec<f32>,             // [kv_dim]
    scratch_v: Vec<f32>,             // [kv_dim]
    scratch_attn_out: Vec<f32>,      // [q_dim]
    scratch_o_out: Vec<f32>,         // [H]
    scratch_post_norm: Vec<f32>,     // [H]
    scratch_gate: Vec<f32>,          // [intermediate]
    scratch_up: Vec<f32>,            // [intermediate]
    scratch_mlp_out: Vec<f32>,       // [H]
    scratch_final_norm: Vec<f32>,    // [H]
    scratch_logits: Vec<f32>,        // [draft_vocab]
    scratch_scores: Vec<f32>,        // [max_seq] for attention
}

impl Eagle3Head {
    /// Load an EAGLE-3 head from a directory containing `model.safetensors`.
    pub fn load(path: &Path, target_config: &Config) -> Result<Self> {
        let safetensors_path = path.join("model.safetensors");
        if !safetensors_path.exists() {
            return Err(HerbertError::ModelLoad(format!(
                "EAGLE-3 head not found: {}",
                safetensors_path.display()
            )));
        }

        let file = std::fs::File::open(&safetensors_path)
            .map_err(|e| HerbertError::ModelLoad(format!("open {}: {}", safetensors_path.display(), e)))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| HerbertError::ModelLoad(format!("mmap {}: {}", safetensors_path.display(), e)))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|e| HerbertError::ModelLoad(format!("parse safetensors: {}", e)))?;

        let h = target_config.hidden_size;
        let num_layers = target_config.num_layers;

        // Load weights
        let fc_weight = load_tensor_bf16(&tensors, "fc.weight")?;
        let hidden_norm = load_tensor_bf16(&tensors, "midlayer.hidden_norm.weight")?;
        let input_layernorm = load_tensor_bf16(&tensors, "midlayer.input_layernorm.weight")?;
        let q_proj = load_tensor_bf16(&tensors, "midlayer.self_attn.q_proj.weight")?;
        let k_proj = load_tensor_bf16(&tensors, "midlayer.self_attn.k_proj.weight")?;
        let v_proj = load_tensor_bf16(&tensors, "midlayer.self_attn.v_proj.weight")?;
        let o_proj = load_tensor_bf16(&tensors, "midlayer.self_attn.o_proj.weight")?;
        let post_attn_norm = load_tensor_bf16(&tensors, "midlayer.post_attention_layernorm.weight")?;
        let mlp_gate = load_tensor_bf16(&tensors, "midlayer.mlp.gate_proj.weight")?;
        let mlp_up = load_tensor_bf16(&tensors, "midlayer.mlp.up_proj.weight")?;
        let mlp_down = load_tensor_bf16(&tensors, "midlayer.mlp.down_proj.weight")?;
        let final_norm = load_tensor_bf16(&tensors, "norm.weight")?;
        let lm_head = load_tensor_bf16(&tensors, "lm_head.weight")?;

        // Load t2d boolean mask: t2d[target_tok] = true if target_tok is in draft vocab
        let t2d_view = tensors.tensor("t2d")
            .map_err(|e| HerbertError::ModelLoad(format!("missing t2d: {}", e)))?;
        let t2d_bool: Vec<bool> = t2d_view.data().iter().map(|&b| b != 0).collect();
        let target_vocab_size = t2d_bool.len();

        // Derive d2t mapping from t2d boolean mask:
        // d2t[draft_idx] = the draft_idx-th True position in t2d
        // This gives us: draft token index → target token ID
        let d2t: Vec<i64> = t2d_bool.iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(target_idx, _)| target_idx as i64)
            .collect();
        let draft_vocab_size = d2t.len();

        // Build t2d reverse mapping: target_tok → draft_idx (-1 if unmapped)
        let mut t2d = vec![-1i32; target_vocab_size];
        for (draft_idx, &target_idx) in d2t.iter().enumerate() {
            t2d[target_idx as usize] = draft_idx as i32;
        }

        // Infer dimensions from weight shapes
        let q_dim = q_proj.len() / (2 * h); // q_proj is [q_dim, 2H]
        let kv_dim_2h = k_proj.len(); // k_proj is [kv_dim, 2H]
        let kv_dim = kv_dim_2h / (2 * h);
        let head_dim = target_config.head_dim;
        let num_heads = q_dim / head_dim;
        let num_kv_heads = kv_dim / head_dim;
        let intermediate_size = mlp_gate.len() / h;

        eprintln!("[eagle3] Loaded: H={}, heads={}/{}, inter={}, draft_vocab={}, layers=[{},{},{}]",
            h, num_heads, num_kv_heads, intermediate_size, draft_vocab_size,
            2, num_layers / 2, num_layers - 3);

        // Compute RoPE cache with EAGLE head's own theta (from config.json)
        // Parse config.json for rope_theta if available, else default 1e6
        let config_path = path.join("config.json");
        let config_str = if config_path.exists() {
            std::fs::read_to_string(&config_path).unwrap_or_default()
        } else {
            String::new()
        };

        let eagle_rope_theta = config_str
            .find("\"rope_theta\"")
            .and_then(|pos| {
                let after = &config_str[pos..];
                after.find(':').and_then(|colon| {
                    let val_str = after[colon + 1..].trim();
                    let end = val_str.find(|c: char| c == ',' || c == '}' || c == '\n').unwrap_or(val_str.len());
                    val_str[..end].trim().parse::<f64>().ok()
                })
            })
            .unwrap_or(1_000_000.0) as f32;

        let mrope_interleaved = config_str.contains("\"mrope_interleaved\": true")
            || config_str.contains("\"mrope_interleaved\":true");

        eprintln!("[eagle3] Using rope_theta={} mrope_interleaved={} (target rope_theta={})",
            eagle_rope_theta, mrope_interleaved, target_config.rope_theta);

        let max_pos = 8192; // eagle3 only needs short contexts
        let rotary_ndims = head_dim; // EAGLE uses full head_dim for RoPE
        let half_dim = rotary_ndims / 2;
        let theta = eagle_rope_theta;
        let mut inv_freq = vec![0.0f32; half_dim];
        for i in 0..half_dim {
            inv_freq[i] = 1.0 / theta.powf((2 * i) as f32 / rotary_ndims as f32);
        }
        let mut cos_cache = vec![0.0f32; max_pos * half_dim];
        let mut sin_cache = vec![0.0f32; max_pos * half_dim];
        for pos in 0..max_pos {
            for i in 0..half_dim {
                let freq = pos as f32 * inv_freq[i];
                cos_cache[pos * half_dim + i] = freq.cos();
                sin_cache[pos * half_dim + i] = freq.sin();
            }
        }

        // Reference: SafeAILab/EAGLE eagle3/traineagle3/modeling_llama_kv.py line 1138
        // Extract layers: [2, N//2, N-3] (NOT [N/4, N/2, 3*N/4])
        let extract_layers = [2, num_layers / 2, num_layers - 3];

        Ok(Self {
            fc_weight,
            hidden_norm,
            input_layernorm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            post_attn_norm,
            mlp_gate,
            mlp_up,
            mlp_down,
            final_norm,
            lm_head,
            d2t,
            t2d,
            hidden_size: h,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            draft_vocab_size,
            target_vocab_size,
            rms_norm_eps: 1e-6,
            mrope_interleaved,
            extract_layers,
            kv_keys: (0..num_kv_heads).map(|_| Vec::with_capacity(1024 * head_dim)).collect(),
            kv_values: (0..num_kv_heads).map(|_| Vec::with_capacity(1024 * head_dim)).collect(),
            kv_seq_len: 0,
            cos_cache,
            sin_cache,
            scratch_fc_out: vec![0.0; h],
            scratch_embed: vec![0.0; h],
            scratch_norm_e: vec![0.0; h],
            scratch_norm_h: vec![0.0; h],
            scratch_concat: vec![0.0; 2 * h],
            scratch_q: vec![0.0; q_dim],
            scratch_k: vec![0.0; kv_dim],
            scratch_v: vec![0.0; kv_dim],
            scratch_attn_out: vec![0.0; q_dim],
            scratch_o_out: vec![0.0; h],
            scratch_post_norm: vec![0.0; h],
            scratch_gate: vec![0.0; intermediate_size],
            scratch_up: vec![0.0; intermediate_size],
            scratch_mlp_out: vec![0.0; h],
            scratch_final_norm: vec![0.0; h],
            scratch_logits: vec![0.0; draft_vocab_size],
            scratch_scores: Vec::with_capacity(1024),
        })
    }

    /// Reset KV cache for a new speculation round.
    pub fn reset_kv(&mut self) {
        for v in &mut self.kv_keys {
            v.clear();
        }
        for v in &mut self.kv_values {
            v.clear();
        }
        self.kv_seq_len = 0;
    }

    /// Truncate KV to keep only first `len` entries.
    pub fn truncate_kv(&mut self, len: usize) {
        if len < self.kv_seq_len {
            let elems = len * self.head_dim;
            for v in &mut self.kv_keys {
                v.truncate(elems);
            }
            for v in &mut self.kv_values {
                v.truncate(elems);
            }
            self.kv_seq_len = len;
        }
    }

    /// Map a target vocab token to draft vocab. Returns None if unmapped.
    #[inline]
    pub fn target_to_draft(&self, target_id: u32) -> Option<u32> {
        let idx = target_id as usize;
        if idx < self.t2d.len() && self.t2d[idx] >= 0 {
            Some(self.t2d[idx] as u32)
        } else {
            None
        }
    }

    /// Map a draft vocab token to target vocab.
    #[inline]
    pub fn draft_to_target(&self, draft_id: u32) -> u32 {
        self.d2t[draft_id as usize] as u32
    }

    /// One draft step. Returns (target_token_id, draft_logits, decoder_output_hidden).
    ///
    /// - `token`: target vocab ID of the current token
    /// - `hidden_states`: from target model (only on first step of a round)
    /// - `prev_output`: decoder output from previous EAGLE step (steps 2..K)
    /// - `target_embed_tokens`: shared embedding table (BF16) from target model
    /// - `rope_offset`: absolute position in the sequence
    pub fn draft_step(
        &mut self,
        token: u32,
        hidden_states: Option<&[Vec<f32>; 3]>,
        prev_output: Option<&[f32]>,
        target_embed_tokens: &[BF16],
        rope_offset: usize,
    ) -> Result<(u32, Vec<f32>, Vec<f32>)> {
        let h = self.hidden_size;

        // Step 1: Get embedding from target's embed_tokens
        let embed = &mut self.scratch_embed;
        let token_idx = token as usize;
        let src_start = token_idx * h;
        for i in 0..h {
            embed[i] = bf16_to_f32(target_embed_tokens[src_start + i]);
        }

        // Step 2: Get fused hidden representation
        let fused = if let Some(hs) = hidden_states {
            let fc_out = &mut self.scratch_fc_out;
            matvec_bf16_3concat(
                &hs[0], &hs[1], &hs[2],
                &self.fc_weight, fc_out, h,
            );
            fc_out as &[f32]
        } else if let Some(prev) = prev_output {
            prev
        } else {
            return Err(HerbertError::Backend(
                "eagle3: either hidden_states or prev_output required".into(),
            ));
        };

        // Step 3: Decoder layer (2H input: embed + fused hidden)
        // 3a: RMS norm on embed (input_layernorm) and fused (hidden_norm) separately
        let norm_e = &mut self.scratch_norm_e;
        rms_norm_bf16(embed, &self.input_layernorm, norm_e, self.rms_norm_eps);
        let norm_h = &mut self.scratch_norm_h;
        rms_norm_bf16(fused, &self.hidden_norm, norm_h, self.rms_norm_eps);

        // 3b: Concat normalized [norm_e, norm_h] → [2H]
        let concat = &mut self.scratch_concat;
        concat[..h].copy_from_slice(norm_e);
        concat[h..2 * h].copy_from_slice(norm_h);

        // 3c: Q/K/V projections from 2H input
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let q = &mut self.scratch_q;
        let k = &mut self.scratch_k;
        let v = &mut self.scratch_v;
        matvec_bf16_raw(concat, &self.q_proj, q, q_dim, 2 * h);
        matvec_bf16_raw(concat, &self.k_proj, k, kv_dim, 2 * h);
        matvec_bf16_raw(concat, &self.v_proj, v, kv_dim, 2 * h);

        // 3d: RoPE on Q and K
        let half_dim = self.head_dim / 2;
        let cos_start = rope_offset * half_dim;
        let cos = &self.cos_cache[cos_start..cos_start + half_dim];
        let sin = &self.sin_cache[cos_start..cos_start + half_dim];

        if self.mrope_interleaved {
            for head in 0..self.num_heads {
                let off = head * self.head_dim;
                apply_rope_interleaved(&mut q[off..off + self.head_dim], cos, sin);
            }
            for head in 0..self.num_kv_heads {
                let off = head * self.head_dim;
                apply_rope_interleaved(&mut k[off..off + self.head_dim], cos, sin);
            }
        } else {
            for head in 0..self.num_heads {
                let off = head * self.head_dim;
                apply_rope(&mut q[off..off + self.head_dim], cos, sin);
            }
            for head in 0..self.num_kv_heads {
                let off = head * self.head_dim;
                apply_rope(&mut k[off..off + self.head_dim], cos, sin);
            }
        }

        // 3e: Append K/V to EAGLE KV cache
        for kv_h in 0..self.num_kv_heads {
            let off = kv_h * self.head_dim;
            self.kv_keys[kv_h].extend_from_slice(&k[off..off + self.head_dim]);
            self.kv_values[kv_h].extend_from_slice(&v[off..off + self.head_dim]);
        }
        self.kv_seq_len += 1;
        let seq_len = self.kv_seq_len;

        // 3f: Attention (GQA: num_heads Q heads, num_kv_heads KV heads)
        let attn_out = &mut self.scratch_attn_out;
        let scores = &mut self.scratch_scores;
        scores.resize(seq_len, 0.0);
        let heads_per_kv = self.num_heads / self.num_kv_heads;
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        for q_head in 0..self.num_heads {
            let kv_head = q_head / heads_per_kv;
            let q_off = q_head * self.head_dim;
            let q_vec = &q[q_off..q_off + self.head_dim];

            // Dot products with all cached keys
            for s in 0..seq_len {
                let k_off = s * self.head_dim;
                let k_vec = &self.kv_keys[kv_head][k_off..k_off + self.head_dim];
                let mut dot = 0.0f32;
                for d in 0..self.head_dim {
                    dot += q_vec[d] * k_vec[d];
                }
                scores[s] = dot * scale;
            }

            // Causal softmax
            softmax_inplace(scores);

            // Weighted sum of values
            let out_off = q_head * self.head_dim;
            for d in 0..self.head_dim {
                attn_out[out_off + d] = 0.0;
            }
            for s in 0..seq_len {
                let v_off = s * self.head_dim;
                let w = scores[s];
                for d in 0..self.head_dim {
                    attn_out[out_off + d] += w * self.kv_values[kv_head][v_off + d];
                }
            }
        }

        // 3g: O projection → [H]
        let o_out = &mut self.scratch_o_out;
        matvec_bf16_raw(attn_out, &self.o_proj, o_out, h, q_dim);

        // 3h: Residual connection with fused hidden (not embed)
        let mut decoder_output = vec![0.0f32; h];
        for i in 0..h {
            decoder_output[i] = fused[i] + o_out[i];
        }

        // 3i: Post-attention norm → MLP (SwiGLU)
        let post_norm = &mut self.scratch_post_norm;
        rms_norm_bf16(&decoder_output, &self.post_attn_norm, post_norm, self.rms_norm_eps);

        let gate = &mut self.scratch_gate;
        let up = &mut self.scratch_up;
        matvec_bf16_raw(post_norm, &self.mlp_gate, gate, self.intermediate_size, h);
        matvec_bf16_raw(post_norm, &self.mlp_up, up, self.intermediate_size, h);

        // SwiGLU: silu(gate) * up
        for i in 0..self.intermediate_size {
            let g = gate[i];
            gate[i] = (g / (1.0 + (-g).exp())) * up[i];
        }

        let mlp_out = &mut self.scratch_mlp_out;
        matvec_bf16_raw(gate, &self.mlp_down, mlp_out, h, self.intermediate_size);

        // Residual
        for i in 0..h {
            decoder_output[i] += mlp_out[i];
        }

        // Step 4: Final norm → lm_head → draft logits
        let final_norm_out = &mut self.scratch_final_norm;
        rms_norm_bf16(&decoder_output, &self.final_norm, final_norm_out, self.rms_norm_eps);

        let logits = &mut self.scratch_logits;
        matvec_bf16_raw_par(final_norm_out, &self.lm_head, logits, self.draft_vocab_size, h);

        // Step 5: Argmax + d2t mapping
        let mut best_draft = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_draft = i as u32;
            }
        }
        let logits_out = logits.clone();
        let target_token = self.draft_to_target(best_draft);

        Ok((target_token, logits_out, decoder_output))
    }

    /// Get draft vocab size.
    pub fn draft_vocab_size(&self) -> usize {
        self.draft_vocab_size
    }

    /// Get target vocab size.
    pub fn target_vocab_size(&self) -> usize {
        self.target_vocab_size
    }

    /// Get the d2t mapping slice.
    pub fn d2t(&self) -> &[i64] {
        &self.d2t
    }

    /// Get the t2d mapping slice.
    pub fn t2d(&self) -> &[i32] {
        &self.t2d
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn load_tensor_bf16(tensors: &SafeTensors, name: &str) -> Result<Vec<BF16>> {
    let view = tensors.tensor(name)
        .map_err(|e| HerbertError::ModelLoad(format!("missing tensor '{}': {}", name, e)))?;
    load_bf16_from_view(&view, name)
}

/// RMS norm with BF16 weights (dispatches to optimized kernels).
fn rms_norm_bf16(input: &[f32], weight: &[BF16], output: &mut [f32], eps: f32) {
    crate::kernels::rms_norm_bf16(input, weight, output, eps);
}

/// Apply RoPE to a head vector in-place (non-interleaved).
/// Pairs are at (i, i + half_dim). `head` is [head_dim], cos/sin are [half_dim].
#[inline]
fn apply_rope(head: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    for i in 0..half {
        let x0 = head[i];
        let x1 = head[i + half];
        head[i] = x0 * cos[i] - x1 * sin[i];
        head[i + half] = x0 * sin[i] + x1 * cos[i];
    }
}

/// Apply RoPE to a head vector in-place (interleaved, for MRoPE).
/// Pairs are at (2*i, 2*i+1). `head` is [head_dim], cos/sin are [half_dim].
#[inline]
fn apply_rope_interleaved(head: &mut [f32], cos: &[f32], sin: &[f32]) {
    let n_pairs = cos.len();
    for i in 0..n_pairs {
        let x0 = head[2 * i];
        let x1 = head[2 * i + 1];
        head[2 * i] = x0 * cos[i] - x1 * sin[i];
        head[2 * i + 1] = x0 * sin[i] + x1 * cos[i];
    }
}

/// In-place softmax over a score vector.
fn softmax_inplace(scores: &mut [f32]) {
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum += *s;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for s in scores.iter_mut() {
            *s *= inv;
        }
    }
}

/// BF16 matvec: y[n] = w[n, k] * x[k], single-threaded.
/// `w` is row-major [n, k] stored as BF16.
fn matvec_bf16_raw(x: &[f32], w: &[BF16], y: &mut [f32], n: usize, k: usize) {
    for row in 0..n {
        let mut acc = 0.0f32;
        let w_off = row * k;
        for col in 0..k {
            acc += bf16_to_f32(w[w_off + col]) * x[col];
        }
        y[row] = acc;
    }
}

/// BF16 matvec: y[n] = w[n, k] * x[k], multi-threaded for large N.
fn matvec_bf16_raw_par(x: &[f32], w: &[BF16], y: &mut [f32], n: usize, k: usize) {
    const PAR_THRESHOLD: usize = 4096;
    if n * k < PAR_THRESHOLD * 128 {
        return matvec_bf16_raw(x, w, y, n, k);
    }

    let pool = global_pool();
    let tile = 64;
    let num_tiles = n.div_ceil(tile);
    let x_ptr = SendPtr::new(x.as_ptr());
    let w_ptr = SendPtr::new(w.as_ptr());
    let y_ptr = SendMutPtr::new(y.as_mut_ptr());

    let _ = pool.parallel_for(num_tiles, move |_, tile_start, tile_end| {
        let x_ptr = x_ptr.ptr();
        let w_ptr = w_ptr.ptr();
        let y_ptr = y_ptr.ptr();
        for t in tile_start..tile_end {
            let row_start = t * tile;
            let row_end = (row_start + tile).min(n);
            for row in row_start..row_end {
                let mut acc = 0.0f32;
                let w_off = row * k;
                for col in 0..k {
                    acc += bf16_to_f32(unsafe { *w_ptr.add(w_off + col) }) * unsafe { *x_ptr.add(col) };
                }
                unsafe { *y_ptr.add(row) = acc; }
            }
        }
    });
}

/// fc matvec with implicit concat of 3 input vectors.
/// w is [H, 3*H], inputs are h0, h1, h2 each [H].
/// output[row] = dot(w[row, :], concat(h0, h1, h2))
fn matvec_bf16_3concat(
    h0: &[f32], h1: &[f32], h2: &[f32],
    w: &[BF16], y: &mut [f32], h: usize,
) {
    let k = 3 * h;
    for row in 0..h {
        let mut acc = 0.0f32;
        let w_off = row * k;
        for col in 0..h {
            acc += bf16_to_f32(w[w_off + col]) * h0[col];
        }
        for col in 0..h {
            acc += bf16_to_f32(w[w_off + h + col]) * h1[col];
        }
        for col in 0..h {
            acc += bf16_to_f32(w[w_off + 2 * h + col]) * h2[col];
        }
        y[row] = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_rope() {
        let mut head = vec![1.0, 0.0, 0.0, 1.0]; // [half=2]
        let cos = vec![1.0, 0.0]; // cos(0)=1, cos(pi/2)=0
        let sin = vec![0.0, 1.0]; // sin(0)=0, sin(pi/2)=1
        apply_rope(&mut head, &cos, &sin);
        // head[0] = 1*1 - 0*0 = 1
        // head[1] = 0*0 - 1*1 = -1
        // head[2] = 1*0 + 0*1 = 0
        // head[3] = 0*1 + 1*0 = 0  -- wait
        // Actually: x0=head[0]=1, x1=head[2]=0
        //   head[0] = 1*1 - 0*0 = 1
        //   head[2] = 1*0 + 0*1 = 0
        // x0=head[1]=0, x1=head[3]=1
        //   head[1] = 0*0 - 1*1 = -1
        //   head[3] = 0*1 + 1*0 = 0
        assert!((head[0] - 1.0).abs() < 1e-5);
        assert!((head[1] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_inplace() {
        let mut scores = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut scores);
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(scores[2] > scores[1]);
        assert!(scores[1] > scores[0]);
    }
}
