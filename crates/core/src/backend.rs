//! Backend trait: the interface every inference backend implements.

use crate::config::KvQuantType;
use crate::error::Result;
use crate::kv_cache::KvHandle;
use std::path::{Path, PathBuf};

/// Options for loading a model
#[derive(Debug, Clone)]
pub struct LoadOpts {
    /// Whether to validate shapes and dtypes during loading
    pub validate: bool,
    /// Optional thread count override
    pub num_threads: Option<usize>,
    /// Optional reservation hint for KV cache (in tokens)
    pub kv_reserve_tokens: Option<usize>,
    /// Print loading progress to stderr
    pub show_progress: bool,
    /// Disable quantized weight cache (force re-quantization)
    pub no_cache: bool,
    /// Chunk size for prefill chunking (None = no chunking)
    pub prefill_chunk_size: Option<usize>,
    /// Max threads during decode (None = auto: 2/3 of thread count, capped at 8)
    pub decode_threads: Option<usize>,
    /// Directory for prefix KV cache (None = disabled)
    pub prefix_cache_dir: Option<PathBuf>,
    /// KV cache quantization type (default: BF16)
    pub kv_quant: KvQuantType,
    /// Use BF16 format for Q in attention dot products (VDPBF16PS optimization)
    pub use_q_bf16: bool,
    /// KV cache budget for H2O eviction (None = unlimited, no eviction)
    pub kv_budget: Option<usize>,
}

impl Default for LoadOpts {
    fn default() -> Self {
        Self {
            validate: true,
            num_threads: None,
            kv_reserve_tokens: None,
            show_progress: false,
            no_cache: false,
            prefill_chunk_size: None,
            decode_threads: None,
            prefix_cache_dir: None,
            kv_quant: KvQuantType::BF16,
            use_q_bf16: false,
            kv_budget: None,
        }
    }
}

/// Options for running inference
#[derive(Debug, Clone, Default)]
pub struct RunOpts {
    /// Whether to return per-layer timings
    pub profile: bool,
    /// Whether to return logits (for validation)
    pub return_logits: bool,
    /// Optional decode target length (used by benchmark-oriented paths)
    pub decode_tokens: Option<usize>,
    /// Continue decoding even if EOS is reached
    pub ignore_eos: bool,
    /// Number of tokens in the system prompt prefix (None = no double prefill)
    pub system_prefix_len: Option<usize>,
}

/// Output from a prefill step
#[derive(Debug, Clone)]
pub struct PrefillOutput {
    /// Logits for the last position (if requested via RunOpts)
    pub logits: Option<Vec<f32>>,
    /// First generated token (from argmax of logits)
    pub first_token: u32,
    /// Timing breakdown (if profiling enabled)
    pub timings: Option<LayerTimings>,
    /// Number of tokens loaded from prefix cache (0 = cache miss)
    pub cached_prefix_len: usize,
}

/// Output from a decode step
#[derive(Debug, Clone)]
pub struct DecodeOutput {
    /// Logits (if requested via RunOpts)
    pub logits: Option<Vec<f32>>,
    /// Chosen token ID
    pub token: u32,
    /// Timing breakdown (if profiling enabled)
    pub timings: Option<LayerTimings>,
}

/// Per-layer timing information
#[derive(Debug, Clone)]
pub struct LayerTimings {
    /// Time spent in each decoder layer (in seconds)
    pub layer_times: Vec<f64>,
    /// Total prefill/decode time (in seconds)
    pub total_time: f64,
}

/// Output from a draft verification step (speculative decoding).
#[derive(Debug, Clone)]
pub struct VerifyOutput {
    /// Logits at each of the K+1 positions (K draft tokens + 1 bonus).
    /// `logits_per_position[i]` is the logit vector after processing draft token i.
    /// The last entry (index K) is the logit vector for the bonus position.
    pub logits_per_position: Vec<Vec<f32>>,
}

/// Vision embedding produced by the vision encoder, ready for injection into the LLM.
#[derive(Debug, Clone)]
pub struct VisionEmbedding {
    /// Flattened hidden states: `[num_tokens * hidden_size]`.
    pub hidden_states: Vec<f32>,
    /// Number of visual tokens.
    pub num_tokens: usize,
    /// Grid height (number of rows of visual tokens).
    pub grid_h: usize,
    /// Grid width (number of columns of visual tokens).
    pub grid_w: usize,
    /// DeepStack features: one `Vec<f32>` per DS layer, each `[num_tokens * hidden_size]`.
    pub deepstack_features: Vec<Vec<f32>>,
}

/// Trait that all inference backends implement.
pub trait Backend: Send + Sync {
    /// Backend name (e.g. "metal-bf16", "vulkan-q4-v7").
    fn name(&self) -> &'static str;

    /// Load model weights from `model_path`.
    fn load(&mut self, model_path: &Path, opts: LoadOpts) -> Result<()>;

    /// Run prefill: process the full prompt and return the first generated token.
    fn prefill(&mut self, input_tokens: &[u32], opts: RunOpts)
        -> Result<(KvHandle, PrefillOutput)>;

    /// Run one decode step: feed a token, advance KV cache, return next token.
    fn decode_next(&mut self, kv: &mut KvHandle, token: u32, opts: RunOpts)
        -> Result<DecodeOutput>;

    /// Get model configuration (available after load)
    fn config(&self) -> Option<&crate::config::Config>;

    /// Continue prefill: append new tokens to an existing KV cache (for multi-turn chat).
    fn continue_prefill(
        &mut self,
        _kv: &mut KvHandle,
        _input_tokens: &[u32],
        _opts: RunOpts,
    ) -> Result<PrefillOutput> {
        Err(crate::error::HerbertError::Backend(
            "continue_prefill not supported".to_string(),
        ))
    }

    /// Compute a text embedding: run a stateless forward pass and return
    /// the L2-normalised hidden state of the last token (no KV cache retained).
    fn embed(&mut self, _input_tokens: &[u32]) -> Result<Vec<f32>> {
        Err(crate::error::HerbertError::Backend(
            "embed not supported".to_string(),
        ))
    }

    /// Compute per-token embeddings: run a stateless forward pass and return
    /// ALL token hidden states (RMS-normed per position).
    /// Returns `seq_len * hidden_size` floats.
    /// When `bidirectional` is true, attention layers use full (non-causal) masking.
    /// When `l2_normalize` is true, each token embedding is also L2-normalized.
    fn embed_all(&mut self, _input_tokens: &[u32], _bidirectional: bool, _l2_normalize: bool) -> Result<Vec<f32>> {
        Err(crate::error::HerbertError::Backend(
            "embed_all not supported".to_string(),
        ))
    }

    /// Compute a vision-language embedding: run a stateless forward pass with
    /// pre-encoded vision tokens injected at `image_positions`, then return
    /// the L2-normalised hidden state of the last token (no KV cache retained).
    fn embed_vl(
        &mut self,
        _tokens: &[u32],
        _images: &[VisionEmbedding],
        _image_positions: &[usize],
    ) -> Result<Vec<f32>> {
        Err(crate::error::HerbertError::Backend(
            "embed_vl not supported".to_string(),
        ))
    }

    /// Run vision encoder on GPU if supported.
    ///
    /// Returns `Some(Ok(embedding))` if GPU vision is available and succeeds,
    /// `Some(Err(...))` on GPU error, or `None` if not supported (use CPU fallback).
    fn encode_vision(
        &mut self,
        _patches: &[f32],
        _grid_t: usize,
        _grid_h: usize,
        _grid_w: usize,
    ) -> Option<Result<VisionEmbedding>> {
        None
    }

    /// Verify K draft tokens in a single forward pass (speculative decoding).
    ///
    /// Runs a prefill-like pass over the `draft_tokens`, appending them to the
    /// existing KV cache, and returns logits at **all K+1 positions** (K draft
    /// positions + 1 bonus position). This allows the caller to check each
    /// draft token against the target model's distribution.
    ///
    /// After verification, the caller should `truncate_to()` the KV cache
    /// to keep only the accepted tokens.
    fn verify_draft(
        &mut self,
        _kv: &mut KvHandle,
        _draft_tokens: &[u32],
    ) -> Result<VerifyOutput> {
        Err(crate::error::HerbertError::Backend(
            "verify_draft not supported".to_string(),
        ))
    }

    /// Get the current sequence length from a KV cache handle.
    ///
    /// Used by speculative decoding to track KV cache positions.
    fn kv_seq_len(
        &self,
        _kv: &KvHandle,
    ) -> Result<usize> {
        Err(crate::error::HerbertError::Backend(
            "kv_seq_len not supported".to_string(),
        ))
    }

    /// Truncate the KV cache to `target_len` positions.
    ///
    /// Used by speculative decoding to roll back rejected draft tokens.
    fn truncate_kv(
        &mut self,
        _kv: &mut KvHandle,
        _target_len: usize,
    ) -> Result<()> {
        Err(crate::error::HerbertError::Backend(
            "truncate_kv not supported".to_string(),
        ))
    }

    /// Like encode_vision, but with a progress callback `(block_done, total_blocks)`.
    /// On Metal, fires asynchronously via MTLSharedEvent as each group of blocks completes.
    fn encode_vision_with_progress(
        &mut self,
        patches: &[f32],
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
        _progress: Box<dyn Fn(usize, usize) + Send + Sync>,
    ) -> Option<Result<VisionEmbedding>> {
        self.encode_vision(patches, grid_t, grid_h, grid_w)
    }

    /// Take the captured EAGLE-3 hidden states from the last decode step.
    ///
    /// Returns an array of 3 hidden state vectors (one per extraction layer).
    /// After calling this, the internal storage is cleared.
    fn take_eagle3_hidden_states(
        &mut self,
        _kv: &mut KvHandle,
    ) -> Result<[Vec<f32>; 3]> {
        Err(crate::error::HerbertError::Backend(
            "take_eagle3_hidden_states not supported".to_string(),
        ))
    }

    /// Enable EAGLE-3 hidden state extraction during decode.
    fn enable_eagle3_extraction(
        &mut self,
        _kv: &mut KvHandle,
        _extract_layers: [usize; 3],
    ) -> Result<()> {
        Err(crate::error::HerbertError::Backend(
            "enable_eagle3_extraction not supported".to_string(),
        ))
    }

    /// Get the target model's embed_tokens table (BF16).
    ///
    /// Used by EAGLE-3 to share the target model's embedding table.
    fn embed_tokens_bf16(&self) -> Result<&[u16]> {
        Err(crate::error::HerbertError::Backend(
            "embed_tokens_bf16 not supported".to_string(),
        ))
    }

    /// Prefill with vision embeddings (VL models).
    ///
    /// `image_positions[i]` is the token index where `images[i]`'s tokens start.
    fn prefill_vl(
        &mut self,
        _tokens: &[u32],
        _images: &[VisionEmbedding],
        _image_positions: &[usize],
        _opts: RunOpts,
    ) -> Result<(KvHandle, PrefillOutput)> {
        Err(crate::error::HerbertError::Backend(
            "prefill_vl not supported by this backend".to_string(),
        ))
    }
}
