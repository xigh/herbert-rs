//! Metal compute backend for Qwen3 inference.
//!
//! On macOS, provides a real GPU inference pipeline using Metal via `objc2-metal`.
//! On other platforms, provides a stub that returns errors.

// Module declarations — real Metal code, macOS only.
#[cfg(target_os = "macos")]
pub mod context;
#[cfg(target_os = "macos")]
pub mod memory;
#[cfg(target_os = "macos")]
pub mod shaders;
#[cfg(target_os = "macos")]
pub mod backend;
#[cfg(target_os = "macos")]
pub mod model;
#[cfg(target_os = "macos")]
pub mod kv_cache;
#[cfg(target_os = "macos")]
pub mod loader;
#[cfg(target_os = "macos")]
pub mod cache;
#[cfg(target_os = "macos")]
pub mod kernels;
#[cfg(target_os = "macos")]
pub mod profiler;
#[cfg(target_os = "macos")]
pub mod vision_model;
#[cfg(target_os = "macos")]
pub mod vision_loader;
#[cfg(target_os = "macos")]
pub mod vision_encoder;

use herbert_core::backend::{
    Backend, DecodeOutput, LoadOpts, PrefillOutput, RunOpts, VerifyOutput, VisionEmbedding,
};
use herbert_core::error::HerbertError;
use herbert_core::kv_cache::KvHandle;
use std::path::Path;

// ============================================================================
// macOS: real Metal backend
// ============================================================================

#[cfg(target_os = "macos")]
pub use loader::QuantMode;

#[cfg(target_os = "macos")]
pub struct MetalBackend {
    inner: Option<backend::MetalBackendInner>,
    quant_mode: loader::QuantMode,
    /// Keeps mmap alive when model is loaded from cache (zero-copy buffers point into it).
    _cache_mmap: Option<cache::CacheMmap>,
    /// Vision model, preloaded during `load()` for VL models.
    vision_model: Option<vision_model::MetalVisionModel>,
    /// Model directory path, stored for vision loading.
    model_path: Option<std::path::PathBuf>,
}

#[cfg(target_os = "macos")]
// SAFETY: MetalBackendInner holds Metal handles that are thread-safe when
// accessed through the Metal API (single queue, sequential command submission).
unsafe impl Send for MetalBackend {}
#[cfg(target_os = "macos")]
unsafe impl Sync for MetalBackend {}

#[cfg(target_os = "macos")]
impl MetalBackend {
    pub fn new() -> Self {
        Self {
            inner: None,
            quant_mode: loader::QuantMode::BF16,
            _cache_mmap: None,
            vision_model: None,
            model_path: None,
        }
    }

    pub fn with_quant_mode(quant_mode: loader::QuantMode) -> Self {
        Self {
            inner: None,
            quant_mode,
            _cache_mmap: None,
            vision_model: None,
            model_path: None,
        }
    }

    fn vision_config_if_present(
        model_path: &Path,
    ) -> herbert_core::error::Result<Option<herbert_vision::config::VisionConfig>> {
        let config_path = model_path.join("config.json");
        let content = std::fs::read_to_string(&config_path)?;
        let root: serde_json::Value = serde_json::from_str(&content)?;
        if root.get("vision_config").is_none() {
            return Ok(None);
        }
        herbert_vision::config::VisionConfig::from_file(&config_path).map(Some)
    }

    fn load_vision_model_if_present(&mut self) -> herbert_core::error::Result<()> {
        if self.vision_model.is_some() {
            return Ok(());
        }
        let inner = self.inner.as_ref().ok_or_else(|| {
            HerbertError::Backend("Metal backend not loaded".into())
        })?;
        // Metal GPU vision encoder only supports Qwen3-VL.
        // Mistral3/Pixtral uses CPU-only vision encoding via the CLI path.
        if inner.config.model_family != herbert_core::config::ModelFamily::Qwen3 {
            return Ok(());
        }
        let model_path = self.model_path.as_ref().ok_or_else(|| {
            HerbertError::Backend("No model path stored for vision loading".into())
        })?;
        let Some(vision_config) = Self::vision_config_if_present(model_path)? else {
            return Ok(());
        };
        eprintln!("[metal-vision] Loading vision encoder to GPU...");
        let t0 = std::time::Instant::now();
        let vision_model = vision_loader::load_vision_model(
            model_path, &inner.ctx.device, &vision_config,
        )?;
        eprintln!(
            "[metal-vision] Vision model loaded: {} blocks, dim={} ({:.1}s)",
            vision_config.num_layers, vision_config.hidden_size,
            t0.elapsed().as_secs_f64(),
        );
        self.vision_model = Some(vision_model);
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        match self.quant_mode {
            loader::QuantMode::Int8 => "metal-int8",
            loader::QuantMode::BF16 => "metal-bf16",
            loader::QuantMode::Q4 => "metal-q4",
        }
    }

    fn load(&mut self, model_path: &Path, opts: LoadOpts) -> herbert_core::error::Result<()> {
        let max_tokens = opts.kv_reserve_tokens.unwrap_or(4096).max(256);

        let ctx = context::MetalContext::new()?;
        eprintln!("[metal] Using device: {}", ctx.device_name());

        // Try loading from cache first (zero-copy mmap, skips BF16→Q4 quantization)
        let (config, model) = if let Some((config, cached, cache_mmap)) = cache::try_load_cache(
            &ctx.device, model_path, max_tokens, self.quant_mode,
        ) {
            self._cache_mmap = Some(cache_mmap);
            (config, cached)
        } else {
            let (config, model) = loader::load_model(
                model_path,
                &ctx.device,
                max_tokens,
                self.quant_mode,
            )?;
            // Save to cache for next time
            if let Err(e) = cache::save_cache(&model, &config, model_path, self.quant_mode) {
                eprintln!("[metal] Warning: failed to save cache: {}", e);
            }
            (config, model)
        };

        eprintln!(
            "[metal] Backend ready: {} layers, hidden={}, vocab={}, max_tokens={}",
            config.num_layers, config.hidden_size, config.vocab_size, max_tokens
        );

        self.model_path = Some(model_path.to_path_buf());
        self.inner = Some(backend::MetalBackendInner {
            model,
            config,
            max_tokens,
            kv_quant: opts.kv_quant,
            kv_budget: opts.kv_budget,
            ctx,
        });
        self.vision_model = None;
        self.load_vision_model_if_present()?;

        Ok(())
    }

    fn prefill(
        &mut self,
        input_tokens: &[u32],
        opts: RunOpts,
    ) -> herbert_core::error::Result<(KvHandle, PrefillOutput)> {
        match self.inner.as_ref() {
            Some(inner) => inner.do_prefill(input_tokens, opts),
            None => Err(HerbertError::Backend(
                "Metal backend not loaded (call load() first)".into(),
            )),
        }
    }

    fn decode_next(
        &mut self,
        kv: &mut KvHandle,
        token: u32,
        opts: RunOpts,
    ) -> herbert_core::error::Result<DecodeOutput> {
        match self.inner.as_ref() {
            Some(inner) => inner.do_decode(kv, token, opts),
            None => Err(HerbertError::Backend(
                "Metal backend not loaded (call load() first)".into(),
            )),
        }
    }

    fn verify_draft(
        &mut self,
        kv: &mut KvHandle,
        draft_tokens: &[u32],
    ) -> herbert_core::error::Result<VerifyOutput> {
        match self.inner.as_ref() {
            Some(inner) => inner.do_verify_draft(kv, draft_tokens),
            None => Err(HerbertError::Backend(
                "Metal backend not loaded (call load() first)".into(),
            )),
        }
    }

    fn kv_seq_len(&self, kv: &KvHandle) -> herbert_core::error::Result<usize> {
        let cache = kv.get_ref::<kv_cache::MetalKvCache>()?;
        Ok(cache.seq_len)
    }

    fn truncate_kv(
        &mut self,
        kv: &mut KvHandle,
        target_len: usize,
    ) -> herbert_core::error::Result<()> {
        let cache = kv.get_mut::<kv_cache::MetalKvCache>()?;
        if cache.seq_len > target_len {
            let removed = cache.seq_len - target_len;
            cache.seq_len = target_len;
            cache.rope_pos = cache.rope_pos.saturating_sub(removed);
            // Metal KV buffers don't need actual truncation — old data is
            // overwritten on the next prefill/decode since kv_offset = seq_len.
            // INT8 quantized cache needs re-quantization of the valid range
            // on the next verify_draft call (handled there).
        }
        Ok(())
    }

    fn encode_vision(
        &mut self,
        patches: &[f32],
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
    ) -> Option<herbert_core::error::Result<VisionEmbedding>> {
        let inner = self.inner.as_ref()?;
        let vision_model = self.vision_model.as_ref()?;
        let merge_size = vision_model.config.spatial_merge_size;
        let merged_h = grid_h / merge_size;
        let merged_w = grid_w / merge_size;

        let result = vision_encoder::vision_encode_gpu(
            vision_model, &inner.ctx, patches, grid_t, grid_h, grid_w,
        );
        Some(result.map(|vo| VisionEmbedding {
            hidden_states: vo.hidden_states,
            num_tokens: vo.num_tokens,
            grid_h: merged_h,
            grid_w: merged_w,
            deepstack_features: vo.deepstack_features,
        }))
    }

    fn encode_vision_with_progress(
        &mut self,
        patches: &[f32],
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
        progress: Box<dyn Fn(usize, usize) + Send + Sync>,
    ) -> Option<herbert_core::error::Result<VisionEmbedding>> {
        let inner = self.inner.as_ref()?;
        let vision_model = self.vision_model.as_ref()?;
        let merge_size = vision_model.config.spatial_merge_size;
        let merged_h = grid_h / merge_size;
        let merged_w = grid_w / merge_size;

        let result = vision_encoder::vision_encode_gpu_with_progress(
            vision_model, &inner.ctx, patches, grid_t, grid_h, grid_w, progress,
        );
        Some(result.map(|vo| VisionEmbedding {
            hidden_states: vo.hidden_states,
            num_tokens: vo.num_tokens,
            grid_h: merged_h,
            grid_w: merged_w,
            deepstack_features: vo.deepstack_features,
        }))
    }

    fn prefill_vl(
        &mut self,
        tokens: &[u32],
        images: &[VisionEmbedding],
        image_positions: &[usize],
        opts: RunOpts,
    ) -> herbert_core::error::Result<(KvHandle, PrefillOutput)> {
        match self.inner.as_ref() {
            Some(inner) => inner.do_prefill_vl(tokens, images, image_positions, opts),
            None => Err(HerbertError::Backend(
                "Metal backend not loaded (call load() first)".into(),
            )),
        }
    }

    fn embed(&mut self, input_tokens: &[u32]) -> herbert_core::error::Result<Vec<f32>> {
        match self.inner.as_ref() {
            Some(inner) => inner.do_embed(input_tokens),
            None => Err(HerbertError::Backend(
                "Metal backend not loaded (call load() first)".into(),
            )),
        }
    }

    fn embed_vl(
        &mut self,
        tokens: &[u32],
        images: &[VisionEmbedding],
        image_positions: &[usize],
    ) -> herbert_core::error::Result<Vec<f32>> {
        match self.inner.as_ref() {
            Some(inner) => inner.do_embed_vl(tokens, images, image_positions),
            None => Err(HerbertError::Backend(
                "Metal backend not loaded (call load() first)".into(),
            )),
        }
    }

    fn config(&self) -> Option<&herbert_core::config::Config> {
        self.inner.as_ref().map(|i| &i.config)
    }
}

// ============================================================================
// Non-macOS: stub backend that returns errors
// ============================================================================

#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    Int8,
    BF16,
    Q4,
}

#[cfg(not(target_os = "macos"))]
pub struct MetalBackend {
    _private: (),
}

#[cfg(not(target_os = "macos"))]
impl MetalBackend {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn with_quant_mode(_quant_mode: QuantMode) -> Self {
        Self { _private: () }
    }
}

#[cfg(not(target_os = "macos"))]
impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn load(&mut self, _model_path: &Path, _opts: LoadOpts) -> herbert_core::error::Result<()> {
        Err(HerbertError::Backend(
            "Metal backend is only supported on macOS".to_string(),
        ))
    }

    fn prefill(
        &mut self,
        _input_tokens: &[u32],
        _opts: RunOpts,
    ) -> herbert_core::error::Result<(KvHandle, PrefillOutput)> {
        Err(HerbertError::Backend(
            "Metal backend is only supported on macOS".to_string(),
        ))
    }

    fn decode_next(
        &mut self,
        _kv: &mut KvHandle,
        _token: u32,
        _opts: RunOpts,
    ) -> herbert_core::error::Result<DecodeOutput> {
        Err(HerbertError::Backend(
            "Metal backend is only supported on macOS".to_string(),
        ))
    }

    fn config(&self) -> Option<&herbert_core::config::Config> {
        None
    }
}
