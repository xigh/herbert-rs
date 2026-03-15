//! Vulkan compute backend for herbert-qwen3-rs.
//!
//! On Linux, provides a real GPU inference pipeline using Vulkan 1.3 via `ash`.
//! On other platforms (macOS), provides a stub that returns errors.

// Module declarations — real Vulkan code, Linux only.
#[cfg(target_os = "linux")]
pub mod context;
#[cfg(target_os = "linux")]
pub mod memory;
#[cfg(target_os = "linux")]
pub mod shaders;
#[cfg(target_os = "linux")]
pub mod backend;
#[cfg(target_os = "linux")]
pub mod model;
#[cfg(target_os = "linux")]
pub mod kv_cache;
#[cfg(target_os = "linux")]
pub mod loader;
#[cfg(target_os = "linux")]
pub mod cache;
#[cfg(target_os = "linux")]
pub mod kernels;

use herbert_core::backend::{
    Backend, DecodeOutput, LoadOpts, PrefillOutput, RunOpts,
};
use herbert_core::error::HerbertError;
use herbert_core::kv_cache::KvHandle;
use std::path::Path;

// ============================================================================
// Linux: real Vulkan backend
// ============================================================================

#[cfg(target_os = "linux")]
pub use loader::QuantMode;

#[cfg(target_os = "linux")]
pub struct VulkanBackend {
    inner: Option<backend::VulkanBackendInner>,
    quant_mode: loader::QuantMode,
}

#[cfg(target_os = "linux")]
// SAFETY: VulkanBackendInner holds Vulkan handles that are thread-safe when
// accessed through the Vulkan API (single queue, sequential command submission).
unsafe impl Send for VulkanBackend {}
#[cfg(target_os = "linux")]
unsafe impl Sync for VulkanBackend {}

#[cfg(target_os = "linux")]
impl VulkanBackend {
    pub fn new() -> Self {
        Self { inner: None, quant_mode: loader::QuantMode::BF16 }
    }

    pub fn with_quant_mode(quant_mode: loader::QuantMode) -> Self {
        Self { inner: None, quant_mode }
    }
}

#[cfg(target_os = "linux")]
impl Backend for VulkanBackend {
    fn name(&self) -> &'static str {
        match self.quant_mode {
            loader::QuantMode::Int8 => "vulkan-int8",
            loader::QuantMode::BF16 => "vulkan-bf16",
            loader::QuantMode::Q4 => "vulkan-q4",
        }
    }

    fn load(&mut self, model_path: &Path, opts: LoadOpts) -> herbert_core::error::Result<()> {
        let max_tokens = opts.kv_reserve_tokens.unwrap_or(4096).max(256);
        let ctx = context::VulkanContext::new(0)?;

        // Try loading from cache first (skips quantization)
        let (config, model) = if !opts.no_cache {
            if let Some((config, cached)) = cache::try_load_cache(
                &ctx, model_path, max_tokens, self.quant_mode,
            ) {
                (config, cached)
            } else {
                let (config, model) = loader::load_model(model_path, &ctx, max_tokens, self.quant_mode)?;
                if let Err(e) = cache::save_cache(&ctx, &model, &config, model_path, self.quant_mode) {
                    eprintln!("[vulkan] Warning: failed to save cache: {}", e);
                }
                (config, model)
            }
        } else {
            loader::load_model(model_path, &ctx, max_tokens, self.quant_mode)?
        };

        eprintln!(
            "[vulkan] Backend ready: {} layers, hidden={}, vocab={}, max_tokens={}",
            config.num_layers, config.hidden_size, config.vocab_size, max_tokens
        );
        self.inner = Some(backend::VulkanBackendInner {
            model,
            config,
            max_tokens,
            ctx,
        });
        Ok(())
    }

    fn prefill(
        &mut self,
        input_tokens: &[u32],
        opts: RunOpts,
    ) -> herbert_core::error::Result<(KvHandle, PrefillOutput)> {
        match self.inner.as_ref() {
            Some(inner) => inner.do_prefill(input_tokens, opts),
            None => Err(HerbertError::Backend("Vulkan backend not loaded (call load() first)".into())),
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
            None => Err(HerbertError::Backend("Vulkan backend not loaded (call load() first)".into())),
        }
    }

    fn config(&self) -> Option<&herbert_core::config::Config> {
        self.inner.as_ref().map(|i| &i.config)
    }
}

// ============================================================================
// Non-Linux (macOS, etc.): stub backend that returns errors
// ============================================================================

#[cfg(not(target_os = "linux"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    Int8,
    BF16,
    Q4,
}

#[cfg(not(target_os = "linux"))]
pub struct VulkanBackend {
    _private: (),
}

#[cfg(not(target_os = "linux"))]
impl VulkanBackend {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn with_quant_mode(_quant_mode: QuantMode) -> Self {
        Self { _private: () }
    }
}

#[cfg(not(target_os = "linux"))]
impl Backend for VulkanBackend {
    fn name(&self) -> &'static str {
        "vulkan"
    }

    fn load(&mut self, _model_path: &Path, _opts: LoadOpts) -> herbert_core::error::Result<()> {
        Err(HerbertError::Backend(
            "Vulkan backend is only supported on Linux".to_string(),
        ))
    }

    fn prefill(
        &mut self,
        _input_tokens: &[u32],
        _opts: RunOpts,
    ) -> herbert_core::error::Result<(KvHandle, PrefillOutput)> {
        Err(HerbertError::Backend(
            "Vulkan backend is only supported on Linux".to_string(),
        ))
    }

    fn decode_next(
        &mut self,
        _kv: &mut KvHandle,
        _token: u32,
        _opts: RunOpts,
    ) -> herbert_core::error::Result<DecodeOutput> {
        Err(HerbertError::Backend(
            "Vulkan backend is only supported on Linux".to_string(),
        ))
    }

    fn config(&self) -> Option<&herbert_core::config::Config> {
        None
    }
}
