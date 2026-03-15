//! Backend abstraction for the server: wraps any Backend behind a Mutex.

use herbert_core::backend::{Backend, DecodeOutput, PrefillOutput, RunOpts};
use herbert_core::config::Config;
use herbert_core::error::Result;
use herbert_core::kv_cache::KvHandle;
use std::sync::Mutex;

/// Trait used by server handlers for inference.
///
/// All methods take `&self` so the backend can be shared across requests
/// (the scheduler's semaphore controls actual concurrency).
pub trait ServerInference: Send + Sync {
    /// Run prefill and return a KV handle + first-token output.
    fn prefill(&self, input_tokens: &[u32], opts: RunOpts) -> Result<(KvHandle, PrefillOutput)>;

    /// Run one decode step.
    fn decode_next(&self, kv: &mut KvHandle, token: u32, opts: RunOpts) -> Result<DecodeOutput>;

    /// Get model configuration.
    fn config(&self) -> &Config;
}

/// Wraps any `Backend` behind a Mutex for thread-safe `&self` access.
///
/// Works for both CPU and GPU backends. The scheduler's semaphore
/// controls actual concurrency (default: 1 = serial).
pub struct GpuServerBackend {
    backend: Mutex<Box<dyn Backend>>,
    config: Config,
}

impl GpuServerBackend {
    /// Create from an already-loaded backend.
    pub fn new(backend: Box<dyn Backend>) -> Result<Self> {
        let config = backend
            .config()
            .cloned()
            .ok_or_else(|| {
                herbert_core::error::HerbertError::Backend(
                    "Cannot create GpuServerBackend: config not available".to_string(),
                )
            })?;
        Ok(Self {
            backend: Mutex::new(backend),
            config,
        })
    }
}

impl ServerInference for GpuServerBackend {
    fn prefill(&self, input_tokens: &[u32], opts: RunOpts) -> Result<(KvHandle, PrefillOutput)> {
        let mut backend = self.backend.lock().unwrap();
        backend.prefill(input_tokens, opts)
    }

    fn decode_next(&self, kv: &mut KvHandle, token: u32, opts: RunOpts) -> Result<DecodeOutput> {
        let mut backend = self.backend.lock().unwrap();
        backend.decode_next(kv, token, opts)
    }

    fn config(&self) -> &Config {
        &self.config
    }
}
