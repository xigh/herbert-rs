//! Backend implementation for unified Q4 quantization.

use herbert_backend_common::cpu_backend::{CpuBackendConfig, GenericCpuBackend};
use herbert_backend_common::generic_model::GenericModel;
use herbert_backend_common::linear_ops::LinearOps;
use herbert_backend_common::weight_cache::CacheWeight;
use herbert_core::backend::{Backend, DecodeOutput, LoadOpts, PrefillOutput, RunOpts, VerifyOutput, VisionEmbedding};
use herbert_core::config::Config;
use herbert_core::error::{HerbertError, Result};
use herbert_core::KvHandle;
use std::path::Path;

use crate::ops::Q4Ops;

pub struct Q4Config;

impl CpuBackendConfig for Q4Config {
    type Ops = Q4Ops;
    const NAME: &'static str = "q4";

    fn configure_thread_pool(num_threads: usize) -> Result<()> {
        crate::thread_pool::configure_global_pool(num_threads)
            .map_err(|e| HerbertError::Backend(format!("failed to configure thread pool: {}", e)))
    }

    fn load_model(dir: &Path, opts: LoadOpts) -> Result<(Config, GenericModel<Q4Ops>)> {
        crate::loader::load_model(dir, opts)
    }
}

pub struct Q4Backend {
    inner: GenericCpuBackend<Q4Config>,
}

impl Q4Backend {
    pub fn new() -> Self {
        Self {
            inner: GenericCpuBackend::new(),
        }
    }
}

impl Default for Q4Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for Q4Backend
where
    <Q4Ops as LinearOps>::Weight: CacheWeight,
{
    fn name(&self) -> &'static str {
        "q4"
    }

    fn load(&mut self, model_path: &Path, opts: LoadOpts) -> Result<()> {
        self.inner.load(model_path, opts)
    }

    fn prefill(&mut self, input_tokens: &[u32], opts: RunOpts) -> Result<(KvHandle, PrefillOutput)> {
        self.inner.prefill(input_tokens, opts)
    }

    fn decode_next(&mut self, kv: &mut KvHandle, token: u32, opts: RunOpts) -> Result<DecodeOutput> {
        self.inner.decode_next(kv, token, opts)
    }

    fn config(&self) -> Option<&Config> {
        self.inner.config()
    }

    fn continue_prefill(
        &mut self,
        kv: &mut KvHandle,
        input_tokens: &[u32],
        opts: RunOpts,
    ) -> Result<PrefillOutput> {
        self.inner.continue_prefill(kv, input_tokens, opts)
    }

    fn embed(&mut self, input_tokens: &[u32]) -> Result<Vec<f32>> {
        self.inner.embed(input_tokens)
    }

    fn embed_all(&mut self, input_tokens: &[u32], bidirectional: bool, l2_normalize: bool) -> Result<Vec<f32>> {
        self.inner.embed_all(input_tokens, bidirectional, l2_normalize)
    }

    fn embed_vl(
        &mut self,
        tokens: &[u32],
        images: &[VisionEmbedding],
        image_positions: &[usize],
    ) -> Result<Vec<f32>> {
        self.inner.embed_vl(tokens, images, image_positions)
    }

    fn prefill_vl(
        &mut self,
        tokens: &[u32],
        images: &[VisionEmbedding],
        image_positions: &[usize],
        opts: RunOpts,
    ) -> Result<(KvHandle, PrefillOutput)> {
        self.inner.prefill_vl(tokens, images, image_positions, opts)
    }

    fn verify_draft(&mut self, kv: &mut KvHandle, draft_tokens: &[u32]) -> Result<VerifyOutput> {
        self.inner.verify_draft(kv, draft_tokens)
    }

    fn kv_seq_len(&self, kv: &KvHandle) -> Result<usize> {
        self.inner.kv_seq_len(kv)
    }

    fn truncate_kv(&mut self, kv: &mut KvHandle, target_len: usize) -> Result<()> {
        self.inner.truncate_kv(kv, target_len)
    }

    fn encode_vision(
        &mut self,
        patches: &[f32],
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
    ) -> Option<Result<VisionEmbedding>> {
        self.inner.encode_vision(patches, grid_t, grid_h, grid_w)
    }

    fn encode_vision_with_progress(
        &mut self,
        patches: &[f32],
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
        progress: Box<dyn Fn(usize, usize) + Send + Sync>,
    ) -> Option<Result<VisionEmbedding>> {
        self.inner
            .encode_vision_with_progress(patches, grid_t, grid_h, grid_w, progress)
    }

    fn take_eagle3_hidden_states(&mut self, kv: &mut KvHandle) -> Result<[Vec<f32>; 3]> {
        self.inner.take_eagle3_hidden_states(kv)
    }

    fn enable_eagle3_extraction(&mut self, kv: &mut KvHandle, extract_layers: [usize; 3]) -> Result<()> {
        self.inner.enable_eagle3_extraction(kv, extract_layers)
    }

    fn embed_tokens_bf16(&self) -> Result<&[u16]> {
        self.inner.embed_tokens_bf16()
    }
}
