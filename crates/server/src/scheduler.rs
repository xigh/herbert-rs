//! Inference scheduler with semaphore-based concurrency control.
//!
//! Replaces `Mutex<InferenceState>` — all fields are immutable or internally
//! synchronized, so multiple requests can proceed concurrently.

use crate::inference::ServerInference;
use crate::metrics::InferenceMetrics;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Shared inference state that supports concurrent access.
///
/// All fields are either immutable after construction or internally
/// synchronized (`Arc<dyn ServerInference>` uses `&self`).
/// The `semaphore` controls how many requests can run simultaneously.
pub(crate) struct InferenceScheduler {
    pub(crate) backend: Arc<dyn ServerInference>,
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) semaphore: Arc<tokio::sync::Semaphore>,
    pub(crate) metrics: Arc<InferenceMetrics>,
    pub(crate) eos_token_id: u32,
    pub(crate) im_end_id: u32,
    pub(crate) think_open_id: u32,
    pub(crate) think_close_id: u32,
    pub(crate) newline_id: u32,
    pub(crate) tool_call_open_id: Option<u32>,
    pub(crate) tool_call_close_id: Option<u32>,
    pub(crate) model_name: String,
    pub(crate) default_temperature: f32,
    pub(crate) default_top_k: usize,
    pub(crate) default_top_p: f32,
    pub(crate) think_budget: usize,
    pub(crate) nothink: bool,
    pub(crate) max_concurrent: usize,
}
