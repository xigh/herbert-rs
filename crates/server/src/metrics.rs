//! Inference metrics collection and reporting.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Global metrics aggregator, shared across all concurrent requests.
pub(crate) struct InferenceMetrics {
    /// Total number of completed requests.
    pub total_requests: AtomicU64,
    /// Total input tokens processed across all requests.
    pub total_input_tokens: AtomicU64,
    /// Total output tokens generated across all requests.
    pub total_output_tokens: AtomicU64,
    /// Total prefill time in microseconds.
    pub total_prefill_us: AtomicU64,
    /// Total decode time in microseconds.
    pub total_decode_us: AtomicU64,
    /// Number of requests that had a shared prefix cache hit.
    pub prefix_cache_hits: AtomicU64,
    /// Peak concurrent requests observed.
    pub peak_concurrent: AtomicUsize,
    /// Currently active inference requests.
    pub active_requests: AtomicUsize,
    /// Requests currently waiting in queue for a semaphore permit.
    pub waiting_count: AtomicUsize,
    /// Server start time for uptime calculation.
    pub start_time: Instant,
}

impl InferenceMetrics {
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            total_input_tokens: AtomicU64::new(0),
            total_output_tokens: AtomicU64::new(0),
            total_prefill_us: AtomicU64::new(0),
            total_decode_us: AtomicU64::new(0),
            prefix_cache_hits: AtomicU64::new(0),
            peak_concurrent: AtomicUsize::new(0),
            active_requests: AtomicUsize::new(0),
            waiting_count: AtomicUsize::new(0),
            start_time: Instant::now(),
        }
    }

    /// Record that a request has started (call after acquiring semaphore).
    pub fn request_started(&self) {
        let current = self.active_requests.fetch_add(1, Ordering::Relaxed) + 1;
        // Update peak if needed
        let mut peak = self.peak_concurrent.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_concurrent.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }
    }

    /// Record that a request has completed.
    pub fn request_completed(&self, stats: &RequestStats) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_input_tokens
            .fetch_add(stats.input_tokens as u64, Ordering::Relaxed);
        self.total_output_tokens
            .fetch_add(stats.output_tokens as u64, Ordering::Relaxed);
        self.total_prefill_us
            .fetch_add(stats.prefill_us, Ordering::Relaxed);
        self.total_decode_us
            .fetch_add(stats.decode_us, Ordering::Relaxed);
        if stats.prefix_hit_len > 0 {
            self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Return a JSON-serializable summary.
    pub fn summary(&self) -> MetricsSummary {
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let total_input = self.total_input_tokens.load(Ordering::Relaxed);
        let total_output = self.total_output_tokens.load(Ordering::Relaxed);
        let total_prefill_us = self.total_prefill_us.load(Ordering::Relaxed);
        let total_decode_us = self.total_decode_us.load(Ordering::Relaxed);
        let uptime_secs = self.start_time.elapsed().as_secs_f64();

        let avg_prefill_tok_s = if total_prefill_us > 0 {
            (total_input as f64) / (total_prefill_us as f64 / 1_000_000.0)
        } else {
            0.0
        };
        let avg_decode_tok_s = if total_decode_us > 0 {
            (total_output as f64) / (total_decode_us as f64 / 1_000_000.0)
        } else {
            0.0
        };

        MetricsSummary {
            uptime_secs,
            total_requests,
            active_requests: self.active_requests.load(Ordering::Relaxed),
            waiting_in_queue: self.waiting_count.load(Ordering::Relaxed),
            peak_concurrent: self.peak_concurrent.load(Ordering::Relaxed),
            total_input_tokens: total_input,
            total_output_tokens: total_output,
            avg_prefill_tok_s,
            avg_decode_tok_s,
            prefix_cache_hits: self.prefix_cache_hits.load(Ordering::Relaxed),
            prefix_cache_hit_rate: if total_requests > 0 {
                self.prefix_cache_hits.load(Ordering::Relaxed) as f64 / total_requests as f64
            } else {
                0.0
            },
        }
    }
}

/// Per-request statistics collected during inference.
pub(crate) struct RequestStats {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub prefill_us: u64,
    pub decode_us: u64,
    pub prefix_hit_len: usize,
}

/// JSON-serializable metrics summary.
#[derive(serde::Serialize)]
pub(crate) struct MetricsSummary {
    pub uptime_secs: f64,
    pub total_requests: u64,
    pub active_requests: usize,
    pub waiting_in_queue: usize,
    pub peak_concurrent: usize,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub avg_prefill_tok_s: f64,
    pub avg_decode_tok_s: f64,
    pub prefix_cache_hits: u64,
    pub prefix_cache_hit_rate: f64,
}
