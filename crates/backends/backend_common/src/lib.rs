#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

//! Shared backend helpers for CPU inference backends.

pub mod cpu_backend;
pub mod attention_common;
pub mod autotune;
pub mod execution_phase;
pub mod generic_attention;
pub mod hugepages;
pub mod generic_layer;
pub mod generic_mlp;
pub mod generic_moe;
pub mod expert_pool;
pub mod generic_model;
pub mod kernels;
pub mod bf16_convert;
pub mod h2o;
pub mod kv_cache;
pub mod linear_ops;
pub mod loader_common;
pub mod mrope;
pub mod numa;
pub mod position_ids;
pub mod prefix_cache;
pub mod prefill_progress;
pub mod profiler;
pub mod shared_prefix_cache;
pub mod progress;
pub mod thread_pool;
pub mod topology;
pub mod weight_cache;
pub mod speculative;
pub mod eagle3;
pub mod eagle3_speculative;

#[cfg(feature = "bench-decode-snapshot")]
pub mod snapshot;

