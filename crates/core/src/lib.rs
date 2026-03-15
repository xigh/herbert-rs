//! Core types and traits for Herbert.
//!
//! Defines the [`Backend`] trait, model [`Config`], KV cache handle, and common
//! tensor operations used by all backend implementations.

pub mod backend;
pub mod cpu_detection;

pub mod config;
pub mod decode_utils;
pub mod error;
pub mod kv_cache;
pub mod moe_stats;
pub mod sampler;
pub mod tensor;

pub use backend::{Backend, DecodeOutput, LoadOpts, PrefillOutput, RunOpts, VerifyOutput};

pub use config::{Config, ModelFamily};
pub use error::HerbertError;
pub use kv_cache::KvHandle;
