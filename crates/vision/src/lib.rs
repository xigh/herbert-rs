//! Vision encoder for Qwen3-VL models.
//!
//! This crate implements the Qwen3-VL vision pipeline:
//! Conv3D patch embed, ViT blocks, spatial merge, DeepStack.

pub mod config;
pub mod encoder;
pub mod image_process;
pub mod loader;
pub mod pixtral;
pub mod pixtral_config;
