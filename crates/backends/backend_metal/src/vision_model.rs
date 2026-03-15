//! Metal vision encoder model structures.

use crate::memory::MetalBuffer;
use herbert_vision::config::VisionConfig;

pub struct MetalVisionLinear {
    pub weight: MetalBuffer, // [out_features, in_features] f32
    pub bias: MetalBuffer,   // [out_features] f32
    pub in_features: usize,
    pub out_features: usize,
}

pub struct MetalVisionLayerNorm {
    pub weight: MetalBuffer, // [dim] f32
    pub bias: MetalBuffer,   // [dim] f32
}

pub struct MetalVisionBlock {
    pub norm1: MetalVisionLayerNorm,
    pub norm2: MetalVisionLayerNorm,
    pub qkv: MetalVisionLinear,  // [3*dim, dim]
    pub proj: MetalVisionLinear, // [dim, dim]
    pub fc1: MetalVisionLinear,  // [intermediate, dim]
    pub fc2: MetalVisionLinear,  // [dim, intermediate]
}

pub struct MetalVisionMerger {
    pub norm: MetalVisionLayerNorm,
    pub fc1: MetalVisionLinear,
    pub fc2: MetalVisionLinear,
    pub use_postshuffle_norm: bool,
}

pub struct MetalVisionModel {
    pub patch_embed: MetalVisionLinear,
    pub pos_embed: MetalBuffer, // [num_position_embeddings * hidden_size] f32
    pub blocks: Vec<MetalVisionBlock>,
    pub merger: MetalVisionMerger,
    pub deepstack_mergers: Vec<MetalVisionMerger>,
    pub config: VisionConfig,
    pub rot_inv_freq: Vec<f32>, // CPU-side for RoPE precompute
}
