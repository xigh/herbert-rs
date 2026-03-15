//! Pixtral vision encoder configuration parsed from `config.json`.

use herbert_core::error::{HerbertError, Result};
use std::path::Path;

/// Pixtral vision encoder configuration (for Mistral3/Devstral).
#[derive(Debug, Clone)]
pub struct PixtralVisionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub rope_theta: f32,
    pub spatial_merge_size: usize,
    pub text_hidden_size: usize,
}

impl PixtralVisionConfig {
    /// Parse from a Mistral3 config.json that has `vision_config` + `text_config`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let root: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| HerbertError::Config(format!("Failed to parse config.json: {}", e)))?;

        let vc = root.get("vision_config").ok_or_else(|| {
            HerbertError::Config("config.json has no vision_config section".into())
        })?;

        let hidden_size = vc.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(1024) as usize;
        let num_heads = vc.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
        let num_layers = vc.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(24) as usize;
        let intermediate_size = vc.get("intermediate_size").and_then(|v| v.as_u64()).unwrap_or(4096) as usize;
        let patch_size = vc.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(14) as usize;
        let image_size = vc.get("image_size").and_then(|v| v.as_u64()).unwrap_or(1540) as usize;
        let rope_theta = vc.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(10000.0) as f32;

        if num_heads == 0 {
            return Err(HerbertError::Config("pixtral vision num_heads must be > 0".into()));
        }
        let head_dim = hidden_size / num_heads;

        // spatial_merge_size from vision_config or top-level
        let spatial_merge_size = vc.get("spatial_merge_size")
            .and_then(|v| v.as_u64())
            .or_else(|| root.get("spatial_merge_size").and_then(|v| v.as_u64()))
            .unwrap_or(2) as usize;

        // text_hidden_size from text_config.hidden_size
        let text_hidden_size = root.get("text_config")
            .and_then(|tc| tc.get("hidden_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(5120) as usize;

        Ok(Self {
            hidden_size,
            num_heads,
            head_dim,
            num_layers,
            intermediate_size,
            patch_size,
            image_size,
            rope_theta,
            spatial_merge_size,
            text_hidden_size,
        })
    }
}
