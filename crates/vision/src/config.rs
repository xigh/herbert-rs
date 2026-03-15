//! Vision encoder configuration parsed from `config.json`.

use herbert_core::error::{HerbertError, Result};
use serde::Deserialize;
use std::path::Path;

/// Vision encoder configuration.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub out_hidden_size: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    /// Number of position embeddings (e.g. 2304 = 48*48).
    pub num_position_embeddings: usize,
    /// Grid side length for position embedding (sqrt of num_position_embeddings).
    pub num_grid_per_side: usize,
    /// DeepStack layer indices (e.g. [5, 11, 17]).
    pub deepstack_visual_indexes: Vec<usize>,
    /// Patch dimension = in_channels * temporal_patch_size * patch_size * patch_size.
    pub patch_dim: usize,
}

impl VisionConfig {
    /// Validate configuration consistency.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(HerbertError::Config("vision hidden_size must be > 0".into()));
        }
        if self.num_heads == 0 || !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(HerbertError::Config(format!(
                "vision hidden_size ({}) must be divisible by num_heads ({})",
                self.hidden_size, self.num_heads
            )));
        }
        if self.patch_size == 0 || self.temporal_patch_size == 0 {
            return Err(HerbertError::Config("patch sizes must be > 0".into()));
        }
        if self.spatial_merge_size == 0 {
            return Err(HerbertError::Config("spatial_merge_size must be > 0".into()));
        }
        let expected_grid = self.num_grid_per_side * self.num_grid_per_side;
        if expected_grid != self.num_position_embeddings {
            return Err(HerbertError::Config(format!(
                "num_grid_per_side^2 ({}) != num_position_embeddings ({})",
                expected_grid, self.num_position_embeddings
            )));
        }
        Ok(())
    }

    /// Merger hidden size = hidden_size * spatial_merge_size^2.
    pub fn merger_hidden_size(&self) -> usize {
        self.hidden_size * self.spatial_merge_size * self.spatial_merge_size
    }
}

/// Raw JSON structure for the vision_config section.
#[derive(Deserialize)]
struct VisionConfigJson {
    #[serde(default = "default_hidden_size")]
    hidden_size: usize,
    #[serde(default = "default_num_heads")]
    num_heads: usize,
    #[serde(default = "default_num_layers")]
    depth: usize,
    #[serde(default = "default_intermediate_size")]
    intermediate_size: usize,
    #[serde(default = "default_out_hidden_size")]
    out_hidden_size: usize,
    #[serde(default = "default_in_channels")]
    in_channels: usize,
    #[serde(default = "default_patch_size")]
    patch_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    temporal_patch_size: usize,
    #[serde(default = "default_spatial_merge_size")]
    spatial_merge_size: usize,
    #[serde(default = "default_num_position_embeddings")]
    num_position_embeddings: usize,
    #[serde(default)]
    deepstack_visual_indexes: Option<Vec<usize>>,
}

fn default_hidden_size() -> usize { 1024 }
fn default_num_heads() -> usize { 16 }
fn default_num_layers() -> usize { 24 }
fn default_intermediate_size() -> usize { 4096 }
fn default_out_hidden_size() -> usize { 2048 }
fn default_in_channels() -> usize { 3 }
fn default_patch_size() -> usize { 16 }
fn default_temporal_patch_size() -> usize { 2 }
fn default_spatial_merge_size() -> usize { 2 }
fn default_num_position_embeddings() -> usize { 2304 }

impl VisionConfig {
    /// Parse from the `vision_config` section of a HuggingFace config.json.
    pub(crate) fn from_json_value(value: &serde_json::Value) -> Result<Self> {
        let json: VisionConfigJson = serde_json::from_value(value.clone())
            .map_err(|e| HerbertError::Config(format!("Failed to parse vision_config: {}", e)))?;

        if json.num_heads == 0 {
            return Err(HerbertError::Config("vision num_heads must be > 0".into()));
        }
        let head_dim = json.hidden_size / json.num_heads;
        if json.num_position_embeddings == 0 {
            return Err(HerbertError::Config("vision num_position_embeddings must be > 0".into()));
        }
        let num_grid_per_side = (json.num_position_embeddings as f64).sqrt() as usize;
        let patch_dim = json.in_channels * json.temporal_patch_size * json.patch_size * json.patch_size;
        let deepstack_visual_indexes = json.deepstack_visual_indexes.unwrap_or_else(|| vec![5, 11, 17]);

        let config = Self {
            hidden_size: json.hidden_size,
            num_heads: json.num_heads,
            head_dim,
            num_layers: json.depth,
            intermediate_size: json.intermediate_size,
            out_hidden_size: json.out_hidden_size,
            in_channels: json.in_channels,
            patch_size: json.patch_size,
            temporal_patch_size: json.temporal_patch_size,
            spatial_merge_size: json.spatial_merge_size,
            num_position_embeddings: json.num_position_embeddings,
            num_grid_per_side,
            deepstack_visual_indexes,
            patch_dim,
        };
        config.validate()?;
        Ok(config)
    }

    /// Load from a config.json file path. Extracts the `vision_config` section.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let root: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| HerbertError::Config(format!("Failed to parse config.json: {}", e)))?;

        let vision_section = root.get("vision_config").ok_or_else(|| {
            HerbertError::Config("config.json has no vision_config section".into())
        })?;
        Self::from_json_value(vision_section)
    }
}
