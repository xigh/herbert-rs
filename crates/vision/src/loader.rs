//! Vision encoder weight loader from safetensors.
//!
//! Loads `model.visual.*` tensors and constructs a `VisionEncoder`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use memmap2::Mmap;
use herbert_core::error::{HerbertError, Result};
use safetensors::SafeTensors;
use tracing::{debug, info};

use crate::config::VisionConfig;
use crate::encoder::*;

/// Load a vision encoder with progress indicators.
pub fn load_vision_encoder_with_progress(model_dir: &Path, config: &VisionConfig) -> Result<VisionEncoder> {
    load_vision_encoder_inner(model_dir, config, true)
}

fn load_vision_encoder_inner(model_dir: &Path, config: &VisionConfig, show_progress: bool) -> Result<VisionEncoder> {
    let store = VisionTensorStore::load(model_dir, show_progress)?;
    let parsed = store.parse_shards()?;
    load_from_tensors(&store, &parsed, config, show_progress)
}

// ─── Internal Tensor Store ───────────────────────────────────────────

struct VisionTensorStore {
    /// Memory-mapped shard data — OS pages in only the accessed regions.
    shard_data: Vec<Mmap>,
    weight_to_shard: HashMap<String, usize>,
}

/// Index structure for multi-shard models.
#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

impl VisionTensorStore {
    fn load(model_dir: &Path, show_progress: bool) -> Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let index_data = fs::read(&index_path)?;
            let index: SafetensorsIndex = serde_json::from_slice(&index_data)?;

            // Find unique shards that contain vision weights
            let mut shard_set = std::collections::HashSet::new();
            for (weight, shard) in &index.weight_map {
                if weight.contains("visual") || weight.contains("model.visual") {
                    shard_set.insert(shard.clone());
                }
            }

            let mut shard_map = HashMap::new();
            let mut shards = Vec::new();
            let mut shard_names: Vec<String> = shard_set.into_iter().collect();
            shard_names.sort();

            let pb = if show_progress {
                use indicatif::{ProgressBar, ProgressStyle};
                let pb = ProgressBar::new(shard_names.len() as u64);
                pb.set_style(
                    ProgressStyle::with_template("  Loading vision shards [{bar:30}] {pos}/{len}")
                        .expect("valid progress template")
                        .progress_chars("█░░"),
                );
                Some(pb)
            } else {
                None
            };

            info!(num_shards = shard_names.len(), "Loading vision shards");
            for shard_name in &shard_names {
                let shard_path = model_dir.join(shard_name);
                let file_size = fs::metadata(&shard_path)
                    .map(|m| m.len())
                    .unwrap_or(0);
                let size_mb = file_size as f64 / (1024.0 * 1024.0);
                info!(shard = %shard_name, size_mb = format!("{:.0}", size_mb), "Mapping vision shard");
                let file = fs::File::open(&shard_path)?;
                // SAFETY: the file is read-only and not modified while mapped.
                let mmap = unsafe { Mmap::map(&file)? };
                debug!(shard = %shard_name, "Vision shard mapped");
                let shard_id = shards.len();
                shards.push(mmap);
                shard_map.insert(shard_name.clone(), shard_id);
                if let Some(ref pb) = pb {
                    pb.inc(1);
                }
            }

            if let Some(pb) = pb {
                pb.finish_and_clear();
            }

            let mut weight_to_shard = HashMap::new();
            for (weight, shard_file) in &index.weight_map {
                if let Some(&shard_id) = shard_map.get(shard_file.as_str()) {
                    weight_to_shard.insert(weight.clone(), shard_id);
                }
            }

            Ok(Self {
                shard_data: shards,
                weight_to_shard,
            })
        } else {
            // Single shard
            let file = fs::File::open(model_dir.join("model.safetensors"))?;
            // SAFETY: the file is read-only and not modified while mapped.
            let mmap = unsafe { Mmap::map(&file)? };
            Ok(Self {
                shard_data: vec![mmap],
                weight_to_shard: HashMap::new(),
            })
        }
    }

    fn parse_shards(&self) -> Result<Vec<SafeTensors<'_>>> {
        let mut parsed = Vec::with_capacity(self.shard_data.len());
        for data in &self.shard_data {
            let tensors = SafeTensors::deserialize(&data[..])
                .map_err(|e| HerbertError::ModelLoad(e.to_string()))?;
            parsed.push(tensors);
        }
        Ok(parsed)
    }

    fn load_f32_tensor(&self, parsed: &[SafeTensors<'_>], name: &str) -> Result<Vec<f32>> {
        let shard_idx = if self.shard_data.len() == 1 {
            0
        } else {
            *self
                .weight_to_shard
                .get(name)
                .ok_or_else(|| HerbertError::ModelLoad(format!("Vision weight not found: {}", name)))?
        };

        let tensors = parsed.get(shard_idx).ok_or_else(|| {
            HerbertError::ModelLoad(format!("Invalid shard index for {}", name))
        })?;
        let view = tensors
            .tensor(name)
            .map_err(|e| HerbertError::ModelLoad(format!("Tensor {} not found: {}", name, e)))?;

        let dtype = view.dtype();
        let data = view.data();

        match dtype {
            safetensors::Dtype::F32 => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                Ok(floats)
            }
            safetensors::Dtype::BF16 => {
                let floats: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                Ok(floats)
            }
            safetensors::Dtype::F16 => {
                let floats: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        half_to_f32(bits)
                    })
                    .collect();
                Ok(floats)
            }
            other => Err(HerbertError::ModelLoad(format!(
                "Unsupported dtype {:?} for vision tensor {}",
                other, name
            ))),
        }
    }
}

/// Convert f16 bits to f32.
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;

    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let mut m = mantissa;
        let mut e = 0i32;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3ff;
        let f32_exp = ((127 - 15 + 1 + e) as u32) & 0xff;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }

    if exp == 31 {
        if mantissa == 0 {
            return f32::from_bits((sign << 31) | (0xff << 23));
        }
        return f32::from_bits((sign << 31) | (0xff << 23) | (mantissa << 13));
    }

    let f32_exp = (exp + 127 - 15) & 0xff;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}

// ─── Load Helpers ────────────────────────────────────────────────────

fn load_linear(
    store: &VisionTensorStore,
    parsed: &[SafeTensors<'_>],
    weight_name: &str,
    bias_name: &str,
    in_features: usize,
    out_features: usize,
) -> Result<Linear> {
    let weight = store.load_f32_tensor(parsed, weight_name)?;
    let bias = store.load_f32_tensor(parsed, bias_name)?;
    if weight.len() != out_features * in_features {
        return Err(HerbertError::ModelLoad(format!(
            "{}: expected {}x{}={} elements, got {}",
            weight_name, out_features, in_features,
            out_features * in_features, weight.len()
        )));
    }
    if bias.len() != out_features {
        return Err(HerbertError::ModelLoad(format!(
            "{}: expected {} elements, got {}",
            bias_name, out_features, bias.len()
        )));
    }
    Ok(Linear {
        weight,
        bias,
        in_features,
        out_features,
    })
}

fn load_layer_norm(
    store: &VisionTensorStore,
    parsed: &[SafeTensors<'_>],
    weight_name: &str,
    bias_name: &str,
    dim: usize,
) -> Result<LayerNorm> {
    let weight = store.load_f32_tensor(parsed, weight_name)?;
    let bias = store.load_f32_tensor(parsed, bias_name)?;
    if weight.len() != dim || bias.len() != dim {
        return Err(HerbertError::ModelLoad(format!(
            "LayerNorm dimension mismatch: weight={}, bias={}, expected dim={}",
            weight.len(), bias.len(), dim
        )));
    }
    Ok(LayerNorm {
        weight,
        bias,
        dim,
        eps: 1e-6, // Qwen2-VL/Qwen3-VL standard LayerNorm eps
    })
}

fn load_merger(
    store: &VisionTensorStore,
    parsed: &[SafeTensors<'_>],
    prefix: &str,
    config: &VisionConfig,
    use_postshuffle_norm: bool,
) -> Result<PatchMerger> {
    let merged_dim = config.merger_hidden_size();
    let norm_dim = if use_postshuffle_norm {
        merged_dim
    } else {
        config.hidden_size
    };

    let norm = load_layer_norm(
        store, parsed,
        &format!("{}.norm.weight", prefix),
        &format!("{}.norm.bias", prefix),
        norm_dim,
    )?;
    let fc1 = load_linear(
        store, parsed,
        &format!("{}.linear_fc1.weight", prefix),
        &format!("{}.linear_fc1.bias", prefix),
        merged_dim,
        merged_dim,
    )?;
    let fc2 = load_linear(
        store, parsed,
        &format!("{}.linear_fc2.weight", prefix),
        &format!("{}.linear_fc2.bias", prefix),
        merged_dim,
        config.out_hidden_size,
    )?;

    Ok(PatchMerger {
        norm,
        fc1,
        fc2,
        use_postshuffle_norm,
    })
}

// ─── Main Load Function ──────────────────────────────────────────────

fn load_from_tensors(
    store: &VisionTensorStore,
    parsed: &[SafeTensors<'_>],
    config: &VisionConfig,
    show_progress: bool,
) -> Result<VisionEncoder> {
    let dim = config.hidden_size;
    let patch_dim = config.patch_dim;

    // Patch embedding (Conv3D weight reshaped to linear: [embed_dim, patch_dim])
    debug!("Loading patch embedding");
    let patch_embed = load_linear(
        store, parsed,
        "model.visual.patch_embed.proj.weight",
        "model.visual.patch_embed.proj.bias",
        patch_dim,
        dim,
    )?;

    // Position embedding: [num_position_embeddings, hidden_size]
    debug!("Loading position embedding");
    let pos_embed = store.load_f32_tensor(parsed, "model.visual.pos_embed.weight")?;
    let expected_pos_len = config.num_position_embeddings * config.hidden_size;
    if pos_embed.len() != expected_pos_len {
        return Err(HerbertError::ModelLoad(format!(
            "pos_embed size mismatch: got {}, expected {}",
            pos_embed.len(), expected_pos_len
        )));
    }

    // 2D rotary inv_freq (computed, not loaded)
    let rot_inv_freq = compute_vision_rot_inv_freq(config.head_dim / 2, 10000.0);

    // Transformer blocks
    let pb = if show_progress {
        use indicatif::{ProgressBar, ProgressStyle};
        let pb = ProgressBar::new(config.num_layers as u64);
        pb.set_style(
            ProgressStyle::with_template("  Loading vision [{bar:30}] {pos}/{len} blocks")
                .expect("valid progress template")
                .progress_chars("█░░"),
        );
        Some(pb)
    } else {
        None
    };
    info!(num_blocks = config.num_layers, "Loading vision blocks");
    let mut blocks = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("model.visual.blocks.{}", i);

        let norm1 = load_layer_norm(
            store, parsed,
            &format!("{}.norm1.weight", prefix),
            &format!("{}.norm1.bias", prefix),
            dim,
        )?;
        let norm2 = load_layer_norm(
            store, parsed,
            &format!("{}.norm2.weight", prefix),
            &format!("{}.norm2.bias", prefix),
            dim,
        )?;

        let attn = VisionAttention {
            qkv: load_linear(
                store, parsed,
                &format!("{}.attn.qkv.weight", prefix),
                &format!("{}.attn.qkv.bias", prefix),
                dim,
                3 * dim,
            )?,
            proj: load_linear(
                store, parsed,
                &format!("{}.attn.proj.weight", prefix),
                &format!("{}.attn.proj.bias", prefix),
                dim,
                dim,
            )?,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            scaling: (config.head_dim as f32).powf(-0.5),
        };

        let mlp = VisionMLP {
            fc1: load_linear(
                store, parsed,
                &format!("{}.mlp.linear_fc1.weight", prefix),
                &format!("{}.mlp.linear_fc1.bias", prefix),
                dim,
                config.intermediate_size,
            )?,
            fc2: load_linear(
                store, parsed,
                &format!("{}.mlp.linear_fc2.weight", prefix),
                &format!("{}.mlp.linear_fc2.bias", prefix),
                config.intermediate_size,
                dim,
            )?,
        };

        blocks.push(VisionBlock {
            norm1,
            norm2,
            attn,
            mlp,
        });

        if let Some(ref pb) = pb {
            pb.set_position((i + 1) as u64);
        }
    }

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }
    info!("Vision blocks loaded");

    // Final merger (use_postshuffle_norm=false)
    debug!("Loading final merger");
    let merger = load_merger(
        store, parsed, "model.visual.merger", config, false,
    )?;

    // DeepStack mergers (use_postshuffle_norm=true)
    debug!(count = config.deepstack_visual_indexes.len(), "Loading deepstack mergers");
    let mut deepstack_mergers = Vec::new();
    for i in 0..config.deepstack_visual_indexes.len() {
        let ds_merger = load_merger(
            store, parsed,
            &format!("model.visual.deepstack_merger_list.{}", i),
            config,
            true,
        )?;
        deepstack_mergers.push(ds_merger);
    }
    info!(num_deepstack = deepstack_mergers.len(), "Vision encoder ready");

    Ok(VisionEncoder {
        config: config.clone(),
        patch_embed,
        pos_embed,
        rot_inv_freq,
        blocks,
        merger,
        deepstack_mergers,
    })
}
