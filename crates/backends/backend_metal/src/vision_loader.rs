//! Vision encoder weight loader for Metal GPU.
//!
//! Loads `model.visual.*` tensors from safetensors into Metal GPU buffers.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use memmap2::Mmap;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use herbert_core::error::{HerbertError, Result};
use herbert_vision::config::VisionConfig;
use herbert_vision::encoder::compute_vision_rot_inv_freq;
use safetensors::SafeTensors;
use tracing::{debug, info};

use crate::memory::MetalBuffer;
use crate::vision_model::*;

struct VisionTensorStore {
    shard_data: Vec<Mmap>,
    weight_to_shard: HashMap<String, usize>,
}

#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

impl VisionTensorStore {
    fn load(model_dir: &Path) -> Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let index_data = fs::read(&index_path)?;
            let index: SafetensorsIndex = serde_json::from_slice(&index_data)?;

            let mut shard_set = std::collections::HashSet::new();
            for (weight, shard) in &index.weight_map {
                if weight.contains("visual") {
                    shard_set.insert(shard.clone());
                }
            }

            let mut shard_map = HashMap::new();
            let mut shards = Vec::new();
            let mut shard_names: Vec<String> = shard_set.into_iter().collect();
            shard_names.sort();

            info!(num_shards = shard_names.len(), "Loading vision shards for Metal");
            for shard_name in &shard_names {
                let shard_path = model_dir.join(shard_name);
                let file = fs::File::open(&shard_path)?;
                let mmap = unsafe { Mmap::map(&file)? };
                let shard_id = shards.len();
                shards.push(mmap);
                shard_map.insert(shard_name.clone(), shard_id);
            }

            let mut weight_to_shard = HashMap::new();
            for (weight, shard_file) in &index.weight_map {
                if let Some(&shard_id) = shard_map.get(shard_file.as_str()) {
                    weight_to_shard.insert(weight.clone(), shard_id);
                }
            }

            Ok(Self { shard_data: shards, weight_to_shard })
        } else {
            let file = fs::File::open(model_dir.join("model.safetensors"))?;
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
            *self.weight_to_shard.get(name)
                .ok_or_else(|| HerbertError::ModelLoad(format!("Vision weight not found: {}", name)))?
        };

        let tensors = parsed.get(shard_idx).ok_or_else(|| {
            HerbertError::ModelLoad(format!("Invalid shard index for {}", name))
        })?;
        let view = tensors.tensor(name)
            .map_err(|e| HerbertError::ModelLoad(format!("Tensor {} not found: {}", name, e)))?;

        let dtype = view.dtype();
        let data = view.data();

        match dtype {
            safetensors::Dtype::F32 => {
                Ok(data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect())
            }
            safetensors::Dtype::BF16 => {
                Ok(data.chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect())
            }
            safetensors::Dtype::F16 => {
                Ok(data.chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        half_to_f32(bits)
                    })
                    .collect())
            }
            other => Err(HerbertError::ModelLoad(format!(
                "Unsupported dtype {:?} for vision tensor {}", other, name
            ))),
        }
    }

    fn load_buffer(
        &self,
        parsed: &[SafeTensors<'_>],
        name: &str,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<MetalBuffer> {
        let data = self.load_f32_tensor(parsed, name)?;
        MetalBuffer::from_f32(device, &data)
    }
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;

    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mantissa;
        let mut e = 0i32;
        while m & 0x400 == 0 { m <<= 1; e -= 1; }
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

fn load_linear(
    store: &VisionTensorStore,
    parsed: &[SafeTensors<'_>],
    device: &ProtocolObject<dyn MTLDevice>,
    weight_name: &str,
    bias_name: &str,
    in_features: usize,
    out_features: usize,
) -> Result<MetalVisionLinear> {
    let weight = store.load_buffer(parsed, weight_name, device)?;
    let bias = store.load_buffer(parsed, bias_name, device)?;
    Ok(MetalVisionLinear { weight, bias, in_features, out_features })
}

fn load_layer_norm(
    store: &VisionTensorStore,
    parsed: &[SafeTensors<'_>],
    device: &ProtocolObject<dyn MTLDevice>,
    weight_name: &str,
    bias_name: &str,
) -> Result<MetalVisionLayerNorm> {
    let weight = store.load_buffer(parsed, weight_name, device)?;
    let bias = store.load_buffer(parsed, bias_name, device)?;
    Ok(MetalVisionLayerNorm { weight, bias })
}

fn load_merger(
    store: &VisionTensorStore,
    parsed: &[SafeTensors<'_>],
    device: &ProtocolObject<dyn MTLDevice>,
    prefix: &str,
    config: &VisionConfig,
    use_postshuffle_norm: bool,
) -> Result<MetalVisionMerger> {
    let merged_dim = config.merger_hidden_size();
    let norm_dim = if use_postshuffle_norm { merged_dim } else { config.hidden_size };

    let norm = load_layer_norm(
        store, parsed, device,
        &format!("{}.norm.weight", prefix),
        &format!("{}.norm.bias", prefix),
    )?;
    // Validate norm dimension
    if norm.weight.size as usize != norm_dim * 4 {
        return Err(HerbertError::ModelLoad(format!(
            "Merger norm weight size {} != expected {}", norm.weight.size, norm_dim * 4
        )));
    }
    let fc1 = load_linear(
        store, parsed, device,
        &format!("{}.linear_fc1.weight", prefix),
        &format!("{}.linear_fc1.bias", prefix),
        merged_dim, merged_dim,
    )?;
    let fc2 = load_linear(
        store, parsed, device,
        &format!("{}.linear_fc2.weight", prefix),
        &format!("{}.linear_fc2.bias", prefix),
        merged_dim, config.out_hidden_size,
    )?;
    Ok(MetalVisionMerger { norm, fc1, fc2, use_postshuffle_norm })
}

pub fn load_vision_model(
    model_dir: &Path,
    device: &ProtocolObject<dyn MTLDevice>,
    config: &VisionConfig,
) -> Result<MetalVisionModel> {
    let store = VisionTensorStore::load(model_dir)?;
    let parsed = store.parse_shards()?;
    let dim = config.hidden_size;
    let patch_dim = config.patch_dim;

    debug!("Loading vision patch embedding to GPU");
    let patch_embed = load_linear(
        &store, &parsed, device,
        "model.visual.patch_embed.proj.weight",
        "model.visual.patch_embed.proj.bias",
        patch_dim, dim,
    )?;

    debug!("Loading vision position embedding to GPU");
    let pos_embed = store.load_buffer(&parsed, "model.visual.pos_embed.weight", device)?;

    let rot_inv_freq = compute_vision_rot_inv_freq(config.head_dim / 2, 10000.0);

    info!(num_blocks = config.num_layers, "Loading vision blocks to GPU");
    let mut blocks = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let prefix = format!("model.visual.blocks.{}", i);
        let norm1 = load_layer_norm(
            &store, &parsed, device,
            &format!("{}.norm1.weight", prefix),
            &format!("{}.norm1.bias", prefix),
        )?;
        let norm2 = load_layer_norm(
            &store, &parsed, device,
            &format!("{}.norm2.weight", prefix),
            &format!("{}.norm2.bias", prefix),
        )?;
        let qkv = load_linear(
            &store, &parsed, device,
            &format!("{}.attn.qkv.weight", prefix),
            &format!("{}.attn.qkv.bias", prefix),
            dim, 3 * dim,
        )?;
        let proj = load_linear(
            &store, &parsed, device,
            &format!("{}.attn.proj.weight", prefix),
            &format!("{}.attn.proj.bias", prefix),
            dim, dim,
        )?;
        let fc1 = load_linear(
            &store, &parsed, device,
            &format!("{}.mlp.linear_fc1.weight", prefix),
            &format!("{}.mlp.linear_fc1.bias", prefix),
            dim, config.intermediate_size,
        )?;
        let fc2 = load_linear(
            &store, &parsed, device,
            &format!("{}.mlp.linear_fc2.weight", prefix),
            &format!("{}.mlp.linear_fc2.bias", prefix),
            config.intermediate_size, dim,
        )?;
        blocks.push(MetalVisionBlock { norm1, norm2, qkv, proj, fc1, fc2 });
    }
    info!("Vision blocks loaded to GPU");

    debug!("Loading final merger to GPU");
    let merger = load_merger(&store, &parsed, device, "model.visual.merger", config, false)?;

    debug!("Loading deepstack mergers to GPU");
    let mut deepstack_mergers = Vec::new();
    for i in 0..config.deepstack_visual_indexes.len() {
        let ds_merger = load_merger(
            &store, &parsed, device,
            &format!("model.visual.deepstack_merger_list.{}", i),
            config, true,
        )?;
        deepstack_mergers.push(ds_merger);
    }
    info!(num_deepstack = deepstack_mergers.len(), "Vision model loaded to GPU");

    Ok(MetalVisionModel {
        patch_embed, pos_embed, blocks, merger, deepstack_mergers,
        config: config.clone(), rot_inv_freq,
    })
}
