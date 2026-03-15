//! Model configuration parsing.

use crate::error::{HerbertError, Result};
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// RoPE variant used by the model.
#[derive(Debug, Clone)]
pub enum RopeType {
    /// Standard RoPE (Qwen3).
    Standard,
    /// Linear RoPE scaling: inv_freq /= factor.
    Linear { factor: f32 },
    /// YARN RoPE: wavelength-dependent frequency scaling (Devstral).
    Yarn {
        factor: f32,
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
    },
}

/// Norm type used by the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// RMS normalization (Qwen3).
    RMSNorm,
    /// Gemma-style RMS normalization: (1 + weight) * rms_norm(input). Used by Qwen3.5.
    GemmaRMSNorm,
}

/// KV cache quantization type (runtime selection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuantType {
    /// Full precision f32 (highest quality, most memory).
    F32,
    /// BF16 (default, good quality/memory trade-off).
    BF16,
    /// INT8 symmetric per-position per-head quantization (~50% of BF16 memory).
    INT8,
    /// INT4 symmetric per-position per-head quantization (~25% of BF16 memory).
    /// Half-split nibble packing: byte[i] = elem[i] | (elem[i+16] << 4).
    INT4,
}

impl KvQuantType {
    /// Bytes per scalar element in this format.
    /// For INT4, returns 1 (conservative estimate; actual is 0.5).
    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::BF16 => 2,
            Self::INT8 => 1,
            Self::INT4 => 1,
        }
    }
}

impl std::fmt::Display for KvQuantType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::BF16 => write!(f, "bf16"),
            Self::INT8 => write!(f, "int8"),
            Self::INT4 => write!(f, "int4"),
        }
    }
}

impl std::str::FromStr for KvQuantType {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "f32" => Ok(Self::F32),
            "bf16" => Ok(Self::BF16),
            "int8" => Ok(Self::INT8),
            "int4" => Ok(Self::INT4),
            _ => Err(format!("unknown KV quant type '{}', expected f32/bf16/int8/int4", s)),
        }
    }
}

/// Supported model families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Qwen3 text and MoE variants.
    Qwen3,
    /// Mistral3 / Devstral / Magistral (text + Pixtral vision).
    Mistral3,
}

impl ModelFamily {
    /// Classify a `model_type` string from config.json.
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "qwen3" | "qwen3_moe" | "qwen3_vl" | "qwen3_vl_moe"
            | "qwen3_vl_text" | "qwen3_vl_moe_text" => Some(Self::Qwen3),
            "mistral3" | "ministral3" | "mistral" | "mixtral" => Some(Self::Mistral3),
            _ => None,
        }
    }
}

/// Model configuration (supports Qwen3 and Qwen3.5 variants).
#[derive(Debug, Clone)]
pub struct Config {
    pub model_type: String,
    pub model_family: ModelFamily,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    /// Whether the model uses per-head QK RMS norms (Qwen3: yes).
    pub has_qk_norm: bool,
    /// Whether Q/K/V projections have bias terms (Qwen3: no).
    pub has_attn_bias: bool,
    /// Whether O-projection has bias (always false for Qwen3).
    pub has_o_bias: bool,
    /// Norm type: RMSNorm (Qwen3) or GemmaRMSNorm (Qwen3.5).
    pub norm_type: NormType,
    /// Number of dimensions that get RoPE (default = head_dim).
    pub rotary_ndims: usize,
    /// EOS token id from config.json (None → fallback to 151643)
    pub eos_token_id: Option<u32>,
    // MoE fields (None for dense models)
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub norm_topk_prob: bool,
    pub decoder_sparse_step: usize,
    pub mlp_only_layers: Vec<usize>,
    /// MRoPE section sizes [T, H, W] for VL models (e.g., [24, 20, 20]).
    /// None for text-only models. sum(mrope_section) == head_dim / 2.
    pub mrope_section: Option<[usize; 3]>,
    /// Image placeholder token id for VL models (e.g. 151655).
    /// None for text-only models.
    pub image_token_id: Option<u32>,
    /// Layer indices using full attention (empty = all attention, i.e. Qwen).
    pub full_attn_idxs: Vec<usize>,
    /// Convolution cache size per layer (0 for pure-attention models like Qwen).
    pub conv_l_cache: usize,
    /// RoPE variant (Standard or Linear).
    pub rope_type: RopeType,
    /// Explicit attention scale (query_pre_attn_scalar^(-0.5)).
    /// None means use default 1/sqrt(head_dim).
    pub attn_scale: Option<f32>,
    /// Dual RoPE: sliding attention layers use standard RoPE with this theta.
    /// None for single-cache models.
    pub sliding_rope_theta: Option<f32>,
    /// Dual RoPE: linear scaling factor for full attention layers' RoPE.
    /// None for single-cache models.
    pub gemma3_rope_factor: Option<f32>,
    // GatedDeltaNet fields (Qwen3.5)
    /// Number of Q/K heads in linear attention layers.
    pub linear_num_key_heads: Option<usize>,
    /// Number of V heads in linear attention layers.
    pub linear_num_value_heads: Option<usize>,
    /// Dimension per K head.
    pub linear_key_head_dim: Option<usize>,
    /// Dimension per V head.
    pub linear_value_head_dim: Option<usize>,
    /// Conv1d kernel size (typically 4).
    pub linear_conv_kernel_dim: Option<usize>,
    /// Full attention interval: every Nth layer is full attention.
    pub full_attention_interval: Option<usize>,
    /// Whether the model has shared experts with a gating mechanism.
    pub has_shared_expert_gate: bool,
    /// Shared expert intermediate size.
    pub shared_expert_intermediate_size: Option<usize>,
    /// Whether attention uses output gating (q_proj fused Q+gate, sigmoid gating).
    pub attn_output_gate: bool,
}

#[derive(Deserialize)]
struct ConfigJson {
    #[serde(default = "default_model_type")]
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    /// Explicit in Qwen3.
    head_dim: Option<usize>,
    intermediate_size: Option<usize>,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    rms_norm_eps: Option<f64>,
    max_position_embeddings: usize,
    /// Qwen3 convention: defaults to true (LM head shares embedding weights).
    /// Missing field -> true; set to false for models with a separate lm_head.
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
    // MoE fields (absent for dense models)
    // Qwen3 uses num_experts, Mixtral uses num_local_experts
    #[serde(default, alias = "num_local_experts")]
    num_experts: Option<usize>,
    #[serde(default)]
    num_experts_per_tok: Option<usize>,
    #[serde(default)]
    moe_intermediate_size: Option<usize>,
    #[serde(default)]
    norm_topk_prob: bool,
    #[serde(default)]
    decoder_sparse_step: Option<usize>,
    #[serde(default)]
    mlp_only_layers: Vec<usize>,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
    /// Qwen3.5 uses rope_parameters for nested config (partial_rotary_factor, rope_theta).
    #[serde(default)]
    rope_parameters: Option<serde_json::Value>,
    #[serde(default)]
    full_attn_idxs: Vec<usize>,
    #[serde(default)]
    conv_l_cache: Option<usize>,
    #[serde(default)]
    num_dense_layers: Option<usize>,
    #[serde(default)]
    image_token_id: Option<serde_json::Value>,
    /// partial_rotary_factor (default 1.0 = full RoPE)
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
    /// query_pre_attn_scalar for explicit attention scaling
    #[serde(default)]
    query_pre_attn_scalar: Option<f64>,
    // GatedDeltaNet (Qwen3.5)
    #[serde(default)]
    linear_num_key_heads: Option<usize>,
    #[serde(default)]
    linear_num_value_heads: Option<usize>,
    #[serde(default)]
    linear_key_head_dim: Option<usize>,
    #[serde(default)]
    linear_value_head_dim: Option<usize>,
    #[serde(default)]
    linear_conv_kernel_dim: Option<usize>,
    #[serde(default)]
    full_attention_interval: Option<usize>,
    #[serde(default)]
    shared_expert_intermediate_size: Option<usize>,
    #[serde(default)]
    attn_output_gate: Option<bool>,
}

fn default_true() -> bool {
    true
}

fn default_model_type() -> String {
    "qwen3".to_string()
}

/// Extract eos_token_id from JSON value (can be a number or an array like [151645]).
fn parse_eos_token_id(val: Option<&serde_json::Value>) -> Option<u32> {
    match val {
        Some(serde_json::Value::Number(n)) => n.as_u64().and_then(|v| u32::try_from(v).ok()),
        Some(serde_json::Value::Array(arr)) => {
            arr.first().and_then(|v| v.as_u64()).and_then(|v| u32::try_from(v).ok())
        }
        _ => None,
    }
}

impl Config {
    /// Load configuration from a JSON file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Self::from_json(&content)
    }

    /// EOS token ID (from config.json, or default 151643).
    #[inline]
    pub fn eos_token(&self) -> u32 {
        self.eos_token_id.unwrap_or(151643)
    }

    /// Whether this is a Mixture-of-Experts model.
    #[inline]
    pub fn is_moe(&self) -> bool {
        self.num_experts.is_some()
    }

    /// Whether a given layer uses MoE (vs dense MLP).
    #[inline]
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.is_moe()
            && !self.mlp_only_layers.contains(&layer_idx)
            && (self.decoder_sparse_step == 0 || layer_idx.is_multiple_of(self.decoder_sparse_step))
    }

    /// Number of KV groups (for GQA/MQA)
    #[inline]
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Q projection dimension (num_attention_heads * head_dim)
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// KV projection dimension (num_key_value_heads * head_dim)
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Whether a given layer uses full attention (vs convolution).
    /// Returns true for Qwen models (full_attn_idxs is empty -> all layers are attention).
    #[inline]
    pub fn is_attn_layer(&self, layer_idx: usize) -> bool {
        self.full_attn_idxs.is_empty() || self.full_attn_idxs.contains(&layer_idx)
    }

    /// Whether this model uses hybrid conv+attention layers (unused, kept for API compat).
    #[inline]
    pub fn is_conv_model(&self) -> bool {
        false
    }

    /// Whether a given layer uses Gated DeltaNet (linear attention).
    #[inline]
    pub fn is_deltanet_layer(&self, layer_idx: usize) -> bool {
        self.full_attention_interval.is_some() && !self.is_attn_layer(layer_idx)
    }

    /// Computed DeltaNet conv dimension: num_kv_heads * (key_dim + value_dim).
    #[inline]
    pub fn deltanet_conv_dim(&self) -> usize {
        let nkv = self.linear_num_value_heads.unwrap_or(0);
        let kd = self.linear_key_head_dim.unwrap_or(0);
        let vd = self.linear_value_head_dim.unwrap_or(0);
        nkv * (kd + vd)
    }

    /// Recurrent state size per DeltaNet layer: num_kv_heads * key_dim * value_dim.
    #[inline]
    pub fn deltanet_recurrent_size(&self) -> usize {
        let nkv = self.linear_num_value_heads.unwrap_or(0);
        let kd = self.linear_key_head_dim.unwrap_or(0);
        let vd = self.linear_value_head_dim.unwrap_or(0);
        nkv * kd * vd
    }

    /// Whether this is a Vision-Language model.
    /// Qwen3-VL has mrope_section; Mistral3-VL has image_token_id.
    #[inline]
    pub fn is_vl(&self) -> bool {
        self.mrope_section.is_some() || self.image_token_id.is_some()
    }

    /// Validate configuration values for consistency.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(HerbertError::Config("hidden_size must be > 0".into()));
        }
        if self.num_layers == 0 {
            return Err(HerbertError::Config("num_layers must be > 0".into()));
        }
        if self.vocab_size == 0 {
            return Err(HerbertError::Config("vocab_size must be > 0".into()));
        }
        if self.num_attention_heads == 0 {
            return Err(HerbertError::Config("num_attention_heads must be > 0".into()));
        }
        if self.num_key_value_heads == 0 {
            return Err(HerbertError::Config("num_key_value_heads must be > 0".into()));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(HerbertError::Config(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return Err(HerbertError::Config(format!(
                "head_dim ({}) must be > 0 and even (required by RoPE)",
                self.head_dim
            )));
        }
        // intermediate_size is only required when there are dense MLP layers
        let has_dense_layers = (0..self.num_layers).any(|i| !self.is_moe_layer(i));
        if self.intermediate_size == 0 && has_dense_layers {
            return Err(HerbertError::Config("intermediate_size must be > 0".into()));
        }
        if self.rms_norm_eps <= 0.0 {
            return Err(HerbertError::Config("rms_norm_eps must be > 0".into()));
        }
        if self.rope_theta <= 0.0 {
            return Err(HerbertError::Config("rope_theta must be > 0".into()));
        }
        if self.max_position_embeddings == 0 {
            return Err(HerbertError::Config("max_position_embeddings must be > 0".into()));
        }
        // MRoPE consistency: sum(mrope_section) must equal rotary_ndims / 2
        if let Some(section) = self.mrope_section {
            let sum: usize = section.iter().sum();
            let expected = self.rotary_ndims / 2;
            if sum != expected {
                return Err(HerbertError::Config(format!(
                    "mrope_section {:?} sums to {} but rotary_ndims/2 = {}",
                    section, sum, expected
                )));
            }
        }
        // MoE consistency
        if let Some(ne) = self.num_experts {
            if ne == 0 {
                return Err(HerbertError::Config("num_experts must be > 0".into()));
            }
            let nept = self.num_experts_per_tok.ok_or_else(|| {
                HerbertError::Config("num_experts_per_tok required when num_experts is set".into())
            })?;
            if nept == 0 || nept > ne {
                return Err(HerbertError::Config(format!(
                    "num_experts_per_tok ({}) must be in 1..=num_experts ({})",
                    nept, ne
                )));
            }
            let mis = self.moe_intermediate_size.ok_or_else(|| {
                HerbertError::Config("moe_intermediate_size required when num_experts is set".into())
            })?;
            if mis == 0 {
                return Err(HerbertError::Config("moe_intermediate_size must be > 0".into()));
            }
        }
        Ok(())
    }

    /// Parse configuration from a JSON string.
    pub fn from_json(content: &str) -> Result<Self> {
        // Handle nested VL configs: if `text_config` exists, merge it with
        // top-level `model_type` and `tie_word_embeddings` to form a flat config.
        let json_value: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| HerbertError::Config(format!("Failed to parse config.json: {}", e)))?;

        let effective_value = if let Some(text_config) = json_value.get("text_config") {
            let mut merged = text_config.clone();
            if let Some(obj) = merged.as_object_mut() {
                // Inject top-level model_type if not in text_config
                if !obj.contains_key("model_type") {
                    if let Some(mt) = json_value.get("model_type") {
                        obj.insert("model_type".to_string(), mt.clone());
                    }
                }
                // Inject top-level tie_word_embeddings if not in text_config
                if !obj.contains_key("tie_word_embeddings") {
                    if let Some(twe) = json_value.get("tie_word_embeddings") {
                        obj.insert("tie_word_embeddings".to_string(), twe.clone());
                    }
                }
                // Inject top-level rope_scaling if not in text_config
                if !obj.contains_key("rope_scaling") {
                    if let Some(rs) = json_value.get("rope_scaling") {
                        obj.insert("rope_scaling".to_string(), rs.clone());
                    }
                }
                // Inject top-level eos_token_id if not in text_config
                if !obj.contains_key("eos_token_id") {
                    if let Some(eos) = json_value.get("eos_token_id") {
                        obj.insert("eos_token_id".to_string(), eos.clone());
                    }
                }
                // Inject top-level image_token_id if not in text_config
                if !obj.contains_key("image_token_id") {
                    if let Some(itok) = json_value.get("image_token_id") {
                        obj.insert("image_token_id".to_string(), itok.clone());
                    }
                }
            }

            merged
        } else {
            json_value
        };

        let json: ConfigJson = serde_json::from_value(effective_value)
            .map_err(|e| HerbertError::Config(format!("Failed to parse config.json: {}", e)))?;

        if json.num_attention_heads == 0 {
            return Err(HerbertError::Config("num_attention_heads must be > 0".into()));
        }
        let head_dim = json
            .head_dim
            .unwrap_or(json.hidden_size / json.num_attention_heads);

        let model_family = ModelFamily::from_model_type(&json.model_type)
            .ok_or_else(|| {
                HerbertError::Config(format!(
                    "unsupported model_type: {:?}. Supported: qwen3, qwen3_moe, qwen3_vl, qwen3_vl_moe, mistral3, ministral3, mistral",
                    json.model_type
                ))
            })?;

        let (has_qk_norm, has_attn_bias) = match model_family {
            ModelFamily::Qwen3 => (true, false),
            ModelFamily::Mistral3 => (false, false),
        };
        let has_o_bias = false;

        let norm_type = NormType::RMSNorm;

        // Partial RoPE: compute rotary_ndims from partial_rotary_factor
        // Check top-level first, then rope_parameters (Qwen3.5 nests it there).
        let partial_rotary_factor = json.partial_rotary_factor
            .or_else(|| {
                json.rope_parameters.as_ref()
                    .and_then(|rp| rp.get("partial_rotary_factor"))
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(1.0) as f32;
        let rotary_ndims = (head_dim as f32 * partial_rotary_factor) as usize;

        let eos_token_id = parse_eos_token_id(json.eos_token_id.as_ref()).or_else(|| {
            match model_family {
                ModelFamily::Mistral3 => Some(2),
                _ => None,
            }
        });
        let image_token_id = parse_eos_token_id(json.image_token_id.as_ref());

        // rope_theta: from json.rope_theta OR rope_parameters.rope_theta
        let rope_theta = json
            .rope_theta
            .or_else(|| {
                json.rope_parameters
                    .as_ref()
                    .and_then(|rp| rp.get("rope_theta").and_then(|v| v.as_f64()))
            })
            .ok_or_else(|| {
                HerbertError::Config("missing rope_theta (or rope_parameters.rope_theta)".into())
            })?;

        // Parse RoPE type from rope_parameters or rope_scaling
        let rope_params = json.rope_parameters.as_ref().or(json.rope_scaling.as_ref());
        let rope_type = match rope_params
            .and_then(|rp| rp.get("rope_type").or_else(|| rp.get("type")))
            .and_then(|v| v.as_str())
        {
            Some("linear") => {
                let rp = rope_params.unwrap();
                let factor = rp.get("factor").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                if factor > 1.0 {
                    RopeType::Linear { factor }
                } else {
                    RopeType::Standard
                }
            }
            Some("yarn") => {
                let rp = rope_params.unwrap();
                let factor = rp.get("factor").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                let original_max_pos = rp
                    .get("original_max_position_embeddings")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(json.max_position_embeddings as u64)
                    as usize;
                let beta_fast = rp.get("beta_fast").and_then(|v| v.as_f64()).unwrap_or(32.0) as f32;
                let beta_slow = rp.get("beta_slow").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                RopeType::Yarn {
                    factor,
                    original_max_position_embeddings: original_max_pos,
                    beta_fast,
                    beta_slow,
                }
            }
            _ => RopeType::Standard,
        };

        // rms_norm_eps: from json.rms_norm_eps, default 1e-5
        let rms_norm_eps = json
            .rms_norm_eps
            .unwrap_or(1e-5);

        // For MoE-only models (like Qwen3-VL-MoE), intermediate_size may be
        // absent in config; fall back to moe_intermediate_size.
        let intermediate_size = json.intermediate_size
            .unwrap_or_else(|| json.moe_intermediate_size.unwrap_or(0));

        // Extract mrope_section from rope_scaling (if present)
        let mrope_section = json
            .rope_scaling
            .as_ref()
            .and_then(|rs| rs.get("mrope_section"))
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                if arr.len() == 3 {
                    let vals: Vec<usize> = arr.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect();
                    if vals.len() == 3 {
                        Some([vals[0], vals[1], vals[2]])
                    } else {
                        None
                    }
                } else {
                    None
                }
            });

        // mlp_only_layers: from json or derived from num_dense_layers
        let mut mlp_only_layers = json.mlp_only_layers.clone();
        if mlp_only_layers.is_empty() {
            if let Some(num_dense_layers) = json.num_dense_layers {
                let dense = num_dense_layers.min(json.num_hidden_layers);
                mlp_only_layers = (0..dense).collect();
            }
        }

        // full_attn_idxs: from json
        let mut full_attn_idxs = json.full_attn_idxs.clone();

        // Qwen3.5: derive full_attn_idxs from full_attention_interval
        if full_attn_idxs.is_empty() {
            if let Some(interval) = json.full_attention_interval {
                full_attn_idxs = (0..json.num_hidden_layers)
                    .filter(|&i| (i + 1) % interval == 0)
                    .collect();
            }
        }

        let conv_l_cache = json.conv_l_cache.unwrap_or(0);

        let num_experts = json.num_experts;

        // moe_intermediate_size: use explicit value, or fall back to intermediate_size for MoE models
        let moe_intermediate_size = json.moe_intermediate_size.or_else(|| {
            if num_experts.is_some() {
                json.intermediate_size
            } else {
                None
            }
        });

        // attn_scale from query_pre_attn_scalar
        let attn_scale = json.query_pre_attn_scalar.map(|s| 1.0 / (s as f32).sqrt());

        let config = Self {
            model_type: json.model_type,
            model_family,
            vocab_size: json.vocab_size,
            hidden_size: json.hidden_size,
            num_layers: json.num_hidden_layers,
            num_attention_heads: json.num_attention_heads,
            num_key_value_heads: json.num_key_value_heads,
            head_dim,
            intermediate_size,
            rope_theta: rope_theta as f32,
            rms_norm_eps: rms_norm_eps as f32,
            max_position_embeddings: json.max_position_embeddings,
            tie_word_embeddings: json.tie_word_embeddings,
            has_qk_norm,
            has_attn_bias,
            has_o_bias,
            norm_type,
            rotary_ndims,
            eos_token_id,
            num_experts,
            num_experts_per_tok: json.num_experts_per_tok,
            moe_intermediate_size,
            norm_topk_prob: json.norm_topk_prob,
            decoder_sparse_step: json.decoder_sparse_step.unwrap_or(1),
            mlp_only_layers,
            mrope_section,
            image_token_id,
            full_attn_idxs,
            conv_l_cache,
            rope_type,
            attn_scale,
            sliding_rope_theta: None,
            gemma3_rope_factor: None,
            linear_num_key_heads: json.linear_num_key_heads,
            linear_num_value_heads: json.linear_num_value_heads,
            linear_key_head_dim: json.linear_key_head_dim,
            linear_value_head_dim: json.linear_value_head_dim,
            linear_conv_kernel_dim: json.linear_conv_kernel_dim,
            full_attention_interval: json.full_attention_interval,
            has_shared_expert_gate: json.shared_expert_intermediate_size.is_some(),
            shared_expert_intermediate_size: json.shared_expert_intermediate_size,
            attn_output_gate: json.attn_output_gate.unwrap_or(false),
        };
        config.validate()?;
        Ok(config)
    }

}
