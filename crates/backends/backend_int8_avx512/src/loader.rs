//! Streaming model loader for INT8 AVX-512 backend.
//!
//! Quantizes ALL BF16 weights to INT8 per-channel symmetric at load time.
//! 1 byte/param vs 2 bytes/param for BF16 — 2x bandwidth savings.

use crate::ops::Int8Avx512Ops;
use crate::weight::quantize_bf16_to_int8;
use herbert_backend_common::generic_model::GenericModel;
use herbert_backend_common::loader_common;
use herbert_backend_common::weight_cache;
use herbert_core::backend::LoadOpts;
use herbert_core::config::Config;
use herbert_core::error::Result;
use std::path::Path;

const BACKEND_ID: &str = "int8-avx512";

pub fn load_model(
    model_dir: &Path,
    opts: LoadOpts,
) -> Result<(Config, GenericModel<Int8Avx512Ops>)> {
    let config = Config::from_file(&model_dir.join("config.json"))?;

    eprintln!("DEBUG config: family={:?} model_type={} hidden={} layers={} heads={} kv_heads={} head_dim={} inter={} rotary_ndims={} rope_theta={} eos={:?} tie_embed={}",
        config.model_family, config.model_type, config.hidden_size, config.num_layers,
        config.num_attention_heads, config.num_key_value_heads, config.head_dim,
        config.intermediate_size, config.rotary_ndims, config.rope_theta,
        config.eos_token_id, config.tie_word_embeddings);

    if !opts.no_cache {
        if let Ok(model_hash) = weight_cache::compute_model_hash(model_dir) {
            if let Some(model) = weight_cache::load_model_cache::<Int8Avx512Ops>(
                &config,
                model_dir,
                BACKEND_ID,
                model_hash,
                opts.show_progress,
            ) {
                return Ok((config, model));
            }
        }
    }

    let (config, model) = loader_common::load_model_streaming_progress::<Int8Avx512Ops>(
        model_dir,
        |bf16_data, out_f, in_f, _name| Ok(quantize_bf16_to_int8(bf16_data, out_f, in_f)),
        opts.show_progress,
    )?;

    if !opts.no_cache {
        if let Ok(model_hash) = weight_cache::compute_model_hash(model_dir) {
            match weight_cache::save_model_cache(&model, &config, model_dir, BACKEND_ID, model_hash)
            {
                Ok(Some(_cache_path)) => {
                    if opts.show_progress {
                        tracing::info!(backend = BACKEND_ID, "saved weight cache");
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "failed to save weight cache");
                }
            }
        }
    }

    Ok((config, model))
}
