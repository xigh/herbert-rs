//! Streaming model loader for BF16 AVX-512 backend.
//!
//! Reuses the same BF16 weight format (row-major [N, K] Vec<BF16>).
//! Shares weight cache with the bf16 backend (same format, same version).

use herbert_backend_bf16::weight::Bf16Weight;
use herbert_backend_common::generic_model::GenericModel;
use herbert_backend_common::loader_common;
use herbert_backend_common::weight_cache;
use herbert_core::backend::LoadOpts;
use herbert_core::config::Config;
use herbert_core::error::Result;
use std::path::Path;

use crate::ops::Bf16Avx512Ops;

const BACKEND_ID: &str = "bf16-avx512";

pub fn load_model(
    model_dir: &Path,
    opts: LoadOpts,
) -> Result<(Config, GenericModel<Bf16Avx512Ops>)> {
    let config = Config::from_file(&model_dir.join("config.json"))?;

    eprintln!("DEBUG config: family={:?} model_type={} hidden={} layers={} heads={} kv_heads={} head_dim={} inter={} rotary_ndims={} rope_theta={} eos={:?} tie_embed={}",
        config.model_family, config.model_type, config.hidden_size, config.num_layers,
        config.num_attention_heads, config.num_key_value_heads, config.head_dim,
        config.intermediate_size, config.rotary_ndims, config.rope_theta,
        config.eos_token_id, config.tie_word_embeddings);
    eprintln!("DEBUG config MoE: experts={:?} per_tok={:?} moe_inter={:?} shared_inter={:?} shared_gate={} norm_topk={}",
        config.num_experts, config.num_experts_per_tok, config.moe_intermediate_size,
        config.shared_expert_intermediate_size, config.has_shared_expert_gate, config.norm_topk_prob);

    if !opts.no_cache {
        if let Ok(model_hash) = weight_cache::compute_model_hash(model_dir) {
            if let Some(model) = weight_cache::load_model_cache::<Bf16Avx512Ops>(
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

    let (config, model) = loader_common::load_model_streaming_progress::<Bf16Avx512Ops>(
        model_dir,
        |bf16_data, out_f, in_f, _name| {
            Ok(Bf16Weight {
                data: bf16_data.to_vec(),
                n: out_f,
                k: in_f,
            })
        },
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
