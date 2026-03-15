//! Generic CPU backend that wraps a `GenericModel<L>`.
//!
//! Individual backends only need to implement `CpuBackendConfig` — a tiny trait
//! that provides the name, thread-pool module, and loader function.

use crate::execution_phase::{self, ExecutionPhase};
use crate::generic_model::{DeepStackInfo, GenericModel};
use crate::kv_cache::CpuKvCache;
use crate::linear_ops::LinearOps;
use crate::position_ids;
use crate::position_ids::ImageInfo;
use crate::prefix_cache::{PrefixCache, PrefixCacheConfig};
use crate::weight_cache::{self, CacheWeight};
use herbert_core::backend::{Backend, DecodeOutput, LayerTimings, LoadOpts, PrefillOutput, RunOpts, VisionEmbedding};
use herbert_core::config::{Config, KvQuantType};
use herbert_core::error::{HerbertError, Result};
use herbert_core::KvHandle;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

/// Configuration trait for CPU backends.
///
/// Each backend implements this to specify its linear ops, name, and loader.
pub trait CpuBackendConfig: 'static {
    /// The LinearOps implementation for this backend's weight format.
    type Ops: LinearOps;

    /// Backend name (e.g., "threads", "q8", "q4").
    const NAME: &'static str;

    /// Configure the global thread pool with the given number of threads.
    /// Return `Ok(())` if successful or if threading is not supported.
    fn configure_thread_pool(num_threads: usize) -> Result<()>;

    /// Load model from disk. Returns config and the generic model.
    fn load_model(dir: &Path, opts: LoadOpts) -> Result<(Config, GenericModel<Self::Ops>)>;
}

/// Generic CPU backend. Each concrete backend is a type alias of this.
pub struct GenericCpuBackend<C: CpuBackendConfig> {
    model: Option<Arc<GenericModel<C::Ops>>>,
    config: Option<Config>,
    kv_reserve_tokens: Option<usize>,
    show_progress: bool,
    prefix_cache: Option<PrefixCache>,
    kv_quant: KvQuantType,
    kv_budget: Option<usize>,
}

impl<C: CpuBackendConfig> GenericCpuBackend<C> {
    pub fn new() -> Self {
        Self {
            model: None,
            config: None,
            kv_reserve_tokens: None,
            show_progress: false,
            prefix_cache: None,
            kv_quant: KvQuantType::BF16,
            kv_budget: None,
        }
    }
}

impl<C: CpuBackendConfig> Default for GenericCpuBackend<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: CpuBackendConfig> Backend for GenericCpuBackend<C>
where
    <C::Ops as LinearOps>::Weight: CacheWeight,
{
    fn name(&self) -> &'static str {
        C::NAME
    }

    fn load(&mut self, model_path: &Path, opts: LoadOpts) -> Result<()> {
        let env_threads = std::env::var("NUM_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let requested_threads = opts.num_threads.or(env_threads).filter(|v| *v > 0);
        if let Some(num_threads) = requested_threads {
            C::configure_thread_pool(num_threads)?;
        }

        // Configure phase-aware execution (decode thread cap + prefill chunking)
        let prefill_chunk_size = opts.prefill_chunk_size.unwrap_or(0);
        let decode_max_workers = match opts.decode_threads {
            Some(0) => 0, // explicit 0 = no cap
            Some(n) => n,
            None => {
                // Auto: 2/3 of thread count, capped at 8 (DRAM saturates at 6-8 cores).
                let pool_size = requested_threads.unwrap_or_else(|| {
                    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
                });
                let auto = (pool_size * 2 / 3).max(1).min(8);
                auto
            }
        };
        execution_phase::configure_phases(decode_max_workers, prefill_chunk_size);
        if decode_max_workers > 0 {
            info!(decode_max_workers, "Decode thread cap configured");
        }
        if prefill_chunk_size > 0 {
            info!(prefill_chunk_size, "Prefill chunking configured");
        }

        self.kv_reserve_tokens = opts.kv_reserve_tokens;
        self.show_progress = opts.show_progress;
        self.kv_quant = opts.kv_quant;
        self.kv_budget = opts.kv_budget;
        info!(kv_quant = %self.kv_quant, "KV cache quantization configured");
        if let Some(budget) = self.kv_budget {
            info!(budget, "H2O KV eviction enabled");
        }
        let _use_q_bf16_opt = opts.use_q_bf16;
        let prefix_cache_dir = opts.prefix_cache_dir.clone();
        #[allow(unused_mut)]
        let (config, mut model) = C::load_model(model_path, opts)?;

        // Propagate Q BF16 flag to attention layers (runtime detection of avx512bf16)
        if _use_q_bf16_opt {
            #[cfg(target_arch = "x86_64")]
            {
                let available = is_x86_feature_detected!("avx512bf16");
                if available {
                    for layer in &mut model.decoder_layers {
                        if let crate::generic_layer::LayerBlock::Attention(ref mut attn) = layer.block {
                            attn.use_q_bf16 = true;
                        }
                    }
                    info!("Q BF16 attention (VDPBF16PS) enabled");
                } else {
                    tracing::warn!("--q-format bf16 requested but avx512bf16 not available, using F32");
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            tracing::warn!("--q-format bf16 is only supported on x86_64");
        }

        // Initialize prefix cache if configured
        if let Some(cache_dir) = prefix_cache_dir {
            match weight_cache::compute_model_hash(model_path) {
                Ok(model_hash) => {
                    let pc_config = PrefixCacheConfig::new(cache_dir);
                    match PrefixCache::new(pc_config, model_hash, config.num_layers, config.kv_dim()) {
                        Ok(pc) => {
                            self.prefix_cache = Some(pc);
                            info!("Prefix KV cache enabled");
                        }
                        Err(e) => warn!("Failed to initialize prefix cache: {}", e),
                    }
                }
                Err(e) => warn!("Failed to compute model hash for prefix cache: {}", e),
            }
        }

        self.config = Some(config);
        self.model = Some(Arc::new(model));
        Ok(())
    }

    fn prefill(
        &mut self,
        input_tokens: &[u32],
        opts: RunOpts,
    ) -> Result<(KvHandle, PrefillOutput)> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| HerbertError::Backend("Model not loaded".to_string()))?;

        execution_phase::set_phase(ExecutionPhase::Prefill);
        debug!(num_tokens = input_tokens.len(), mrope = model.is_mrope(), "prefill: creating KV cache");
        let start_time = std::time::Instant::now();
        let reserve_tokens = self
            .kv_reserve_tokens
            .or(opts.decode_tokens)
            .map(|decode_hint| input_tokens.len().saturating_add(decode_hint));
        let mut kv_cache = model.create_kv_cache_with_quant(reserve_tokens, self.kv_quant);
        debug!(reserve = ?reserve_tokens, kv_quant = %self.kv_quant, "prefill: KV cache created");
        let chunk_size = execution_phase::phase_config().prefill_chunk_size;

        // Try prefix cache lookup
        let prefix_hit_len = if let Some(ref mut pc) = self.prefix_cache {
            if let Some((_cached_len, path)) = pc.lookup(input_tokens) {
                match pc.load_kv_data(&path, &mut kv_cache) {
                    Ok(len) => len,
                    Err(e) => {
                        warn!("Prefix cache load failed: {}, falling back to full prefill", e);
                        // Reset KV cache on failure
                        kv_cache = model.create_kv_cache_with_quant(reserve_tokens, self.kv_quant);
                        0
                    }
                }
            } else {
                0
            }
        } else {
            0
        };

        let (logits, first_token, layer_times) = if model.is_mrope() && prefix_hit_len > 0 && prefix_hit_len < input_tokens.len() {
            // MRoPE partial prefix cache hit: prefill remaining tokens
            let remaining = &input_tokens[prefix_hit_len..];
            debug!(prefix_hit_len, remaining = remaining.len(), "prefill: MRoPE partial prefix cache hit");
            kv_cache.kv.text_pos = prefix_hit_len as u32;
            let positions: Vec<position_ids::Position3D> = (prefix_hit_len..input_tokens.len())
                .map(|i| position_ids::Position3D::text(i as u32))
                .collect();
            let hidden_size = model.config.hidden_size;
            let mut embeds = vec![0.0f32; remaining.len() * hidden_size];
            for (i, &token_id) in remaining.iter().enumerate() {
                let token_idx = token_id as usize;
                if token_idx >= model.config.vocab_size {
                    return Err(HerbertError::Backend(format!(
                        "token_id {} >= vocab_size {}", token_id, model.config.vocab_size
                    )));
                }
                model.embed_lookup(token_idx, &mut embeds[i * hidden_size..(i + 1) * hidden_size]);
            }
            let result = model.prefill_with_embeds_mrope(&embeds, &positions, &mut kv_cache, opts.return_logits, None, self.show_progress, prefix_hit_len)?;
            kv_cache.kv.text_pos = input_tokens.len() as u32;
            result
        } else if model.is_mrope() && prefix_hit_len > 0 && prefix_hit_len >= input_tokens.len() {
            // MRoPE exact prefix cache hit: trim last token, re-prefill it to get logits
            debug!(prefix_hit_len, "prefill: MRoPE exact prefix cache hit, re-prefilling last token");
            let trim_to = prefix_hit_len - 1;
            let num_layers = kv_cache.kv.layer_data.len();
            for l in 0..num_layers {
                kv_cache.truncate_to(l, trim_to);
            }
            kv_cache.kv.seq_len = trim_to;
            kv_cache.kv.text_pos = trim_to as u32;
            let positions = vec![position_ids::Position3D::text(trim_to as u32)];
            let hidden_size = model.config.hidden_size;
            let token_id = input_tokens[trim_to];
            let token_idx = token_id as usize;
            if token_idx >= model.config.vocab_size {
                return Err(HerbertError::Backend(format!(
                    "token_id {} >= vocab_size {}", token_id, model.config.vocab_size
                )));
            }
            let mut embeds = vec![0.0f32; hidden_size];
            model.embed_lookup(token_idx, &mut embeds);
            let result = model.prefill_with_embeds_mrope(&embeds, &positions, &mut kv_cache, opts.return_logits, None, self.show_progress, trim_to)?;
            kv_cache.kv.text_pos = input_tokens.len() as u32;
            result
        } else if model.is_mrope() {
            // MRoPE full prefill (no cache hit)
            debug!("prefill: building MRoPE text positions + embeds");
            let positions = position_ids::build_text_position_ids(input_tokens.len());
            let hidden_size = model.config.hidden_size;
            let mut embeds = vec![0.0f32; input_tokens.len() * hidden_size];
            for (i, &token_id) in input_tokens.iter().enumerate() {
                let token_idx = token_id as usize;
                if token_idx >= model.config.vocab_size {
                    return Err(HerbertError::Backend(format!(
                        "token_id {} >= vocab_size {}", token_id, model.config.vocab_size
                    )));
                }
                model.embed_lookup(token_idx, &mut embeds[i * hidden_size..(i + 1) * hidden_size]);
            }
            debug!("prefill: running prefill_with_embeds_mrope");
            let result = model.prefill_with_embeds_mrope(&embeds, &positions, &mut kv_cache, opts.return_logits, None, self.show_progress, 0)?;
            kv_cache.kv.text_pos = input_tokens.len() as u32;
            // Save system prefix for double prefill
            if let Some(sys_len) = opts.system_prefix_len {
                if sys_len > 0 && sys_len < input_tokens.len() {
                    if let Some(ref mut pc) = self.prefix_cache {
                        if let Err(e) = pc.save_system(&input_tokens[..sys_len], &kv_cache) {
                            warn!("Failed to save system prefix cache: {}", e);
                        }
                    }
                }
            }
            // Save full sequence to prefix cache
            if let Some(ref mut pc) = self.prefix_cache {
                if let Err(e) = pc.save(input_tokens, &kv_cache) {
                    warn!("Failed to save prefix cache: {}", e);
                }
            }
            result
        } else if prefix_hit_len > 0 && prefix_hit_len < input_tokens.len() {
            // Partial prefix cache hit: only prefill the remaining tokens
            let remaining = &input_tokens[prefix_hit_len..];
            debug!(prefix_hit_len, remaining = remaining.len(), "prefill: partial prefix cache hit");
            model.prefill_with_token(remaining, &mut kv_cache, prefix_hit_len, opts.return_logits, self.show_progress)?
        } else if prefix_hit_len > 0 && prefix_hit_len >= input_tokens.len() {
            // Exact hit: trim last token from cache, re-prefill it to get logits
            debug!(prefix_hit_len, "prefill: exact prefix cache hit, re-prefilling last token");
            let trim_to = prefix_hit_len - 1;
            let num_layers = kv_cache.kv.layer_data.len();
            for l in 0..num_layers {
                kv_cache.truncate_to(l, trim_to);
            }
            kv_cache.kv.seq_len = trim_to;
            model.prefill_with_token(&input_tokens[trim_to..], &mut kv_cache, trim_to, opts.return_logits, self.show_progress)?
        } else if chunk_size > 0 {
            debug!(chunk_size, "prefill: running chunked prefill");
            let result = model.prefill_with_token_chunked(input_tokens, &mut kv_cache, chunk_size, opts.return_logits, self.show_progress)?;
            // Save system prefix for double prefill
            if let Some(sys_len) = opts.system_prefix_len {
                if sys_len > 0 && sys_len < input_tokens.len() {
                    if let Some(ref mut pc) = self.prefix_cache {
                        if let Err(e) = pc.save_system(&input_tokens[..sys_len], &kv_cache) {
                            warn!("Failed to save system prefix cache: {}", e);
                        }
                    }
                }
            }
            // Save full sequence to prefix cache on miss
            if let Some(ref mut pc) = self.prefix_cache {
                if let Err(e) = pc.save(input_tokens, &kv_cache) {
                    warn!("Failed to save prefix cache: {}", e);
                }
            }
            result
        } else {
            debug!("prefill: running prefill_with_token");
            let result = model.prefill_with_token(input_tokens, &mut kv_cache, 0, opts.return_logits, self.show_progress)?;
            // Save system prefix for double prefill
            if let Some(sys_len) = opts.system_prefix_len {
                if sys_len > 0 && sys_len < input_tokens.len() {
                    if let Some(ref mut pc) = self.prefix_cache {
                        if let Err(e) = pc.save_system(&input_tokens[..sys_len], &kv_cache) {
                            warn!("Failed to save system prefix cache: {}", e);
                        }
                    }
                }
            }
            // Save full sequence to prefix cache on miss
            if let Some(ref mut pc) = self.prefix_cache {
                if let Err(e) = pc.save(input_tokens, &kv_cache) {
                    warn!("Failed to save prefix cache: {}", e);
                }
            }
            result
        };
        info!(first_token, elapsed_ms = (start_time.elapsed().as_secs_f64() * 1000.0) as u64, "prefill: done");
        let elapsed = start_time.elapsed().as_secs_f64();

        let timings = if opts.profile {
            Some(LayerTimings {
                layer_times,
                total_time: elapsed,
            })
        } else {
            None
        };

        Ok((
            KvHandle::new(kv_cache),
            PrefillOutput {
                logits,
                first_token,
                timings,
                cached_prefix_len: prefix_hit_len,
            },
        ))
    }

    fn continue_prefill(
        &mut self,
        kv: &mut KvHandle,
        input_tokens: &[u32],
        opts: RunOpts,
    ) -> Result<PrefillOutput> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| HerbertError::Backend("Model not loaded".to_string()))?;

        execution_phase::set_phase(ExecutionPhase::Prefill);
        let start_time = std::time::Instant::now();
        let kv_cache = kv.get_mut::<CpuKvCache>()?;
        let start_pos = kv_cache.kv.seq_len;
        debug!(start_pos, new_tokens = input_tokens.len(), "continue_prefill: appending to existing KV cache");

        let (logits, first_token, layer_times) = if model.is_mrope() {
            // VL model: compute MRoPE positions + embeddings, use MRoPE prefill path
            let text_pos = kv_cache.kv.text_pos;
            let positions: Vec<position_ids::Position3D> = (0..input_tokens.len())
                .map(|i| position_ids::Position3D::text(text_pos + i as u32))
                .collect();
            let hidden_size = model.config.hidden_size;
            let mut embeds = vec![0.0f32; input_tokens.len() * hidden_size];
            for (i, &token_id) in input_tokens.iter().enumerate() {
                let token_idx = token_id as usize;
                if token_idx >= model.config.vocab_size {
                    return Err(HerbertError::Backend(format!(
                        "token_id {} >= vocab_size {}", token_id, model.config.vocab_size
                    )));
                }
                model.embed_lookup(token_idx, &mut embeds[i * hidden_size..(i + 1) * hidden_size]);
            }
            let result = model.prefill_with_embeds_mrope(
                &embeds, &positions, kv_cache, opts.return_logits, None, self.show_progress, start_pos,
            )?;
            kv_cache.kv.text_pos = text_pos + input_tokens.len() as u32;
            result
        } else {
            model.prefill_with_token(input_tokens, kv_cache, start_pos, opts.return_logits, self.show_progress)?
        };

        let elapsed = start_time.elapsed().as_secs_f64();
        info!(first_token, elapsed_ms = (elapsed * 1000.0) as u64, "continue_prefill: done");

        let timings = if opts.profile {
            Some(LayerTimings {
                layer_times,
                total_time: elapsed,
            })
        } else {
            None
        };

        Ok(PrefillOutput {
            logits,
            first_token,
            timings,
            cached_prefix_len: 0,
        })
    }

    fn decode_next(
        &mut self,
        kv: &mut KvHandle,
        token: u32,
        opts: RunOpts,
    ) -> Result<DecodeOutput> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| HerbertError::Backend("Model not loaded".to_string()))?;
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| HerbertError::Backend("Config not loaded".to_string()))?;

        execution_phase::set_phase(ExecutionPhase::Decode);
        let start_time = std::time::Instant::now();
        let kv_cache = kv.get_mut::<CpuKvCache>()?;
        let (logits, next_token, layer_times) = if model.is_mrope() {
            model.decode_step_mrope(token, kv_cache, opts.return_logits)?
        } else {
            model.decode_step(token, kv_cache, opts.return_logits)?
        };

        // H2O eviction: if budget is set and sequence exceeds budget + hysteresis
        if let Some(budget) = self.kv_budget {
            let hyst = crate::h2o::hysteresis(budget);
            if kv_cache.kv.seq_len > budget + hyst {
                let evicted = crate::h2o::h2o_evict(kv_cache, config, budget);
                if evicted > 0 {
                    eprintln!("[H2O] evicted {} positions, {} → {}", evicted, kv_cache.kv.seq_len + evicted, kv_cache.kv.seq_len);
                }
            }
        }

        let elapsed = start_time.elapsed().as_secs_f64();

        let timings = if opts.profile {
            Some(LayerTimings {
                layer_times,
                total_time: elapsed,
            })
        } else {
            None
        };

        Ok(DecodeOutput {
            logits,
            token: next_token,
            timings,
        })
    }

    fn verify_draft(
        &mut self,
        kv: &mut KvHandle,
        draft_tokens: &[u32],
    ) -> Result<herbert_core::backend::VerifyOutput> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| HerbertError::Backend("Model not loaded".to_string()))?;

        execution_phase::set_phase(ExecutionPhase::Prefill);
        let kv_cache = kv.get_mut::<CpuKvCache>()?;
        let start_pos = kv_cache.kv.seq_len;

        let all_logits = if model.is_mrope() {
            // VL model: compute MRoPE 3D text positions
            let text_pos = kv_cache.kv.text_pos;
            let positions: Vec<crate::position_ids::Position3D> = (0..draft_tokens.len())
                .map(|i| crate::position_ids::Position3D::text(text_pos + i as u32))
                .collect();
            let result = model.verify_draft_tokens_mrope(draft_tokens, &positions, kv_cache, start_pos)?;
            kv_cache.kv.text_pos = text_pos + draft_tokens.len() as u32;
            result
        } else {
            model.verify_draft_tokens(draft_tokens, kv_cache, start_pos)?
        };

        Ok(herbert_core::backend::VerifyOutput {
            logits_per_position: all_logits,
        })
    }

    fn embed(&mut self, input_tokens: &[u32]) -> Result<Vec<f32>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| HerbertError::Backend("Model not loaded".to_string()))?;
        let mut kv_cache = model.create_kv_cache(Some(input_tokens.len()));
        model.prefill_for_embedding(input_tokens, &mut kv_cache)
    }

    fn kv_seq_len(&self, kv: &KvHandle) -> Result<usize> {
        let cache = kv.get_ref::<CpuKvCache>()?;
        Ok(cache.kv.seq_len)
    }

    fn truncate_kv(
        &mut self,
        kv: &mut KvHandle,
        target_len: usize,
    ) -> Result<()> {
        let cache = kv.get_mut::<CpuKvCache>()?;
        if cache.kv.seq_len > target_len {
            let removed = cache.kv.seq_len - target_len;
            let num_layers = cache.kv.layer_data.len();
            for l in 0..num_layers {
                cache.truncate_to(l, target_len);
            }
            cache.kv.seq_len = target_len;
            cache.kv.text_pos = cache.kv.text_pos.saturating_sub(removed as u32);
        }
        Ok(())
    }

    fn take_eagle3_hidden_states(
        &mut self,
        kv: &mut KvHandle,
    ) -> Result<[Vec<f32>; 3]> {
        let cache = kv.get_mut::<CpuKvCache>()?;
        if !cache.ctx.eagle3_active {
            return Err(HerbertError::Backend(
                "EAGLE-3 extraction not active".to_string(),
            ));
        }
        Ok([
            std::mem::take(&mut cache.ctx.eagle3_hidden_states[0]),
            std::mem::take(&mut cache.ctx.eagle3_hidden_states[1]),
            std::mem::take(&mut cache.ctx.eagle3_hidden_states[2]),
        ])
    }

    fn enable_eagle3_extraction(
        &mut self,
        kv: &mut KvHandle,
        extract_layers: [usize; 3],
    ) -> Result<()> {
        let cache = kv.get_mut::<CpuKvCache>()?;
        cache.ctx.eagle3_active = true;
        cache.ctx.eagle3_extract_layers = extract_layers;
        Ok(())
    }

    fn embed_tokens_bf16(&self) -> Result<&[u16]> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| HerbertError::Backend("Model not loaded".to_string()))?;
        Ok(model.embed_tokens())
    }

    fn embed_all(&mut self, input_tokens: &[u32], bidirectional: bool, l2_normalize: bool) -> Result<Vec<f32>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| HerbertError::Backend("Model not loaded".to_string()))?;
        let mut kv_cache = model.create_kv_cache(Some(input_tokens.len()));
        model.prefill_all_embeddings(input_tokens, &mut kv_cache, bidirectional, l2_normalize)
    }

    fn config(&self) -> Option<&Config> {
        self.config.as_ref()
    }

    fn prefill_vl(
        &mut self,
        tokens: &[u32],
        images: &[VisionEmbedding],
        image_positions: &[usize],
        opts: RunOpts,
    ) -> Result<(KvHandle, PrefillOutput)> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| HerbertError::Backend("Model not loaded".to_string()))?;

        if images.len() != image_positions.len() {
            return Err(HerbertError::Backend(format!(
                "images ({}) and image_positions ({}) must have the same length",
                images.len(),
                image_positions.len()
            )));
        }

        let hidden_size = model.config.hidden_size;
        let use_mrope = model.is_mrope();
        debug!(total_tokens = tokens.len(), num_images = images.len(), hidden_size, use_mrope, "prefill_vl: start");

        // Build ImageInfo list
        let image_infos: Vec<ImageInfo> = images
            .iter()
            .zip(image_positions.iter())
            .map(|(img, &pos)| ImageInfo {
                start_idx: pos,
                grid_h: img.grid_h,
                grid_w: img.grid_w,
            })
            .collect();

        let total_len = tokens.len();

        // Build combined embeddings (text lookup + image hidden states)
        debug!("prefill_vl: building combined embeddings");
        let image_hs_refs: Vec<&[f32]> = images.iter().map(|img| img.hidden_states.as_slice()).collect();
        let embeds = position_ids::build_vl_embeddings(
            tokens,
            &image_infos,
            &image_hs_refs,
            &model.embed_tokens,
            hidden_size,
        );

        // Compute text_pos for subsequent decode
        let text_token_count = total_len - images.iter().map(|img| img.num_tokens).sum::<usize>();

        let start_time = std::time::Instant::now();
        let reserve_tokens = self
            .kv_reserve_tokens
            .or(opts.decode_tokens)
            .map(|decode_hint| total_len.saturating_add(decode_hint));
        let mut kv_cache = model.create_kv_cache_with_quant(reserve_tokens, self.kv_quant);

        let (logits, first_token, layer_times) = if use_mrope {
            // ── M-RoPE path (Qwen3-VL): 3D positions + DeepStack ──
            debug!("prefill_vl: building 3D position IDs (MRoPE)");
            let positions = position_ids::build_vl_position_ids(total_len, &image_infos);

            // Build image token indices for DeepStack injection
            let image_token_indices: Vec<usize> = image_infos
                .iter()
                .flat_map(|img| img.start_idx..img.end_idx())
                .collect();

            // Collect deepstack features from all images (concatenated per layer)
            let all_deepstack: Vec<Vec<f32>> = {
                let max_ds_layers = images.iter().map(|img| img.deepstack_features.len()).max().unwrap_or(0);
                let mut layers = Vec::with_capacity(max_ds_layers);
                for layer_idx in 0..max_ds_layers {
                    let mut combined = Vec::new();
                    for img in images.iter() {
                        if layer_idx < img.deepstack_features.len() {
                            combined.extend_from_slice(&img.deepstack_features[layer_idx]);
                        }
                    }
                    layers.push(combined);
                }
                layers
            };

            let deepstack_info = if !all_deepstack.is_empty() {
                Some(DeepStackInfo {
                    image_token_indices: &image_token_indices,
                    layer_features: &all_deepstack,
                })
            } else {
                None
            };

            debug!(
                total_len, text_token_count,
                image_token_indices = image_token_indices.len(),
                deepstack_layers = all_deepstack.len(),
                "prefill_vl: MRoPE path"
            );

            model.prefill_with_embeds_mrope(
                &embeds, &positions, &mut kv_cache,
                opts.return_logits, deepstack_info.as_ref(), self.show_progress, 0,
            )?
        } else {
            // ── Standard 1D RoPE path (Mistral3/Pixtral, LFM2) ──
            // All tokens get sequential positions, no DeepStack
            debug!(total_len, text_token_count, "prefill_vl: standard 1D RoPE path");
            model.prefill_with_embeds(
                &embeds, &mut kv_cache, 0,
                opts.return_logits, self.show_progress,
            )?
        };

        info!(first_token, elapsed_ms = (start_time.elapsed().as_secs_f64() * 1000.0) as u64, "prefill_vl: done");

        kv_cache.kv.text_pos = text_token_count as u32;
        let elapsed = start_time.elapsed().as_secs_f64();

        let timings = if opts.profile {
            Some(LayerTimings {
                layer_times,
                total_time: elapsed,
            })
        } else {
            None
        };

        Ok((
            KvHandle::new(kv_cache),
            PrefillOutput {
                logits,
                first_token,
                timings,
                cached_prefix_len: 0,
            },
        ))
    }
}

// ============================================================================
// ConcurrentBackend: &self inference for multi-request concurrency
// ============================================================================

/// Backend that supports concurrent inference via `&self` methods.
///
/// Creates a fresh `CpuKvCache` (KvStore + InferenceContext) per request,
/// so multiple requests can run simultaneously without exclusive access.
/// The only shared mutable state is the `PrefixCache` (disk I/O), which
/// is protected by a `Mutex`.
pub struct ConcurrentBackend<C: CpuBackendConfig>
where
    <C::Ops as LinearOps>::Weight: CacheWeight,
{
    model: Arc<GenericModel<C::Ops>>,
    config: Config,
    kv_quant: KvQuantType,
    kv_reserve_tokens: Option<usize>,
    prefix_cache: Option<Mutex<PrefixCache>>,
}

impl<C: CpuBackendConfig> ConcurrentBackend<C>
where
    <C::Ops as LinearOps>::Weight: CacheWeight,
{
    /// Convert a fully-loaded `GenericCpuBackend` into a `ConcurrentBackend`,
    /// consuming the original backend.
    ///
    /// The backend must have been loaded (i.e., `load()` must have been called)
    /// or this returns an error.
    pub fn from_loaded_backend(backend: GenericCpuBackend<C>) -> Result<Self> {
        let model = backend
            .model
            .ok_or_else(|| HerbertError::Backend("Cannot create ConcurrentBackend: model not loaded".to_string()))?;
        let config = backend
            .config
            .ok_or_else(|| HerbertError::Backend("Cannot create ConcurrentBackend: config not loaded".to_string()))?;

        let prefix_cache = backend.prefix_cache.map(Mutex::new);

        Ok(Self {
            model,
            config,
            kv_quant: backend.kv_quant,
            kv_reserve_tokens: backend.kv_reserve_tokens,
            prefix_cache,
        })
    }

    /// Return the model configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Whether this model uses MRoPE (VL model).
    pub fn is_mrope(&self) -> bool {
        self.model.is_mrope()
    }

    /// Run prefill on a fresh KV cache, returning both the cache and the output.
    ///
    /// This method takes `&self` — it creates a new `CpuKvCache` per call and
    /// only touches the shared `PrefixCache` under a short-held `Mutex` lock.
    ///
    /// Handles both MRoPE (VL) and standard 1D RoPE paths.
    pub fn prefill(
        &self,
        input_tokens: &[u32],
        opts: RunOpts,
    ) -> Result<(CpuKvCache, PrefillOutput)> {
        execution_phase::set_phase(ExecutionPhase::Prefill);
        debug!(num_tokens = input_tokens.len(), mrope = self.model.is_mrope(), "concurrent_prefill: creating KV cache");
        let start_time = std::time::Instant::now();

        let reserve_tokens = self
            .kv_reserve_tokens
            .or(opts.decode_tokens)
            .map(|decode_hint| input_tokens.len().saturating_add(decode_hint));
        let mut kv_cache = self.model.create_kv_cache_with_quant(reserve_tokens, self.kv_quant);
        debug!(reserve = ?reserve_tokens, kv_quant = %self.kv_quant, "concurrent_prefill: KV cache created");

        let chunk_size = execution_phase::phase_config().prefill_chunk_size;

        // Try prefix cache lookup (short Mutex hold for lookup + load)
        let prefix_hit_len = if let Some(ref pc_mutex) = self.prefix_cache {
            let lookup_result = {
                let mut pc = pc_mutex.lock().unwrap();
                pc.lookup(input_tokens)
            };
            if let Some((_cached_len, path)) = lookup_result {
                // load_kv_data takes &self on PrefixCache, but we still need the lock
                // since lookup returned a path. Actually load_kv_data is &self, so no lock needed.
                let pc = pc_mutex.lock().unwrap();
                match pc.load_kv_data(&path, &mut kv_cache) {
                    Ok(len) => len,
                    Err(e) => {
                        warn!("Prefix cache load failed: {}, falling back to full prefill", e);
                        kv_cache = self.model.create_kv_cache_with_quant(reserve_tokens, self.kv_quant);
                        0
                    }
                }
            } else {
                0
            }
        } else {
            0
        };

        // Report prefill start (0/N); chunked path reports intermediate progress
        crate::prefill_progress::report(0, input_tokens.len());

        let (logits, first_token, layer_times) = if self.model.is_mrope() && prefix_hit_len > 0 && prefix_hit_len < input_tokens.len() {
            // MRoPE partial prefix cache hit: prefill remaining tokens
            let remaining = &input_tokens[prefix_hit_len..];
            debug!(prefix_hit_len, remaining = remaining.len(), "concurrent_prefill: MRoPE partial prefix cache hit");
            kv_cache.kv.text_pos = prefix_hit_len as u32;
            let positions: Vec<position_ids::Position3D> = (prefix_hit_len..input_tokens.len())
                .map(|i| position_ids::Position3D::text(i as u32))
                .collect();
            let hidden_size = self.model.config.hidden_size;
            let mut embeds = vec![0.0f32; remaining.len() * hidden_size];
            for (i, &token_id) in remaining.iter().enumerate() {
                let token_idx = token_id as usize;
                if token_idx >= self.model.config.vocab_size {
                    return Err(HerbertError::Backend(format!(
                        "token_id {} >= vocab_size {}", token_id, self.model.config.vocab_size
                    )));
                }
                self.model.embed_lookup(token_idx, &mut embeds[i * hidden_size..(i + 1) * hidden_size]);
            }
            let result = self.model.prefill_with_embeds_mrope(&embeds, &positions, &mut kv_cache, opts.return_logits, None, false, prefix_hit_len)?;
            kv_cache.kv.text_pos = input_tokens.len() as u32;
            result
        } else if self.model.is_mrope() && prefix_hit_len > 0 && prefix_hit_len >= input_tokens.len() {
            // MRoPE exact prefix cache hit: trim last token, re-prefill it to get logits
            debug!(prefix_hit_len, "concurrent_prefill: MRoPE exact prefix cache hit, re-prefilling last token");
            let trim_to = prefix_hit_len - 1;
            let num_layers = kv_cache.kv.layer_data.len();
            for l in 0..num_layers {
                kv_cache.truncate_to(l, trim_to);
            }
            kv_cache.kv.seq_len = trim_to;
            kv_cache.kv.text_pos = trim_to as u32;
            let positions = vec![position_ids::Position3D::text(trim_to as u32)];
            let hidden_size = self.model.config.hidden_size;
            let token_id = input_tokens[trim_to];
            let token_idx = token_id as usize;
            if token_idx >= self.model.config.vocab_size {
                return Err(HerbertError::Backend(format!(
                    "token_id {} >= vocab_size {}", token_id, self.model.config.vocab_size
                )));
            }
            let mut embeds = vec![0.0f32; hidden_size];
            self.model.embed_lookup(token_idx, &mut embeds);
            let result = self.model.prefill_with_embeds_mrope(&embeds, &positions, &mut kv_cache, opts.return_logits, None, false, trim_to)?;
            kv_cache.kv.text_pos = input_tokens.len() as u32;
            result
        } else if self.model.is_mrope() {
            // MRoPE full prefill (no cache hit)
            debug!("concurrent_prefill: building MRoPE text positions + embeds");
            let positions = position_ids::build_text_position_ids(input_tokens.len());
            let hidden_size = self.model.config.hidden_size;
            let mut embeds = vec![0.0f32; input_tokens.len() * hidden_size];
            for (i, &token_id) in input_tokens.iter().enumerate() {
                let token_idx = token_id as usize;
                if token_idx >= self.model.config.vocab_size {
                    return Err(HerbertError::Backend(format!(
                        "token_id {} >= vocab_size {}", token_id, self.model.config.vocab_size
                    )));
                }
                self.model.embed_lookup(token_idx, &mut embeds[i * hidden_size..(i + 1) * hidden_size]);
            }
            debug!("concurrent_prefill: running prefill_with_embeds_mrope");
            let result = self.model.prefill_with_embeds_mrope(&embeds, &positions, &mut kv_cache, opts.return_logits, None, false, 0)?;
            kv_cache.kv.text_pos = input_tokens.len() as u32;
            // Save to prefix cache (short Mutex hold)
            if let Some(sys_len) = opts.system_prefix_len {
                if sys_len > 0 && sys_len < input_tokens.len() {
                    if let Some(ref pc_mutex) = self.prefix_cache {
                        let mut pc = pc_mutex.lock().unwrap();
                        if let Err(e) = pc.save_system(&input_tokens[..sys_len], &kv_cache) {
                            warn!("Failed to save system prefix cache: {}", e);
                        }
                    }
                }
            }
            if let Some(ref pc_mutex) = self.prefix_cache {
                let mut pc = pc_mutex.lock().unwrap();
                if let Err(e) = pc.save(input_tokens, &kv_cache) {
                    warn!("Failed to save prefix cache: {}", e);
                }
            }
            result
        } else if prefix_hit_len > 0 && prefix_hit_len < input_tokens.len() {
            // Partial prefix cache hit: only prefill the remaining tokens
            let remaining = &input_tokens[prefix_hit_len..];
            debug!(prefix_hit_len, remaining = remaining.len(), "concurrent_prefill: partial prefix cache hit");
            self.model.prefill_with_token(remaining, &mut kv_cache, prefix_hit_len, opts.return_logits, false)?
        } else if prefix_hit_len > 0 && prefix_hit_len >= input_tokens.len() {
            // Exact hit: trim last token from cache, re-prefill it to get logits
            debug!(prefix_hit_len, "concurrent_prefill: exact prefix cache hit, re-prefilling last token");
            let trim_to = prefix_hit_len - 1;
            let num_layers = kv_cache.kv.layer_data.len();
            for l in 0..num_layers {
                kv_cache.truncate_to(l, trim_to);
            }
            kv_cache.kv.seq_len = trim_to;
            self.model.prefill_with_token(&input_tokens[trim_to..], &mut kv_cache, trim_to, opts.return_logits, false)?
        } else if chunk_size > 0 {
            debug!(chunk_size, "concurrent_prefill: running chunked prefill");
            let result = self.model.prefill_with_token_chunked(input_tokens, &mut kv_cache, chunk_size, opts.return_logits, false)?;
            // Save to prefix cache (short Mutex hold)
            if let Some(sys_len) = opts.system_prefix_len {
                if sys_len > 0 && sys_len < input_tokens.len() {
                    if let Some(ref pc_mutex) = self.prefix_cache {
                        let mut pc = pc_mutex.lock().unwrap();
                        if let Err(e) = pc.save_system(&input_tokens[..sys_len], &kv_cache) {
                            warn!("Failed to save system prefix cache: {}", e);
                        }
                    }
                }
            }
            if let Some(ref pc_mutex) = self.prefix_cache {
                let mut pc = pc_mutex.lock().unwrap();
                if let Err(e) = pc.save(input_tokens, &kv_cache) {
                    warn!("Failed to save prefix cache: {}", e);
                }
            }
            result
        } else {
            debug!("concurrent_prefill: running prefill_with_token");
            let result = self.model.prefill_with_token(input_tokens, &mut kv_cache, 0, opts.return_logits, false)?;
            // Save to prefix cache (short Mutex hold)
            if let Some(sys_len) = opts.system_prefix_len {
                if sys_len > 0 && sys_len < input_tokens.len() {
                    if let Some(ref pc_mutex) = self.prefix_cache {
                        let mut pc = pc_mutex.lock().unwrap();
                        if let Err(e) = pc.save_system(&input_tokens[..sys_len], &kv_cache) {
                            warn!("Failed to save system prefix cache: {}", e);
                        }
                    }
                }
            }
            if let Some(ref pc_mutex) = self.prefix_cache {
                let mut pc = pc_mutex.lock().unwrap();
                if let Err(e) = pc.save(input_tokens, &kv_cache) {
                    warn!("Failed to save prefix cache: {}", e);
                }
            }
            result
        };

        // Report prefill completion (N/N)
        crate::prefill_progress::report(input_tokens.len(), input_tokens.len());

        info!(first_token, elapsed_ms = (start_time.elapsed().as_secs_f64() * 1000.0) as u64, "concurrent_prefill: done");
        let elapsed = start_time.elapsed().as_secs_f64();

        let timings = if opts.profile {
            Some(LayerTimings {
                layer_times,
                total_time: elapsed,
            })
        } else {
            None
        };

        Ok((
            kv_cache,
            PrefillOutput {
                logits,
                first_token,
                timings,
                cached_prefix_len: prefix_hit_len,
            },
        ))
    }

    /// Run one decode step, producing the next token.
    ///
    /// Takes `&self` — all mutable state is in the caller-owned `kv_cache`.
    pub fn decode_next(
        &self,
        kv_cache: &mut CpuKvCache,
        token: u32,
        opts: RunOpts,
    ) -> Result<DecodeOutput> {
        execution_phase::set_phase(ExecutionPhase::Decode);
        let start_time = std::time::Instant::now();

        let (logits, next_token, layer_times) = if self.model.is_mrope() {
            self.model.decode_step_mrope(token, kv_cache, opts.return_logits)?
        } else {
            self.model.decode_step(token, kv_cache, opts.return_logits)?
        };

        let elapsed = start_time.elapsed().as_secs_f64();

        let timings = if opts.profile {
            Some(LayerTimings {
                layer_times,
                total_time: elapsed,
            })
        } else {
            None
        };

        Ok(DecodeOutput {
            logits,
            token: next_token,
            timings,
        })
    }
}

// ============================================================================
// ConcurrentInference: object-safe trait for runtime backend dispatch
// ============================================================================

/// Object-safe trait for concurrent inference backends.
///
/// Allows different backend types (Q4, BF16) to be used interchangeably
/// behind `Arc<dyn ConcurrentInference>` in the HTTP server.
pub trait ConcurrentInference: Send + Sync {
    /// Run prefill on a fresh KV cache.
    fn prefill(&self, input_tokens: &[u32], opts: RunOpts)
        -> Result<(CpuKvCache, PrefillOutput)>;

    /// Run one decode step.
    fn decode_next(&self, kv_cache: &mut CpuKvCache, token: u32, opts: RunOpts)
        -> Result<DecodeOutput>;

    /// Get model configuration.
    fn config(&self) -> &Config;

    /// Whether this model uses MRoPE (VL model).
    fn is_mrope(&self) -> bool;
}

impl<C: CpuBackendConfig> ConcurrentInference for ConcurrentBackend<C>
where
    <C::Ops as LinearOps>::Weight: CacheWeight,
{
    fn prefill(&self, input_tokens: &[u32], opts: RunOpts)
        -> Result<(CpuKvCache, PrefillOutput)>
    {
        ConcurrentBackend::prefill(self, input_tokens, opts)
    }

    fn decode_next(&self, kv_cache: &mut CpuKvCache, token: u32, opts: RunOpts)
        -> Result<DecodeOutput>
    {
        ConcurrentBackend::decode_next(self, kv_cache, token, opts)
    }

    fn config(&self) -> &Config {
        ConcurrentBackend::config(self)
    }

    fn is_mrope(&self) -> bool {
        ConcurrentBackend::is_mrope(self)
    }
}
