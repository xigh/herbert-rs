use crate::inference::{self, Sampler, SamplerConfig};
use crate::state::AppState;
use crate::storage;
use crate::types::{GenerationStats, Message, TokenEvent};
use herbert_core::backend::{RunOpts, VisionEmbedding};
use std::sync::atomic::Ordering;
use std::time::Instant;
use tauri::ipc::Channel;
use tauri::State;
use tracing::info;

#[tauri::command]
pub async fn generate(
    conversation_id: String,
    image_ids: Option<Vec<String>>,
    channel: Channel<TokenEvent>,
    state: State<'_, AppState>,
) -> Result<(), String> {
    // Check not already generating
    if state
        .generating
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("Already generating".to_string());
    }
    state.cancel_flag.store(false, Ordering::SeqCst);

    let data_dir = state.data_dir.clone();

    // Load conversation
    let conv = storage::load_conversation(&data_dir, &conversation_id)
        .map_err(|e| {
            state.generating.store(false, Ordering::SeqCst);
            e.to_string()
        })?;

    // Collect vision embeddings if image_ids provided
    let vision_images: Vec<(String, VisionEmbedding)> = if let Some(ref ids) = image_ids {
        let mut images = Vec::new();
        let mut embeddings_lock = state.vision_embeddings.lock().map_err(|e| {
            state.generating.store(false, Ordering::SeqCst);
            format!("Lock error: {}", e)
        })?;
        for id in ids {
            if let Some(embed) = embeddings_lock.remove(id) {
                images.push((id.clone(), embed));
            }
        }
        images
    } else {
        Vec::new()
    };
    let has_images = !vision_images.is_empty();

    // Build multi-turn prompt
    let messages: Vec<(String, String)> = conv
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    let prompt_text = if has_images {
        inference::build_multi_turn_prompt_vl(&conv.system_prompt, &messages, vision_images.len())
    } else {
        inference::build_multi_turn_prompt(&conv.system_prompt, &messages)
    };

    // Take model lock
    let mut model_lock = state.model.lock().map_err(|e| {
        state.generating.store(false, Ordering::SeqCst);
        format!("Lock error: {}", e)
    })?;
    let model_state = model_lock.as_mut().ok_or_else(|| {
        state.generating.store(false, Ordering::SeqCst);
        "Model not loaded".to_string()
    })?;

    let settings = &conv.settings;
    let eos_token_id = model_state.eos_token_id;
    let im_end_id = model_state.im_end_id;
    let think_id = model_state.think_id;

    // Load app settings to check nothink
    let app_settings = storage::load_settings(&data_dir).unwrap_or_default();
    let nothink = app_settings.nothink;

    // Tokenize prompt
    let encoding = model_state
        .tokenizer
        .encode(prompt_text.as_str(), false)
        .map_err(|e| {
            state.generating.store(false, Ordering::SeqCst);
            format!("Tokenization failed: {}", e)
        })?;
    let base_tokens = encoding.get_ids().to_vec();

    // Prefill — always request logits so we can suppress <think> on first token
    let prefill_start = Instant::now();
    let prefill_opts = RunOpts {
        return_logits: true,
        profile: false,
        decode_tokens: Some(settings.max_tokens),
        ignore_eos: false,
        system_prefix_len: None,
    };

    let (mut kv, prefill_out, prompt_len) = if has_images {
        // VL path: expand image_pad tokens and call prefill_vl
        const IMAGE_PAD_ID: u32 = 151655;
        let embeds: Vec<VisionEmbedding> = vision_images.into_iter().map(|(_, e)| e).collect();

        // Find each image_pad position and expand to num_tokens pads
        let mut expanded_tokens = Vec::with_capacity(base_tokens.len() + 1024);
        let mut image_positions = Vec::new();
        let mut embed_idx = 0;
        for &tok in &base_tokens {
            if tok == IMAGE_PAD_ID && embed_idx < embeds.len() {
                let pos = expanded_tokens.len();
                image_positions.push(pos);
                let num_vis = embeds[embed_idx].num_tokens;
                expanded_tokens.resize(expanded_tokens.len() + num_vis, IMAGE_PAD_ID);
                embed_idx += 1;
            } else {
                expanded_tokens.push(tok);
            }
        }

        let prompt_len = expanded_tokens.len();
        info!("VL Prefilling {} tokens ({} images)", prompt_len, embeds.len());

        let (kv, out) = model_state
            .backend
            .prefill_vl(&expanded_tokens, &embeds, &image_positions, prefill_opts)
            .map_err(|e| {
                state.generating.store(false, Ordering::SeqCst);
                format!("VL Prefill failed: {}", e)
            })?;
        (kv, out, prompt_len)
    } else {
        // Text-only path
        let prompt_len = base_tokens.len();
        info!("Prefilling {} tokens", prompt_len);
        let (kv, out) = model_state
            .backend
            .prefill(&base_tokens, prefill_opts)
            .map_err(|e| {
                state.generating.store(false, Ordering::SeqCst);
                format!("Prefill failed: {}", e)
            })?;
        (kv, out, prompt_len)
    };
    let prefill_ms = prefill_start.elapsed().as_millis() as u64;

    // Create sampler
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: settings.temperature,
        top_k: settings.top_k,
        top_p: settings.top_p,
    });

    // Determine first token — optionally suppress <think> when nothink is enabled
    let first_token = if let Some(mut logits) = prefill_out.logits {
        if nothink {
            if let Some(tid) = think_id {
                if (tid as usize) < logits.len() {
                    logits[tid as usize] = f32::NEG_INFINITY;
                }
            }
        }
        if sampler.is_greedy() {
            inference::argmax(&logits)
        } else {
            sampler.sample(&logits)
        }
    } else {
        prefill_out.first_token
    };

    // Start decode loop
    let decode_start = Instant::now();
    let mut all_tokens = vec![first_token];
    let mut current_token = first_token;

    // Send initial delta
    let full_text = model_state
        .tokenizer
        .decode(&all_tokens, true)
        .unwrap_or_default();
    let safe = inference::safe_len(&full_text);
    if safe > 0 {
        let _ = channel.send(TokenEvent::Delta {
            text: full_text[..safe].to_string(),
        });
    }
    let mut printed_len = safe;
    let mut prev_full = full_text;

    // Check first token for stop
    if first_token == eos_token_id || first_token == im_end_id {
        let decode_ms = decode_start.elapsed().as_millis() as u64;
        let _ = channel.send(TokenEvent::Done {
            stats: GenerationStats {
                prefill_tokens: prompt_len,
                decode_tokens: 1,
                prefill_ms,
                decode_ms,
                tokens_per_sec: if decode_ms > 0 {
                    1000.0 / decode_ms as f64
                } else {
                    0.0
                },
            },
        });
        // Save assistant message
        let generated_text = model_state
            .tokenizer
            .decode(&all_tokens, true)
            .unwrap_or_default();
        save_assistant_message(
            &data_dir,
            &conversation_id,
            &generated_text,
            prompt_len,
            1,
            prefill_ms,
            decode_ms,
        );
        state.generating.store(false, Ordering::SeqCst);
        return Ok(());
    }

    let use_sampling = !sampler.is_greedy();
    let decode_opts = RunOpts {
        return_logits: use_sampling,
        profile: false,
        decode_tokens: Some(settings.max_tokens),
        ignore_eos: false,
        system_prefix_len: None,
    };

    for _ in 1..settings.max_tokens {
        // Check cancel
        if state.cancel_flag.load(Ordering::SeqCst) {
            let _ = channel.send(TokenEvent::Cancelled);
            // Save partial
            let generated_text = model_state
                .tokenizer
                .decode(&all_tokens, true)
                .unwrap_or_default();
            let decode_ms = decode_start.elapsed().as_millis() as u64;
            save_assistant_message(
                &data_dir,
                &conversation_id,
                &generated_text,
                prompt_len,
                all_tokens.len(),
                prefill_ms,
                decode_ms,
            );
            state.generating.store(false, Ordering::SeqCst);
            return Ok(());
        }

        let output = match model_state
            .backend
            .decode_next(&mut kv, current_token, decode_opts.clone())
        {
            Ok(o) => o,
            Err(e) => {
                let _ = channel.send(TokenEvent::Error {
                    message: format!("Decode error: {}", e),
                });
                state.generating.store(false, Ordering::SeqCst);
                return Err(format!("Decode error: {}", e));
            }
        };

        let next_token = if let Some(logits) = output.logits {
            if use_sampling {
                sampler.sample(&logits)
            } else {
                inference::argmax(&logits)
            }
        } else {
            output.token
        };

        all_tokens.push(next_token);

        // Compute text delta
        let full_text = model_state
            .tokenizer
            .decode(&all_tokens, true)
            .unwrap_or_default();

        let print_from = if full_text.is_char_boundary(printed_len)
            && full_text.as_bytes().get(..printed_len)
                == prev_full.as_bytes().get(..printed_len)
        {
            printed_len
        } else {
            let common = prev_full
                .bytes()
                .zip(full_text.bytes())
                .take_while(|(a, b)| a == b)
                .count();
            let mut start = common.min(printed_len);
            while start > 0 && !full_text.is_char_boundary(start) {
                start -= 1;
            }
            start
        };

        let safe = inference::safe_len(&full_text);
        if safe > print_from {
            let _ = channel.send(TokenEvent::Delta {
                text: full_text[print_from..safe].to_string(),
            });
        }
        printed_len = safe;
        prev_full = full_text;

        // Stop conditions
        if next_token == eos_token_id || next_token == im_end_id {
            break;
        }

        // Repetition loop detection
        if let Some(pat_len) = inference::detect_repetition_loop(&all_tokens) {
            info!("Repetition loop detected ({} tokens x 3)", pat_len);
            break;
        }

        current_token = next_token;
    }

    let decode_ms = decode_start.elapsed().as_millis() as u64;
    let decode_tokens = all_tokens.len();
    let tokens_per_sec = if decode_ms > 0 {
        decode_tokens as f64 * 1000.0 / decode_ms as f64
    } else {
        0.0
    };

    let _ = channel.send(TokenEvent::Done {
        stats: GenerationStats {
            prefill_tokens: prompt_len,
            decode_tokens,
            prefill_ms,
            decode_ms,
            tokens_per_sec,
        },
    });

    // Save assistant message
    let generated_text = model_state
        .tokenizer
        .decode(&all_tokens, true)
        .unwrap_or_default();
    save_assistant_message(
        &data_dir,
        &conversation_id,
        &generated_text,
        prompt_len,
        decode_tokens,
        prefill_ms,
        decode_ms,
    );

    state.generating.store(false, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
pub async fn cancel_generation(state: State<'_, AppState>) -> Result<(), String> {
    state.cancel_flag.store(true, Ordering::SeqCst);
    Ok(())
}

fn save_assistant_message(
    data_dir: &std::path::Path,
    conversation_id: &str,
    content: &str,
    prefill_tokens: usize,
    decode_tokens: usize,
    prefill_ms: u64,
    decode_ms: u64,
) {
    if let Ok(mut conv) = storage::load_conversation(data_dir, conversation_id) {
        let tokens_per_sec = if decode_ms > 0 {
            decode_tokens as f64 * 1000.0 / decode_ms as f64
        } else {
            0.0
        };
        conv.messages.push(Message {
            role: "assistant".to_string(),
            content: content.to_string(),
            timestamp: chrono::Utc::now(),
            images: None,
            stats: Some(GenerationStats {
                prefill_tokens,
                decode_tokens,
                prefill_ms,
                decode_ms,
                tokens_per_sec,
            }),
        });
        conv.updated_at = chrono::Utc::now();
        let _ = storage::save_conversation(data_dir, &conv);
    }
}
