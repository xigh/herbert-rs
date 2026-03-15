//! HTTP route handlers and inference execution.

use crate::metrics::RequestStats;
use crate::prompt::*;
use crate::session_log::{SessionLog, DECODE_FLUSH_INTERVAL};
use crate::sse::*;
use crate::types::*;
use crate::AppState;
use herbert_core::decode_utils::{detect_repetition_loop, StopReason};
use herbert_core::sampler::{self, Sampler, SamplerConfig, ThinkBudget};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::Sse;
use axum::response::{IntoResponse, Json};
use herbert_core::backend::RunOpts;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tracing::{error, info, warn};
use uuid::Uuid;

pub async fn health_check(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "model": state.scheduler.as_ref().map(|s| s.model_name.as_str()).unwrap_or("none"),
        "embeddings": state.embed.is_some(),
        "embed_only": state.scheduler.is_none(),
    }))
}

pub async fn metrics_handler(
    State(state): State<Arc<AppState>>,
) -> axum::response::Response {
    match &state.scheduler {
        Some(s) => Json(s.metrics.summary()).into_response(),
        None => make_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "api_error",
            "LLM not loaded (embed-only mode)",
        ).into_response(),
    }
}

pub async fn count_tokens_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<CountTokensRequest>,
) -> Result<Json<CountTokensResponse>, axum::response::Response> {
    if let Err(resp) = check_auth(&state, &headers) {
        return Err(resp.into_response());
    }

    let scheduler = state.scheduler.as_ref().ok_or_else(|| {
        make_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "api_error",
            "LLM not loaded (embed-only mode)",
        ).into_response()
    })?;

    let prompt = messages_to_prompt(&req.system, &req.messages, &[]);
    let encoding = scheduler
        .tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| {
            make_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                &format!("Tokenization failed: {}", e),
            )
            .into_response()
        })?;

    Ok(Json(CountTokensResponse {
        input_tokens: encoding.get_ids().len(),
    }))
}

pub async fn tokenize_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<TokenizeRequest>,
) -> axum::response::Response {
    if let Err(resp) = check_auth(&state, &headers) {
        return resp.into_response();
    }

    let (tokenizer, model_name) = match req.model.as_str() {
        "chat" => match &state.scheduler {
            Some(s) => (Arc::clone(&s.tokenizer), s.model_name.clone()),
            None => {
                return make_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "api_error",
                    "LLM not loaded (embed-only mode), use model=\"embed\" instead",
                ).into_response();
            }
        },
        _ => match &state.embed {
            Some(e) => (Arc::clone(&e.tokenizer), e.model_name.clone()),
            None => match &state.scheduler {
                Some(s) => (Arc::clone(&s.tokenizer), s.model_name.clone()),
                None => {
                    return make_error_response(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "api_error",
                        "No tokenizer available",
                    ).into_response();
                }
            },
        },
    };

    let encoding = match tokenizer.encode(req.text.as_str(), false) {
        Ok(enc) => enc,
        Err(e) => {
            return make_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                &format!("Tokenization failed: {}", e),
            )
            .into_response();
        }
    };

    let ids = encoding.get_ids().to_vec();
    let tokens: Vec<String> = ids
        .iter()
        .map(|&id| tokenizer.decode(&[id], false).unwrap_or_default())
        .collect();
    let count = ids.len();

    Json(TokenizeResponse {
        token_ids: ids,
        tokens,
        count,
        model: model_name,
    })
    .into_response()
}

pub async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<EmbeddingsRequest>,
) -> axum::response::Response {
    if let Err(resp) = check_auth(&state, &headers) {
        return resp.into_response();
    }

    let embed = match &state.embed {
        Some(e) => e,
        None => {
            return make_error_response(
                StatusCode::NOT_FOUND,
                "not_found",
                "Embedding endpoint not enabled. Start server with --embed-model <path>",
            )
            .into_response();
        }
    };

    if req.encoding_format != "float" {
        return make_error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            &format!(
                "Unsupported encoding_format: \"{}\". Only \"float\" is supported.",
                req.encoding_format
            ),
        )
        .into_response();
    }

    let texts: Vec<String> = match req.input {
        EmbeddingInput::Single(s) => vec![s],
        EmbeddingInput::Batch(v) => v,
    };

    if texts.is_empty() {
        return make_error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Input must not be empty",
        )
        .into_response();
    }

    let _permit = embed.semaphore.acquire().await.unwrap();

    let tokenizer = Arc::clone(&embed.tokenizer);
    let model_name = embed.model_name.clone();

    // Tokenize all inputs outside the blocking task (tokenizer is Send+Sync)
    let mut all_token_ids: Vec<Vec<u32>> = Vec::with_capacity(texts.len());
    let mut total_tokens: usize = 0;
    for text in &texts {
        let prompt = format!(
            "<|im_start|>system\nRepresent the user's input.<|im_end|>\n\
             <|im_start|>user\n{}<|im_end|>\n\
             <|im_start|>assistant\n",
            text
        );
        let encoding = match tokenizer.encode(prompt.as_str(), false) {
            Ok(enc) => enc,
            Err(e) => {
                return make_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    &format!("Tokenization failed: {}", e),
                )
                .into_response();
            }
        };
        let ids = encoding.get_ids().to_vec();
        total_tokens += ids.len();
        all_token_ids.push(ids);
    }

    // Run embed calls sequentially in a single blocking task (Backend::embed takes &mut self)
    let state_clone = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<(Vec<Vec<f32>>, f64)> {
        let embed = state_clone.embed.as_ref().unwrap();
        let mut backend = embed.backend.lock().unwrap();
        let mut results = Vec::with_capacity(all_token_ids.len());
        let t0 = Instant::now();
        for tokens in &all_token_ids {
            let embedding = backend
                .embed(tokens)
                .map_err(|e| anyhow::anyhow!("Embedding failed: {}", e))?;
            results.push(embedding);
        }
        let duration_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok((results, duration_ms))
    })
    .await;

    let (embeddings, duration_ms) = match result {
        Ok(Ok(pair)) => pair,
        Ok(Err(e)) => {
            error!("Embedding error: {}", e);
            return make_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                &format!("Embedding failed: {}", e),
            )
            .into_response();
        }
        Err(e) => {
            error!("Embedding task panicked: {}", e);
            return make_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "Internal error during embedding",
            )
            .into_response();
        }
    };

    let data: Vec<EmbeddingData> = embeddings
        .into_iter()
        .enumerate()
        .map(|(i, embedding)| EmbeddingData {
            object: "embedding",
            embedding,
            index: i,
        })
        .collect();

    let tokens_per_second = if duration_ms > 0.0 {
        total_tokens as f64 / (duration_ms / 1000.0)
    } else {
        0.0
    };

    info!(
        count = data.len(),
        total_tokens = total_tokens,
        duration_ms = format!("{:.1}", duration_ms),
        tok_s = format!("{:.1}", tokens_per_second),
        model = %model_name,
        "Embeddings complete"
    );

    Json(EmbeddingsResponse {
        object: "list",
        data,
        model: model_name,
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
            tokens_per_second: (tokens_per_second * 10.0).round() / 10.0,
            duration_ms: (duration_ms * 100.0).round() / 100.0,
        },
    })
    .into_response()
}

pub async fn messages_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<MessagesRequest>,
) -> axum::response::Response {
    if let Err(resp) = check_auth(&state, &headers) {
        return resp.into_response();
    }

    let is_stream = req.stream.unwrap_or(false);

    // Parse dispatch priority from X-Priority header (0=highest, 9=lowest, default=5)
    let priority: u8 = headers
        .get("x-priority")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .map(|p: u8| p.min(9))
        .unwrap_or(5);

    if is_stream {
        // ── Streaming mode (SSE) ──
        let (tx, rx) = mpsc::channel::<SsePayload>(64);
        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(e) = run_inference(state_clone, req, tx.clone(), priority).await {
                error!("Inference error: {}", e);
                let _ = tx.send(SsePayload::Error { message: format!("{}", e) }).await;
            }
        });
        let stream = ReceiverStream::new(rx).map(|payload| Ok::<_, Infallible>(sse_event(&payload)));
        Sse::new(stream)
            .keep_alive(
                axum::response::sse::KeepAlive::new()
                    .interval(std::time::Duration::from_secs(15)),
            )
            .into_response()
    } else {
        // ── Non-streaming mode (JSON) ──
        let (tx, mut rx) = mpsc::channel::<SsePayload>(64);
        let state_clone = Arc::clone(&state);

        tokio::spawn(async move {
            if let Err(e) = run_inference(state_clone, req, tx.clone(), priority).await {
                error!("Inference error (non-stream): {}", e);
                let _ = tx.send(SsePayload::Error { message: format!("{}", e) }).await;
            }
        });

        // Collect SsePayload events into a non-streaming response
        let mut msg_id = String::new();
        let mut model = String::new();
        let mut content_blocks: Vec<serde_json::Value> = Vec::new();
        let mut current_text = String::new();
        let mut current_thinking = String::new();
        let mut current_tool: Option<(String, String, String)> = None; // (id, name, json)
        let mut current_block_type = String::new();
        let mut stop_reason = "end_turn".to_string();
        let mut output_tokens: usize = 0;
        let mut input_tokens: usize = 0;
        let mut error_message: Option<String> = None;

        while let Some(payload) = rx.recv().await {
            match payload {
                SsePayload::MessageStart { msg_id: id, model: m, input_tokens: it } => {
                    msg_id = id;
                    model = m;
                    input_tokens = it;
                }
                SsePayload::ContentBlockStart { block_type, .. } => {
                    current_block_type = block_type;
                }
                SsePayload::ToolUseStart { id, name, .. } => {
                    current_tool = Some((id, name, String::new()));
                }
                SsePayload::ThinkingDelta { thinking, .. } => {
                    current_thinking.push_str(&thinking);
                }
                SsePayload::TextDelta { text, .. } => {
                    current_text.push_str(&text);
                }
                SsePayload::InputJsonDelta { partial_json, .. } => {
                    if let Some((_, _, ref mut json)) = current_tool {
                        json.push_str(&partial_json);
                    }
                }
                SsePayload::ContentBlockStop { .. } => {
                    if let Some((id, name, json_str)) = current_tool.take() {
                        let input: serde_json::Value = serde_json::from_str(&json_str)
                            .unwrap_or(serde_json::Value::Object(Default::default()));
                        content_blocks.push(serde_json::json!({
                            "type": "tool_use", "id": id, "name": name, "input": input
                        }));
                    } else if current_block_type == "thinking" {
                        content_blocks.push(serde_json::json!({
                            "type": "thinking", "thinking": std::mem::take(&mut current_thinking)
                        }));
                    } else {
                        content_blocks.push(serde_json::json!({
                            "type": "text", "text": std::mem::take(&mut current_text)
                        }));
                    }
                    current_block_type.clear();
                }
                SsePayload::MessageDelta { stop_reason: sr, output_tokens: ot } => {
                    stop_reason = sr;
                    output_tokens = ot;
                }
                SsePayload::Error { message } => {
                    error_message = Some(message);
                }
                SsePayload::Queue { .. } | SsePayload::PrefillProgress { .. } | SsePayload::MessageStop => {}
            }
        }

        if let Some(err_msg) = error_message {
            return make_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                &err_msg,
            )
            .into_response();
        }

        // If no content blocks were produced, add empty text block
        if content_blocks.is_empty() {
            content_blocks.push(serde_json::json!({"type": "text", "text": ""}));
        }

        let response = serde_json::json!({
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": model,
            "stop_reason": stop_reason,
            "stop_sequence": null,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        });

        Json(response).into_response()
    }
}

fn check_auth(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let expected = match &state.api_key {
        Some(k) => k,
        None => return Ok(()),
    };

    if let Some(val) = headers.get("x-api-key") {
        if let Ok(key) = val.to_str() {
            if key == expected {
                return Ok(());
            }
        }
    }

    if let Some(val) = headers.get("authorization") {
        if let Ok(auth) = val.to_str() {
            if let Some(token) = auth.strip_prefix("Bearer ") {
                if token == expected {
                    return Ok(());
                }
            }
        }
    }

    Err((
        StatusCode::UNAUTHORIZED,
        Json(ErrorResponse {
            error_type: "error".to_string(),
            error: ErrorDetail {
                error_type: "authentication_error".to_string(),
                message: "Invalid API key".to_string(),
            },
        }),
    ))
}

fn make_error_response(status: StatusCode, error_type: &str, message: &str) -> impl IntoResponse {
    (
        status,
        Json(ErrorResponse {
            error_type: "error".to_string(),
            error: ErrorDetail {
                error_type: error_type.to_string(),
                message: message.to_string(),
            },
        }),
    )
}

async fn run_inference(
    state: Arc<AppState>,
    req: MessagesRequest,
    tx: mpsc::Sender<SsePayload>,
    priority: u8,
) -> anyhow::Result<()> {
    let sched = state.scheduler.as_ref()
        .ok_or_else(|| anyhow::anyhow!("LLM not loaded (embed-only mode)"))?;
    let max_concurrent = sched.max_concurrent;

    // Fast path: try to acquire a permit immediately
    let _permit = match sched.semaphore.try_acquire() {
        Ok(permit) => permit,
        Err(_) => {
            // All slots busy — enter queue polling loop with SSE notifications
            sched.metrics.waiting_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let _ = tx
                .send(SsePayload::Queue {
                    position: sched.metrics.waiting_count.load(std::sync::atomic::Ordering::Relaxed),
                    active: sched.metrics.active_requests.load(std::sync::atomic::Ordering::Relaxed),
                    max_concurrent,
                })
                .await;

            let permit = loop {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(1),
                    sched.semaphore.acquire(),
                )
                .await
                {
                    Ok(Ok(permit)) => break permit,
                    Ok(Err(_)) => {
                        sched.metrics.waiting_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                        return Err(anyhow::anyhow!("Server shutting down"));
                    }
                    Err(_timeout) => {
                        // Still waiting — send updated queue position
                        let _ = tx
                            .send(SsePayload::Queue {
                                position: sched.metrics.waiting_count.load(std::sync::atomic::Ordering::Relaxed),
                                active: sched.metrics.active_requests.load(std::sync::atomic::Ordering::Relaxed),
                                max_concurrent,
                            })
                            .await;
                    }
                }
            };
            sched.metrics.waiting_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            permit
        }
    };

    sched.metrics.request_started();
    let msg_id = generate_msg_id();
    let max_tokens = req.max_tokens;
    let seed = req.seed;
    let stop_sequences = req.stop_sequences;

    let log_system = extract_system_text(&req.system);
    let log_messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), extract_message_text(&m.content)))
        .collect();

    let has_tools = !req.tools.is_empty();
    if has_tools {
        info!(num_tools = req.tools.len(), "Tools provided in request");
    }
    let prompt = messages_to_prompt(&req.system, &req.messages, &req.tools);

    let model_name = sched.model_name.clone();

    let encoding = sched
        .tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let tokens = encoding.get_ids().to_vec();
    let input_token_count = tokens.len();

    if tokens.is_empty() {
        return Err(anyhow::anyhow!("Prompt produced zero tokens"));
    }

    info!(
        input_tokens = input_token_count,
        max_tokens = max_tokens,
        priority = priority,
        "Starting inference"
    );

    let temperature = req.temperature.unwrap_or(sched.default_temperature);
    let top_k = req.top_k.unwrap_or(sched.default_top_k);
    let top_p = req.top_p.unwrap_or(sched.default_top_p);
    let use_sampling = temperature > 0.0;

    let effective_think_budget = if sched.nothink { 1 } else { sched.think_budget };
    let think_budget_active = ThinkBudget::new(
        effective_think_budget,
        sched.think_open_id,
        sched.think_close_id,
        sched.newline_id,
    )
    .is_active();
    let need_logits = use_sampling || think_budget_active;

    let eos_token_id = sched.eos_token_id;
    let im_end_id = sched.im_end_id;
    let think_open_id = sched.think_open_id;
    let think_close_id = sched.think_close_id;
    let nothink = sched.nothink;
    let newline_id = sched.newline_id;
    let tool_call_open_id = if has_tools { sched.tool_call_open_id } else { None };
    let tool_call_close_id = if has_tools { sched.tool_call_close_id } else { None };

    // Clone Arc refs for the blocking task
    let backend = Arc::clone(&sched.backend);
    let tokenizer = Arc::clone(&sched.tokenizer);

    // Create incremental session log (writes file immediately with status=prefilling)
    let mut session = SessionLog::new(
        msg_id.clone(),
        model_name.clone(),
        max_tokens,
        temperature,
        top_k,
        top_p,
        log_system.clone(),
        log_messages.clone(),
        input_token_count,
    );

    // Send message_start before entering blocking context
    let _ = tx
        .send(SsePayload::MessageStart {
            msg_id: msg_id.clone(),
            model: model_name.clone(),
            input_tokens: input_token_count,
        })
        .await;

    let run_opts = RunOpts {
        return_logits: need_logits,
        profile: false,
        decode_tokens: Some(max_tokens),
        ignore_eos: false,
        system_prefix_len: None,
    };

    // Run all CPU-bound inference + SSE streaming in a blocking thread.
    // Returns only the data needed for session logging.
    let tx_progress = tx.clone();
    let log_data = tokio::task::spawn_blocking(move || -> anyhow::Result<LogData> {
        let session = &mut session;
        // Set dispatch priority for this request's thread pool calls
        herbert_backend_common::thread_pool::set_dispatch_priority(priority);

        // Install prefill progress callback (sends SSE events + logs progress)
        herbert_backend_common::prefill_progress::set_callback(Some(Box::new(
            move |done, total| {
                info!(done, total, "Prefill progress");
                let _ = tx_progress.blocking_send(SsePayload::PrefillProgress {
                    tokens_processed: done,
                    tokens_total: total,
                });
            },
        )));

        let sampler_config = SamplerConfig {
            temperature,
            top_k,
            top_p,
        };
        let mut sampler = match seed {
            Some(s) => Sampler::new_with_seed(sampler_config, s),
            None => Sampler::new(sampler_config),
        };
        let use_sampling = !sampler.is_greedy();

        let effective_think_budget = if nothink { 1 } else { effective_think_budget };
        let mut think_budget = ThinkBudget::new(
            effective_think_budget,
            think_open_id,
            think_close_id,
            newline_id,
        );

        let prefill_start = Instant::now();
        let (mut kv, prefill_output) = backend.prefill(&tokens, run_opts.clone())?;
        let prefill_us = prefill_start.elapsed().as_micros() as u64;
        let prefix_hit_len = prefill_output.cached_prefix_len;

        // Clear prefill progress callback (no longer needed during decode)
        herbert_backend_common::prefill_progress::set_callback(None);

        // Session log: prefill done, entering decode
        if let Some(s) = session.as_mut() {
            s.status = "decoding".to_string();
            s.prefill_us = prefill_us;
            s.flush();
        }

        let decode_start = Instant::now();

        let first_token = if let Some(ref logits) = prefill_output.logits {
            let mut logits_buf = logits.clone();
            think_budget.apply_to_logits(&mut logits_buf);
            if use_sampling {
                sampler.sample(&logits_buf)
            } else {
                sampler::argmax(&logits_buf)
            }
        } else {
            prefill_output.first_token
        };
        think_budget.track(first_token);

        let mut all_tokens: Vec<u32> = vec![first_token];
        let mut output_tokens: usize = 0;
        let mut stop_reason = StopReason::MaxTokens;

        let mut in_think = false;
        let mut text_block_started = false;
        let mut current_block_index: usize = 0;

        let mut in_tool_call = false;
        let mut tool_call_tokens: Vec<u32> = Vec::new();
        let mut tool_calls_emitted = false;

        let decode_opts = RunOpts {
            return_logits: need_logits,
            profile: false,
            decode_tokens: Some(max_tokens),
            ignore_eos: false,
            system_prefix_len: None,
        };

        let mut current_token = first_token;
        let mut generated_text = String::new();
        let mut thinking_text = String::new();
        let use_stop_sequences = !stop_sequences.is_empty();

        // Process first token
        {
            output_tokens += 1;

            if current_token == think_open_id {
                in_think = true;
                let _ = tx.blocking_send(SsePayload::ContentBlockStart {
                    index: current_block_index,
                    block_type: "thinking".to_string(),
                });
            } else if tool_call_open_id == Some(current_token) {
                in_tool_call = true;
                tool_call_tokens.clear();
            } else if current_token == eos_token_id {
                stop_reason = StopReason::Eos;
            } else if current_token == im_end_id {
                stop_reason = StopReason::ImEnd;
            } else {
                if !text_block_started {
                    let _ = tx.blocking_send(SsePayload::ContentBlockStart {
                        index: current_block_index,
                        block_type: "text".to_string(),
                    });
                    text_block_started = true;
                }
                let text = tokenizer
                    .decode(&[current_token], true)
                    .unwrap_or_default();
                if !text.is_empty() {
                    generated_text.push_str(&text);
                    let _ = tx.blocking_send(SsePayload::TextDelta {
                        index: current_block_index,
                        text,
                    });
                }
            }
        }

        // Decode loop
        if stop_reason == StopReason::MaxTokens {
            for _step in 1..max_tokens {
                let output = backend.decode_next(&mut kv, current_token, decode_opts.clone())?;

                let next_token = if let Some(mut logits) = output.logits {
                    think_budget.apply_to_logits(&mut logits);
                    if use_sampling {
                        sampler.sample(&logits)
                    } else {
                        sampler::argmax(&logits)
                    }
                } else {
                    output.token
                };
                think_budget.track(next_token);
                all_tokens.push(next_token);
                output_tokens += 1;
                current_token = next_token;

                // Per-token decode logging (compile with --features decode-debug)
                #[cfg(feature = "decode-debug")]
                {
                    let tok_text = tokenizer.decode(&[next_token], true).unwrap_or_default();
                    tracing::debug!(
                        step = output_tokens,
                        token_id = next_token,
                        text = %tok_text,
                        "decode"
                    );
                }

                // Periodic session log flush during decode
                if output_tokens % DECODE_FLUSH_INTERVAL == 0 {
                    if let Some(s) = session.as_mut() {
                        s.output_tokens = output_tokens;
                        s.thinking_text = thinking_text.clone();
                        s.generated_text = generated_text.clone();
                        s.decode_us = decode_start.elapsed().as_micros() as u64;
                        s.flush();
                    }
                }

                // EOS/im_end checks FIRST — must break even inside tool calls or thinking
                if next_token == eos_token_id {
                    stop_reason = StopReason::Eos;
                    break;
                }
                if next_token == im_end_id {
                    stop_reason = StopReason::ImEnd;
                    break;
                }

                if tool_call_open_id == Some(next_token) {
                    if text_block_started {
                        let _ = tx.blocking_send(SsePayload::ContentBlockStop {
                            index: current_block_index,
                        });
                        current_block_index += 1;
                        text_block_started = false;
                    }
                    in_tool_call = true;
                    tool_call_tokens.clear();
                    continue;
                }

                if in_tool_call && tool_call_close_id == Some(next_token) {
                    in_tool_call = false;
                    let tool_json_str = tokenizer
                        .decode(&tool_call_tokens, true)
                        .unwrap_or_default();
                    let tool_json_str = tool_json_str.trim();
                    info!(tool_json = tool_json_str, "Tool call detected (token)");

                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tool_json_str) {
                        let tool_name = parsed.get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        let tool_args = parsed.get("arguments")
                            .cloned()
                            .unwrap_or(serde_json::json!({}));
                        let tool_id = format!("toolu_{}", Uuid::new_v4().to_string().replace('-', ""));

                        let _ = tx.blocking_send(SsePayload::ToolUseStart {
                            index: current_block_index,
                            id: tool_id,
                            name: tool_name,
                        });

                        let args_str = tool_args.to_string();
                        let _ = tx.blocking_send(SsePayload::InputJsonDelta {
                            index: current_block_index,
                            partial_json: args_str,
                        });

                        let _ = tx.blocking_send(SsePayload::ContentBlockStop {
                            index: current_block_index,
                        });
                        current_block_index += 1;
                        tool_calls_emitted = true;
                    } else {
                        warn!(json = tool_json_str, "Failed to parse tool call JSON, emitting as text");
                        if !text_block_started {
                            let _ = tx.blocking_send(SsePayload::ContentBlockStart {
                                index: current_block_index,
                                block_type: "text".to_string(),
                            });
                            text_block_started = true;
                        }
                        let fallback = format!("<tool_call>\n{}\n</tool_call>", tool_json_str);
                        generated_text.push_str(&fallback);
                        let _ = tx.blocking_send(SsePayload::TextDelta {
                            index: current_block_index,
                            text: fallback,
                        });
                    }
                    continue;
                }

                if in_tool_call {
                    tool_call_tokens.push(next_token);
                    continue;
                }

                if next_token == think_open_id {
                    if !in_think {
                        in_think = true;
                        let _ = tx.blocking_send(SsePayload::ContentBlockStart {
                            index: current_block_index,
                            block_type: "thinking".to_string(),
                        });
                    }
                    continue;
                }

                if next_token == think_close_id {
                    if in_think {
                        let _ = tx.blocking_send(SsePayload::ContentBlockStop {
                            index: current_block_index,
                        });
                        current_block_index += 1;
                        in_think = false;
                    }
                    continue;
                }

                if in_think {
                    let token_text = tokenizer
                        .decode(&[next_token], true)
                        .unwrap_or_default();
                    if !token_text.is_empty() {
                        thinking_text.push_str(&token_text);
                        let _ = tx.blocking_send(SsePayload::ThinkingDelta {
                            index: current_block_index,
                            thinking: token_text,
                        });
                    }
                } else {
                    if !text_block_started {
                        let _ = tx.blocking_send(SsePayload::ContentBlockStart {
                            index: current_block_index,
                            block_type: "text".to_string(),
                        });
                        text_block_started = true;
                    }

                    let token_text = tokenizer
                        .decode(&[next_token], true)
                        .unwrap_or_default();
                    if !token_text.is_empty() {
                        generated_text.push_str(&token_text);
                        let _ = tx.blocking_send(SsePayload::TextDelta {
                            index: current_block_index,
                            text: token_text,
                        });
                    }
                }

                if use_stop_sequences
                    && find_stop_sequence(&generated_text, &stop_sequences).is_some()
                {
                    stop_reason = StopReason::Eos;
                    break;
                }

                if let Some(pat_len) = detect_repetition_loop(&all_tokens) {
                    info!(pattern_len = pat_len, "Repetition loop detected, stopping");
                    stop_reason = StopReason::RepetitionLoop;
                    break;
                }

                if tx.is_closed() {
                    info!("Client disconnected, stopping inference");
                    return Ok(LogData {
                        output_tokens,
                        stop_reason_str: "client_disconnect".to_string(),
                        thinking_text,
                        generated_text,
                        prefill_us,
                        decode_us: decode_start.elapsed().as_micros() as u64,
                        prefix_hit_len,
                        session: session.take(),
                    });
                }
            }
        }

        // Fallback 1: incomplete token-based tool call (model forgot </tool_call>)
        if in_tool_call && !tool_call_tokens.is_empty() && !tool_calls_emitted {
            let tool_json_str = tokenizer
                .decode(&tool_call_tokens, true)
                .unwrap_or_default();
            let tool_json_str = tool_json_str.trim();
            info!(tool_json = tool_json_str, "Incomplete tool call (no close token), attempting parse");

            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tool_json_str) {
                let tool_name = parsed.get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let tool_args = parsed.get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::json!({}));
                let tool_id = format!("toolu_{}", Uuid::new_v4().to_string().replace('-', ""));

                let _ = tx.blocking_send(SsePayload::ToolUseStart {
                    index: current_block_index,
                    id: tool_id,
                    name: tool_name,
                });
                let args_str = tool_args.to_string();
                let _ = tx.blocking_send(SsePayload::InputJsonDelta {
                    index: current_block_index,
                    partial_json: args_str,
                });
                let _ = tx.blocking_send(SsePayload::ContentBlockStop {
                    index: current_block_index,
                });
                current_block_index += 1;
                tool_calls_emitted = true;
                in_tool_call = false;
            }
        }

        // Fallback 2: text-based tool call detection (model used <tool_call> as text, not special token)
        if !tool_calls_emitted && generated_text.contains("<tool_call>") {
            let mut search_from = 0;
            loop {
                let start_tag = "<tool_call>";
                let end_tag = "</tool_call>";
                let start = match generated_text[search_from..].find(start_tag) {
                    Some(pos) => search_from + pos + start_tag.len(),
                    None => break,
                };
                let end = match generated_text[start..].find(end_tag) {
                    Some(pos) => start + pos,
                    None => break,
                };
                let body = generated_text[start..end].trim();
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(body) {
                    let tool_name = parsed.get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let tool_args = parsed.get("arguments")
                        .cloned()
                        .unwrap_or(serde_json::json!({}));
                    let tool_id = format!("toolu_{}", Uuid::new_v4().to_string().replace('-', ""));

                    info!(name = %tool_name, "Tool call detected (text fallback)");

                    // Close any open text block before emitting tool_use
                    if text_block_started {
                        let _ = tx.blocking_send(SsePayload::ContentBlockStop {
                            index: current_block_index,
                        });
                        current_block_index += 1;
                        text_block_started = false;
                    }

                    let _ = tx.blocking_send(SsePayload::ToolUseStart {
                        index: current_block_index,
                        id: tool_id,
                        name: tool_name,
                    });
                    let args_str = tool_args.to_string();
                    let _ = tx.blocking_send(SsePayload::InputJsonDelta {
                        index: current_block_index,
                        partial_json: args_str,
                    });
                    let _ = tx.blocking_send(SsePayload::ContentBlockStop {
                        index: current_block_index,
                    });
                    current_block_index += 1;
                    tool_calls_emitted = true;
                }
                search_from = end + end_tag.len();
            }
        }

        // Close open content blocks
        if in_think {
            let _ = tx.blocking_send(SsePayload::ContentBlockStop {
                index: current_block_index,
            });
            current_block_index += 1;
        }
        if text_block_started {
            let _ = tx.blocking_send(SsePayload::ContentBlockStop {
                index: current_block_index,
            });
        } else if !in_think && !tool_calls_emitted {
            let _ = tx.blocking_send(SsePayload::ContentBlockStart {
                index: current_block_index,
                block_type: "text".to_string(),
            });
            let _ = tx.blocking_send(SsePayload::ContentBlockStop {
                index: current_block_index,
            });
        }

        let anthropic_stop_reason = if tool_calls_emitted {
            "tool_use"
        } else {
            match stop_reason {
                StopReason::Eos | StopReason::ImEnd | StopReason::RepetitionLoop => "end_turn",
                StopReason::MaxTokens => "max_tokens",
            }
        };

        let _ = tx.blocking_send(SsePayload::MessageDelta {
            stop_reason: anthropic_stop_reason.to_string(),
            output_tokens,
        });

        let _ = tx.blocking_send(SsePayload::MessageStop);

        let decode_us = decode_start.elapsed().as_micros() as u64;
        let decode_tok_s = if decode_us > 0 && output_tokens > 1 {
            ((output_tokens - 1) as f64) / (decode_us as f64 / 1_000_000.0)
        } else {
            0.0
        };
        let prefill_tok_s = if prefill_us > 0 {
            (input_token_count as f64) / (prefill_us as f64 / 1_000_000.0)
        } else {
            0.0
        };

        info!(
            output_tokens = output_tokens,
            stop_reason = anthropic_stop_reason,
            prefill_tok_s = format!("{:.1}", prefill_tok_s),
            decode_tok_s = format!("{:.1}", decode_tok_s),
            prefix_hit_len = prefix_hit_len,
            "Inference complete"
        );

        Ok(LogData {
            output_tokens,
            stop_reason_str: anthropic_stop_reason.to_string(),
            thinking_text,
            generated_text,
            prefill_us,
            decode_us,
            prefix_hit_len,
            session: session.take(),
        })
    })
    .await??;

    // Record metrics
    sched.metrics.request_completed(&RequestStats {
        input_tokens: input_token_count,
        output_tokens: log_data.output_tokens,
        prefill_us: log_data.prefill_us,
        decode_us: log_data.decode_us,
        prefix_hit_len: log_data.prefix_hit_len,
    });

    // Final session log flush with complete data
    if let Some(mut session) = log_data.session {
        session.status = "complete".to_string();
        session.output_tokens = log_data.output_tokens;
        session.stop_reason = log_data.stop_reason_str.clone();
        session.thinking_text = log_data.thinking_text.clone();
        session.generated_text = log_data.generated_text.clone();
        session.prefill_us = log_data.prefill_us;
        session.decode_us = log_data.decode_us;
        session.flush();
    }

    Ok(())
}

/// Data returned from the blocking inference task for session logging + metrics.
struct LogData {
    output_tokens: usize,
    stop_reason_str: String,
    thinking_text: String,
    generated_text: String,
    prefill_us: u64,
    decode_us: u64,
    prefix_hit_len: usize,
    session: Option<SessionLog>,
}
