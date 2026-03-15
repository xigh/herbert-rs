//! SSE event types and builders for the streaming Messages API.

use axum::response::sse::Event;

#[derive(Debug)]
pub enum SsePayload {
    MessageStart {
        msg_id: String,
        model: String,
        input_tokens: usize,
    },
    ContentBlockStart {
        index: usize,
        block_type: String,
    },
    ToolUseStart {
        index: usize,
        id: String,
        name: String,
    },
    ThinkingDelta {
        index: usize,
        thinking: String,
    },
    TextDelta {
        index: usize,
        text: String,
    },
    InputJsonDelta {
        index: usize,
        partial_json: String,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        stop_reason: String,
        output_tokens: usize,
    },
    MessageStop,
    Queue {
        position: usize,
        active: usize,
        max_concurrent: usize,
    },
    PrefillProgress {
        tokens_processed: usize,
        tokens_total: usize,
    },
    Error {
        message: String,
    },
}

pub fn sse_event(payload: &SsePayload) -> Event {
    match payload {
        SsePayload::MessageStart { msg_id, model, input_tokens } => {
            let data = serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 0
                    }
                }
            });
            Event::default()
                .event("message_start")
                .data(data.to_string())
        }
        SsePayload::ContentBlockStart { index, block_type } => {
            let content_block = if block_type == "thinking" {
                serde_json::json!({"type": "thinking", "thinking": ""})
            } else {
                serde_json::json!({"type": "text", "text": ""})
            };
            let data = serde_json::json!({
                "type": "content_block_start",
                "index": index,
                "content_block": content_block,
            });
            Event::default()
                .event("content_block_start")
                .data(data.to_string())
        }
        SsePayload::ToolUseStart { index, id, name } => {
            let data = serde_json::json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": {},
                },
            });
            Event::default()
                .event("content_block_start")
                .data(data.to_string())
        }
        SsePayload::ThinkingDelta { index, thinking } => {
            let data = serde_json::json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "thinking_delta",
                    "thinking": thinking,
                }
            });
            Event::default()
                .event("content_block_delta")
                .data(data.to_string())
        }
        SsePayload::TextDelta { index, text } => {
            let data = serde_json::json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "text_delta",
                    "text": text,
                }
            });
            Event::default()
                .event("content_block_delta")
                .data(data.to_string())
        }
        SsePayload::InputJsonDelta { index, partial_json } => {
            let data = serde_json::json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": partial_json,
                }
            });
            Event::default()
                .event("content_block_delta")
                .data(data.to_string())
        }
        SsePayload::ContentBlockStop { index } => {
            let data = serde_json::json!({
                "type": "content_block_stop",
                "index": index,
            });
            Event::default()
                .event("content_block_stop")
                .data(data.to_string())
        }
        SsePayload::MessageDelta { stop_reason, output_tokens } => {
            let data = serde_json::json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": null,
                },
                "usage": {
                    "output_tokens": output_tokens,
                }
            });
            Event::default()
                .event("message_delta")
                .data(data.to_string())
        }
        SsePayload::MessageStop => {
            let data = serde_json::json!({"type": "message_stop"});
            Event::default()
                .event("message_stop")
                .data(data.to_string())
        }
        SsePayload::Queue { position, active, max_concurrent } => {
            let data = serde_json::json!({
                "type": "queue",
                "position": position,
                "active": active,
                "max_concurrent": max_concurrent,
            });
            Event::default()
                .event("queue")
                .data(data.to_string())
        }
        SsePayload::PrefillProgress { tokens_processed, tokens_total } => {
            let data = serde_json::json!({
                "type": "prefill_progress",
                "tokens_processed": tokens_processed,
                "tokens_total": tokens_total,
            });
            Event::default()
                .event("prefill_progress")
                .data(data.to_string())
        }
        SsePayload::Error { message } => {
            let data = serde_json::json!({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": message,
                }
            });
            Event::default()
                .event("error")
                .data(data.to_string())
        }
    }
}
