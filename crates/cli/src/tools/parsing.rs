use super::types::{ToolCall, KNOWN_TOOLS};

/// Extract tool calls from raw token IDs using the special token IDs.
///
/// Finds tool_call_open ... tool_call_close spans in the token stream, decodes the
/// tokens between them, and parses each tool call (Qwen3 JSON format).
pub fn extract_tool_calls_from_tokens(
    tokens: &[u32],
    tool_call_open_id: u32,
    tool_call_close_id: u32,
    tokenizer: &tokenizers::Tokenizer,
) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] == tool_call_open_id {
            if let Some(close_offset) = tokens[i + 1..].iter().position(|&t| t == tool_call_close_id) {
                let body_start = i + 1;
                let body_end = i + 1 + close_offset;
                let body_tokens = &tokens[body_start..body_end];
                if let Ok(body_str) = tokenizer.decode(body_tokens, true) {
                    let body_str = body_str.trim();
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(body_str) {
                        let name = val
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let arguments = val
                            .get("arguments")
                            .cloned()
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                        if !name.is_empty() {
                            calls.push(ToolCall { name, arguments });
                        }
                    }
                }
                i = body_end + 1;
            } else {
                break;
            }
        } else {
            i += 1;
        }
    }
    calls
}

/// Parse `<tool_call>...</tool_call>` blocks from generated text (fallback).
pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut search_from = 0;
    loop {
        let start_tag = "<tool_call>";
        let end_tag = "</tool_call>";
        let start = match text[search_from..].find(start_tag) {
            Some(pos) => search_from + pos + start_tag.len(),
            None => break,
        };
        let end = match text[start..].find(end_tag) {
            Some(pos) => start + pos,
            None => break,
        };
        let body_str = text[start..end].trim();
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(body_str) {
            let name = val
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let arguments = val
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            if !name.is_empty() {
                calls.push(ToolCall { name, arguments });
            }
        }
        search_from = end + end_tag.len();
    }
    calls
}

/// Parse a raw JSON tool call (no `<tool_call>` wrapper).
pub fn parse_raw_tool_call(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let trimmed = text.trim();
    let bytes = trimmed.as_bytes();
    let mut pos = 0;
    while pos < bytes.len() {
        let start = match trimmed[pos..].find('{') {
            Some(p) => pos + p,
            None => break,
        };
        let mut depth = 0i32;
        let mut end = start;
        for (i, &b) in bytes[start..].iter().enumerate() {
            if b == b'{' {
                depth += 1;
            } else if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    end = start + i + 1;
                    break;
                }
            }
        }
        if depth != 0 {
            break;
        }
        let json_candidate = &trimmed[start..end];
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_candidate) {
            let name = val
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if KNOWN_TOOLS.contains(&name.as_str()) {
                let arguments = val
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                calls.push(ToolCall { name, arguments });
            }
        }
        pos = end;
    }
    calls
}
