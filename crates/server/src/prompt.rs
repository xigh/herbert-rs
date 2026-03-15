//! Prompt construction from Anthropic Messages API format to Qwen3 chat template.

use crate::types::*;
use std::collections::HashMap;
use tracing::{info, warn};

pub fn generate_msg_id() -> String {
    let uuid = uuid::Uuid::new_v4().to_string().replace('-', "");
    format!("msg_{}", &uuid[..24.min(uuid.len())])
}

pub fn extract_system_text(system: &Option<SystemPrompt>) -> Option<String> {
    match system {
        None => None,
        Some(SystemPrompt::Text(s)) => {
            if s.is_empty() { None } else { Some(s.clone()) }
        }
        Some(SystemPrompt::Blocks(blocks)) => {
            let text: String = blocks
                .iter()
                .filter_map(|b| b.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n");
            if text.is_empty() { None } else { Some(text) }
        }
    }
}

pub fn extract_message_text(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(s) => s.clone(),
        MessageContent::Blocks(blocks) => {
            let mut parts = Vec::new();
            for block in blocks {
                match block {
                    ContentBlock::Text { text } => parts.push(text.clone()),
                    ContentBlock::ToolResult { content, .. } => {
                        if let Some(tc) = content {
                            match tc {
                                ToolResultContent::Text(t) => parts.push(t.clone()),
                                ToolResultContent::Blocks(bs) => {
                                    for b in bs {
                                        if let Some(ref t) = b.text {
                                            parts.push(t.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    ContentBlock::ToolUse { name, input, .. } => {
                        if let (Some(n), Some(inp)) = (name, input) {
                            parts.push(format!("[Tool use: {} with input: {}]", n, inp));
                        }
                    }
                    ContentBlock::Thinking { thinking, .. } => {
                        if let Some(t) = thinking {
                            parts.push(t.clone());
                        }
                    }
                    ContentBlock::Image { .. } => {
                        parts.push("[Image omitted]".to_string());
                    }
                }
            }
            parts.join("\n")
        }
    }
}

/// Convert Anthropic messages into Qwen3 chat template prompt.
pub fn messages_to_prompt(
    system: &Option<SystemPrompt>,
    messages: &[Message],
    tools: &[Tool],
) -> String {
    let mut prompt = String::new();

    // Build tool_use_id → name mapping for tool response formatting.
    // Qwen3 expects {"name": "tool_name", "content": "..."} in tool responses,
    // but Anthropic API uses tool_use_id references.
    let tool_id_map = build_tool_id_map(messages);

    let sys_text = extract_system_text(system);
    let has_tools = !tools.is_empty();
    if sys_text.is_some() || has_tools {
        prompt.push_str("<|im_start|>system\n");
        if let Some(ref sys) = sys_text {
            prompt.push_str(sys);
        }
        if has_tools {
            prompt.push_str(&build_tools_system_block(tools));
        }
        prompt.push_str("<|im_end|>\n");
    }

    for msg in messages {
        match msg.role.as_str() {
            "user" => {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(&format_message_content_for_prompt(&msg.content, "user", &tool_id_map));
                prompt.push_str("<|im_end|>\n");
            }
            "assistant" => {
                prompt.push_str("<|im_start|>assistant\n");
                prompt.push_str(&format_message_content_for_prompt(&msg.content, "assistant", &tool_id_map));
                prompt.push_str("<|im_end|>\n");
            }
            other => {
                warn!(role = other, "Unknown message role, treating as user");
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(&format_message_content_for_prompt(&msg.content, "user", &tool_id_map));
                prompt.push_str("<|im_end|>\n");
            }
        }
    }

    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Build a mapping from tool_use_id → tool_name by scanning ToolUse blocks.
fn build_tool_id_map(messages: &[Message]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for msg in messages {
        if let MessageContent::Blocks(blocks) = &msg.content {
            for block in blocks {
                if let ContentBlock::ToolUse { id, name, .. } = block {
                    if let (Some(id), Some(name)) = (id, name) {
                        map.insert(id.clone(), name.clone());
                    }
                }
            }
        }
    }
    map
}

fn format_message_content_for_prompt(
    content: &MessageContent,
    role: &str,
    tool_id_map: &HashMap<String, String>,
) -> String {
    match content {
        MessageContent::Text(s) => {
            if role == "user" { strip_system_reminders(s) } else { s.clone() }
        }
        MessageContent::Blocks(blocks) => {
            let mut parts = Vec::new();
            for block in blocks {
                match block {
                    ContentBlock::Text { text } => {
                        let t = if role == "user" { strip_system_reminders(text) } else { text.clone() };
                        if !t.trim().is_empty() {
                            parts.push(t);
                        }
                    }
                    ContentBlock::ToolUse { name, input, .. } => {
                        if role == "assistant" {
                            if let (Some(n), Some(inp)) = (name, input) {
                                let call_json = serde_json::json!({
                                    "name": n,
                                    "arguments": inp,
                                });
                                parts.push(format!(
                                    "<tool_call>\n{}\n</tool_call>",
                                    call_json
                                ));
                            }
                        }
                    }
                    ContentBlock::ToolResult { content, tool_use_id, .. } => {
                        if role == "user" {
                            let result_text = match content {
                                Some(ToolResultContent::Text(t)) => t.clone(),
                                Some(ToolResultContent::Blocks(bs)) => {
                                    bs.iter()
                                        .filter_map(|b| b.text.as_deref())
                                        .collect::<Vec<_>>()
                                        .join("\n")
                                }
                                None => String::new(),
                            };
                            // Qwen3 expects tool name in responses, not tool_use_id.
                            // Look up the name from the ToolUse blocks mapping.
                            let tool_name = tool_use_id
                                .as_ref()
                                .and_then(|id| tool_id_map.get(id))
                                .map(|s| s.as_str())
                                .unwrap_or("unknown");
                            parts.push(format!(
                                "<tool_response>\n{}\n</tool_response>",
                                serde_json::json!({
                                    "name": tool_name,
                                    "content": result_text,
                                })
                            ));
                        }
                    }
                    ContentBlock::Thinking { .. } => {
                        // Skip thinking blocks in message history.
                        // The model's prior reasoning is noise for follow-up turns
                        // and wastes context tokens (often 500+ tokens).
                    }
                    ContentBlock::Image { .. } => {
                        parts.push("[Image omitted]".to_string());
                    }
                }
            }
            parts.join("\n")
        }
    }
}

fn build_tools_system_block(tools: &[Tool]) -> String {
    let mut block = String::new();
    block.push_str("\n\n# Tools\n\n");
    block.push_str("You may call one or more functions to assist with the user query.\n\n");
    block.push_str("You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
    for tool in tools {
        let params = tool.input_schema.clone().unwrap_or(serde_json::json!({}));
        let func = serde_json::json!({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description.as_deref().unwrap_or(""),
                "parameters": params,
            }
        });
        block.push_str(&func.to_string());
        block.push('\n');
    }
    block.push_str("</tools>\n\n");
    block.push_str("For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n");
    block.push_str("{\"name\": <function-name>, \"arguments\": <args-json-object>}\n");
    block.push_str("</tool_call>");
    block
}

/// Strip `<system-reminder>...</system-reminder>` blocks from text.
///
/// Claude Code injects internal metadata (skills, hooks, dates, etc.) as
/// `<system-reminder>` XML tags inside user messages. These are meaningless
/// to Qwen3 and bloat the context with noise. We remove them entirely so the
/// model can focus on the actual user content.
fn strip_system_reminders(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;
    let mut stripped_count = 0usize;
    let mut stripped_chars = 0usize;

    loop {
        match remaining.find("<system-reminder>") {
            Some(start) => {
                result.push_str(&remaining[..start]);
                match remaining[start..].find("</system-reminder>") {
                    Some(end_offset) => {
                        let block_end = start + end_offset + "</system-reminder>".len();
                        stripped_count += 1;
                        stripped_chars += block_end - start;
                        remaining = &remaining[block_end..];
                        // Skip trailing whitespace/newlines after the block.
                        remaining = remaining.trim_start_matches(|c: char| c == '\n' || c == '\r');
                    }
                    None => {
                        // Unclosed tag — keep everything.
                        result.push_str(&remaining[start..]);
                        remaining = "";
                    }
                }
            }
            None => {
                result.push_str(remaining);
                break;
            }
        }
    }

    if stripped_count > 0 {
        info!(
            count = stripped_count,
            chars = stripped_chars,
            "Stripped <system-reminder> blocks from user message"
        );
    }

    result
}

pub fn find_stop_sequence<'a>(text: &str, stop_sequences: &'a [String]) -> Option<&'a str> {
    let max_stop_len = stop_sequences.iter().map(|s| s.len()).max().unwrap_or(0);
    let search_start = text.len().saturating_sub(max_stop_len + 64);
    let search_text = &text[search_start..];
    for seq in stop_sequences {
        if search_text.contains(seq.as_str()) {
            return Some(seq.as_str());
        }
    }
    None
}
