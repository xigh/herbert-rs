//! Request and response types for the Anthropic Messages API.

use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct Tool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<serde_json::Value>,
}

#[derive(Deserialize)]
pub struct MessagesRequest {
    #[serde(rename = "model", default)]
    pub _model: Option<String>,
    pub max_tokens: usize,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub system: Option<SystemPrompt>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub tools: Vec<Tool>,
    #[serde(rename = "tool_choice", default)]
    pub _tool_choice: Option<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<TextBlock>),
}

#[derive(Deserialize)]
pub struct TextBlock {
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Deserialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_result")]
    ToolResult {
        #[serde(default)]
        content: Option<ToolResultContent>,
        #[serde(default)]
        tool_use_id: Option<String>,
        #[serde(rename = "is_error", default)]
        _is_error: Option<bool>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        #[serde(rename = "id", default)]
        id: Option<String>,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        input: Option<serde_json::Value>,
    },
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(default)]
        thinking: Option<String>,
        #[serde(rename = "signature", default)]
        _signature: Option<String>,
    },
    #[serde(rename = "image")]
    Image {
        #[serde(rename = "source", default)]
        _source: Option<serde_json::Value>,
    },
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultBlock>),
}

#[derive(Deserialize)]
pub struct ToolResultBlock {
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Deserialize)]
pub struct CountTokensRequest {
    #[serde(rename = "model", default)]
    pub _model: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub system: Option<SystemPrompt>,
}

#[derive(Serialize)]
pub struct CountTokensResponse {
    pub input_tokens: usize,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    #[serde(rename = "type")]
    pub error_type: String,
    pub error: ErrorDetail,
}

#[derive(Serialize)]
pub struct ErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

// ── OpenAI Embeddings API ──

#[derive(Deserialize)]
pub struct EmbeddingsRequest {
    #[serde(rename = "model", default)]
    pub _model: Option<String>,
    pub input: EmbeddingInput,
    #[serde(default = "default_float")]
    pub encoding_format: String,
}

fn default_float() -> String {
    "float".to_string()
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    pub tokens_per_second: f64,
    pub duration_ms: f64,
}

// ── Tokenize API ──

#[derive(Deserialize)]
pub struct TokenizeRequest {
    pub text: String,
    /// "embed" (default) or "chat"
    #[serde(default = "default_embed")]
    pub model: String,
}

fn default_embed() -> String {
    "embed".to_string()
}

#[derive(Serialize)]
pub struct TokenizeResponse {
    pub token_ids: Vec<u32>,
    pub tokens: Vec<String>,
    pub count: usize,
    pub model: String,
}
