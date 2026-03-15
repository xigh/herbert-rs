use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<GenerationStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub prefill_ms: u64,
    pub decode_ms: u64,
    pub tokens_per_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSettings {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_temperature() -> f32 { 0.4 }
fn default_top_k() -> usize { 40 }
fn default_top_p() -> f32 { 0.9 }
fn default_max_tokens() -> usize { 2048 }

impl Default for ConversationSettings {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_k: default_top_k(),
            top_p: default_top_p(),
            max_tokens: default_max_tokens(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub settings: ConversationSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSummary {
    pub id: String,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub message_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub backend: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub model_path: Option<String>,
    pub backend: String,
    pub default_system_prompt: String,
    pub default_settings: ConversationSettings,
    #[serde(default)]
    pub nothink: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            model_path: None,
            backend: "metal-q4".to_string(),
            default_system_prompt: "You are a helpful assistant.".to_string(),
            default_settings: ConversationSettings::default(),
            nothink: false,
        }
    }
}

/// Events sent through the Tauri Channel during model loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LoadingEvent {
    /// Progress update: step description + progress percentage (0-100).
    #[serde(rename = "progress")]
    Progress { step: String, percent: u32 },
    /// Loading completed successfully.
    #[serde(rename = "done")]
    Done,
    /// Loading failed.
    #[serde(rename = "error")]
    Error { message: String },
}

/// Events sent through the Tauri Channel during vision encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VisionEvent {
    #[serde(rename = "progress")]
    Progress {
        image_id: String,
        percent: u32,
        label: String,
    },
    #[serde(rename = "done")]
    Done {
        image_id: String,
        num_tokens: usize,
    },
    #[serde(rename = "error")]
    Error {
        image_id: String,
        message: String,
    },
}

/// Events sent through the Tauri Channel during streaming generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TokenEvent {
    /// A new text delta (incremental text).
    #[serde(rename = "delta")]
    Delta { text: String },
    /// Generation finished successfully.
    #[serde(rename = "done")]
    Done { stats: GenerationStats },
    /// Generation was cancelled by user.
    #[serde(rename = "cancelled")]
    Cancelled,
    /// An error occurred.
    #[serde(rename = "error")]
    Error { message: String },
}
