use crate::state::AppState;
use crate::storage;
use crate::types::{Conversation, ConversationSettings, ConversationSummary, Message};
use chrono::Utc;
use tauri::State;

#[tauri::command]
pub async fn list_conversations(
    state: State<'_, AppState>,
) -> Result<Vec<ConversationSummary>, String> {
    storage::list_conversations(&state.data_dir).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_conversation(
    id: String,
    state: State<'_, AppState>,
) -> Result<Conversation, String> {
    storage::load_conversation(&state.data_dir, &id).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn create_conversation(
    system_prompt: Option<String>,
    state: State<'_, AppState>,
) -> Result<Conversation, String> {
    let now = Utc::now();
    let conv = Conversation {
        id: format!("conv_{}", now.timestamp_millis()),
        title: "New Chat".to_string(),
        created_at: now,
        updated_at: now,
        system_prompt: system_prompt.unwrap_or_else(|| "You are a helpful assistant.".to_string()),
        messages: Vec::new(),
        settings: ConversationSettings::default(),
    };
    storage::save_conversation(&state.data_dir, &conv).map_err(|e| e.to_string())?;
    Ok(conv)
}

#[tauri::command]
pub async fn delete_conversation(
    id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    storage::delete_conversation(&state.data_dir, &id).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn update_conversation_title(
    id: String,
    title: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let mut conv =
        storage::load_conversation(&state.data_dir, &id).map_err(|e| e.to_string())?;
    conv.title = title;
    conv.updated_at = Utc::now();
    storage::save_conversation(&state.data_dir, &conv).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn add_message(
    conversation_id: String,
    role: String,
    content: String,
    images: Option<Vec<String>>,
    state: State<'_, AppState>,
) -> Result<Message, String> {
    let mut conv =
        storage::load_conversation(&state.data_dir, &conversation_id).map_err(|e| e.to_string())?;

    let msg = Message {
        role,
        content,
        timestamp: Utc::now(),
        images,
        stats: None,
    };
    conv.messages.push(msg.clone());
    conv.updated_at = Utc::now();

    // Auto-title from first user message
    if conv.title == "New Chat" {
        if let Some(first_user) = conv.messages.iter().find(|m| m.role == "user") {
            let title: String = first_user
                .content
                .chars()
                .take(50)
                .collect::<String>()
                .lines()
                .next()
                .unwrap_or("New Chat")
                .to_string();
            conv.title = if title.len() < first_user.content.len() {
                format!("{}...", title.trim_end())
            } else {
                title
            };
        }
    }

    storage::save_conversation(&state.data_dir, &conv).map_err(|e| e.to_string())?;
    Ok(msg)
}

#[tauri::command]
pub async fn export_conversation(
    id: String,
    path: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let conv =
        storage::load_conversation(&state.data_dir, &id).map_err(|e| e.to_string())?;
    let json =
        serde_json::to_string_pretty(&conv).map_err(|e| e.to_string())?;

    std::fs::write(&path, &json)
        .map_err(|e| format!("Failed to write file: {}", e))?;

    Ok(())
}
