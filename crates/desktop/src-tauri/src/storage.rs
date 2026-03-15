use crate::types::{AppSettings, Conversation, ConversationSummary};
use anyhow::Result;
use std::fs;
use std::path::Path;

/// Ensure directory exists.
fn ensure_dir(path: &Path) -> Result<()> {
    if !path.exists() {
        fs::create_dir_all(path)?;
    }
    Ok(())
}

/// Get conversations directory.
fn conversations_dir(data_dir: &Path) -> std::path::PathBuf {
    data_dir.join("conversations")
}

/// List all conversations (summaries only).
pub fn list_conversations(data_dir: &Path) -> Result<Vec<ConversationSummary>> {
    let dir = conversations_dir(data_dir);
    ensure_dir(&dir)?;

    let mut summaries = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            match fs::read_to_string(&path) {
                Ok(content) => {
                    if let Ok(conv) = serde_json::from_str::<Conversation>(&content) {
                        summaries.push(ConversationSummary {
                            id: conv.id,
                            title: conv.title,
                            created_at: conv.created_at,
                            updated_at: conv.updated_at,
                            message_count: conv.messages.len(),
                        });
                    }
                }
                Err(_) => continue,
            }
        }
    }
    summaries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(summaries)
}

/// Load a conversation by ID.
pub fn load_conversation(data_dir: &Path, id: &str) -> Result<Conversation> {
    let path = conversations_dir(data_dir).join(format!("{}.json", id));
    let content = fs::read_to_string(&path)?;
    let conv = serde_json::from_str(&content)?;
    Ok(conv)
}

/// Save a conversation.
pub fn save_conversation(data_dir: &Path, conv: &Conversation) -> Result<()> {
    let dir = conversations_dir(data_dir);
    ensure_dir(&dir)?;
    let path = dir.join(format!("{}.json", conv.id));
    let content = serde_json::to_string_pretty(conv)?;
    fs::write(&path, content)?;
    Ok(())
}

/// Delete a conversation.
pub fn delete_conversation(data_dir: &Path, id: &str) -> Result<()> {
    let path = conversations_dir(data_dir).join(format!("{}.json", id));
    if path.exists() {
        fs::remove_file(&path)?;
    }
    Ok(())
}

/// Load app settings.
pub fn load_settings(data_dir: &Path) -> Result<AppSettings> {
    let path = data_dir.join("settings.json");
    if !path.exists() {
        return Ok(AppSettings::default());
    }
    let content = fs::read_to_string(&path)?;
    let settings = serde_json::from_str(&content)?;
    Ok(settings)
}

/// Save app settings.
pub fn save_settings(data_dir: &Path, settings: &AppSettings) -> Result<()> {
    ensure_dir(data_dir)?;
    let path = data_dir.join("settings.json");
    let content = serde_json::to_string_pretty(settings)?;
    fs::write(&path, content)?;
    Ok(())
}
