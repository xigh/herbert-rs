use base64::Engine;
use std::fs;
use std::path::Path;

#[tauri::command]
pub async fn save_screenshot_data(data: String, path: String) -> Result<(), String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&data)
        .map_err(|e| format!("Base64 decode error: {}", e))?;
    let dir = Path::new(&path).parent();
    if let Some(d) = dir {
        fs::create_dir_all(d).map_err(|e| format!("mkdir error: {}", e))?;
    }
    fs::write(&path, &bytes).map_err(|e| format!("Write error: {}", e))?;
    Ok(())
}
