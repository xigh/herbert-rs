use crate::state::AppState;
use crate::storage;
use crate::types::AppSettings;
use tauri::State;

#[tauri::command]
pub async fn get_settings(state: State<'_, AppState>) -> Result<AppSettings, String> {
    storage::load_settings(&state.data_dir).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn save_settings(
    settings: AppSettings,
    state: State<'_, AppState>,
) -> Result<(), String> {
    storage::save_settings(&state.data_dir, &settings).map_err(|e| e.to_string())
}
