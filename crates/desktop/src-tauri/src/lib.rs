mod commands;
mod inference;
mod state;
mod storage;
mod types;

use state::AppState;
use tauri::Emitter;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .without_time()
        .init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            commands::model::load_model,
            commands::model::unload_model,
            commands::model::get_model_info,
            commands::generate::generate,
            commands::generate::cancel_generation,
            commands::conversation::list_conversations,
            commands::conversation::get_conversation,
            commands::conversation::create_conversation,
            commands::conversation::delete_conversation,
            commands::conversation::update_conversation_title,
            commands::conversation::add_message,
            commands::conversation::export_conversation,
            commands::settings::get_settings,
            commands::settings::save_settings,
            commands::screenshot::save_screenshot_data,
            commands::vision::read_image_thumbnail,
            commands::vision::encode_image,
            commands::vision::remove_image,
        ])
        .setup(|app| {
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                let trigger = std::path::PathBuf::from("/tmp/herbert-take-screenshot");
                loop {
                    if trigger.exists() {
                        let _ = std::fs::remove_file(&trigger);
                        let _ = handle.emit("take-screenshot", ());
                    }
                    std::thread::sleep(std::time::Duration::from_millis(300));
                }
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Herbert desktop");
}
