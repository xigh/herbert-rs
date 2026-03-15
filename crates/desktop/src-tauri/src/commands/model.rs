use crate::state::{AppState, ModelState};
use crate::types::{LoadingEvent, ModelInfo};
use herbert_core::backend::{Backend, LoadOpts};
use herbert_core::config::{Config, KvQuantType};
use std::path::Path;
use tauri::ipc::Channel;
use tauri::State;
use tokenizers::Tokenizer;
use tracing::info;

#[tauri::command]
pub async fn load_model(
    model_path: String,
    backend_name: String,
    channel: Channel<LoadingEvent>,
    state: State<'_, AppState>,
) -> Result<ModelInfo, String> {
    let model_path_clone = model_path.clone();
    let backend_name_clone = backend_name.clone();

    let result = tauri::async_runtime::spawn_blocking(move || {
        let path = Path::new(&model_path_clone);
        if !path.exists() {
            let _ = channel.send(LoadingEvent::Error {
                message: format!("Path does not exist: {}", model_path_clone),
            });
            return Err(format!("Model path does not exist: {}", model_path_clone));
        }

        // Step 1: Load config (5%)
        let _ = channel.send(LoadingEvent::Progress {
            step: "Loading configuration...".to_string(),
            percent: 5,
        });
        let config_path = path.join("config.json");
        let config = Config::from_file(&config_path)
            .map_err(|e| format!("Failed to load config: {}", e))?;

        // Step 2: Load tokenizer (10%)
        let _ = channel.send(LoadingEvent::Progress {
            step: "Loading tokenizer...".to_string(),
            percent: 10,
        });
        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        let eos_token_id = config.eos_token();
        let im_end_id = tokenizer
            .token_to_id("<|im_end|>")
            .unwrap_or(eos_token_id);
        let think_id = tokenizer.token_to_id("<think>");
        let end_think_id = tokenizer.token_to_id("</think>");

        // Step 3: Create backend (15%)
        let _ = channel.send(LoadingEvent::Progress {
            step: format!("Initializing {} backend...", backend_name_clone),
            percent: 15,
        });
        let mut backend: Box<dyn Backend> = match backend_name_clone.as_str() {
            #[cfg(target_os = "macos")]
            "metal-q4" => Box::new(
                herbert_backend_metal::MetalBackend::with_quant_mode(
                    herbert_backend_metal::QuantMode::Q4,
                ),
            ),
            #[cfg(target_os = "macos")]
            "metal-bf16" => Box::new(
                herbert_backend_metal::MetalBackend::with_quant_mode(
                    herbert_backend_metal::QuantMode::BF16,
                ),
            ),
            #[cfg(target_os = "macos")]
            "metal-int8" => Box::new(
                herbert_backend_metal::MetalBackend::with_quant_mode(
                    herbert_backend_metal::QuantMode::Int8,
                ),
            ),
            _ => return Err(format!("Unsupported backend: {}", backend_name_clone)),
        };

        // Step 4: Load weights (20% -> 95%)
        let _ = channel.send(LoadingEvent::Progress {
            step: format!(
                "Loading model weights ({} layers, {}d)...",
                config.num_layers, config.hidden_size
            ),
            percent: 20,
        });
        let load_opts = LoadOpts {
            validate: false,
            num_threads: None,
            kv_reserve_tokens: Some(8192),
            show_progress: false,
            no_cache: false,
            prefill_chunk_size: None,
            decode_threads: None,
            prefix_cache_dir: None,
            kv_quant: KvQuantType::BF16,
            kv_budget: None,
            use_q_bf16: false,
        };
        backend
            .load(path, load_opts)
            .map_err(|e| format!("Failed to load model: {}", e))?;

        // Step 5: Done (100%)
        let _ = channel.send(LoadingEvent::Progress {
            step: "Model loaded!".to_string(),
            percent: 100,
        });
        let _ = channel.send(LoadingEvent::Done);

        info!("Model loaded from {}", model_path_clone);

        let info = ModelInfo {
            name: config.model_type.clone(),
            path: model_path_clone,
            backend: backend_name_clone,
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
        };

        Ok((backend, tokenizer, config, eos_token_id, im_end_id, think_id, end_think_id, info))
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?;

    let (backend, tokenizer, config, eos_token_id, im_end_id, think_id, end_think_id, info) = result?;

    let mut model_lock = state.model.lock().map_err(|e| format!("Lock error: {}", e))?;
    *model_lock = Some(ModelState {
        backend,
        tokenizer,
        config,
        eos_token_id,
        im_end_id,
        think_id,
        end_think_id,
        model_path: std::path::PathBuf::from(&model_path),
    });

    Ok(info)
}

#[tauri::command]
pub async fn unload_model(state: State<'_, AppState>) -> Result<(), String> {
    let mut model_lock = state.model.lock().map_err(|e| format!("Lock error: {}", e))?;
    *model_lock = None;
    info!("Model unloaded");
    Ok(())
}

#[tauri::command]
pub async fn get_model_info(state: State<'_, AppState>) -> Result<Option<ModelInfo>, String> {
    let model_lock = state.model.lock().map_err(|e| format!("Lock error: {}", e))?;
    match model_lock.as_ref() {
        Some(ms) => Ok(Some(ModelInfo {
            name: ms.config.model_type.clone(),
            path: String::new(),
            backend: ms.backend.name().to_string(),
            num_layers: ms.config.num_layers,
            hidden_size: ms.config.hidden_size,
            vocab_size: ms.config.vocab_size,
        })),
        None => Ok(None),
    }
}
