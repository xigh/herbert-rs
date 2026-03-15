use crate::state::AppState;
use crate::types::VisionEvent;
use base64::Engine;
use herbert_core::backend::VisionEmbedding;
use herbert_vision::config::VisionConfig;
use herbert_vision::image_process::preprocess_rgb;
use std::path::PathBuf;
use tauri::ipc::Channel;
use tauri::State;
use std::time::Instant;
use tracing::info;

/// Read an image from disk and return a base64 data URL thumbnail (max 240x240 JPEG).
#[tauri::command]
pub async fn read_image_thumbnail(path: String) -> Result<String, String> {
    let t0 = Instant::now();
    let img = image::open(&path).map_err(|e| format!("Failed to open image: {}", e))?;
    let thumb = img.thumbnail(240, 240);
    let mut buf = std::io::Cursor::new(Vec::new());
    thumb
        .write_to(&mut buf, image::ImageFormat::Jpeg)
        .map_err(|e| format!("Failed to encode thumbnail: {}", e))?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());
    info!("[vision-thumb] {} -> {:.0}ms", path, t0.elapsed().as_secs_f64() * 1000.0);
    Ok(format!("data:image/jpeg;base64,{}", b64))
}

/// Encode an image for vision: load, preprocess, run vision encoder, store embedding.
/// Progress is streamed via Channel<VisionEvent>.
#[tauri::command]
pub async fn encode_image(
    image_id: String,
    path: String,
    channel: Channel<VisionEvent>,
    state: State<'_, AppState>,
) -> Result<(), String> {
    // Get model_path from loaded model
    let model_path: PathBuf = {
        let model_lock = state.model.lock().map_err(|e| format!("Lock error: {}", e))?;
        let ms = model_lock
            .as_ref()
            .ok_or_else(|| "Model not loaded".to_string())?;
        ms.model_path.clone()
    };

    let id = image_id;
    let t0 = Instant::now();
    info!("[vision-encode] START id={} path={}", id, path);

    // Step 1: Load and preprocess (CPU work, in spawn_blocking)
    let preprocess_result = {
        let id = id.clone();
        let channel = channel.clone();
        let model_path = model_path.clone();
        tauri::async_runtime::spawn_blocking(move || -> Result<(Vec<f32>, usize, usize, usize, VisionConfig), String> {
            info!("[vision-encode] +{:.0}ms  sending progress 0% Loading...", t0.elapsed().as_secs_f64() * 1000.0);
            let _ = channel.send(VisionEvent::Progress {
                image_id: id.clone(),
                percent: 0,
                label: "Loading...".to_string(),
            });

            let img = image::open(&path).map_err(|e| format!("Failed to load image: {}", e))?;
            let rgb = img.to_rgb8();
            let (width, height) = (rgb.width() as usize, rgb.height() as usize);
            let rgb_bytes = rgb.into_raw();
            info!("[vision-encode] +{:.0}ms  image loaded {}x{}", t0.elapsed().as_secs_f64() * 1000.0, width, height);

            let config_path = model_path.join("config.json");
            let vision_config = VisionConfig::from_file(&config_path)
                .map_err(|e| format!("Failed to load vision config: {}", e))?;

            info!("[vision-encode] +{:.0}ms  sending progress 5% Preprocessing...", t0.elapsed().as_secs_f64() * 1000.0);
            let _ = channel.send(VisionEvent::Progress {
                image_id: id.clone(),
                percent: 5,
                label: "Preprocessing...".to_string(),
            });

            let min_pixels = 256 * 28 * 28;
            let max_pixels = 1280 * 28 * 28;
            let (patches, grid_t, grid_h, grid_w) =
                preprocess_rgb(&rgb_bytes, height, width, &vision_config, min_pixels, max_pixels)
                    .map_err(|e| format!("Preprocessing failed: {}", e))?;

            info!("[vision-encode] +{:.0}ms  preprocess done grid={}x{}x{} patches={}", t0.elapsed().as_secs_f64() * 1000.0, grid_t, grid_h, grid_w, patches.len());
            Ok((patches, grid_t, grid_h, grid_w, vision_config))
        })
        .await
        .map_err(|e| format!("Task join error: {}", e))?
    };

    let (patches, grid_t, grid_h, grid_w, vision_config) = match preprocess_result {
        Ok(r) => r,
        Err(e) => {
            let _ = channel.send(VisionEvent::Error {
                image_id: id,
                message: e.clone(),
            });
            return Err(e);
        }
    };

    // Check cancel
    if is_cancelled(&state, &id) {
        return Ok(());
    }

    // Step 2: Try GPU encode (fast ~100ms, hold model lock briefly)
    info!("[vision-encode] +{:.0}ms  acquiring model lock for encode...", t0.elapsed().as_secs_f64() * 1000.0);
    let gpu_result: Option<Result<VisionEmbedding, String>> = {
        let mut model_lock = state.model.lock().map_err(|e| format!("Lock error: {}", e))?;
        if let Some(ms) = model_lock.as_mut() {
            info!("[vision-encode] +{:.0}ms  sending progress 10% Encoding...", t0.elapsed().as_secs_f64() * 1000.0);
            let _ = channel.send(VisionEvent::Progress {
                image_id: id.clone(),
                percent: 10,
                label: "Encoding...".to_string(),
            });
            let t_enc = Instant::now();
            // Progress callback: GPU signals each block via MTLSharedEvent
            let progress_channel = channel.clone();
            let progress_id = id.clone();
            let progress_t0 = t0;
            let progress_fn: Box<dyn Fn(usize, usize) + Send + Sync> = Box::new(move |block_done, total_blocks| {
                // Map block progress to 10%..90% range
                let pct = 10 + (block_done * 80 / total_blocks) as u32;
                info!("[vision-encode] +{:.0}ms  GPU block {}/{}", progress_t0.elapsed().as_secs_f64() * 1000.0, block_done, total_blocks);
                let _ = progress_channel.send(VisionEvent::Progress {
                    image_id: progress_id.clone(),
                    percent: pct,
                    label: format!("Encoding block {}/{}...", block_done, total_blocks),
                });
            });
            let result = match ms.backend.encode_vision_with_progress(&patches, grid_t, grid_h, grid_w, progress_fn) {
                Some(Ok(embed)) => Some(Ok(embed)),
                Some(Err(e)) => Some(Err(format!("GPU vision encode failed: {}", e))),
                None => None,
            };
            info!("[vision-encode] +{:.0}ms  encode_vision returned ({:.0}ms)", t0.elapsed().as_secs_f64() * 1000.0, t_enc.elapsed().as_secs_f64() * 1000.0);
            result
        } else {
            return Err("Model not loaded".to_string());
        }
    };

    let embed = match gpu_result {
        Some(Ok(embed)) => embed,
        Some(Err(e)) => {
            let _ = channel.send(VisionEvent::Error {
                image_id: id,
                message: e.clone(),
            });
            return Err(e);
        }
        None => {
            // CPU fallback in spawn_blocking
            info!("[vision-encode] +{:.0}ms  GPU not available, falling back to CPU", t0.elapsed().as_secs_f64() * 1000.0);
            let id_cpu = id.clone();
            let channel_cpu = channel.clone();
            let vc = vision_config.clone();
            let mp = model_path;

            let cpu_result = tauri::async_runtime::spawn_blocking(move || -> Result<VisionEmbedding, String> {
                info!("[vision-encode] +{:.0}ms  sending progress 10% Loading vision encoder...", t0.elapsed().as_secs_f64() * 1000.0);
                let _ = channel_cpu.send(VisionEvent::Progress {
                    image_id: id_cpu.clone(),
                    percent: 10,
                    label: "Loading vision encoder...".to_string(),
                });

                let encoder = herbert_vision::loader::load_vision_encoder_with_progress(&mp, &vc)
                    .map_err(|e| format!("Failed to load vision encoder: {}", e))?;

                info!("[vision-encode] +{:.0}ms  sending progress 20% Encoding (CPU)...", t0.elapsed().as_secs_f64() * 1000.0);
                let _ = channel_cpu.send(VisionEvent::Progress {
                    image_id: id_cpu.clone(),
                    percent: 20,
                    label: "Encoding (CPU)...".to_string(),
                });

                let t_fwd = Instant::now();
                let output = encoder
                    .forward(&patches, grid_t, grid_h, grid_w)
                    .map_err(|e| format!("Vision encode failed: {}", e))?;
                info!("[vision-encode] +{:.0}ms  CPU forward done ({:.0}ms)", t0.elapsed().as_secs_f64() * 1000.0, t_fwd.elapsed().as_secs_f64() * 1000.0);

                let merge_size = vc.spatial_merge_size;
                Ok(VisionEmbedding {
                    hidden_states: output.hidden_states,
                    num_tokens: output.num_tokens,
                    grid_h: grid_h / merge_size,
                    grid_w: grid_w / merge_size,
                    deepstack_features: output.deepstack_features,
                })
            })
            .await
            .map_err(|e| format!("Task join error: {}", e))?;

            match cpu_result {
                Ok(embed) => embed,
                Err(e) => {
                    let _ = channel.send(VisionEvent::Error {
                        image_id: id,
                        message: e.clone(),
                    });
                    return Err(e);
                }
            }
        }
    };

    // Check cancel before storing
    if is_cancelled(&state, &id) {
        return Ok(());
    }

    let num_tokens = embed.num_tokens;
    info!("[vision-encode] +{:.0}ms  storing embedding ({} tokens)", t0.elapsed().as_secs_f64() * 1000.0, num_tokens);

    // Store embedding
    if let Ok(mut embeddings) = state.vision_embeddings.lock() {
        embeddings.insert(id.clone(), embed);
    }

    info!("[vision-encode] +{:.0}ms  sending Done event", t0.elapsed().as_secs_f64() * 1000.0);
    let _ = channel.send(VisionEvent::Done {
        image_id: id.clone(),
        num_tokens,
    });

    info!("[vision-encode] DONE id={} total={:.0}ms tokens={}", id, t0.elapsed().as_secs_f64() * 1000.0, num_tokens);
    Ok(())
}

/// Remove an image embedding and cancel in-progress encoding.
#[tauri::command]
pub async fn remove_image(
    image_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    if let Ok(mut cancel_set) = state.cancel_vision.lock() {
        cancel_set.insert(image_id.clone());
    }
    if let Ok(mut embeddings) = state.vision_embeddings.lock() {
        embeddings.remove(&image_id);
    }
    Ok(())
}

fn is_cancelled(state: &State<'_, AppState>, id: &str) -> bool {
    state
        .cancel_vision
        .lock()
        .map(|s| s.contains(id))
        .unwrap_or(false)
}
