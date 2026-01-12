//! Model management commands
//!
//! These commands handle model discovery, downloading, and management.

use crate::llm::types::{LocalModel, RecommendedModel};
use crate::models::{self, get_models_dir, list_local_models, registry};
use crate::state::AppState;
use tauri::{AppHandle, Emitter, State};
use std::path::PathBuf;

/// List all locally available models
#[tauri::command]
pub async fn list_models() -> Result<Vec<LocalModel>, String> {
    list_local_models().map_err(|e| e.to_string())
}

/// Get recommended models for download
#[tauri::command]
pub async fn get_recommended_models() -> Result<Vec<RecommendedModel>, String> {
    Ok(registry::get_recommended_models())
}

/// Get recommended models that fit in available memory
#[tauri::command]
pub async fn get_recommended_models_for_system() -> Result<Vec<RecommendedModel>, String> {
    let memory = registry::estimate_available_memory();
    let max_gb = (memory as f64 / (1024.0 * 1024.0 * 1024.0)) * 0.7; // Use 70% of available
    Ok(registry::get_recommended_models_by_size(max_gb))
}

/// Suggest the best model for the current system
#[tauri::command]
pub async fn suggest_model() -> Result<RecommendedModel, String> {
    let memory = registry::estimate_available_memory();
    Ok(registry::suggest_model_for_memory(memory))
}

/// Download a model from HuggingFace Hub
#[tauri::command]
pub async fn download_model(
    app: AppHandle,
    state: State<'_, AppState>,
    repo_id: String,
    filename: String,
) -> Result<String, String> {
    let models_dir = get_models_dir().map_err(|e| e.to_string())?;

    // Check if already downloaded
    let dest_path = models_dir.join(&filename);
    if dest_path.exists() {
        return Ok(dest_path.to_string_lossy().to_string());
    }

    // Get HF token from settings if available
    let settings = state.settings.read().await;
    let hf_token = settings.hf_api_key.clone();
    drop(settings);

    // Start download with progress events
    let handle = models::download::download_model(
        repo_id.clone(),
        filename.clone(),
        models_dir,
        hf_token,
    );

    // Spawn task to emit progress events
    let app_clone = app.clone();
    tokio::spawn(async move {
        let mut progress_rx = handle.progress_rx;
        while let Some(progress) = progress_rx.recv().await {
            let _ = app_clone.emit("model-download-progress", &progress);
        }
    });

    // Wait for result
    let result = handle.result_rx.await
        .map_err(|_| "Download task cancelled".to_string())?
        .map_err(|e| e.to_string())?;

    // Register in cache
    let mut cache = state.model_cache.write().await;
    if let Some(ref mut cache) = *cache {
        let metadata = std::fs::metadata(&result).ok();
        cache.register_model(
            filename,
            result.to_string_lossy().to_string(),
            metadata.map(|m| m.len()).unwrap_or(0),
            Some(repo_id),
        ).map_err(|e| e.to_string())?;
    }

    Ok(result.to_string_lossy().to_string())
}

/// Delete a local model
#[tauri::command]
pub async fn delete_model(
    state: State<'_, AppState>,
    model_path: String,
) -> Result<(), String> {
    models::delete_model(&model_path).map_err(|e| e.to_string())?;

    // Remove from cache
    let mut cache = state.model_cache.write().await;
    if let Some(ref mut cache) = *cache {
        let name = PathBuf::from(&model_path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let _ = cache.remove_model(&name);
    }

    Ok(())
}

/// Get the models directory path
#[tauri::command]
pub async fn get_models_directory() -> Result<String, String> {
    get_models_dir()
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| e.to_string())
}

/// Check if a model is downloaded
#[tauri::command]
pub async fn is_model_downloaded(_repo_id: String, filename: String) -> Result<bool, String> {
    let models_dir = get_models_dir().map_err(|e| e.to_string())?;
    Ok(models_dir.join(&filename).exists())
}

/// Get cache statistics
#[tauri::command]
pub async fn get_cache_stats(
    state: State<'_, AppState>,
) -> Result<CacheStats, String> {
    let cache = state.model_cache.read().await;
    match cache.as_ref() {
        Some(cache) => {
            let models = cache.list_models();
            Ok(CacheStats {
                total_models: models.len(),
                total_size_bytes: cache.total_size(),
                models,
            })
        }
        None => Ok(CacheStats {
            total_models: 0,
            total_size_bytes: 0,
            models: Vec::new(),
        }),
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheStats {
    pub total_models: usize,
    pub total_size_bytes: u64,
    pub models: Vec<LocalModel>,
}

/// Clean up old models from cache
#[tauri::command]
pub async fn cleanup_cache(
    state: State<'_, AppState>,
    max_age_days: Option<i64>,
) -> Result<Vec<String>, String> {
    let mut cache = state.model_cache.write().await;
    match cache.as_mut() {
        Some(cache) => {
            cache.cleanup_old_models(max_age_days.unwrap_or(30))
                .map_err(|e| e.to_string())
        }
        None => Ok(Vec::new()),
    }
}
