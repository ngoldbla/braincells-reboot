//! Inference commands for LLM generation
//!
//! These commands handle text generation requests from the frontend,
//! supporting both single requests and batch processing.

use crate::llm::{LLMRequest, LLMResponse};
use crate::state::AppState;
use tauri::State;

/// Generate a response for a single cell
#[tauri::command]
pub async fn generate_cell(
    state: State<'_, AppState>,
    request: LLMRequest,
) -> Result<LLMResponse, String> {
    let pool = state.inference_pool.read().await;
    let pool = pool.as_ref().ok_or("No LLM engine configured. Please configure a backend in settings.")?;

    pool.generate_single(request)
        .await
        .map_err(|e| e.to_string())
}

/// Generate responses for multiple cells in parallel
#[tauri::command]
pub async fn generate_batch(
    state: State<'_, AppState>,
    requests: Vec<LLMRequest>,
) -> Result<Vec<Result<LLMResponse, String>>, String> {
    let pool = state.inference_pool.read().await;
    let pool = pool.as_ref().ok_or("No LLM engine configured. Please configure a backend in settings.")?;

    let results = pool.generate_batch(requests).await;
    Ok(results.into_iter().map(|r| r.map_err(|e| e.to_string())).collect())
}

/// Check if an inference engine is ready
#[tauri::command]
pub async fn is_inference_ready(state: State<'_, AppState>) -> Result<bool, String> {
    let pool = state.inference_pool.read().await;
    match pool.as_ref() {
        Some(pool) => Ok(pool.is_ready().await),
        None => Ok(false),
    }
}

/// Get the current backend name
#[tauri::command]
pub async fn get_backend_name(state: State<'_, AppState>) -> Result<Option<String>, String> {
    let pool = state.inference_pool.read().await;
    Ok(pool.as_ref().map(|p| p.backend_name().to_string()))
}

/// Unload the current model (for local backends)
#[tauri::command]
pub async fn unload_model(state: State<'_, AppState>) -> Result<(), String> {
    let pool = state.inference_pool.read().await;
    if let Some(pool) = pool.as_ref() {
        pool.unload().await.map_err(|e| e.to_string())?;
    }
    Ok(())
}
