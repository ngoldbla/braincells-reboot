//! Settings commands for app configuration
//!
//! These commands handle loading, saving, and updating app settings.

use crate::llm::{self, BackendConfig, InferencePool, LocalEngine, Settings};
use crate::state::AppState;
use std::sync::Arc;
use tauri::State;

/// Get current settings
#[tauri::command]
pub async fn get_settings(state: State<'_, AppState>) -> Result<Settings, String> {
    let settings = state.settings.read().await;
    Ok(settings.clone())
}

/// Update settings
#[tauri::command]
pub async fn update_settings(
    state: State<'_, AppState>,
    new_settings: Settings,
) -> Result<(), String> {
    let mut settings = state.settings.write().await;
    *settings = new_settings;

    // Save to disk
    state.save_settings(&settings).await.map_err(|e| e.to_string())
}

/// Configure the inference backend
#[tauri::command]
pub async fn configure_backend(
    state: State<'_, AppState>,
    config: BackendConfig,
    max_concurrent: Option<usize>,
) -> Result<(), String> {
    let max_concurrent = max_concurrent.unwrap_or(5);

    // Create the engine
    let engine = llm::create_engine(&config).map_err(|e| e.to_string())?;

    // For local backends, load the model
    // Note: Local engines are loaded in the quick_configure_local function

    // Create the pool
    let pool = InferencePool::new(engine, max_concurrent);

    // For local backends, we need to load the model
    if let BackendConfig::Local { model_path, gpu_layers, context_size } = &config {
        let local_engine = LocalEngine::new(model_path, *gpu_layers, *context_size);
        local_engine.load_model().await.map_err(|e| e.to_string())?;
        let engine: Arc<dyn llm::LLMEngine> = Arc::new(local_engine);
        let pool = InferencePool::new(engine, max_concurrent);

        let mut pool_guard = state.inference_pool.write().await;
        *pool_guard = Some(pool);
    } else {
        let mut pool_guard = state.inference_pool.write().await;
        *pool_guard = Some(pool);
    }

    // Update settings
    let mut settings = state.settings.write().await;
    settings.active_backend = Some(config);
    settings.max_concurrent_requests = max_concurrent;

    // Save settings
    state.save_settings(&settings).await.map_err(|e| e.to_string())?;

    Ok(())
}

/// Set the HuggingFace API key
#[tauri::command]
pub async fn set_hf_api_key(
    state: State<'_, AppState>,
    api_key: String,
) -> Result<(), String> {
    let mut settings = state.settings.write().await;
    settings.hf_api_key = if api_key.is_empty() { None } else { Some(api_key) };
    state.save_settings(&settings).await.map_err(|e| e.to_string())
}

/// Set the OpenRouter API key
#[tauri::command]
pub async fn set_openrouter_api_key(
    state: State<'_, AppState>,
    api_key: String,
) -> Result<(), String> {
    let mut settings = state.settings.write().await;
    settings.openrouter_api_key = if api_key.is_empty() { None } else { Some(api_key) };
    state.save_settings(&settings).await.map_err(|e| e.to_string())
}

/// Set the OpenAI API key
#[tauri::command]
pub async fn set_openai_api_key(
    state: State<'_, AppState>,
    api_key: String,
) -> Result<(), String> {
    let mut settings = state.settings.write().await;
    settings.openai_api_key = if api_key.is_empty() { None } else { Some(api_key) };
    state.save_settings(&settings).await.map_err(|e| e.to_string())
}

/// Set the default local model
#[tauri::command]
pub async fn set_default_model(
    state: State<'_, AppState>,
    model_path: Option<String>,
) -> Result<(), String> {
    let mut settings = state.settings.write().await;
    settings.default_local_model = model_path;
    state.save_settings(&settings).await.map_err(|e| e.to_string())
}

/// Set concurrency level
#[tauri::command]
pub async fn set_max_concurrent(
    state: State<'_, AppState>,
    max_concurrent: usize,
) -> Result<(), String> {
    let max_concurrent = max_concurrent.clamp(1, 10);

    let mut settings = state.settings.write().await;
    settings.max_concurrent_requests = max_concurrent;
    state.save_settings(&settings).await.map_err(|e| e.to_string())
}

/// Quick configure for local model
#[tauri::command]
pub async fn quick_configure_local(
    state: State<'_, AppState>,
    model_path: String,
) -> Result<(), String> {
    let gpu_layers = llm::detect_optimal_gpu_layers();

    let config = BackendConfig::Local {
        model_path,
        gpu_layers,
        context_size: 4096,
    };

    let settings = state.settings.read().await;
    let max_concurrent = settings.max_concurrent_requests;
    drop(settings);

    configure_backend(state, config, Some(max_concurrent)).await
}

/// Quick configure for HuggingFace cloud
#[tauri::command]
pub async fn quick_configure_huggingface(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<(), String> {
    let settings = state.settings.read().await;
    let api_key = settings.hf_api_key.clone()
        .ok_or("HuggingFace API key not configured")?;
    let max_concurrent = settings.max_concurrent_requests;
    drop(settings);

    let config = BackendConfig::HuggingFace {
        api_key,
        model_id,
        provider: None,
    };

    configure_backend(state, config, Some(max_concurrent)).await
}

/// Quick configure for OpenRouter cloud
#[tauri::command]
pub async fn quick_configure_openrouter(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<(), String> {
    let settings = state.settings.read().await;
    let api_key = settings.openrouter_api_key.clone()
        .ok_or("OpenRouter API key not configured")?;
    let max_concurrent = settings.max_concurrent_requests;
    drop(settings);

    let config = BackendConfig::OpenRouter {
        api_key,
        model_id,
    };

    configure_backend(state, config, Some(max_concurrent)).await
}

/// Quick configure for OpenAI cloud
#[tauri::command]
pub async fn quick_configure_openai(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<(), String> {
    let settings = state.settings.read().await;
    let api_key = settings.openai_api_key.clone()
        .ok_or("OpenAI API key not configured")?;
    let max_concurrent = settings.max_concurrent_requests;
    drop(settings);

    let config = BackendConfig::OpenAI {
        api_key,
        model_id,
    };

    configure_backend(state, config, Some(max_concurrent)).await
}

/// Get available cloud providers based on configured API keys
#[tauri::command]
pub async fn get_available_providers(
    state: State<'_, AppState>,
) -> Result<Vec<ProviderInfo>, String> {
    let settings = state.settings.read().await;

    let mut providers = Vec::new();

    // Always show local as available
    providers.push(ProviderInfo {
        id: "local".to_string(),
        name: "Local (llama.cpp)".to_string(),
        available: true,
        configured: true,
    });

    providers.push(ProviderInfo {
        id: "huggingface".to_string(),
        name: "HuggingFace".to_string(),
        available: true,
        configured: settings.hf_api_key.is_some(),
    });

    providers.push(ProviderInfo {
        id: "openrouter".to_string(),
        name: "OpenRouter".to_string(),
        available: true,
        configured: settings.openrouter_api_key.is_some(),
    });

    providers.push(ProviderInfo {
        id: "openai".to_string(),
        name: "OpenAI".to_string(),
        available: true,
        configured: settings.openai_api_key.is_some(),
    });

    Ok(providers)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProviderInfo {
    pub id: String,
    pub name: String,
    pub available: bool,
    pub configured: bool,
}
