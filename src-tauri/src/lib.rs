//! Braincells - Intelligent Spreadsheet Automation
//!
//! A desktop application for AI-powered data manipulation using local LLM inference.
//!
//! Developed by Dylan Goldblatt at Kennesaw State University Office of Research.

pub mod commands;
pub mod llm;
pub mod models;
pub mod state;

use state::{AppState, get_data_dir};
use tauri::Manager;

/// Run the Tauri application
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::default()
                .level(log::LevelFilter::Info)
                .build(),
        )
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            log::info!("Starting Braincells Desktop App");

            // Initialize app state
            let data_dir = get_data_dir();
            let state = AppState::new(data_dir);

            // Initialize state in background
            let state_clone = state.inference_pool.clone();
            let settings_clone = state.settings.clone();
            let cache_clone = state.model_cache.clone();
            let data_dir_clone = state.data_dir.clone();

            tauri::async_runtime::spawn(async move {
                let temp_state = AppState {
                    inference_pool: state_clone,
                    settings: settings_clone,
                    model_cache: cache_clone,
                    data_dir: data_dir_clone,
                };
                if let Err(e) = temp_state.initialize().await {
                    log::error!("Failed to initialize app state: {}", e);
                }
            });

            // Manage state
            app.manage(state);

            log::info!("Braincells setup complete");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Inference commands
            commands::generate_cell,
            commands::generate_batch,
            commands::is_inference_ready,
            commands::get_backend_name,
            commands::unload_model,

            // Model commands
            commands::list_models,
            commands::get_recommended_models,
            commands::get_recommended_models_for_system,
            commands::suggest_model,
            commands::download_model,
            commands::delete_model,
            commands::get_models_directory,
            commands::is_model_downloaded,
            commands::get_cache_stats,
            commands::cleanup_cache,

            // Settings commands
            commands::get_settings,
            commands::update_settings,
            commands::configure_backend,
            commands::set_hf_api_key,
            commands::set_openrouter_api_key,
            commands::set_openai_api_key,
            commands::set_default_model,
            commands::set_max_concurrent,
            commands::quick_configure_local,
            commands::quick_configure_huggingface,
            commands::quick_configure_openrouter,
            commands::quick_configure_openai,
            commands::get_available_providers,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Braincells");
}
