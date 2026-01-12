//! Application state management
//!
//! This module manages the global application state, including
//! the inference pool, settings, and model cache.

use crate::llm::{InferencePool, Settings};
use crate::models::cache::ModelCache;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Application state shared across all commands
pub struct AppState {
    /// The active inference pool (if configured)
    pub inference_pool: Arc<RwLock<Option<InferencePool>>>,

    /// Application settings
    pub settings: Arc<RwLock<Settings>>,

    /// Model cache
    pub model_cache: Arc<RwLock<Option<ModelCache>>>,

    /// Data directory path
    pub data_dir: PathBuf,
}

impl AppState {
    /// Create a new app state with the given data directory
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            inference_pool: Arc::new(RwLock::new(None)),
            settings: Arc::new(RwLock::new(Settings::default())),
            model_cache: Arc::new(RwLock::new(None)),
            data_dir,
        }
    }

    /// Initialize the app state (load settings, setup cache, etc.)
    pub async fn initialize(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Create data directory if needed
        std::fs::create_dir_all(&self.data_dir)?;

        // Load settings
        let settings_path = self.data_dir.join("settings.json");
        if settings_path.exists() {
            let content = std::fs::read_to_string(&settings_path)?;
            if let Ok(settings) = serde_json::from_str::<Settings>(&content) {
                let mut guard = self.settings.write().await;
                *guard = settings;
            }
        }

        // Initialize model cache
        let models_dir = crate::models::get_models_dir()?;
        let mut cache = ModelCache::new(models_dir)?;
        cache.sync_with_disk()?;

        let mut cache_guard = self.model_cache.write().await;
        *cache_guard = Some(cache);

        log::info!("AppState initialized with data_dir: {}", self.data_dir.display());

        Ok(())
    }

    /// Save settings to disk
    pub async fn save_settings(&self, settings: &Settings) -> Result<(), std::io::Error> {
        let settings_path = self.data_dir.join("settings.json");
        let content = serde_json::to_string_pretty(settings)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(settings_path, content)?;
        Ok(())
    }

    /// Get the settings path
    pub fn settings_path(&self) -> PathBuf {
        self.data_dir.join("settings.json")
    }

    /// Check if a backend is configured and ready
    pub async fn is_ready(&self) -> bool {
        let pool = self.inference_pool.read().await;
        match pool.as_ref() {
            Some(pool) => pool.is_ready().await,
            None => false,
        }
    }
}

/// Get the default data directory for the app
pub fn get_data_dir() -> PathBuf {
    dirs::data_dir()
        .or_else(|| dirs::home_dir().map(|h| h.join(".local/share")))
        .map(|d| d.join("braincells"))
        .unwrap_or_else(|| PathBuf::from(".braincells"))
}
