//! Model management module
//!
//! Handles model discovery, downloading, and caching.

pub mod cache;
pub mod download;
pub mod registry;

pub use cache::ModelCache;
pub use download::{download_model, DownloadHandle};
pub use registry::{get_recommended_models, RECOMMENDED_MODELS};

use crate::llm::types::{LocalModel, LLMError};
use std::path::PathBuf;

/// Get the default models directory
pub fn get_models_dir() -> Result<PathBuf, LLMError> {
    let base_dir = dirs::data_dir()
        .or_else(|| dirs::home_dir().map(|h| h.join(".local/share")))
        .ok_or_else(|| LLMError::IoError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Could not determine data directory",
        )))?;

    let models_dir = base_dir.join("braincells").join("models");

    // Create directory if it doesn't exist
    std::fs::create_dir_all(&models_dir)?;

    Ok(models_dir)
}

/// List all local models
pub fn list_local_models() -> Result<Vec<LocalModel>, LLMError> {
    let models_dir = get_models_dir()?;
    let mut models = Vec::new();

    if !models_dir.exists() {
        return Ok(models);
    }

    for entry in std::fs::read_dir(&models_dir)? {
        let entry = entry?;
        let path = entry.path();

        // Only include GGUF files
        if path.extension().map(|e| e == "gguf").unwrap_or(false) {
            let metadata = entry.metadata()?;
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();

            models.push(LocalModel {
                name,
                path: path.to_string_lossy().to_string(),
                size_bytes: metadata.len(),
                format: "gguf".to_string(),
            });
        }
    }

    // Sort by name
    models.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(models)
}

/// Delete a local model
pub fn delete_model(model_path: &str) -> Result<(), LLMError> {
    let path = PathBuf::from(model_path);

    // Safety check: only allow deleting from models directory
    let models_dir = get_models_dir()?;
    if !path.starts_with(&models_dir) {
        return Err(LLMError::InvalidConfig(
            "Cannot delete model outside of models directory".to_string(),
        ));
    }

    std::fs::remove_file(&path)?;
    log::info!("Deleted model: {}", model_path);

    Ok(())
}
