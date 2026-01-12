//! Model download functionality using HuggingFace Hub
//!
//! This module handles downloading models from the HuggingFace Hub
//! with progress tracking and resume support.

use crate::llm::types::{DownloadProgress, LLMError};
use hf_hub::api::sync::ApiBuilder;
use std::path::PathBuf;
use tokio::sync::mpsc;

/// Handle for tracking download progress
pub struct DownloadHandle {
    pub progress_rx: mpsc::Receiver<DownloadProgress>,
    pub result_rx: tokio::sync::oneshot::Receiver<Result<PathBuf, LLMError>>,
}

/// Download a model from HuggingFace Hub
///
/// Returns a handle that can be used to track progress and get the final result.
pub fn download_model(
    repo_id: String,
    filename: String,
    dest_dir: PathBuf,
    hf_token: Option<String>,
) -> DownloadHandle {
    let (progress_tx, progress_rx) = mpsc::channel(100);
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();

    // Spawn blocking task for download
    std::thread::spawn(move || {
        let result = download_model_sync(&repo_id, &filename, &dest_dir, hf_token.as_deref(), |progress| {
            let _ = progress_tx.blocking_send(progress);
        });
        let _ = result_tx.send(result);
    });

    DownloadHandle {
        progress_rx,
        result_rx,
    }
}

/// Synchronous model download with progress callback
fn download_model_sync<F>(
    repo_id: &str,
    filename: &str,
    dest_dir: &PathBuf,
    hf_token: Option<&str>,
    mut on_progress: F,
) -> Result<PathBuf, LLMError>
where
    F: FnMut(DownloadProgress),
{
    log::info!("Starting download: {}/{}", repo_id, filename);

    // Build API client
    let mut builder = ApiBuilder::new();
    if let Some(token) = hf_token {
        builder = builder.with_token(Some(token.to_string()));
    }

    let api = builder
        .build()
        .map_err(|e| LLMError::DownloadFailed(format!("Failed to build API: {}", e)))?;

    // Get the repo
    let repo = api.model(repo_id.to_string());

    // Send initial progress
    on_progress(DownloadProgress {
        model_name: filename.to_string(),
        downloaded_bytes: 0,
        total_bytes: 0,
        percentage: 0.0,
    });

    // Download the file
    // Note: hf-hub doesn't provide progress callbacks directly,
    // so we'll report completion at the end
    let cache_path = repo
        .get(filename)
        .map_err(|e| LLMError::DownloadFailed(format!("Download failed: {}", e)))?;

    // Copy from cache to destination
    std::fs::create_dir_all(dest_dir)?;
    let dest_path = dest_dir.join(filename);

    // Get file size
    let metadata = std::fs::metadata(&cache_path)?;
    let total_bytes = metadata.len();

    // Copy with progress
    let mut src = std::fs::File::open(&cache_path)?;
    let mut dest = std::fs::File::create(&dest_path)?;

    let mut downloaded: u64 = 0;
    let mut buffer = vec![0u8; 1024 * 1024]; // 1MB buffer

    loop {
        use std::io::Read;
        let bytes_read = src.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }

        use std::io::Write;
        dest.write_all(&buffer[..bytes_read])?;
        downloaded += bytes_read as u64;

        on_progress(DownloadProgress {
            model_name: filename.to_string(),
            downloaded_bytes: downloaded,
            total_bytes,
            percentage: (downloaded as f32 / total_bytes as f32) * 100.0,
        });
    }

    log::info!("Download complete: {}", dest_path.display());

    Ok(dest_path)
}

/// Download a model asynchronously (wrapper for async contexts)
pub async fn download_model_async(
    repo_id: String,
    filename: String,
    dest_dir: PathBuf,
    hf_token: Option<String>,
) -> Result<PathBuf, LLMError> {
    tokio::task::spawn_blocking(move || {
        download_model_sync(&repo_id, &filename, &dest_dir, hf_token.as_deref(), |_| {})
    })
    .await
    .map_err(|e| LLMError::DownloadFailed(format!("Task join error: {}", e)))?
}

/// Check if a model is already downloaded
pub fn is_model_downloaded(dest_dir: &PathBuf, filename: &str) -> bool {
    dest_dir.join(filename).exists()
}

/// Get the expected path for a downloaded model
pub fn get_model_path(dest_dir: &PathBuf, filename: &str) -> PathBuf {
    dest_dir.join(filename)
}
