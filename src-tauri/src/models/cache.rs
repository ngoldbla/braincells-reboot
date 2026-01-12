//! Model cache management
//!
//! This module handles caching of models and cleanup of old/unused models.

use crate::llm::types::{LocalModel, LLMError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Cache metadata for tracking model usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    pub models: HashMap<String, ModelCacheEntry>,
    pub total_size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCacheEntry {
    pub path: String,
    pub size_bytes: u64,
    pub last_used: chrono::DateTime<chrono::Utc>,
    pub download_date: chrono::DateTime<chrono::Utc>,
    pub repo_id: Option<String>,
}

impl Default for CacheMetadata {
    fn default() -> Self {
        Self {
            models: HashMap::new(),
            total_size_bytes: 0,
        }
    }
}

/// Model cache manager
pub struct ModelCache {
    cache_dir: PathBuf,
    metadata_path: PathBuf,
    metadata: CacheMetadata,
}

impl ModelCache {
    /// Create a new model cache at the given directory
    pub fn new(cache_dir: PathBuf) -> Result<Self, LLMError> {
        std::fs::create_dir_all(&cache_dir)?;

        let metadata_path = cache_dir.join("cache_metadata.json");
        let metadata = if metadata_path.exists() {
            let content = std::fs::read_to_string(&metadata_path)?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            CacheMetadata::default()
        };

        Ok(Self {
            cache_dir,
            metadata_path,
            metadata,
        })
    }

    /// Get the cache directory
    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }

    /// Save metadata to disk
    fn save_metadata(&self) -> Result<(), LLMError> {
        let content = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| LLMError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to serialize metadata: {}", e),
            )))?;
        std::fs::write(&self.metadata_path, content)?;
        Ok(())
    }

    /// Register a model in the cache
    pub fn register_model(
        &mut self,
        name: String,
        path: String,
        size_bytes: u64,
        repo_id: Option<String>,
    ) -> Result<(), LLMError> {
        let now = chrono::Utc::now();

        self.metadata.models.insert(
            name,
            ModelCacheEntry {
                path,
                size_bytes,
                last_used: now,
                download_date: now,
                repo_id,
            },
        );

        self.recalculate_total_size();
        self.save_metadata()
    }

    /// Mark a model as recently used
    pub fn touch_model(&mut self, name: &str) -> Result<(), LLMError> {
        if let Some(entry) = self.metadata.models.get_mut(name) {
            entry.last_used = chrono::Utc::now();
            self.save_metadata()?;
        }
        Ok(())
    }

    /// Remove a model from the cache
    pub fn remove_model(&mut self, name: &str) -> Result<(), LLMError> {
        if let Some(entry) = self.metadata.models.remove(name) {
            // Delete the file
            let path = PathBuf::from(&entry.path);
            if path.exists() {
                std::fs::remove_file(&path)?;
            }
            self.recalculate_total_size();
            self.save_metadata()?;
        }
        Ok(())
    }

    /// Get all cached models
    pub fn list_models(&self) -> Vec<LocalModel> {
        self.metadata
            .models
            .iter()
            .map(|(name, entry)| LocalModel {
                name: name.clone(),
                path: entry.path.clone(),
                size_bytes: entry.size_bytes,
                format: "gguf".to_string(),
            })
            .collect()
    }

    /// Get total cache size in bytes
    pub fn total_size(&self) -> u64 {
        self.metadata.total_size_bytes
    }

    /// Recalculate total size from entries
    fn recalculate_total_size(&mut self) {
        self.metadata.total_size_bytes = self
            .metadata
            .models
            .values()
            .map(|e| e.size_bytes)
            .sum();
    }

    /// Clean up models that haven't been used recently
    pub fn cleanup_old_models(&mut self, max_age_days: i64) -> Result<Vec<String>, LLMError> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(max_age_days);
        let to_remove: Vec<String> = self
            .metadata
            .models
            .iter()
            .filter(|(_, entry)| entry.last_used < cutoff)
            .map(|(name, _)| name.clone())
            .collect();

        for name in &to_remove {
            self.remove_model(name)?;
        }

        Ok(to_remove)
    }

    /// Clean up to stay under a size limit
    pub fn cleanup_to_size(&mut self, max_size_bytes: u64) -> Result<Vec<String>, LLMError> {
        if self.metadata.total_size_bytes <= max_size_bytes {
            return Ok(Vec::new());
        }

        // Sort by last_used (oldest first) and collect names
        let mut models: Vec<_> = self.metadata.models.iter()
            .map(|(name, entry)| (name.clone(), entry.last_used))
            .collect();
        models.sort_by_key(|(_, last_used)| *last_used);

        let mut removed = Vec::new();
        for (name, _) in models {
            if self.metadata.total_size_bytes <= max_size_bytes {
                break;
            }
            self.remove_model(&name)?;
            removed.push(name);
        }

        Ok(removed)
    }

    /// Sync cache metadata with actual files on disk
    pub fn sync_with_disk(&mut self) -> Result<(), LLMError> {
        // Remove entries for files that no longer exist
        let to_remove: Vec<String> = self
            .metadata
            .models
            .iter()
            .filter(|(_, entry)| !PathBuf::from(&entry.path).exists())
            .map(|(name, _)| name.clone())
            .collect();

        for name in to_remove {
            self.metadata.models.remove(&name);
        }

        // Add entries for new files
        if self.cache_dir.exists() {
            for entry in std::fs::read_dir(&self.cache_dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.extension().map(|e| e == "gguf").unwrap_or(false) {
                    let name = path
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_default();

                    if !self.metadata.models.contains_key(&name) {
                        let metadata = entry.metadata()?;
                        self.register_model(
                            name,
                            path.to_string_lossy().to_string(),
                            metadata.len(),
                            None,
                        )?;
                    }
                }
            }
        }

        self.recalculate_total_size();
        self.save_metadata()
    }
}
