//! LLM types and traits for unified inference interface
//!
//! This module defines the core abstractions for LLM inference,
//! supporting both local (llama.cpp) and cloud (HF, OpenRouter, OpenAI) backends.

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

/// A message in a chat conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

/// Request for LLM generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMRequest {
    pub messages: Vec<Message>,
    pub model: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
}

impl Default for LLMRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            model: String::new(),
            max_tokens: Some(2048),
            temperature: Some(0.7),
            stop_sequences: None,
            stream: false,
        }
    }
}

/// Response from LLM generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    pub content: String,
    pub tokens_used: u32,
    pub model: String,
    pub finish_reason: String,
}

/// Configuration for different LLM backends
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum BackendConfig {
    Local {
        model_path: String,
        #[serde(default = "default_gpu_layers")]
        gpu_layers: u32,
        #[serde(default = "default_context_size")]
        context_size: u32,
    },
    HuggingFace {
        api_key: String,
        model_id: String,
        #[serde(default)]
        provider: Option<String>,
    },
    OpenRouter {
        api_key: String,
        model_id: String,
    },
    OpenAI {
        api_key: String,
        model_id: String,
    },
}

fn default_gpu_layers() -> u32 {
    999 // Offload all layers to GPU by default
}

fn default_context_size() -> u32 {
    4096
}

/// Errors that can occur during LLM operations
#[derive(Debug, thiserror::Error)]
pub enum LLMError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Rate limited - please wait before retrying")]
    RateLimited,

    #[error("Context length exceeded - input too long")]
    ContextLengthExceeded,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Download failed: {0}")]
    DownloadFailed(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

// Implement conversion to String for Tauri
impl From<LLMError> for String {
    fn from(err: LLMError) -> Self {
        err.to_string()
    }
}

/// Stream item for streaming responses
pub type StreamItem = Result<String, LLMError>;
pub type LLMStream = Pin<Box<dyn Stream<Item = StreamItem> + Send>>;

/// Trait for LLM inference engines
#[async_trait]
pub trait LLMEngine: Send + Sync {
    /// Generate a complete response
    async fn generate(&self, request: LLMRequest) -> Result<LLMResponse, LLMError>;

    /// Generate with streaming (optional, default falls back to generate)
    async fn generate_stream(&self, request: LLMRequest) -> Result<LLMStream, LLMError> {
        // Default implementation: just return the full response as a single chunk
        let response = self.generate(request).await?;
        let stream = futures::stream::once(async move { Ok(response.content) });
        Ok(Box::pin(stream))
    }

    /// Check if the engine is ready for inference
    async fn is_ready(&self) -> bool;

    /// Get the backend type name
    fn backend_name(&self) -> &'static str;

    /// Unload/cleanup resources (optional)
    async fn unload(&self) -> Result<(), LLMError> {
        Ok(())
    }
}

/// Information about a local model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModel {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub format: String,
}

/// Information about a recommended model for download
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedModel {
    pub repo_id: String,
    pub filename: String,
    pub display_name: String,
    pub size_gb: f64,
    pub description: String,
    pub quantization: String,
}

/// Model download progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub model_name: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub percentage: f32,
}

/// Application settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,
    #[serde(default)]
    pub active_backend: Option<BackendConfig>,
    #[serde(default)]
    pub hf_api_key: Option<String>,
    #[serde(default)]
    pub openrouter_api_key: Option<String>,
    #[serde(default)]
    pub openai_api_key: Option<String>,
    #[serde(default)]
    pub default_local_model: Option<String>,
    #[serde(default = "default_temperature")]
    pub default_temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub default_max_tokens: u32,
}

fn default_max_concurrent() -> usize {
    5
}

fn default_temperature() -> f32 {
    0.7
}

fn default_max_tokens() -> u32 {
    2048
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_concurrent_requests: default_max_concurrent(),
            active_backend: None,
            hf_api_key: None,
            openrouter_api_key: None,
            openai_api_key: None,
            default_local_model: None,
            default_temperature: default_temperature(),
            default_max_tokens: default_max_tokens(),
        }
    }
}
