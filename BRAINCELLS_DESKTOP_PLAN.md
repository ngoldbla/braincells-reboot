# Braincells Desktop App - Implementation Plan

## Executive Summary

Transform Hugging Face AI Sheets into **Braincells**, a standalone cross-platform desktop application with native LLM inference. The app will use llama.cpp directly (no Ollama) for maximum performance, with Hugging Face Inference API as a cloud fallback.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BRAINCELLS DESKTOP                               │
│                      (Tauri 2.0 + Rust + Web)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    WEB FRONTEND (Qwik)                          │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │   │
│  │  │ Spreadsheet  │ │   Model      │ │   Settings Panel         │ │   │
│  │  │ UI (existing)│ │   Picker     │ │   - Backend selection    │ │   │
│  │  │              │ │   (local/    │ │   - HF API key           │ │   │
│  │  │              │ │    cloud)    │ │   - Concurrency settings │ │   │
│  │  └──────────────┘ └──────────────┘ └──────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              │ Tauri IPC (invoke)                       │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    RUST BACKEND (src-tauri)                     │   │
│  │                                                                  │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │              UNIFIED LLM INTERFACE (trait)               │   │   │
│  │  │                                                          │   │   │
│  │  │   async fn generate(&self, request: LLMRequest)          │   │   │
│  │  │       -> Result<LLMResponse>                             │   │   │
│  │  │                                                          │   │   │
│  │  │   async fn generate_stream(&self, request: LLMRequest)   │   │   │
│  │  │       -> impl Stream<Item = Result<String>>              │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │                              │                                   │   │
│  │              ┌───────────────┼───────────────┐                   │   │
│  │              ▼               ▼               ▼                   │   │
│  │  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐       │   │
│  │  │  LOCAL ENGINE  │ │  LOCAL ENGINE  │ │  CLOUD ENGINE  │       │   │
│  │  │    (macOS)     │ │ (Win/Linux)    │ │  (HF Inference)│       │   │
│  │  │                │ │                │ │                │       │   │
│  │  │  llama.cpp     │ │  llama.cpp     │ │  HF Pro API    │       │   │
│  │  │  + Metal       │ │  + CUDA        │ │  OpenRouter    │       │   │
│  │  │                │ │                │ │  OpenAI        │       │   │
│  │  │  Optional:     │ │                │ │                │       │   │
│  │  │  MLX Swift     │ │                │ │                │       │   │
│  │  │  sidecar       │ │                │ │                │       │   │
│  │  └────────────────┘ └────────────────┘ └────────────────┘       │   │
│  │                                                                  │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │                    DATA LAYER                            │   │   │
│  │  │  DuckDB (duckdb-rs) │ Model Cache │ Settings (SQLite)    │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Technical Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Desktop Framework** | Tauri 2.0 | Small binary (~15MB), native performance, Rust backend |
| **Frontend** | Qwik (existing) | Reuse existing codebase, minimal porting |
| **Local LLM** | llama.cpp via `llama-cpp-2` crate | Direct integration, no Ollama overhead, GPU acceleration |
| **macOS Optimization** | Optional MLX Swift sidecar | 2x faster on Apple Silicon |
| **Cloud Fallback** | HF Inference API | For users without local compute |
| **Data Storage** | DuckDB + SQLite | Existing compatibility, excellent performance |
| **Model Format** | GGUF | Universal format, quantization options |

---

## Phase 1: Foundation (Core Tauri Setup)

### 1.1 Project Structure

```
braincells/
├── src/                          # Existing Qwik frontend (minimal changes)
│   ├── components/
│   ├── features/
│   ├── routes/
│   ├── services/
│   │   └── inference/            # MODIFY: Bridge to Tauri backend
│   └── config.ts                 # MODIFY: Desktop-aware config
│
├── src-tauri/                    # NEW: Rust backend
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   ├── build.rs                  # Platform-specific llama.cpp compilation
│   ├── src/
│   │   ├── main.rs               # Tauri entry point
│   │   ├── lib.rs
│   │   ├── commands/             # Tauri IPC commands
│   │   │   ├── mod.rs
│   │   │   ├── inference.rs      # LLM generation commands
│   │   │   ├── models.rs         # Model management
│   │   │   └── data.rs           # Import/export, DuckDB
│   │   ├── llm/                  # LLM abstraction layer
│   │   │   ├── mod.rs
│   │   │   ├── traits.rs         # Unified LLM interface
│   │   │   ├── local.rs          # llama.cpp backend
│   │   │   ├── cloud.rs          # HF/OpenRouter/OpenAI
│   │   │   └── pool.rs           # Connection pooling for parallel calls
│   │   ├── models/               # Model management
│   │   │   ├── mod.rs
│   │   │   ├── download.rs       # HF Hub model downloads
│   │   │   ├── cache.rs          # Model cache management
│   │   │   └── registry.rs       # Available models
│   │   └── data/                 # Data layer
│   │       ├── mod.rs
│   │       ├── duckdb.rs         # DuckDB operations
│   │       └── settings.rs       # App settings persistence
│   └── resources/                # Bundled resources
│       └── mlx-inference/        # Optional: MLX sidecar binary
│
├── package.json                  # Add tauri dependencies
├── tauri.conf.json              # Tauri configuration
└── CLAUDE.md
```

### 1.2 Tauri Configuration

```json
// src-tauri/tauri.conf.json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "Braincells",
  "identifier": "ai.braincells.app",
  "version": "0.1.0",
  "build": {
    "beforeDevCommand": "pnpm dev",
    "devUrl": "http://localhost:5173",
    "beforeBuildCommand": "pnpm build",
    "frontendDist": "../dist"
  },
  "app": {
    "windows": [
      {
        "title": "Braincells",
        "width": 1400,
        "height": 900,
        "minWidth": 800,
        "minHeight": 600,
        "decorations": true,
        "transparent": false
      }
    ],
    "security": {
      "csp": null
    }
  },
  "bundle": {
    "active": true,
    "targets": ["dmg", "msi", "deb", "appimage"],
    "icon": [
      "icons/32x32.png",
      "icons/128x128.png",
      "icons/128x128@2x.png",
      "icons/icon.icns",
      "icons/icon.ico"
    ],
    "resources": ["resources/*"],
    "macOS": {
      "minimumSystemVersion": "11.0",
      "frameworks": ["Metal.framework", "Accelerate.framework"]
    },
    "windows": {
      "webviewInstallMode": { "type": "embedBootstrapper" }
    }
  },
  "plugins": {
    "shell": { "open": true },
    "fs": {
      "scope": ["$APPDATA/**", "$DOWNLOAD/**", "$HOME/.braincells/**"]
    }
  }
}
```

### 1.3 Cargo Dependencies

```toml
# src-tauri/Cargo.toml
[package]
name = "braincells"
version = "0.1.0"
edition = "2021"

[dependencies]
tauri = { version = "2", features = ["protocol-asset"] }
tauri-plugin-shell = "2"
tauri-plugin-fs = "2"
tauri-plugin-dialog = "2"
tauri-plugin-http = "2"

# LLM inference
llama-cpp-2 = "0.1"

# Async runtime
tokio = { version = "1", features = ["full"] }
futures = "0.3"

# Data
duckdb = { version = "1.0", features = ["bundled"] }
rusqlite = { version = "0.32", features = ["bundled"] }

# HTTP for cloud APIs
reqwest = { version = "0.12", features = ["json", "stream"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# HuggingFace Hub
hf-hub = "0.3"

# Utilities
thiserror = "1"
tracing = "0.1"
tracing-subscriber = "0.3"

[target.'cfg(target_os = "macos")'.dependencies]
llama-cpp-2 = { version = "0.1", features = ["metal"] }

[target.'cfg(target_os = "windows")'.dependencies]
llama-cpp-2 = { version = "0.1", features = ["cuda"] }

[target.'cfg(target_os = "linux")'.dependencies]
llama-cpp-2 = { version = "0.1", features = ["cuda"] }

[build-dependencies]
tauri-build = { version = "2", features = [] }
```

---

## Phase 2: Unified LLM Interface

### 2.1 Core Traits

```rust
// src-tauri/src/llm/traits.rs

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMRequest {
    pub messages: Vec<Message>,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,      // "system", "user", "assistant"
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    pub content: String,
    pub tokens_used: u32,
    pub model: String,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LLMBackend {
    Local { model_path: String },
    HuggingFace { api_key: String, model_id: String },
    OpenRouter { api_key: String, model_id: String },
    OpenAI { api_key: String, model_id: String },
}

#[async_trait]
pub trait LLMEngine: Send + Sync {
    /// Generate a complete response
    async fn generate(&self, request: LLMRequest) -> Result<LLMResponse, LLMError>;

    /// Generate with streaming
    async fn generate_stream(
        &self,
        request: LLMRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>;

    /// Check if the engine is ready
    async fn is_ready(&self) -> bool;

    /// Get the backend type
    fn backend(&self) -> LLMBackend;
}

#[derive(Debug, thiserror::Error)]
pub enum LLMError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Rate limited")]
    RateLimited,
    #[error("Context length exceeded")]
    ContextLengthExceeded,
}
```

### 2.2 Local llama.cpp Backend

```rust
// src-tauri/src/llm/local.rs

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::context::params::LlamaContextParams;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct LocalLLMEngine {
    model: Arc<RwLock<Option<LoadedModel>>>,
    model_path: String,
    gpu_layers: u32,
}

struct LoadedModel {
    model: LlamaModel,
    context: LlamaContext,
}

impl LocalLLMEngine {
    pub fn new(model_path: String) -> Self {
        // Auto-detect GPU layers based on platform
        let gpu_layers = Self::detect_gpu_layers();

        Self {
            model: Arc::new(RwLock::new(None)),
            model_path,
            gpu_layers,
        }
    }

    fn detect_gpu_layers() -> u32 {
        #[cfg(target_os = "macos")]
        {
            // Metal: offload all layers to GPU
            999
        }
        #[cfg(target_os = "windows")]
        {
            // CUDA: check VRAM and decide
            if Self::has_cuda() { 35 } else { 0 }
        }
        #[cfg(target_os = "linux")]
        {
            if Self::has_cuda() { 35 } else { 0 }
        }
    }

    pub async fn load_model(&self) -> Result<(), LLMError> {
        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(self.gpu_layers);

        let model = LlamaModel::load_from_file(&self.model_path, model_params)
            .map_err(|e| LLMError::ModelNotFound(e.to_string()))?;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(4096));

        let context = model.new_context(&ctx_params)
            .map_err(|e| LLMError::InferenceFailed(e.to_string()))?;

        let mut guard = self.model.write().await;
        *guard = Some(LoadedModel { model, context });

        Ok(())
    }

    pub async fn unload_model(&self) {
        let mut guard = self.model.write().await;
        *guard = None;
    }
}

#[async_trait]
impl LLMEngine for LocalLLMEngine {
    async fn generate(&self, request: LLMRequest) -> Result<LLMResponse, LLMError> {
        let guard = self.model.read().await;
        let loaded = guard.as_ref()
            .ok_or_else(|| LLMError::ModelNotFound("Model not loaded".into()))?;

        // Build prompt from messages
        let prompt = Self::format_chat_prompt(&request.messages);

        // Tokenize and generate
        // ... llama.cpp generation logic ...

        Ok(LLMResponse {
            content: generated_text,
            tokens_used: token_count,
            model: request.model,
            finish_reason: "stop".into(),
        })
    }

    async fn generate_stream(
        &self,
        request: LLMRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        // Streaming implementation
        todo!()
    }

    async fn is_ready(&self) -> bool {
        self.model.read().await.is_some()
    }

    fn backend(&self) -> LLMBackend {
        LLMBackend::Local { model_path: self.model_path.clone() }
    }
}
```

### 2.3 Cloud Backend (HuggingFace + OpenRouter + OpenAI)

```rust
// src-tauri/src/llm/cloud.rs

use reqwest::Client;
use serde_json::json;

pub struct CloudLLMEngine {
    client: Client,
    provider: CloudProvider,
    api_key: String,
    model_id: String,
}

pub enum CloudProvider {
    HuggingFace,
    OpenRouter,
    OpenAI,
}

impl CloudProvider {
    fn base_url(&self) -> &str {
        match self {
            CloudProvider::HuggingFace => "https://api-inference.huggingface.co/v1",
            CloudProvider::OpenRouter => "https://openrouter.ai/api/v1",
            CloudProvider::OpenAI => "https://api.openai.com/v1",
        }
    }

    fn auth_header(&self) -> &str {
        match self {
            CloudProvider::HuggingFace => "Bearer",
            CloudProvider::OpenRouter => "Bearer",
            CloudProvider::OpenAI => "Bearer",
        }
    }
}

impl CloudLLMEngine {
    pub fn new(provider: CloudProvider, api_key: String, model_id: String) -> Self {
        Self {
            client: Client::new(),
            provider,
            api_key,
            model_id,
        }
    }
}

#[async_trait]
impl LLMEngine for CloudLLMEngine {
    async fn generate(&self, request: LLMRequest) -> Result<LLMResponse, LLMError> {
        let url = format!("{}/chat/completions", self.provider.base_url());

        let body = json!({
            "model": self.model_id,
            "messages": request.messages.iter().map(|m| {
                json!({ "role": m.role, "content": m.content })
            }).collect::<Vec<_>>(),
            "max_tokens": request.max_tokens.unwrap_or(2048),
            "temperature": request.temperature.unwrap_or(0.7),
        });

        let response = self.client
            .post(&url)
            .header("Authorization", format!("{} {}", self.provider.auth_header(), self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::ApiError(e.to_string()))?;

        if response.status() == 429 {
            return Err(LLMError::RateLimited);
        }

        let result: serde_json::Value = response.json().await
            .map_err(|e| LLMError::ApiError(e.to_string()))?;

        let content = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(LLMResponse {
            content,
            tokens_used: result["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
            model: self.model_id.clone(),
            finish_reason: result["choices"][0]["finish_reason"]
                .as_str()
                .unwrap_or("stop")
                .to_string(),
        })
    }

    async fn generate_stream(
        &self,
        request: LLMRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        // SSE streaming implementation
        todo!()
    }

    async fn is_ready(&self) -> bool {
        !self.api_key.is_empty()
    }

    fn backend(&self) -> LLMBackend {
        match self.provider {
            CloudProvider::HuggingFace => LLMBackend::HuggingFace {
                api_key: self.api_key.clone(),
                model_id: self.model_id.clone(),
            },
            CloudProvider::OpenRouter => LLMBackend::OpenRouter {
                api_key: self.api_key.clone(),
                model_id: self.model_id.clone(),
            },
            CloudProvider::OpenAI => LLMBackend::OpenAI {
                api_key: self.api_key.clone(),
                model_id: self.model_id.clone(),
            },
        }
    }
}
```

### 2.4 Parallel Inference Pool

```rust
// src-tauri/src/llm/pool.rs

use std::sync::Arc;
use tokio::sync::Semaphore;
use futures::stream::{self, StreamExt};

pub struct InferencePool {
    engine: Arc<dyn LLMEngine>,
    semaphore: Arc<Semaphore>,
    max_concurrent: usize,
}

impl InferencePool {
    pub fn new(engine: Arc<dyn LLMEngine>, max_concurrent: usize) -> Self {
        Self {
            engine,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        }
    }

    /// Execute multiple inference requests in parallel
    pub async fn batch_generate(
        &self,
        requests: Vec<LLMRequest>,
    ) -> Vec<Result<LLMResponse, LLMError>> {
        stream::iter(requests)
            .map(|request| {
                let engine = self.engine.clone();
                let semaphore = self.semaphore.clone();

                async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    engine.generate(request).await
                }
            })
            .buffer_unordered(self.max_concurrent)
            .collect()
            .await
    }

    /// Generate for a single cell (rate-limited)
    pub async fn generate_single(&self, request: LLMRequest) -> Result<LLMResponse, LLMError> {
        let _permit = self.semaphore.acquire().await.unwrap();
        self.engine.generate(request).await
    }
}
```

---

## Phase 3: Tauri Commands (IPC Bridge)

### 3.1 Inference Commands

```rust
// src-tauri/src/commands/inference.rs

use crate::llm::{LLMRequest, LLMResponse, LLMError, InferencePool};
use tauri::State;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct AppState {
    pub inference_pool: Arc<RwLock<Option<InferencePool>>>,
    pub settings: Arc<RwLock<Settings>>,
}

#[tauri::command]
pub async fn generate_cell(
    state: State<'_, AppState>,
    request: LLMRequest,
) -> Result<LLMResponse, String> {
    let pool = state.inference_pool.read().await;
    let pool = pool.as_ref().ok_or("No LLM engine configured")?;

    pool.generate_single(request)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn generate_batch(
    state: State<'_, AppState>,
    requests: Vec<LLMRequest>,
) -> Result<Vec<Result<LLMResponse, String>>, String> {
    let pool = state.inference_pool.read().await;
    let pool = pool.as_ref().ok_or("No LLM engine configured")?;

    let results = pool.batch_generate(requests).await;
    Ok(results.into_iter().map(|r| r.map_err(|e| e.to_string())).collect())
}

#[tauri::command]
pub async fn configure_backend(
    state: State<'_, AppState>,
    backend_config: BackendConfig,
) -> Result<(), String> {
    let engine: Arc<dyn LLMEngine> = match backend_config {
        BackendConfig::Local { model_path } => {
            let engine = LocalLLMEngine::new(model_path);
            engine.load_model().await.map_err(|e| e.to_string())?;
            Arc::new(engine)
        }
        BackendConfig::HuggingFace { api_key, model_id } => {
            Arc::new(CloudLLMEngine::new(
                CloudProvider::HuggingFace,
                api_key,
                model_id,
            ))
        }
        BackendConfig::OpenRouter { api_key, model_id } => {
            Arc::new(CloudLLMEngine::new(
                CloudProvider::OpenRouter,
                api_key,
                model_id,
            ))
        }
        BackendConfig::OpenAI { api_key, model_id } => {
            Arc::new(CloudLLMEngine::new(
                CloudProvider::OpenAI,
                api_key,
                model_id,
            ))
        }
    };

    let pool = InferencePool::new(engine, backend_config.max_concurrent.unwrap_or(5));

    let mut guard = state.inference_pool.write().await;
    *guard = Some(pool);

    Ok(())
}
```

### 3.2 Model Management Commands

```rust
// src-tauri/src/commands/models.rs

use hf_hub::api::sync::Api;
use std::path::PathBuf;

#[tauri::command]
pub async fn list_local_models(
    state: State<'_, AppState>,
) -> Result<Vec<LocalModel>, String> {
    let models_dir = get_models_dir()?;

    let mut models = Vec::new();
    for entry in std::fs::read_dir(&models_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();

        if path.extension().map(|e| e == "gguf").unwrap_or(false) {
            models.push(LocalModel {
                name: path.file_stem().unwrap().to_string_lossy().to_string(),
                path: path.to_string_lossy().to_string(),
                size_bytes: entry.metadata().map(|m| m.len()).unwrap_or(0),
            });
        }
    }

    Ok(models)
}

#[tauri::command]
pub async fn download_model(
    app: tauri::AppHandle,
    repo_id: String,
    filename: String,
) -> Result<String, String> {
    let api = Api::new().map_err(|e| e.to_string())?;
    let repo = api.model(repo_id);

    // Emit progress events
    let path = repo.get(&filename).map_err(|e| e.to_string())?;

    // Move to models directory
    let models_dir = get_models_dir()?;
    let dest = models_dir.join(&filename);
    std::fs::copy(&path, &dest).map_err(|e| e.to_string())?;

    Ok(dest.to_string_lossy().to_string())
}

#[tauri::command]
pub async fn get_recommended_models() -> Result<Vec<RecommendedModel>, String> {
    Ok(vec![
        RecommendedModel {
            repo_id: "bartowski/Llama-3.2-3B-Instruct-GGUF".into(),
            filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf".into(),
            display_name: "Llama 3.2 3B (Fast, 2GB)".into(),
            size_gb: 2.0,
            description: "Fast general-purpose model, good for most tasks".into(),
        },
        RecommendedModel {
            repo_id: "bartowski/Qwen2.5-7B-Instruct-GGUF".into(),
            filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf".into(),
            display_name: "Qwen 2.5 7B (Balanced, 4.5GB)".into(),
            size_gb: 4.5,
            description: "Excellent reasoning and instruction following".into(),
        },
        RecommendedModel {
            repo_id: "bartowski/Mistral-Nemo-Instruct-2407-GGUF".into(),
            filename: "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf".into(),
            display_name: "Mistral Nemo 12B (Quality, 7GB)".into(),
            size_gb: 7.0,
            description: "High quality outputs, longer context".into(),
        },
    ])
}
```

---

## Phase 4: Frontend Integration

### 4.1 TypeScript Bindings

```typescript
// src/services/inference/tauri-bridge.ts

import { invoke } from '@tauri-apps/api/core';

export interface LLMRequest {
  messages: { role: string; content: string }[];
  model: string;
  max_tokens?: number;
  temperature?: number;
}

export interface LLMResponse {
  content: string;
  tokens_used: number;
  model: string;
  finish_reason: string;
}

export type BackendConfig =
  | { type: 'local'; model_path: string; max_concurrent?: number }
  | { type: 'huggingface'; api_key: string; model_id: string; max_concurrent?: number }
  | { type: 'openrouter'; api_key: string; model_id: string; max_concurrent?: number }
  | { type: 'openai'; api_key: string; model_id: string; max_concurrent?: number };

// Single cell generation
export async function generateCell(request: LLMRequest): Promise<LLMResponse> {
  return invoke('generate_cell', { request });
}

// Batch generation for parallel inference
export async function generateBatch(requests: LLMRequest[]): Promise<LLMResponse[]> {
  return invoke('generate_batch', { requests });
}

// Configure the LLM backend
export async function configureBackend(config: BackendConfig): Promise<void> {
  return invoke('configure_backend', { backendConfig: config });
}

// Model management
export async function listLocalModels(): Promise<LocalModel[]> {
  return invoke('list_local_models');
}

export async function downloadModel(repoId: string, filename: string): Promise<string> {
  return invoke('download_model', { repoId, filename });
}

export async function getRecommendedModels(): Promise<RecommendedModel[]> {
  return invoke('get_recommended_models');
}
```

### 4.2 Modify Existing Inference Service

```typescript
// src/services/inference/index.ts (MODIFIED)

import { isTauri } from '@tauri-apps/api/core';
import * as tauriBridge from './tauri-bridge';
import * as webBridge from './web-bridge'; // Existing HF inference

export async function generateCells(
  requests: GenerationRequest[],
  onProgress?: (completed: number, total: number) => void
): Promise<GenerationResult[]> {
  if (isTauri()) {
    // Desktop app: use Rust backend
    const llmRequests = requests.map(r => ({
      messages: buildMessages(r),
      model: r.model,
      max_tokens: r.maxTokens,
      temperature: r.temperature,
    }));

    const results = await tauriBridge.generateBatch(llmRequests);
    return results.map((r, i) => ({
      rowId: requests[i].rowId,
      content: r.content,
      success: true,
    }));
  } else {
    // Web: use existing HF inference
    return webBridge.generateCells(requests, onProgress);
  }
}
```

---

## Phase 5: Settings & Configuration UI

### 5.1 Settings Panel Component

```typescript
// src/features/settings/settings-panel.tsx

import { component$, useSignal, useTask$ } from '@builder.io/qwik';
import { configureBackend, listLocalModels, getRecommendedModels } from '~/services/inference/tauri-bridge';

export const SettingsPanel = component$(() => {
  const backendType = useSignal<'local' | 'huggingface' | 'openrouter' | 'openai'>('local');
  const localModels = useSignal<LocalModel[]>([]);
  const selectedModel = useSignal<string>('');
  const apiKey = useSignal<string>('');
  const maxConcurrent = useSignal<number>(5);

  useTask$(async () => {
    const models = await listLocalModels();
    localModels.value = models;
    if (models.length > 0) {
      selectedModel.value = models[0].path;
    }
  });

  return (
    <div class="settings-panel p-6 bg-neutral-900 rounded-lg">
      <h2 class="text-xl font-bold mb-4">Inference Settings</h2>

      {/* Backend Selection */}
      <div class="mb-6">
        <label class="block text-sm font-medium mb-2">Backend</label>
        <div class="flex gap-2">
          <button
            class={`px-4 py-2 rounded ${backendType.value === 'local' ? 'bg-primary-600' : 'bg-neutral-700'}`}
            onClick$={() => backendType.value = 'local'}
          >
            Local (llama.cpp)
          </button>
          <button
            class={`px-4 py-2 rounded ${backendType.value === 'huggingface' ? 'bg-primary-600' : 'bg-neutral-700'}`}
            onClick$={() => backendType.value = 'huggingface'}
          >
            HuggingFace Pro
          </button>
          <button
            class={`px-4 py-2 rounded ${backendType.value === 'openrouter' ? 'bg-primary-600' : 'bg-neutral-700'}`}
            onClick$={() => backendType.value = 'openrouter'}
          >
            OpenRouter
          </button>
          <button
            class={`px-4 py-2 rounded ${backendType.value === 'openai' ? 'bg-primary-600' : 'bg-neutral-700'}`}
            onClick$={() => backendType.value = 'openai'}
          >
            OpenAI
          </button>
        </div>
      </div>

      {/* Local Model Selection */}
      {backendType.value === 'local' && (
        <div class="mb-6">
          <label class="block text-sm font-medium mb-2">Model</label>
          <select
            class="w-full p-2 bg-neutral-800 rounded"
            value={selectedModel.value}
            onChange$={(e) => selectedModel.value = (e.target as HTMLSelectElement).value}
          >
            {localModels.value.map(model => (
              <option key={model.path} value={model.path}>
                {model.name} ({(model.size_bytes / 1e9).toFixed(1)}GB)
              </option>
            ))}
          </select>
          <button class="mt-2 text-sm text-primary-400 hover:underline">
            + Download new model
          </button>
        </div>
      )}

      {/* API Key for cloud backends */}
      {backendType.value !== 'local' && (
        <div class="mb-6">
          <label class="block text-sm font-medium mb-2">API Key</label>
          <input
            type="password"
            class="w-full p-2 bg-neutral-800 rounded"
            value={apiKey.value}
            onInput$={(e) => apiKey.value = (e.target as HTMLInputElement).value}
            placeholder={`Enter your ${backendType.value} API key`}
          />
        </div>
      )}

      {/* Concurrency */}
      <div class="mb-6">
        <label class="block text-sm font-medium mb-2">
          Parallel Requests: {maxConcurrent.value}
        </label>
        <input
          type="range"
          min="1"
          max="10"
          value={maxConcurrent.value}
          onInput$={(e) => maxConcurrent.value = parseInt((e.target as HTMLInputElement).value)}
          class="w-full"
        />
        <p class="text-xs text-neutral-400 mt-1">
          Higher values = faster batch processing, more resource usage
        </p>
      </div>

      {/* Save Button */}
      <button
        class="w-full py-3 bg-primary-600 hover:bg-primary-700 rounded font-medium"
        onClick$={async () => {
          await configureBackend({
            type: backendType.value,
            ...(backendType.value === 'local'
              ? { model_path: selectedModel.value }
              : { api_key: apiKey.value, model_id: 'default' }),
            max_concurrent: maxConcurrent.value,
          });
        }}
      >
        Apply Settings
      </button>
    </div>
  );
});
```

---

## Phase 6: Build & Distribution

### 6.1 GitHub Actions CI/CD

```yaml
# .github/workflows/release.yml

name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    strategy:
      fail-fast: false
      matrix:
        include:
          - platform: macos-latest
            target: aarch64-apple-darwin
            name: Braincells-macOS-arm64
          - platform: macos-latest
            target: x86_64-apple-darwin
            name: Braincells-macOS-x64
          - platform: windows-latest
            target: x86_64-pc-windows-msvc
            name: Braincells-Windows-x64
          - platform: ubuntu-22.04
            target: x86_64-unknown-linux-gnu
            name: Braincells-Linux-x64

    runs-on: ${{ matrix.platform }}

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: 20

      - name: Setup pnpm
        uses: pnpm/action-setup@v3
        with:
          version: 9

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}

      - name: Install dependencies (Ubuntu)
        if: matrix.platform == 'ubuntu-22.04'
        run: |
          sudo apt-get update
          sudo apt-get install -y libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev

      - name: Install CUDA toolkit (Windows)
        if: matrix.platform == 'windows-latest'
        uses: Jimver/cuda-toolkit@v0.2.11
        with:
          cuda: '12.2.0'

      - name: Install frontend dependencies
        run: pnpm install --frozen-lockfile

      - name: Build Tauri app
        uses: tauri-apps/tauri-action@v0
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tagName: v__VERSION__
          releaseName: 'Braincells v__VERSION__'
          releaseBody: 'See the assets below to download for your platform.'
          releaseDraft: true
          prerelease: false
          args: --target ${{ matrix.target }}
```

### 6.2 Platform-Specific Build Scripts

```bash
#!/bin/bash
# scripts/build-macos.sh

# Build with Metal support
export LLAMA_METAL=1
cargo build --release --target aarch64-apple-darwin

# Create universal binary (optional)
# lipo -create -output target/universal/braincells \
#   target/aarch64-apple-darwin/release/braincells \
#   target/x86_64-apple-darwin/release/braincells
```

```powershell
# scripts/build-windows.ps1

# Build with CUDA support
$env:LLAMA_CUDA = "1"
cargo build --release --target x86_64-pc-windows-msvc
```

---

## Phase 7: Optional MLX Sidecar (macOS Optimization)

For maximum performance on Apple Silicon, bundle an MLX-based Swift binary:

```
src-tauri/resources/
└── mlx-inference/
    ├── braincells-mlx          # Compiled Swift binary
    └── models/                  # MLX-format models
```

```rust
// src-tauri/src/llm/mlx.rs

#[cfg(target_os = "macos")]
pub struct MLXEngine {
    sidecar_path: PathBuf,
}

#[cfg(target_os = "macos")]
impl MLXEngine {
    pub async fn generate(&self, request: LLMRequest) -> Result<LLMResponse, LLMError> {
        use tokio::process::Command;

        let output = Command::new(&self.sidecar_path)
            .args(["--model", &request.model])
            .args(["--prompt", &self.format_prompt(&request.messages)])
            .args(["--max-tokens", &request.max_tokens.unwrap_or(2048).to_string()])
            .output()
            .await
            .map_err(|e| LLMError::InferenceFailed(e.to_string()))?;

        let content = String::from_utf8(output.stdout)
            .map_err(|e| LLMError::InferenceFailed(e.to_string()))?;

        Ok(LLMResponse {
            content,
            tokens_used: 0, // MLX doesn't report this easily
            model: request.model,
            finish_reason: "stop".into(),
        })
    }
}
```

---

## Implementation Milestones

### Milestone 1: Core Tauri Shell (Week 1-2)
- [ ] Initialize Tauri 2.0 project structure
- [ ] Configure build for macOS, Windows, Linux
- [ ] Embed existing Qwik frontend
- [ ] Basic window management and settings persistence

### Milestone 2: Local LLM Integration (Week 3-4)
- [ ] Integrate llama-cpp-2 crate
- [ ] Implement model loading/unloading
- [ ] Metal acceleration (macOS)
- [ ] CUDA acceleration (Windows/Linux)
- [ ] Basic generation command

### Milestone 3: Cloud Backends (Week 5)
- [ ] HuggingFace Inference API integration
- [ ] OpenRouter integration
- [ ] OpenAI integration
- [ ] Unified interface for all backends

### Milestone 4: Parallel Inference (Week 6)
- [ ] Implement inference pool with semaphore
- [ ] Batch generation for spreadsheet cells
- [ ] Progress reporting to frontend
- [ ] Error handling and retry logic

### Milestone 5: Model Management (Week 7)
- [ ] HuggingFace Hub model browser
- [ ] Model download with progress
- [ ] Local model cache management
- [ ] Recommended models list

### Milestone 6: UI Polish (Week 8)
- [ ] Settings panel for backend selection
- [ ] Model picker component
- [ ] Download manager UI
- [ ] Error states and loading indicators

### Milestone 7: Testing & Release (Week 9-10)
- [ ] Cross-platform testing
- [ ] Performance benchmarking
- [ ] CI/CD pipeline for releases
- [ ] Documentation and onboarding flow

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| App startup | < 2s | Cold start to usable |
| Model load (7B) | < 10s | First load, cached after |
| Inference (7B Q4) | 30+ tok/s | Apple M1/M2/M3 |
| Inference (7B Q4) | 50+ tok/s | RTX 3080+ |
| Parallel cells | 5 concurrent | Default, configurable to 10 |
| Memory (idle) | < 200MB | Without model loaded |
| Bundle size | < 50MB | Without bundled models |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| llama.cpp build complexity | Use pre-built binaries from llama-cpp-2 crate |
| CUDA dependency on Windows | Ship CPU fallback, CUDA optional |
| Large model downloads | Stream downloads, show progress, allow resume |
| Cross-platform UI differences | Test on all platforms in CI, use system webview |
| Memory pressure with large models | Implement model unloading, memory monitoring |

---

## Open Questions

1. **Branding**: Final name "Braincells" or something else?
2. **Default model**: Ship a small bundled model (500MB) or prompt download on first run?
3. **Pricing model**: Free/open-source only, or freemium with cloud features?
4. **Update mechanism**: Auto-update via Tauri updater, or manual downloads?
