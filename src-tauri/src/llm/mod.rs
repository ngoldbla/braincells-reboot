//! LLM inference module
//!
//! This module provides a unified interface for LLM inference,
//! supporting both local (llama.cpp) and cloud backends.

pub mod cloud;
pub mod local;
pub mod pool;
pub mod types;

pub use cloud::{CloudEngine, CloudProvider};
pub use local::{LocalEngine, detect_optimal_gpu_layers};
pub use pool::{InferencePool, InferencePoolBuilder};
pub use types::*;

use std::sync::Arc;

/// Create an engine from a backend configuration
pub fn create_engine(config: &BackendConfig) -> Result<Arc<dyn LLMEngine>, LLMError> {
    match config {
        BackendConfig::Local {
            model_path,
            gpu_layers,
            context_size,
        } => {
            let engine = LocalEngine::new(model_path, *gpu_layers, *context_size);
            Ok(Arc::new(engine))
        }
        BackendConfig::HuggingFace {
            api_key,
            model_id,
            provider,
        } => {
            let engine = CloudEngine::huggingface(api_key, model_id, provider.clone());
            Ok(Arc::new(engine))
        }
        BackendConfig::OpenRouter { api_key, model_id } => {
            let engine = CloudEngine::openrouter(api_key, model_id);
            Ok(Arc::new(engine))
        }
        BackendConfig::OpenAI { api_key, model_id } => {
            let engine = CloudEngine::openai(api_key, model_id);
            Ok(Arc::new(engine))
        }
    }
}

/// Create an inference pool from a backend configuration
pub async fn create_pool(
    config: &BackendConfig,
    max_concurrent: usize,
) -> Result<InferencePool, LLMError> {
    let engine = create_engine(config)?;

    // Load local models
    if let BackendConfig::Local { .. } = config {
        if let Some(local_engine) = engine.as_ref().as_any().downcast_ref::<LocalEngine>() {
            local_engine.load_model().await?;
        }
    }

    Ok(InferencePool::new(engine, max_concurrent))
}

/// Extension trait to allow downcasting
pub trait AsAny {
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: LLMEngine + 'static> AsAny for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Extend LLMEngine to include AsAny
impl dyn LLMEngine {
    pub fn as_any(&self) -> &dyn std::any::Any {
        // This is a workaround - in practice we'd use a different pattern
        // For now, we'll handle this in the specific implementations
        unsafe { std::mem::transmute::<&dyn LLMEngine, &dyn std::any::Any>(self) }
    }
}
