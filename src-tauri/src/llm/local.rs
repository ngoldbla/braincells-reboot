//! Local LLM inference using llama.cpp
//!
//! This module provides a local inference backend using the llama.cpp library
//! through Rust bindings. It supports Metal acceleration on macOS and CUDA on
//! Windows/Linux.

use super::types::{LLMEngine, LLMError, LLMRequest, LLMResponse, Message};
use async_trait::async_trait;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, RwLock};

/// Request sent to the inference worker
struct InferenceRequest {
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    response_tx: oneshot::Sender<Result<(String, u32), LLMError>>,
}

/// Local LLM engine using llama.cpp
pub struct LocalEngine {
    request_tx: Arc<RwLock<Option<mpsc::Sender<InferenceRequest>>>>,
    model_path: PathBuf,
    gpu_layers: u32,
    context_size: u32,
    is_loaded: Arc<RwLock<bool>>,
}

impl LocalEngine {
    /// Create a new local engine with the specified model path
    pub fn new(model_path: impl Into<PathBuf>, gpu_layers: u32, context_size: u32) -> Self {
        Self {
            request_tx: Arc::new(RwLock::new(None)),
            model_path: model_path.into(),
            gpu_layers,
            context_size,
            is_loaded: Arc::new(RwLock::new(false)),
        }
    }

    /// Load the model into memory and start the inference worker
    pub async fn load_model(&self) -> Result<(), LLMError> {
        let model_path = self.model_path.clone();
        let gpu_layers = self.gpu_layers;
        let context_size = self.context_size;

        // Channel for sending inference requests
        let (request_tx, mut request_rx) = mpsc::channel::<InferenceRequest>(32);

        // Spawn blocking thread for model loading and inference
        std::thread::spawn(move || {
            // Initialize the backend
            let backend = match LlamaBackend::init() {
                Ok(b) => b,
                Err(e) => {
                    log::error!("Failed to init backend: {}", e);
                    return;
                }
            };

            // Set up model parameters
            let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);

            // Load the model
            let model = match LlamaModel::load_from_file(&backend, &model_path, &model_params) {
                Ok(m) => m,
                Err(e) => {
                    log::error!("Failed to load model: {}", e);
                    return;
                }
            };

            log::info!("Model loaded successfully: {}", model_path.display());

            // Process inference requests
            while let Some(request) = request_rx.blocking_recv() {
                let result = run_inference(&backend, &model, &request.prompt, request.max_tokens, request.temperature, context_size);
                let _ = request.response_tx.send(result);
            }

            log::info!("Inference worker shutting down");
        });

        // Store the request channel
        let mut tx_guard = self.request_tx.write().await;
        *tx_guard = Some(request_tx);

        let mut loaded_guard = self.is_loaded.write().await;
        *loaded_guard = true;

        log::info!("Model ready: {}", self.model_path.display());
        Ok(())
    }

    /// Unload the model from memory
    pub async fn unload_model(&self) {
        let mut tx_guard = self.request_tx.write().await;
        *tx_guard = None; // Dropping the sender will cause the worker to exit

        let mut loaded_guard = self.is_loaded.write().await;
        *loaded_guard = false;

        log::info!("Model unloaded: {}", self.model_path.display());
    }

    /// Format messages into a prompt string using ChatML format
    fn format_prompt(messages: &[Message]) -> String {
        let mut prompt = String::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    prompt.push_str("<|im_start|>system\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|im_end|>\n");
                }
                "user" => {
                    prompt.push_str("<|im_start|>user\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|im_end|>\n");
                }
                "assistant" => {
                    prompt.push_str("<|im_start|>assistant\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|im_end|>\n");
                }
                _ => {
                    prompt.push_str(&msg.content);
                    prompt.push('\n');
                }
            }
        }

        // Add the assistant prefix for the response
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }
}

/// Run inference synchronously (called from the worker thread)
fn run_inference(
    backend: &LlamaBackend,
    model: &LlamaModel,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    context_size: u32,
) -> Result<(String, u32), LLMError> {
    // Create context
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(context_size));

    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| LLMError::InferenceFailed(format!("Failed to create context: {}", e)))?;

    // Tokenize the prompt
    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .map_err(|e| LLMError::InferenceFailed(format!("Tokenization failed: {}", e)))?;

    // Check context size
    if tokens.len() >= context_size as usize {
        return Err(LLMError::ContextLengthExceeded);
    }

    // Create batch and add tokens
    let mut batch = LlamaBatch::new(context_size as usize, 1);

    for (i, token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        batch
            .add(*token, i as i32, &[0], is_last)
            .map_err(|e| LLMError::InferenceFailed(format!("Failed to add token: {}", e)))?;
    }

    // Process the batch (prefill)
    ctx.decode(&mut batch)
        .map_err(|e| LLMError::InferenceFailed(format!("Decode failed: {}", e)))?;

    // Set up sampler
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(temperature),
        LlamaSampler::dist(0), // seed
    ]);

    // Generate tokens
    let mut generated = String::new();
    let mut n_cur = tokens.len();
    let mut tokens_generated = 0u32;

    while n_cur < max_tokens + tokens.len() {
        // Sample next token using the last position in the batch
        let idx = (batch.n_tokens() - 1) as i32;
        let token = sampler.sample(&ctx, idx);

        // Check for EOS
        if model.is_eog_token(token) {
            break;
        }

        // Decode token to string
        let token_str = model
            .token_to_str(token, Special::Tokenize)
            .map_err(|e| LLMError::InferenceFailed(format!("Token decode failed: {}", e)))?;

        // Check for end markers
        if token_str.contains("<|im_end|>") || token_str.contains("<|endoftext|>") {
            break;
        }

        generated.push_str(&token_str);
        tokens_generated += 1;

        // Prepare next batch
        batch.clear();
        batch
            .add(token, n_cur as i32, &[0], true)
            .map_err(|e| LLMError::InferenceFailed(format!("Failed to add token: {}", e)))?;

        ctx.decode(&mut batch)
            .map_err(|e| LLMError::InferenceFailed(format!("Decode failed: {}", e)))?;

        n_cur += 1;
    }

    Ok((generated.trim().to_string(), tokens_generated))
}

#[async_trait]
impl LLMEngine for LocalEngine {
    async fn generate(&self, request: LLMRequest) -> Result<LLMResponse, LLMError> {
        let tx_guard = self.request_tx.read().await;
        let request_tx = tx_guard
            .as_ref()
            .ok_or_else(|| LLMError::ModelNotLoaded)?;

        let prompt = Self::format_prompt(&request.messages);
        let max_tokens = request.max_tokens.unwrap_or(2048) as usize;
        let temperature = request.temperature.unwrap_or(0.7);
        let model_name = request.model.clone();

        // Create response channel
        let (response_tx, response_rx) = oneshot::channel();

        // Send inference request
        request_tx
            .send(InferenceRequest {
                prompt,
                max_tokens,
                temperature,
                response_tx,
            })
            .await
            .map_err(|_| LLMError::InferenceFailed("Failed to send request to worker".to_string()))?;

        // Wait for response
        let result = response_rx
            .await
            .map_err(|_| LLMError::InferenceFailed("Worker closed response channel".to_string()))??;

        Ok(LLMResponse {
            content: result.0,
            tokens_used: result.1,
            model: model_name,
            finish_reason: "stop".to_string(),
        })
    }

    async fn is_ready(&self) -> bool {
        *self.is_loaded.read().await
    }

    fn backend_name(&self) -> &'static str {
        "local-llama-cpp"
    }

    async fn unload(&self) -> Result<(), LLMError> {
        self.unload_model().await;
        Ok(())
    }
}

/// Detect optimal GPU layers based on platform
pub fn detect_optimal_gpu_layers() -> u32 {
    #[cfg(target_os = "macos")]
    {
        // Metal: offload all layers
        999
    }

    #[cfg(target_os = "windows")]
    {
        // Check for NVIDIA GPU
        if has_cuda_support() {
            35 // Conservative default for CUDA
        } else {
            0 // CPU only
        }
    }

    #[cfg(target_os = "linux")]
    {
        if has_cuda_support() {
            35
        } else {
            0
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        0
    }
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn has_cuda_support() -> bool {
    // Simple check for NVIDIA driver
    std::process::Command::new("nvidia-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "macos")]
fn has_cuda_support() -> bool {
    false // macOS uses Metal, not CUDA
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_prompt() {
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::user("Hello!"),
        ];

        let prompt = LocalEngine::format_prompt(&messages);
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("You are a helpful assistant."));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }
}
