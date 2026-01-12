//! Cloud LLM backends (HuggingFace, OpenRouter, OpenAI)
//!
//! This module provides cloud-based inference backends for users without
//! local compute resources or when using proprietary models.

use super::types::{LLMEngine, LLMError, LLMRequest, LLMResponse};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Cloud provider type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloudProvider {
    HuggingFace,
    OpenRouter,
    OpenAI,
}

impl CloudProvider {
    fn base_url(&self) -> &'static str {
        match self {
            CloudProvider::HuggingFace => "https://api-inference.huggingface.co/v1",
            CloudProvider::OpenRouter => "https://openrouter.ai/api/v1",
            CloudProvider::OpenAI => "https://api.openai.com/v1",
        }
    }

    fn name(&self) -> &'static str {
        match self {
            CloudProvider::HuggingFace => "huggingface",
            CloudProvider::OpenRouter => "openrouter",
            CloudProvider::OpenAI => "openai",
        }
    }
}

/// Cloud LLM engine supporting multiple providers
pub struct CloudEngine {
    client: Client,
    provider: CloudProvider,
    api_key: String,
    model_id: String,
    hf_provider: Option<String>, // For HuggingFace inference providers
}

impl CloudEngine {
    /// Create a new cloud engine
    pub fn new(
        provider: CloudProvider,
        api_key: impl Into<String>,
        model_id: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            provider,
            api_key: api_key.into(),
            model_id: model_id.into(),
            hf_provider: None,
        }
    }

    /// Create a HuggingFace engine with a specific inference provider
    pub fn huggingface(
        api_key: impl Into<String>,
        model_id: impl Into<String>,
        provider: Option<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            provider: CloudProvider::HuggingFace,
            api_key: api_key.into(),
            model_id: model_id.into(),
            hf_provider: provider,
        }
    }

    /// Create an OpenRouter engine
    pub fn openrouter(api_key: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self::new(CloudProvider::OpenRouter, api_key, model_id)
    }

    /// Create an OpenAI engine
    pub fn openai(api_key: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self::new(CloudProvider::OpenAI, api_key, model_id)
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    total_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ApiErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
}

#[async_trait]
impl LLMEngine for CloudEngine {
    async fn generate(&self, request: LLMRequest) -> Result<LLMResponse, LLMError> {
        let url = format!("{}/chat/completions", self.provider.base_url());

        let messages: Vec<ChatMessage> = request
            .messages
            .iter()
            .map(|m| ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();

        let body = ChatCompletionRequest {
            model: self.model_id.clone(),
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stop: request.stop_sequences,
            stream: false,
        };

        let mut req = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json");

        // Add provider-specific headers
        match self.provider {
            CloudProvider::HuggingFace => {
                if let Some(ref provider) = self.hf_provider {
                    req = req.header("X-Use-Inference-Provider", provider);
                }
            }
            CloudProvider::OpenRouter => {
                req = req.header("HTTP-Referer", "https://braincells.app");
                req = req.header("X-Title", "Braincells");
            }
            CloudProvider::OpenAI => {}
        }

        let response = req
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::ApiError(format!("Request failed: {}", e)))?;

        let status = response.status();

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(LLMError::RateLimited);
        }

        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();

            // Try to parse structured error
            if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&error_text) {
                if error_resp
                    .error
                    .error_type
                    .as_deref()
                    .map(|t| t.contains("context_length"))
                    .unwrap_or(false)
                {
                    return Err(LLMError::ContextLengthExceeded);
                }
                return Err(LLMError::ApiError(format!(
                    "API error ({}): {}",
                    status, error_resp.error.message
                )));
            }

            return Err(LLMError::ApiError(format!(
                "API error ({}): {}",
                status, error_text
            )));
        }

        let result: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| LLMError::ApiError(format!("Failed to parse response: {}", e)))?;

        let content = result
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let tokens_used = result
            .usage
            .and_then(|u| u.total_tokens)
            .unwrap_or(0);

        let finish_reason = result
            .choices
            .first()
            .and_then(|c| c.finish_reason.clone())
            .unwrap_or_else(|| "stop".to_string());

        Ok(LLMResponse {
            content,
            tokens_used,
            model: self.model_id.clone(),
            finish_reason,
        })
    }

    async fn is_ready(&self) -> bool {
        !self.api_key.is_empty()
    }

    fn backend_name(&self) -> &'static str {
        match self.provider {
            CloudProvider::HuggingFace => "cloud-huggingface",
            CloudProvider::OpenRouter => "cloud-openrouter",
            CloudProvider::OpenAI => "cloud-openai",
        }
    }
}

/// Get default models for each provider
pub fn get_default_models(provider: CloudProvider) -> Vec<(String, String)> {
    match provider {
        CloudProvider::HuggingFace => vec![
            ("meta-llama/Llama-3.3-70B-Instruct".to_string(), "Llama 3.3 70B".to_string()),
            ("Qwen/Qwen2.5-72B-Instruct".to_string(), "Qwen 2.5 72B".to_string()),
            ("mistralai/Mixtral-8x7B-Instruct-v0.1".to_string(), "Mixtral 8x7B".to_string()),
            ("microsoft/Phi-3-medium-128k-instruct".to_string(), "Phi-3 Medium".to_string()),
        ],
        CloudProvider::OpenRouter => vec![
            ("meta-llama/llama-3.3-70b-instruct".to_string(), "Llama 3.3 70B".to_string()),
            ("anthropic/claude-3.5-sonnet".to_string(), "Claude 3.5 Sonnet".to_string()),
            ("google/gemini-2.0-flash-exp:free".to_string(), "Gemini 2.0 Flash".to_string()),
            ("qwen/qwen-2.5-72b-instruct".to_string(), "Qwen 2.5 72B".to_string()),
        ],
        CloudProvider::OpenAI => vec![
            ("gpt-4o".to_string(), "GPT-4o".to_string()),
            ("gpt-4o-mini".to_string(), "GPT-4o Mini".to_string()),
            ("gpt-4-turbo".to_string(), "GPT-4 Turbo".to_string()),
            ("gpt-3.5-turbo".to_string(), "GPT-3.5 Turbo".to_string()),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_urls() {
        assert!(CloudProvider::HuggingFace
            .base_url()
            .contains("huggingface"));
        assert!(CloudProvider::OpenRouter.base_url().contains("openrouter"));
        assert!(CloudProvider::OpenAI.base_url().contains("openai"));
    }
}
