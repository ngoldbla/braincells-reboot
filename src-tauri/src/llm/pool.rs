//! Parallel inference pool for concurrent LLM requests
//!
//! This module provides a pool for managing concurrent inference requests,
//! essential for processing multiple spreadsheet cells in parallel.

use super::types::{LLMEngine, LLMError, LLMRequest, LLMResponse};
use futures::stream::{self, StreamExt};
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Pool for managing concurrent inference requests
pub struct InferencePool {
    engine: Arc<dyn LLMEngine>,
    semaphore: Arc<Semaphore>,
    max_concurrent: usize,
}

impl InferencePool {
    /// Create a new inference pool with the given engine and concurrency limit
    pub fn new(engine: Arc<dyn LLMEngine>, max_concurrent: usize) -> Self {
        let max_concurrent = max_concurrent.clamp(1, 10); // Clamp between 1 and 10
        Self {
            engine,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        }
    }

    /// Get the maximum concurrent requests
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Get the current number of available permits
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    /// Execute a single inference request (rate-limited)
    pub async fn generate_single(&self, request: LLMRequest) -> Result<LLMResponse, LLMError> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|e| LLMError::InferenceFailed(format!("Semaphore error: {}", e)))?;

        self.engine.generate(request).await
    }

    /// Execute multiple inference requests in parallel
    ///
    /// Results are returned in the same order as the input requests.
    pub async fn generate_batch(
        &self,
        requests: Vec<LLMRequest>,
    ) -> Vec<Result<LLMResponse, LLMError>> {
        if requests.is_empty() {
            return Vec::new();
        }

        // Process requests concurrently with semaphore limiting
        let results: Vec<_> = stream::iter(requests.into_iter().enumerate())
            .map(|(idx, request)| {
                let engine = self.engine.clone();
                let semaphore = self.semaphore.clone();

                async move {
                    let _permit = semaphore.acquire().await;
                    let result = engine.generate(request).await;
                    (idx, result)
                }
            })
            .buffer_unordered(self.max_concurrent)
            .collect()
            .await;

        // Sort by original index to maintain order
        let mut sorted: Vec<_> = results.into_iter().collect();
        sorted.sort_by_key(|(idx, _)| *idx);
        sorted.into_iter().map(|(_, result)| result).collect()
    }

    /// Execute multiple inference requests with progress callback
    pub async fn generate_batch_with_progress<F>(
        &self,
        requests: Vec<LLMRequest>,
        mut on_progress: F,
    ) -> Vec<Result<LLMResponse, LLMError>>
    where
        F: FnMut(usize, usize) + Send,
    {
        if requests.is_empty() {
            return Vec::new();
        }

        let total = requests.len();
        let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let results: Vec<_> = stream::iter(requests.into_iter().enumerate())
            .map(|(idx, request)| {
                let engine = self.engine.clone();
                let semaphore = self.semaphore.clone();
                let completed = completed.clone();

                async move {
                    let _permit = semaphore.acquire().await;
                    let result = engine.generate(request).await;
                    completed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    (idx, result)
                }
            })
            .buffer_unordered(self.max_concurrent)
            .inspect(|_| {
                let current = completed.load(std::sync::atomic::Ordering::SeqCst);
                on_progress(current, total);
            })
            .collect()
            .await;

        // Sort by original index to maintain order
        let mut sorted: Vec<_> = results.into_iter().collect();
        sorted.sort_by_key(|(idx, _)| *idx);
        sorted.into_iter().map(|(_, result)| result).collect()
    }

    /// Check if the underlying engine is ready
    pub async fn is_ready(&self) -> bool {
        self.engine.is_ready().await
    }

    /// Get the backend name
    pub fn backend_name(&self) -> &'static str {
        self.engine.backend_name()
    }

    /// Unload the underlying engine
    pub async fn unload(&self) -> Result<(), LLMError> {
        self.engine.unload().await
    }
}

/// Builder for InferencePool
pub struct InferencePoolBuilder {
    engine: Option<Arc<dyn LLMEngine>>,
    max_concurrent: usize,
}

impl Default for InferencePoolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl InferencePoolBuilder {
    pub fn new() -> Self {
        Self {
            engine: None,
            max_concurrent: 5,
        }
    }

    pub fn engine(mut self, engine: Arc<dyn LLMEngine>) -> Self {
        self.engine = Some(engine);
        self
    }

    pub fn max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    pub fn build(self) -> Result<InferencePool, LLMError> {
        let engine = self
            .engine
            .ok_or_else(|| LLMError::InvalidConfig("No engine provided".to_string()))?;

        Ok(InferencePool::new(engine, self.max_concurrent))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::types::Message;

    // Mock engine for testing
    struct MockEngine;

    #[async_trait::async_trait]
    impl LLMEngine for MockEngine {
        async fn generate(&self, request: LLMRequest) -> Result<LLMResponse, LLMError> {
            // Simulate some processing time
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            Ok(LLMResponse {
                content: format!("Response to: {}", request.messages.last().map(|m| m.content.as_str()).unwrap_or("")),
                tokens_used: 10,
                model: request.model,
                finish_reason: "stop".to_string(),
            })
        }

        async fn is_ready(&self) -> bool {
            true
        }

        fn backend_name(&self) -> &'static str {
            "mock"
        }
    }

    #[tokio::test]
    async fn test_pool_creation() {
        let engine = Arc::new(MockEngine);
        let pool = InferencePool::new(engine, 5);
        assert_eq!(pool.max_concurrent(), 5);
        assert!(pool.is_ready().await);
    }

    #[tokio::test]
    async fn test_single_request() {
        let engine = Arc::new(MockEngine);
        let pool = InferencePool::new(engine, 5);

        let request = LLMRequest {
            messages: vec![Message::user("Hello")],
            model: "test".to_string(),
            ..Default::default()
        };

        let result = pool.generate_single(request).await;
        assert!(result.is_ok());
        assert!(result.unwrap().content.contains("Hello"));
    }

    #[tokio::test]
    async fn test_batch_requests() {
        let engine = Arc::new(MockEngine);
        let pool = InferencePool::new(engine, 3);

        let requests: Vec<LLMRequest> = (0..5)
            .map(|i| LLMRequest {
                messages: vec![Message::user(format!("Message {}", i))],
                model: "test".to_string(),
                ..Default::default()
            })
            .collect();

        let results = pool.generate_batch(requests).await;
        assert_eq!(results.len(), 5);

        // Check order is preserved
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok());
            assert!(result.as_ref().unwrap().content.contains(&format!("Message {}", i)));
        }
    }
}
