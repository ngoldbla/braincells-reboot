//! Model registry with recommended models for download
//!
//! This module provides a curated list of recommended models
//! that work well with Braincells.

use crate::llm::types::RecommendedModel;
use once_cell::sync::Lazy;

/// Curated list of recommended models
pub static RECOMMENDED_MODELS: Lazy<Vec<RecommendedModel>> = Lazy::new(|| {
    vec![
        // Small models (< 3GB) - Fast, good for quick tasks
        RecommendedModel {
            repo_id: "bartowski/Llama-3.2-1B-Instruct-GGUF".to_string(),
            filename: "Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string(),
            display_name: "Llama 3.2 1B (Tiny, 0.8GB)".to_string(),
            size_gb: 0.8,
            description: "Ultra-fast for simple tasks. Good starting point.".to_string(),
            quantization: "Q4_K_M".to_string(),
        },
        RecommendedModel {
            repo_id: "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
            filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string(),
            display_name: "Llama 3.2 3B (Fast, 2GB)".to_string(),
            size_gb: 2.0,
            description: "Great balance of speed and quality for most tasks.".to_string(),
            quantization: "Q4_K_M".to_string(),
        },
        RecommendedModel {
            repo_id: "bartowski/Phi-3.5-mini-instruct-GGUF".to_string(),
            filename: "Phi-3.5-mini-instruct-Q4_K_M.gguf".to_string(),
            display_name: "Phi 3.5 Mini (Fast, 2.3GB)".to_string(),
            size_gb: 2.3,
            description: "Microsoft's efficient model. Excellent for reasoning.".to_string(),
            quantization: "Q4_K_M".to_string(),
        },

        // Medium models (3-6GB) - Balanced performance
        RecommendedModel {
            repo_id: "bartowski/Qwen2.5-7B-Instruct-GGUF".to_string(),
            filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf".to_string(),
            display_name: "Qwen 2.5 7B (Balanced, 4.5GB)".to_string(),
            size_gb: 4.5,
            description: "Excellent instruction following and reasoning.".to_string(),
            quantization: "Q4_K_M".to_string(),
        },
        RecommendedModel {
            repo_id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF".to_string(),
            filename: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf".to_string(),
            display_name: "Llama 3.1 8B (Balanced, 4.9GB)".to_string(),
            size_gb: 4.9,
            description: "Meta's flagship 8B model. Great all-rounder.".to_string(),
            quantization: "Q4_K_M".to_string(),
        },
        RecommendedModel {
            repo_id: "bartowski/gemma-2-9b-it-GGUF".to_string(),
            filename: "gemma-2-9b-it-Q4_K_M.gguf".to_string(),
            display_name: "Gemma 2 9B (Balanced, 5.8GB)".to_string(),
            size_gb: 5.8,
            description: "Google's efficient model. Strong on diverse tasks.".to_string(),
            quantization: "Q4_K_M".to_string(),
        },

        // Large models (6-12GB) - High quality
        RecommendedModel {
            repo_id: "bartowski/Mistral-Nemo-Instruct-2407-GGUF".to_string(),
            filename: "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf".to_string(),
            display_name: "Mistral Nemo 12B (Quality, 7.4GB)".to_string(),
            size_gb: 7.4,
            description: "High quality outputs with 128k context window.".to_string(),
            quantization: "Q4_K_M".to_string(),
        },
        RecommendedModel {
            repo_id: "bartowski/Qwen2.5-14B-Instruct-GGUF".to_string(),
            filename: "Qwen2.5-14B-Instruct-Q4_K_M.gguf".to_string(),
            display_name: "Qwen 2.5 14B (Quality, 8.9GB)".to_string(),
            size_gb: 8.9,
            description: "Excellent for complex reasoning and analysis.".to_string(),
            quantization: "Q4_K_M".to_string(),
        },

        // Extra large models (12GB+) - Best quality
        RecommendedModel {
            repo_id: "bartowski/Qwen2.5-32B-Instruct-GGUF".to_string(),
            filename: "Qwen2.5-32B-Instruct-Q4_K_M.gguf".to_string(),
            display_name: "Qwen 2.5 32B (Premium, 19GB)".to_string(),
            size_gb: 19.0,
            description: "Top-tier quality. Requires 24GB+ RAM.".to_string(),
            quantization: "Q4_K_M".to_string(),
        },
        RecommendedModel {
            repo_id: "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF".to_string(),
            filename: "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf".to_string(),
            display_name: "Llama 3.1 70B (Premium, 40GB)".to_string(),
            size_gb: 40.0,
            description: "Near frontier quality. Requires 48GB+ RAM.".to_string(),
            quantization: "Q4_K_M".to_string(),
        },
    ]
});

/// Get all recommended models
pub fn get_recommended_models() -> Vec<RecommendedModel> {
    RECOMMENDED_MODELS.clone()
}

/// Get recommended models filtered by max size
pub fn get_recommended_models_by_size(max_size_gb: f64) -> Vec<RecommendedModel> {
    RECOMMENDED_MODELS
        .iter()
        .filter(|m| m.size_gb <= max_size_gb)
        .cloned()
        .collect()
}

/// Get a specific recommended model by repo_id
pub fn get_model_by_repo(repo_id: &str) -> Option<RecommendedModel> {
    RECOMMENDED_MODELS
        .iter()
        .find(|m| m.repo_id == repo_id)
        .cloned()
}

/// Estimate available memory for model selection
pub fn estimate_available_memory() -> u64 {
    #[cfg(target_os = "macos")]
    {
        // On macOS, use sysctl to get memory info
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if let Ok(mem_str) = String::from_utf8(output.stdout) {
                if let Ok(mem) = mem_str.trim().parse::<u64>() {
                    return mem;
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Default fallback for Windows
        // In a real implementation, use Windows API
    }

    // Default: assume 8GB
    8 * 1024 * 1024 * 1024
}

/// Suggest a model based on available memory
pub fn suggest_model_for_memory(available_bytes: u64) -> RecommendedModel {
    let available_gb = available_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    // Leave some headroom for the system
    let usable_gb = available_gb * 0.7;

    // Find the largest model that fits
    let suitable: Vec<_> = RECOMMENDED_MODELS
        .iter()
        .filter(|m| m.size_gb <= usable_gb)
        .collect();

    suitable
        .last()
        .cloned()
        .cloned()
        .unwrap_or_else(|| RECOMMENDED_MODELS[0].clone())
}
