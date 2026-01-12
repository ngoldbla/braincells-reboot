/**
 * Tauri Desktop Bridge
 *
 * TypeScript bindings for the Braincells Tauri backend.
 * Provides local LLM inference and cloud fallback capabilities.
 */

// Types
export interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface LLMRequest {
  messages: Message[];
  model: string;
  max_tokens?: number;
  temperature?: number;
  stop_sequences?: string[];
  stream?: boolean;
}

export interface LLMResponse {
  content: string;
  tokens_used: number;
  model: string;
  finish_reason: string;
}

export type BackendConfig =
  | {
      type: 'local';
      model_path: string;
      gpu_layers?: number;
      context_size?: number;
    }
  | {
      type: 'huggingface';
      api_key: string;
      model_id: string;
      provider?: string;
    }
  | {
      type: 'openrouter';
      api_key: string;
      model_id: string;
    }
  | {
      type: 'openai';
      api_key: string;
      model_id: string;
    };

export interface LocalModel {
  name: string;
  path: string;
  size_bytes: number;
  format: string;
}

export interface RecommendedModel {
  repo_id: string;
  filename: string;
  display_name: string;
  size_gb: number;
  description: string;
  quantization: string;
}

export interface DownloadProgress {
  model_name: string;
  downloaded_bytes: number;
  total_bytes: number;
  percentage: number;
}

export interface Settings {
  max_concurrent_requests: number;
  active_backend?: BackendConfig;
  hf_api_key?: string;
  openrouter_api_key?: string;
  openai_api_key?: string;
  default_local_model?: string;
  default_temperature: number;
  default_max_tokens: number;
}

export interface ProviderInfo {
  id: string;
  name: string;
  available: boolean;
  configured: boolean;
}

export interface CacheStats {
  total_models: number;
  total_size_bytes: number;
  models: LocalModel[];
}

// Check if we're running in Tauri
export function isTauri(): boolean {
  return typeof window !== 'undefined' && '__TAURI__' in window;
}

// Lazy load Tauri API to avoid import errors in non-Tauri environments
let tauriInvoke:
  | ((cmd: string, args?: Record<string, unknown>) => Promise<unknown>)
  | null = null;
let tauriListen:
  | ((
      event: string,
      handler: (payload: unknown) => void,
    ) => Promise<() => void>)
  | null = null;

async function getTauriInvoke() {
  if (!isTauri()) {
    throw new Error('Not running in Tauri environment');
  }
  if (!tauriInvoke) {
    const { invoke } = await import('@tauri-apps/api/core');
    tauriInvoke = invoke;
  }
  return tauriInvoke;
}

async function getTauriListen() {
  if (!isTauri()) {
    throw new Error('Not running in Tauri environment');
  }
  if (!tauriListen) {
    const { listen } = await import('@tauri-apps/api/event');
    tauriListen = listen;
  }
  return tauriListen;
}

// =====================
// Inference Commands
// =====================

/**
 * Generate a response for a single cell
 */
export async function generateCell(request: LLMRequest): Promise<LLMResponse> {
  const invoke = await getTauriInvoke();
  return invoke('generate_cell', { request }) as Promise<LLMResponse>;
}

/**
 * Generate responses for multiple cells in parallel
 */
export async function generateBatch(
  requests: LLMRequest[],
): Promise<Array<LLMResponse | { error: string }>> {
  const invoke = await getTauriInvoke();
  const results = (await invoke('generate_batch', { requests })) as Array<
    { Ok: LLMResponse } | { Err: string }
  >;
  return results.map((r) => {
    if ('Ok' in r) return r.Ok;
    return { error: r.Err };
  });
}

/**
 * Check if an inference engine is ready
 */
export async function isInferenceReady(): Promise<boolean> {
  const invoke = await getTauriInvoke();
  return invoke('is_inference_ready') as Promise<boolean>;
}

/**
 * Get the current backend name
 */
export async function getBackendName(): Promise<string | null> {
  const invoke = await getTauriInvoke();
  return invoke('get_backend_name') as Promise<string | null>;
}

/**
 * Unload the current model
 */
export async function unloadModel(): Promise<void> {
  const invoke = await getTauriInvoke();
  await invoke('unload_model');
}

// =====================
// Model Commands
// =====================

/**
 * List all locally available models
 */
export async function listModels(): Promise<LocalModel[]> {
  const invoke = await getTauriInvoke();
  return invoke('list_models') as Promise<LocalModel[]>;
}

/**
 * Get recommended models for download
 */
export async function getRecommendedModels(): Promise<RecommendedModel[]> {
  const invoke = await getTauriInvoke();
  return invoke('get_recommended_models') as Promise<RecommendedModel[]>;
}

/**
 * Get recommended models that fit in available memory
 */
export async function getRecommendedModelsForSystem(): Promise<
  RecommendedModel[]
> {
  const invoke = await getTauriInvoke();
  return invoke('get_recommended_models_for_system') as Promise<
    RecommendedModel[]
  >;
}

/**
 * Suggest the best model for the current system
 */
export async function suggestModel(): Promise<RecommendedModel> {
  const invoke = await getTauriInvoke();
  return invoke('suggest_model') as Promise<RecommendedModel>;
}

/**
 * Download a model from HuggingFace Hub
 * Returns the path to the downloaded model
 */
export async function downloadModel(
  repoId: string,
  filename: string,
): Promise<string> {
  const invoke = await getTauriInvoke();
  return invoke('download_model', { repoId, filename }) as Promise<string>;
}

/**
 * Listen for model download progress events
 */
export async function onDownloadProgress(
  callback: (progress: DownloadProgress) => void,
): Promise<() => void> {
  const listen = await getTauriListen();
  return listen('model-download-progress', (event) => {
    callback(event.payload as DownloadProgress);
  });
}

/**
 * Delete a local model
 */
export async function deleteModel(modelPath: string): Promise<void> {
  const invoke = await getTauriInvoke();
  await invoke('delete_model', { modelPath });
}

/**
 * Get the models directory path
 */
export async function getModelsDirectory(): Promise<string> {
  const invoke = await getTauriInvoke();
  return invoke('get_models_directory') as Promise<string>;
}

/**
 * Check if a model is downloaded
 */
export async function isModelDownloaded(
  repoId: string,
  filename: string,
): Promise<boolean> {
  const invoke = await getTauriInvoke();
  return invoke('is_model_downloaded', {
    repoId,
    filename,
  }) as Promise<boolean>;
}

/**
 * Get cache statistics
 */
export async function getCacheStats(): Promise<CacheStats> {
  const invoke = await getTauriInvoke();
  return invoke('get_cache_stats') as Promise<CacheStats>;
}

/**
 * Clean up old models from cache
 */
export async function cleanupCache(maxAgeDays?: number): Promise<string[]> {
  const invoke = await getTauriInvoke();
  return invoke('cleanup_cache', { maxAgeDays }) as Promise<string[]>;
}

// =====================
// Settings Commands
// =====================

/**
 * Get current settings
 */
export async function getSettings(): Promise<Settings> {
  const invoke = await getTauriInvoke();
  return invoke('get_settings') as Promise<Settings>;
}

/**
 * Update settings
 */
export async function updateSettings(settings: Settings): Promise<void> {
  const invoke = await getTauriInvoke();
  await invoke('update_settings', { newSettings: settings });
}

/**
 * Configure the inference backend
 */
export async function configureBackend(
  config: BackendConfig,
  maxConcurrent?: number,
): Promise<void> {
  const invoke = await getTauriInvoke();
  await invoke('configure_backend', { config, maxConcurrent });
}

/**
 * Set the HuggingFace API key
 */
export async function setHfApiKey(apiKey: string): Promise<void> {
  const invoke = await getTauriInvoke();
  await invoke('set_hf_api_key', { apiKey });
}

/**
 * Set the OpenRouter API key
 */
export async function setOpenrouterApiKey(apiKey: string): Promise<void> {
  const invoke = await getTauriInvoke();
  await invoke('set_openrouter_api_key', { apiKey });
}

/**
 * Set the OpenAI API key
 */
export async function setOpenaiApiKey(apiKey: string): Promise<void> {
  const invoke = await getTauriInvoke();
  await invoke('set_openai_api_key', { apiKey });
}

/**
 * Set the default local model
 */
export async function setDefaultModel(modelPath?: string): Promise<void> {
  const invoke = await getTauriInvoke();
  await invoke('set_default_model', { modelPath });
}

/**
 * Set concurrency level
 */
export async function setMaxConcurrent(maxConcurrent: number): Promise<void> {
  const invoke = await getTauriInvoke();
  await invoke('set_max_concurrent', { maxConcurrent });
}

/**
 * Quick configure for local model
 */
export async function quickConfigureLocal(modelPath: string): Promise<void> {
  const invoke = await getTauriInvoke();
  await invoke('quick_configure_local', { modelPath });
}

/**
 * Quick configure for HuggingFace cloud
 */
export async function quickConfigureHuggingface(
  modelId: string,
): Promise<void> {
  const invoke = await getTauriInvoke();
  await invoke('quick_configure_huggingface', { modelId });
}

/**
 * Quick configure for OpenRouter cloud
 */
export async function quickConfigureOpenrouter(modelId: string): Promise<void> {
  const invoke = await getTauriInvoke();
  await invoke('quick_configure_openrouter', { modelId });
}

/**
 * Quick configure for OpenAI cloud
 */
export async function quickConfigureOpenai(modelId: string): Promise<void> {
  const invoke = await getTauriInvoke();
  await invoke('quick_configure_openai', { modelId });
}

/**
 * Get available cloud providers based on configured API keys
 */
export async function getAvailableProviders(): Promise<ProviderInfo[]> {
  const invoke = await getTauriInvoke();
  return invoke('get_available_providers') as Promise<ProviderInfo[]>;
}
