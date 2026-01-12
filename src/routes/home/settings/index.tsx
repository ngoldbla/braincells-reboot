import {
  $,
  component$,
  useSignal,
  useStore,
  useVisibleTask$,
} from '@builder.io/qwik';
import type { DocumentHead } from '@builder.io/qwik-city';
import { cn } from '@qwik-ui/utils';
import {
  LuCheck,
  LuCloud,
  LuCpu,
  LuDownload,
  LuHardDrive,
  LuKey,
  LuLoader2,
  LuSettings,
  LuTrash2,
  LuZap,
} from '@qwikest/icons/lucide';
import { Button, Input, Label } from '~/components';
import { MainSidebarButton } from '~/features/main-sidebar';
import * as tauri from '~/services/tauri';

type BackendType = 'local' | 'huggingface' | 'openrouter' | 'openai';

interface SettingsState {
  isTauri: boolean;
  loading: boolean;
  error: string | null;
  success: string | null;

  // Settings
  activeBackend: BackendType;
  maxConcurrent: number;

  // API Keys
  hfApiKey: string;
  openrouterApiKey: string;
  openaiApiKey: string;

  // Local models
  localModels: tauri.LocalModel[];
  recommendedModels: tauri.RecommendedModel[];
  selectedLocalModel: string;
  downloadingModel: string | null;
  downloadProgress: number;

  // Providers
  providers: tauri.ProviderInfo[];
}

const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${Number.parseFloat((bytes / k ** i).toFixed(2))} ${sizes[i]}`;
};

export default component$(() => {
  const state = useStore<SettingsState>({
    isTauri: false,
    loading: true,
    error: null,
    success: null,
    activeBackend: 'local',
    maxConcurrent: 4,
    hfApiKey: '',
    openrouterApiKey: '',
    openaiApiKey: '',
    localModels: [],
    recommendedModels: [],
    selectedLocalModel: '',
    downloadingModel: null,
    downloadProgress: 0,
    providers: [],
  });

  const savingKey = useSignal<string | null>(null);

  // Initialize settings when component mounts
  useVisibleTask$(async () => {
    state.isTauri = tauri.isTauri();

    if (!state.isTauri) {
      state.loading = false;
      return;
    }

    try {
      // Load settings and models in parallel
      const [settings, localModels, recommendedModels, providers] =
        await Promise.all([
          tauri.getSettings(),
          tauri.listModels(),
          tauri.getRecommendedModelsForSystem(),
          tauri.getAvailableProviders(),
        ]);

      state.maxConcurrent = settings.max_concurrent_requests;
      state.hfApiKey = settings.hf_api_key || '';
      state.openrouterApiKey = settings.openrouter_api_key || '';
      state.openaiApiKey = settings.openai_api_key || '';
      state.selectedLocalModel = settings.default_local_model || '';
      state.localModels = localModels;
      state.recommendedModels = recommendedModels;
      state.providers = providers;

      // Determine active backend type
      if (settings.active_backend) {
        state.activeBackend = settings.active_backend.type as BackendType;
      }
    } catch (err) {
      state.error = `Failed to load settings: ${err}`;
    } finally {
      state.loading = false;
    }
  });

  const selectBackend = $(async (backend: BackendType) => {
    if (!state.isTauri) return;

    state.error = null;
    state.success = null;

    try {
      if (backend === 'local') {
        if (state.selectedLocalModel) {
          await tauri.quickConfigureLocal(state.selectedLocalModel);
        } else if (state.localModels.length > 0) {
          await tauri.quickConfigureLocal(state.localModels[0].path);
          state.selectedLocalModel = state.localModels[0].path;
        } else {
          state.error =
            'No local models available. Please download a model first.';
          return;
        }
      } else if (backend === 'huggingface') {
        if (!state.hfApiKey) {
          state.error = 'Please set your HuggingFace API key first.';
          return;
        }
        await tauri.quickConfigureHuggingface(
          'meta-llama/Llama-3.2-3B-Instruct',
        );
      } else if (backend === 'openrouter') {
        if (!state.openrouterApiKey) {
          state.error = 'Please set your OpenRouter API key first.';
          return;
        }
        await tauri.quickConfigureOpenrouter(
          'meta-llama/llama-3.2-3b-instruct',
        );
      } else if (backend === 'openai') {
        if (!state.openaiApiKey) {
          state.error = 'Please set your OpenAI API key first.';
          return;
        }
        await tauri.quickConfigureOpenai('gpt-4o-mini');
      }

      state.activeBackend = backend;
      state.success = `Switched to ${backend} backend`;
    } catch (err) {
      state.error = `Failed to switch backend: ${err}`;
    }
  });

  const saveApiKey = $(async (provider: 'hf' | 'openrouter' | 'openai') => {
    if (!state.isTauri) return;

    savingKey.value = provider;
    state.error = null;
    state.success = null;

    try {
      if (provider === 'hf') {
        await tauri.setHfApiKey(state.hfApiKey);
      } else if (provider === 'openrouter') {
        await tauri.setOpenrouterApiKey(state.openrouterApiKey);
      } else {
        await tauri.setOpenaiApiKey(state.openaiApiKey);
      }

      // Refresh providers
      state.providers = await tauri.getAvailableProviders();
      state.success = 'API key saved successfully';
    } catch (err) {
      state.error = `Failed to save API key: ${err}`;
    } finally {
      savingKey.value = null;
    }
  });

  const downloadModel = $(async (model: tauri.RecommendedModel) => {
    if (!state.isTauri) return;

    state.downloadingModel = model.repo_id;
    state.downloadProgress = 0;
    state.error = null;

    try {
      // Set up progress listener
      const unlisten = await tauri.onDownloadProgress((progress) => {
        state.downloadProgress = progress.percentage;
      });

      // Download the model
      const modelPath = await tauri.downloadModel(
        model.repo_id,
        model.filename,
      );

      // Clean up listener
      unlisten();

      // Refresh local models list
      state.localModels = await tauri.listModels();

      // Auto-select the newly downloaded model
      state.selectedLocalModel = modelPath;

      state.success = `Downloaded ${model.display_name}`;
    } catch (err) {
      state.error = `Failed to download model: ${err}`;
    } finally {
      state.downloadingModel = null;
      state.downloadProgress = 0;
    }
  });

  const deleteModel = $(async (modelPath: string) => {
    if (!state.isTauri) return;

    state.error = null;
    state.success = null;

    try {
      await tauri.deleteModel(modelPath);
      state.localModels = await tauri.listModels();

      if (state.selectedLocalModel === modelPath) {
        state.selectedLocalModel =
          state.localModels.length > 0 ? state.localModels[0].path : '';
      }

      state.success = 'Model deleted';
    } catch (err) {
      state.error = `Failed to delete model: ${err}`;
    }
  });

  const selectLocalModel = $(async (modelPath: string) => {
    if (!state.isTauri) return;

    state.selectedLocalModel = modelPath;
    state.error = null;
    state.success = null;

    try {
      await tauri.setDefaultModel(modelPath);
      if (state.activeBackend === 'local') {
        await tauri.quickConfigureLocal(modelPath);
      }
      state.success = 'Model selected';
    } catch (err) {
      state.error = `Failed to select model: ${err}`;
    }
  });

  if (!state.isTauri) {
    return (
      <div class="max-w-4xl mx-auto">
        <div class="flex items-center gap-3 mb-8">
          <MainSidebarButton />
          <h1 class="text-2xl font-semibold">Settings</h1>
        </div>

        <div class="bg-amber-50 border border-amber-200 rounded-lg p-6">
          <div class="flex items-start gap-4">
            <LuSettings class="w-6 h-6 text-amber-600 mt-1" />
            <div>
              <h2 class="font-semibold text-amber-800 mb-2">
                Desktop App Required
              </h2>
              <p class="text-amber-700">
                Local inference settings are only available when running the
                Braincells desktop application. The desktop app enables:
              </p>
              <ul class="mt-3 space-y-2 text-amber-700">
                <li class="flex items-center gap-2">
                  <LuCpu class="w-4 h-4" />
                  Local LLM inference with GPU acceleration
                </li>
                <li class="flex items-center gap-2">
                  <LuCloud class="w-4 h-4" />
                  Cloud fallback (HuggingFace, OpenRouter, OpenAI)
                </li>
                <li class="flex items-center gap-2">
                  <LuZap class="w-4 h-4" />
                  Parallel inference for spreadsheet cells
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (state.loading) {
    return (
      <div class="flex items-center justify-center h-64">
        <LuLoader2 class="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div class="max-w-4xl mx-auto pb-12">
      <div class="flex items-center gap-3 mb-8">
        <MainSidebarButton />
        <h1 class="text-2xl font-semibold">Settings</h1>
      </div>

      {/* Status Messages */}
      {state.error && (
        <div class="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
          {state.error}
        </div>
      )}
      {state.success && (
        <div class="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg text-green-700 flex items-center gap-2">
          <LuCheck class="w-4 h-4" />
          {state.success}
        </div>
      )}

      {/* Backend Selection */}
      <section class="mb-10">
        <h2 class="text-lg font-semibold mb-4 flex items-center gap-2">
          <LuCpu class="w-5 h-5" />
          Inference Backend
        </h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Local */}
          <button
            type="button"
            onClick$={() => selectBackend('local')}
            class={cn(
              'p-4 rounded-lg border-2 text-left transition-all',
              state.activeBackend === 'local'
                ? 'border-primary bg-primary/5'
                : 'border-gray-200 hover:border-gray-300',
            )}
          >
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center gap-2">
                <LuHardDrive class="w-5 h-5 text-primary" />
                <span class="font-medium">Local (llama.cpp)</span>
              </div>
              {state.activeBackend === 'local' && (
                <LuCheck class="w-5 h-5 text-primary" />
              )}
            </div>
            <p class="text-sm text-gray-600">
              Run models locally with GPU acceleration. Private and offline.
            </p>
          </button>

          {/* HuggingFace */}
          <button
            type="button"
            onClick$={() => selectBackend('huggingface')}
            class={cn(
              'p-4 rounded-lg border-2 text-left transition-all',
              state.activeBackend === 'huggingface'
                ? 'border-primary bg-primary/5'
                : 'border-gray-200 hover:border-gray-300',
            )}
          >
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center gap-2">
                <LuCloud class="w-5 h-5 text-yellow-600" />
                <span class="font-medium">HuggingFace</span>
              </div>
              {state.activeBackend === 'huggingface' && (
                <LuCheck class="w-5 h-5 text-primary" />
              )}
            </div>
            <p class="text-sm text-gray-600">
              Use HuggingFace Inference API. Requires API key.
            </p>
          </button>

          {/* OpenRouter */}
          <button
            type="button"
            onClick$={() => selectBackend('openrouter')}
            class={cn(
              'p-4 rounded-lg border-2 text-left transition-all',
              state.activeBackend === 'openrouter'
                ? 'border-primary bg-primary/5'
                : 'border-gray-200 hover:border-gray-300',
            )}
          >
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center gap-2">
                <LuCloud class="w-5 h-5 text-purple-600" />
                <span class="font-medium">OpenRouter</span>
              </div>
              {state.activeBackend === 'openrouter' && (
                <LuCheck class="w-5 h-5 text-primary" />
              )}
            </div>
            <p class="text-sm text-gray-600">
              Access many models through OpenRouter. Pay-per-token.
            </p>
          </button>

          {/* OpenAI */}
          <button
            type="button"
            onClick$={() => selectBackend('openai')}
            class={cn(
              'p-4 rounded-lg border-2 text-left transition-all',
              state.activeBackend === 'openai'
                ? 'border-primary bg-primary/5'
                : 'border-gray-200 hover:border-gray-300',
            )}
          >
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center gap-2">
                <LuCloud class="w-5 h-5 text-green-600" />
                <span class="font-medium">OpenAI</span>
              </div>
              {state.activeBackend === 'openai' && (
                <LuCheck class="w-5 h-5 text-primary" />
              )}
            </div>
            <p class="text-sm text-gray-600">
              Use GPT-4o and other OpenAI models. Requires API key.
            </p>
          </button>
        </div>
      </section>

      {/* Local Models */}
      <section class="mb-10">
        <h2 class="text-lg font-semibold mb-4 flex items-center gap-2">
          <LuDownload class="w-5 h-5" />
          Local Models
        </h2>

        {/* Downloaded Models */}
        {state.localModels.length > 0 && (
          <div class="mb-6">
            <h3 class="text-sm font-medium text-gray-600 mb-3">
              Downloaded Models
            </h3>
            <div class="space-y-2">
              {state.localModels.map((model) => (
                <div
                  key={model.path}
                  class={cn(
                    'flex items-center justify-between p-3 rounded-lg border',
                    state.selectedLocalModel === model.path
                      ? 'border-primary bg-primary/5'
                      : 'border-gray-200',
                  )}
                >
                  <button
                    type="button"
                    onClick$={() => selectLocalModel(model.path)}
                    class="flex-1 text-left flex items-center gap-3"
                  >
                    {state.selectedLocalModel === model.path ? (
                      <LuCheck class="w-4 h-4 text-primary" />
                    ) : (
                      <div class="w-4 h-4 rounded-full border-2 border-gray-300" />
                    )}
                    <div>
                      <div class="font-medium text-sm">{model.name}</div>
                      <div class="text-xs text-gray-500">
                        {formatBytes(model.size_bytes)} - {model.format}
                      </div>
                    </div>
                  </button>
                  <button
                    type="button"
                    onClick$={() => deleteModel(model.path)}
                    class="p-2 text-red-500 hover:bg-red-50 rounded"
                  >
                    <LuTrash2 class="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recommended Models */}
        <div>
          <h3 class="text-sm font-medium text-gray-600 mb-3">
            Recommended Models for Your System
          </h3>
          <div class="space-y-2">
            {state.recommendedModels.map((model) => {
              const isDownloaded = state.localModels.some(
                (m) =>
                  m.name.includes(model.filename) ||
                  m.path.includes(model.filename),
              );
              const isDownloading = state.downloadingModel === model.repo_id;

              return (
                <div
                  key={model.repo_id}
                  class="flex items-center justify-between p-3 rounded-lg border border-gray-200"
                >
                  <div class="flex-1">
                    <div class="font-medium text-sm">{model.display_name}</div>
                    <div class="text-xs text-gray-500">
                      {model.size_gb} GB - {model.quantization} -{' '}
                      {model.description}
                    </div>
                  </div>
                  {isDownloaded ? (
                    <span class="text-sm text-green-600 flex items-center gap-1">
                      <LuCheck class="w-4 h-4" />
                      Downloaded
                    </span>
                  ) : isDownloading ? (
                    <div class="flex items-center gap-2">
                      <div class="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          class="h-full bg-primary transition-all"
                          style={{ width: `${state.downloadProgress}%` }}
                        />
                      </div>
                      <span class="text-xs text-gray-500">
                        {Math.round(state.downloadProgress)}%
                      </span>
                    </div>
                  ) : (
                    <Button
                      look="secondary"
                      size="sm"
                      onClick$={() => downloadModel(model)}
                    >
                      <LuDownload class="w-4 h-4 mr-1" />
                      Download
                    </Button>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* API Keys */}
      <section class="mb-10">
        <h2 class="text-lg font-semibold mb-4 flex items-center gap-2">
          <LuKey class="w-5 h-5" />
          API Keys
        </h2>
        <div class="space-y-6">
          {/* HuggingFace */}
          <div>
            <Label class="block text-sm font-medium mb-2">
              HuggingFace API Key
            </Label>
            <div class="flex gap-2">
              <Input
                type="password"
                value={state.hfApiKey}
                onInput$={(e) =>
                  (state.hfApiKey = (e.target as HTMLInputElement).value)
                }
                placeholder="hf_..."
                class="flex-1"
              />
              <Button
                look="secondary"
                onClick$={() => saveApiKey('hf')}
                disabled={savingKey.value === 'hf'}
              >
                {savingKey.value === 'hf' ? (
                  <LuLoader2 class="w-4 h-4 animate-spin" />
                ) : (
                  'Save'
                )}
              </Button>
            </div>
            <p class="text-xs text-gray-500 mt-1">
              Get your key at{' '}
              <a
                href="https://huggingface.co/settings/tokens"
                target="_blank"
                rel="noopener noreferrer"
                class="text-primary hover:underline"
              >
                huggingface.co/settings/tokens
              </a>
            </p>
          </div>

          {/* OpenRouter */}
          <div>
            <Label class="block text-sm font-medium mb-2">
              OpenRouter API Key
            </Label>
            <div class="flex gap-2">
              <Input
                type="password"
                value={state.openrouterApiKey}
                onInput$={(e) =>
                  (state.openrouterApiKey = (
                    e.target as HTMLInputElement
                  ).value)
                }
                placeholder="sk-or-..."
                class="flex-1"
              />
              <Button
                look="secondary"
                onClick$={() => saveApiKey('openrouter')}
                disabled={savingKey.value === 'openrouter'}
              >
                {savingKey.value === 'openrouter' ? (
                  <LuLoader2 class="w-4 h-4 animate-spin" />
                ) : (
                  'Save'
                )}
              </Button>
            </div>
            <p class="text-xs text-gray-500 mt-1">
              Get your key at{' '}
              <a
                href="https://openrouter.ai/settings/keys"
                target="_blank"
                rel="noopener noreferrer"
                class="text-primary hover:underline"
              >
                openrouter.ai/settings/keys
              </a>
            </p>
          </div>

          {/* OpenAI */}
          <div>
            <Label class="block text-sm font-medium mb-2">OpenAI API Key</Label>
            <div class="flex gap-2">
              <Input
                type="password"
                value={state.openaiApiKey}
                onInput$={(e) =>
                  (state.openaiApiKey = (e.target as HTMLInputElement).value)
                }
                placeholder="sk-..."
                class="flex-1"
              />
              <Button
                look="secondary"
                onClick$={() => saveApiKey('openai')}
                disabled={savingKey.value === 'openai'}
              >
                {savingKey.value === 'openai' ? (
                  <LuLoader2 class="w-4 h-4 animate-spin" />
                ) : (
                  'Save'
                )}
              </Button>
            </div>
            <p class="text-xs text-gray-500 mt-1">
              Get your key at{' '}
              <a
                href="https://platform.openai.com/api-keys"
                target="_blank"
                rel="noopener noreferrer"
                class="text-primary hover:underline"
              >
                platform.openai.com/api-keys
              </a>
            </p>
          </div>
        </div>
      </section>

      {/* Advanced Settings */}
      <section>
        <h2 class="text-lg font-semibold mb-4 flex items-center gap-2">
          <LuSettings class="w-5 h-5" />
          Advanced
        </h2>
        <div class="space-y-4">
          <div>
            <Label class="block text-sm font-medium mb-2">
              Max Concurrent Requests
            </Label>
            <Input
              type="number"
              value={state.maxConcurrent}
              onInput$={(e) =>
                (state.maxConcurrent =
                  Number.parseInt((e.target as HTMLInputElement).value, 10) ||
                  1)
              }
              min={1}
              max={16}
              class="w-32"
            />
            <p class="text-xs text-gray-500 mt-1">
              Number of parallel inference requests (1-16)
            </p>
          </div>
        </div>
      </section>
    </div>
  );
});

export const head: DocumentHead = {
  title: 'Braincells - Settings',
  meta: [
    {
      name: 'description',
      content: 'Configure your inference backend and API keys',
    },
  ],
};
