/**
 * Tauri Inference Bridge
 *
 * Provides a unified interface for inference that works both in Tauri desktop app
 * and in the web browser. When running in Tauri, it uses the local llama.cpp backend
 * or configured cloud backend. When running in browser, it falls back to HuggingFace.
 */

import * as tauri from '~/services/tauri';
import type {
  PromptExecutionParams,
  PromptExecutionResponse,
} from './run-prompt-execution';
import { materializePrompt } from './materialize-prompt';

/**
 * Check if we should use Tauri for inference
 */
export function shouldUseTauriInference(): boolean {
  return tauri.isTauri();
}

/**
 * Run inference using the Tauri backend
 * This uses the configured backend (local llama.cpp or cloud)
 */
export async function runTauriInference(
  params: PromptExecutionParams,
): Promise<PromptExecutionResponse> {
  const { instruction, sourcesContext, data, examples, task } = params;

  // Materialize the prompt using the same logic as HuggingFace
  const inputPrompt = materializePrompt({
    instruction,
    sourcesContext,
    data,
    examples,
    task,
  });

  try {
    // Check if inference is ready
    const isReady = await tauri.isInferenceReady();

    if (!isReady) {
      // Return error but don't throw - let the caller decide what to do
      return {
        error:
          'Inference backend not ready. Please configure a model in Settings.',
        done: true,
      };
    }

    // Create the request
    const request: tauri.LLMRequest = {
      messages: [
        {
          role: 'user',
          content: inputPrompt,
        },
      ],
      model: 'default', // Will use the configured model
      max_tokens: 2048,
      temperature: 0.7,
      stream: false,
    };

    // Call the Tauri backend
    const response = await tauri.generateCell(request);

    return {
      value: response.content,
      done: true,
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error('[TauriBridge] Inference error:', errorMessage);

    return {
      error: errorMessage,
      done: true,
    };
  }
}

/**
 * Run batch inference using the Tauri backend
 * This processes multiple prompts in parallel using the inference pool
 */
export async function runTauriBatchInference(
  paramsArray: PromptExecutionParams[],
): Promise<PromptExecutionResponse[]> {
  try {
    // Check if inference is ready
    const isReady = await tauri.isInferenceReady();

    if (!isReady) {
      return paramsArray.map(() => ({
        error:
          'Inference backend not ready. Please configure a model in Settings.',
        done: true,
      }));
    }

    // Convert to LLMRequests
    const requests: tauri.LLMRequest[] = paramsArray.map((params) => {
      const inputPrompt = materializePrompt({
        instruction: params.instruction,
        sourcesContext: params.sourcesContext,
        data: params.data,
        examples: params.examples,
        task: params.task,
      });

      return {
        messages: [
          {
            role: 'user',
            content: inputPrompt,
          },
        ],
        model: 'default',
        max_tokens: 2048,
        temperature: 0.7,
        stream: false,
      };
    });

    // Call batch inference
    const responses = await tauri.generateBatch(requests);

    // Convert responses
    return responses.map((response) => {
      if ('error' in response) {
        return {
          error: response.error,
          done: true,
        };
      }

      return {
        value: response.content,
        done: true,
      };
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error('[TauriBridge] Batch inference error:', errorMessage);

    return paramsArray.map(() => ({
      error: errorMessage,
      done: true,
    }));
  }
}

/**
 * Get information about the current inference backend
 */
export async function getInferenceInfo(): Promise<{
  isAvailable: boolean;
  backendName: string | null;
  isTauri: boolean;
}> {
  if (!tauri.isTauri()) {
    return {
      isAvailable: false,
      backendName: null,
      isTauri: false,
    };
  }

  try {
    const [isReady, backendName] = await Promise.all([
      tauri.isInferenceReady(),
      tauri.getBackendName(),
    ]);

    return {
      isAvailable: isReady,
      backendName,
      isTauri: true,
    };
  } catch {
    return {
      isAvailable: false,
      backendName: null,
      isTauri: true,
    };
  }
}
