import modelManager from "../helpers/ModelManager";
import { inferenceConfig } from "../config/InferenceConfig";
import { BaseReasoningService } from "./BaseReasoningService";
import { TOKEN_LIMITS } from "../config/constants";
import logger from "../utils/logger";

interface LocalReasoningConfig {
  maxTokens?: number;
  temperature?: number;
  contextSize?: number;
}

// Ollama API client
class OllamaClient {
  private baseUrl = "http://localhost:11434";

  async isAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`, {
        method: "GET",
        signal: AbortSignal.timeout(2000)
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  async listModels(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`);
      const data = await response.json();
      return data.models?.map((m: any) => m.name) || [];
    } catch {
      return [];
    }
  }

  async generate(model: string, prompt: string, options: { temperature?: number; maxTokens?: number } = {}): Promise<string> {
    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        prompt,
        stream: false,
        options: {
          temperature: options.temperature || 0.7,
          num_predict: options.maxTokens || 500
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    return data.response || "";
  }
}

class LocalReasoningService extends BaseReasoningService {
  private ollamaClient = new OllamaClient();

  async processText(
    text: string,
    modelId: string = "qwen2.5-7b-instruct-q5_k_m",
    agentName: string | null = null,
    config: LocalReasoningConfig = {}
  ): Promise<string> {
    logger.logReasoning("LOCAL_MODEL_START", {
      modelId,
      agentName,
      textLength: text.length,
      configKeys: Object.keys(config)
    });

    if (this.isProcessing) {
      throw new Error("Already processing a request");
    }

    this.isProcessing = true;
    const startTime = Date.now();

    try {
      // Get prompt using the base class method
      const reasoningPrompt = this.getReasoningPrompt(text, agentName, config);
      
      logger.logReasoning("LOCAL_MODEL_PROMPT_PREPARED", {
        promptLength: reasoningPrompt.length,
        hasAgentName: !!agentName
      });

      // Get optimized config for reasoning use case
      const inferenceOptions = inferenceConfig.getConfigForUseCase('reasoning');

      // Calculate max tokens with configurable limits
      const maxTokens = config.maxTokens || this.calculateMaxTokens(
        text.length,
        TOKEN_LIMITS.MIN_TOKENS,
        TOKEN_LIMITS.MAX_TOKENS,
        TOKEN_LIMITS.TOKEN_MULTIPLIER
      );
      
      logger.logReasoning("LOCAL_MODEL_INFERENCE_CONFIG", {
        modelId,
        maxTokens,
        temperature: config.temperature || inferenceOptions.temperature,
        contextSize: config.contextSize || TOKEN_LIMITS.REASONING_CONTEXT_SIZE
      });

      let result: string;

      // Try Ollama first (if running and has the model)
      const ollamaAvailable = await this.ollamaClient.isAvailable();
      if (ollamaAvailable) {
        const ollamaModels = await this.ollamaClient.listModels();

        // Map common model names to Ollama names
        let ollamaModelName = modelId;
        if (modelId.includes("qwen") && !modelId.includes(":")) {
          ollamaModelName = "qwen3:4b"; // Use available Ollama model
        }

        if (ollamaModels.includes(ollamaModelName) || ollamaModels.some(m => m.startsWith(modelId.split(/[-_]/)[0]))) {
          logger.logReasoning("LOCAL_MODEL_USING_OLLAMA", {
            modelId,
            ollamaModelName,
            availableModels: ollamaModels
          });

          result = await this.ollamaClient.generate(ollamaModelName, reasoningPrompt, {
            temperature: config.temperature || inferenceOptions.temperature,
            maxTokens
          });
        } else {
          logger.logReasoning("LOCAL_MODEL_OLLAMA_NO_MATCH", {
            requestedModel: modelId,
            availableModels: ollamaModels
          });
          // Fall back to llama.cpp
          result = await modelManager.runInference(modelId, reasoningPrompt, {
            ...inferenceOptions,
            maxTokens,
            temperature: config.temperature || inferenceOptions.temperature,
            contextSize: config.contextSize || TOKEN_LIMITS.REASONING_CONTEXT_SIZE,
          });
        }
      } else {
        logger.logReasoning("LOCAL_MODEL_OLLAMA_UNAVAILABLE", {
          fallingBackTo: "llama.cpp"
        });
        // Use llama.cpp
        result = await modelManager.runInference(modelId, reasoningPrompt, {
          ...inferenceOptions,
          maxTokens,
          temperature: config.temperature || inferenceOptions.temperature,
          contextSize: config.contextSize || TOKEN_LIMITS.REASONING_CONTEXT_SIZE,
        });
      }

      const processingTime = Date.now() - startTime;
      
      logger.logReasoning("LOCAL_MODEL_SUCCESS", {
        modelId,
        processingTimeMs: processingTime,
        resultLength: result.length,
        resultPreview: result.substring(0, 100) + (result.length > 100 ? "..." : "")
      });

      return result;
    } catch (error) {
      const processingTime = Date.now() - startTime;
      
      logger.logReasoning("LOCAL_MODEL_ERROR", {
        modelId,
        processingTimeMs: processingTime,
        error: (error as Error).message,
        stack: (error as Error).stack
      });

      console.error("LocalReasoningService error:", error);
      throw error;
    } finally {
      this.isProcessing = false;
    }
  }

  // Check if local reasoning is available
  async isAvailable(): Promise<boolean> {
    try {
      // Check Ollama first (faster and simpler)
      const ollamaAvailable = await this.ollamaClient.isAvailable();
      if (ollamaAvailable) {
        const models = await this.ollamaClient.listModels();
        if (models.length > 0) {
          return true;
        }
      }
      // Fall back to llama.cpp
      await modelManager.ensureLlamaCpp();
      return true;
    } catch (error) {
      console.warn("Local reasoning not available:", (error as Error).message);
      return false;
    }
  }

  // Get list of downloaded models
  async getDownloadedModels() {
    try {
      const models: any[] = [];

      // Get Ollama models first
      const ollamaAvailable = await this.ollamaClient.isAvailable();
      if (ollamaAvailable) {
        const ollamaModels = await this.ollamaClient.listModels();
        models.push(...ollamaModels.map(name => ({
          id: name,
          name: name,
          isDownloaded: true,
          source: 'ollama'
        })));
      }

      // Also get llama.cpp models
      const modelsWithStatus = await modelManager.getModelsWithStatus();
      models.push(...modelsWithStatus.filter(model => model.isDownloaded).map(m => ({
        ...m,
        source: 'llama.cpp'
      })));

      return models;
    } catch (error) {
      console.error("Failed to get downloaded models:", error);
      return [];
    }
  }
}

export default new LocalReasoningService();
