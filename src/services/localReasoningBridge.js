const modelManager = require("../helpers/modelManagerBridge").default;
const debugLogger = require("../helpers/debugLogger");

// Use native fetch if available (Node 18+), otherwise polyfill
const fetch = globalThis.fetch || require('node-fetch');

class OllamaClient {
  constructor() {
    this.baseUrl = "http://localhost:11434";
  }

  async isAvailable() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);

      const response = await fetch(`${this.baseUrl}/api/tags`, {
        method: "GET",
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      return response.ok;
    } catch {
      return false;
    }
  }

  async generate(model, prompt, options = {}) {
    // Add system prompt to disable thinking for Qwen models
    const systemPrompt = model.includes('qwen')
      ? "You are a helpful AI assistant. Provide direct, concise responses without showing your thinking process. Do not include <think> tags or reasoning steps. Just give the final answer."
      : "You are a helpful AI assistant.";

    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        prompt,
        system: systemPrompt,
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
    let response_text = data.response || "";

    // Strip thinking tags if they somehow still appear
    if (response_text.includes('<think>')) {
      response_text = response_text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
    }

    return response_text;
  }
}

class LocalReasoningService {
  constructor() {
    this.isProcessing = false;
    this.ollamaClient = new OllamaClient();
  }

  async isAvailable() {
    try {
      // Check Ollama first
      const ollamaAvailable = await this.ollamaClient.isAvailable();
      if (ollamaAvailable) {
        return true;
      }

      // Fall back to llama.cpp
      await modelManager.ensureLlamaCpp();

      // Check if at least one model is downloaded
      const models = await modelManager.getAllModels();
      return models.some(model => model.isDownloaded);
    } catch (error) {
      return false;
    }
  }

  async processText(text, modelId, agentName = null, config = {}) {
    debugLogger.logReasoning("LOCAL_BRIDGE_START", {
      modelId,
      agentName,
      textLength: text.length,
      hasConfig: Object.keys(config).length > 0
    });
    
    if (this.isProcessing) {
      throw new Error("Already processing a request");
    }

    this.isProcessing = true;
    const startTime = Date.now();

    try {
      // Get custom prompts from the request context
      const customPrompts = config.customPrompts || null;
      
      // Build the reasoning prompt
      const reasoningPrompt = this.getReasoningPrompt(text, agentName, customPrompts);
      
      debugLogger.logReasoning("LOCAL_BRIDGE_PROMPT", {
        promptLength: reasoningPrompt.length,
        hasAgentName: !!agentName,
        hasCustomPrompts: !!customPrompts
      });
      
      const inferenceConfig = {
        maxTokens: config.maxTokens || this.calculateMaxTokens(text.length),
        temperature: config.temperature || 0.7,
        topK: config.topK || 40,
        topP: config.topP || 0.9,
        repeatPenalty: config.repeatPenalty || 1.1,
        contextSize: config.contextSize || 4096,
        threads: config.threads || 4,
        systemPrompt: "You are a helpful AI assistant that processes and improves text."
      };
      
      debugLogger.logReasoning("LOCAL_BRIDGE_INFERENCE", {
        modelId,
        config: inferenceConfig
      });

      let result;

      // Check if this is an Ollama model (contains colon like "qwen3:4b")
      if (modelId.includes(':')) {
        debugLogger.logReasoning("LOCAL_BRIDGE_USING_OLLAMA", {
          modelId,
          ollamaAvailable: await this.ollamaClient.isAvailable()
        });

        const ollamaAvailable = await this.ollamaClient.isAvailable();
        if (ollamaAvailable) {
          result = await this.ollamaClient.generate(modelId, reasoningPrompt, {
            temperature: inferenceConfig.temperature,
            maxTokens: inferenceConfig.maxTokens
          });
        } else {
          throw new Error(`Ollama is not running. Please start Ollama to use model: ${modelId}`);
        }
      } else {
        // Use llama.cpp for GGUF models
        result = await modelManager.runInference(modelId, reasoningPrompt, inferenceConfig);
      }

      const processingTime = Date.now() - startTime;
      
      debugLogger.logReasoning("LOCAL_BRIDGE_SUCCESS", {
        modelId,
        processingTimeMs: processingTime,
        resultLength: result.length,
        resultPreview: result.substring(0, 100) + (result.length > 100 ? "..." : "")
      });

      return result;
    } catch (error) {
      const processingTime = Date.now() - startTime;
      
      debugLogger.logReasoning("LOCAL_BRIDGE_ERROR", {
        modelId,
        processingTimeMs: processingTime,
        error: error.message,
        stack: error.stack
      });
      
      throw error;
    } finally {
      this.isProcessing = false;
    }
  }

  getCustomPrompts() {
    // In main process, we can't access localStorage directly
    // This should be passed from the renderer process
    return null;
  }

  getReasoningPrompt(text, agentName, customPrompts) {
    // Default prompts
    const DEFAULT_AGENT_PROMPT = `You are {{agentName}}, a helpful AI assistant. Process and improve the following text, removing any reference to your name from the output:\n\n{{text}}\n\nImproved text:`;
    const DEFAULT_REGULAR_PROMPT = `Process and improve the following text:\n\n{{text}}\n\nImproved text:`;

    let agentPrompt = DEFAULT_AGENT_PROMPT;
    let regularPrompt = DEFAULT_REGULAR_PROMPT;

    if (customPrompts) {
      agentPrompt = customPrompts.agent || DEFAULT_AGENT_PROMPT;
      regularPrompt = customPrompts.regular || DEFAULT_REGULAR_PROMPT;
    }

    // Check if agent name is mentioned
    if (agentName && text.toLowerCase().includes(agentName.toLowerCase())) {
      return agentPrompt
        .replace(/\{\{agentName\}\}/g, agentName)
        .replace(/\{\{text\}\}/g, text);
    }
    
    return regularPrompt.replace(/\{\{text\}\}/g, text);
  }

  calculateMaxTokens(textLength, minTokens = 100, maxTokens = 2048, multiplier = 2) {
    return Math.max(minTokens, Math.min(textLength * multiplier, maxTokens));
  }
}

module.exports = {
  default: new LocalReasoningService()
};