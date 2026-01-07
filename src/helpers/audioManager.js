import ReasoningService from "../services/ReasoningService";
import { API_ENDPOINTS, buildApiUrl, normalizeBaseUrl } from "../config/constants";
import logger from "../utils/logger";

const SHORT_CLIP_DURATION_SECONDS = 2.5;
const REASONING_CACHE_TTL = 30000; // 30 seconds


class AudioManager {
  constructor() {
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.isProcessing = false;
    this.onStateChange = null;
    this.onError = null;
    this.onTranscriptionComplete = null;
    this.cachedApiKey = null;
    this.cachedTranscriptionEndpoint = null;
    this.recordingStartTime = null;
    this.processingStartTime = null; // Track when processing starts
    this.reasoningAvailabilityCache = { value: false, expiresAt: 0 };
    this.cachedReasoningPreference = null;
  }

  setCallbacks({ onStateChange, onError, onTranscriptionComplete }) {
    this.onStateChange = onStateChange;
    this.onError = onError;
    this.onTranscriptionComplete = onTranscriptionComplete;
  }

  clearCache() {
    // Clear cached values when settings change
    this.cachedApiKey = null;
    this.cachedTranscriptionEndpoint = null;
  }

  async startRecording() {
    try {
      if (this.isRecording) {
        return false;
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });


      this.mediaRecorder = new MediaRecorder(stream);
      this.audioChunks = [];
      this.recordingStartTime = Date.now();

      this.mediaRecorder.ondataavailable = (event) => {
        this.audioChunks.push(event.data);
      };

      this.mediaRecorder.onstop = async () => {
        this.isRecording = false;
        this.isProcessing = true;
        this.onStateChange?.({ isRecording: false, isProcessing: true });

        const audioBlob = new Blob(this.audioChunks, { type: "audio/wav" });
        
        if (audioBlob.size === 0) {
        }
        
        const durationSeconds = this.recordingStartTime
          ? (Date.now() - this.recordingStartTime) / 1000
          : null;
        this.recordingStartTime = null;
        await this.processAudio(audioBlob, { durationSeconds });

        // Clean up stream
        stream.getTracks().forEach((track) => track.stop());
      };

      this.mediaRecorder.start();
      this.isRecording = true;
      this.onStateChange?.({ isRecording: true, isProcessing: false });

      return true;
    } catch (error) {
      
      // Provide more specific error messages
      let errorTitle = "Recording Error";
      let errorDescription = `Failed to access microphone: ${error.message}`;
      
      if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
        errorTitle = "Microphone Access Denied";
        errorDescription = "Please grant microphone permission in your system settings and try again.";
      } else if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
        errorTitle = "No Microphone Found";
        errorDescription = "No microphone was detected. Please connect a microphone and try again.";
      } else if (error.name === "NotReadableError" || error.name === "TrackStartError") {
        errorTitle = "Microphone In Use";
        errorDescription = "The microphone is being used by another application. Please close other apps and try again.";
      }
      
      this.onError?.({
        title: errorTitle,
        description: errorDescription,
      });
      return false;
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.processingStartTime = Date.now();
      // Log to both console and terminal
      const msg = '[TIMING:FRONTEND] ⏹️  RECORDING STOPPED - Processing pipeline starting...';
      console.log(msg);
      if (typeof window !== 'undefined' && window.electronAPI?.logToTerminal) {
        window.electronAPI.logToTerminal(msg).catch(() => {});
      }
      this.mediaRecorder.stop();
      // State change will be handled in onstop callback
      return true;
    }
    return false;
  }

  async processAudio(audioBlob, metadata = {}) {
    const overallStartTime = Date.now();
    console.log('[TIMING] processAudio() started');

    try {
      const useLocalWhisper = localStorage.getItem("useLocalWhisper") === "true";
      const whisperModel = localStorage.getItem("whisperModel") || "base";

      let result;
      if (useLocalWhisper) {
        result = await this.processWithLocalWhisper(audioBlob, whisperModel, metadata);
      } else {
        result = await this.processWithOpenAIAPI(audioBlob, metadata);
      }

      const pasteStartTime = Date.now();
      this.onTranscriptionComplete?.(result);
      const pasteEndTime = Date.now();

      const overallEndTime = Date.now();
      const overallTime = overallEndTime - overallStartTime;
      const callbackTime = pasteEndTime - pasteStartTime;
      const absoluteEndToEndTime = this.processingStartTime
        ? overallEndTime - this.processingStartTime
        : overallTime;

      const msg1 = `[TIMING:FRONTEND] ✅ TOTAL END-TO-END TIME (from stopRecording): ${absoluteEndToEndTime}ms (${(absoluteEndToEndTime / 1000).toFixed(2)}s)`;
      const msg2 = `[TIMING:FRONTEND]   - processAudio() time: ${overallTime}ms`;
      const msg3 = `[TIMING:FRONTEND]   - Transcription callback time: ${callbackTime}ms`;
      const msg4 = `[TIMING:FRONTEND]   - Transcription + reasoning time: ${overallTime - callbackTime}ms`;

      console.log(msg1);
      console.log(msg2);
      console.log(msg3);
      console.log(msg4);

      if (typeof window !== 'undefined' && window.electronAPI?.logToTerminal) {
        window.electronAPI.logToTerminal(msg1).catch(() => {});
        window.electronAPI.logToTerminal(msg2).catch(() => {});
        window.electronAPI.logToTerminal(msg3).catch(() => {});
        window.electronAPI.logToTerminal(msg4).catch(() => {});
      }

      this.processingStartTime = null; // Reset for next recording
    } catch (error) {
      if (error.message !== "No audio detected") {
        this.onError?.({
          title: "Transcription Error",
          description: `Transcription failed: ${error.message}`,
        });
      }

      const overallEndTime = Date.now();
      console.log(`[TIMING] ❌ processAudio() failed after ${overallEndTime - overallStartTime}ms`);
    } finally {
      this.isProcessing = false;
      this.onStateChange?.({ isRecording: false, isProcessing: false });
    }
  }

  async processWithLocalWhisper(audioBlob, model = "base", metadata = {}) {
    const startTime = Date.now();
    console.log(`[TIMING] processWithLocalWhisper() started with model: ${model}`);

    try {
      const bufferStartTime = Date.now();
      const arrayBuffer = await audioBlob.arrayBuffer();
      const bufferTime = Date.now() - bufferStartTime;
      console.log(`[TIMING]   - Blob to ArrayBuffer conversion: ${bufferTime}ms`);

      const language = localStorage.getItem("preferredLanguage");
      const options = { model };
      if (language && language !== "auto") {
        options.language = language;
      }

      const transcribeStartTime = Date.now();
      const result = await window.electronAPI.transcribeLocalWhisper(
        arrayBuffer,
        options
      );
      const transcribeTime = Date.now() - transcribeStartTime;
      console.log(`[TIMING]   - Local Whisper IPC call: ${transcribeTime}ms`);

      if (result.success && result.text) {
        const reasoningStartTime = Date.now();
        console.log(`[TIMING]   - Starting AI cleanup/reasoning...`);
        const text = await this.processTranscription(result.text, "local");
        const reasoningTime = Date.now() - reasoningStartTime;
        console.log(`[TIMING]   - AI cleanup/reasoning completed: ${reasoningTime}ms`);

        const totalTime = Date.now() - startTime;
        console.log(`[TIMING] ✅ processWithLocalWhisper() completed in ${totalTime}ms`);

        if (text !== null && text !== undefined) {
          return { success: true, text: text || result.text, source: "local" };
        } else {
          throw new Error("No text transcribed");
        }
      } else if (result.success === false && result.message === "No audio detected") {
        this.onError?.({
          title: "No Audio Detected",
          description: "The recording contained no detectable audio. Please check your microphone settings.",
        });
        throw new Error("No audio detected");
      } else {
        throw new Error(result.error || "Local Whisper transcription failed");
      }
    } catch (error) {
      const errorTime = Date.now() - startTime;
      console.log(`[TIMING] ❌ processWithLocalWhisper() failed after ${errorTime}ms: ${error.message}`);

      if (error.message === "No audio detected") {
        throw error;
      }

      const allowOpenAIFallback = localStorage.getItem("allowOpenAIFallback") === "true";
      const isLocalMode = localStorage.getItem("useLocalWhisper") === "true";

      if (allowOpenAIFallback && isLocalMode) {
        try {
          console.log('[TIMING] Attempting OpenAI fallback...');
          const fallbackResult = await this.processWithOpenAIAPI(audioBlob, metadata);
          return { ...fallbackResult, source: "openai-fallback" };
        } catch (fallbackError) {
          throw new Error(`Local Whisper failed: ${error.message}. OpenAI fallback also failed: ${fallbackError.message}`);
        }
      } else {
        throw new Error(`Local Whisper failed: ${error.message}`);
      }
    }
  }

  async getAPIKey() {
    if (this.cachedApiKey) {
      return this.cachedApiKey;
    }

    // Check if using a custom transcription endpoint (e.g., Groq)
    const customBaseUrl = typeof localStorage !== "undefined"
      ? (localStorage.getItem("cloudTranscriptionBaseUrl") || "").trim()
      : "";
    const isCustomEndpoint = customBaseUrl && 
      !customBaseUrl.includes("api.openai.com");

    // For custom endpoints, prefer the transcription-specific API key
    if (isCustomEndpoint) {
      const transcriptionApiKey = localStorage.getItem("cloudTranscriptionApiKey");
      if (transcriptionApiKey && transcriptionApiKey.trim()) {
        this.cachedApiKey = transcriptionApiKey.trim();
        return this.cachedApiKey;
      }
    }

    // Fall back to OpenAI key
    let apiKey = await window.electronAPI.getOpenAIKey();
    if (
      !apiKey ||
      apiKey.trim() === "" ||
      apiKey === "your_openai_api_key_here"
    ) {
      apiKey = localStorage.getItem("openaiApiKey");
    }

    if (
      !apiKey ||
      apiKey.trim() === "" ||
      apiKey === "your_openai_api_key_here"
    ) {
      const errorMsg = isCustomEndpoint
        ? "API key not found. Please set your API key for the custom transcription endpoint in Settings."
        : "OpenAI API key not found. Please set your API key in the .env file or Control Panel.";
      throw new Error(errorMsg);
    }

    this.cachedApiKey = apiKey;
    return apiKey;
  }

  async optimizeAudio(audioBlob) {
    return new Promise((resolve) => {
      const audioContext = new (window.AudioContext ||
        window.webkitAudioContext)();
      const reader = new FileReader();

      reader.onload = async () => {
        try {
          const arrayBuffer = reader.result;
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

          // Convert to 16kHz mono for smaller size and faster upload
          const sampleRate = 16000;
          const channels = 1;
          const length = Math.floor(audioBuffer.duration * sampleRate);
          const offlineContext = new OfflineAudioContext(
            channels,
            length,
            sampleRate
          );

          const source = offlineContext.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(offlineContext.destination);
          source.start();

          const renderedBuffer = await offlineContext.startRendering();
          const wavBlob = this.audioBufferToWav(renderedBuffer);
          resolve(wavBlob);
        } catch (error) {
          // If optimization fails, use original
          resolve(audioBlob);
        }
      };

      reader.onerror = () => resolve(audioBlob);
      reader.readAsArrayBuffer(audioBlob);
    });
  }

  audioBufferToWav(buffer) {
    const length = buffer.length;
    const arrayBuffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(arrayBuffer);
    const sampleRate = buffer.sampleRate;
    const channelData = buffer.getChannelData(0);

    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(0, "RIFF");
    view.setUint32(4, 36 + length * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, length * 2, true);

    let offset = 44;
    for (let i = 0; i < length; i++) {
      const sample = Math.max(-1, Math.min(1, channelData[i]));
      view.setInt16(
        offset,
        sample < 0 ? sample * 0x8000 : sample * 0x7fff,
        true
      );
      offset += 2;
    }

    return new Blob([arrayBuffer], { type: "audio/wav" });
  }

  async processWithReasoningModel(text, model, agentName) {
    logger.logReasoning("CALLING_REASONING_SERVICE", {
      model,
      agentName,
      textLength: text.length
    });
    
    const startTime = Date.now();
    
    try {
      const result = await ReasoningService.processText(text, model, agentName);
      
      const processingTime = Date.now() - startTime;
      
      logger.logReasoning("REASONING_SERVICE_COMPLETE", {
        model,
        processingTimeMs: processingTime,
        resultLength: result.length,
        success: true
      });
      
      return result;
    } catch (error) {
      const processingTime = Date.now() - startTime;
      
      logger.logReasoning("REASONING_SERVICE_ERROR", {
        model,
        processingTimeMs: processingTime,
        error: error.message,
        stack: error.stack
      });
      
      throw error;
    }
  }

  async isReasoningAvailable() {
    if (typeof window === "undefined" || !window.localStorage) {
      return false;
    }

    const storedValue = localStorage.getItem("useReasoningModel");
    const now = Date.now();
    const cacheValid =
      this.reasoningAvailabilityCache &&
      now < this.reasoningAvailabilityCache.expiresAt &&
      this.cachedReasoningPreference === storedValue;

    if (cacheValid) {
      return this.reasoningAvailabilityCache.value;
    }

    logger.logReasoning("REASONING_STORAGE_CHECK", {
      storedValue,
      typeOfStoredValue: typeof storedValue,
      isTrue: storedValue === "true",
      isTruthy: !!storedValue && storedValue !== "false",
    });

    const useReasoning =
      storedValue === "true" || (!!storedValue && storedValue !== "false");

    if (!useReasoning) {
      this.reasoningAvailabilityCache = {
        value: false,
        expiresAt: now + REASONING_CACHE_TTL,
      };
      this.cachedReasoningPreference = storedValue;
      return false;
    }

    try {
      const isAvailable = await ReasoningService.isAvailable();

      logger.logReasoning("REASONING_AVAILABILITY", {
        isAvailable,
        reasoningEnabled: useReasoning,
        finalDecision: useReasoning && isAvailable,
      });

      this.reasoningAvailabilityCache = {
        value: isAvailable,
        expiresAt: now + REASONING_CACHE_TTL,
      };
      this.cachedReasoningPreference = storedValue;

      return isAvailable;
    } catch (error) {
      logger.logReasoning("REASONING_AVAILABILITY_ERROR", {
        error: error.message,
        stack: error.stack,
      });

      this.reasoningAvailabilityCache = {
        value: false,
        expiresAt: now + REASONING_CACHE_TTL,
      };
      this.cachedReasoningPreference = storedValue;
      return false;
    }
  }

  async processTranscription(text, source) {
    const normalizedText = typeof text === "string" ? text.trim() : "";

    logger.logReasoning("TRANSCRIPTION_RECEIVED", {
      source,
      textLength: normalizedText.length,
      textPreview: normalizedText.substring(0, 100) + (normalizedText.length > 100 ? "..." : ""),
      timestamp: new Date().toISOString()
    });

    const reasoningModel = (typeof window !== 'undefined' && window.localStorage)
      ? (localStorage.getItem("reasoningModel") || "gpt-4o-mini")
      : "gpt-4o-mini";
    const reasoningProvider = (typeof window !== 'undefined' && window.localStorage)
      ? (localStorage.getItem("reasoningProvider") || "auto")
      : "auto";
    const agentName = (typeof window !== 'undefined' && window.localStorage)
      ? (localStorage.getItem("agentName") || null)
      : null;
    const useReasoning = await this.isReasoningAvailable();

    logger.logReasoning("REASONING_CHECK", {
      useReasoning,
      reasoningModel,
      reasoningProvider,
      agentName
    });

    if (useReasoning) {
      try {
        const preparedText = normalizedText;

        logger.logReasoning("SENDING_TO_REASONING", {
          preparedTextLength: preparedText.length,
          model: reasoningModel,
          provider: reasoningProvider
        });

        const aiStartTime = Date.now();
        const result = await this.processWithReasoningModel(preparedText, reasoningModel, agentName);
        
        logger.logReasoning("REASONING_SUCCESS", {
          resultLength: result.length,
          resultPreview: result.substring(0, 100) + (result.length > 100 ? "..." : ""),
          processingTime: new Date().toISOString()
        });

        return result;
      } catch (error) {
        logger.logReasoning("REASONING_FAILED", {
          error: error.message,
          stack: error.stack,
          fallbackToCleanup: true
        });
      }
    }

    logger.logReasoning("USING_STANDARD_CLEANUP", {
      reason: useReasoning ? "Reasoning failed" : "Reasoning not enabled"
    });

    return normalizedText;
  }

  async processWithOpenAIAPI(audioBlob, metadata = {}) {
    const startTime = Date.now();
    console.log('[TIMING] processWithOpenAIAPI() started');

    const language = localStorage.getItem("preferredLanguage");
    const allowLocalFallback =
      localStorage.getItem("allowLocalFallback") === "true";
    const fallbackModel = localStorage.getItem("fallbackWhisperModel") || "base";

    try {
      const durationSeconds = metadata.durationSeconds ?? null;
      const shouldSkipOptimizationForDuration =
        typeof durationSeconds === "number" &&
        durationSeconds > 0 &&
        durationSeconds < SHORT_CLIP_DURATION_SECONDS;

      const shouldOptimize =
        !shouldSkipOptimizationForDuration && audioBlob.size > 1024 * 1024;

      const prepStartTime = Date.now();
      const [apiKey, optimizedAudio] = await Promise.all([
        this.getAPIKey(),
        shouldOptimize ? this.optimizeAudio(audioBlob) : Promise.resolve(audioBlob),
      ]);
      const prepTime = Date.now() - prepStartTime;
      console.log(`[TIMING]   - API key + audio optimization: ${prepTime}ms (optimized: ${shouldOptimize})`);

      // Get the transcription model - use custom model for custom endpoints, default to whisper-1 for OpenAI
      const customBaseUrl = (localStorage.getItem("cloudTranscriptionBaseUrl") || "").trim();
      const isCustomEndpoint = customBaseUrl && !customBaseUrl.includes("api.openai.com");
      const defaultModel = isCustomEndpoint ? "whisper-large-v3" : "whisper-1";
      const transcriptionModel = localStorage.getItem("cloudTranscriptionModel") || defaultModel;

      const formData = new FormData();
      formData.append("file", optimizedAudio, "audio.wav");
      formData.append("model", transcriptionModel);

      if (language && language !== "auto") {
        formData.append("language", language);
      }

      const apiStartTime = Date.now();
      const response = await fetch(
        this.getTranscriptionEndpoint(),
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${apiKey}`,
          },
          body: formData,
        }
      );
      const apiTime = Date.now() - apiStartTime;
      console.log(`[TIMING]   - OpenAI API call: ${apiTime}ms`);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API Error: ${response.status} ${errorText}`);
      }

      const result = await response.json();

      if (result.text) {
        const reasoningStartTime = Date.now();
        const text = await this.processTranscription(result.text, "openai");
        const reasoningTime = Date.now() - reasoningStartTime;
        console.log(`[TIMING]   - Reasoning/processing: ${reasoningTime}ms`);

        const totalTime = Date.now() - startTime;
        console.log(`[TIMING] ✅ processWithOpenAIAPI() completed in ${totalTime}ms`);

        const source = await this.isReasoningAvailable() ? "openai-reasoned" : "openai";
        return { success: true, text, source };
      } else {
        throw new Error("No text transcribed");
      }
    } catch (error) {
      const errorTime = Date.now() - startTime;
      console.log(`[TIMING] ❌ processWithOpenAIAPI() failed after ${errorTime}ms: ${error.message}`);

      const isOpenAIMode = localStorage.getItem("useLocalWhisper") !== "true";

      if (allowLocalFallback && isOpenAIMode) {
        try {
          console.log('[TIMING] Attempting local fallback...');
          const arrayBuffer = await audioBlob.arrayBuffer();
          const options = { model: fallbackModel };
          if (language && language !== "auto") {
            options.language = language;
          }

          const result = await window.electronAPI.transcribeLocalWhisper(
            arrayBuffer,
            options
          );

          if (result.success && result.text) {
            const text = await this.processTranscription(result.text, "local-fallback");
            if (text) {
              return { success: true, text, source: "local-fallback" };
            }
          }
          throw error;
        } catch (fallbackError) {
          throw new Error(
            `OpenAI API failed: ${error.message}. Local fallback also failed: ${fallbackError.message}`
          );
        }
      }

      throw error;
    }
  }

  getTranscriptionEndpoint() {
    if (this.cachedTranscriptionEndpoint) {
      return this.cachedTranscriptionEndpoint;
    }

    try {
      const stored = typeof localStorage !== "undefined"
        ? localStorage.getItem("cloudTranscriptionBaseUrl") || ""
        : "";
      const trimmed = stored.trim();
      const base = trimmed ? trimmed : API_ENDPOINTS.TRANSCRIPTION_BASE;
      const normalizedBase = normalizeBaseUrl(base);

      if (!normalizedBase) {
        this.cachedTranscriptionEndpoint = API_ENDPOINTS.TRANSCRIPTION;
        return API_ENDPOINTS.TRANSCRIPTION;
      }

      const isLocalhost = normalizedBase.includes('://localhost') || normalizedBase.includes('://127.0.0.1');
      if (!normalizedBase.startsWith('https://') && !isLocalhost) {
        console.warn('Non-HTTPS endpoint rejected for security. Using default.');
        this.cachedTranscriptionEndpoint = API_ENDPOINTS.TRANSCRIPTION;
        return API_ENDPOINTS.TRANSCRIPTION;
      }

      let endpoint;
      if (/\/audio\/(transcriptions|translations)$/i.test(normalizedBase)) {
        endpoint = normalizedBase;
      } else {
        endpoint = buildApiUrl(normalizedBase, '/audio/transcriptions');
      }

      this.cachedTranscriptionEndpoint = endpoint;
      return endpoint;
    } catch (error) {
      console.warn('Failed to resolve transcription endpoint:', error);
      this.cachedTranscriptionEndpoint = API_ENDPOINTS.TRANSCRIPTION;
      return API_ENDPOINTS.TRANSCRIPTION;
    }
  }

  async safePaste(text) {
    const startTime = Date.now();
    console.log('[TIMING] safePaste() started');

    try {
      const pasteStartTime = Date.now();
      await window.electronAPI.pasteText(text);
      const pasteTime = Date.now() - pasteStartTime;

      const totalTime = Date.now() - startTime;
      console.log(`[TIMING] ✅ safePaste() completed in ${totalTime}ms`);
      console.log(`[TIMING]   - Actual paste operation: ${pasteTime}ms`);

      return true;
    } catch (error) {
      const errorTime = Date.now() - startTime;
      console.log(`[TIMING] ❌ safePaste() failed after ${errorTime}ms: ${error.message}`);

      this.onError?.({
        title: "Paste Error",
        description: `Failed to paste text. Please check accessibility permissions. ${error.message}`,
      });
      return false;
    }
  }

  async saveTranscription(text) {
    try {
      await window.electronAPI.saveTranscription(text);
      return true;
    } catch (error) {
      return false;
    }
  }

  getState() {
    return {
      isRecording: this.isRecording,
      isProcessing: this.isProcessing,
    };
  }

  cleanup() {
    if (this.mediaRecorder && this.isRecording) {
      this.stopRecording();
    }
    this.onStateChange = null;
    this.onError = null;
    this.onTranscriptionComplete = null;
  }
}

export default AudioManager;
