/**
 * SloughGPT TypeScript SDK
 * Client for interacting with the SloughGPT API from React Native / Expo
 */

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface GenerateRequest {
  prompt: string;
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  personality?: string;
  model?: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  model?: string;
  temperature?: number;
  max_new_tokens?: number;
  stream?: boolean;
}

export interface GenerationResult {
  text: string;
  model: string;
  personality?: string;
  inference_time_ms?: number;
}

export interface ChatResult {
  message: ChatMessage;
  model: string;
  inference_time_ms?: number;
}

export interface HealthStatus {
  status: 'healthy' | 'unhealthy';
  model_loaded: boolean;
  model_type: string;
}

export interface SystemInfo {
  name: string;
  version: string;
  model: {
    type: string;
    loaded: boolean;
  };
}

export interface ModelInfo {
  model_id: string;
  name: string;
  description?: string;
  model_type?: string;
}

export interface MetricsData {
  requests_today: number;
  tokens_today: number;
  cache_hit_rate: number;
}

export interface TrainingJob {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: number;
}

export interface Experiment {
  experiment_id: string;
  name: string;
  description?: string;
  metrics?: Record<string, number>;
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface SloughGPTConfig {
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
  headers?: Record<string, string>;
  onLog?: (level: LogLevel, message: string) => void;
}

export class SloughGPTError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public response?: any
  ) {
    super(message);
    this.name = 'SloughGPTError';
  }
}

export class SloughGPTClient {
  private baseUrl: string;
  private timeout: number;
  private headers: Record<string, string>;
  private onLog?: (level: LogLevel, message: string) => void;

  constructor(config: SloughGPTConfig = {}) {
    this.baseUrl = (config.baseUrl || 'http://localhost:8000').replace(/\/$/, '');
    this.timeout = config.timeout || 30000;
    this.headers = {
      'Content-Type': 'application/json',
      ...(config.apiKey ? { 'X-API-Key': config.apiKey } : {}),
      ...config.headers,
    };
    this.onLog = config.onLog;
  }

  private log(level: LogLevel, message: string) {
    if (this.onLog) {
      this.onLog(level, message);
    }
  }

  private async request<T>(
    method: string,
    endpoint: string,
    body?: any
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    this.log('debug', `${method} ${url}`);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method,
        headers: this.headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new SloughGPTError(
          `HTTP ${response.status}: ${response.statusText}`,
          response.status
        );
      }

      return await response.json();
    } catch (error: any) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        throw new SloughGPTError('Request timeout', 408);
      }
      throw error;
    }
  }

  // Health & Status

  async health(): Promise<HealthStatus> {
    return this.request('GET', '/health');
  }

  async liveness(): Promise<{ status: string }> {
    return this.request('GET', '/health/live');
  }

  async readiness(): Promise<{ status: string; model_loaded: boolean }> {
    return this.request('GET', '/health/ready');
  }

  async info(): Promise<SystemInfo> {
    return this.request('GET', '/info');
  }

  // Text Generation

  async generate(request: GenerateRequest): Promise<GenerationResult> {
    this.log('info', `Generating: "${request.prompt.slice(0, 50)}..."`);
    return this.request('POST', '/generate', {
      prompt: request.prompt,
      max_new_tokens: request.max_new_tokens || 100,
      temperature: request.temperature || 0.8,
      top_k: request.top_k || 50,
      top_p: request.top_p || 0.9,
      personality: request.personality,
      model: request.model,
    });
  }

  async *generateStream(
    request: GenerateRequest
  ): AsyncGenerator<string, void, unknown> {
    const url = `${this.baseUrl}/generate/stream`;
    this.log('info', `Streaming: "${request.prompt.slice(0, 50)}..."`);

    const response = await fetch(url, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        prompt: request.prompt,
        max_new_tokens: request.max_new_tokens || 100,
        temperature: request.temperature || 0.8,
        personality: request.personality,
        model: request.model,
      }),
    });

    if (!response.ok) {
      throw new SloughGPTError(
        `HTTP ${response.status}`,
        response.status
      );
    }

    const reader = response.body?.getReader();
    if (!reader) throw new SloughGPTError('No response body');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data:')) {
          const data = line.slice(5).trim();
          if (data && data !== '[DONE]') {
            yield data;
          }
        }
      }
    }
  }

  // Chat Completions

  async chat(request: ChatRequest): Promise<ChatResult> {
    this.log('info', `Chat: ${request.messages.length} messages`);
    return this.request('POST', '/chat/completions', {
      messages: request.messages,
      model: request.model,
      temperature: request.temperature || 0.8,
      max_new_tokens: request.max_new_tokens || 100,
      stream: request.stream,
    });
  }

  async *chatStream(
    request: ChatRequest
  ): AsyncGenerator<string, void, unknown> {
    const url = `${this.baseUrl}/chat/completions`;
    this.log('info', `Chat stream: ${request.messages.length} messages`);

    const response = await fetch(url, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        messages: request.messages,
        model: request.model,
        temperature: request.temperature || 0.8,
        max_new_tokens: request.max_new_tokens || 100,
        stream: true,
      }),
    });

    if (!response.ok) {
      throw new SloughGPTError(
        `HTTP ${response.status}`,
        response.status
      );
    }

    const reader = response.body?.getReader();
    if (!reader) throw new SloughGPTError('No response body');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data:')) {
          const data = line.slice(5).trim();
          if (data && data !== '[DONE]') {
            yield data;
          }
        }
      }
    }
  }

  // Batch Processing

  async batchGenerate(prompts: string[]): Promise<GenerationResult[]> {
    return this.request('POST', '/generate/batch', { prompts });
  }

  // Models

  async listModels(): Promise<ModelInfo[]> {
    return this.request('GET', '/models');
  }

  async getModel(modelId: string): Promise<ModelInfo> {
    return this.request('GET', `/models/${modelId}`);
  }

  async loadModel(modelId: string): Promise<{ status: string }> {
    return this.request('POST', '/models/load', { model_id: modelId });
  }

  // Metrics

  async metrics(): Promise<MetricsData> {
    return this.request('GET', '/metrics');
  }

  // Training

  async startTraining(config: {
    model_name: string;
    dataset_id: string;
    epochs?: number;
    batch_size?: number;
    learning_rate?: number;
  }): Promise<TrainingJob> {
    return this.request('POST', '/training/start', config);
  }

  async getTrainingStatus(jobId: string): Promise<TrainingJob> {
    return this.request('GET', `/training/jobs/${jobId}`);
  }

  async listTrainingJobs(): Promise<TrainingJob[]> {
    return this.request('GET', '/training/jobs');
  }

  // Experiments

  async createExperiment(
    name: string,
    description?: string
  ): Promise<Experiment> {
    return this.request('POST', '/experiments', { name, description });
  }

  async listExperiments(): Promise<Experiment[]> {
    return this.request('GET', '/experiments');
  }

  async getExperiment(experimentId: string): Promise<Experiment> {
    return this.request('GET', `/experiments/${experimentId}`);
  }

  async logMetric(
    experimentId: string,
    metric: string,
    value: number,
    step?: number
  ): Promise<void> {
    await this.request('POST', `/experiments/${experimentId}/log_metric`, {
      metric,
      value,
      step,
    });
  }

  // Personalities

  async getPersonalities(): Promise<Array<{ name: string; description: string }>> {
    return this.request('GET', '/personalities');
  }

  async setPersonality(personality: string): Promise<{ status: string }> {
    return this.request('POST', '/personalities', { personality });
  }

  // Convenience Methods

  async quickGenerate(prompt: string): Promise<string> {
    const result = await this.generate({ prompt });
    return result.text;
  }

  async quickChat(message: string): Promise<string> {
    const result = await this.chat({
      messages: [{ role: 'user', content: message }],
    });
    return result.message.content;
  }
}

// React hook for easy integration
import { useState, useEffect, useCallback } from 'react';

export interface UseSloughGPTOptions extends SloughGPTConfig {
  autoConnect?: boolean;
}

export function useSloughGPT(options: UseSloughGPTOptions = {}) {
  const [client] = useState(() => new SloughGPTClient(options));
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);

  const checkHealth = useCallback(async () => {
    try {
      const status = await client.health();
      setHealth(status);
      setIsReady(status.status === 'healthy');
      setError(null);
    } catch (e) {
      setError(e as Error);
      setIsReady(false);
    }
  }, [client]);

  useEffect(() => {
    if (options.autoConnect !== false) {
      checkHealth();
    }
  }, [checkHealth, options.autoConnect]);

  const generate = useCallback(
    async (prompt: string, opts?: Partial<GenerateRequest>): Promise<string> => {
      setIsLoading(true);
      try {
        const result = await client.generate({ prompt, ...opts });
        return result.text;
      } finally {
        setIsLoading(false);
      }
    },
    [client]
  );

  const chat = useCallback(
    async (
      messages: ChatMessage[],
      opts?: Partial<ChatRequest>
    ): Promise<string> => {
      setIsLoading(true);
      try {
        const result = await client.chat({ messages, ...opts });
        return result.message.content;
      } finally {
        setIsLoading(false);
      }
    },
    [client]
  );

  return {
    client,
    isReady,
    isLoading,
    error,
    health,
    generate,
    chat,
    checkHealth,
  };
}

export default SloughGPTClient;
