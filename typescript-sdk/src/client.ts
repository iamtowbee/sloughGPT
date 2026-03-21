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
  model: { type: string; loaded: boolean };
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
    public response?: unknown
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
    body?: unknown
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
    } catch (error: unknown) {
      clearTimeout(timeoutId);
      if ((error as { name?: string }).name === 'AbortError') {
        throw new SloughGPTError('Request timeout', 408);
      }
      throw error;
    }
  }

  async health(): Promise<HealthStatus> {
    return this.request<HealthStatus>('GET', '/health');
  }

  async liveness(): Promise<{ status: string }> {
    return this.request('GET', '/health/live');
  }

  async readiness(): Promise<{ status: string; model_loaded: boolean }> {
    return this.request('GET', '/health/ready');
  }

  async info(): Promise<SystemInfo> {
    return this.request<SystemInfo>('GET', '/info');
  }

  async generate(request: GenerateRequest): Promise<GenerationResult> {
    this.log('info', `Generating: "${request.prompt.slice(0, 50)}..."`);
    return this.request<GenerationResult>('POST', '/generate', {
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
      throw new SloughGPTError(`HTTP ${response.status}`, response.status);
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

  async chat(request: ChatRequest): Promise<ChatResult> {
    this.log('info', `Chat: ${request.messages.length} messages`);
    return this.request<ChatResult>('POST', '/chat/completions', {
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
      throw new SloughGPTError(`HTTP ${response.status}`, response.status);
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

  async batchGenerate(prompts: string[]): Promise<GenerationResult[]> {
    return this.request<GenerationResult[]>('POST', '/generate/batch', { prompts });
  }

  async listModels(): Promise<ModelInfo[]> {
    return this.request<ModelInfo[]>('GET', '/models');
  }

  async getModel(modelId: string): Promise<ModelInfo> {
    return this.request<ModelInfo>('GET', `/models/${modelId}`);
  }

  async loadModel(modelId: string): Promise<{ status: string }> {
    return this.request('POST', '/models/load', { model_id: modelId });
  }

  async metrics(): Promise<MetricsData> {
    return this.request<MetricsData>('GET', '/metrics');
  }

  async startTraining(config: {
    model_name: string;
    dataset_id: string;
    epochs?: number;
    batch_size?: number;
    learning_rate?: number;
  }): Promise<TrainingJob> {
    return this.request<TrainingJob>('POST', '/training/start', config);
  }

  async getTrainingStatus(jobId: string): Promise<TrainingJob> {
    return this.request<TrainingJob>('GET', `/training/jobs/${jobId}`);
  }

  async listTrainingJobs(): Promise<TrainingJob[]> {
    return this.request<TrainingJob[]>('GET', '/training/jobs');
  }

  async createExperiment(
    name: string,
    description?: string
  ): Promise<Experiment> {
    return this.request<Experiment>('POST', '/experiments', { name, description });
  }

  async listExperiments(): Promise<Experiment[]> {
    return this.request<Experiment[]>('GET', '/experiments');
  }

  async getExperiment(experimentId: string): Promise<Experiment> {
    return this.request<Experiment>('GET', `/experiments/${experimentId}`);
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

  async getPersonalities(): Promise<Array<{ name: string; description: string }>> {
    return this.request('GET', '/personalities');
  }

  async setPersonality(personality: string): Promise<{ status: string }> {
    return this.request('POST', '/personalities', { personality });
  }

  async listDatasets(): Promise<unknown[]> {
    return this.request('GET', '/datasets');
  }

  async getDataset(datasetId: string): Promise<unknown> {
    return this.request('GET', `/datasets/${datasetId}`);
  }

  async getDatasetStats(datasetId: string): Promise<unknown> {
    return this.request('GET', `/datasets/${datasetId}/stats`);
  }

  async inferenceGenerate(request: GenerateRequest): Promise<GenerationResult> {
    return this.request<GenerationResult>('POST', '/inference/generate', request);
  }

  async *inferenceStream(
    request: GenerateRequest
  ): AsyncGenerator<string, void, unknown> {
    const url = `${this.baseUrl}/inference/generate/stream`;
    const response = await fetch(url, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new SloughGPTError(`HTTP ${response.status}`, response.status);
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
          if (data && data !== '[DONE]') yield data;
        }
      }
    }
  }

  async inferenceStats(): Promise<unknown> {
    return this.request('GET', '/inference/stats');
  }

  async inferenceBatch(prompts: string[]): Promise<unknown> {
    return this.request('POST', '/inference/batch', { prompts });
  }

  async registerModel(model: {
    name: string;
    model_type?: string;
    description?: string;
    config?: Record<string, unknown>;
  }): Promise<unknown> {
    return this.request('POST', '/registry/models', model);
  }

  async listRegisteredModels(): Promise<unknown[]> {
    return this.request('GET', '/registry/models');
  }

  async getRegisteredModel(modelId: string): Promise<unknown> {
    return this.request('GET', `/registry/models/${modelId}`);
  }

  async unregisterModel(modelId: string): Promise<unknown> {
    return this.request('DELETE', `/registry/models/${modelId}`);
  }

  async recordToRegistry(
    modelId: string,
    metrics: {
      latency_ms?: number;
      tokens_generated?: number;
      cache_hit?: boolean;
      error?: string;
    }
  ): Promise<unknown> {
    return this.request('POST', `/registry/models/${modelId}/record`, metrics);
  }

  async getRegistryMetrics(modelId: string): Promise<unknown> {
    return this.request('GET', `/registry/models/${modelId}/metrics`);
  }

  async getBestRegisteredModel(criteria: {
    metric?: string;
    order?: 'asc' | 'desc';
    model_type?: string;
  }): Promise<unknown> {
    return this.request('GET', '/registry/best', criteria);
  }

  async getRegistryStats(): Promise<unknown> {
    return this.request('GET', '/registry/stats');
  }

  async runBenchmark(config: {
    model?: string;
    num_samples?: number;
    dataset?: string;
  }): Promise<unknown> {
    return this.request('POST', '/benchmark/run', config);
  }

  async runPerplexityBenchmark(config: {
    model?: string;
    dataset?: string;
  }): Promise<unknown> {
    return this.request('POST', '/benchmark/perplexity', config);
  }

  async compareBenchmarks(modelIds: string[]): Promise<unknown> {
    return this.request('GET', `/benchmark/compare?models=${modelIds.join(',')}`);
  }

  async clearCache(): Promise<unknown> {
    return this.request('DELETE', '/cache');
  }

  async cacheStats(): Promise<unknown> {
    return this.request('GET', '/cache/stats');
  }

  async rateLimitStatus(): Promise<unknown> {
    return this.request('GET', '/rate-limit/status');
  }

  async rateLimitCheck(resource: string, cost?: number): Promise<unknown> {
    return this.request('GET', '/rate-limit/check', { resource, cost });
  }

  async getAuditLog(): Promise<unknown> {
    return this.request('GET', '/security/audit');
  }

  async getSecurityKeys(): Promise<unknown> {
    return this.request('GET', '/security/keys');
  }

  async getToken(username: string, password: string): Promise<unknown> {
    return this.request('POST', '/auth/token', { username, password });
  }

  async refreshToken(refreshToken: string): Promise<unknown> {
    return this.request('POST', '/auth/refresh', { refresh_token: refreshToken });
  }

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

export default SloughGPTClient;
