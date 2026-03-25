import { useAuthStore } from './auth'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const getAuthHeaders = () => {
  const token = useAuthStore.getState().token
  return token ? { 'Authorization': `Bearer ${token}` } : {}
}

export interface BenchmarkResult {
  model_name: string
  num_parameters: number
  memory_mb: number
  inference_time_ms: number
  throughput_tokens_per_sec: number
  latency_p50_ms: number
  latency_p95_ms: number
  latency_p99_ms: number
  /** Present when the benchmark API returns an error payload instead of metrics */
  error?: string
}

/** `/benchmark/compare` returns either a top-level error or per-quantization rows (rows may embed `error`). */
export type BenchmarkCompareResponse =
  | { error: string }
  | Record<string, BenchmarkResult | { error: string }>

export interface ExportResult {
  status: string
  format: string
  files: Record<string, string>
  error?: string
}

export interface ModelExport {
  name: string
  path: string
  size_mb: number
}

export interface ModelList {
  models: ModelExport[]
}

export interface User {
  id: string
  username: string
  email: string
}

export interface Model {
  id: string
  name: string
  size: string
  type: string
  quantization?: string
  loaded?: boolean
}

export interface TrainingJob {
  id: string
  name: string
  model: string
  dataset: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  epochs?: number
  current_epoch?: number
  loss?: number
  data_path?: string
  output_checkpoint_stem?: string
  data_source?: 'legacy' | 'manifest' | 'ref'
  manifest?: { dataset_id: string; version: string }
  checkpoint?: string
  error?: string
}

export interface Dataset {
  id: string
  name: string
  size: string
  samples: number
  type: string
}

export interface GenerateRequest {
  prompt: string
  model?: string
  max_tokens?: number
  temperature?: number
  top_p?: number
}

export interface GenerateResponse {
  text: string
  model: string
  tokens_generated: number
}

export interface TrainDatasetRef {
  dataset_id: string
  version: string
  manifest_uri: string
}

export interface TrainingRequest {
  name: string
  model: string
  /** Exactly one of `dataset`, `manifest_uri`, or `dataset_ref` (server validates). */
  dataset?: string
  manifest_uri?: string
  dataset_ref?: TrainDatasetRef
  epochs?: number
  batch_size?: number
  learning_rate?: number
  n_embed?: number
  n_layer?: number
  n_head?: number
  block_size?: number
  max_steps?: number
}

export interface TrainResolveRequest {
  dataset?: string
  manifest_uri?: string
  dataset_ref?: TrainDatasetRef
}

export interface TrainResolveResponse {
  ok: boolean
  data_path: string
  output_checkpoint_stem: string
  data_source: 'legacy' | 'manifest' | 'ref'
  dataset?: string
  manifest?: { dataset_id: string; version: string }
}

/** POST /v1/infer (SloughGPT Standard v1) */
export interface StandardInferInput {
  prompt?: string
  messages?: Array<{ role: string; content: string }>
  context?: string
}

export interface StandardInferRequest {
  trace_id?: string
  tenant_id?: string
  model_id?: string
  task_type?: string
  mode: 'generate' | 'chat' | 'structured'
  safety?: { level?: string; policy_bundle?: string }
  output_schema_ref?: string
  retrieval?: Record<string, unknown>
  input: StandardInferInput
  generation?: {
    max_new_tokens?: number
    temperature?: number
    top_p?: number
    top_k?: number
    repetition_penalty?: number
    seed?: number
  }
}

export interface StandardInferResponse {
  trace_id: string
  model_id: string
  model_version: string
  task_type: string
  mode: string
  output: { text?: string; structured?: Record<string, unknown> }
  usage: {
    prompt_tokens?: number
    completion_tokens?: number
    total_tokens?: number
    latency_ms: number
  }
  safety_flags?: Array<{ rule_id: string; severity: string; message?: string }>
  citations?: Record<string, unknown>[]
}

export interface Experiment {
  id: string
  name: string
  status: string
  metrics: Record<string, number>
  params: Record<string, any>
  created_at: string
}

export interface Run {
  id: string
  experiment_id: string
  status: string
  metrics: Record<string, number>
  params: Record<string, any>
  created_at: string
}

const fetchWithAuth = async (url: string, options: RequestInit = {}) => {
  const authHeaders = getAuthHeaders()
  const headers: Record<string, string> = {}
  
  if (authHeaders.Authorization) {
    headers.Authorization = authHeaders.Authorization
  }
  
  const optionHeaders = options.headers as Record<string, string> | undefined
  if (optionHeaders) {
    Object.assign(headers, optionHeaders)
  }
  
  return fetch(url, { ...options, headers })
}

export const api = {
  async healthCheck() {
    const res = await fetch(`${API_URL}/health`)
    return res.json()
  },

  async login(username: string, password: string) {
    // Server expects query params, not JSON body
    const res = await fetch(`${API_URL}/auth/login?username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`, {
      method: 'POST',
    })
    return res.json()
  },

  async register(username: string, email: string, password: string) {
    // Server expects query params, not JSON body
    const res = await fetch(`${API_URL}/auth/register?username=${encodeURIComponent(username)}&email=${encodeURIComponent(email)}&password=${encodeURIComponent(password)}`, {
      method: 'POST',
    })
    return res.json()
  },

  async logout() {
    await fetchWithAuth(`${API_URL}/auth/logout`, { method: 'POST' })
  },

  async getCurrentUser(): Promise<User> {
    const res = await fetchWithAuth(`${API_URL}/auth/me`)
    return res.json()
  },

  async getModels(): Promise<Model[]> {
    const res = await fetch(`${API_URL}/models`)
    return res.json()
  },

  async loadModel(modelId: string) {
    const res = await fetchWithAuth(`${API_URL}/models/${modelId}/load`, {
      method: 'POST',
    })
    return res.json()
  },

  async unloadModel(modelId: string) {
    const res = await fetchWithAuth(`${API_URL}/models/${modelId}/unload`, {
      method: 'POST',
    })
    return res.json()
  },

  async generate(req: GenerateRequest): Promise<GenerateResponse> {
    const res = await fetchWithAuth(`${API_URL}/inference/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    return res.json()
  },

  generateStream(req: GenerateRequest, onToken: (token: string) => void, onDone: () => void) {
    const eventSource = new EventSource(`${API_URL}/inference/generate/stream`)
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.token) {
        onToken(data.token)
      }
      if (data.done) {
        eventSource.close()
        onDone()
      }
      if (data.error) {
        eventSource.close()
        onDone()
      }
    }
    
    eventSource.onerror = () => {
      eventSource.close()
      onDone()
    }
    
    return () => eventSource.close()
  },

  async getTrainingJobs(): Promise<TrainingJob[]> {
    const res = await fetchWithAuth(`${API_URL}/training/jobs`)
    const data = await res.json()
    return Array.isArray(data) ? data : (data as { jobs?: TrainingJob[] }).jobs ?? []
  },

  async getTrainingJob(jobId: string): Promise<TrainingJob> {
    const res = await fetchWithAuth(`${API_URL}/training/jobs/${jobId}`)
    return res.json()
  },

  async startTraining(req: TrainingRequest) {
    const res = await fetchWithAuth(`${API_URL}/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    return res.json()
  },

  async inferV1(body: StandardInferRequest): Promise<StandardInferResponse> {
    const res = await fetchWithAuth(`${API_URL}/v1/infer`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-SloughGPT-Standard': '1',
      },
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      const err = (await res.json().catch(() => ({}))) as { detail?: string | Array<{ msg?: string }> }
      const d = err.detail
      const msg =
        typeof d === 'string'
          ? d
          : Array.isArray(d)
            ? d.map((x) => x.msg || JSON.stringify(x)).join('; ')
            : res.statusText
      throw new Error(msg)
    }
    return res.json()
  },

  async resolveTrainingData(body: TrainResolveRequest): Promise<TrainResolveResponse> {
    const res = await fetchWithAuth(`${API_URL}/train/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      const err = (await res.json().catch(() => ({}))) as { detail?: string | Array<{ msg?: string }> }
      const d = err.detail
      const msg =
        typeof d === 'string'
          ? d
          : Array.isArray(d)
            ? d.map((x) => x.msg || JSON.stringify(x)).join('; ')
            : res.statusText
      throw new Error(msg)
    }
    return res.json()
  },

  async getDatasets(): Promise<Dataset[]> {
    const res = await fetch(`${API_URL}/datasets`)
    const body = (await res.json()) as {
      datasets?: Array<{
        id?: string
        name?: string
        path?: string
        size_bytes?: number
        size_formatted?: string
        size_kb?: number
        type?: string
      }>
    }
    const rows = body.datasets ?? []
    return rows.map((d) => ({
      id: String(d.id ?? d.name ?? ''),
      name: String(d.name ?? d.id ?? ''),
      size:
        d.size_formatted ??
        (typeof d.size_bytes === 'number'
          ? `${(d.size_bytes / 1024).toFixed(1)} KB`
          : typeof d.size_kb === 'number'
            ? `${d.size_kb.toFixed(1)} KB`
            : '—'),
      samples: 0,
      type: d.type ?? 'text',
    }))
  },

  async downloadDataset(datasetId: string) {
    const res = await fetchWithAuth(`${API_URL}/datasets/${datasetId}/download`, {
      method: 'POST',
    })
    return res.json()
  },

  async getExperiments(): Promise<Experiment[]> {
    const res = await fetchWithAuth(`${API_URL}/experiments`)
    return res.json()
  },

  async getExperiment(experimentId: string): Promise<Experiment> {
    const res = await fetchWithAuth(`${API_URL}/experiments/${experimentId}`)
    return res.json()
  },

  async getRuns(experimentId: string): Promise<Run[]> {
    const res = await fetchWithAuth(`${API_URL}/experiments/${experimentId}/runs`)
    return res.json()
  },

  async getRun(runId: string): Promise<Run> {
    const res = await fetchWithAuth(`${API_URL}/runs/${runId}`)
    return res.json()
  },

  async runBenchmark(prompt?: string, maxTokens?: number, numRuns?: number): Promise<BenchmarkResult> {
    const params = new URLSearchParams()
    if (prompt) params.append('prompt', prompt)
    if (maxTokens) params.append('max_new_tokens', maxTokens.toString())
    if (numRuns) params.append('num_runs', numRuns.toString())
    
    const res = await fetchWithAuth(`${API_URL}/benchmark/run?${params}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
    return res.json()
  },

  async calculatePerplexity(text: string): Promise<{ perplexity: number; text_length: number }> {
    const res = await fetchWithAuth(`${API_URL}/benchmark/perplexity?text=${encodeURIComponent(text)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
    return res.json()
  },

  async compareBenchmarks(): Promise<BenchmarkCompareResponse> {
    const res = await fetchWithAuth(`${API_URL}/benchmark/compare`, {
      method: 'GET',
    })
    return res.json()
  },

  async exportModel(outputPath: string, format?: string, includeTokenizer?: boolean): Promise<ExportResult> {
    const res = await fetchWithAuth(`${API_URL}/model/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        output_path: outputPath,
        format: format || 'sou',
        include_tokenizer: includeTokenizer ?? true,
      }),
    })
    return res.json()
  },

  async getExportFormats(): Promise<{ formats: Record<string, string> }> {
    const res = await fetch(`${API_URL}/model/export/formats`)
    return res.json()
  },

  async listModels(): Promise<ModelList> {
    const res = await fetch(`${API_URL}/models`)
    return res.json()
  },

  async logMetric(experimentId: string, metricName: string, value: number, step?: number): Promise<{ status: string }> {
    const params = new URLSearchParams()
    params.append('metric_name', metricName)
    params.append('value', value.toString())
    if (step !== undefined) params.append('step', step.toString())
    
    const res = await fetchWithAuth(`${API_URL}/experiments/${experimentId}/log_metric?${params}`, {
      method: 'POST',
    })
    return res.json()
  },

  async logParam(experimentId: string, paramName: string, value: string | number): Promise<{ status: string }> {
    const params = new URLSearchParams()
    params.append('param_name', paramName)
    params.append('value', String(value))
    
    const res = await fetchWithAuth(`${API_URL}/experiments/${experimentId}/log_param?${params}`, {
      method: 'POST',
    })
    return res.json()
  },

  subscribeToTraining(jobId: string, callback: (job: TrainingJob) => void) {
    const eventSource = new EventSource(`${API_URL}/training/jobs/${jobId}/stream`)
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data)
      callback(data)
    }
    
    eventSource.onerror = () => {
      eventSource.close()
    }
    
    return () => eventSource.close()
  },
}
