import { useAuthStore } from './auth'
import { PUBLIC_API_URL } from './config'
import { devDebug } from './dev-log'

const API_URL = PUBLIC_API_URL

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
  /** Mirrors API `source` (e.g. local / huggingface). */
  type: string
  quantization?: string
  loaded?: boolean
  description?: string
  tags?: string[]
  /** Numeric size when the API provides `size_mb`. */
  size_mb?: number
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
  /** Last reported optimizer step while running (from API trainer progress). */
  global_step?: number
  /** Last batch training loss (running). */
  train_loss?: number
  /** Last eval loss when an eval ran during training (running). */
  eval_loss?: number
  loss?: number
  data_path?: string
  output_checkpoint_stem?: string
  data_source?: 'legacy' | 'manifest' | 'ref'
  manifest?: { dataset_id: string; version: string }
  /** Path to trainer output when set; native `step_*.pt` embeds char vocab for `cli.py eval` — see repo `docs/policies/CONTRIBUTING.md` (*Checkpoint vocabulary*). */
  checkpoint?: string
  error?: string
}

export interface Dataset {
  id: string
  name: string
  size: string
  samples: number
  type: string
  /** Present when the API includes a path field. */
  path?: string
}

export type ImportSource = 'github' | 'huggingface' | 'url' | 'local' | 'kaggle'

export interface DatasetPreview {
  dataset_id: string
  samples: DatasetSample[]
  total_samples: number
  total_chars: number
  languages: Record<string, number>
}

export interface DatasetSample {
  path: string
  language: string
  content: string
  size: number
}

export interface GitHubImportRequest {
  url: string
  name: string
  extensions?: string[]
  max_files?: number
}

export interface HuggingFaceImportRequest {
  dataset_id: string
  name?: string
}

export interface URLImportRequest {
  url: string
  name: string
}

export interface LocalImportRequest {
  path: string
  name: string
  extensions?: string[]
}

export interface KaggleImportRequest {
  dataset: string
  name?: string
}

export interface ImportResponse {
  success: boolean
  dataset_id: string
  message: string
  output_path: string
}

export interface GitHubRepo {
  id: string
  name: string
  full_name: string
  description: string | null
  stars: number
  url: string
  language: string | null
}

export interface GenerateRequest {
  prompt: string
  model?: string
  /** Legacy alias; server expects `max_new_tokens`. */
  max_tokens?: number
  max_new_tokens?: number
  temperature?: number
  top_p?: number
  top_k?: number
  personality?: string
}

export interface GenerateResponse {
  text: string
  model: string
  tokens_generated: number
}

/** `POST /chat` and `/chat/stream` — OpenAI-style message list (see `ChatRequest` in `apps/api/server/main.py`). */
export interface ChatMessagePayload {
  role: string
  content: string
}

export interface ChatCompletionRequest {
  messages: ChatMessagePayload[]
  model?: string
  max_new_tokens?: number
  temperature?: number
  top_p?: number
  top_k?: number
}

function buildChatPayload(req: ChatCompletionRequest) {
  return {
    messages: req.messages,
    max_new_tokens: req.max_new_tokens ?? 100,
    temperature: req.temperature ?? 0.8,
    top_p: req.top_p ?? 0.9,
    top_k: req.top_k ?? 50,
    ...(req.model != null && req.model !== '' ? { model: req.model } : {}),
  }
}

/** FastAPI may return `detail` as a string, object, or Pydantic validation list — normalize for errors. */
function formatFastApiDetail(detail: unknown): string | null {
  if (detail == null) return null
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === 'object' && 'msg' in item) {
          const m = (item as { msg?: string }).msg
          return typeof m === 'string' ? m : JSON.stringify(item)
        }
        return typeof item === 'string' ? item : JSON.stringify(item)
      })
      .join('; ')
  }
  if (typeof detail === 'object' && 'msg' in (detail as object)) {
    const m = (detail as { msg?: string }).msg
    return typeof m === 'string' ? m : null
  }
  return null
}

/** `GET /health` — drives UI for inference readiness (see `apps/api/server/main.py`). */
export interface ApiHealth {
  status: string
  model_loaded: boolean
  model_type: string
  soul_engine_active?: boolean
  soul_name?: string | null
}

/** Body shape for `POST /inference/generate` and `/inference/generate/stream` (see server `GenerateRequest`). */
function buildInferenceGeneratePayload(req: GenerateRequest) {
  return {
    prompt: req.prompt,
    max_new_tokens: req.max_new_tokens ?? req.max_tokens ?? 100,
    temperature: req.temperature ?? 0.8,
    top_p: req.top_p ?? 0.9,
    top_k: req.top_k ?? 50,
    ...(req.personality != null && req.personality !== ''
      ? { personality: req.personality }
      : {}),
  }
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
  log_interval?: number
  eval_interval?: number
  dropout?: number
  weight_decay?: number
  gradient_accumulation_steps?: number
  max_grad_norm?: number
  use_mixed_precision?: boolean
  mixed_precision_dtype?: string
  warmup_steps?: number
  min_lr?: number
  scheduler?: string
  use_lora?: boolean
  lora_rank?: number
  lora_alpha?: number
  checkpoint_dir?: string
  checkpoint_interval?: number
  save_best_only?: boolean
  max_checkpoints?: number
  /** Explicit device on the training host (`auto` is omitted by `cli.py train --api`). */
  device?: string
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
  /**
   * Lightweight probe for whether the API process has weights/tokenizers for `/inference/*`.
   * Returns `null` when the server is unreachable or returns a non-OK status.
   */
  async getHealth(): Promise<ApiHealth | null> {
    try {
      const res = await fetch(`${API_URL}/health`, { cache: 'no-store' })
      if (!res.ok) return null
      return (await res.json()) as ApiHealth
    } catch {
      return null
    }
  },

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
    if (!res.ok) {
      throw new Error(`GET /models failed (${res.status})`)
    }
    const body = (await res.json()) as {
      models?: Array<{
        id?: string
        name?: string
        source?: string
        size_mb?: number
        description?: string
        tags?: unknown
      }>
    }
    const rows = body.models ?? []
    return rows.map((m) => {
      const tags = Array.isArray(m.tags) ? m.tags.filter((t): t is string => typeof t === 'string') : undefined
      const sizeMb = typeof m.size_mb === 'number' ? m.size_mb : undefined
      return {
        id: String(m.id ?? m.name ?? ''),
        name: String(m.name ?? m.id ?? ''),
        size: typeof sizeMb === 'number' ? `${sizeMb.toFixed(2)} MB` : '—',
        type: m.source ?? 'unknown',
        description: typeof m.description === 'string' ? m.description : undefined,
        tags,
        size_mb: sizeMb,
      }
    })
  },

  /**
   * `POST /models/load` (apps/api/server `LoadModelRequest`) — loads HF weights into the API process.
   * Not to be confused with a RESTful `/models/{id}/load` path (not used by this server).
   */
  async loadModel(
    modelId: string,
    opts?: { mode?: string; device?: string },
  ): Promise<{
    status?: string
    model?: string
    mode?: string
    device?: string
    effective_device?: string | null
    model_type?: string
    error?: string
  }> {
    const res = await fetchWithAuth(`${API_URL}/models/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_id: modelId,
        mode: opts?.mode ?? 'local',
        device: opts?.device ?? 'auto',
      }),
    })
    const body = (await res.json().catch(() => ({}))) as {
      status?: string
      error?: string
      model?: string
    }
    if (!res.ok) {
      throw new Error(typeof body.error === 'string' ? body.error : `HTTP ${res.status}`)
    }
    if (body.status === 'error') {
      throw new Error(typeof body.error === 'string' ? body.error : 'Load failed')
    }
    return body
  },

  /** No `unload` route on `apps/api/server/main.py` yet; callers should handle failure. */
  async unloadModel(modelId: string) {
    const res = await fetchWithAuth(`${API_URL}/models/${encodeURIComponent(modelId)}/unload`, {
      method: 'POST',
    })
    if (res.status === 404) {
      throw new Error('Model unload is not implemented on this API build')
    }
    return res.json()
  },

  async generate(req: GenerateRequest): Promise<GenerateResponse> {
    const res = await fetchWithAuth(`${API_URL}/inference/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildInferenceGeneratePayload(req)),
    })
    const body = (await res.json().catch(() => ({}))) as Partial<GenerateResponse> & {
      error?: string
      detail?: unknown
    }

    if (!res.ok) {
      const msg =
        typeof body.error === 'string'
          ? body.error
          : typeof body.detail === 'string'
            ? body.detail
            : `HTTP ${res.status}`
      throw new Error(msg)
    }

    // FastAPI often returns 200 with `{ error: "Model not loaded", text: "" }` when no weights are loaded.
    if (typeof body.error === 'string' && body.error.trim() !== '') {
      throw new Error(body.error)
    }

    const text = typeof body.text === 'string' ? body.text : ''
    if (!text.trim()) {
      throw new Error(
        'Model returned no text. Start the API, load a model (e.g. via /models), then try again.',
      )
    }

    return {
      text,
      model: typeof body.model === 'string' ? body.model : 'unknown',
      tokens_generated: typeof body.tokens_generated === 'number' ? body.tokens_generated : 0,
    }
  },

  /**
   * Stream tokens from `POST /inference/generate/stream` (SSE). Uses fetch + body so prompt
   * and generation params are sent; EventSource cannot POST a JSON body.
   */
  generateStream(req: GenerateRequest, onToken: (token: string) => void, onDone: () => void) {
    const ac = new AbortController()
    let settled = false
    const finish = () => {
      if (settled) return
      settled = true
      onDone()
    }

    ;(async () => {
      try {
        const res = await fetchWithAuth(`${API_URL}/inference/generate/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(buildInferenceGeneratePayload(req)),
          signal: ac.signal,
        })
        if (!res.ok) {
          try {
            const raw = await res.text()
            const j = JSON.parse(raw) as { detail?: unknown; error?: string }
            const msg =
              typeof j.error === 'string'
                ? j.error
                : typeof j.detail === 'string'
                  ? j.detail
                  : raw.slice(0, 200)
            if (msg) {
              devDebug('[api] inference stream HTTP error:', res.status, msg)
            }
          } catch {
            /* ignore */
          }
          finish()
          return
        }
        if (!res.body) {
          finish()
          return
        }
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''
          for (const line of lines) {
            const trimmed = line.trimEnd()
            if (!trimmed.startsWith('data:')) continue
            const payload = trimmed.slice(5).trim()
            if (!payload || payload === '[DONE]') continue
            try {
              const data = JSON.parse(payload) as {
                token?: string
                done?: boolean
                error?: string
              }
              if (data.error) {
                finish()
                return
              }
              if (data.token) onToken(data.token)
              if (data.done) {
                finish()
                return
              }
            } catch {
              /* ignore partial / malformed frames */
            }
          }
        }
        finish()
      } catch {
        finish()
      }
    })()

    return () => {
      ac.abort()
      finish()
    }
  },

  /**
   * Non-streaming chat completion (`POST /chat`). Same engine as `/inference/generate`, chat-formatted prompt.
   */
  async chat(req: ChatCompletionRequest): Promise<GenerateResponse> {
    const res = await fetchWithAuth(`${API_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildChatPayload(req)),
    })
    const body = (await res.json().catch(() => ({}))) as Partial<GenerateResponse> & {
      error?: string
      detail?: unknown
    }

    if (!res.ok) {
      const msg =
        typeof body.error === 'string' && body.error.trim() !== ''
          ? body.error
          : formatFastApiDetail(body.detail) ?? `HTTP ${res.status}`
      throw new Error(msg)
    }

    if (typeof body.error === 'string' && body.error.trim() !== '') {
      throw new Error(body.error)
    }

    const text = typeof body.text === 'string' ? body.text : ''
    if (!text.trim()) {
      throw new Error(
        'Model returned no text. Start the API, load a model (e.g. via /models), then try again.',
      )
    }

    return {
      text,
      model: typeof body.model === 'string' ? body.model : 'unknown',
      tokens_generated: typeof body.tokens_generated === 'number' ? body.tokens_generated : 0,
    }
  },

  /**
   * Stream tokens from `POST /chat/stream` (SSE). Message history is sent in the JSON body.
   */
  chatStream(
    req: ChatCompletionRequest,
    onToken: (token: string) => void,
    onDone: () => void,
  ) {
    const ac = new AbortController()
    let settled = false
    const finish = () => {
      if (settled) return
      settled = true
      onDone()
    }

    ;(async () => {
      try {
        const res = await fetchWithAuth(`${API_URL}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(buildChatPayload(req)),
          signal: ac.signal,
        })
        if (!res.ok) {
          try {
            const raw = await res.text()
            const j = JSON.parse(raw) as { detail?: unknown; error?: string }
            const msg =
              typeof j.error === 'string'
                ? j.error
                : typeof j.detail === 'string'
                  ? j.detail
                  : raw.slice(0, 200)
            if (msg) {
              devDebug('[api] chat stream HTTP error:', res.status, msg)
            }
          } catch {
            /* ignore */
          }
          finish()
          return
        }
        if (!res.body) {
          finish()
          return
        }
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''
          for (const line of lines) {
            const trimmed = line.trimEnd()
            if (!trimmed.startsWith('data:')) continue
            const payload = trimmed.slice(5).trim()
            if (!payload || payload === '[DONE]') continue
            try {
              const data = JSON.parse(payload) as {
                token?: string
                done?: boolean
                error?: string
              }
              if (data.error) {
                finish()
                return
              }
              if (data.token) onToken(data.token)
              if (data.done) {
                finish()
                return
              }
            } catch {
              /* ignore partial / malformed frames */
            }
          }
        }
        finish()
      } catch {
        finish()
      }
    })()

    return () => {
      ac.abort()
      finish()
    }
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
    if (!res.ok) {
      throw new Error(`GET /datasets failed (${res.status})`)
    }
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
      path: typeof d.path === 'string' ? d.path : undefined,
    }))
  },

  async downloadDataset(datasetId: string) {
    const res = await fetchWithAuth(`${API_URL}/datasets/${datasetId}/download`, {
      method: 'POST',
    })
    return res.json()
  },

  async previewDataset(datasetId: string, limit = 10): Promise<DatasetPreview> {
    const res = await fetchWithAuth(`${API_URL}/datasets/${datasetId}/preview?limit=${limit}`)
    if (!res.ok) {
      throw new Error(`GET /datasets/${datasetId}/preview failed (${res.status})`)
    }
    return res.json()
  },

  async importFromGitHub(request: GitHubImportRequest): Promise<ImportResponse> {
    const res = await fetchWithAuth(`${API_URL}/datasets/import/github`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Import failed' }))
      throw new Error(error.detail || `Import failed (${res.status})`)
    }
    return res.json()
  },

  async importFromHuggingFace(request: HuggingFaceImportRequest): Promise<ImportResponse> {
    const res = await fetchWithAuth(`${API_URL}/datasets/import/huggingface`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Import failed' }))
      throw new Error(error.detail || `Import failed (${res.status})`)
    }
    return res.json()
  },

  async importFromURL(request: URLImportRequest): Promise<ImportResponse> {
    const res = await fetchWithAuth(`${API_URL}/datasets/import/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Import failed' }))
      throw new Error(error.detail || `Import failed (${res.status})`)
    }
    return res.json()
  },

  async importFromLocal(request: LocalImportRequest): Promise<ImportResponse> {
    const res = await fetchWithAuth(`${API_URL}/datasets/import/local`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Import failed' }))
      throw new Error(error.detail || `Import failed (${res.status})`)
    }
    return res.json()
  },

  async importFromKaggle(request: KaggleImportRequest): Promise<ImportResponse> {
    const res = await fetchWithAuth(`${API_URL}/datasets/import/kaggle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Import failed' }))
      throw new Error(error.detail || `Import failed (${res.status})`)
    }
    return res.json()
  },

  async deleteDataset(datasetId: string): Promise<{ success: boolean; message: string }> {
    const res = await fetchWithAuth(`${API_URL}/datasets/${datasetId}`, {
      method: 'DELETE',
    })
    if (!res.ok) {
      throw new Error(`Delete failed (${res.status})`)
    }
    return res.json()
  },

  async exportDataset(datasetId: string, format = 'json'): Promise<{
    dataset_id: string
    format: string
    total_samples: number
    content: string
    size_bytes: number
  }> {
    const res = await fetchWithAuth(`${API_URL}/datasets/${datasetId}/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ format }),
    })
    if (!res.ok) {
      throw new Error(`Export failed (${res.status})`)
    }
    return res.json()
  },

  async searchGitHubRepos(query: string, limit = 10): Promise<{ repos: GitHubRepo[] }> {
    const res = await fetchWithAuth(`${API_URL}/datasets/search/github?query=${encodeURIComponent(query)}&limit=${limit}`)
    if (!res.ok) {
      throw new Error(`GitHub search failed (${res.status})`)
    }
    return res.json()
  },

  async validateDataset(datasetId: string): Promise<{
    dataset_id: string
    valid: boolean
    issues: string[]
    warnings: string[]
    stats: Record<string, unknown>
  }> {
    const res = await fetchWithAuth(`${API_URL}/datasets/${datasetId}/validate`, { method: 'POST' })
    if (!res.ok) {
      throw new Error(`Validation failed (${res.status})`)
    }
    return res.json()
  },

  async combineDatasets(sourceIds: string[], name: string): Promise<ImportResponse> {
    const res = await fetchWithAuth(`${API_URL}/datasets/combine`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_ids: sourceIds, name }),
    })
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Combine failed' }))
      throw new Error(error.detail || `Combine failed (${res.status})`)
    }
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
