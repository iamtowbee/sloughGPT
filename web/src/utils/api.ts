const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export interface ApiResponse<T> {
  data?: T
  error?: string
  status: number
}

// Chat types
export interface ChatMessageRequest {
  message: string
  model?: string
  conversation_id?: string
  temperature?: number
  max_tokens?: number
}

export interface ChatMessageResponse {
  conversation_id: string
  message: {
    role: string
    content: string
    timestamp: string
  }
  model: string
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}

// Model types
export interface Model {
  id: string
  name: string
  provider: string
  status: string
  description?: string
  context_length?: number
  pricing?: {
    prompt: number
    completion: number
  }
}

// Dataset types
export interface Dataset {
  name: string
  path: string
  size: number
  created_at: string
  updated_at: string
  has_train: boolean
  has_val: boolean
  has_meta: boolean
  description?: string
}

export interface DatasetCreate {
  name: string
  content: string
  description?: string
}

// Conversation types
export interface Conversation {
  id: string
  name: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface ConversationDetail {
  id: string
  messages: Array<{
    role: string
    content: string
    timestamp: string
  }>
  metadata: Conversation
}

// Training types
export interface TrainingJob {
  id: string
  status: string
  dataset_name: string
  model_id: string
  progress: number
  current_epoch: number
  total_epochs: number
  loss: number
  started_at?: string
  completed_at?: string
  error?: string
}

export interface TrainingLog {
  epoch: number
  loss: number
  timestamp: string
}

export interface TrainingConfig {
  dataset_name: string
  model_id: string
  epochs: number
  batch_size: number
  learning_rate: number
  vocab_size: number
  n_embed: number
  n_layer: number
  n_head: number
  optimizer?: string
  scheduler?: string
  validation_split?: number
  early_stopping_patience?: number
  save_checkpoint_every?: number
  gradient_clip?: number
  warmup_steps?: number
  weight_decay?: number
}

export interface TrainingConfig {
  dataset_name: string
  model_id: string
  epochs: number
  batch_size: number
  learning_rate: number
  vocab_size: number
  n_embed: number
  n_layer: number
  n_head: number
}

// System types
export interface SystemMetrics {
  cpu_percent: number
  memory_percent: number
  memory_used_mb: number
  memory_total_mb: number
  disk_percent: number
  disk_used_gb: number
  disk_total_gb: number
  network_sent_mb: number
  network_recv_mb: number
  timestamp: string
}

export interface HealthStatus {
  status: string
  version: string
  uptime_seconds: number
  timestamp: string
  services: Record<string, string>
}

export interface SystemInfo {
  python_version: string
  platform: string
  cpu_count: number
  total_memory_gb: number
  disk_total_gb: number
  conversations_count: number
  datasets_count: number
  training_jobs_count: number
}

// Generation types
export interface GenerateRequest {
  prompt: string
  model: string
  max_length: number
  temperature: number
  top_k?: number
}

export interface GenerateResponse {
  text: string
  model: string
  tokens_generated: number
  processing_time_ms: number
}

class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const url = `${this.baseUrl}${endpoint}`
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
        ...options.headers
      }

      const response = await fetch(url, {
        ...options,
        headers
      })

      const data = await response.json()

      if (!response.ok) {
        return {
          error: data.detail || 'An error occurred',
          status: response.status
        }
      }

      return { data, status: response.status }
    } catch (error) {
      return {
        error: error instanceof Error ? error.message : 'Network error',
        status: 0
      }
    }
  }

  // =====================
  // Health & System
  // =====================
  
  async healthCheck() {
    return this.request<HealthStatus>('/health')
  }

  async getMetrics() {
    return this.request<SystemMetrics>('/metrics')
  }

  async getSystemInfo() {
    return this.request<SystemInfo>('/info')
  }

  // =====================
  // Conversations
  // =====================
  
  async createConversation(name?: string) {
    return this.request<Conversation>('/conversations', {
      method: 'POST',
      body: JSON.stringify({ name })
    })
  }

  async listConversations() {
    return this.request<{ conversations: Conversation[] }>('/conversations')
  }

  async getConversation(conversationId: string) {
    return this.request<ConversationDetail>(`/conversations/${conversationId}`)
  }

  async deleteConversation(conversationId: string) {
    return this.request<{ status: string }>(`/conversations/${conversationId}`, {
      method: 'DELETE'
    })
  }

  // =====================
  // Chat
  // =====================
  
  async sendChatMessage(
    message: string, 
    model: string = 'gpt-3.5-turbo', 
    conversationId?: string,
    temperature: number = 0.7,
    maxTokens: number = 1000
  ) {
    return this.request<ChatMessageResponse>('/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message, 
        model, 
        conversation_id: conversationId,
        temperature,
        max_tokens: maxTokens
      })
    })
  }

  // =====================
  // Models
  // =====================
  
  async getModels() {
    return this.request<{ models: Model[] }>('/models')
  }

  async getModel(modelId: string) {
    return this.request<Model>(`/models/${modelId}`)
  }

  // =====================
  // Datasets
  // =====================
  
  async listDatasets() {
    return this.request<{ datasets: Dataset[] }>('/datasets')
  }

  async createDataset(name: string, content: string, description?: string) {
    return this.request<Dataset>('/datasets', {
      method: 'POST',
      body: JSON.stringify({ name, content, description })
    })
  }

  async getDataset(name: string) {
    return this.request<Dataset>(`/datasets/${name}`)
  }

  async deleteDataset(name: string) {
    return this.request<{ status: string }>(`/datasets/${name}`, {
      method: 'DELETE'
    })
  }

  // =====================
  // Training
  // =====================
  
  async listTrainingJobs() {
    return this.request<{ jobs: TrainingJob[] }>('/training')
  }

  async createTrainingJob(config: TrainingConfig) {
    return this.request<TrainingJob>('/training', {
      method: 'POST',
      body: JSON.stringify(config)
    })
  }

  async getTrainingJob(jobId: string) {
    return this.request<TrainingJob>(`/training/${jobId}`)
  }

  async cancelTrainingJob(jobId: string) {
    return this.request<{ status: string }>(`/training/${jobId}`, {
      method: 'DELETE'
    })
  }

  async restartTrainingJob(jobId: string) {
    return this.request<{ new_job_id: string; original_job_id: string; status: string }>(`/training/${jobId}/restart`, {
      method: 'POST'
    })
  }

  async getTrainingLogs(jobId: string) {
    return this.request<{ job_id: string; logs: TrainingLog[] }>(`/training/${jobId}/logs`)
  }

  async getTrainingHistory(jobId: string) {
    return this.request<{ job_id: string; history: any[] }>(`/training/${jobId}/history`)
  }

  // =====================
  // Generation
  // =====================
  
  async generate(
    prompt: string, 
    model: string = 'nanogpt',
    maxLength: number = 100,
    temperature: number = 0.8,
    topK?: number
  ) {
    return this.request<GenerateResponse>('/generate', {
      method: 'POST',
      body: JSON.stringify({ 
        prompt, 
        model, 
        max_length: maxLength, 
        temperature,
        top_k: topK
      })
    })
  }

  // =====================
  // Root
  // =====================
  
  async getRoot() {
    return this.request<{ message: string; version: string; docs: string }>('/')
  }
}

export const api = new ApiClient()
export default api
