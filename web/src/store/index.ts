import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  model?: string
}

export interface Conversation {
  id: string
  name: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface Model {
  id: string
  name: string
  provider: string
  status: string
  description?: string
  context_length?: number
  pricing?: { prompt: number; completion: number }
}

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

export interface TrainingJob {
  id: string
  status: string
  dataset_name: string
  model_id: string
  progress: number
  current_epoch: number
  total_epochs: number
  loss: number
  val_loss?: number
  best_loss?: number
  learning_rate?: number
  batch_size?: number
  optimizer?: string
  scheduler?: string
  gradient_clip?: number
  weight_decay?: number
  steps_per_epoch?: number
  total_steps?: number
  started_at?: string
  completed_at?: string
  error?: string
  checkpoints?: Checkpoint[]
  metrics_history?: MetricHistory[]
  early_stopping_patience?: number
  epochs_without_improvement?: number
}

export interface Checkpoint {
  epoch: number
  loss: number
  val_loss?: number
  best_loss?: number
  timestamp: string
  path: string
}

export interface MetricHistory {
  epoch: number
  loss: number
  val_loss?: number
  learning_rate?: number
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

export interface TrainingLog {
  epoch: number
  loss: number
  timestamp: string
}

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

export interface AppState {
  // Chat
  messages: ChatMessage[]
  conversations: Conversation[]
  currentConversationId: string
  selectedModel: string
  
  // Data
  datasets: Dataset[]
  models: Model[]
  trainingJobs: TrainingJob[]
  
  // System
  metrics: SystemMetrics | null
  isConnected: boolean
  
  // UI
  activeTab: 'home' | 'chat' | 'datasets' | 'models' | 'training' | 'monitoring'
  isLoading: boolean
  error?: string
  
  // Actions - Chat
  addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => void
  setConversations: (conversations: Conversation[]) => void
  addConversation: (conversation: Conversation) => void
  setCurrentConversationId: (id: string) => void
  setSelectedModel: (modelId: string) => void
  
  // Actions - Data
  setDatasets: (datasets: Dataset[]) => void
  addDataset: (dataset: Dataset) => void
  removeDataset: (name: string) => void
  setModels: (models: Model[]) => void
  setTrainingJobs: (jobs: TrainingJob[]) => void
  addTrainingJob: (job: TrainingJob) => void
  updateTrainingJob: (job: TrainingJob) => void
  
  // Actions - System
  setMetrics: (metrics: SystemMetrics) => void
  setConnected: (connected: boolean) => void
  
  // Actions - UI
  setActiveTab: (tab: AppState['activeTab']) => void
  setLoading: (loading: boolean) => void
  setError: (error?: string) => void
  clearError: () => void
}

export const useStore = create<AppState>(
  persist(
    (set, get) => ({
      // Initial state
      messages: [],
      conversations: [],
      currentConversationId: 'conv_default',
      selectedModel: 'gpt-3.5-turbo',
      datasets: [],
      models: [],
      trainingJobs: [],
      metrics: null,
      isConnected: false,
      activeTab: 'home',
      isLoading: false,
      
      // Chat actions
      addMessage: (message) => {
        set((state) => ({
          messages: [
            ...state.messages,
            {
              ...message,
              id: `${state.currentConversationId}-${Date.now()}`,
              timestamp: new Date()
            }
          ]
        }))
      },
      
      setConversations: (conversations) => set({ conversations }),
      
      addConversation: (conversation) => {
        set((state) => ({
          conversations: [conversation, ...state.conversations]
        }))
      },
      
      setCurrentConversationId: (id) => set({ 
        currentConversationId: id,
        messages: [] // Clear messages when switching conversations
      }),
      
      setSelectedModel: (modelId) => set({ selectedModel: modelId }),
      
      // Data actions
      setDatasets: (datasets) => set({ datasets }),
      
      addDataset: (dataset) => {
        set((state) => ({
          datasets: [dataset, ...state.datasets]
        }))
      },
      
      removeDataset: (name) => {
        set((state) => ({
          datasets: state.datasets.filter(d => d.name !== name)
        }))
      },
      
      setModels: (models) => set({ models }),
      
      setTrainingJobs: (jobs) => set({ trainingJobs: jobs }),
      
      addTrainingJob: (job) => {
        set((state) => ({
          trainingJobs: [job, ...state.trainingJobs]
        }))
      },
      
      updateTrainingJob: (job) => {
        set((state) => ({
          trainingJobs: state.trainingJobs.map(j => 
            j.id === job.id ? job : j
          )
        }))
      },
      
      // System actions
      setMetrics: (metrics) => set({ metrics }),
      setConnected: (connected) => set({ isConnected: connected }),
      
      // UI actions
      setActiveTab: (tab) => set({ activeTab: tab }),
      setLoading: (loading) => set({ isLoading: loading }),
      setError: (error) => set({ error }),
      clearError: () => set({ error: undefined })
    }),
    {
      name: 'sloughgpt-web-ui-store',
      partialize: (state) => ({
        messages: state.messages,
        conversations: state.conversations,
        currentConversationId: state.currentConversationId,
        selectedModel: state.selectedModel,
        datasets: state.datasets,
        activeTab: state.activeTab
      })
    }
  )
)
