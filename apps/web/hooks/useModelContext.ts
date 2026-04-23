'use client'

import { useCallback, useEffect, useState } from 'react'
import { api, type Model, type ApiHealth } from '@/lib/api'
import { useModelLoader, type ModelLoadResult } from './useModelLoader'
import { useApiHealth } from './useApiHealth'

export interface ModelInfo {
  id: string
  name: string
  type: string
  loaded: boolean
  sizeMb?: number
  params?: string
  description?: string
  tags?: string[]
}

export interface ModelContextState {
  models: ModelInfo[]
  currentModel: string | null
  loading: boolean
  loadingModelId: string | null
  error: string | null
  isModelLoaded: boolean
}

export interface UseModelContextReturn {
  state: ModelContextState
  models: ModelInfo[]
  currentModel: string | null
  isModelLoaded: boolean
  loading: boolean
  loadingModelId: string | null
  error: string | null
  loadModel: (modelId: string, opts?: { mode?: string; device?: string }) => Promise<ModelLoadResult>
  loadModelPath: (path: string) => Promise<ModelLoadResult>
  unloadModel: (modelId: string) => Promise<ModelLoadResult>
  refreshModels: () => Promise<void>
  selectModel: (modelId: string) => void
  clearError: () => void
}

export function useModelContext(): UseModelContextReturn {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [currentModel, setCurrentModel] = useState<string | null>(null)
  const { state: health, refresh: refreshHealth } = useApiHealth()
  const loader = useModelLoader()

  const isModelLoaded = health !== null && health !== 'offline' && (health as ApiHealth).model_loaded

  const refreshModels = useCallback(async () => {
    try {
      const fetchedModels = await api.getModels()
      setModels(fetchedModels.map(m => ({
        id: m.id,
        name: m.name,
        type: m.type,
        loaded: m.loaded ?? false,
        sizeMb: m.size_mb,
        params: m.params,
        description: m.description,
        tags: m.tags,
      })))

      if (health !== null && health !== 'offline') {
        const h = health as ApiHealth
        if (h.model_loaded && h.model_type) {
          setCurrentModel(h.model_type)
        }
      }
    } catch (err) {
      console.error('Failed to fetch models:', err)
    }
  }, [health])

  useEffect(() => {
    void refreshModels()
  }, [refreshModels])

  useEffect(() => {
    if (health !== null && health !== 'offline') {
      const h = health as ApiHealth
      if (h.model_loaded && h.model_type) {
        setCurrentModel(h.model_type)
      }
    }
  }, [health])

  const selectModel = useCallback((modelId: string) => {
    setCurrentModel(modelId)
  }, [])

  return {
    state: {
      models,
      currentModel,
      loading: loader.state.loading,
      loadingModelId: loader.state.loadingModelId,
      error: loader.state.error,
      isModelLoaded,
    },
    models,
    currentModel,
    isModelLoaded,
    loading: loader.state.loading,
    loadingModelId: loader.state.loadingModelId,
    error: loader.state.error,
    loadModel: async (modelId: string, opts?: { mode?: string; device?: string }) => {
      const result = await loader.loadModel(modelId, opts)
      if (result.success) {
        await refreshModels()
        await refreshHealth()
      }
      return result
    },
    loadModelPath: async (path: string) => {
      const result = await loader.loadModelPath(path)
      if (result.success) {
        await refreshModels()
        await refreshHealth()
      }
      return result
    },
    unloadModel: async (modelId: string) => {
      const result = await loader.unloadModel(modelId)
      if (result.success) {
        await refreshModels()
        await refreshHealth()
      }
      return result
    },
    refreshModels,
    selectModel,
    clearError: loader.clearError,
  }
}

export function useCurrentModel() {
  const { currentModel, isModelLoaded, models } = useModelContext()
  
  return {
    modelId: currentModel,
    isLoaded: isModelLoaded,
    modelInfo: models.find(m => m.id === currentModel || m.name === currentModel),
  }
}

export function useAvailableModels(filter?: { type?: 'local' | 'huggingface' | 'all' }) {
  const { models, loading, error, refreshModels } = useModelContext()

  const filteredModels = models.filter(m => {
    if (!filter?.type || filter.type === 'all') return true
    return m.type === filter.type
  })

  return {
    models: filteredModels,
    loading,
    error,
    refreshModels,
  }
}