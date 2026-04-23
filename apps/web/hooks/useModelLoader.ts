'use client'

import { useCallback, useState } from 'react'
import { api, type Model } from '@/lib/api'

export interface ModelLoadResult {
  success: boolean
  modelId?: string
  modelType?: string
  device?: string
  error?: string
}

export interface ModelLoaderState {
  loading: boolean
  loadingModelId: string | null
  error: string | null
  lastLoadResult: ModelLoadResult | null
}

export interface UseModelLoaderReturn {
  state: ModelLoaderState
  loadModel: (modelId: string, opts?: { mode?: string; device?: string }) => Promise<ModelLoadResult>
  loadModelPath: (path: string) => Promise<ModelLoadResult>
  unloadModel: (modelId: string) => Promise<ModelLoadResult>
  clearError: () => void
}

const initialState: ModelLoaderState = {
  loading: false,
  loadingModelId: null,
  error: null,
  lastLoadResult: null,
}

export function useModelLoader(): UseModelLoaderReturn {
  const [state, setState] = useState<ModelLoaderState>(initialState)

  const loadModel = useCallback(async (
    modelId: string,
    opts?: { mode?: string; device?: string }
  ): Promise<ModelLoadResult> => {
    setState(prev => ({ ...prev, loading: true, loadingModelId: modelId, error: null }))

    try {
      const data = await api.loadModel(modelId, opts)

      if (data.error || data.status === 'error') {
        const error = data.error || 'Load failed'
        setState(prev => ({
          ...prev,
          loading: false,
          loadingModelId: null,
          error,
          lastLoadResult: { success: false, error },
        }))
        return { success: false, error }
      }

      const result: ModelLoadResult = {
        success: true,
        modelId: data.model || modelId,
        modelType: data.model_type,
        device: data.effective_device ?? undefined,
      }

      setState(prev => ({
        ...prev,
        loading: false,
        loadingModelId: null,
        lastLoadResult: result,
      }))

      return result
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Unknown error'
      setState(prev => ({
        ...prev,
        loading: false,
        loadingModelId: null,
        error,
        lastLoadResult: { success: false, error },
      }))
      return { success: false, error }
    }
  }, [])

  const loadModelPath = useCallback(async (path: string): Promise<ModelLoadResult> => {
    setState(prev => ({ ...prev, loading: true, loadingModelId: path, error: null }))

    try {
      const data = await api.loadModelPath(path)

      if (data.error) {
        const error = String(data.error)
        setState(prev => ({
          ...prev,
          loading: false,
          loadingModelId: null,
          error,
          lastLoadResult: { success: false, error },
        }))
        return { success: false, error }
      }

      const result: ModelLoadResult = {
        success: true,
        modelId: path,
        modelType: 'local',
      }

      setState(prev => ({
        ...prev,
        loading: false,
        loadingModelId: null,
        lastLoadResult: result,
      }))

      return result
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Unknown error'
      setState(prev => ({
        ...prev,
        loading: false,
        loadingModelId: null,
        error,
        lastLoadResult: { success: false, error },
      }))
      return { success: false, error }
    }
  }, [])

  const unloadModel = useCallback(async (modelId: string): Promise<ModelLoadResult> => {
    setState(prev => ({ ...prev, loading: true, loadingModelId: modelId, error: null }))

    try {
      const data = await api.unloadModel(modelId)

      if (data.error) {
        const error = data.error
        setState(prev => ({
          ...prev,
          loading: false,
          loadingModelId: null,
          error,
          lastLoadResult: { success: false, error },
        }))
        return { success: false, error }
      }

      const result: ModelLoadResult = {
        success: true,
        modelId,
        modelType: 'unloaded',
      }

      setState(prev => ({
        ...prev,
        loading: false,
        loadingModelId: null,
        lastLoadResult: result,
      }))

      return result
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Unknown error'
      setState(prev => ({
        ...prev,
        loading: false,
        loadingModelId: null,
        error,
        lastLoadResult: { success: false, error },
      }))
      return { success: false, error }
    }
  }, [])

  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }))
  }, [])

  return {
    state,
    loadModel,
    loadModelPath,
    unloadModel,
    clearError,
  }
}

export type { Model }