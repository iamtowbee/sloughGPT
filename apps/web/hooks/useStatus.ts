'use client'

import { useState, useEffect, useCallback } from 'react'
import { api } from '@/lib/api'
import type { ApiHealth } from '@/lib/api'

interface StatusData {
  models: string[]
  config: Record<string, unknown>
  health: ApiHealth | null
  datasets: Array<{ id: string; name: string }>
}

/**
 * useStatus - Single hook for all initialization data.
 * Reduces 5+ API calls on load to just 2 (health + models).
 */
export function useStatus() {
  const [data, setData] = useState<StatusData>({
    models: [],
    config: {},
    health: null,
    datasets: [],
  })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStatus = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      const [models, config, datasets] = await Promise.all([
        api.getModels().then((m: unknown) => (m as Array<{ id: string }>).map(x => x.id)),
        api.getGenerationConfig().catch(() => ({})),
        api.getDatasets().catch(() => []),
      ])

      setData({
        models: models as string[],
        config: config as Record<string, unknown>,
        health: null, // Health comes from useApiHealth hook
        datasets: datasets as Array<{ id: string; name: string }>,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch status')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
  }, [fetchStatus])

  return {
    data,
    loading,
    error,
    refresh: fetchStatus,
  }
}