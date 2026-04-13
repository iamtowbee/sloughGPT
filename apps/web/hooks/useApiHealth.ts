'use client'

import { useCallback, useEffect, useState } from 'react'

import { api, type ApiHealth } from '@/lib/api'

const POLL_MS = 28_000

/** `null` until the first `GET /health` completes. */
export type ApiHealthSnapshot = ApiHealth | 'offline' | null

/** One-line status for headers (Models page, tooltips). */
export function inferenceHealthLabel(state: ApiHealthSnapshot): string {
  if (state === null) return 'checking...'
  if (state === 'offline') return 'disconnected'
  if (state.model_loaded) return `inference ready · ${state.model_type}`
  return `connected · no weights (${state.model_type})`
}

/**
 * Polls `GET /health` on an interval and when the tab becomes visible again.
 * Use for chat, models, and anywhere inference readiness matters.
 */
export function useApiHealth() {
  const [state, setState] = useState<ApiHealthSnapshot>(null)

  const refresh = useCallback(async () => {
    const h = await api.getHealth()
    setState(h ?? 'offline')
  }, [])

  useEffect(() => {
    let cancelled = false
    
    const run = async () => {
      try {
        const h = await api.getHealth()
        if (!cancelled) setState(h ?? 'offline')
      } catch {
        if (!cancelled) setState('offline')
      }
    }
    
    void run()
    const id = setInterval(run, POLL_MS)
    const onVis = () => {
      if (document.visibilityState === 'visible') void run()
    }
    document.addEventListener('visibilitychange', onVis)
    return () => {
      cancelled = true
      clearInterval(id)
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [])

  return { state, refresh }
}
