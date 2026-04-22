'use client'

import { useCallback, useEffect, useState, useRef } from 'react'

import { api, type ApiHealth } from '@/lib/api'

const POLL_MS = 28_000

let _sharedState: ApiHealthSnapshot = null
let _pollId: ReturnType<typeof setInterval> | null = null
let _subscribers: Set<(s: ApiHealthSnapshot) => void> = new Set()

function _notify() {
  _subscribers.forEach(fn => fn(_sharedState))
}

function _startPolling() {
  if (_pollId) return
  const run = async () => {
    const h = await api.getHealth()
    _sharedState = h ?? 'offline'
    _notify()
  }
  void run()
  _pollId = setInterval(run, POLL_MS)
}

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
 * Uses shared singleton polling to avoid redundant requests.
 */
export function useApiHealth() {
  const [state, setState] = useState<ApiHealthSnapshot>(_sharedState)

  useEffect(() => {
    _startPolling()
    const fn = (s: ApiHealthSnapshot) => setState(s)
    _subscribers.add(fn)
    return () => { _subscribers.delete(fn) }
  }, [])

  const refresh = useCallback(async () => {
    const h = await api.getHealth()
    const s = h ?? 'offline'
    _sharedState = s
    _notify()
  }, [])

  return { state, refresh }
}
