'use client'

import { useState, useEffect } from 'react'

import { PUBLIC_API_URL } from '@/lib/config'

interface Model {
  id: string
  name: string
  source?: string
  description?: string
  size_mb?: number
  tags?: string[]
}

export default function ModelsPage() {
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(true)
  const [loadingModel, setLoadingModel] = useState<string | null>(null)
  const [filter, setFilter] = useState<'all' | 'local' | 'huggingface'>('all')
  const [apiHealth, setApiHealth] = useState<string>('checking...')

  useEffect(() => {
    checkHealth()
    fetchModels()
  }, [])

  const checkHealth = async () => {
    try {
      const res = await fetch(`${PUBLIC_API_URL}/health`)
      const data = await res.json()
      setApiHealth(data.model_type || 'connected')
    } catch {
      setApiHealth('disconnected')
    }
  }

  const fetchModels = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${PUBLIC_API_URL}/models`)
      const data = await res.json()
      setModels(data.models || [])
    } catch (err) {
      console.error('Failed to fetch models:', err)
      setModels([])
    } finally {
      setLoading(false)
    }
  }

  const loadModel = async (modelId: string) => {
    setLoadingModel(modelId)
    try {
      const res = await fetch(`${PUBLIC_API_URL}/models/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId }),
      })
      const data = await res.json()
      alert(`${data.status}: ${data.model || modelId}`)
    } catch (err) {
      alert(`Failed to load model: ${err}`)
    } finally {
      setLoadingModel(null)
    }
  }

  const filteredModels = models.filter((m) => {
    if (filter === 'all') return true
    if (filter === 'local') return m.source === 'local'
    if (filter === 'huggingface') return m.source === 'huggingface'
    return true
  })

  const sourceBadge = (src: string) =>
    src === 'local'
      ? 'bg-success/15 text-success border border-success/25'
      : 'bg-primary/15 text-primary border border-primary/25'

  return (
    <div className="sl-page max-w-6xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="sl-h1">Models</h1>
          <p className="text-muted-foreground mt-1">
            API:{' '}
            <span className={apiHealth === 'disconnected' ? 'text-destructive' : 'text-success'}>
              {apiHealth}
            </span>
          </p>
        </div>
        <div className="flex gap-2">
          {(['all', 'local', 'huggingface'] as const).map((f) => (
            <button
              type="button"
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 rounded-lg text-sm border transition-colors ${
                filter === f
                  ? 'bg-primary/20 text-primary border-primary/30'
                  : 'bg-muted/40 text-muted-foreground border-border hover:text-foreground'
              }`}
            >
              {f === 'all' ? 'All' : f === 'local' ? 'Local' : 'HuggingFace'}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="text-center py-12 text-muted-foreground">Loading models...</div>
      ) : filteredModels.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          No {filter === 'all' ? '' : filter} models found.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredModels.map((model) => (
            <div
              key={model.id}
              className="sl-card p-4 hover:border-primary/25 transition-colors ring-1 ring-primary/5"
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h3 className="font-semibold text-foreground">{model.name}</h3>
                  {model.description && (
                    <p className="text-sm text-muted-foreground mt-1 line-clamp-2">{model.description}</p>
                  )}
                </div>
                {model.source && (
                  <span className={`text-xs px-2 py-1 rounded-md font-medium ${sourceBadge(model.source)}`}>
                    {model.source}
                  </span>
                )}
              </div>

              {model.tags && model.tags.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-3">
                  {model.tags.slice(0, 4).map((tag) => (
                    <span
                      key={tag}
                      className="text-xs bg-muted text-muted-foreground px-2 py-0.5 rounded border border-border"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}

              <div className="flex justify-between items-center pt-3 border-t border-border">
                <span className="text-sm text-muted-foreground">
                  {model.size_mb ? `${model.size_mb.toFixed(1)} MB` : model.id}
                </span>
                <button
                  type="button"
                  onClick={() => loadModel(model.id)}
                  disabled={loadingModel !== null}
                  className="sl-btn-primary text-sm px-3 py-1.5 rounded-lg"
                >
                  {loadingModel === model.id ? 'Loading...' : 'Load'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mt-8 sl-card p-4">
        <h2 className="text-lg font-semibold text-foreground mb-2">API Endpoints</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
          <code className="sl-code block p-2">GET /models</code>
          <code className="sl-code block p-2">POST /models/load</code>
          <code className="sl-code block p-2">GET /models/hf</code>
          <code className="sl-code block p-2">GET /health</code>
        </div>
      </div>
    </div>
  )
}
