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

  const sourceColors: Record<string, string> = {
    local: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300',
    huggingface: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300',
  }

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold text-white">Models</h1>
          <p className="text-zinc-400 mt-1">
            API: <span className={apiHealth === 'connected' ? 'text-green-400' : 'text-red-400'}>{apiHealth}</span>
          </p>
        </div>
        <div className="flex gap-2">
          {(['all', 'local', 'huggingface'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 rounded-lg text-sm ${
                filter === f
                  ? 'bg-blue-600 text-white'
                  : 'bg-white/5 text-zinc-400 hover:bg-white/10'
              }`}
            >
              {f === 'all' ? 'All' : f === 'local' ? 'Local' : 'HuggingFace'}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="text-center py-12 text-zinc-500">Loading models...</div>
      ) : filteredModels.length === 0 ? (
        <div className="text-center py-12 text-zinc-500">
          No {filter === 'all' ? '' : filter} models found.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredModels.map((model) => (
            <div
              key={model.id}
              className="bg-white/5 border border-white/10 rounded-xl p-4 hover:border-white/20 transition-colors"
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h3 className="font-semibold text-white">{model.name}</h3>
                  {model.description && (
                    <p className="text-sm text-zinc-500 mt-1 line-clamp-2">{model.description}</p>
                  )}
                </div>
                {model.source && (
                  <span
                    className={`text-xs px-2 py-1 rounded ${
                      sourceColors[model.source] || 'bg-zinc-100 text-zinc-800'
                    }`}
                  >
                    {model.source}
                  </span>
                )}
              </div>

              {model.tags && model.tags.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-3">
                  {model.tags.slice(0, 4).map((tag) => (
                    <span
                      key={tag}
                      className="text-xs bg-white/5 text-zinc-400 px-2 py-0.5 rounded"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}

              <div className="flex justify-between items-center pt-3 border-t border-white/5">
                <span className="text-sm text-zinc-500">
                  {model.size_mb ? `${model.size_mb.toFixed(1)} MB` : model.id}
                </span>
                <button
                  onClick={() => loadModel(model.id)}
                  disabled={loadingModel !== null}
                  className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white text-sm px-3 py-1.5 rounded-lg transition-colors"
                >
                  {loadingModel === model.id ? 'Loading...' : 'Load'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mt-8 p-4 bg-white/5 rounded-xl border border-white/10">
        <h2 className="text-lg font-semibold text-white mb-2">API Endpoints</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
          <code className="bg-black/20 p-2 rounded text-zinc-300">GET /models</code>
          <code className="bg-black/20 p-2 rounded text-zinc-300">POST /models/load</code>
          <code className="bg-black/20 p-2 rounded text-zinc-300">GET /models/hf</code>
          <code className="bg-black/20 p-2 rounded text-zinc-300">GET /health</code>
        </div>
      </div>
    </div>
  )
}
