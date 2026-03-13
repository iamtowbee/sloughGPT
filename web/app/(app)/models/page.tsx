'use client'

import { useState, useEffect } from 'react'
import { api, Model } from '@/lib/api'

export default function ModelsPage() {
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(true)
  const [loadingModel, setLoadingModel] = useState<string | null>(null)

  useEffect(() => {
    api.getModels()
      .then(setModels)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  const loadModel = async (modelId: string) => {
    setLoadingModel(modelId)
    setTimeout(() => {
      alert(`Model ${modelId} loaded! (Demo)`)
      setLoadingModel(null)
    }, 1000)
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-slate-800 dark:text-white">Models</h1>
        <button className="bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-4 py-2">
          Add Model
        </button>
      </div>

      {loading ? (
        <div className="text-center py-8 text-slate-500">Loading models...</div>
      ) : models.length === 0 ? (
        <div className="text-center py-8 text-slate-500">No models available.</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((model) => (
            <div
              key={model.id}
              className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4"
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h3 className="font-semibold text-slate-800 dark:text-white">{model.name}</h3>
                  <p className="text-sm text-slate-500">{model.type}</p>
                </div>
                <span className="bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 text-xs px-2 py-1 rounded">
                  {model.quantization || 'F16'}
                </span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-500">{model.size}</span>
                <button 
                  onClick={() => loadModel(model.id)}
                  disabled={loadingModel !== null}
                  className="text-blue-600 hover:text-blue-700 text-sm font-medium disabled:opacity-50"
                >
                  {loadingModel === model.id ? 'Loading...' : 'Load'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
