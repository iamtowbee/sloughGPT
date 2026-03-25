'use client'

import { useState, useEffect } from 'react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Dataset {
  name: string
  path?: string
  size_kb?: number
  size_bytes?: number
  size_formatted?: string
}

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDatasets()
  }, [])

  const fetchDatasets = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/datasets`)
      const data = await res.json()
      setDatasets(data.datasets || [])
    } catch (err) {
      console.error('Failed to fetch datasets:', err)
      setDatasets([])
    } finally {
      setLoading(false)
    }
  }

  const formatSize = (d: Dataset) => {
    if (d.size_formatted) return d.size_formatted
    if (typeof d.size_bytes === 'number' && d.size_bytes > 0)
      return `${(d.size_bytes / 1024).toFixed(1)} KB`
    if (typeof d.size_kb === 'number' && d.size_kb > 0) {
      const kb = d.size_kb
      if (kb < 1024) return `${kb.toFixed(1)} KB`
      if (kb < 1024 * 1024) return `${(kb / 1024).toFixed(1)} MB`
      return `${(kb / (1024 * 1024)).toFixed(1)} GB`
    }
    return 'Unknown'
  }

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-white">Datasets</h1>
        <button className="bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-4 py-2">
          Add Dataset
        </button>
      </div>

      {loading ? (
        <div className="text-center py-12 text-zinc-500">Loading datasets...</div>
      ) : datasets.length === 0 ? (
        <div className="text-center py-12 text-zinc-500">
          No datasets found. Add datasets to train your models.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {datasets.map((dataset, i) => (
            <div
              key={dataset.name || i}
              className="bg-white/5 border border-white/10 rounded-xl p-4 hover:border-white/20 transition-colors"
            >
              <div className="flex items-start justify-between mb-2">
                <h3 className="font-semibold text-white">{dataset.name}</h3>
                <span className="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded">
                  text
                </span>
              </div>
              {dataset.path && (
                <p className="text-sm text-zinc-500 mb-3 truncate">{dataset.path}</p>
              )}
              <div className="flex justify-between items-center text-sm text-zinc-400">
                <span>{formatSize(dataset)}</span>
                <button className="text-blue-400 hover:text-blue-300">View</button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mt-8 p-4 bg-white/5 rounded-xl border border-white/10">
        <h2 className="text-lg font-semibold text-white mb-2">Quick Commands</h2>
        <div className="text-sm text-zinc-400 space-y-1 font-mono">
          <p>python cli.py data validate datasets/my_data/</p>
          <p>python cli.py data stats datasets/my_data/</p>
          <p>python cli.py data split datasets/my_data/ --train 0.9</p>
        </div>
      </div>
    </div>
  )
}
