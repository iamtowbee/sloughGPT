'use client'

import { useState, useEffect } from 'react'

import { PUBLIC_API_URL } from '@/lib/config'

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
      const res = await fetch(`${PUBLIC_API_URL}/datasets`)
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
    <div className="sl-page max-w-6xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="sl-h1">Datasets</h1>
        <button type="button" className="sl-btn-primary rounded-lg px-4 py-2">
          Add Dataset
        </button>
      </div>

      {loading ? (
        <div className="text-center py-12 text-muted-foreground">Loading datasets...</div>
      ) : datasets.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          No datasets found. Add datasets to train your models.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {datasets.map((dataset, i) => (
            <div key={dataset.name || i} className="sl-card p-4 hover:border-primary/20 transition-colors">
              <div className="flex items-start justify-between mb-2">
                <h3 className="font-semibold text-foreground">{dataset.name}</h3>
                <span className="text-xs px-2 py-1 rounded-md bg-muted text-muted-foreground border border-border font-mono">
                  text
                </span>
              </div>
              {dataset.path && (
                <p className="text-sm text-muted-foreground mb-3 truncate font-mono text-xs">{dataset.path}</p>
              )}
              <div className="flex justify-between items-center text-sm text-muted-foreground">
                <span className="text-chart-3 font-medium">{formatSize(dataset)}</span>
                <button type="button" className="text-primary hover:underline text-sm font-medium">
                  View
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mt-8 sl-card p-4">
        <h2 className="sl-h2 mb-2">Quick Commands</h2>
        <div className="text-sm text-muted-foreground space-y-1 font-mono">
          <p>python cli.py data validate datasets/my_data/</p>
          <p>python cli.py data stats datasets/my_data/</p>
          <p>python cli.py data split datasets/my_data/ --train 0.9</p>
        </div>
      </div>
    </div>
  )
}
