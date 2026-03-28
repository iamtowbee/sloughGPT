'use client'

import { useState, useEffect } from 'react'
import { api } from '@/lib/api'

interface ComparisonResult {
  model_name: string
  num_parameters: number
  memory_mb: number
  throughput_tokens_per_sec: number
  latency_p50_ms: number
}

export default function ComparePage() {
  const [results, setResults] = useState<Record<string, ComparisonResult>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const runComparison = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.compareBenchmarks()
      const topErr = (res as { error?: unknown }).error
      if (typeof topErr === 'string') {
        setError(topErr)
      } else {
        const ok: Record<string, ComparisonResult> = {}
        for (const [k, v] of Object.entries(res)) {
          if (v && typeof v === 'object' && 'error' in v) {
            continue
          }
          ok[k] = v as ComparisonResult
        }
        setResults(ok)
      }
    } catch (e) {
      setError('Failed to compare benchmarks')
    } finally {
      setLoading(false)
    }
  }

  const metrics = [
    { key: 'num_parameters', label: 'Parameters', format: (v: number) => v.toLocaleString(), unit: '' },
    { key: 'memory_mb', label: 'Memory', format: (v: number) => v.toFixed(1), unit: 'MB' },
    { key: 'throughput_tokens_per_sec', label: 'Speed', format: (v: number) => v.toFixed(2), unit: 'tok/s' },
    { key: 'latency_p50_ms', label: 'Latency P50', format: (v: number) => v.toFixed(0), unit: 'ms' },
  ]

  const comparisonData = Object.entries(results).filter(([k]) => !k.startsWith('error'))

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Model Comparison</h1>
          <p className="text-gray-600">Compare performance across quantization levels</p>
        </div>
        <button
          onClick={runComparison}
          disabled={loading}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Running...' : 'Run Comparison'}
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
          {error}
        </div>
      )}

      {comparisonData.length > 0 && (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Metric</th>
                {comparisonData.map(([name]) => (
                  <th key={name} className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase">
                    {name.toUpperCase()}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y">
              {metrics.map((metric) => (
                <tr key={metric.key} className="hover:bg-gray-50">
                  <td className="px-6 py-4 font-medium text-gray-900">
                    {metric.label}
                    <span className="text-gray-500 text-sm ml-1">({metric.unit})</span>
                  </td>
                  {comparisonData.map(([name, data]) => (
                    <td key={name} className="px-6 py-4 text-center text-gray-600">
                      {metric.format(data[metric.key as keyof ComparisonResult] as number)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {comparisonData.length > 1 && (
        <div className="mt-8">
          <h2 className="text-lg font-semibold mb-4">Visual Comparison</h2>
          
          <div className="space-y-6">
            {metrics.filter(m => m.key !== 'num_parameters').map((metric) => (
              <div key={metric.key} className="bg-white rounded-lg shadow p-4">
                <h3 className="text-sm font-medium text-gray-600 mb-3">{metric.label}</h3>
                <div className="h-8 bg-gray-100 rounded-lg overflow-hidden flex">
                  {comparisonData.map(([name, data]) => {
                    const values = comparisonData.map(d => d[1][metric.key as keyof ComparisonResult] as number)
                    const maxVal = Math.max(...values)
                    const width = ((data[metric.key as keyof ComparisonResult] as number) / maxVal) * 100
                    const colors = ['bg-blue-500', 'bg-green-500', 'bg-purple-500', 'bg-orange-500', 'bg-red-500']
                    const colorIndex = comparisonData.findIndex(d => d[0] === name) % colors.length
                    
                    return (
                      <div
                        key={name}
                        className={`${colors[colorIndex]} flex items-center justify-center text-white text-sm font-medium`}
                        style={{ width: `${width}%` }}
                      >
                        {width > 15 && metric.format(data[metric.key as keyof ComparisonResult] as number)}
                      </div>
                    )
                  })}
                </div>
                <div className="flex gap-4 mt-2">
                  {comparisonData.map(([name], i) => {
                    const colors = ['bg-blue-500', 'bg-green-500', 'bg-purple-500', 'bg-orange-500', 'bg-red-500']
                    return (
                      <div key={name} className="flex items-center gap-1 text-xs text-gray-600">
                        <div className={`w-3 h-3 rounded ${colors[i % colors.length]}`}></div>
                        {name.toUpperCase()}
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {comparisonData.length === 0 && !loading && (
        <div className="bg-gray-50 rounded-lg p-8 text-center">
          <p className="text-gray-500">
            Click &quot;Run Comparison&quot; to compare benchmark results
          </p>
        </div>
      )}
    </div>
  )
}
