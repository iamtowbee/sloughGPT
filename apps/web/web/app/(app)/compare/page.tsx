'use client'

import { useState } from 'react'

import { api } from '@/lib/api'

interface ComparisonResult {
  model_name: string
  num_parameters: number
  memory_mb: number
  throughput_tokens_per_sec: number
  latency_p50_ms: number
}

const BAR = ['bg-chart-1', 'bg-chart-2', 'bg-chart-3', 'bg-chart-4', 'bg-chart-5'] as const

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
    } catch {
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
    <div className="sl-page max-w-6xl mx-auto">
      <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="sl-h1">Model Comparison</h1>
          <p className="text-muted-foreground text-sm">Compare performance across quantization levels</p>
        </div>
        <button
          type="button"
          onClick={runComparison}
          disabled={loading}
          className="sl-btn-primary rounded-lg px-4 py-2 disabled:opacity-50"
        >
          {loading ? 'Running...' : 'Run Comparison'}
        </button>
      </div>

      {error && (
        <div className="mb-6 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-destructive text-sm">
          {error}
        </div>
      )}

      {comparisonData.length > 0 && (
        <div className="sl-card-solid overflow-hidden">
          <table className="w-full text-sm">
            <thead className="border-b border-border bg-muted/30">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-mono font-medium uppercase tracking-wider text-muted-foreground">
                  Metric
                </th>
                {comparisonData.map(([name]) => (
                  <th
                    key={name}
                    className="px-4 py-3 text-center text-xs font-mono font-medium uppercase tracking-wider text-muted-foreground"
                  >
                    {name.toUpperCase()}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {metrics.map((metric) => (
                <tr key={metric.key} className="hover:bg-muted/20">
                  <td className="px-4 py-3 font-medium text-foreground">
                    {metric.label}
                    <span className="ml-1 text-muted-foreground">({metric.unit})</span>
                  </td>
                  {comparisonData.map(([name, data]) => (
                    <td key={name} className="px-4 py-3 text-center text-muted-foreground tabular-nums">
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
          <h2 className="sl-h2 mb-4">Visual comparison</h2>
          <div className="space-y-6">
            {metrics
              .filter((m) => m.key !== 'num_parameters')
              .map((metric) => (
                <div key={metric.key} className="sl-card p-4">
                  <h3 className="mb-3 text-sm font-medium text-muted-foreground">{metric.label}</h3>
                  <div className="flex h-8 overflow-hidden rounded-lg">
                    {comparisonData.map(([name, data]) => {
                      const values = comparisonData.map((d) => d[1][metric.key as keyof ComparisonResult] as number)
                      const maxVal = Math.max(...values)
                      const width = ((data[metric.key as keyof ComparisonResult] as number) / maxVal) * 100
                      const colorIndex = comparisonData.findIndex((d) => d[0] === name) % BAR.length
                      return (
                        <div
                          key={name}
                          className={`${BAR[colorIndex]} flex items-center justify-center text-primary-foreground text-xs font-medium`}
                          style={{ width: `${width}%` }}
                        >
                          {width > 15 && metric.format(data[metric.key as keyof ComparisonResult] as number)}
                        </div>
                      )
                    })}
                  </div>
                  <div className="mt-2 flex flex-wrap gap-4">
                    {comparisonData.map(([name], i) => (
                      <div key={name} className="flex items-center gap-1 text-xs text-muted-foreground">
                        <span className={`h-3 w-3 rounded ${BAR[i % BAR.length]}`} />
                        {name.toUpperCase()}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      {comparisonData.length === 0 && !loading && (
        <div className="rounded-xl border border-border bg-muted/30 p-8 text-center text-muted-foreground text-sm">
          Click &quot;Run Comparison&quot; to compare benchmark results
        </div>
      )}
    </div>
  )
}
