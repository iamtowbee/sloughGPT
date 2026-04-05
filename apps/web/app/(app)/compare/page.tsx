'use client'

import { useState } from 'react'

import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { api } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

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
    <div className="sl-page mx-auto max-w-6xl">
      <AppRouteHeader
        className="mb-6 items-start"
        left={
          <AppRouteHeaderLead
            title="Model comparison"
            subtitle="Compare performance across quantization levels"
          />
        }
        right={
          <Button type="button" onClick={runComparison} disabled={loading}>
            {loading ? 'Running…' : 'Run comparison'}
          </Button>
        }
      />

      {error && (
        <Card className="mb-6 border-destructive/40 bg-destructive/10">
          <CardContent className="py-4 text-sm text-destructive">{error}</CardContent>
        </Card>
      )}

      {comparisonData.length > 0 && (
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-border bg-muted/30">
                <tr>
                  <th className="px-4 py-3 text-left font-mono text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Metric
                  </th>
                  {comparisonData.map(([name]) => (
                    <th
                      key={name}
                      className="px-4 py-3 text-center font-mono text-xs font-medium uppercase tracking-wider text-muted-foreground"
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
                      <td key={name} className="px-4 py-3 text-center tabular-nums text-muted-foreground">
                        {metric.format(data[metric.key as keyof ComparisonResult] as number)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {comparisonData.length > 1 && (
        <div className="mt-8 space-y-6">
          <h2 className="text-lg font-semibold text-foreground">Visual comparison</h2>
          {metrics
            .filter((m) => m.key !== 'num_parameters')
            .map((metric) => (
              <Card key={metric.key}>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">{metric.label}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex h-8 overflow-hidden">
                    {comparisonData.map(([name, data]) => {
                      const values = comparisonData.map((d) => d[1][metric.key as keyof ComparisonResult] as number)
                      const maxVal = Math.max(...values)
                      const width = ((data[metric.key as keyof ComparisonResult] as number) / maxVal) * 100
                      const colorIndex = comparisonData.findIndex((d) => d[0] === name) % BAR.length
                      return (
                        <div
                          key={name}
                          className={`${BAR[colorIndex]} flex items-center justify-center text-xs font-medium text-primary-foreground`}
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
                        <span className={`h-3 w-3 ${BAR[i % BAR.length]}`} />
                        {name.toUpperCase()}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
        </div>
      )}

      {comparisonData.length === 0 && !loading && (
        <Card className="border-dashed bg-muted/20">
          <CardContent className="py-12 text-center text-sm text-muted-foreground">
            Click &quot;Run comparison&quot; to compare benchmark results from the API.
          </CardContent>
        </Card>
      )}
    </div>
  )
}
