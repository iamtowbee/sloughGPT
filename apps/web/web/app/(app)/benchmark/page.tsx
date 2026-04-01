'use client'

import { useState } from 'react'

import { api, BenchmarkResult } from '@/lib/api'

export default function BenchmarkPage() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<BenchmarkResult | null>(null)
  const [prompt, setPrompt] = useState('The quick brown fox jumps over the lazy dog')
  const [maxTokens, setMaxTokens] = useState(50)
  const [numRuns, setNumRuns] = useState(3)
  const [error, setError] = useState<string | null>(null)

  const runBenchmark = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.runBenchmark(prompt, maxTokens, numRuns)
      if (res.error) {
        setError(res.error)
      } else {
        setResult(res)
      }
    } catch {
      setError('Failed to run benchmark')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="sl-page max-w-4xl mx-auto">
      <h1 className="sl-h1 mb-6">Benchmark</h1>

      <div className="sl-card mb-6 p-6">
        <h2 className="sl-h2 mb-4">Run Inference Benchmark</h2>

        <div className="mb-4 grid gap-4">
          <div>
            <label className="sl-label normal-case tracking-normal">Prompt</label>
            <input type="text" value={prompt} onChange={(e) => setPrompt(e.target.value)} className="sl-input" />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="sl-label normal-case tracking-normal">Max Tokens</label>
              <input
                type="number"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value, 10) || 50)}
                className="sl-input"
              />
            </div>
            <div>
              <label className="sl-label normal-case tracking-normal">Number of Runs</label>
              <input
                type="number"
                value={numRuns}
                onChange={(e) => setNumRuns(parseInt(e.target.value, 10) || 3)}
                className="sl-input"
              />
            </div>
          </div>
        </div>

        <button type="button" onClick={runBenchmark} disabled={loading} className="sl-btn-primary rounded-lg px-4 py-2">
          {loading ? 'Running...' : 'Run Benchmark'}
        </button>

        {error && <p className="mt-4 text-sm text-destructive">{error}</p>}
      </div>

      {result && !result.error && (
        <div className="sl-card p-6">
          <h2 className="sl-h2 mb-4">Results</h2>

          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <div className="rounded-lg border border-border bg-muted/30 p-4">
              <p className="text-xs font-mono uppercase tracking-wider text-muted-foreground">Parameters</p>
              <p className="text-xl font-semibold tabular-nums text-foreground">{result.num_parameters.toLocaleString()}</p>
            </div>
            <div className="rounded-lg border border-border bg-muted/30 p-4">
              <p className="text-xs font-mono uppercase tracking-wider text-muted-foreground">Memory</p>
              <p className="text-xl font-semibold tabular-nums text-foreground">{result.memory_mb.toFixed(1)} MB</p>
            </div>
            <div className="rounded-lg border border-border bg-muted/30 p-4">
              <p className="text-xs font-mono uppercase tracking-wider text-muted-foreground">Throughput</p>
              <p className="text-xl font-semibold tabular-nums text-foreground">
                {result.throughput_tokens_per_sec.toFixed(2)} tok/s
              </p>
            </div>
            <div className="rounded-lg border border-border bg-muted/30 p-4">
              <p className="text-xs font-mono uppercase tracking-wider text-muted-foreground">Avg Latency</p>
              <p className="text-xl font-semibold tabular-nums text-foreground">{result.inference_time_ms.toFixed(0)} ms</p>
            </div>
          </div>

          <div className="mt-4 grid grid-cols-3 gap-4">
            <div className="rounded-lg border border-primary/25 bg-primary/10 p-4">
              <p className="text-xs text-primary">P50 Latency</p>
              <p className="text-lg font-semibold text-foreground">{result.latency_p50_ms?.toFixed(0) || '-'} ms</p>
            </div>
            <div className="rounded-lg border border-primary/25 bg-primary/10 p-4">
              <p className="text-xs text-primary">P95 Latency</p>
              <p className="text-lg font-semibold text-foreground">{result.latency_p95_ms?.toFixed(0) || '-'} ms</p>
            </div>
            <div className="rounded-lg border border-primary/25 bg-primary/10 p-4">
              <p className="text-xs text-primary">P99 Latency</p>
              <p className="text-lg font-semibold text-foreground">{result.latency_p99_ms?.toFixed(0) || '-'} ms</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
