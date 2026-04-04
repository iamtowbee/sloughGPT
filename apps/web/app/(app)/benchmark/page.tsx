'use client'

import { useState } from 'react'

import { api, BenchmarkResult } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'

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

  const statCard = (label: string, value: string, sub?: string) => (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription className="text-xs font-mono uppercase tracking-wider">{label}</CardDescription>
        <p className="text-xl font-semibold tabular-nums text-foreground">{value}</p>
        {sub && <p className="text-xs text-muted-foreground">{sub}</p>}
      </CardHeader>
    </Card>
  )

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <h1 className="sl-h1 mb-6">Benchmark</h1>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Run inference benchmark</CardTitle>
          <CardDescription>Uses the API benchmark endpoint with your prompt and token budget.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="bench-prompt">Prompt</Label>
            <Input id="bench-prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="bench-max">Max tokens</Label>
              <Input
                id="bench-max"
                type="number"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value, 10) || 50)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="bench-runs">Number of runs</Label>
              <Input
                id="bench-runs"
                type="number"
                value={numRuns}
                onChange={(e) => setNumRuns(parseInt(e.target.value, 10) || 3)}
              />
            </div>
          </div>
          <Button type="button" onClick={runBenchmark} disabled={loading}>
            {loading ? 'Running…' : 'Run benchmark'}
          </Button>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>

      {result && !result.error && (
        <>
          <h2 className="mb-3 text-lg font-semibold text-foreground">Results</h2>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {statCard('Parameters', result.num_parameters.toLocaleString())}
            {statCard('Memory', `${result.memory_mb.toFixed(1)} MB`)}
            {statCard('Throughput', `${result.throughput_tokens_per_sec.toFixed(2)} tok/s`)}
            {statCard('Avg latency', `${result.inference_time_ms.toFixed(0)} ms`)}
          </div>
          <div className="mt-4 grid grid-cols-3 gap-3">
            <Card className="border-primary/25 bg-primary/5">
              <CardHeader className="pb-2">
                <CardDescription className="text-primary">P50 latency</CardDescription>
                <p className="text-lg font-semibold text-foreground">
                  {result.latency_p50_ms?.toFixed(0) ?? '—'} ms
                </p>
              </CardHeader>
            </Card>
            <Card className="border-primary/25 bg-primary/5">
              <CardHeader className="pb-2">
                <CardDescription className="text-primary">P95 latency</CardDescription>
                <p className="text-lg font-semibold text-foreground">
                  {result.latency_p95_ms?.toFixed(0) ?? '—'} ms
                </p>
              </CardHeader>
            </Card>
            <Card className="border-primary/25 bg-primary/5">
              <CardHeader className="pb-2">
                <CardDescription className="text-primary">P99 latency</CardDescription>
                <p className="text-lg font-semibold text-foreground">
                  {result.latency_p99_ms?.toFixed(0) ?? '—'} ms
                </p>
              </CardHeader>
            </Card>
          </div>
          <Separator className="my-6" />
        </>
      )}
    </div>
  )
}
