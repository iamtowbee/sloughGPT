'use client'

import { useState, useEffect } from 'react'
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
    } catch (e) {
      setError('Failed to run benchmark')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Benchmark</h1>
      
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h2 className="text-lg font-semibold mb-4">Run Inference Benchmark</h2>
        
        <div className="grid gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium mb-1">Prompt</label>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full px-3 py-2 border rounded-lg"
            />
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Max Tokens</label>
              <input
                type="number"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value) || 50)}
                className="w-full px-3 py-2 border rounded-lg"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Number of Runs</label>
              <input
                type="number"
                value={numRuns}
                onChange={(e) => setNumRuns(parseInt(e.target.value) || 3)}
                className="w-full px-3 py-2 border rounded-lg"
              />
            </div>
          </div>
        </div>
        
        <button
          onClick={runBenchmark}
          disabled={loading}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Running...' : 'Run Benchmark'}
        </button>
        
        {error && (
          <p className="mt-4 text-red-600">{error}</p>
        )}
      </div>

      {result && !result.error && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Results</h2>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Parameters</p>
              <p className="text-xl font-bold">{result.num_parameters.toLocaleString()}</p>
            </div>
            
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Memory</p>
              <p className="text-xl font-bold">{result.memory_mb.toFixed(1)} MB</p>
            </div>
            
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Throughput</p>
              <p className="text-xl font-bold">{result.throughput_tokens_per_sec.toFixed(2)} tok/s</p>
            </div>
            
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">Avg Latency</p>
              <p className="text-xl font-bold">{result.inference_time_ms.toFixed(0)} ms</p>
            </div>
          </div>
          
          <div className="mt-4 grid grid-cols-3 gap-4">
            <div className="p-4 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-600">P50 Latency</p>
              <p className="text-lg font-bold">{result.latency_p50_ms?.toFixed(0) || '-'} ms</p>
            </div>
            <div className="p-4 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-600">P95 Latency</p>
              <p className="text-lg font-bold">{result.latency_p95_ms?.toFixed(0) || '-'} ms</p>
            </div>
            <div className="p-4 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-600">P99 Latency</p>
              <p className="text-lg font-bold">{result.latency_p99_ms?.toFixed(0) || '-'} ms</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
