'use client'

import { useState } from 'react'

interface Endpoint {
  method: string
  path: string
  description: string
  body?: { field: string; type: string; required: boolean }[]
}

const endpoints: Endpoint[] = [
  {
    method: 'GET',
    path: '/',
    description: 'API root with server info and available endpoints',
  },
  {
    method: 'GET',
    path: '/health',
    description: 'Health check endpoint',
  },
  {
    method: 'GET',
    path: '/info',
    description: 'Detailed system and model information',
  },
  {
    method: 'POST',
    path: '/generate',
    description: 'Text generation (non-streaming)',
    body: [
      { field: 'prompt', type: 'string', required: true },
      { field: 'max_new_tokens', type: 'number', required: false },
      { field: 'temperature', type: 'number', required: false },
      { field: 'top_p', type: 'number', required: false },
      { field: 'personality', type: 'string', required: false },
    ],
  },
  {
    method: 'POST',
    path: '/generate/stream',
    description: 'Streaming text generation via Server-Sent Events',
    body: [
      { field: 'prompt', type: 'string', required: true },
      { field: 'max_new_tokens', type: 'number', required: false },
      { field: 'temperature', type: 'number', required: false },
    ],
  },
  {
    method: 'POST',
    path: '/chat/stream',
    description: 'Streaming chat completion',
    body: [
      { field: 'messages', type: 'array', required: true },
      { field: 'max_new_tokens', type: 'number', required: false },
      { field: 'temperature', type: 'number', required: false },
    ],
  },
  {
    method: 'WS',
    path: '/ws/generate',
    description: 'WebSocket endpoint for real-time generation',
  },
  {
    method: 'GET',
    path: '/models',
    description: 'List available models (local + HuggingFace)',
  },
  {
    method: 'POST',
    path: '/models/load',
    description: 'Load a HuggingFace model',
    body: [
      { field: 'model_id', type: 'string', required: true },
      { field: 'mode', type: 'string', required: false },
    ],
  },
  {
    method: 'GET',
    path: '/models/hf',
    description: 'List HuggingFace models',
  },
  {
    method: 'POST',
    path: '/load',
    description: 'Load inference model on-demand',
  },
  {
    method: 'GET',
    path: '/datasets',
    description: 'List available datasets',
  },
  {
    method: 'POST',
    path: '/train',
    description: 'Start a training job',
    body: [
      { field: 'dataset', type: 'string', required: true },
      { field: 'epochs', type: 'number', required: false },
      { field: 'batch_size', type: 'number', required: false },
      { field: 'learning_rate', type: 'number', required: false },
    ],
  },
  {
    method: 'GET',
    path: '/train/status',
    description: 'Get training status',
  },
  {
    method: 'GET',
    path: '/personalities',
    description: 'List available AI personalities',
  },
  {
    method: 'POST',
    path: '/inference/generate',
    description: 'Production inference engine - Generate text',
    body: [
      { field: 'prompt', type: 'string', required: true },
      { field: 'max_new_tokens', type: 'number', required: false },
      { field: 'temperature', type: 'number', required: false },
      { field: 'top_p', type: 'number', required: false },
      { field: 'top_k', type: 'number', required: false },
    ],
  },
  {
    method: 'POST',
    path: '/inference/generate/stream',
    description: 'Production inference - Streaming text (SSE)',
    body: [
      { field: 'prompt', type: 'string', required: true },
      { field: 'max_new_tokens', type: 'number', required: false },
      { field: 'temperature', type: 'number', required: false },
    ],
  },
  {
    method: 'POST',
    path: '/inference/quantize',
    description: 'Quantize loaded model (fp16, int8, etc.)',
    body: [
      { field: 'quantization_type', type: 'string', required: true },
    ],
  },
  {
    method: 'GET',
    path: '/inference/stats',
    description: 'Get inference engine statistics',
  },
  {
    method: 'POST',
    path: '/benchmark/run',
    description: 'Run inference benchmark',
    body: [
      { field: 'prompt', type: 'string', required: false },
      { field: 'max_new_tokens', type: 'number', required: false },
      { field: 'num_runs', type: 'number', required: false },
    ],
  },
  {
    method: 'POST',
    path: '/benchmark/perplexity',
    description: 'Calculate model perplexity',
    body: [
      { field: 'text', type: 'string', required: true },
    ],
  },
  {
    method: 'GET',
    path: '/benchmark/compare',
    description: 'Compare quantization levels',
  },
  {
    method: 'POST',
    path: '/model/export',
    description: 'Export current model to file',
    body: [
      { field: 'output_path', type: 'string', required: true },
      { field: 'format', type: 'string', required: false },
      { field: 'include_tokenizer', type: 'boolean', required: false },
    ],
  },
  {
    method: 'GET',
    path: '/model/export/formats',
    description: 'List supported export formats',
  },
  {
    method: 'GET',
    path: '/experiments',
    description: 'List all experiments',
  },
  {
    method: 'POST',
    path: '/experiments',
    description: 'Create new experiment',
    body: [
      { field: 'name', type: 'string', required: true },
      { field: 'description', type: 'string', required: false },
      { field: 'parameters', type: 'string', required: false },
    ],
  },
  {
    method: 'GET',
    path: '/experiments/{id}',
    description: 'Get specific experiment',
  },
  {
    method: 'POST',
    path: '/experiments/{id}/log_metric',
    description: 'Log a metric to experiment',
    body: [
      { field: 'metric_name', type: 'string', required: true },
      { field: 'value', type: 'number', required: true },
      { field: 'step', type: 'number', required: false },
    ],
  },
  {
    method: 'POST',
    path: '/experiments/{id}/log_param',
    description: 'Log a parameter to experiment',
    body: [
      { field: 'param_name', type: 'string', required: true },
      { field: 'value', type: 'any', required: true },
    ],
  },
  {
    method: 'POST',
    path: '/training/start',
    description: 'Start a new training job',
    body: [
      { field: 'name', type: 'string', required: true },
      { field: 'model', type: 'string', required: true },
      { field: 'dataset', type: 'string', required: true },
      { field: 'epochs', type: 'number', required: false },
      { field: 'batch_size', type: 'number', required: false },
      { field: 'learning_rate', type: 'number', required: false },
    ],
  },
  {
    method: 'GET',
    path: '/training/jobs',
    description: 'List all training jobs',
  },
]

export default function ApiDocsPage() {
  const [expanded, setExpanded] = useState<string | null>(null)

  const methodColors: Record<string, string> = {
    GET: 'bg-green-500/20 text-green-400',
    POST: 'bg-blue-500/20 text-blue-400',
    PUT: 'bg-yellow-500/20 text-yellow-400',
    DELETE: 'bg-red-500/20 text-red-400',
    WS: 'bg-purple-500/20 text-purple-400',
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-white mb-2">API Documentation</h1>
      <p className="text-zinc-400 mb-6">
        Base URL: <code className="bg-white/5 px-2 py-1 rounded">http://localhost:8000</code>
      </p>

      <div className="space-y-3">
        {endpoints.map((ep) => (
          <div
            key={ep.path}
            className="bg-white/5 border border-white/10 rounded-xl overflow-hidden"
          >
            <button
              onClick={() => setExpanded(expanded === ep.path ? null : ep.path)}
              className="w-full flex items-center gap-3 p-4 text-left hover:bg-white/5 transition-colors"
            >
              <span
                className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${methodColors[ep.method]}`}
              >
                {ep.method}
              </span>
              <code className="text-white font-mono">{ep.path}</code>
              <span className="text-zinc-500 text-sm flex-1">{ep.description}</span>
              <svg
                className={`w-4 h-4 text-zinc-400 transition-transform ${
                  expanded === ep.path ? 'rotate-180' : ''
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </button>

            {expanded === ep.path && ep.body && (
              <div className="px-4 pb-4 border-t border-white/10">
                <p className="text-sm text-zinc-400 mt-3 mb-2">Request Body:</p>
                <div className="bg-black/20 rounded-lg overflow-hidden">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/10">
                        <th className="px-3 py-2 text-left text-zinc-400 font-medium">Field</th>
                        <th className="px-3 py-2 text-left text-zinc-400 font-medium">Type</th>
                        <th className="px-3 py-2 text-left text-zinc-400 font-medium">Required</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ep.body.map((field) => (
                        <tr key={field.field} className="border-b border-white/5">
                          <td className="px-3 py-2 text-cyan-400 font-mono">{field.field}</td>
                          <td className="px-3 py-2 text-zinc-300">{field.type}</td>
                          <td className="px-3 py-2">
                            {field.required ? (
                              <span className="text-red-400">Yes</span>
                            ) : (
                              <span className="text-zinc-500">No</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="mt-8 p-4 bg-white/5 border border-white/10 rounded-xl">
        <h2 className="text-lg font-semibold text-white mb-2">Quick Examples</h2>
        <pre className="bg-black/30 rounded-lg p-3 text-sm text-zinc-300 overflow-x-auto">{`# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/inference/generate \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "Hello world", "max_new_tokens": 50}'

# Streaming (SSE)
curl -X POST http://localhost:8000/inference/generate/stream \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "Hello", "max_new_tokens": 100}'

# Run benchmark
curl -X POST "http://localhost:8000/benchmark/run?max_new_tokens=20" \\
  -H "Content-Type: application/json" -d '{}'

# Create experiment
curl -X POST "http://localhost:8000/experiments?name=test&description=Testing"

# Export model
curl -X POST http://localhost:8000/model/export \\
  -H "Content-Type: application/json" \\
  -d '{"output_path": "models/exported", "format": "torch"}'`}
        </pre>
      </div>
    </div>
  )
}
