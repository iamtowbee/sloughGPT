'use client'

import { useState } from 'react'

import { PUBLIC_API_URL } from '@/lib/config'

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
    path: '/v1/infer',
    description:
      'SloughGPT Standard v1 inference envelope. Optional header X-SloughGPT-Standard: 1. See standards/SLOUGHGPT_STANDARD_V1.md.',
    body: [
      { field: 'mode', type: "'generate' | 'chat' | 'structured'", required: true },
      { field: 'task_type', type: 'string', required: false },
      { field: 'model_id', type: 'string', required: false },
      { field: 'input', type: 'object (prompt?, messages?, context?)', required: true },
      { field: 'generation', type: 'object', required: false },
      { field: 'retrieval', type: 'object', required: false },
      { field: 'output_schema_ref', type: 'string', required: false },
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
    description:
      'Start char-level training in the background. Exactly one of: dataset (folder under datasets/), manifest_uri, or dataset_ref.',
    body: [
      { field: 'dataset', type: 'string', required: false },
      { field: 'manifest_uri', type: 'string', required: false },
      { field: 'dataset_ref.dataset_id', type: 'string', required: false },
      { field: 'dataset_ref.version', type: 'string', required: false },
      { field: 'dataset_ref.manifest_uri', type: 'string', required: false },
      { field: 'epochs', type: 'number', required: false },
      { field: 'batch_size', type: 'number', required: false },
      { field: 'learning_rate', type: 'number', required: false },
      { field: 'n_embed', type: 'number', required: false },
      { field: 'n_layer', type: 'number', required: false },
      { field: 'n_head', type: 'number', required: false },
      { field: 'block_size', type: 'number', required: false },
      { field: 'max_steps', type: 'number', required: false },
    ],
  },
  {
    method: 'POST',
    path: '/train/resolve',
    description:
      'Resolve corpus to data_path and checkpoint stem (dry run; no training). Same mutually exclusive source fields as POST /train.',
    body: [
      { field: 'dataset', type: 'string', required: false },
      { field: 'manifest_uri', type: 'string', required: false },
      { field: 'dataset_ref.dataset_id', type: 'string', required: false },
      { field: 'dataset_ref.version', type: 'string', required: false },
      { field: 'dataset_ref.manifest_uri', type: 'string', required: false },
    ],
  },
  {
    method: 'GET',
    path: '/train/status',
    description: 'Get training status',
  },
  {
    method: 'GET',
    path: '/metrics/prometheus',
    description: 'Prometheus text exposition format (HTTP + SloughGPT metrics when enabled)',
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
    description:
      'Start a tracked training job (web UI). Exactly one of: dataset, manifest_uri, or nested dataset_ref.',
    body: [
      { field: 'name', type: 'string', required: true },
      { field: 'model', type: 'string', required: true },
      { field: 'dataset', type: 'string', required: false },
      { field: 'manifest_uri', type: 'string', required: false },
      { field: 'dataset_ref.dataset_id', type: 'string', required: false },
      { field: 'dataset_ref.version', type: 'string', required: false },
      { field: 'dataset_ref.manifest_uri', type: 'string', required: false },
      { field: 'epochs', type: 'number', required: false },
      { field: 'batch_size', type: 'number', required: false },
      { field: 'learning_rate', type: 'number', required: false },
      { field: 'n_embed', type: 'number', required: false },
      { field: 'n_layer', type: 'number', required: false },
      { field: 'n_head', type: 'number', required: false },
      { field: 'block_size', type: 'number', required: false },
      { field: 'max_steps', type: 'number', required: false },
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
    GET: 'bg-success/15 text-success border border-success/25',
    POST: 'bg-primary/15 text-primary border border-primary/25',
    PUT: 'bg-warning/15 text-warning border border-warning/25',
    DELETE: 'bg-destructive/15 text-destructive border border-destructive/25',
    WS: 'bg-chart-4/15 text-chart-4 border border-chart-4/25',
  }

  return (
    <div className="sl-page max-w-4xl mx-auto">
      <h1 className="sl-h1 mb-2">API Documentation</h1>
      <p className="text-muted-foreground mb-6">
        Base URL: <code className="sl-code">{PUBLIC_API_URL}</code>
      </p>

      <div className="space-y-3">
        {endpoints.map((ep) => (
          <div key={ep.path} className="sl-card overflow-hidden p-0 ring-1 ring-primary/5">
            <button
              type="button"
              onClick={() => setExpanded(expanded === ep.path ? null : ep.path)}
              className="w-full flex items-center gap-3 p-4 text-left hover:bg-muted/40 transition-colors"
            >
              <span
                className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${methodColors[ep.method]}`}
              >
                {ep.method}
              </span>
              <code className="text-foreground font-mono text-sm">{ep.path}</code>
              <span className="text-muted-foreground text-sm flex-1">{ep.description}</span>
              <svg
                className={`w-4 h-4 text-muted-foreground transition-transform ${
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
              <div className="px-4 pb-4 border-t border-border">
                <p className="text-sm text-muted-foreground mt-3 mb-2">Request Body:</p>
                <div className="bg-muted/40 border border-border rounded-lg overflow-hidden">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="px-3 py-2 text-left text-muted-foreground font-medium">Field</th>
                        <th className="px-3 py-2 text-left text-muted-foreground font-medium">Type</th>
                        <th className="px-3 py-2 text-left text-muted-foreground font-medium">Required</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ep.body.map((field) => (
                        <tr key={field.field} className="border-b border-border/60">
                          <td className="px-3 py-2 text-primary font-mono">{field.field}</td>
                          <td className="px-3 py-2 text-foreground">{field.type}</td>
                          <td className="px-3 py-2">
                            {field.required ? (
                              <span className="text-destructive font-medium">Yes</span>
                            ) : (
                              <span className="text-muted-foreground">No</span>
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

      <div className="mt-8 sl-card p-4">
        <h2 className="sl-h2 mb-2">Quick Examples</h2>
        <pre className="bg-muted/50 border border-border rounded-lg p-3 text-sm text-foreground overflow-x-auto font-mono">{`# Health check
curl ${PUBLIC_API_URL}/health

# Generate text
curl -X POST ${PUBLIC_API_URL}/inference/generate \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "Hello world", "max_new_tokens": 50}'

# Streaming (SSE)
curl -X POST ${PUBLIC_API_URL}/inference/generate/stream \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "Hello", "max_new_tokens": 100}'

# Run benchmark
curl -X POST "${PUBLIC_API_URL}/benchmark/run?max_new_tokens=20" \\
  -H "Content-Type: application/json" -d '{}'

# Create experiment
curl -X POST "${PUBLIC_API_URL}/experiments?name=test&description=Testing"

# Export model
curl -X POST ${PUBLIC_API_URL}/model/export \\
  -H "Content-Type: application/json" \\
  -d '{"output_path": "models/exported", "format": "torch"}'`}
        </pre>
      </div>
    </div>
  )
}
