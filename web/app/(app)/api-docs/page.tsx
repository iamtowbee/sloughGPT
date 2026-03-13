export default function ApiDocsPage() {
  const endpoints = [
    { method: 'GET', path: '/health', description: 'Check API health status' },
    { method: 'POST', path: '/inference/generate', description: 'Generate text with a model' },
    { method: 'GET', path: '/models', description: 'List available models' },
    { method: 'POST', path: '/training/start', description: 'Start a training job' },
    { method: 'GET', path: '/training/jobs', description: 'List training jobs' },
    { method: 'GET', path: '/datasets', description: 'List available datasets' },
  ]

  return (
    <div>
      <h1 className="text-3xl font-bold text-slate-800 dark:text-white mb-6">API Documentation</h1>

      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
        <div className="p-4 bg-slate-50 dark:bg-slate-700 border-b border-slate-200 dark:border-slate-700">
          <h2 className="font-semibold text-slate-800 dark:text-white">Base URL</h2>
          <code className="text-sm text-blue-600">http://localhost:8000</code>
        </div>
        
        <div className="divide-y divide-slate-200 dark:divide-slate-700">
          {endpoints.map((endpoint, i) => (
            <div key={i} className="p-4">
              <div className="flex items-center gap-3 mb-2">
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  endpoint.method === 'GET' ? 'bg-green-100 text-green-700' :
                  endpoint.method === 'POST' ? 'bg-blue-100 text-blue-700' :
                  'bg-red-100 text-red-700'
                }`}>
                  {endpoint.method}
                </span>
                <code className="text-sm text-slate-800 dark:text-white">{endpoint.path}</code>
              </div>
              <p className="text-sm text-slate-500">{endpoint.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
