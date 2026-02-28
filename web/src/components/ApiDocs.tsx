import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription
} from '@base-ui/react'

interface Endpoint {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH'
  path: string
  description: string
  category: string
}

const endpoints: Endpoint[] = [
  // Health
  { method: 'GET', path: '/health', description: 'Health check endpoint', category: 'Health' },
  { method: 'GET', path: '/metrics', description: 'System metrics', category: 'Health' },
  { method: 'GET', path: '/info', description: 'System information', category: 'Health' },
  
  // Conversations
  { method: 'POST', path: '/conversations', description: 'Create new conversation', category: 'Conversations' },
  { method: 'GET', path: '/conversations', description: 'List all conversations', category: 'Conversations' },
  { method: 'GET', path: '/conversations/{id}', description: 'Get conversation by ID', category: 'Conversations' },
  { method: 'DELETE', path: '/conversations/{id}', description: 'Delete conversation', category: 'Conversations' },
  
  // Chat
  { method: 'POST', path: '/chat', description: 'Send chat message', category: 'Chat' },
  { method: 'POST', path: '/chat/stream', description: 'Send chat message with streaming', category: 'Chat' },
  
  // Models
  { method: 'GET', path: '/models', description: 'List available models', category: 'Models' },
  { method: 'GET', path: '/models/{id}', description: 'Get model by ID', category: 'Models' },
  
  // Datasets
  { method: 'GET', path: '/datasets', description: 'List all datasets', category: 'Datasets' },
  { method: 'POST', path: '/datasets', description: 'Create new dataset', category: 'Datasets' },
  { method: 'GET', path: '/datasets/{name}', description: 'Get dataset by name', category: 'Datasets' },
  { method: 'DELETE', path: '/datasets/{name}', description: 'Delete dataset', category: 'Datasets' },
  
  // Training
  { method: 'GET', path: '/training', description: 'List training jobs', category: 'Training' },
  { method: 'POST', path: '/training', description: 'Create training job', category: 'Training' },
  { method: 'GET', path: '/training/{id}', description: 'Get training job', category: 'Training' },
  { method: 'DELETE', path: '/training/{id}', description: 'Cancel training job', category: 'Training' },
  
  // Generation
  { method: 'POST', path: '/generate', description: 'Generate text from prompt', category: 'Generation' },
  
  // WebSocket
  { method: 'WS', path: '/ws', description: 'WebSocket for real-time updates', category: 'WebSocket' }
]

const getMethodColor = (method: string) => {
  const colors: Record<string, string> = {
    GET: 'bg-green-100 text-green-800 border-green-200',
    POST: 'bg-blue-100 text-blue-800 border-blue-200',
    PUT: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    DELETE: 'bg-red-100 text-red-800 border-red-200',
    PATCH: 'bg-purple-100 text-purple-800 border-purple-200',
    WS: 'bg-orange-100 text-orange-800 border-orange-200'
  }
  return colors[method] || 'bg-gray-100 text-gray-800 border-gray-200'
}

export const ApiDocs: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')

  const categories = ['all', ...new Set(endpoints.map(e => e.category))]

  const filteredEndpoints = endpoints.filter(endpoint => {
    const matchesCategory = selectedCategory === 'all' || endpoint.category === selectedCategory
    const matchesSearch = endpoint.path.toLowerCase().includes(searchQuery.toLowerCase()) ||
      endpoint.description.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesCategory && matchesSearch
  })

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">API Documentation</CardTitle>
          <CardDescription>
            Complete reference for the SloughGPT API
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search endpoints..."
                className="w-full px-3 py-2 border border-slate-300 rounded-lg"
              />
            </div>
            <Link
              to="/settings"
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
            >
              Base URL: localhost:8000
            </Link>
          </div>
        </CardContent>
      </Card>

      {/* Categories */}
      <div className="flex gap-2 flex-wrap">
        {categories.map(category => (
          <button
            key={category}
            onClick={() => setSelectedCategory(category)}
            className={`px-3 py-1.5 rounded-full text-sm transition-colors ${
              selectedCategory === category
                ? 'bg-blue-500 text-white'
                : 'bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700'
            }`}
          >
            {category.charAt(0).toUpperCase() + category.slice(1)}
          </button>
        ))}
      </div>

      {/* Endpoints */}
      <div className="space-y-3">
        {filteredEndpoints.map((endpoint, index) => (
          <Card key={index}>
            <CardContent className="p-4">
              <div className="flex items-center gap-4">
                <span className={`px-2 py-1 rounded text-xs font-mono border ${getMethodColor(endpoint.method)}`}>
                  {endpoint.method}
                </span>
                <code className="flex-1 font-mono text-sm dark:text-white">
                  {endpoint.path}
                </code>
                <span className="text-sm text-slate-500">
                  {endpoint.description}
                </span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Quick Links */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Links</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-3">
          <Link to="/chat" className="px-3 py-1.5 bg-slate-100 rounded-lg hover:bg-slate-200">
            💬 Try Chat API
          </Link>
          <Link to="/datasets" className="px-3 py-1.5 bg-slate-100 rounded-lg hover:bg-slate-200">
            📊 Try Datasets API
          </Link>
          <Link to="/training" className="px-3 py-1.5 bg-slate-100 rounded-lg hover:bg-slate-200">
            🧠 Try Training API
          </Link>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="px-3 py-1.5 bg-blue-100 rounded-lg hover:bg-blue-200"
          >
            📚 OpenAPI Docs ↗
          </a>
        </CardContent>
      </Card>
    </div>
  )
}

export default ApiDocs
