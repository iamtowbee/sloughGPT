'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'

import { PUBLIC_API_URL } from '@/lib/config'

const features = [
  { title: 'Chat', icon: '💬', href: '/chat', desc: 'AI conversation' },
  { title: 'Models', icon: '🧠', href: '/models', desc: 'Model management' },
  { title: 'Training', icon: '⚡', href: '/training', desc: 'Train models' },
  { title: 'Datasets', icon: '📁', href: '/datasets', desc: 'Data sources' },
  { title: 'Monitor', icon: '📊', href: '/monitoring', desc: 'System metrics' },
  { title: 'API Docs', icon: '📖', href: '/api-docs', desc: 'Endpoints' },
]

export default function HomePage() {
  const [apiStatus, setApiStatus] = useState<'loading' | 'online' | 'offline'>('loading')
  const [modelCount, setModelCount] = useState(0)

  useEffect(() => {
    const checkApi = async () => {
      try {
        const res = await fetch(`${PUBLIC_API_URL}/health`)
        if (res.ok) {
          setApiStatus('online')
          const modelsRes = await fetch(`${PUBLIC_API_URL}/models`)
          const data = await modelsRes.json()
          setModelCount(data.models?.length || 0)
        } else {
          setApiStatus('offline')
        }
      } catch {
        setApiStatus('offline')
      }
    }
    checkApi()
  }, [])

  return (
    <div className="p-6 space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">SloughGPT</h1>
        <p className="text-zinc-400">Your AI platform for training and inference.</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <p className="text-sm text-zinc-400">API Status</p>
          <p className={`text-2xl font-bold ${
            apiStatus === 'online' ? 'text-green-400' :
            apiStatus === 'offline' ? 'text-red-400' : 'text-zinc-500'
          }`}>
            {apiStatus === 'loading' ? '...' :
             apiStatus === 'online' ? 'Online' : 'Offline'}
          </p>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <p className="text-sm text-zinc-400">Models</p>
          <p className="text-2xl font-bold text-white">{modelCount}</p>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <p className="text-sm text-zinc-400">Datasets</p>
          <p className="text-2xl font-bold text-white">13</p>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <p className="text-sm text-zinc-400">Version</p>
          <p className="text-2xl font-bold text-cyan-400">1.0</p>
        </div>
      </div>

      <div>
        <h2 className="text-lg font-semibold text-white mb-4">Quick Actions</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {features.map((f) => (
            <Link
              key={f.href}
              href={f.href}
              className="bg-white/5 border border-white/10 rounded-xl p-4 hover:border-blue-500/50 hover:bg-white/10 transition-all text-center group"
            >
              <span className="text-3xl mb-2 block">{f.icon}</span>
              <span className="text-white font-medium block">{f.title}</span>
              <span className="text-zinc-500 text-xs">{f.desc}</span>
            </Link>
          ))}
        </div>
      </div>

      <div className="bg-white/5 border border-white/10 rounded-xl p-4">
        <h2 className="text-lg font-semibold text-white mb-3">Quick Start</h2>
        <div className="space-y-2 text-sm font-mono">
          <p className="text-zinc-400">
            <span className="text-cyan-400">#</span> Start API server
          </p>
          <p className="text-white/80 bg-black/20 px-3 py-1 rounded">
            uvicorn server.main:app --port 8000
          </p>
          <p className="text-zinc-400 mt-2">
            <span className="text-cyan-400">#</span> Quick train & generate
          </p>
          <p className="text-white/80 bg-black/20 px-3 py-1 rounded">
            python cli.py quick --prompt "Hello"
          </p>
          <p className="text-zinc-400 mt-2">
            <span className="text-cyan-400">#</span> List models
          </p>
          <p className="text-white/80 bg-black/20 px-3 py-1 rounded">
            python cli.py models
          </p>
        </div>
      </div>
    </div>
  )
}
