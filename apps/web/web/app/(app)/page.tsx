'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'

import {
  IconApiDocs,
  IconChat,
  IconDatasets,
  IconModels,
  IconMonitor,
  IconTraining,
} from '@/components/icons/NavIcons'
import { PUBLIC_API_URL } from '@/lib/config'

const features = [
  { title: 'Chat', Icon: IconChat, href: '/chat', desc: 'AI conversation' },
  { title: 'Models', Icon: IconModels, href: '/models', desc: 'Model management' },
  { title: 'Training', Icon: IconTraining, href: '/training', desc: 'Train models' },
  { title: 'Datasets', Icon: IconDatasets, href: '/datasets', desc: 'Data sources' },
  { title: 'Monitor', Icon: IconMonitor, href: '/monitoring', desc: 'System metrics' },
  { title: 'API Docs', Icon: IconApiDocs, href: '/api-docs', desc: 'Endpoints' },
] as const

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
    <div className="p-8 md:p-10 max-w-6xl space-y-10">
      <header className="space-y-2">
        <p className="text-xs font-mono uppercase tracking-[0.2em] text-muted-foreground">Overview</p>
        <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground">SloughGPT</h1>
        <p className="text-muted-foreground max-w-xl text-base leading-relaxed">
          Training and inference in one console—models, jobs, and API health at a glance.
        </p>
      </header>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 md:gap-4">
        <div className="rounded-xl border border-border bg-card/50 backdrop-blur-sm p-4 ring-1 ring-white/[0.04]">
          <p className="text-xs font-mono uppercase tracking-wider text-muted-foreground">API status</p>
          <div className="mt-2 min-h-8 flex items-center">
            {apiStatus === 'loading' ? (
              <span
                className="h-7 w-28 rounded-md bg-muted/80 animate-pulse"
                aria-busy
                aria-label="Checking API"
              />
            ) : (
              <p
                className={`text-xl font-semibold tracking-tight ${
                  apiStatus === 'online' ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {apiStatus === 'online' ? 'Online' : 'Offline'}
              </p>
            )}
          </div>
        </div>
        <div className="rounded-xl border border-border bg-card/50 backdrop-blur-sm p-4 ring-1 ring-white/[0.04]">
          <p className="text-xs font-mono uppercase tracking-wider text-muted-foreground">Models</p>
          <p className="mt-2 text-xl font-semibold tracking-tight text-foreground tabular-nums">{modelCount}</p>
        </div>
        <div className="rounded-xl border border-border bg-card/50 backdrop-blur-sm p-4 ring-1 ring-white/[0.04]">
          <p className="text-xs font-mono uppercase tracking-wider text-muted-foreground">Datasets</p>
          <p className="mt-2 text-xl font-semibold tracking-tight text-foreground tabular-nums">13</p>
        </div>
        <div className="rounded-xl border border-border bg-card/50 backdrop-blur-sm p-4 ring-1 ring-white/[0.04]">
          <p className="text-xs font-mono uppercase tracking-wider text-muted-foreground">Version</p>
          <p className="mt-2 text-xl font-semibold tracking-tight text-primary tabular-nums">1.0</p>
        </div>
      </div>

      <section>
        <h2 className="text-sm font-semibold text-foreground mb-4 tracking-tight">Quick actions</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {features.map((f) => (
            <Link
              key={f.href}
              href={f.href}
              className="group rounded-xl border border-border bg-card/40 hover:bg-card/70 backdrop-blur-sm p-4 transition-colors ring-1 ring-white/[0.04] hover:ring-primary/25 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            >
              <f.Icon className="w-7 h-7 text-primary mb-3 opacity-90 group-hover:opacity-100 transition-opacity" />
              <span className="text-foreground font-medium text-sm block">{f.title}</span>
              <span className="text-muted-foreground text-xs mt-0.5 block">{f.desc}</span>
            </Link>
          ))}
        </div>
      </section>

      <section className="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-5 ring-1 ring-white/[0.04]">
        <h2 className="text-sm font-semibold text-foreground mb-4 tracking-tight">Quick start</h2>
        <div className="space-y-3 text-sm font-mono">
          <div>
            <p className="text-muted-foreground">
              <span className="text-primary">#</span> Start API server
            </p>
            <p className="text-foreground/90 bg-muted/40 border border-border px-3 py-2 rounded-lg mt-1">
              uvicorn server.main:app --port 8000
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">
              <span className="text-primary">#</span> Quick train &amp; generate
            </p>
            <p className="text-foreground/90 bg-muted/40 border border-border px-3 py-2 rounded-lg mt-1">
              python cli.py quick --prompt &quot;Hello&quot;
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">
              <span className="text-primary">#</span> List models
            </p>
            <p className="text-foreground/90 bg-muted/40 border border-border px-3 py-2 rounded-lg mt-1">
              python cli.py models
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
