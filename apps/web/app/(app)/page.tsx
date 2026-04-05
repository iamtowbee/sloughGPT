'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  IconApiDocs,
  IconChat,
  IconDatasets,
  IconModels,
  IconMonitor,
  IconTraining,
} from '@/components/icons/NavIcons'
import { useApiHealth } from '@/hooks/useApiHealth'
import { api } from '@/lib/api'
import { PUBLIC_API_URL, WEB_UI_VERSION } from '@/lib/config'

const features = [
  { title: 'Chat', Icon: IconChat, href: '/chat', desc: 'Talk to a loaded model' },
  { title: 'Models', Icon: IconModels, href: '/models', desc: 'Load & manage models' },
  { title: 'Training', Icon: IconTraining, href: '/training', desc: 'Jobs via HTTP API' },
  { title: 'Datasets', Icon: IconDatasets, href: '/datasets', desc: 'Corpus listing' },
  { title: 'Monitor', Icon: IconMonitor, href: '/monitoring', desc: 'System metrics' },
  { title: 'API Docs', Icon: IconApiDocs, href: '/api-docs', desc: 'Swagger / endpoints' },
] as const

export default function HomePage() {
  const { state: health } = useApiHealth()
  const [modelCount, setModelCount] = useState<number | null>(null)
  const [datasetCount, setDatasetCount] = useState<number | null>(null)

  const apiStatus = useMemo<'loading' | 'online' | 'offline'>(() => {
    if (health === null) return 'loading'
    if (health === 'offline') return 'offline'
    return 'online'
  }, [health])

  const inferenceReady = useMemo(() => {
    if (health === null || health === 'offline') return null
    return health.model_loaded
  }, [health])

  useEffect(() => {
    if (health === null || health === 'offline') {
      setModelCount(null)
      setDatasetCount(null)
      return
    }
    let cancelled = false
    ;(async () => {
      try {
        const [models, datasets] = await Promise.all([api.getModels(), api.getDatasets()])
        if (cancelled) return
        setModelCount(models.length)
        setDatasetCount(datasets.length)
      } catch {
        if (!cancelled) {
          setModelCount(null)
          setDatasetCount(null)
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [health])

  return (
    <div className="max-w-6xl space-y-10 p-8 md:p-10">
      <header className="space-y-2">
        <p className="text-xs font-mono uppercase tracking-[0.2em] text-muted-foreground">Overview</p>
        <h1 className="text-3xl font-semibold tracking-tight text-foreground md:text-4xl">SloughGPT</h1>
        <p className="max-w-2xl text-base leading-relaxed text-muted-foreground">
          Local training and inference: use this console when the API is running, or the CLI from the repo
          root for the same stack without the browser.
        </p>
      </header>

      {apiStatus === 'offline' ? (
        <Card className="border-amber-500/35 bg-amber-500/5" aria-live="polite">
          <CardHeader className="pb-2">
            <CardTitle className="text-base font-medium text-amber-800 dark:text-amber-400/95">
              API not reachable
            </CardTitle>
            <CardDescription className="text-foreground/80">
              The web app talks to <span className="font-mono text-foreground">{PUBLIC_API_URL}</span>. From the{' '}
              <strong>repository root</strong>, start the API first, then refresh this page.
            </CardDescription>
          </CardHeader>
        </Card>
      ) : null}

      <div className="grid grid-cols-2 gap-3 md:gap-4 lg:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-mono uppercase tracking-wider">API status</CardDescription>
            <div className="mt-2 min-h-8">
              {apiStatus === 'loading' ? (
                <span
                  className="block h-7 w-28 animate-pulse bg-muted"
                  aria-busy
                  aria-label="Checking API"
                />
              ) : (
                <p
                  className={`text-xl font-semibold tabular-nums tracking-tight ${
                    apiStatus === 'online' ? 'text-success' : 'text-destructive'
                  }`}
                >
                  {apiStatus === 'online' ? 'Online' : 'Offline'}
                </p>
              )}
              {apiStatus === 'online' && inferenceReady !== null ? (
                <p className="mt-1.5 text-xs leading-snug text-muted-foreground">
                  Inference weights: {inferenceReady ? 'loaded' : 'not loaded'}
                </p>
              ) : null}
            </div>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-mono uppercase tracking-wider">Models</CardDescription>
            <p className="mt-2 text-xl font-semibold tabular-nums tracking-tight text-foreground">
              {modelCount !== null ? modelCount : apiStatus === 'loading' ? '…' : '—'}
            </p>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-mono uppercase tracking-wider">Dataset folders</CardDescription>
            <p className="mt-2 text-xl font-semibold tabular-nums tracking-tight text-foreground">
              {datasetCount !== null ? datasetCount : apiStatus === 'loading' ? '…' : '—'}
            </p>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-mono uppercase tracking-wider">Web UI</CardDescription>
            <p className="mt-2 text-xl font-semibold tabular-nums tracking-tight text-primary">{WEB_UI_VERSION}</p>
          </CardHeader>
        </Card>
      </div>

      <section>
        <h2 className="mb-4 text-sm font-semibold tracking-tight text-foreground">Quick actions</h2>
        <div className="grid grid-cols-2 gap-3 sm:gap-4 md:grid-cols-3 xl:grid-cols-6">
          {features.map((f) => (
            <Button
              key={f.href}
              variant="outline"
              className="flex h-auto w-full min-w-0 flex-col items-start justify-start gap-3 p-3 text-left sm:p-4"
              asChild
            >
              <Link href={f.href} className="min-w-0">
                <f.Icon className="h-7 w-7 shrink-0 text-primary opacity-90" aria-hidden />
                <div className="flex min-w-0 flex-col gap-0.5">
                  <span className="text-sm font-medium leading-tight text-foreground">{f.title}</span>
                  <span className="text-balance text-xs font-normal leading-snug text-muted-foreground">
                    {f.desc}
                  </span>
                </div>
              </Link>
            </Button>
          ))}
        </div>
      </section>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Quick start (terminal)</CardTitle>
          <CardDescription>
            Run from the <strong>repository root</strong> (where <span className="font-mono">cli.py</span> and{' '}
            <span className="font-mono">pyproject.toml</span> live).
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 font-mono text-sm">
          {[
            { label: 'Guided onboarding', cmd: 'python3 cli.py start' },
            { label: 'Install & sanity check', cmd: 'python3 -m pip install -e ".[dev]" && python3 cli.py config check' },
            { label: 'Start API (this web app expects it)', cmd: 'python3 apps/api/server/main.py' },
            { label: 'Short training smoke (CLI, no browser)', cmd: 'make train-demo' },
            {
              label: 'One-shot generate (after a checkpoint or .sou exists)',
              cmd: 'python3 cli.py generate "Hello" --max-tokens 80',
            },
          ].map((row) => (
            <div key={row.cmd}>
              <p className="text-muted-foreground">
                <span className="text-primary">#</span> {row.label}
              </p>
              <p className="mt-1 break-all border border-border bg-muted/40 px-3 py-2 text-foreground/90">{row.cmd}</p>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  )
}
