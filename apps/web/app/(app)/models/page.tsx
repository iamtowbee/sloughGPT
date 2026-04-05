'use client'

import { useState, useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { api } from '@/lib/api'
import { PUBLIC_API_URL } from '@/lib/config'

interface Model {
  id: string
  name: string
  source?: string
  description?: string
  size_mb?: number
  tags?: string[]
}

export default function ModelsPage() {
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(true)
  const [loadingModel, setLoadingModel] = useState<string | null>(null)
  const [filter, setFilter] = useState<'all' | 'local' | 'huggingface'>('all')
  const [apiHealth, setApiHealth] = useState<string>('checking...')
  const [loadDialog, setLoadDialog] = useState<{ open: boolean; title: string; body: string }>({
    open: false,
    title: '',
    body: '',
  })

  useEffect(() => {
    checkHealth()
    fetchModels()
  }, [])

  const checkHealth = async () => {
    const h = await api.getHealth()
    if (!h) {
      setApiHealth('disconnected')
      return
    }
    if (h.model_loaded) {
      setApiHealth(`inference ready · ${h.model_type}`)
    } else {
      setApiHealth(`connected · no weights (${h.model_type})`)
    }
  }

  const fetchModels = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${PUBLIC_API_URL}/models`)
      const data = await res.json()
      setModels(data.models || [])
    } catch (err) {
      console.error('Failed to fetch models:', err)
      setModels([])
    } finally {
      setLoading(false)
    }
  }

  const loadModel = async (modelId: string) => {
    setLoadingModel(modelId)
    try {
      const data = await api.loadModel(modelId)
      setLoadDialog({
        open: true,
        title: 'Load model',
        body: `${data.status ?? 'ok'}: ${data.model ?? modelId}${data.effective_device != null ? ` (${data.effective_device})` : ''}`,
      })
    } catch (err) {
      setLoadDialog({
        open: true,
        title: 'Load failed',
        body: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setLoadingModel(null)
      void checkHealth()
    }
  }

  const filteredModels = models.filter((m) => {
    if (filter === 'all') return true
    if (filter === 'local') return m.source === 'local'
    if (filter === 'huggingface') return m.source === 'huggingface'
    return true
  })

  const sourceBadge = (src: string) =>
    src === 'local'
      ? 'border border-success/30 bg-success/10 text-success'
      : 'border border-primary/30 bg-primary/10 text-primary'

  return (
    <div className="sl-page mx-auto max-w-6xl">
      <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="sl-h1">Models</h1>
          <p className="mt-1 text-muted-foreground">
            API:{' '}
            <span
              className={
                apiHealth === 'checking...'
                  ? 'text-muted-foreground'
                  : apiHealth === 'disconnected'
                    ? 'text-destructive'
                    : apiHealth.startsWith('connected ·')
                      ? 'text-warning'
                      : 'text-success'
              }
            >
              {apiHealth}
            </span>
          </p>
        </div>
        <Button type="button" variant="secondary" size="sm" onClick={() => void fetchModels()}>
          Refresh list
        </Button>
      </div>

      <div className="mb-6 inline-flex flex-wrap gap-0 border border-border bg-muted/30 p-0.5">
        {(['all', 'local', 'huggingface'] as const).map((f) => (
          <Button
            key={f}
            type="button"
            variant={filter === f ? 'default' : 'ghost'}
            size="sm"
            className="rounded-none px-4"
            onClick={() => setFilter(f)}
          >
            {f === 'all' ? 'All' : f === 'local' ? 'Local' : 'HuggingFace'}
          </Button>
        ))}
      </div>

      {loading ? (
        <div className="py-12 text-center text-muted-foreground">Loading models…</div>
      ) : filteredModels.length === 0 ? (
        <Card className="border-dashed">
          <CardContent className="py-12 text-center text-muted-foreground">
            No {filter === 'all' ? '' : filter} models found.
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredModels.map((model) => (
            <Card key={model.id} className="transition-colors duration-200 ease-smooth hover:border-primary/30">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-2">
                  <CardTitle className="text-base leading-snug">{model.name}</CardTitle>
                  {model.source && (
                    <span className={`shrink-0 px-2 py-0.5 text-xs font-medium ${sourceBadge(model.source)}`}>
                      {model.source}
                    </span>
                  )}
                </div>
                {model.description && (
                  <p className="text-sm text-muted-foreground line-clamp-2">{model.description}</p>
                )}
              </CardHeader>
              <CardContent className="pt-0">
                {model.tags && model.tags.length > 0 && (
                  <div className="mb-3 flex flex-wrap gap-1">
                    {model.tags.slice(0, 4).map((tag) => (
                      <span
                        key={tag}
                        className="border border-border bg-muted/50 px-2 py-0.5 text-xs text-muted-foreground"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </CardContent>
              <CardFooter className="flex justify-between border-t border-border pt-4">
                <span className="text-sm text-muted-foreground">
                  {model.size_mb ? `${model.size_mb.toFixed(1)} MB` : model.id}
                </span>
                <Button
                  type="button"
                  size="sm"
                  onClick={() => loadModel(model.id)}
                  disabled={loadingModel !== null}
                >
                  {loadingModel === model.id ? 'Loading…' : 'Load'}
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      )}

      <Card className="mt-8">
        <CardHeader>
          <CardTitle className="text-base">API endpoints</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-2 text-sm md:grid-cols-4">
            <code className="sl-code block p-2">GET /models</code>
            <code className="sl-code block p-2">POST /models/load</code>
            <code className="sl-code block p-2">GET /models/hf</code>
            <code className="sl-code block p-2">GET /health</code>
          </div>
        </CardContent>
      </Card>

      <Dialog open={loadDialog.open} onOpenChange={(o) => setLoadDialog((d) => ({ ...d, open: o }))}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{loadDialog.title}</DialogTitle>
            <DialogDescription className="font-mono text-xs text-foreground/90">{loadDialog.body}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button type="button" onClick={() => setLoadDialog((d) => ({ ...d, open: false }))}>
              OK
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
