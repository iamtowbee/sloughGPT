'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'

import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
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
import { inferenceHealthLabel, useApiHealth } from '@/hooks/useApiHealth'
import { api } from '@/lib/api'
import { catalogIdMatchesRuntime } from '@/lib/inference-display'
import { devDebug } from '@/lib/dev-log'
import { ToastContainer, type Toast } from '@/components/chat/Toast'
import { cn } from '@/lib/cn'

interface Model {
  id: string
  name: string
  source?: string
  description?: string
  size_mb?: number
  tags?: string[]
  thumbnail?: string
  params?: string
}

function ModelCardSkeleton() {
  return (
    <div className="animate-pulse rounded-xl bg-muted/50 border border-border/50 overflow-hidden">
      <div className="p-5">
        <div className="flex items-start justify-between mb-4">
          <div className="h-6 w-32 bg-muted rounded"></div>
          <div className="h-5 w-16 bg-muted rounded-full"></div>
        </div>
        <div className="space-y-2 mb-4">
          <div className="h-4 w-full bg-muted rounded"></div>
          <div className="h-4 w-3/4 bg-muted rounded"></div>
        </div>
        <div className="flex gap-2">
          <div className="h-5 w-16 bg-muted rounded-full"></div>
          <div className="h-5 w-20 bg-muted rounded-full"></div>
        </div>
      </div>
      <div className="px-5 py-3 border-t border-border/50 flex justify-between items-center bg-muted/30">
        <div className="h-4 w-20 bg-muted rounded"></div>
        <div className="h-8 w-16 bg-muted rounded-lg"></div>
      </div>
    </div>
  )
}

function ModelIcon({ modelId, size = 'lg' }: { modelId: string; size?: 'sm' | 'md' | 'lg' }) {
  const iconSize = size === 'lg' ? 'w-12 h-12' : size === 'md' ? 'w-10 h-10' : 'w-8 h-8'
  const textSize = size === 'lg' ? 'text-2xl' : size === 'md' ? 'text-xl' : 'text-lg'
  
  const lowerId = modelId.toLowerCase()
  
  let gradient = 'from-blue-500 to-cyan-500'
  let emoji = '🤖'
  
  if (lowerId.includes('gpt')) {
    gradient = 'from-green-500 to-emerald-500'
    emoji = '🧠'
  } else if (lowerId.includes('llama')) {
    gradient = 'from-orange-500 to-amber-500'
    emoji = '🦙'
  } else if (lowerId.includes('mistral')) {
    gradient = 'from-violet-500 to-purple-500'
    emoji = '🌫️'
  } else if (lowerId.includes('phi')) {
    gradient = 'from-pink-500 to-rose-500'
    emoji = 'Φ'
  } else if (lowerId.includes('bert') || lowerId.includes('albert')) {
    gradient = 'from-teal-500 to-green-500'
    emoji = '📚'
  } else if (lowerId.includes('t5') || lowerId.includes('flan')) {
    gradient = 'from-yellow-500 to-orange-500'
    emoji = '🔢'
  } else if (lowerId.includes('gemma')) {
    gradient = 'from-indigo-500 to-blue-500'
    emoji = '💎'
  } else if (lowerId.includes('qwen')) {
    gradient = 'from-red-500 to-pink-500'
    emoji = '🐉'
  } else if (lowerId.includes('babbage') || lowerId.includes('davinci') || lowerId.includes('gpt2')) {
    gradient = 'from-amber-500 to-yellow-500'
    emoji = '✨'
  }
  
  return (
    <div className={cn('rounded-xl bg-gradient-to-br flex items-center justify-center shadow-sm', gradient, iconSize)}>
      <span className={textSize}>{emoji}</span>
    </div>
  )
}

export default function ModelsPage() {
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(true)
  const [loadingModel, setLoadingModel] = useState<string | null>(null)
  const [filter, setFilter] = useState<'all' | 'local' | 'huggingface'>('all')
  const { state: health, refresh: refreshHealth } = useApiHealth()
  const [loadDialog, setLoadDialog] = useState<{ open: boolean; title: string; body: string }>({
    open: false,
    title: '',
    body: '',
  })

  const [toasts, setToasts] = useState<Toast[]>([])
  const addToast = useCallback((message: string, type: Toast['type'] = 'info') => {
    const id = Date.now().toString()
    setToasts(prev => [...prev, { id, message, type }])
  }, [])
  const dismissToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  const [souls, setSouls] = useState<{ name: string; path: string; description: string; personality: Record<string, number>; traits: string[] }[]>([])
  const [currentSoul, setCurrentSoul] = useState<string | null>(null)
  const [switchingSoul, setSwitchingSoul] = useState<string | null>(null)

  const fetchSouls = useCallback(async () => {
    try {
      const data = await api.getSouls()
      setSouls(data.souls)
      setCurrentSoul(data.current_soul)
    } catch (err) {
      devDebug('Failed to fetch souls:', err)
    }
  }, [])

  const switchSoul = async (name: string) => {
    setSwitchingSoul(name)
    try {
      await api.switchSoul(name)
      setCurrentSoul(name)
      addToast(`Switched to ${name}`, 'success')
    } catch (err) {
      devDebug('Failed to switch soul:', err)
      addToast(err instanceof Error ? err.message : 'Failed to switch soul', 'error')
    } finally {
      setSwitchingSoul(null)
    }
  }

  useEffect(() => {
    void fetchSouls()
  }, [fetchSouls])

  const apiHealthLabel = useMemo(() => inferenceHealthLabel(health), [health])

  const healthToneClass = useMemo(() => {
    if (health === null) return 'text-muted-foreground'
    if (health === 'offline') return 'text-destructive'
    if (health.model_loaded) return 'text-success'
    return 'text-warning'
  }, [health])

  useEffect(() => {
    void fetchModels()
  }, [])

  const fetchModels = async () => {
    setLoading(true)
    try {
      const rows = await api.getModels()
      setModels(
        rows.map((m) => ({
          id: m.id,
          name: m.name,
          source: m.type,
          description: m.description,
          tags: m.tags,
          size_mb: m.size_mb,
          params: m.params,
        })),
      )
    } catch (err) {
      devDebug('Failed to fetch models:', err)
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
        title: 'Model Loaded',
        body: `${data.status ?? 'ok'}: ${data.model ?? modelId}${data.effective_device != null ? ` (${data.effective_device})` : ''}`,
      })
    } catch (err) {
      setLoadDialog({
        open: true,
        title: 'Load Failed',
        body: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setLoadingModel(null)
      void refreshHealth()
    }
  }

  const filteredModels = models.filter((m) => {
    if (filter === 'all') return true
    if (filter === 'local') return m.source === 'local'
    if (filter === 'huggingface') return m.source === 'huggingface'
    return true
  })

  const activeRuntimeId =
    health !== null && health !== 'offline' && health.model_loaded ? health.model_type : null

  const sourceBadge = (src: string) =>
    src === 'local'
      ? 'border border-success/30 bg-success/10 text-success'
      : 'border border-primary/30 bg-primary/10 text-primary'

  const getModelTypeBadge = (modelId: string) => {
    const lowerId = modelId.toLowerCase()
    if (lowerId.includes('gpt')) return { label: 'GPT', class: 'bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/30' }
    if (lowerId.includes('llama')) return { label: 'LLaMA', class: 'bg-orange-500/10 text-orange-600 dark:text-orange-400 border-orange-500/30' }
    if (lowerId.includes('mistral')) return { label: 'Mistral', class: 'bg-violet-500/10 text-violet-600 dark:text-violet-400 border-violet-500/30' }
    if (lowerId.includes('phi')) return { label: 'Phi', class: 'bg-pink-500/10 text-pink-600 dark:text-pink-400 border-pink-500/30' }
    if (lowerId.includes('gemma')) return { label: 'Gemma', class: 'bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border-indigo-500/30' }
    return null
  }

  return (
    <div className="sl-page mx-auto max-w-6xl">
      <AppRouteHeader
        className="mb-6 items-start"
        left={
          <AppRouteHeaderLead
            title="Models"
            subtitle={
              <>
                API:{' '}
                <span className={healthToneClass} data-testid="models-api-status">
                  {apiHealthLabel}
                </span>
              </>
            }
          />
        }
        right={
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => {
              void fetchModels()
              void refreshHealth()
            }}
          >
            <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Refresh
          </Button>
        }
      />

      <div className="mb-6 inline-flex flex-wrap gap-0 border border-border bg-muted/30 p-0.5 rounded-lg">
        {(['all', 'local', 'huggingface'] as const).map((f) => (
          <Button
            key={f}
            type="button"
            variant={filter === f ? 'default' : 'ghost'}
            size="sm"
            className={cn(
              "rounded-md px-4 transition-all",
              filter === f && "shadow-sm"
            )}
            onClick={() => setFilter(f)}
          >
            {f === 'all' ? 'All' : f === 'local' ? 'Local' : 'HuggingFace'}
            {f === 'all' && ` (${models.length})`}
            {f === 'local' && ` (${models.filter(m => m.source === 'local').length})`}
            {f === 'huggingface' && ` (${models.filter(m => m.source === 'huggingface').length})`}
          </Button>
        ))}
      </div>

      {souls.length > 0 && (
        <Card className="mb-6 border-primary/20 bg-gradient-to-br from-primary/5 to-transparent">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary/20 to-violet-500/20 flex items-center justify-center">
                  <svg className="w-4 h-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <CardTitle className="text-base">Soul Personality</CardTitle>
              </div>
              {currentSoul && (
                <span className="text-xs text-muted-foreground">
                  Active: <span className="text-primary font-medium">{currentSoul}</span>
                </span>
              )}
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="flex flex-wrap gap-2 mb-4">
              {souls.map((soul) => (
                <Button
                  key={soul.name}
                  size="sm"
                  variant={currentSoul === soul.name ? 'default' : 'outline'}
                  onClick={() => void switchSoul(soul.name)}
                  disabled={switchingSoul !== null}
                  className={cn(
                    "gap-1.5 transition-all",
                    currentSoul === soul.name && "shadow-md"
                  )}
                >
                  {switchingSoul === soul.name ? (
                    <>
                      <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Switching...
                    </>
                  ) : soul.name}
                </Button>
              ))}
            </div>
            {currentSoul && (
              <div className="p-4 rounded-lg bg-muted/30 border border-border/50">
                <p className="text-sm text-muted-foreground mb-2">
                  {souls.find(s => s.name === currentSoul)?.description || 'Soul personality'}
                </p>
                {souls.find(s => s.name === currentSoul)?.traits && (
                  <div className="flex flex-wrap gap-1.5">
                    {souls.find(s => s.name === currentSoul)?.traits.map((trait: string) => (
                      <span key={trait} className="text-xs px-2.5 py-1 rounded-full bg-primary/10 text-primary border border-primary/20">
                        {trait}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {loading ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <ModelCardSkeleton key={i} />
          ))}
        </div>
      ) : filteredModels.length === 0 ? (
        <Card className="border-dashed">
          <CardContent className="py-12 text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-muted/50 flex items-center justify-center">
              <svg className="w-8 h-8 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
              </svg>
            </div>
            <p className="text-muted-foreground">No {filter === 'all' ? '' : filter} models found.</p>
            <Button variant="ghost" size="sm" onClick={() => setFilter('all')} className="mt-2">
              View all models
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredModels.map((model) => {
            const isRuntimeActive =
              activeRuntimeId != null && catalogIdMatchesRuntime(model.id, activeRuntimeId)
            const typeBadge = getModelTypeBadge(model.id)
            
            return (
              <Card
                key={model.id}
                data-testid={isRuntimeActive ? 'model-card-active-runtime' : undefined}
                className={cn(
                  "group transition-all duration-300 ease-out hover:shadow-lg hover:-translate-y-1",
                  "border-border/60 hover:border-primary/30",
                  isRuntimeActive && 'border-primary/50 bg-primary/5 ring-1 ring-primary/20 shadow-primary/10'
                )}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-start gap-4">
                    <ModelIcon modelId={model.id} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2 mb-1">
                        <CardTitle className="text-base leading-tight truncate">{model.name}</CardTitle>
                        {model.source && (
                          <span className={cn("shrink-0 px-2 py-0.5 text-xs font-medium rounded-full", sourceBadge(model.source))}>
                            {model.source}
                          </span>
                        )}
                      </div>
                      {typeBadge && (
                        <span className={cn("inline-block text-xs px-2 py-0.5 rounded-full border mb-2", typeBadge.class)}>
                          {typeBadge.label}
                        </span>
                      )}
                    </div>
                  </div>
                  {model.description && (
                    <p className="text-sm text-muted-foreground line-clamp-2 mt-2">{model.description}</p>
                  )}
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="flex items-center gap-4 text-xs text-muted-foreground mb-3">
                    {model.size_mb && (
                      <span className="flex items-center gap-1">
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                        </svg>
                        {model.size_mb < 1 ? `${(model.size_mb * 1024).toFixed(0)} KB` : `${model.size_mb.toFixed(1)} MB`}
                      </span>
                    )}
                    {model.params && (
                      <span className="flex items-center gap-1">
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                        {model.params}
                      </span>
                    )}
                  </div>
                  {model.tags && model.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {model.tags.slice(0, 4).map((tag) => (
                        <span
                          key={tag}
                          className="border border-border/60 bg-muted/50 px-2 py-0.5 text-xs text-muted-foreground/80 rounded-md"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </CardContent>
                <CardFooter className="flex justify-between items-center border-t border-border/50 pt-4">
                  <span className="text-xs text-muted-foreground truncate max-w-[120px]" title={model.id}>
                    {model.id.length > 20 ? model.id.slice(0, 20) + '...' : model.id}
                  </span>
                  <Button
                    type="button"
                    size="sm"
                    className={cn(
                      "transition-all",
                      isRuntimeActive && "bg-success hover:bg-success/90",
                      loadingModel === model.id && "opacity-70"
                    )}
                    onClick={() => loadModel(model.id)}
                    disabled={loadingModel !== null || isRuntimeActive}
                  >
                    {loadingModel === model.id ? (
                      <>
                        <svg className="w-3 h-3 mr-1.5 animate-spin" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                        Loading
                      </>
                    ) : isRuntimeActive ? (
                      <>
                        <svg className="w-3.5 h-3.5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        Active
                      </>
                    ) : (
                      'Load'
                    )}
                  </Button>
                </CardFooter>
              </Card>
            )
          })}
        </div>
      )}

      <Card className="mt-8">
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <svg className="w-4 h-4 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            API Endpoints
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-2 text-sm md:grid-cols-4">
            <code className="sl-code block p-3 rounded-lg bg-muted/50 text-muted-foreground hover:text-foreground transition-colors">GET /models</code>
            <code className="sl-code block p-3 rounded-lg bg-muted/50 text-muted-foreground hover:text-foreground transition-colors">POST /models/load</code>
            <code className="sl-code block p-3 rounded-lg bg-muted/50 text-muted-foreground hover:text-foreground transition-colors">GET /models/hf</code>
            <code className="sl-code block p-3 rounded-lg bg-muted/50 text-muted-foreground hover:text-foreground transition-colors">GET /health</code>
          </div>
        </CardContent>
      </Card>

      <Dialog open={loadDialog.open} onOpenChange={(o) => setLoadDialog((d) => ({ ...d, open: o }))}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className={loadDialog.title.includes('Failed') ? 'text-destructive' : 'text-success'}>
              {loadDialog.title}
            </DialogTitle>
            <DialogDescription className="font-mono text-sm text-foreground/90 bg-muted/50 p-3 rounded-lg mt-2">
              {loadDialog.body}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button type="button" onClick={() => setLoadDialog((d) => ({ ...d, open: false }))}>
              Got it
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </div>
  )
}