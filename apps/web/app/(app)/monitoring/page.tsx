'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'

import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { LogConsole } from '@/components/LogConsole'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { inferenceHealthLabel, useApiHealth } from '@/hooks/useApiHealth'
import { api } from '@/lib/api'
import { cn } from '@/lib/cn'

import { mapInfoToSystemInfo, type SystemInfo, type InfoJson } from '@/lib/monitoring-info'

function MetricCard({ label, value, subValue, color, icon }: {
  label: string
  value: string
  subValue?: string
  color: 'cpu' | 'memory' | 'gpu' | 'inference'
  icon?: React.ReactNode
}) {
  const colors = {
    cpu: 'from-blue-500/10 to-blue-600/5 border-blue-500/20',
    memory: 'from-emerald-500/10 to-emerald-600/5 border-emerald-500/20',
    gpu: 'from-violet-500/10 to-violet-600/5 border-violet-500/20',
    inference: 'from-amber-500/10 to-amber-600/5 border-amber-500/20',
  }
  const valueColors = {
    cpu: 'text-blue-600 dark:text-blue-400',
    memory: 'text-emerald-600 dark:text-emerald-400',
    gpu: 'text-violet-600 dark:text-violet-400',
    inference: 'text-amber-600 dark:text-amber-400',
  }
  
  return (
    <Card className={cn("bg-gradient-to-br border-2", colors[color])}>
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2 mb-1">
          {icon && <span className="text-muted-foreground">{icon}</span>}
          <CardDescription className="text-xs font-mono uppercase tracking-wider">{label}</CardDescription>
        </div>
        <p className={cn("text-2xl font-semibold tabular-nums", valueColors[color])}>
          {value}
        </p>
        {subValue && (
          <p className="text-xs text-muted-foreground mt-1">{subValue}</p>
        )}
      </CardHeader>
    </Card>
  )
}

function StatusIndicator({ status }: { status: 'online' | 'offline' | 'loading' }) {
  const config = {
    online: { color: 'bg-green-500', label: 'Online' },
    offline: { color: 'bg-red-500', label: 'Offline' },
    loading: { color: 'bg-yellow-500 animate-pulse', label: 'Loading' },
  }
  const { color, label } = config[status]
  
  return (
    <span className="flex items-center gap-1.5">
      <span className={cn("h-2 w-2 rounded-full", color)} />
      <span className="text-xs text-muted-foreground">{label}</span>
    </span>
  )
}

function HistoryBar({ value, maxValue, color }: { value: number; maxValue?: number; color: string }) {
  const percentage = maxValue ? Math.min((value / maxValue) * 100, 100) : Math.min(value, 100)
  return (
    <div className="flex-1 overflow-hidden bg-muted/50 rounded">
      <div
        className={cn("h-4 rounded transition-all duration-500", color)}
        style={{ width: `${percentage}%` }}
      />
    </div>
  )
}

export default function MonitoringPage() {
  const [sysInfo, setSysInfo] = useState<SystemInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [history, setHistory] = useState<{ time: string; cpu: number; memory: number }[]>([])
  const [logTick, setLogTick] = useState(0)
  const { state: health, refresh: refreshHealth } = useApiHealth()

  const inferenceSummary = useMemo(() => {
    if (health === null) return '…'
    if (health === 'offline') return 'API offline'
    if (!health.model_loaded) return 'No weights'
    return health.model_type
  }, [health])

  const inferenceTitle = useMemo(() => inferenceHealthLabel(health), [health])

  const fetchInfo = useCallback(async () => {
    try {
      const data = await api.getSystemInfo()
      setSysInfo(mapInfoToSystemInfo(data as InfoJson))
      setLoading(false)
      setLogTick((t) => t + 1)
    } catch {
      setSysInfo({
        platform: 'Unavailable',
        python: 'N/A',
        cpu_cores: typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 0 : 0,
        cpu_percent: null,
        memory_total: 0,
        memory_used: 0,
        memory_percent: null,
        gpu_available: false,
        gpu_percent: null,
        process_rss_bytes: null,
        host_metrics_available: false,
      })
      setLoading(false)
      setLogTick((t) => t + 1)
    }
  }, [])

  useEffect(() => {
    void fetchInfo()
    const interval = setInterval(() => void fetchInfo(), 5000)
    return () => clearInterval(interval)
  }, [fetchInfo])

  useEffect(() => {
    if (!sysInfo?.host_metrics_available) {
      setHistory([])
    }
  }, [sysInfo?.host_metrics_available])

  useEffect(() => {
    if (
      !sysInfo ||
      !sysInfo.host_metrics_available ||
      sysInfo.cpu_percent == null ||
      sysInfo.memory_percent == null
    ) {
      return
    }
    const cpu = sysInfo.cpu_percent
    const mem = sysInfo.memory_percent
    const now = new Date()
    const time = now.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
    setHistory((prev) => {
      const next = [...prev, { time, cpu, memory: mem }]
      return next.slice(-30)
    })
  }, [sysInfo])

  const formatBytes = (bytes: number) => {
    if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`
    if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
    return `${(bytes / 1024).toFixed(1)} KB`
  }

  if (loading) {
    return (
      <div className="sl-page">
        <AppRouteHeader className="mb-6 items-start" left={<AppRouteHeaderLead title="Monitoring" />} />
        <p className="text-muted-foreground">Loading system info…</p>
      </div>
    )
  }

  return (
    <div className="sl-page mx-auto max-w-6xl">
      <AppRouteHeader
        className="mb-6 items-start"
        left={
          <AppRouteHeaderLead
            title="Monitoring"
            subtitle={
              <>
                Inference:{' '}
                <span className="text-foreground/90" title={inferenceTitle} data-testid="monitoring-inference-line">
                  {inferenceSummary}
                </span>
              </>
            }
          >
            {sysInfo && !sysInfo.host_metrics_available ? (
              <p className="mt-2 max-w-prose text-xs text-muted-foreground">
                Host CPU/RAM are not included in the last <code className="font-mono">/info</code> response (API
                offline, or install <code className="font-mono">psutil</code> on the server). Charts use real
                samples only when the API exposes a <code className="font-mono">host</code> block.
              </p>
            ) : null}
          </AppRouteHeaderLead>
        }
        right={
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => {
              void fetchInfo()
              void refreshHealth()
            }}
          >
            Refresh now
          </Button>
        }
      />

      <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricCard
          label="CPU Usage"
          value={sysInfo?.cpu_percent != null ? `${sysInfo.cpu_percent.toFixed(0)}%` : '—'}
          color="cpu"
          icon={
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
            </svg>
          }
        />
        <MetricCard
          label="Memory"
          value={
            sysInfo?.host_metrics_available && sysInfo.memory_total > 0
              ? `${sysInfo.memory_percent != null ? `${sysInfo.memory_percent.toFixed(0)}%` : '—'}`
              : '—'
          }
          subValue={
            sysInfo?.host_metrics_available && sysInfo.memory_total > 0
              ? `${formatBytes(sysInfo.memory_used)} / ${formatBytes(sysInfo.memory_total)}`
              : undefined
          }
          color="memory"
          icon={
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
          }
        />
        <MetricCard
          label="GPU"
          value={
            !sysInfo?.gpu_available
              ? 'N/A'
              : sysInfo.gpu_percent != null
                ? `${sysInfo.gpu_percent.toFixed(0)}%`
                : sysInfo.gpu_memory != null && sysInfo.gpu_used != null
                  ? `${formatBytes(sysInfo.gpu_used)} / ${formatBytes(sysInfo.gpu_memory)}`
                  : '—'
          }
          subValue={sysInfo?.gpu_name}
          color="gpu"
          icon={
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
            </svg>
          }
        />
        <Card className="bg-gradient-to-br from-amber-500/10 to-amber-600/5 border-2 border-amber-500/20">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2 mb-1">
              <svg className="w-4 h-4 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <CardDescription className="text-xs font-mono uppercase tracking-wider">Inference</CardDescription>
            </div>
            <p className="break-words text-lg font-semibold leading-tight text-amber-600 dark:text-amber-400 md:text-2xl">
              {inferenceSummary}
            </p>
            <div className="mt-1">
              <StatusIndicator status={health === 'offline' ? 'offline' : health === null ? 'loading' : 'online'} />
            </div>
          </CardHeader>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">CPU & memory history</CardTitle>
            <CardDescription>
              Last samples from the API host (<code className="font-mono">GET /info</code> →{' '}
              <code className="font-mono">host</code>, polled every 5s)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {history.slice(-10).map((h, i) => (
              <div key={i} className="flex items-center gap-2 text-sm">
                <span className="w-20 shrink-0 font-mono text-xs text-muted-foreground">{h.time}</span>
                <div className="flex flex-1 gap-2">
                  <div className="flex-1 overflow-hidden bg-chart-1/20">
                    <div
                      className="h-4 bg-chart-1 transition-all duration-300"
                      style={{ width: `${Math.min(h.cpu, 100)}%` }}
                    />
                  </div>
                  <div className="flex-1 overflow-hidden bg-chart-2/20">
                    <div
                      className="h-4 bg-chart-2 transition-all duration-300"
                      style={{ width: `${Math.min(h.memory, 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
            <div className="mt-2 flex gap-4 text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 bg-chart-1" /> CPU
              </span>
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 bg-chart-2" /> Memory
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">System info</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Platform</span>
              <span className="text-right text-foreground">{sysInfo?.platform}</span>
            </div>
            <Separator />
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Python</span>
              <span className="text-right font-mono text-xs text-foreground">{sysInfo?.python}</span>
            </div>
            <Separator />
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">CPU cores</span>
              <span className="text-foreground">{sysInfo?.cpu_cores}</span>
            </div>
            {sysInfo?.process_rss_bytes != null && sysInfo.process_rss_bytes > 0 && (
              <>
                <Separator />
                <div className="flex justify-between gap-4">
                  <span className="text-muted-foreground">API process RSS</span>
                  <span className="font-mono text-foreground">{formatBytes(sysInfo.process_rss_bytes)}</span>
                </div>
              </>
            )}
            <Separator />
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">GPU</span>
              <span className="text-right text-foreground">{sysInfo?.gpu_name || 'Not detected'}</span>
            </div>
            {sysInfo?.gpu_memory != null && sysInfo.gpu_memory > 0 && (
              <>
                <Separator />
                <div className="flex justify-between gap-4">
                  <span className="text-muted-foreground">GPU memory</span>
                  <span className="text-foreground">
                    {sysInfo.gpu_used != null
                      ? `${formatBytes(sysInfo.gpu_used)} / ${formatBytes(sysInfo.gpu_memory)}`
                      : formatBytes(sysInfo.gpu_memory)}
                  </span>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="mt-8">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Activity log</CardTitle>
          <CardDescription>Short client-side trace — filter by level; new line on each metrics poll</CardDescription>
        </CardHeader>
        <CardContent className="p-0 sm:px-0">
          <LogConsole tick={logTick} />
        </CardContent>
      </Card>
    </div>
  )
}
