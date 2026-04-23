'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'

import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { LogConsole } from '@/components/LogConsole'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { inferenceHealthLabel, useApiHealth } from '@/hooks/useApiHealth'
import { api } from '@/lib/api'

import { mapInfoToSystemInfo, type SystemInfo, type InfoJson } from '@/lib/monitoring-info'

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
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-mono uppercase tracking-wider">CPU usage</CardDescription>
            <p className="text-2xl font-semibold tabular-nums text-chart-1">
              {sysInfo?.cpu_percent != null ? `${sysInfo.cpu_percent.toFixed(0)}%` : '—'}
            </p>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-mono uppercase tracking-wider">Memory</CardDescription>
            <p className="text-lg font-semibold leading-tight text-chart-2">
              {sysInfo?.host_metrics_available && sysInfo.memory_total > 0
                ? `${formatBytes(sysInfo.memory_used)} / ${formatBytes(sysInfo.memory_total)}${
                    sysInfo.memory_percent != null ? ` (${sysInfo.memory_percent.toFixed(0)}%)` : ''
                  }`
                : '—'}
            </p>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-mono uppercase tracking-wider">GPU</CardDescription>
            <p className="text-2xl font-semibold tabular-nums text-chart-4">
              {!sysInfo?.gpu_available
                ? 'N/A'
                : sysInfo.gpu_percent != null
                  ? `${sysInfo.gpu_percent.toFixed(0)}%`
                  : sysInfo.gpu_memory != null && sysInfo.gpu_used != null
                    ? `${formatBytes(sysInfo.gpu_used)} / ${formatBytes(sysInfo.gpu_memory)}`
                    : '—'}
            </p>
          </CardHeader>
        </Card>
        <Card title={inferenceTitle}>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-mono uppercase tracking-wider">Inference</CardDescription>
            <p className="break-words text-lg font-semibold leading-tight text-chart-3 md:text-2xl">
              {inferenceSummary}
            </p>
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
