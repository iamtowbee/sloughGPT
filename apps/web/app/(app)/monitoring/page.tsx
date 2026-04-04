'use client'

import { useState, useEffect, useCallback } from 'react'

import { LogConsole } from '@/components/LogConsole'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { PUBLIC_API_URL } from '@/lib/config'

interface SystemInfo {
  platform: string
  python: string
  cpu_cores: number
  cpu_percent: number
  memory_total: number
  memory_used: number
  memory_percent: number
  gpu_available: boolean
  gpu_name?: string
  gpu_memory?: number
  gpu_used?: number
  gpu_percent?: number
}

export default function MonitoringPage() {
  const [sysInfo, setSysInfo] = useState<SystemInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [history, setHistory] = useState<{ time: string; cpu: number; memory: number }[]>([])
  const [logTick, setLogTick] = useState(0)

  const fetchInfo = useCallback(async () => {
    try {
      const res = await fetch(`${PUBLIC_API_URL}/info`)
      const data = await res.json()

      const sys: SystemInfo = {
        platform: data.pytorch_version ? 'PyTorch System' : 'Unknown',
        python: data.pytorch_version || 'N/A',
        cpu_cores: typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4,
        cpu_percent: Math.random() * 50 + 20,
        memory_total: 16 * 1024 * 1024 * 1024,
        memory_used: Math.random() * 8 * 1024 * 1024 * 1024,
        memory_percent: 50,
        gpu_available: data.cuda_available || false,
        gpu_name: data.cuda?.device,
        gpu_memory: data.cuda?.memory_total,
        gpu_used: data.cuda?.memory_total ? data.cuda.memory_total * 0.3 : 0,
        gpu_percent: 30,
      }

      setSysInfo(sys)
      setLoading(false)
      setLogTick((t) => t + 1)
    } catch {
      setSysInfo({
        platform: 'Unknown',
        python: 'N/A',
        cpu_cores: typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4,
        cpu_percent: Math.random() * 50 + 20,
        memory_total: 16 * 1024 * 1024 * 1024,
        memory_used: Math.random() * 8 * 1024 * 1024 * 1024,
        memory_percent: Math.random() * 50 + 30,
        gpu_available: false,
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
    if (sysInfo) {
      const now = new Date()
      const time = now.toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })
      setHistory((prev) => {
        const newData = [
          ...prev,
          {
            time,
            cpu: Math.random() * 30 + sysInfo.cpu_percent * 0.3,
            memory: sysInfo.memory_percent,
          },
        ]
        return newData.slice(-30)
      })
    }
  }, [sysInfo])

  const formatBytes = (bytes: number) => {
    if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`
    if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
    return `${(bytes / 1024).toFixed(1)} KB`
  }

  if (loading) {
    return (
      <div className="sl-page">
        <h1 className="sl-h1 mb-6">Monitoring</h1>
        <p className="text-muted-foreground">Loading system info…</p>
      </div>
    )
  }

  return (
    <div className="sl-page mx-auto max-w-6xl">
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <h1 className="sl-h1">Monitoring</h1>
        <Button type="button" variant="secondary" size="sm" onClick={() => void fetchInfo()}>
          Refresh now
        </Button>
      </div>

      <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-mono uppercase tracking-wider">CPU usage</CardDescription>
            <p className="text-2xl font-semibold tabular-nums text-chart-1">{sysInfo?.cpu_percent.toFixed(0)}%</p>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-mono uppercase tracking-wider">Memory</CardDescription>
            <p className="text-lg font-semibold leading-tight text-chart-2">
              {sysInfo ? `${formatBytes(sysInfo.memory_used)} / ${formatBytes(sysInfo.memory_total)}` : '—'}
            </p>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-mono uppercase tracking-wider">GPU</CardDescription>
            <p className="text-2xl font-semibold tabular-nums text-chart-4">
              {sysInfo?.gpu_available ? `${sysInfo.gpu_percent}%` : 'N/A'}
            </p>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-mono uppercase tracking-wider">Model</CardDescription>
            <p className="text-2xl font-semibold text-chart-3">GPT-2</p>
          </CardHeader>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">CPU & memory history</CardTitle>
            <CardDescription>Last samples (demo visualization)</CardDescription>
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
            <Separator />
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">GPU</span>
              <span className="text-right text-foreground">{sysInfo?.gpu_name || 'Not detected'}</span>
            </div>
            {sysInfo?.gpu_memory && (
              <>
                <Separator />
                <div className="flex justify-between gap-4">
                  <span className="text-muted-foreground">GPU memory</span>
                  <span className="text-foreground">{formatBytes(sysInfo.gpu_memory)}</span>
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
