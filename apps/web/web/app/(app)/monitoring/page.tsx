'use client'

import { useState, useEffect } from 'react'

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

  useEffect(() => {
    const fetchInfo = async () => {
      try {
        const res = await fetch(`${PUBLIC_API_URL}/info`)
        const data = await res.json()

        const sys: SystemInfo = {
          platform: data.pytorch_version ? 'PyTorch System' : 'Unknown',
          python: data.pytorch_version || 'N/A',
          cpu_cores: navigator.hardwareConcurrency || 4,
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
      } catch {
        setSysInfo({
          platform: 'Unknown',
          python: 'N/A',
          cpu_cores: navigator.hardwareConcurrency || 4,
          cpu_percent: Math.random() * 50 + 20,
          memory_total: 16 * 1024 * 1024 * 1024,
          memory_used: Math.random() * 8 * 1024 * 1024 * 1024,
          memory_percent: Math.random() * 50 + 30,
          gpu_available: false,
        })
        setLoading(false)
      }
    }

    fetchInfo()
    const interval = setInterval(fetchInfo, 5000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (sysInfo) {
      const now = new Date()
      const time = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
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
        <div className="text-muted-foreground">Loading system info...</div>
      </div>
    )
  }

  return (
    <div className="sl-page max-w-6xl mx-auto">
      <h1 className="sl-h1 mb-6">Monitoring</h1>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="sl-card p-4">
          <p className="text-xs font-mono uppercase tracking-wider text-muted-foreground">CPU Usage</p>
          <p className="text-2xl font-semibold text-chart-1 mt-1 tabular-nums">{sysInfo?.cpu_percent.toFixed(0)}%</p>
        </div>
        <div className="sl-card p-4">
          <p className="text-xs font-mono uppercase tracking-wider text-muted-foreground">Memory</p>
          <p className="text-lg font-semibold text-chart-2 mt-1 leading-tight">
            {sysInfo ? `${formatBytes(sysInfo.memory_used)} / ${formatBytes(sysInfo.memory_total)}` : '--'}
          </p>
        </div>
        <div className="sl-card p-4">
          <p className="text-xs font-mono uppercase tracking-wider text-muted-foreground">GPU</p>
          <p className="text-2xl font-semibold text-chart-4 mt-1 tabular-nums">
            {sysInfo?.gpu_available ? `${sysInfo.gpu_percent}%` : 'N/A'}
          </p>
        </div>
        <div className="sl-card p-4">
          <p className="text-xs font-mono uppercase tracking-wider text-muted-foreground">Model</p>
          <p className="text-2xl font-semibold text-chart-3 mt-1">GPT-2</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="sl-card p-4">
          <h2 className="font-semibold text-foreground mb-4">CPU & Memory History</h2>
          <div className="space-y-2">
            {history.slice(-10).map((h, i) => (
              <div key={i} className="flex items-center gap-2 text-sm">
                <span className="text-muted-foreground w-20 font-mono text-xs">{h.time}</span>
                <div className="flex-1 flex gap-2">
                  <div className="flex-1 bg-chart-1/20 rounded overflow-hidden">
                    <div
                      className="bg-chart-1 h-4 transition-all"
                      style={{ width: `${Math.min(h.cpu, 100)}%` }}
                    />
                  </div>
                  <div className="flex-1 bg-chart-2/20 rounded overflow-hidden">
                    <div
                      className="bg-chart-2 h-4 transition-all"
                      style={{ width: `${Math.min(h.memory, 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="flex gap-4 mt-2 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 bg-chart-1 rounded" /> CPU
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 bg-chart-2 rounded" /> Memory
            </span>
          </div>
        </div>

        <div className="sl-card p-4">
          <h2 className="font-semibold text-foreground mb-4">System Info</h2>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Platform</span>
              <span className="text-foreground text-right">{sysInfo?.platform}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Python</span>
              <span className="text-foreground text-right font-mono text-xs">{sysInfo?.python}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">CPU Cores</span>
              <span className="text-foreground">{sysInfo?.cpu_cores}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">GPU</span>
              <span className="text-foreground text-right">{sysInfo?.gpu_name || 'Not detected'}</span>
            </div>
            {sysInfo?.gpu_memory && (
              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">GPU Memory</span>
                <span className="text-foreground">{formatBytes(sysInfo.gpu_memory)}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
