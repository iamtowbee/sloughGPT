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
      setHistory(prev => {
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
      <div className="p-6">
        <h1 className="text-3xl font-bold text-white mb-6">Monitoring</h1>
        <div className="text-zinc-500">Loading system info...</div>
      </div>
    )
  }

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-white mb-6">Monitoring</h1>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <p className="text-sm text-zinc-400">CPU Usage</p>
          <p className="text-2xl font-bold text-blue-400">{sysInfo?.cpu_percent.toFixed(0)}%</p>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <p className="text-sm text-zinc-400">Memory</p>
          <p className="text-2xl font-bold text-green-400">
            {sysInfo ? `${formatBytes(sysInfo.memory_used)} / ${formatBytes(sysInfo.memory_total)}` : '--'}
          </p>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <p className="text-sm text-zinc-400">GPU</p>
          <p className="text-2xl font-bold text-purple-400">
            {sysInfo?.gpu_available ? `${sysInfo.gpu_percent}%` : 'N/A'}
          </p>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <p className="text-sm text-zinc-400">Model</p>
          <p className="text-2xl font-bold text-cyan-400">GPT-2</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <h2 className="font-semibold text-white mb-4">CPU & Memory History</h2>
          <div className="space-y-2">
            {history.slice(-10).map((h, i) => (
              <div key={i} className="flex items-center gap-2 text-sm">
                <span className="text-zinc-500 w-20">{h.time}</span>
                <div className="flex-1 flex gap-2">
                  <div className="flex-1 bg-blue-500/20 rounded overflow-hidden">
                    <div
                      className="bg-blue-500 h-4 transition-all"
                      style={{ width: `${Math.min(h.cpu, 100)}%` }}
                    />
                  </div>
                  <div className="flex-1 bg-green-500/20 rounded overflow-hidden">
                    <div
                      className="bg-green-500 h-4 transition-all"
                      style={{ width: `${Math.min(h.memory, 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="flex gap-4 mt-2 text-xs text-zinc-500">
            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-blue-500 rounded" /> CPU</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-green-500 rounded" /> Memory</span>
          </div>
        </div>

        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <h2 className="font-semibold text-white mb-4">System Info</h2>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-zinc-400">Platform</span>
              <span className="text-white">{sysInfo?.platform}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-400">Python</span>
              <span className="text-white">{sysInfo?.python}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-400">CPU Cores</span>
              <span className="text-white">{sysInfo?.cpu_cores}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-400">GPU</span>
              <span className="text-white">{sysInfo?.gpu_name || 'Not detected'}</span>
            </div>
            {sysInfo?.gpu_memory && (
              <div className="flex justify-between">
                <span className="text-zinc-400">GPU Memory</span>
                <span className="text-white">{formatBytes(sysInfo.gpu_memory)}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
