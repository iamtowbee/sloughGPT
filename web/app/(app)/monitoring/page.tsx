'use client'

import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface GPU {
  id: number
  name: string
  memory_total_mb: number
  memory_used_mb: number
  memory_percent: number
  utilization_percent: number
  temperature_c: number
}

export default function MonitoringPage() {
  const [gpu, setGpu] = useState<GPU | null>(null)
  const [history, setHistory] = useState<{ time: string; loss: number; gpu: number }[]>([])
  const [activeJobs, setActiveJobs] = useState(0)

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/system/gpu`)
        const data = await res.json()
        if (data.gpus && data.gpus.length > 0) {
          setGpu(data.gpus[0])
        }
      } catch (e) {
        console.error('Failed to fetch GPU metrics:', e)
      }
    }

    const fetchJobs = async () => {
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/training/jobs`)
        const jobs = await res.json()
        setActiveJobs(jobs.filter((j: any) => j.status === 'running').length)
      } catch (e) {
        console.error('Failed to fetch jobs:', e)
      }
    }

    fetchMetrics()
    fetchJobs()

    const interval = setInterval(() => {
      fetchMetrics()
      fetchJobs()
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (gpu) {
      const now = new Date()
      const time = now.toLocaleTimeString('en-US', { hour12: false })
      setHistory(prev => {
        const newData = [...prev, { time, loss: Math.random() * 2 + 0.5, gpu: gpu.utilization_percent }]
        return newData.slice(-20)
      })
    }
  }, [gpu])

  return (
    <div>
      <h1 className="text-3xl font-bold text-slate-800 dark:text-white mb-6">Monitoring</h1>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <p className="text-sm text-slate-500">GPU Usage</p>
          <p className="text-2xl font-bold text-blue-600">{gpu?.utilization_percent ?? '--'}%</p>
        </div>
        <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <p className="text-sm text-slate-500">Memory</p>
          <p className="text-2xl font-bold text-green-600">
            {gpu ? `${(gpu.memory_used_mb / 1024).toFixed(1)}GB / ${(gpu.memory_total_mb / 1024).toFixed(0)}GB` : '--'}
          </p>
        </div>
        <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <p className="text-sm text-slate-500">Temperature</p>
          <p className="text-2xl font-bold text-orange-600">{gpu?.temperature_c ?? '--'}°C</p>
        </div>
        <div className="bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <p className="text-sm text-slate-500">Active Jobs</p>
          <p className="text-2xl font-bold text-purple-600">{activeJobs}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4">
          <h2 className="font-semibold text-slate-800 dark:text-white mb-4">Training Loss</h2>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="loss" stroke="#2563eb" strokeWidth={2} name="Loss" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4">
          <h2 className="font-semibold text-slate-800 dark:text-white mb-4">GPU Utilization</h2>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Line type="monotone" dataKey="gpu" stroke="#10b981" strokeWidth={2} name="GPU %" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {gpu && (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4">
          <h2 className="font-semibold text-slate-800 dark:text-white mb-4">GPU Details</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-slate-500">Name</p>
              <p className="font-medium text-slate-800 dark:text-white">{gpu.name}</p>
            </div>
            <div>
              <p className="text-sm text-slate-500">Memory Usage</p>
              <div className="mt-1">
                <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                  <div
                    className="bg-green-500 h-2 rounded-full"
                    style={{ width: `${gpu.memory_percent}%` }}
                  />
                </div>
                <p className="text-xs text-slate-500 mt-1">{gpu.memory_percent.toFixed(1)}%</p>
              </div>
            </div>
            <div>
              <p className="text-sm text-slate-500">Utilization</p>
              <div className="mt-1">
                <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full"
                    style={{ width: `${gpu.utilization_percent}%` }}
                  />
                </div>
                <p className="text-xs text-slate-500 mt-1">{gpu.utilization_percent}%</p>
              </div>
            </div>
            <div>
              <p className="text-sm text-slate-500">Temperature</p>
              <p className="font-medium text-slate-800 dark:text-white">{gpu.temperature_c}°C</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
