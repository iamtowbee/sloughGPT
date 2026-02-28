import React, { useState, useEffect } from 'react'
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Spinner,
  Badge
} from '@base-ui/react'
import { SystemMetrics } from '../store'
import { api } from '../utils/api'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts'

interface LogEntry {
  id: string
  level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG'
  message: string
  timestamp: Date
}

const Monitoring: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [metricsHistory, setMetricsHistory] = useState<SystemMetrics[]>([])
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedTab, setSelectedTab] = useState<'overview' | 'charts' | 'logs'>('overview')
  const [systemInfo, setSystemInfo] = useState<any>(null)

  useEffect(() => {
    loadData()
  }, [])

  useEffect(() => {
    const interval = setInterval(() => {
      loadMetrics()
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  const loadData = async () => {
    try {
      const [metricsRes, infoRes] = await Promise.all([
        api.getMetrics(),
        api.getSystemInfo()
      ])
      
      if (metricsRes.data) {
        setMetrics(metricsRes.data)
        setMetricsHistory([metricsRes.data])
        setMetrics(metricsRes.data)
      }
      
      if (infoRes.data) {
        setSystemInfo(infoRes.data)
      }
      
      // Add initial log
      setLogs([
        { id: '1', level: 'INFO', message: 'Monitoring started', timestamp: new Date() }
      ])
    } catch (error) {
      console.error('Error loading data:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const loadMetrics = async () => {
    try {
      const response = await api.getMetrics()
      if (response.data) {
        setMetrics(response.data)
        setMetricsHistory(prev => [...prev.slice(-29), response.data!])
        setMetrics(response.data)
      }
    } catch (error) {
      console.error('Error loading metrics:', error)
    }
  }

  const getMetricColor = (value: number, max: number = 100) => {
    const percent = (value / max) * 100
    if (percent < 50) return 'text-green-500'
    if (percent < 80) return 'text-yellow-500'
    return 'text-red-500'
  }

  const renderMetricCard = (title: string, value: number, unit: string, max: number = 100) => {
    const colorClass = getMetricColor(value, max)
    
    return (
      <Card className="hover:shadow-md transition-shadow">
        <CardContent className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-600">{title}</span>
            <span className={`text-lg font-bold ${colorClass}`}>
              {value.toFixed(1)}{unit}
            </span>
          </div>
          <div className="w-full bg-slate-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all ${
                value < 50 ? 'bg-green-500' : value < 80 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              style={{ width: `${Math.min(100, (value / max) * 100)}%` }}
            />
          </div>
        </CardContent>
      </Card>
    )
  }

  const renderLogEntry = (log: LogEntry) => {
    const levelColors = {
      INFO: 'bg-blue-100 text-blue-800',
      WARN: 'bg-yellow-100 text-yellow-800',
      ERROR: 'bg-red-100 text-red-800',
      DEBUG: 'bg-gray-100 text-gray-800'
    }

    return (
      <div key={log.id} className="flex items-start gap-2 p-2 hover:bg-slate-50">
        <Badge className={`${levelColors[log.level]} flex-shrink-0`}>
          {log.level}
        </Badge>
        <div className="flex-1 text-sm">
          {log.message}
          <span className="text-xs text-slate-400 ml-2">
            {log.timestamp.toLocaleTimeString()}
          </span>
        </div>
      </div>
    )
  }

  const chartData = metricsHistory.map((m, i) => ({
    time: i,
    cpu: m.cpu_percent,
    memory: m.memory_percent,
    disk: m.disk_percent
  }))

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Spinner className="h-8 w-8 text-blue-500" size="8" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="text-xl font-semibold">Monitoring Dashboard</span>
            <div className="flex gap-2">
              <Button
                variant={selectedTab === 'overview' ? 'primary' : 'outline'}
                size="sm"
                onClick={() => setSelectedTab('overview')}
              >
                Overview
              </Button>
              <Button
                variant={selectedTab === 'charts' ? 'primary' : 'outline'}
                size="sm"
                onClick={() => setSelectedTab('charts')}
              >
                Charts
              </Button>
              <Button
                variant={selectedTab === 'logs' ? 'primary' : 'outline'}
                size="sm"
                onClick={() => setSelectedTab('logs')}
              >
                Logs
              </Button>
            </div>
          </CardTitle>
          <CardDescription>
            Real-time system performance and health metrics
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Overview Tab */}
      {selectedTab === 'overview' && metrics && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {renderMetricCard('CPU Usage', metrics.cpu_percent, '%')}
            {renderMetricCard('Memory Usage', metrics.memory_percent, '%')}
            {renderMetricCard('Disk Usage', metrics.disk_percent, '%')}
            {renderMetricCard('Network I/O', metrics.network_recv_mb + metrics.network_sent_mb, ' MB')}
          </div>

          {/* System Info */}
          {systemInfo && (
            <Card>
              <CardHeader>
                <CardTitle>System Information</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-slate-500">Python Version:</span>
                    <span className="ml-2 font-medium">{systemInfo.python_version}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">CPU Cores:</span>
                    <span className="ml-2 font-medium">{systemInfo.cpu_count}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Total Memory:</span>
                    <span className="ml-2 font-medium">{systemInfo.total_memory_gb?.toFixed(1)} GB</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Total Disk:</span>
                    <span className="ml-2 font-medium">{systemInfo.disk_total_gb?.toFixed(1)} GB</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Conversations:</span>
                    <span className="ml-2 font-medium">{systemInfo.conversations_count}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Datasets:</span>
                    <span className="ml-2 font-medium">{systemInfo.datasets_count}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Training Jobs:</span>
                    <span className="ml-2 font-medium">{systemInfo.training_jobs_count}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Platform:</span>
                    <span className="ml-2 font-medium">{systemInfo.platform}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Memory Details */}
          <Card>
            <CardHeader>
              <CardTitle>Memory Details</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-slate-500">Used:</span>
                  <span className="ml-2 font-medium">{metrics.memory_used_mb?.toFixed(0)} MB</span>
                </div>
                <div>
                  <span className="text-slate-500">Total:</span>
                  <span className="ml-2 font-medium">{metrics.memory_total_mb?.toFixed(0)} MB</span>
                </div>
                <div>
                  <span className="text-slate-500">Disk Used:</span>
                  <span className="ml-2 font-medium">{metrics.disk_used_gb?.toFixed(1)} GB</span>
                </div>
                <div>
                  <span className="text-slate-500">Disk Total:</span>
                  <span className="ml-2 font-medium">{metrics.disk_total_gb?.toFixed(1)} GB</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Charts Tab */}
      {selectedTab === 'charts' && (
        <Card>
          <CardHeader>
            <CardTitle>Resource Usage Over Time</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="cpu"
                  stackId="1"
                  stroke="#0088FE"
                  fill="#0088FE"
                  fillOpacity={0.6}
                  name="CPU %"
                />
                <Area
                  type="monotone"
                  dataKey="memory"
                  stackId="2"
                  stroke="#00C49F"
                  fill="#00C49F"
                  fillOpacity={0.6}
                  name="Memory %"
                />
                <Area
                  type="monotone"
                  dataKey="disk"
                  stackId="3"
                  stroke="#FFBB28"
                  fill="#FFBB28"
                  fillOpacity={0.6}
                  name="Disk %"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Logs Tab */}
      {selectedTab === 'logs' && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>System Logs</span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setLogs([])}
              >
                Clear
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent className="max-h-[600px] overflow-y-auto">
            {logs.length === 0 ? (
              <div className="text-center text-slate-400 py-8">
                No logs available
              </div>
            ) : (
              <div className="space-y-1">
                {logs.map(renderLogEntry)}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export default Monitoring
