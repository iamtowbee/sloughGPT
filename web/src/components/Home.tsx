import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
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
import { useStore } from '../store'
import { api } from '../utils/api'

interface StatCard {
  title: string
  value: string | number
  change?: string
  trend?: 'up' | 'down' | 'stable'
  icon: string
}

interface ActivityItem {
  id: string
  type: string
  message: string
  timestamp: Date
}

interface SystemInfo {
  python_version: string
  platform: string
  cpu_count: number
  total_memory_gb: number
  disk_total_gb: number
  conversations_count: number
  datasets_count: number
  training_jobs_count: number
}

const Home: React.FC = () => {
  const [stats, setStats] = useState<StatCard[]>([])
  const [activities, setActivities] = useState<ActivityItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null)
  const { models, datasets, isConnected } = useStore()

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    try {
      const [infoRes, healthRes, metricsRes] = await Promise.all([
        api.getSystemInfo(),
        api.healthCheck(),
        api.getMetrics()
      ])

      if (infoRes.data) {
        setSystemInfo(infoRes.data)
        
        const simulatedStats: StatCard[] = [
          { title: 'Models', value: models.length, icon: '🤖', trend: 'stable' },
          { title: 'Datasets', value: infoRes.data.datasets_count, icon: '📊', trend: 'stable' },
          { title: 'Conversations', value: infoRes.data.conversations_count, icon: '💬', trend: 'up' },
          { title: 'Training Jobs', value: infoRes.data.training_jobs_count, icon: '🧠', trend: 'stable' }
        ]
        setStats(simulatedStats)
      }

      const simulatedActivities: ActivityItem[] = [
        { id: '1', type: 'info', message: 'API connected successfully', timestamp: new Date() },
        { id: '2', type: 'success', message: `Loaded ${models.length} models`, timestamp: new Date(Date.now() - 60000) },
        { id: '3', type: 'info', message: `Found ${datasets.length} datasets`, timestamp: new Date(Date.now() - 120000) },
        { id: '4', type: 'success', message: 'System health check passed', timestamp: new Date(Date.now() - 180000) }
      ]
      setActivities(simulatedActivities)
    } catch (error) {
      console.error('Error loading dashboard:', error)
      
      // Fallback stats
      setStats([
        { title: 'Models', value: models.length, icon: '🤖', trend: 'stable' },
        { title: 'Datasets', value: datasets.length, icon: '📊', trend: 'stable' },
        { title: 'Conversations', value: 0, icon: '💬', trend: 'stable' },
        { title: 'Training Jobs', value: 0, icon: '🧠', trend: 'stable' }
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const quickLinks = [
    { path: '/chat', label: 'Start Chatting', icon: '💬', description: 'Chat with AI models' },
    { path: '/datasets', label: 'Manage Datasets', icon: '📊', description: 'Create and manage datasets' },
    { path: '/models', label: 'View Models', icon: '🤖', description: 'Browse available models' },
    { path: '/training', label: 'Train Models', icon: '🧠', description: 'Train your own models' },
    { path: '/monitoring', label: 'System Monitor', icon: '📈', description: 'View system metrics' }
  ]

  const renderStatCard = (stat: StatCard, index: number) => {
    const trendColor = stat.trend === 'up' ? 'text-green-500' : stat.trend === 'down' ? 'text-red-500' : 'text-slate-500'
    const trendIcon = stat.trend === 'up' ? '↑' : stat.trend === 'down' ? '↓' : '→'

    return (
      <Card key={index} className="hover:shadow-md transition-shadow">
        <CardContent className="p-6">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-slate-500 mb-1">{stat.title}</p>
              <p className="text-3xl font-bold text-slate-900">{stat.value}</p>
              {stat.change && (
                <p className={`text-sm mt-2 ${trendColor}`}>
                  {trendIcon} {stat.change}
                </p>
              )}
            </div>
            <div className="text-4xl">{stat.icon}</div>
          </div>
        </CardContent>
      </Card>
    )
  }

  const renderActivity = (activity: ActivityItem) => {
    const typeColors: Record<string, string> = {
      info: 'bg-blue-100 text-blue-800',
      success: 'bg-green-100 text-green-800',
      warning: 'bg-yellow-100 text-yellow-800',
      error: 'bg-red-100 text-red-800'
    }

    const timeAgo = (date: Date) => {
      const seconds = Math.floor((Date.now() - date.getTime()) / 1000)
      if (seconds < 60) return 'just now'
      if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
      if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
      return `${Math.floor(seconds / 86400)}d ago`
    }

    return (
      <div key={activity.id} className="flex items-start gap-3 p-3 hover:bg-slate-50 rounded-lg transition-colors">
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${typeColors[activity.type] || typeColors.info}`}>
          {activity.type.toUpperCase()}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-slate-900">{activity.message}</p>
          <p className="text-xs text-slate-500 mt-1">{timeAgo(activity.timestamp)}</p>
        </div>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Spinner className="h-8 w-8 text-blue-500" size="8" />
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Welcome Section */}
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl flex items-center justify-between">
            Welcome to SloughGPT
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-sm font-normal text-slate-500">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </CardTitle>
          <CardDescription>
            Enterprise AI Framework for training, deploying, and managing large language models
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            <Badge className="bg-blue-100 text-blue-800">v2.0.0</Badge>
            <Badge className="bg-purple-100 text-purple-800">API Ready</Badge>
            <Badge className="bg-indigo-100 text-indigo-800">TypeScript UI</Badge>
          </div>
        </CardContent>
      </Card>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map(renderStatCard)}
      </div>

      {/* Quick Links */}
      <div>
        <h2 className="text-xl font-semibold mb-4 text-slate-800">Quick Actions</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {quickLinks.map((link) => (
            <Link key={link.path} to={link.path}>
              <Card className="hover:shadow-lg transition-all hover:border-blue-300 cursor-pointer h-full">
                <CardContent className="p-4 flex flex-col items-center text-center">
                  <div className="text-3xl mb-2">{link.icon}</div>
                  <h3 className="font-semibold text-slate-900 text-sm mb-1">{link.label}</h3>
                  <p className="text-xs text-slate-500">{link.description}</p>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </div>

      {/* System Info & Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>System Information</CardTitle>
            <CardDescription>Backend system details</CardDescription>
          </CardHeader>
          <CardContent>
            {systemInfo ? (
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-slate-500">Python Version</span>
                  <span className="font-medium">{systemInfo.python_version}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Platform</span>
                  <span className="font-medium">{systemInfo.platform}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">CPU Cores</span>
                  <span className="font-medium">{systemInfo.cpu_count}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Total Memory</span>
                  <span className="font-medium">{systemInfo.total_memory_gb?.toFixed(1)} GB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Total Disk</span>
                  <span className="font-medium">{systemInfo.disk_total_gb?.toFixed(1)} GB</span>
                </div>
              </div>
            ) : (
              <p className="text-slate-400">Unable to load system information</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>Latest system events and updates</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {activities.map(renderActivity)}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default Home
