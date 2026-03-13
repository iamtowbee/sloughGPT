'use client'

import Link from 'next/link'

const features = [
  { title: 'Chat', icon: '💬', href: '/chat' },
  { title: 'Agents', icon: '🤖', href: '/agents' },
  { title: 'Training', icon: '⚡', href: '/training' },
  { title: 'Models', icon: '🧠', href: '/models' },
  { title: 'Plugins', icon: '🔌', href: '/plugins' },
  { title: 'Monitor', icon: '📊', href: '/monitoring' },
]

export default function HomePage() {
  return (
    <div className="space-y-6">
      {/* Welcome */}
      <div>
        <h1 className="text-2xl font-semibold">Welcome back</h1>
        <p className="text-sm text-muted-foreground">Here's what's happening with your AI platform.</p>
      </div>

      {/* Quick Actions */}
      <div>
        <h2 className="text-sm font-medium mb-3">Quick Actions</h2>
        <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
          {features.map((feature) => (
            <Link
              key={feature.href}
              href={feature.href}
              className="flex flex-col items-center gap-2 p-4 rounded-xl border bg-card hover:bg-secondary transition-colors"
            >
              <span className="text-2xl">{feature.icon}</span>
              <span className="text-xs font-medium">{feature.title}</span>
            </Link>
          ))}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="p-4 rounded-xl border bg-card">
          <p className="text-xs text-muted-foreground">API Status</p>
          <p className="text-lg font-semibold text-green-600">Online</p>
        </div>
        <div className="p-4 rounded-xl border bg-card">
          <p className="text-xs text-muted-foreground">Models</p>
          <p className="text-lg font-semibold">4 loaded</p>
        </div>
        <div className="p-4 rounded-xl border bg-card">
          <p className="text-xs text-muted-foreground">GPU</p>
          <p className="text-lg font-semibold text-blue-600">Ready</p>
        </div>
      </div>
    </div>
  )
}
