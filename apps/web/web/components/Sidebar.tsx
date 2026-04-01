'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

import {
  IconAgents,
  IconApiDocs,
  IconChat,
  IconModels,
  IconMonitor,
  IconPlugins,
  IconSettings,
  IconTraining,
} from '@/components/icons/NavIcons'
import { ThemeSwitcher } from './ThemeSwitcher'

const navItems = [
  { path: '/chat', label: 'Chat', Icon: IconChat },
  { path: '/agents', label: 'Agents', Icon: IconAgents },
  { path: '/plugins', label: 'Plugins', Icon: IconPlugins },
  { path: '/models', label: 'Models', Icon: IconModels },
  { path: '/training', label: 'Training', Icon: IconTraining },
  { path: '/monitoring', label: 'Monitor', Icon: IconMonitor },
] as const

const bottomItems = [
  { path: '/settings', label: 'Settings', Icon: IconSettings },
  { path: '/api-docs', label: 'API', Icon: IconApiDocs },
] as const

export function Sidebar() {
  const pathname = usePathname()

  return (
    <aside className="flex h-screen w-[240px] flex-col border-r border-border bg-card/80 backdrop-blur-sm" aria-label="Main navigation">
      <div className="flex h-14 items-center gap-3 border-b border-border px-4">
        <Link
          href="/"
          className="flex min-w-0 flex-1 items-center gap-3 rounded-lg outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring"
          aria-label="SloughGPT home"
        >
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/15 font-mono text-xs font-semibold tracking-tight text-primary ring-1 ring-primary/25">
            S
          </div>
          <div className="flex min-w-0 flex-col">
            <span className="truncate text-sm font-semibold tracking-tight text-foreground">SloughGPT</span>
            <span style={{ fontSize: '10px' }} className="font-mono uppercase tracking-wider text-muted-foreground">
              Console
            </span>
          </div>
        </Link>
      </div>

      <nav className="flex-1 overflow-y-auto p-2" aria-label="Primary">
        <div className="space-y-0.5">
          {navItems.map((item) => {
            const active = pathname === item.path
            return (
              <Link
                key={item.path}
                href={item.path}
                aria-current={active ? 'page' : undefined}
                className={`flex items-center gap-3 rounded-lg px-2.5 py-2 text-sm transition-colors ${
                  active
                    ? 'bg-primary/10 font-medium text-primary'
                    : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
                }`}
              >
                <item.Icon className={active ? 'opacity-100' : 'opacity-80'} />
                {item.label}
              </Link>
            )
          })}
        </div>

        <div className="mt-6 border-t border-border pt-3">
          <p
            style={{ fontSize: '10px' }}
            className="mb-1.5 px-2.5 font-mono uppercase tracking-wider text-muted-foreground"
            id="sidebar-system-heading"
          >
            System
          </p>
          <div className="space-y-0.5" role="group" aria-labelledby="sidebar-system-heading">
            {bottomItems.map((item) => {
              const active = pathname === item.path
              return (
                <Link
                  key={item.path}
                  href={item.path}
                  aria-current={active ? 'page' : undefined}
                  className={`flex items-center gap-3 rounded-lg px-2.5 py-2 text-sm transition-colors ${
                    active
                      ? 'bg-primary/10 font-medium text-primary'
                      : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
                  }`}
                >
                  <item.Icon className={active ? 'opacity-100' : 'opacity-80'} />
                  {item.label}
                </Link>
              )
            })}
          </div>
        </div>
      </nav>

      <div className="border-t border-border p-3">
        <ThemeSwitcher />
      </div>
    </aside>
  )
}
