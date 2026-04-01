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
    <aside className="w-[240px] h-screen bg-card/80 backdrop-blur-sm border-r border-border flex flex-col">
      <div className="h-14 flex items-center gap-3 px-4 border-b border-border">
        <div className="w-8 h-8 rounded-lg bg-primary/15 ring-1 ring-primary/25 flex items-center justify-center text-primary text-xs font-semibold tracking-tight font-mono">
          S
        </div>
        <div className="flex flex-col min-w-0">
          <span className="font-semibold text-sm tracking-tight text-foreground truncate">SloughGPT</span>
          <span style={{ fontSize: '10px' }} className="text-muted-foreground font-mono uppercase tracking-wider">
            Console
          </span>
        </div>
      </div>

      <nav className="flex-1 p-2 overflow-y-auto">
        <div className="space-y-0.5">
          {navItems.map((item) => {
            const active = pathname === item.path
            return (
              <Link
                key={item.path}
                href={item.path}
                className={`flex items-center gap-3 px-2.5 py-2 rounded-lg text-sm transition-colors ${
                  active
                    ? 'bg-primary/10 text-primary font-medium'
                    : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
                }`}
              >
                <item.Icon className={active ? 'opacity-100' : 'opacity-80'} />
                {item.label}
              </Link>
            )
          })}
        </div>

        <div className="mt-6 pt-3 border-t border-border">
          <p style={{ fontSize: '10px' }} className="px-2.5 mb-1.5 text-muted-foreground font-mono uppercase tracking-wider">
            System
          </p>
          <div className="space-y-0.5">
            {bottomItems.map((item) => {
              const active = pathname === item.path
              return (
                <Link
                  key={item.path}
                  href={item.path}
                  className={`flex items-center gap-3 px-2.5 py-2 rounded-lg text-sm transition-colors ${
                    active
                      ? 'bg-primary/10 text-primary font-medium'
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

      <div className="p-3 border-t border-border">
        <ThemeSwitcher />
      </div>
    </aside>
  )
}
