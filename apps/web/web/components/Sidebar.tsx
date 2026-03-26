'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { ThemeSwitcher } from './ThemeSwitcher'

const navItems = [
  { path: '/chat', label: 'Chat', icon: '💬' },
  { path: '/agents', label: 'Agents', icon: '🤖' },
  { path: '/plugins', label: 'Plugins', icon: '🔌' },
  { path: '/models', label: 'Models', icon: '🧠' },
  { path: '/training', label: 'Training', icon: '⚡' },
  { path: '/monitoring', label: 'Monitor', icon: '📊' },
]

const bottomItems = [
  { path: '/settings', label: 'Settings', icon: '⚙️' },
  { path: '/api-docs', label: 'API', icon: '📚' },
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <aside className="w-[240px] h-screen bg-card border-r border-border flex flex-col">
      {/* Logo */}
      <div className="h-14 flex items-center gap-2.5 px-4 border-b border-border">
        <div className="w-7 h-7 bg-primary rounded-md flex items-center justify-center text-primary-foreground text-sm font-semibold">
          S
        </div>
        <span className="font-semibold text-sm">SloughGPT</span>
      </div>
      
      {/* Nav */}
      <nav className="flex-1 p-2 overflow-y-auto">
        <div className="space-y-0.5">
          {navItems.map((item) => (
            <Link
              key={item.path}
              href={item.path}
              className={`flex items-center gap-2.5 px-2.5 py-2 rounded-md text-sm transition-colors ${
                pathname === item.path
                  ? 'bg-primary/10 text-primary font-medium'
                  : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
              }`}
            >
              <span className="text-base">{item.icon}</span>
              {item.label}
            </Link>
          ))}
        </div>
        
        <div className="mt-6 pt-2 border-t border-border">
          <div className="space-y-0.5">
            {bottomItems.map((item) => (
              <Link
                key={item.path}
                href={item.path}
                className={`flex items-center gap-2.5 px-2.5 py-2 rounded-md text-sm transition-colors ${
                  pathname === item.path
                    ? 'bg-primary/10 text-primary font-medium'
                    : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
                }`}
              >
                <span className="text-base">{item.icon}</span>
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      </nav>
      
      {/* Theme */}
      <div className="p-3 border-t border-border">
        <ThemeSwitcher />
      </div>
    </aside>
  )
}
