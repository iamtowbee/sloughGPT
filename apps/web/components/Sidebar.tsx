'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useMemo } from 'react'

import { inferenceHealthLabel, useApiHealth } from '@/hooks/useApiHealth'
import {
  IconAgents,
  IconApiDocs,
  IconChat,
  IconClose,
  IconModels,
  IconMonitor,
  IconPlugins,
  IconSettings,
  IconTraining,
} from '@/components/icons/NavIcons'
import { cn } from '@/lib/cn'
import { routeMatchesPath } from '@/lib/route-match'
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

/** Sidebar list — smaller than home quick-action tiles */
const NAV_ICON = 'h-4 w-4 shrink-0'

export type SidebarVariant = 'desktop' | 'drawer'

export type SidebarProps = {
  variant?: SidebarVariant
  /** Called after choosing a nav link (e.g. close mobile drawer). */
  onNavigate?: () => void
  /** Close control for drawer variant. */
  onClose?: () => void
}

export function Sidebar({ variant = 'desktop', onNavigate, onClose }: SidebarProps) {
  const pathname = usePathname()
  const isDrawer = variant === 'drawer'
  const { state: apiHealth } = useApiHealth()

  const apiStatusDot = useMemo(() => {
    if (apiHealth === null) return 'bg-muted-foreground'
    if (apiHealth === 'offline') return 'bg-destructive'
    return apiHealth.model_loaded ? 'bg-success' : 'bg-warning'
  }, [apiHealth])

  const apiStatusShort = useMemo(() => {
    if (apiHealth === null) return 'API…'
    if (apiHealth === 'offline') return 'Offline'
    return apiHealth.model_loaded ? apiHealth.model_type : 'No weights'
  }, [apiHealth])

  const apiStatusTitle = useMemo(() => inferenceHealthLabel(apiHealth), [apiHealth])

  const navLinkClass = (active: boolean) =>
    cn(
      'group relative flex min-h-[2.25rem] items-center gap-2.5 rounded-none px-3 py-1.5 text-sm transition-colors duration-200 ease-smooth',
      active
        ? 'bg-primary/[0.13] font-medium text-primary shadow-[inset_3px_0_0_0] shadow-primary dark:bg-primary/[0.11]'
        : 'text-foreground/78 hover:bg-secondary hover:text-foreground dark:text-muted-foreground',
    )

  const afterNav = onNavigate

  return (
    <aside
      className={cn(
        'sl-sidebar-surface flex shrink-0 flex-col',
        isDrawer
          ? 'h-full w-full min-w-0 pb-[max(0px,env(safe-area-inset-bottom))]'
          : 'h-dvh w-[var(--sidebar-width)]',
      )}
      aria-label="Main navigation"
    >
      <div
        className={cn(
          'flex h-[3.25rem] shrink-0 items-center border-b border-border/80 dark:border-border',
          isDrawer ? 'gap-1 px-2' : 'px-3',
        )}
      >
        {isDrawer ? (
          <button
            type="button"
            className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-none text-foreground transition-colors hover:bg-secondary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label="Close menu"
            onClick={onClose}
          >
            <IconClose className="h-4 w-4" aria-hidden />
          </button>
        ) : null}
        <Link
          href="/"
          className="flex min-w-0 flex-1 items-center gap-2.5 rounded-none outline-none ring-offset-background focus-visible:ring-2 focus-visible:ring-ring"
          aria-label="SloughGPT home"
          onClick={afterNav}
        >
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-none bg-primary/15 font-mono text-[11px] font-semibold tracking-tight text-primary ring-1 ring-primary/25">
            S
          </div>
          <div className="flex min-w-0 flex-col justify-center gap-0.5 leading-none">
            <span className="truncate text-sm font-semibold tracking-tight text-foreground">SloughGPT</span>
            <span className="font-mono text-[10px] uppercase tracking-wider text-foreground/55 dark:text-muted-foreground">
              Console
            </span>
          </div>
        </Link>
      </div>

      <nav
        className="flex min-h-0 flex-1 flex-col p-2"
        aria-label="Primary"
      >
        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
          <p
            className="mb-2 px-3 font-mono text-[10px] uppercase tracking-wider text-foreground/48 dark:text-muted-foreground"
            id="sidebar-workspace-heading"
          >
            Workspace
          </p>
          <ul className="space-y-0.5" aria-labelledby="sidebar-workspace-heading">
            {navItems.map((item) => {
              const active = routeMatchesPath(pathname, item.path)
              return (
                <li key={item.path}>
                  <Link
                    href={item.path}
                    aria-current={active ? 'page' : undefined}
                    className={navLinkClass(active)}
                    onClick={afterNav}
                  >
                    <item.Icon
                      className={cn(NAV_ICON, active ? 'opacity-100' : 'opacity-90 dark:opacity-80')}
                      aria-hidden
                    />
                    {item.label}
                  </Link>
                </li>
              )
            })}
          </ul>
        </div>

        <div className="shrink-0 border-t border-border/80 pt-3 dark:border-border">
          <p
            className="mb-1.5 px-3 font-mono text-[10px] uppercase tracking-wider text-foreground/48 dark:text-muted-foreground"
            id="sidebar-system-heading"
          >
            System
          </p>
          <div className="space-y-0.5" role="group" aria-labelledby="sidebar-system-heading">
            {bottomItems.map((item) => {
              const active = routeMatchesPath(pathname, item.path)
              return (
                <Link
                  key={item.path}
                  href={item.path}
                  aria-current={active ? 'page' : undefined}
                  className={navLinkClass(active)}
                  onClick={afterNav}
                >
                  <item.Icon
                    className={cn(NAV_ICON, active ? 'opacity-100' : 'opacity-90 dark:opacity-80')}
                    aria-hidden
                  />
                  {item.label}
                </Link>
              )
            })}
          </div>
        </div>
      </nav>

      <div className="shrink-0 space-y-2 border-t border-border/80 px-3 py-2.5 dark:border-border">
        <div
          className="flex min-w-0 items-center gap-2"
          title={apiStatusTitle}
          data-testid="sidebar-api-status"
        >
          <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${apiStatusDot}`} aria-hidden />
          <span className="min-w-0 truncate font-mono text-[10px] leading-tight text-foreground/70 dark:text-muted-foreground">
            {apiStatusShort}
          </span>
        </div>
        <ThemeSwitcher />
      </div>
    </aside>
  )
}
