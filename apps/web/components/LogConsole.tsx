'use client'

import { useCallback, useEffect, useId, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/cn'

export type LogLevel = 'all' | 'info' | 'warn' | 'error'

export interface LogLine {
  id: string
  ts: string
  level: Exclude<LogLevel, 'all'>
  message: string
}

function levelStyle(level: Exclude<LogLevel, 'all'>) {
  switch (level) {
    case 'error':
      return 'text-destructive'
    case 'warn':
      return 'text-warning'
    default:
      return 'text-chart-3'
  }
}

function levelDot(level: Exclude<LogLevel, 'all'>) {
  switch (level) {
    case 'error':
      return 'bg-destructive'
    case 'warn':
      return 'bg-warning'
    default:
      return 'bg-chart-3'
  }
}

const TAB_ITEMS: { value: LogLevel; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'info', label: 'Info' },
  { value: 'warn', label: 'Warn' },
  { value: 'error', label: 'Error' },
]

export interface LogConsoleProps {
  /** Bump after an external event (e.g. monitoring poll) to append a status line. */
  tick?: number
  className?: string
}

export function LogConsole({ tick = 0, className }: LogConsoleProps) {
  const baseId = useId()
  const [lines, setLines] = useState<LogLine[]>(() => seedLines())
  const [filter, setFilter] = useState<LogLevel>('all')

  const append = useCallback(
    (level: Exclude<LogLevel, 'all'>, message: string) => {
      const ts = new Date().toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })
      setLines((prev) => {
        const next: LogLine[] = [
          ...prev,
          {
            id: `${baseId}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            ts,
            level,
            message,
          },
        ]
        return next.slice(-200)
      })
    },
    [baseId]
  )

  useEffect(() => {
    if (tick <= 0) return
    append('info', `Poll #${tick} — metrics refresh`)
  }, [tick, append])

  const filtered = useMemo(() => {
    if (filter === 'all') return lines
    return lines.filter((l) => l.level === filter)
  }, [lines, filter])

  const clear = () => {
    setLines(seedLines())
  }

  return (
    <div
      className={cn(
        'overflow-hidden rounded-none border border-border bg-card shadow-sm ring-1 ring-border/40',
        className
      )}
    >
      <div className="flex flex-col border-b border-border/80 bg-muted/25 sm:flex-row sm:items-stretch sm:justify-between">
        <div
          className="flex flex-wrap items-center gap-0 p-1 sm:flex-nowrap"
          role="tablist"
          aria-label="Log level filter"
        >
          {TAB_ITEMS.map(({ value, label }) => {
            const active = filter === value
            return (
              <button
                key={value}
                type="button"
                role="tab"
                aria-selected={active}
                onClick={() => setFilter(value)}
                className={cn(
                  'h-7 min-w-[3rem] rounded-none px-2.5 py-0 text-[11px] font-semibold uppercase tracking-wider transition-colors duration-200',
                  active
                    ? 'bg-primary/15 text-primary shadow-[inset_0_-2px_0_0_var(--primary)]'
                    : 'text-muted-foreground hover:bg-secondary/70 hover:text-foreground'
                )}
              >
                {label}
              </button>
            )
          })}
        </div>
        <div className="flex items-center justify-between gap-2 border-t border-border/60 px-2 py-1.5 sm:border-t-0 sm:border-l sm:border-border/60 sm:py-0">
          <span className="font-mono text-[10px] text-muted-foreground">
            {filtered.length} line{filtered.length === 1 ? '' : 's'}
          </span>
          <Button type="button" variant="ghost" size="sm" className="h-7 rounded-none px-2 text-xs" onClick={clear}>
            Clear
          </Button>
        </div>
      </div>

      <div
        className="max-h-44 overflow-y-auto overscroll-contain bg-muted/35 px-3 py-2 font-mono text-[11px] leading-relaxed dark:bg-secondary/25"
        role="log"
        aria-live="polite"
        aria-relevant="additions"
      >
        {filtered.length === 0 ? (
          <p className="py-6 text-center text-muted-foreground">No lines for this filter.</p>
        ) : (
          <ul className="space-y-1">
            {filtered.map((line) => (
              <li key={line.id} className="flex gap-2 break-all">
                <span className="shrink-0 tabular-nums text-muted-foreground">{line.ts}</span>
                <span className={cn('mt-[3px] h-1.5 w-1.5 shrink-0 rounded-full', levelDot(line.level))} aria-hidden />
                <span className={cn('min-w-0 flex-1', levelStyle(line.level))}>{line.message}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

function seedLines(): LogLine[] {
  const ts = () =>
    new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
  return [
    { id: 's1', ts: ts(), level: 'info', message: 'SloughGPT web console ready — set NEXT_PUBLIC_API_URL for API' },
    { id: 's2', ts: ts(), level: 'info', message: 'Session provider initialized' },
    { id: 's3', ts: ts(), level: 'warn', message: 'If the API is offline, the home card shows disconnected' },
  ]
}
