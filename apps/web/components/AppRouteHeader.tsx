import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'

/** Standard title block for `AppRouteHeader` `left` slot (heading + optional muted line). */
export function AppRouteHeaderLead({
  title,
  subtitle,
  children,
}: {
  title: ReactNode
  subtitle?: ReactNode
  children?: ReactNode
}) {
  return (
    <div className="min-w-0">
      {typeof title === 'string' ? <h1 className="sl-h1">{title}</h1> : title}
      {subtitle != null ? <div className="mt-1 text-sm text-muted-foreground">{subtitle}</div> : null}
      {children}
    </div>
  )
}

export type AppRouteHeaderProps = {
  /** Primary actions / title — stays left at `lg`, wraps on small screens. */
  left: ReactNode
  /** Secondary cluster (status, tools) — stays right; use `justify-end` content. */
  right?: ReactNode
  className?: string
}

/**
 * Shared page header: one row, `justify-between`, gap between left and right clusters.
 * Use inside a max-width column (`mx-auto max-w-*`) per route; slots are composable.
 */
export function AppRouteHeader({ left, right, className }: AppRouteHeaderProps) {
  return (
    <header
      className={cn(
        'flex w-full min-w-0 flex-wrap items-center justify-between gap-x-4 gap-y-2',
        className,
      )}
    >
      <div className="flex min-w-0 flex-wrap items-center gap-2 md:gap-3">{left}</div>
      {right != null ? (
        <div className="flex min-w-0 shrink-0 flex-wrap items-center justify-end gap-1.5">{right}</div>
      ) : null}
    </header>
  )
}
