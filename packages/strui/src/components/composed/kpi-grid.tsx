import type { ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface KpiGridProps {
  children: ReactNode
  className?: string
}

/** Responsive grid for **StatCard** tiles. */
export function KpiGrid({ children, className }: KpiGridProps) {
  return (
    <div
      className={cn('str-safe-x grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4', className)}
    >
      {children}
    </div>
  )
}
