import type { HTMLAttributes } from 'react'

import { cn } from '../../lib/cn'

export interface ProgressBarProps extends Omit<HTMLAttributes<HTMLDivElement>, 'children'> {
  /** 0–`max` (ignored when `indeterminate`). */
  value?: number
  max?: number
  /** Visually indeterminate (ignores value). */
  indeterminate?: boolean
}

/** Accessible linear progress for jobs, uploads, and context fill. */
export function ProgressBar({
  value = 0,
  max = 100,
  className,
  indeterminate,
  ...props
}: ProgressBarProps) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))

  return (
    <div
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={max}
      aria-valuenow={indeterminate ? undefined : Math.round(value)}
      aria-valuetext={indeterminate ? undefined : `${Math.round(pct)}%`}
      className={cn('relative h-2 w-full overflow-hidden rounded-none bg-muted/50', className)}
      {...props}
    >
      {indeterminate ? (
        <div className="absolute inset-0 w-full h-full animate-pulse bg-primary/30" />
      ) : (
        <>
          <div className="absolute inset-0 w-full h-full animate-pulse bg-primary/20" />
          <div
            className="absolute inset-y-0 left-0 h-full bg-primary transition-[width] duration-300 ease-smooth"
            style={{ width: `${pct}%` }}
          />
        </>
      )}
    </div>
  )
}
