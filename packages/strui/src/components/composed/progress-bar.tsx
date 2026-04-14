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
      className={cn('relative h-2 w-full overflow-hidden rounded-none bg-muted', className)}
      {...props}
    >
      {indeterminate ? (
        <div 
          className="absolute inset-0 h-full"
          style={{
            background: `repeating-linear-gradient(
              -45deg,
              #22c55e,
              #22c55e 8px,
              #16a34a 8px,
              #16a34a 16px
            )`,
            backgroundSize: '200% 100%',
            animation: 'progress-stripes 0.6s linear infinite',
          }}
        />
      ) : (
        <>
          <div 
            className="absolute inset-0 h-full"
            style={{
              background: `repeating-linear-gradient(
                -45deg,
                rgba(34, 197, 94, 0.25),
                rgba(34, 197, 94, 0.25) 6px,
                rgba(34, 197, 94, 0.4) 6px,
                rgba(34, 197, 94, 0.4) 12px
              )`,
              backgroundSize: '200% 100%',
              animation: 'progress-stripes 0.5s linear infinite',
            }}
          />
          <div
            className="absolute inset-y-0 left-0 h-full bg-gradient-to-r from-green-500 to-emerald-500 transition-[width] duration-300 ease-smooth"
            style={{ width: `${pct}%` }}
          />
        </>
      )}
      <style>{`
        @keyframes progress-stripes {
          0% { background-position: 0 0; }
          100% { background-position: 200% 0; }
        }
      `}</style>
    </div>
  )
}
