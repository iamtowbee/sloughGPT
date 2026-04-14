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
        <div 
          className="absolute inset-0 h-full"
          style={{
            background: `repeating-linear-gradient(
              -45deg,
              transparent,
              transparent 8px,
              rgba(59, 130, 246, 0.4) 8px,
              rgba(59, 130, 246, 0.4) 16px
            )`,
            backgroundSize: '200% 100%',
            animation: 'progress-stripes 1s linear infinite',
          }}
        />
      ) : (
        <>
          <div 
            className="absolute inset-0 h-full"
            style={{
              background: `repeating-linear-gradient(
                -45deg,
                transparent,
                transparent 6px,
                rgba(59, 130, 246, 0.15) 6px,
                rgba(59, 130, 246, 0.15) 12px
              )`,
              backgroundSize: '200% 100%',
              animation: 'progress-stripes 0.8s linear infinite',
            }}
          />
          <div
            className="absolute inset-y-0 left-0 h-full bg-primary transition-[width] duration-300 ease-smooth"
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
