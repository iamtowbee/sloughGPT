import * as React from 'react'

import { cn } from '@/lib/cn'

export interface TokenMeterProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Approximate prompt + completion tokens. */
  total?: number
  contextLimit?: number
}

function formatTok(n: number) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}

/** Inline usage strip for chat headers (mobile-friendly, no chart deps). */
export function TokenMeter({ className, total = 0, contextLimit, ...props }: TokenMeterProps) {
  const pct =
    contextLimit && contextLimit > 0 ? Math.min(100, Math.round((total / contextLimit) * 100)) : null

  return (
    <div
      className={cn(
        'flex flex-wrap items-center gap-2 text-xs text-muted-foreground sm:text-[13px]',
        className,
      )}
      {...props}
    >
      <span className="font-mono tabular-nums">{formatTok(total)} tok</span>
      {pct !== null ? (
        <span
          className="h-1.5 min-w-[4rem] max-w-[8rem] flex-1 rounded-none bg-muted"
          title={`${pct}% of context`}
          aria-hidden
        >
          <span
            className="block h-full rounded-none bg-primary/70 transition-all duration-200 ease-smooth"
            style={{ width: `${pct}%` }}
          />
        </span>
      ) : null}
      {contextLimit ? (
        <span className="font-mono tabular-nums text-muted-foreground/80">/ {formatTok(contextLimit)}</span>
      ) : null}
    </div>
  )
}
