import { cn } from '../../lib/cn'

export interface StepIndicatorProps {
  steps: readonly string[]
  /** Zero-based index of the active step. */
  current: number
  className?: string
}

/** Horizontal wizard / pipeline steps (training stages, onboarding). */
export function StepIndicator({ steps, current, className }: StepIndicatorProps) {
  return (
    <ol className={cn('flex flex-wrap items-center gap-2 text-xs sm:gap-3', className)}>
      {steps.map((label, i) => {
        const done = i < current
        const active = i === current
        return (
          <li key={i} className="flex items-center gap-2">
            {i > 0 ? (
              <span className="hidden h-px w-4 bg-border sm:block" aria-hidden />
            ) : null}
            <span
              className={cn(
                'inline-flex items-center gap-2 rounded-none border px-2 py-1 font-medium',
                done && 'border-success/40 bg-success/10 text-success',
                active && 'border-primary bg-primary/15 text-foreground',
                !done && !active && 'border-border bg-muted/30 text-muted-foreground',
              )}
            >
              <span className="tabular-nums text-[0.65rem] opacity-80">{i + 1}</span>
              {label}
            </span>
          </li>
        )
      })}
    </ol>
  )
}
