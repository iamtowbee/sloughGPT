import type { ButtonHTMLAttributes, ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface ListRowProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  title: string
  description?: ReactNode
  /** Trailing icon or meta (chevron, badge). */
  trailing?: ReactNode
}

/**
 * Selectable row for conversation lists, datasets, experiments (use `type="button"`).
 */
export function ListRow({ title, description, trailing, className, ...props }: ListRowProps) {
  return (
    <button
      type="button"
      className={cn(
        'flex w-full items-center gap-3 rounded-none border border-transparent px-3 py-3 text-left transition-colors',
        'hover:border-border hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        className,
      )}
      {...props}
    >
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-foreground">{title}</p>
        {description ? (
          <div className="mt-0.5 line-clamp-2 text-xs text-muted-foreground">{description}</div>
        ) : null}
      </div>
      {trailing ? <div className="shrink-0 text-muted-foreground">{trailing}</div> : null}
    </button>
  )
}
