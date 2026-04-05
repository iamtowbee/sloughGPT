import type { HTMLAttributes } from 'react'

import { cn } from '../../lib/cn'

export interface SkeletonProps extends HTMLAttributes<HTMLDivElement> {}

/** Pulse placeholder for loading rows, cards, and chat bubbles. */
export function Skeleton({ className, ...props }: SkeletonProps) {
  return (
    <div
      className={cn('animate-pulse rounded-none bg-muted/80', className)}
      aria-hidden
      {...props}
    />
  )
}
