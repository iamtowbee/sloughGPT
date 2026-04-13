'use client'

import { cn } from '@/lib/cn'

interface TypingDotsProps {
  className?: string
}

export function TypingDots({ className }: TypingDotsProps) {
  return (
    <div className={cn("flex items-center gap-1", className)}>
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 animate-bounce rounded-full bg-current"
          style={{ animationDelay: `${i * 150}ms`, animationDuration: '600ms' }}
        />
      ))}
    </div>
  )
}
