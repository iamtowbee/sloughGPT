'use client'

import { cn } from '@/lib/cn'

interface TypingDotsProps {
  className?: string
  size?: 'sm' | 'md' | 'lg'
  color?: 'muted' | 'primary' | 'gradient'
}

export function TypingDots({ className, size = 'md', color = 'primary' }: TypingDotsProps) {
  const sizes = {
    sm: 'h-1.5 w-1.5',
    md: 'h-2 w-2',
    lg: 'h-2.5 w-2.5',
  }

  const gradients = {
    muted: 'bg-muted-foreground/60',
    primary: 'bg-primary',
    gradient: 'bg-gradient-to-r from-primary to-violet-500',
  }

  return (
    <div className={cn('flex items-center gap-1', className)}>
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className={cn(
            'rounded-full animate-bounce shadow-sm',
            sizes[size],
            gradients[color],
            color === 'gradient' && 'opacity-90'
          )}
          style={{
            animationDelay: `${i * 150}ms`,
            animationDuration: '600ms',
          }}
        />
      ))}
    </div>
  )
}

interface TypingIndicatorProps {
  className?: string
  showLabel?: boolean
  label?: string
}

export function TypingIndicator({ className, showLabel = true, label = 'Thinking' }: TypingIndicatorProps) {
  return (
    <div className={cn('flex items-center gap-3', className)}>
      <div className="relative">
        <div className="h-8 w-8 rounded-full bg-gradient-to-br from-primary/20 to-violet-500/20 flex items-center justify-center">
          <span className="text-sm">🤖</span>
        </div>
        <div className="absolute -bottom-0.5 -right-0.5">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-primary"></span>
          </span>
        </div>
      </div>
      <div className="flex items-center gap-2">
        {showLabel && (
          <span className="text-xs text-muted-foreground/70 font-medium">{label}</span>
        )}
        <TypingDots size="sm" color="primary" />
      </div>
    </div>
  )
}