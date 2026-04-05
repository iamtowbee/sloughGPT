import * as React from 'react'

import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/cn'

export interface PromptComposerProps extends Omit<React.FormHTMLAttributes<HTMLFormElement>, 'onSubmit'> {
  value: string
  onValueChange: (value: string) => void
  onSubmit: () => void
  placeholder?: string
  sendLabel?: string
  disabled?: boolean
  /** Shown while a request is in flight; send button stays touch-sized. */
  busy?: boolean
}

/**
 * Bottom-stacked prompt area for mobile; row layout from `sm` and up.
 * Applies `str-safe-bottom` so content clears the iOS home indicator in standalone PWA.
 */
export function PromptComposer({
  className,
  value,
  onValueChange,
  onSubmit,
  placeholder = 'Message…',
  sendLabel = 'Send',
  disabled,
  busy,
  ...formProps
}: PromptComposerProps) {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!disabled && !busy && value.trim()) onSubmit()
  }

  return (
    <form
      className={cn(
        'str-safe-bottom border-t border-border bg-card/90 backdrop-blur-md',
        'p-3 sm:p-4',
        className,
      )}
      onSubmit={handleSubmit}
      {...formProps}
    >
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
        <Textarea
          value={value}
          onChange={(e) => onValueChange(e.target.value)}
          placeholder={placeholder}
          disabled={disabled || busy}
          rows={3}
          className="min-h-[100px] flex-1 sm:min-h-20"
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              if (!disabled && !busy && value.trim()) onSubmit()
            }
          }}
          aria-label={placeholder}
        />
        <Button
          type="submit"
          disabled={disabled || busy || !value.trim()}
          className="str-touch-target w-full shrink-0 sm:w-auto"
        >
          {busy ? '…' : sendLabel}
        </Button>
      </div>
    </form>
  )
}
