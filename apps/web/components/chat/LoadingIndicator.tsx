'use client'

import { TypingDots } from './TypingDots'

export function LoadingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="rounded-2xl rounded-bl-md bg-muted/90 px-4 py-3 text-sm text-muted-foreground backdrop-blur-sm dark:bg-muted/70">
        <TypingDots />
      </div>
    </div>
  )
}
