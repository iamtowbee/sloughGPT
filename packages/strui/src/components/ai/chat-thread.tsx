import * as React from 'react'

import { cn } from '@/lib/cn'

export interface ChatThreadProps extends React.HTMLAttributes<HTMLDivElement> {
  /** When true, thread is announced politely for new messages (screen readers). */
  live?: boolean
}

/**
 * Scrollable message list. Uses `str-chat-scroll` + safe horizontal insets for notched phones / PWA.
 * Pair with {@link PromptComposer} using `str-safe-bottom` so the composer clears the home indicator.
 */
export const ChatThread = React.forwardRef<HTMLDivElement, ChatThreadProps>(
  ({ className, children, live = true, ...props }, ref) => (
    <div
      ref={ref}
      role="log"
      aria-live={live ? 'polite' : 'off'}
      aria-relevant="additions"
      className={cn(
        'str-chat-scroll str-safe-x flex flex-col gap-3 overflow-y-auto p-3 sm:gap-3.5 sm:p-4',
        'max-h-[min(70dvh,36rem)] sm:max-h-[min(75vh,40rem)]',
        className,
      )}
      {...props}
    >
      {children}
    </div>
  ),
)
ChatThread.displayName = 'ChatThread'
