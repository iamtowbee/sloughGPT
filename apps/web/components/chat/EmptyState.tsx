'use client'

interface EmptyStateProps {
  hasModel: boolean
}

function ChatBubbleIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
    </svg>
  )
}

export function EmptyState({ hasModel }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-12 text-center">
      <div className="relative">
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
          <ChatBubbleIcon className="h-8 w-8 text-primary/60" />
        </div>
        <div className="absolute -bottom-1 -right-1 flex h-6 w-6 items-center justify-center rounded-full bg-primary/20">
          <span className="text-[10px] font-medium">AI</span>
        </div>
      </div>
      <div className="space-y-1">
        <p className="text-sm font-medium text-foreground">
          {hasModel ? 'Ready to chat' : 'Model loading...'}
        </p>
        <p className="text-xs text-muted-foreground max-w-[200px]">
          {hasModel 
            ? 'Send a message to start a conversation' 
            : 'Please wait while the model initializes'}
        </p>
      </div>
      <div className="flex items-center gap-4 text-[10px] text-muted-foreground/60">
        <span className="flex items-center gap-1">
          <kbd className="rounded bg-muted px-1.5 py-0.5 font-mono">Enter</kbd>
          to send
        </span>
        <span>•</span>
        <span className="flex items-center gap-1">
          <kbd className="rounded bg-muted px-1.5 py-0.5 font-mono">Shift</kbd>
          + Enter for new line
        </span>
      </div>
    </div>
  )
}
