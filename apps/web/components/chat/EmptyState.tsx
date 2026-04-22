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
    <div className="flex flex-col items-center justify-center gap-4 py-10 text-center">
      <div className="relative">
        <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br from-primary/10 to-primary/5 shadow-sm">
          <ChatBubbleIcon className="h-7 w-7 text-primary/70" />
        </div>
        <div className="absolute -bottom-0.5 -right-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-primary/20 border-2 border-background">
          <span className="text-[9px] font-semibold text-primary">AI</span>
        </div>
      </div>
      <div className="space-y-1.5">
        <p className="text-sm font-semibold text-foreground">
          {hasModel ? 'Ready to chat' : 'Loading model...'}
        </p>
        <p className="text-xs text-muted-foreground/80 max-w-[220px]">
          {hasModel 
            ? 'Send a message to start a conversation' 
            : 'Please wait while the model initializes'}
        </p>
      </div>
      <div className="flex items-center gap-2 text-[10px] text-muted-foreground/60 bg-muted/40 px-3 py-1.5 rounded-full">
        <span className="flex items-center gap-1">
          <kbd className="rounded bg-background px-1.5 py-0.5 font-mono text-[9px] shadow-sm border">↵</kbd>
          <span>send</span>
        </span>
        <span className="text-muted-foreground/40">|</span>
        <span className="flex items-center gap-1">
          <kbd className="rounded bg-background px-1.5 py-0.5 font-mono text-[9px] shadow-sm border">Shift</kbd>
          <span>+ Enter for new line</span>
        </span>
      </div>
    </div>
  )
}
