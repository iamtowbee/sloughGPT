'use client'

import { useState, useCallback } from 'react'
import { cn } from '@/lib/cn'

interface MessageActionsProps {
  content: string
  messageId: string
  onCopy?: (text: string) => void
  onRegenerate?: () => void
  onThumbsUp?: (messageId: string) => void
  onThumbsDown?: (messageId: string) => void
  onEdit?: (messageId: string) => void
}

function CopyIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
    </svg>
  )
}

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
    </svg>
  )
}

function RefreshIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
    </svg>
  )
}

function ThumbsUpIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 19V5M5 12l7-7 7 7" />
    </svg>
  )
}

function ThumbsDownIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 5v14M5 12l7 7 7-7" />
    </svg>
  )
}

function EditIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
    </svg>
  )
}

export function MessageActions({ content, messageId, onCopy, onRegenerate, onThumbsUp, onThumbsDown, onEdit }: MessageActionsProps) {
  const [copied, setCopied] = useState(false)
  const [thumbsUp, setThumbsUp] = useState(false)
  const [thumbsDown, setThumbsDown] = useState(false)

  const handleCopy = useCallback(async () => {
    if (!content || !onCopy) return
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      onCopy(content)
      setTimeout(() => setCopied(false), 1500)
    } catch {}
  }, [content, onCopy])

  const handleThumbsUp = useCallback(() => {
    const newVal = !thumbsUp
    setThumbsUp(newVal)
    if (newVal) setThumbsDown(false)
    onThumbsUp?.(messageId)
  }, [thumbsUp, messageId, onThumbsUp])

  const handleThumbsDown = useCallback(() => {
    const newVal = !thumbsDown
    setThumbsDown(newVal)
    if (newVal) setThumbsUp(false)
    onThumbsDown?.(messageId)
  }, [thumbsDown, messageId, onThumbsDown])

  return (
    <div className="flex items-center gap-1 mt-1">
      {onCopy && (
        <button
          onClick={handleCopy}
          className={cn(
            "p-1 rounded text-muted-foreground/60 hover:bg-secondary/60 hover:text-foreground transition-all",
            copied && "text-green-500"
          )}
          aria-label="Copy"
          title="Copy"
        >
          {copied ? <CheckIcon className="h-4 w-4" /> : <CopyIcon className="h-4 w-4" />}
        </button>
      )}
      
      {onRegenerate && (
        <button
          onClick={onRegenerate}
          className="p-1 rounded text-muted-foreground/60 hover:bg-secondary/60 hover:text-foreground transition-all"
          aria-label="Regenerate"
          title="Regenerate"
        >
          <RefreshIcon className="h-4 w-4" />
        </button>
      )}
      
      {onThumbsUp && (
        <button
          onClick={handleThumbsUp}
          className={cn(
            "p-1 rounded text-muted-foreground/60 hover:bg-secondary/60 hover:text-foreground transition-all",
            thumbsUp && "text-green-500"
          )}
          aria-label="Good"
          title="Good"
        >
          <ThumbsUpIcon className="h-4 w-4" />
        </button>
      )}
      
      {onThumbsDown && (
        <button
          onClick={handleThumbsDown}
          className={cn(
            "p-1 rounded text-muted-foreground/60 hover:bg-secondary/60 hover:text-foreground transition-all",
            thumbsDown && "text-red-500"
          )}
          aria-label="Bad"
          title="Bad"
        >
          <ThumbsDownIcon className="h-4 w-4" />
        </button>
      )}
      
      {onEdit && (
        <button
          onClick={() => onEdit(messageId)}
          className="p-1 rounded text-muted-foreground/60 hover:bg-secondary/60 hover:text-foreground transition-all"
          aria-label="Edit"
          title="Edit and resend"
        >
          <EditIcon className="h-4 w-4" />
        </button>
      )}
    </div>
  )
}
