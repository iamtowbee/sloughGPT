'use client'

import { useEffect, useState } from 'react'
import { cn } from '@/lib/cn'
import { Markdown } from './Markdown'
import { MessageActions } from './MessageActions'
import type { ImageAttachment } from './ImageUpload'

export interface MessageBubbleProps {
  content: string
  role: 'user' | 'assistant'
  timestamp: Date | string
  showTimestamp: boolean
  images?: ImageAttachment[]
  onCopy?: (text: string) => void
  onRegenerate?: () => void
  onThumbsUp?: (messageId: string) => void
  onThumbsDown?: (messageId: string) => void
  messageId?: string
  isStreaming?: boolean
}

function formatTime(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function MessageBubble({ 
  content, 
  role, 
  timestamp, 
  showTimestamp, 
  images, 
  onCopy, 
  onRegenerate,
  onThumbsUp,
  onThumbsDown,
  messageId,
  isStreaming = false,
}: MessageBubbleProps) {
  const [displayContent, setDisplayContent] = useState(content)
  const [isVisible, setIsVisible] = useState(false)
  
  useEffect(() => {
    setIsVisible(true)
  }, [])
  
  useEffect(() => {
    setDisplayContent(content)
  }, [content])

  const hasContent = displayContent && displayContent.trim().length > 0
  const showActions = role === 'assistant' && hasContent && !isStreaming
  const id = messageId || 'msg'

  return (
    <div
      className={cn(
        "flex flex-col transition-all duration-300 ease-out",
        isVisible 
          ? "opacity-100 translate-y-0" 
          : "opacity-0 translate-y-2",
        role === 'user' ? 'items-end' : 'items-start'
      )}
    >
      <div
        className={cn(
          "relative rounded-2xl px-4 py-3 text-sm sm:px-5 sm:py-3.5 max-w-[85%] sm:max-w-[75%] transition-shadow duration-200",
          role === 'user'
            ? 'bg-primary text-primary-foreground rounded-br-md shadow-sm hover:shadow-md'
            : 'bg-muted/90 text-foreground rounded-bl-md border border-border/50 hover:bg-muted',
          isStreaming && role === 'assistant' && "cursor-wait"
        )}
      >
        {images && images.length > 0 && (
          <div className={cn(
            "flex gap-2 mb-3 flex-wrap",
            role === 'user' && "flex-row-reverse"
          )}>
            {images.map((img) => (
              <img
                key={img.id}
                src={img.dataUrl}
                alt={img.name}
                className="h-24 w-24 rounded-xl object-cover border border-current/20 shadow-sm hover:shadow-md transition-shadow"
              />
            ))}
          </div>
        )}
        
        {hasContent && (
          role === 'assistant' ? (
            <div className="leading-relaxed">
              <Markdown content={displayContent} />
              {isStreaming && (
                <span className="inline-block ml-1 animate-pulse">▊</span>
              )}
            </div>
          ) : (
            <p className="whitespace-pre-wrap break-words leading-relaxed">{displayContent}</p>
          )
        )}
        
        {!hasContent && role === 'assistant' && (
          <div className="flex gap-1 items-center h-5">
            <span className="w-2 h-2 bg-foreground/30 rounded-full animate-bounce [animation-delay:0ms]" />
            <span className="w-2 h-2 bg-foreground/30 rounded-full animate-bounce [animation-delay:150ms]" />
            <span className="w-2 h-2 bg-foreground/30 rounded-full animate-bounce [animation-delay:300ms]" />
          </div>
        )}
        
        {showTimestamp && (
          <p className={cn(
            "mt-1.5 text-[11px] opacity-50 font-medium",
            role === 'user' ? 'text-right' : 'text-left'
          )}>
            {formatTime(timestamp)}
          </p>
        )}
      </div>

      {showActions && (
        <MessageActions
          content={content}
          messageId={id}
          onCopy={onCopy}
          onRegenerate={onRegenerate}
          onThumbsUp={onThumbsUp}
          onThumbsDown={onThumbsDown}
        />
      )}
    </div>
  )
}
