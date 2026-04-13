'use client'

import { useState } from 'react'
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
  messageId 
}: MessageBubbleProps) {
  const hasContent = content && content.trim().length > 0
  const showActions = role === 'assistant' && hasContent
  const id = messageId || 'msg'

  return (
    <div
      className={cn(
        "flex flex-col animate-in slide-in-from-bottom-1 fade-in duration-300 ease-out",
        role === 'user' ? 'items-end' : 'items-start'
      )}
    >
      <div
        className={cn(
          "relative rounded-lg px-3 py-2 text-sm sm:px-4 sm:py-3 max-w-[85%] sm:max-w-[80%]",
          role === 'user'
            ? 'bg-primary text-primary-foreground rounded-br-sm'
            : 'bg-muted/80 text-foreground rounded-bl-sm'
        )}
      >
        {images && images.length > 0 && (
          <div className={cn(
            "flex gap-2 mb-2 flex-wrap",
            role === 'user' && "flex-row-reverse"
          )}>
            {images.map((img) => (
              <img
                key={img.id}
                src={img.dataUrl}
                alt={img.name}
                className="h-20 w-20 rounded-lg object-cover border border-current/20"
              />
            ))}
          </div>
        )}
        
        {content && (
          role === 'assistant' ? (
            <Markdown content={content} />
          ) : (
            <p className="whitespace-pre-wrap break-words leading-relaxed">{content}</p>
          )
        )}
        
        {showTimestamp && (
          <p className={cn(
            "mt-1 text-xs opacity-40",
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
