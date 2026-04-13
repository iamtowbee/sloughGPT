'use client'

import { forwardRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { MessageBubble } from './MessageBubble'
import { EmptyState } from './EmptyState'
import { LoadingIndicator } from './LoadingIndicator'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

interface ChatMessagesProps {
  messages: ChatMessage[]
  loading: boolean
  isStreaming: boolean
  health: ApiHealthSnapshot
  onRefreshHealth: () => void
  onCopy: (text: string) => void
  onRegenerate?: () => void
  onThumbsUp?: (messageId: string) => void
  onThumbsDown?: (messageId: string) => void
}

export const ChatMessages = forwardRef<HTMLDivElement, ChatMessagesProps>(
  function ChatMessages({ messages, loading, isStreaming, health, onRefreshHealth, onCopy, onRegenerate, onThumbsUp, onThumbsDown }, ref) {
    const isOffline = health === 'offline'
    const hasModel = health !== null && health !== 'offline' && health.model_loaded

    return (
      <section className="flex-1 min-h-0 overflow-y-auto">
        <div className="mx-auto max-w-2xl px-3 py-4 sm:px-4 sm:py-6">
          {isOffline && (
            <div className="mb-4 rounded-lg border border-yellow-200/50 bg-yellow-50/80 p-3 text-xs text-yellow-800 dark:border-yellow-900/50 dark:bg-yellow-950/50 dark:text-yellow-300 sm:text-sm sm:p-4">
              <p className="font-medium">API Server Offline</p>
              <p className="mt-1 text-muted-foreground">The API server is not responding. Make sure it is running.</p>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={onRefreshHealth}
                className="mt-2 text-[10px] sm:text-xs hover:opacity-80 active:opacity-70"
              >
                Check Again
              </Button>
            </div>
          )}
          
          {messages.length === 0 && !isOffline && (
            <EmptyState hasModel={hasModel} />
          )}
          
          <div className="space-y-3 sm:space-y-4">
            {messages.map((msg, idx) => {
              const isLast = idx === messages.length - 1
              const isStreamingThis = isLast && loading && isStreaming
              const hasContent = msg.content.length > 0
              const isAssistantWithContent = msg.role === 'assistant' && hasContent && !isStreamingThis
              return (
                <MessageBubble
                  key={msg.id}
                  content={msg.content}
                  role={msg.role}
                  timestamp={msg.timestamp}
                  showTimestamp={isAssistantWithContent}
                  messageId={msg.id}
                  onCopy={msg.role === 'assistant' ? onCopy : undefined}
                  onRegenerate={msg.role === 'assistant' && isLast ? onRegenerate : undefined}
                  onThumbsUp={msg.role === 'assistant' ? onThumbsUp : undefined}
                  onThumbsDown={msg.role === 'assistant' ? onThumbsDown : undefined}
                />
              )
            })}
            
            {loading && (
              <LoadingIndicator />
            )}
          </div>
          
          <div ref={ref} className="h-4" />
        </div>
      </section>
    )
  }
)
