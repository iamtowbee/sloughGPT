'use client'

import { useRef, useCallback, KeyboardEvent, ChangeEvent } from 'react'
import { Button } from '@/components/ui/button'
import { VoiceInput } from './VoiceInput'
import { ImageUpload, ImagePreview, type ImageAttachment } from './ImageUpload'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import { cn } from '@/lib/cn'

interface ChatInputProps {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  loading: boolean
  health: ApiHealthSnapshot
  images?: ImageAttachment[]
  onAddImage?: (dataUrl: string) => void
  onRemoveImage?: (id: string) => void
}

function autoResize(textarea: HTMLTextAreaElement | null) {
  if (textarea) {
    textarea.style.height = 'auto'
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`
  }
}

export function ChatInput({ 
  value, 
  onChange, 
  onSend, 
  loading, 
  health,
  images = [],
  onAddImage,
  onRemoveImage,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleChange = useCallback((e: ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value)
    autoResize(e.target.value ? textareaRef.current : null)
  }, [onChange])

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      onSend()
    }
  }, [onSend])

  const handleSend = useCallback(() => {
    onSend()
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }, [onSend])

  const handleVoiceTranscript = useCallback((text: string) => {
    onChange(value ? `${value} ${text}` : text)
    if (textareaRef.current) {
      autoResize(textareaRef.current)
    }
  }, [value, onChange])

  const handleAddImage = useCallback((dataUrl: string) => {
    if (onAddImage) {
      onAddImage(dataUrl)
    }
  }, [onAddImage])

  const handleRemoveImage = useCallback((id: string) => {
    if (onRemoveImage) {
      onRemoveImage(id)
    }
  }, [onRemoveImage])

  const isDisabled = loading || health === 'offline'
  const hasModel = health !== null && health !== 'offline' && 'model_loaded' in health && health.model_loaded
  const placeholder = health === 'offline' 
    ? 'API offline...' 
    : hasModel 
      ? 'Type a message...' 
      : 'Loading model...'

  return (
    <section 
      className="shrink-0 border-t border-border/50 px-3 py-3 sm:px-4 sm:py-4" 
      style={{ paddingBottom: 'max(0.75rem, env(safe-area-inset-bottom))' }}
    >
      <div className="mx-auto max-w-2xl space-y-2">
        {images.length > 0 && (
          <div className="flex gap-2 flex-wrap">
            {images.map((img) => (
              <ImagePreview 
                key={img.id} 
                image={img} 
                onRemove={handleRemoveImage}
              />
            ))}
          </div>
        )}
        
        <div className="flex items-end gap-2 sm:gap-3">
          <div className="flex items-end gap-1">
            <ImageUpload onImage={handleAddImage} disabled={isDisabled} />
            <VoiceInput onTranscript={handleVoiceTranscript} disabled={isDisabled} />
          </div>
          
          <textarea
            ref={textareaRef}
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={isDisabled}
            className="flex-1 resize-none rounded-xl border border-input bg-background px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary disabled:cursor-not-allowed disabled:opacity-50 placeholder:text-muted-foreground/50 transition-all duration-200"
            style={{ minHeight: '44px', maxHeight: '160px' }}
            rows={1}
            aria-label="Message input"
          />
          
          <Button 
            onClick={handleSend} 
            disabled={isDisabled || (!value.trim() && images.length === 0)}
            className="h-11 w-11 shrink-0 p-0 sm:h-12 sm:w-auto sm:px-5 hover:opacity-80 active:opacity-70 disabled:opacity-50"
            aria-label="Send message"
          >
            <svg className="h-4 w-4 sm:hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
            <span className="hidden sm:inline">{loading ? '...' : 'Send'}</span>
          </Button>
        </div>
      </div>
    </section>
  )
}
