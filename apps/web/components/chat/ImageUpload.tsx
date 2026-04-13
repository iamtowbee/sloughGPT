'use client'

import { useCallback, useRef } from 'react'
import { cn } from '@/lib/cn'

interface ImageUploadProps {
  onImage: (dataUrl: string) => void
  disabled?: boolean
}

function ImageIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
  )
}

function XIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  )
}

export interface ImageAttachment {
  id: string
  dataUrl: string
  name: string
}

interface ImagePreviewProps {
  image: ImageAttachment
  onRemove: (id: string) => void
}

export function ImagePreview({ image, onRemove }: ImagePreviewProps) {
  return (
    <div className="relative group">
      <img 
        src={image.dataUrl} 
        alt={image.name}
        className="h-16 w-16 rounded-lg object-cover border border-border"
      />
      <button
        type="button"
        onClick={() => onRemove(image.id)}
        className="absolute -top-1.5 -right-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-destructive/80 text-destructive-foreground opacity-0 transition-opacity hover:bg-destructive group-hover:opacity-100"
        aria-label={`Remove ${image.name}`}
      >
        <XIcon className="h-3 w-3" />
      </button>
    </div>
  )
}

export function ImageUpload({ onImage, disabled }: ImageUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.type.startsWith('image/')) return

    const reader = new FileReader()
    reader.onload = (event) => {
      const dataUrl = event.target?.result as string
      onImage(dataUrl)
    }
    reader.readAsDataURL(file)

    e.target.value = ''
  }, [onImage])

  return (
    <div className="relative">
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        disabled={disabled}
        className="absolute inset-0 z-10 h-full w-full cursor-pointer opacity-0"
        aria-label="Upload image"
      />
      <button
        type="button"
        disabled={disabled}
        className={cn(
          "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground",
          disabled && "opacity-50 cursor-not-allowed"
        )}
        aria-label="Upload image"
        title="Upload image"
      >
        <ImageIcon className="h-5 w-5" />
      </button>
    </div>
  )
}
