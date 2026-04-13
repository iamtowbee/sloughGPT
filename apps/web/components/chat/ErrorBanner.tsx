'use client'

import { Button } from '@/components/ui/button'

type ErrorType = 'network' | 'server' | 'model' | 'timeout' | 'unknown'

const ERROR_MESSAGES: Record<ErrorType, { title: string; suggestion: string }> = {
  network: {
    title: 'Connection failed',
    suggestion: 'Check if the API server is running on port 8000.',
  },
  server: {
    title: 'Server error',
    suggestion: 'The API returned an error. Try again or restart the server.',
  },
  model: {
    title: 'Model not loaded',
    suggestion: 'No model is currently loaded. Load a model first.',
  },
  timeout: {
    title: 'Request timed out',
    suggestion: 'The model took too long. Try a shorter response.',
  },
  unknown: {
    title: 'Something went wrong',
    suggestion: 'An unexpected error occurred. Please try again.',
  },
}

interface ErrorBannerProps {
  error: {
    type: ErrorType
    message: string
    canRetry: boolean
  }
  onRetry: () => void
  onDismiss: () => void
}

export function ErrorBanner({ error, onRetry, onDismiss }: ErrorBannerProps) {
  const info = ERROR_MESSAGES[error.type]

  return (
    <section className="shrink-0 border-b border-red-200 bg-red-50 px-3 py-2.5 dark:border-red-900/50 dark:bg-red-950/30 sm:px-4 sm:py-3">
      <div className="mx-auto flex max-w-2xl items-start gap-2 sm:gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-red-600 dark:text-red-400 sm:text-sm">
            {info.title}
          </p>
          <p className="mt-0.5 text-[10px] text-red-500/80 dark:text-red-400/60 sm:text-xs">
            {error.message}
          </p>
        </div>
        <div className="flex shrink-0 gap-1.5 sm:gap-2">
          {error.canRetry && (
            <Button 
              variant="outline" 
              size="sm" 
              onClick={onRetry}
              className="h-7 text-[10px] sm:h-8 sm:text-xs hover:opacity-80 active:opacity-70"
            >
              Retry
            </Button>
          )}
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={onDismiss}
            className="h-7 text-[10px] sm:h-8 sm:text-xs"
          >
            Dismiss
          </Button>
        </div>
      </div>
    </section>
  )
}

export function getErrorInfo(status: number, message?: string): { type: ErrorType; message: string; canRetry: boolean } {
  if (status === 0) {
    return { type: 'network', message: 'Could not connect to API server.', canRetry: true }
  }
  if (status === 404) {
    return { type: 'server', message: 'Chat endpoint not found. Is the API server running?', canRetry: true }
  }
  if (status === 500) {
    return { type: 'server', message: message || 'Internal server error.', canRetry: true }
  }
  if (status === 503) {
    return { type: 'model', message: message || 'Model not available.', canRetry: true }
  }
  return { type: 'unknown', message: message || `HTTP ${status}`, canRetry: true }
}
