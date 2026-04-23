'use client'

import { useCallback, useRef, useState } from 'react'
import { api } from '@/lib/api'
import type { ChatMessage } from '@/components/chat'

export interface StreamingState {
  isStreaming: boolean
  isPaused: boolean
  streamingMessageId: string | null
  tokensGenerated: number
  timeElapsed: number
  error: string | null
}

export interface UseStreamingChatOptions {
  onToken?: (messageId: string, token: string) => void
  onComplete?: (messageId: string, fullContent: string) => void
  onError?: (messageId: string, error: string) => void
  onStart?: (messageId: string) => void
}

export interface UseStreamingChatReturn {
  state: StreamingState
  startStream: (params: {
    messageId: string
    messages: Array<{ role: string; content: string }>
    model?: string
    systemPrompt?: string
    temperature?: number
    maxTokens?: number
    knowledge?: Array<{ content: string }>
  }) => Promise<void>
  stopStream: () => void
  pauseStream: () => void
  resumeStream: () => void
  resetState: () => void
}

export function useStreamingChat(options: UseStreamingChatOptions = {}): UseStreamingChatReturn {
  const [state, setState] = useState<StreamingState>({
    isStreaming: false,
    isPaused: false,
    streamingMessageId: null,
    tokensGenerated: 0,
    timeElapsed: 0,
    error: null,
  })

  const abortControllerRef = useRef<AbortController | null>(null)
  const startTimeRef = useRef<number>(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const tokensRef = useRef<number>(0)
  const fullContentRef = useRef<string>('')

  const stopStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    setState(prev => ({
      ...prev,
      isStreaming: false,
      isPaused: false,
    }))
  }, [])

  const pauseStream = useCallback(() => {
    setState(prev => ({
      ...prev,
      isPaused: true,
    }))
  }, [])

  const resumeStream = useCallback(() => {
    setState(prev => ({
      ...prev,
      isPaused: false,
    }))
  }, [])

  const resetState = useCallback(() => {
    stopStream()
    setState({
      isStreaming: false,
      isPaused: false,
      streamingMessageId: null,
      tokensGenerated: 0,
      timeElapsed: 0,
      error: null,
    })
    tokensRef.current = 0
    fullContentRef.current = ''
  }, [stopStream])

  const startStream = useCallback(async (params: {
    messageId: string
    messages: Array<{ role: string; content: string }>
    model?: string
    systemPrompt?: string
    temperature?: number
    maxTokens?: number
    knowledge?: Array<{ content: string }>
  }) => {
    const { messageId, messages, model, systemPrompt, temperature, maxTokens, knowledge } = params

    abortControllerRef.current = new AbortController()
    startTimeRef.current = Date.now()
    tokensRef.current = 0
    fullContentRef.current = ''

    setState(prev => ({
      ...prev,
      isStreaming: true,
      isPaused: false,
      streamingMessageId: messageId,
      tokensGenerated: 0,
      timeElapsed: 0,
      error: null,
    }))

    timerRef.current = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000)
      setState(prev => ({
        ...prev,
        timeElapsed: elapsed,
        tokensGenerated: tokensRef.current,
      }))
    }, 100)

    options.onStart?.(messageId)

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages,
          model: model || 'gpt2',
          system_prompt: systemPrompt,
          max_new_tokens: maxTokens || 200,
          temperature: temperature || 0.8,
          knowledge: knowledge || [],
        }),
        signal: abortControllerRef.current.signal,
      })

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown error')
        throw new Error(errorText)
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error('No response body')
      }

      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))

              if (data.error) {
                throw new Error(data.error)
              }

              if (data.token) {
                tokensRef.current++
                fullContentRef.current += data.token
                setState(prev => ({
                  ...prev,
                  tokensGenerated: tokensRef.current,
                }))
                options.onToken?.(messageId, data.token)
              }

              if (data.done) {
                options.onComplete?.(messageId, fullContentRef.current)
                break
              }
            } catch (e) {
              if (e instanceof SyntaxError) continue
              throw e
            }
          }
        }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      
      if (errorMessage === 'The user aborted the request') {
        options.onComplete?.(messageId, fullContentRef.current)
      } else {
        setState(prev => ({
          ...prev,
          error: errorMessage,
        }))
        options.onError?.(messageId, errorMessage)
      }
    } finally {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
      setState(prev => ({
        ...prev,
        isStreaming: false,
        isPaused: false,
      }))
    }
  }, [options])

  return {
    state,
    startStream,
    stopStream,
    pauseStream,
    resumeStream,
    resetState,
  }
}

export interface StreamIndicatorProps {
  tokensGenerated: number
  timeElapsed: number
  isStreaming: boolean
  isPaused: boolean
}

export function StreamStatsDisplay({ tokensGenerated, timeElapsed, isStreaming }: StreamIndicatorProps) {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`
  }

  const tokensPerSecond = timeElapsed > 0 ? (tokensGenerated / timeElapsed).toFixed(1) : '0'

  if (!isStreaming) return null

  return (
    <div className="flex items-center gap-4 text-xs text-muted-foreground">
      <span>{tokensGenerated} tokens</span>
      <span>{formatTime(timeElapsed)}</span>
      <span>{tokensPerSecond} tok/s</span>
    </div>
  )
}