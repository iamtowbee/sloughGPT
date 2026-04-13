'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import { useApiHealth } from '@/hooks/useApiHealth'
import { API_CHAT_ENDPOINT } from '@/lib/config'
import { api } from '@/lib/api'
import { useFeedbackStore } from '@/lib/feedback-store'
import {
  ChatHeader,
  ChatSettings,
  ChatMessages,
  ChatInput,
  ToastContainer,
  ErrorBanner,
  SessionSidebar,
  getErrorInfo,
  type ChatMessage,
  type Toast,
  type ImageAttachment,
} from '@/components/chat'

const STORAGE_KEY = 'sloughgpt_chat_sessions'
const CURRENT_SESSION_KEY = 'sloughgpt_current_session'
const USER_ID_KEY = 'sloughgpt_user_id'
const API_SESSION_CONTEXT_ENDPOINT = `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/session`

function getOrCreateUserId(): string {
  if (typeof window === 'undefined') return 'default'
  let userId = localStorage.getItem(USER_ID_KEY)
  if (!userId) {
    userId = `user_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
    localStorage.setItem(USER_ID_KEY, userId)
  }
  return userId
}

interface ChatSession {
  id: string
  name: string
  messages: ChatMessage[]
  createdAt: string
  updatedAt: string
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [showSidebar, setShowSidebar] = useState(false)
  const [model, setModel] = useState('gpt2')
  const [temperature, setTemperature] = useState(0.8)
  const [maxTokens, setMaxTokens] = useState(200)
  const [currentError, setCurrentError] = useState<ReturnType<typeof getErrorInfo> | null>(null)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [images, setImages] = useState<ImageAttachment[]>([])
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const sessionIdRef = useRef<string>('')
  const userIdRef = useRef<string>('default')
  const { state: health, refresh: refreshHealth } = useApiHealth()
  
  // Feedback store
  const { recordFeedback, fetchStats, fetchAdapterStats, stats, adapterStats } = useFeedbackStore()

  useEffect(() => {
    sessionIdRef.current = localStorage.getItem(CURRENT_SESSION_KEY) || `session-${Date.now()}`
    userIdRef.current = getOrCreateUserId()
    // Fetch initial feedback stats
    fetchStats()
    fetchAdapterStats()
  }, [])

  const showToast = useCallback((message: string, type: Toast['type'] = 'success') => {
    const id = Date.now().toString()
    setToasts(prev => [...prev, { id, message, type }])
  }, [])

  const dismissToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  const saveSession = useCallback(() => {
    if (messages.length === 0) return
    const sessionId = localStorage.getItem(CURRENT_SESSION_KEY) || `session-${Date.now()}`
    const sessionName = messages[0]?.content?.slice(0, 30) || 'New Chat'
    
    const session: ChatSession = {
      id: sessionId,
      name: sessionName,
      messages,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    }
    
    const sessions = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]') as ChatSession[]
    const existingIndex = sessions.findIndex(s => s.id === sessionId)
    
    if (existingIndex >= 0) {
      sessions[existingIndex] = session
    } else {
      sessions.unshift(session)
    }
    
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions.slice(0, 20)))
    localStorage.setItem(CURRENT_SESSION_KEY, sessionId)
  }, [messages])

  const loadSession = useCallback((sessionId: string) => {
    const sessions = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]') as ChatSession[]
    const session = sessions.find(s => s.id === sessionId)
    if (session) {
      setMessages(session.messages)
      localStorage.setItem(CURRENT_SESSION_KEY, sessionId)
      showToast(`Loaded: ${session.name}`)
    }
  }, [showToast])

  const newChat = useCallback(() => {
    const currentId = localStorage.getItem(CURRENT_SESSION_KEY)
    if (currentId && messages.length > 0) {
      saveSession()
    }
    setMessages([])
    localStorage.setItem(CURRENT_SESSION_KEY, `session-${Date.now()}`)
    showToast('New chat started')
  }, [messages, saveSession, showToast])

  const deleteSession = useCallback((sessionId: string) => {
    const sessions = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]') as ChatSession[]
    const filtered = sessions.filter(s => s.id !== sessionId)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered))
    if (localStorage.getItem(CURRENT_SESSION_KEY) === sessionId) {
      setMessages([])
      localStorage.removeItem(CURRENT_SESSION_KEY)
    }
    showToast('Session deleted')
  }, [showToast])

  useEffect(() => {
    if (messages.length > 0) {
      const timeout = setTimeout(saveSession, 1000)
      return () => clearTimeout(timeout)
    }
  }, [messages, saveSession])

  useEffect(() => {
    const currentId = localStorage.getItem(CURRENT_SESSION_KEY)
    if (currentId) {
      loadSession(currentId)
    }
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, isStreaming])

  const handleCopy = useCallback((text: string) => {
    showToast('Copied to clipboard')
  }, [showToast])

  const handleRegenerate = useCallback(() => {
    if (messages.length < 2) return
    
    const lastAssistantIdx = messages.findLastIndex(m => m.role === 'assistant')
    if (lastAssistantIdx === -1) return

    const contextMessages = messages.slice(0, lastAssistantIdx)
    const originalMsgId = messages[lastAssistantIdx].id

    storeSessionContext(sessionIdRef.current, contextMessages)

    setLoading(true)
    setIsStreaming(true)

    const assistantId = (Date.now() + 1).toString()
    setMessages(prev => [
      ...prev.slice(0, lastAssistantIdx),
      { ...prev[lastAssistantIdx], id: assistantId, content: '', timestamp: new Date() }
    ])

    fetch(`${API_SESSION_CONTEXT_ENDPOINT}/${sessionIdRef.current}/regenerate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    })
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          showToast('Regeneration failed', 'error')
        } else {
          const words = (data.text || '').split(' ')
          let idx = 0
          const interval = setInterval(() => {
            if (idx >= words.length) {
              clearInterval(interval)
              setIsStreaming(false)
              setLoading(false)
              return
            }
            setMessages(prev => prev.map(msg =>
              msg.id === assistantId
                ? { ...msg, content: prev.find(m => m.id === assistantId)?.content + words[idx] + ' ' }
                : msg
            ))
            idx++
          }, 15)
        }
      })
      .catch(() => {
        showToast('Regeneration failed', 'error')
        setLoading(false)
        setIsStreaming(false)
      })
  }, [messages, showToast])

  const storeSessionContext = async (sessionId: string, msgs: ChatMessage[]) => {
    try {
      await fetch(`${API_SESSION_CONTEXT_ENDPOINT}/${sessionId}/context`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: msgs.map(m => ({ role: m.role, content: m.content })) }),
      })
    } catch {}
  }

  const handleThumbsUp = useCallback(async (messageId: string) => {
    const msgIdx = messages.findIndex(m => m.id === messageId)
    const assistantMsg = messages[msgIdx]
    const userMsg = msgIdx > 0 ? messages[msgIdx - 1] : null

    const success = await recordFeedback({
      userMessage: userMsg?.content || '',
      assistantResponse: assistantMsg?.content || '',
      rating: 'thumbs_up',
      conversationId: sessionIdRef.current,
      userId: userIdRef.current,
    })
    
    if (success) {
      showToast('Thanks for the feedback!')
    } else {
      showToast('Failed to submit feedback', 'error')
    }
  }, [showToast, messages])

  const handleThumbsDown = useCallback(async (messageId: string) => {
    const msgIdx = messages.findIndex(m => m.id === messageId)
    const assistantMsg = messages[msgIdx]
    const userMsg = msgIdx > 0 ? messages[msgIdx - 1] : null

    const success = await recordFeedback({
      userMessage: userMsg?.content || '',
      assistantResponse: assistantMsg?.content || '',
      rating: 'thumbs_down',
      conversationId: sessionIdRef.current,
      userId: userIdRef.current,
    })
    
    if (success) {
      showToast('Thanks for the feedback!')
    } else {
      showToast('Failed to submit feedback', 'error')
    }
  }, [showToast, messages])

  const handleRetry = useCallback(() => {
    setCurrentError(null)
    sendMessage()
  }, [currentError])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (currentError) {
          setCurrentError(null)
        } else if (showSettings) {
          setShowSettings(false)
        }
      }
      if (e.key === '?' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        setShowSettings(prev => !prev)
      }
      if (e.key === 'n' && (e.metaKey || e.ctrlKey) && !e.shiftKey) {
        e.preventDefault()
        newChat()
      }
      if (e.key === 'r' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        handleRegenerate()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [currentError, showSettings, handleRegenerate])

  const handleAddImage = useCallback((dataUrl: string) => {
    const newImage: ImageAttachment = {
      id: Date.now().toString(),
      dataUrl,
      name: `image-${Date.now()}.png`,
    }
    setImages(prev => [...prev, newImage])
  }, [])

  const handleRemoveImage = useCallback((id: string) => {
    setImages(prev => prev.filter(img => img.id !== id))
  }, [])

  const sendMessage = async () => {
    if ((!input.trim() && images.length === 0) || loading) return
    
    const userImages = [...images]
    
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    }
    
    const assistantId = (Date.now() + 1).toString()
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    }
    
    setMessages(prev => [...prev, userMessage, assistantMessage])
    setInput('')
    setImages([])
    setCurrentError(null)
    setIsStreaming(true)
    setLoading(true)

    try {
      const response = await fetch(API_CHAT_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...messages, userMessage].map(m => ({ role: m.role, content: m.content })),
          model,
          max_new_tokens: maxTokens,
          temperature,
          user_id: userIdRef.current,
        }),
      })

      if (!response.ok) {
        const errorText = await response.text().catch(() => '')
        setCurrentError(getErrorInfo(response.status, errorText))
        setMessages(prev => prev.filter(msg => msg.id !== assistantId))
        setLoading(false)
        setIsStreaming(false)
        return
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let hasContent = false
      
      if (reader) {
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
                  setCurrentError(getErrorInfo(500, data.error))
                  setMessages(prev => prev.filter(msg => msg.id !== assistantId))
                  setIsStreaming(false)
                  return
                }
                if (data.token !== undefined && data.token) {
                  hasContent = true
                  setMessages(prev => prev.map(msg => 
                    msg.id === assistantId 
                      ? { ...msg, content: msg.content + data.token }
                      : msg
                  ))
                }
                if (data.done) break
              } catch {}
            }
          }
        }
      }

      if (!hasContent) {
        setMessages(prev => prev.map(msg => 
          msg.id === assistantId 
            ? { ...msg, content: '(empty response)' }
            : msg
        ))
      }
    } catch (err) {
      setCurrentError(getErrorInfo(0, err instanceof Error ? err.message : 'Network error'))
      setMessages(prev => prev.filter(msg => msg.id !== assistantId))
    } finally {
      setLoading(false)
      setIsStreaming(false)
    }
  }

  const clearChat = useCallback(() => {
    newChat()
  }, [newChat])

  const toggleSettings = useCallback(() => {
    setShowSettings(prev => !prev)
  }, [])

  return (
    <div className="flex flex-1 flex-col min-h-0">
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />

      <SessionSidebar
        isOpen={showSidebar}
        onClose={() => setShowSidebar(false)}
        currentSessionId={sessionIdRef.current}
        onLoadSession={loadSession}
        onDeleteSession={deleteSession}
        onNewChat={newChat}
      />

      <ChatHeader
        health={health}
        showSettings={showSettings}
        showSidebar={showSidebar}
        onToggleSettings={toggleSettings}
        onToggleSidebar={() => setShowSidebar(prev => !prev)}
        onNewChat={newChat}
      />

      <ChatSettings
        isOpen={showSettings}
        model={model}
        temperature={temperature}
        maxTokens={maxTokens}
        onModelChange={setModel}
        onTemperatureChange={setTemperature}
        onMaxTokensChange={setMaxTokens}
        onClear={clearChat}
        hasMessages={messages.length > 0}
      />

      {currentError && (
        <ErrorBanner
          error={currentError}
          onRetry={handleRetry}
          onDismiss={() => setCurrentError(null)}
        />
      )}

      <ChatMessages
        ref={messagesEndRef}
        messages={messages}
        loading={loading}
        isStreaming={isStreaming}
        health={health}
        onRefreshHealth={refreshHealth}
        onCopy={handleCopy}
        onRegenerate={handleRegenerate}
        onThumbsUp={handleThumbsUp}
        onThumbsDown={handleThumbsDown}
      />

      <ChatInput
        value={input}
        onChange={setInput}
        onSend={sendMessage}
        loading={loading}
        health={health}
        images={images}
        onAddImage={handleAddImage}
        onRemoveImage={handleRemoveImage}
      />
    </div>
  )
}
