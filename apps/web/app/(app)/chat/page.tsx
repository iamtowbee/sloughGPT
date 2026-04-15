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
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [currentError, setCurrentError] = useState<ReturnType<typeof getErrorInfo> | null>(null)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [images, setImages] = useState<ImageAttachment[]>([])
  const [sessionSaved, setSessionSaved] = useState(false)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const sessionIdRef = useRef<string>('')
  const userIdRef = useRef<string>('default')
  const { state: health, refresh: refreshHealth } = useApiHealth()
  
  // Feedback store
  const { recordFeedback, fetchStats, fetchAdapterStats, stats, adapterStats } = useFeedbackStore()

  // Generate hash ID for session
  const generateSessionId = () => {
    const chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    let hash = ''
    for (let i = 0; i < 8; i++) {
      hash += chars[Math.floor(Math.random() * chars.length)]
    }
    return `chat_${hash}`
  }

  useEffect(() => {
    sessionIdRef.current = localStorage.getItem(CURRENT_SESSION_KEY) || generateSessionId()
    userIdRef.current = getOrCreateUserId()
    // Fetch initial feedback stats
    fetchStats()
    fetchAdapterStats()
    // Fetch available models
    api.getModels().then((models) => {
      setAvailableModels(models.map((m: { id: string }) => m.id))
    }).catch(() => {})
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
      setSessionSaved(true)
      showToast(`Loaded: ${session.name}`)
    }
  }, [showToast])

  const newChat = useCallback(() => {
    setMessages([])
    setSessionSaved(false)
    const newId = generateSessionId()
    sessionIdRef.current = newId
    localStorage.setItem(CURRENT_SESSION_KEY, newId)
    showToast('New chat started')
  }, [showToast])

  const deleteSession = useCallback((sessionId: string) => {
    const sessions = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]') as ChatSession[]
    const filtered = sessions.filter(s => s.id !== sessionId)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered))
    if (localStorage.getItem(CURRENT_SESSION_KEY) === sessionId) {
      setMessages([])
      setSessionSaved(false)
      localStorage.removeItem(CURRENT_SESSION_KEY)
    }
    showToast('Session deleted')
  }, [showToast])

  // Only auto-save if session has received a successful response
  useEffect(() => {
    if (messages.length > 0 && sessionSaved) {
      const timeout = setTimeout(saveSession, 1000)
      return () => clearTimeout(timeout)
    }
  }, [messages, saveSession, sessionSaved])

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
    
    // Get injected knowledge
    const injectedKnowledge = (() => {
      try {
        const stored = localStorage.getItem(KNOWLEDGE_STORAGE_KEY)
        return stored ? JSON.parse(stored) : []
      } catch {
        return []
      }
    })()
    
    // Build knowledge context
    let knowledgeContext = ''
    if (injectedKnowledge.length > 0) {
      knowledgeContext = `\n\n[IMPORTANT KNOWLEDGE - Use this information when responding:]\n${injectedKnowledge.map((k: { content: string }) => `• ${k.content}`).join('\n')}\n[/IMPORTANT KNOWLEDGE]`
    }
    
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim() + knowledgeContext,
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
          // Also send knowledge to API for storage
          injected_knowledge: injectedKnowledge,
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
      // Mark session as saved after successful response
      setSessionSaved(true)
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
        model={model}
        onModelChange={setModel}
        models={availableModels}
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

      {/* Knowledge Injection Panel */}
      <KnowledgePanel />

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

// ===== Knowledge Injection Panel =====

const KNOWLEDGE_STORAGE_KEY = 'sloughgpt_injected_knowledge'

interface InjectedKnowledge {
  id: string
  content: string
  timestamp: number
}

function KnowledgePanel() {
  const [isOpen, setIsOpen] = useState(false)
  const [knowledge, setKnowledge] = useState<InjectedKnowledge[]>([])
  const [newKnowledge, setNewKnowledge] = useState('')
  const [showAdd, setShowAdd] = useState(false)

  useEffect(() => {
    const stored = localStorage.getItem(KNOWLEDGE_STORAGE_KEY)
    if (stored) {
      try {
        setKnowledge(JSON.parse(stored))
      } catch {
        setKnowledge([])
      }
    }
  }, [])

  const saveKnowledge = (updated: InjectedKnowledge[]) => {
    setKnowledge(updated)
    localStorage.setItem(KNOWLEDGE_STORAGE_KEY, JSON.stringify(updated))
  }

  const addKnowledge = () => {
    if (!newKnowledge.trim()) return
    
    const item: InjectedKnowledge = {
      id: `know_${Date.now()}`,
      content: newKnowledge.trim(),
      timestamp: Date.now(),
    }
    
    saveKnowledge([...knowledge, item])
    setNewKnowledge('')
    setShowAdd(false)
  }

  const removeKnowledge = (id: string) => {
    saveKnowledge(knowledge.filter(k => k.id !== id))
  }

  const clearAll = () => {
    if (confirm('Clear all injected knowledge?')) {
      saveKnowledge([])
    }
  }

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-20 right-4 z-40 p-2 rounded-full bg-muted hover:bg-muted/80 shadow-lg transition-all"
        title="Knowledge Panel"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
        {knowledge.length > 0 && (
          <span className="absolute -top-1 -right-1 w-4 h-4 bg-primary text-primary-foreground text-xs rounded-full flex items-center justify-center">
            {knowledge.length}
          </span>
        )}
      </button>
    )
  }

  return (
    <div className="fixed bottom-20 right-4 z-40 w-80 max-h-96 bg-background border rounded-lg shadow-xl flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b">
        <h3 className="font-medium text-sm">Injected Knowledge</h3>
        <button
          onClick={() => setIsOpen(false)}
          className="text-muted-foreground hover:text-foreground"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {knowledge.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">
            No knowledge injected. Add facts the AI should know.
          </p>
        ) : (
          knowledge.map(item => (
            <div key={item.id} className="p-2 bg-muted/50 rounded text-sm">
              <p className="whitespace-pre-wrap">{item.content}</p>
              <button
                onClick={() => removeKnowledge(item.id)}
                className="text-xs text-muted-foreground hover:text-destructive mt-1"
              >
                Remove
              </button>
            </div>
          ))
        )}
      </div>

      {/* Add Form */}
      <div className="p-2 border-t space-y-2">
        {showAdd ? (
          <>
            <textarea
              className="w-full p-2 text-sm border rounded resize-none h-20"
              placeholder="Enter knowledge to inject..."
              value={newKnowledge}
              onChange={e => setNewKnowledge(e.target.value)}
              autoFocus
            />
            <div className="flex gap-2">
              <button
                onClick={addKnowledge}
                className="flex-1 px-3 py-1.5 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90"
              >
                Add
              </button>
              <button
                onClick={() => { setShowAdd(false); setNewKnowledge('') }}
                className="px-3 py-1.5 text-sm border rounded hover:bg-muted"
              >
                Cancel
              </button>
            </div>
          </>
        ) : (
          <div className="flex gap-2">
            <button
              onClick={() => setShowAdd(true)}
              className="flex-1 px-3 py-1.5 text-sm border rounded hover:bg-muted"
            >
              + Add Knowledge
            </button>
            {knowledge.length > 0 && (
              <button
                onClick={clearAll}
                className="px-3 py-1.5 text-sm text-destructive border rounded hover:bg-destructive/10"
              >
                Clear
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
