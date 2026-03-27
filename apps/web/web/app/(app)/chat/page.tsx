'use client'

import { useEffect, useRef, useState } from 'react'
import { useTheme } from '@/components/ThemeProvider'
import { api } from '@/lib/api'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

interface ChatSettings {
  temperature: number
  maxNewTokens: number
  topP: number
  topK: number
}

interface ChatSession {
  id: string
  title: string
  messages: Message[]
  selectedModel: string
  settings: ChatSettings
  updatedAt: string
}

const CHAT_SESSIONS_KEY = 'sloughgpt_chat_sessions_v1'
const ACTIVE_CHAT_KEY = 'sloughgpt_active_chat_v1'

const defaultSettings: ChatSettings = {
  temperature: 0.8,
  maxNewTokens: 200,
  topP: 0.9,
  topK: 50,
}

const createSession = (): ChatSession => {
  const id = Date.now().toString()
  return {
    id,
    title: 'New chat',
    messages: [],
    selectedModel: 'gpt2',
    settings: defaultSettings,
    updatedAt: new Date().toISOString(),
  }
}

export default function ChatPage() {
  const { theme } = useTheme()
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [showModelSelector, setShowModelSelector] = useState(false)
  const [availableModels, setAvailableModels] = useState<Array<{ id: string; name: string; source?: string }>>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const activeSession = sessions.find((s) => s.id === activeSessionId) ?? null
  const messages = activeSession?.messages ?? []
  const selectedModel = activeSession?.selectedModel ?? 'gpt2'
  const settings = activeSession?.settings ?? defaultSettings

  const updateActiveSession = (updater: (session: ChatSession) => ChatSession) => {
    if (!activeSessionId) return
    setSessions((prev) =>
      prev.map((session) =>
        session.id === activeSessionId
          ? { ...updater(session), updatedAt: new Date().toISOString() }
          : session,
      ),
    )
  }

  const upsertSessionTitle = (prompt: string) => {
    updateActiveSession((session) => {
      if (session.title !== 'New chat') return session
      return { ...session, title: prompt.slice(0, 40) || 'New chat' }
    })
  }

  useEffect(() => {
    const rawSessions = localStorage.getItem(CHAT_SESSIONS_KEY)
    const rawActive = localStorage.getItem(ACTIVE_CHAT_KEY)
    if (rawSessions) {
      try {
        const parsed = JSON.parse(rawSessions) as Array<
          Omit<ChatSession, 'messages'> & {
            messages: Array<Omit<Message, 'timestamp'> & { timestamp: string }>
          }
        >
        const hydrated = parsed.map((session) => ({
          ...session,
          settings: session.settings ?? defaultSettings,
          messages: session.messages.map((m) => ({ ...m, timestamp: new Date(m.timestamp) })),
        }))
        if (hydrated.length > 0) {
          setSessions(hydrated)
          if (rawActive && hydrated.some((s) => s.id === rawActive)) {
            setActiveSessionId(rawActive)
          } else {
            setActiveSessionId(hydrated[0].id)
          }
          return
        }
      } catch {
        // Fallback to clean state when storage is malformed.
      }
    }
    const session = createSession()
    setSessions([session])
    setActiveSessionId(session.id)
  }, [])

  useEffect(() => {
    ;(async () => {
      try {
        const models = await api.getModels()
        setAvailableModels(models.map((m) => ({ id: m.id, name: m.name, source: m.type })))
      } catch {
        setAvailableModels([{ id: 'gpt2', name: 'GPT-2', source: 'huggingface' }])
      }
    })()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (sessions.length === 0) return
    localStorage.setItem(CHAT_SESSIONS_KEY, JSON.stringify(sessions))
  }, [sessions])

  useEffect(() => {
    if (!activeSessionId) return
    localStorage.setItem(ACTIVE_CHAT_KEY, activeSessionId)
  }, [activeSessionId])

  const startNewConversation = () => {
    const session = createSession()
    setSessions((prev) => [session, ...prev])
    setActiveSessionId(session.id)
    setShowModelSelector(false)
  }

  const deleteSession = (id: string) => {
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id)
      if (next.length === 0) {
        const fresh = createSession()
        setActiveSessionId(fresh.id)
        return [fresh]
      }
      if (id === activeSessionId) {
        setActiveSessionId(next[0].id)
      }
      return next
    })
  }

  const generateForPrompt = async (prompt: string) => {
    if (!activeSession) return
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: prompt,
      timestamp: new Date(),
    }
    updateActiveSession((session) => ({ ...session, messages: [...session.messages, userMessage] }))
    upsertSessionTitle(prompt)
    setInput('')
    setIsLoading(true)

    const assistantId = (Date.now() + 1).toString()
    const assistantMessage: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    }
    updateActiveSession((session) => ({ ...session, messages: [...session.messages, assistantMessage] }))

    const appendAssistantToken = (token: string) => {
      updateActiveSession((session) => ({
        ...session,
        messages: session.messages.map((m) => (m.id === assistantId ? { ...m, content: m.content + token } : m)),
      }))
    }

    const streamViaInference = (): Promise<boolean> =>
      new Promise((resolve) => {
        let gotToken = false
        api.generateStream(
          {
            prompt,
            model: selectedModel,
            max_new_tokens: settings.maxNewTokens,
            temperature: settings.temperature,
            top_p: settings.topP,
            top_k: settings.topK,
          },
          (token) => {
            gotToken = true
            appendAssistantToken(token)
          },
          () => resolve(gotToken),
        )
      })

    try {
      if (await streamViaInference()) {
        setIsLoading(false)
        return
      }
    } catch (err) {
      console.log('Inference streaming failed, falling back to non-stream generation:', err)
    }

    try {
      const data = await api.generate({
        prompt,
        model: selectedModel,
        max_new_tokens: settings.maxNewTokens,
        temperature: settings.temperature,
        top_p: settings.topP,
        top_k: settings.topK,
      })
      const fullContent = data.text || ''
      for (let i = 0; i <= fullContent.length; i += 3) {
        updateActiveSession((session) => ({
          ...session,
          messages: session.messages.map((m) => (m.id === assistantId ? { ...m, content: fullContent.slice(0, i) } : m)),
        }))
        await new Promise((r) => setTimeout(r, 10))
      }
    } catch (err) {
      console.log('Generation failed, using demo response:', err)
      const demoResponses = [
        `Hey! I'm SloughGPT. Start the API server for real AI responses.`,
        `Interesting question! Let me think about that.\n\nThis is a demo response.`,
        `Got it! I can help with:\n\n- Writing code\n- Answering questions\n- Creative writing\n- And more!`,
      ]
      const fullContent = demoResponses[Math.floor(Math.random() * demoResponses.length)]
      for (let i = 0; i <= fullContent.length; i += 3) {
        updateActiveSession((session) => ({
          ...session,
          messages: session.messages.map((m) => (m.id === assistantId ? { ...m, content: fullContent.slice(0, i) } : m)),
        }))
        await new Promise((r) => setTimeout(r, 15))
      }
    }

    setIsLoading(false)
  }

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return
    await generateForPrompt(input.trim())
  }

  const retryAssistantMessage = async (assistantMessageId: string) => {
    if (!activeSession || isLoading) return
    const idx = activeSession.messages.findIndex((m) => m.id === assistantMessageId && m.role === 'assistant')
    if (idx <= 0) return
    const prompt = [...activeSession.messages]
      .slice(0, idx)
      .reverse()
      .find((m) => m.role === 'user')?.content
    if (!prompt) return
    updateActiveSession((session) => ({ ...session, messages: session.messages.slice(0, idx) }))
    await generateForPrompt(prompt)
  }

  const editFromUserMessage = (userMessageId: string) => {
    if (!activeSession || isLoading) return
    const idx = activeSession.messages.findIndex((m) => m.id === userMessageId && m.role === 'user')
    if (idx < 0) return
    const msg = activeSession.messages[idx]
    setInput(msg.content)
    updateActiveSession((session) => ({ ...session, messages: session.messages.slice(0, idx) }))
    inputRef.current?.focus()
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const filteredSessions = sessions.filter((s) =>
    s.title.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  const themeColors: Record<string, string> = {
    blue: 'from-blue-500 to-cyan-400',
    purple: 'from-purple-500 to-pink-400',
    pink: 'from-pink-500 to-rose-400',
    red: 'from-red-500 to-orange-400',
    orange: 'from-orange-500 to-yellow-400',
    green: 'from-green-500 to-emerald-400',
    teal: 'from-teal-500 to-cyan-400',
  }

  return (
    <div className="h-screen flex gap-4">
      <aside className="w-72 border-r border-white/5 pr-3 py-3 flex flex-col">
        <button
          onClick={startNewConversation}
          className="w-full mb-3 px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-sm text-zinc-200"
        >
          + New chat
        </button>
        <input
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search chats..."
          className="mb-3 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-500 focus:outline-none"
        />
        <div className="flex-1 overflow-y-auto space-y-1">
          {filteredSessions.map((session) => (
            <div
              key={session.id}
              className={`group rounded-lg px-2 py-2 cursor-pointer ${
                session.id === activeSessionId ? 'bg-white/10' : 'hover:bg-white/5'
              }`}
              onClick={() => setActiveSessionId(session.id)}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="text-sm text-zinc-200 truncate">{session.title}</div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    deleteSession(session.id)
                  }}
                  className="opacity-0 group-hover:opacity-100 text-zinc-500 hover:text-zinc-300 text-xs"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      </aside>

      <div className="flex-1 flex flex-col">
        <div className="flex items-center justify-between py-3 border-b border-white/5">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold text-white">{activeSession?.title ?? 'Chat'}</h1>
            <div className="relative">
              <button
                onClick={() => setShowModelSelector(!showModelSelector)}
                className="flex items-center gap-2 px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-zinc-300"
              >
                <span>{availableModels.find((m) => m.id === selectedModel)?.name || selectedModel}</span>
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {showModelSelector && (
                <div className="absolute top-full left-0 mt-1 bg-[#1a1a2e] border border-white/10 rounded-lg shadow-xl py-1 z-50 min-w-[180px] max-h-64 overflow-y-auto">
                  {availableModels.length === 0 && (
                    <div className="px-3 py-2 text-sm text-zinc-500">Loading models...</div>
                  )}
                  {availableModels.map((model) => (
                    <button
                      key={model.id}
                      onClick={() => {
                        updateActiveSession((session) => ({ ...session, selectedModel: model.id }))
                        setShowModelSelector(false)
                      }}
                      className={`w-full text-left px-3 py-2 text-sm ${
                        selectedModel === model.id ? 'text-white bg-white/10' : 'text-zinc-400 hover:bg-white/5'
                      }`}
                    >
                      <div>{model.name}</div>
                      <div className="text-xs text-zinc-500">{model.source || 'local'}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>
            <button
              onClick={() => setShowSettings((v) => !v)}
              className="px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-zinc-300"
            >
              Settings
            </button>
          </div>
        </div>

        {showSettings && (
          <div className="grid grid-cols-2 gap-3 py-3 border-b border-white/5">
            <label className="text-xs text-zinc-400">
              Temperature
              <input
                type="number"
                step="0.1"
                min="0"
                max="2"
                value={settings.temperature}
                onChange={(e) =>
                  updateActiveSession((session) => ({
                    ...session,
                    settings: {
                      ...session.settings,
                      temperature: Number(e.target.value) || defaultSettings.temperature,
                    },
                  }))
                }
                className="mt-1 w-full bg-white/5 border border-white/10 rounded px-2 py-1 text-sm text-white"
              />
            </label>
            <label className="text-xs text-zinc-400">
              Max tokens
              <input
                type="number"
                min="1"
                value={settings.maxNewTokens}
                onChange={(e) =>
                  updateActiveSession((session) => ({
                    ...session,
                    settings: {
                      ...session.settings,
                      maxNewTokens: Number(e.target.value) || defaultSettings.maxNewTokens,
                    },
                  }))
                }
                className="mt-1 w-full bg-white/5 border border-white/10 rounded px-2 py-1 text-sm text-white"
              />
            </label>
            <label className="text-xs text-zinc-400">
              Top P
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={settings.topP}
                onChange={(e) =>
                  updateActiveSession((session) => ({
                    ...session,
                    settings: {
                      ...session.settings,
                      topP: Number(e.target.value) || defaultSettings.topP,
                    },
                  }))
                }
                className="mt-1 w-full bg-white/5 border border-white/10 rounded px-2 py-1 text-sm text-white"
              />
            </label>
            <label className="text-xs text-zinc-400">
              Top K
              <input
                type="number"
                min="1"
                value={settings.topK}
                onChange={(e) =>
                  updateActiveSession((session) => ({
                    ...session,
                    settings: {
                      ...session.settings,
                      topK: Number(e.target.value) || defaultSettings.topK,
                    },
                  }))
                }
                className="mt-1 w-full bg-white/5 border border-white/10 rounded px-2 py-1 text-sm text-white"
              />
            </label>
          </div>
        )}

        <div className="flex-1 overflow-y-auto py-4 space-y-3">
          {messages.length === 0 && (
            <div className="flex-1 flex flex-col items-center justify-center h-full text-center py-20">
              <div
                className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${themeColors[theme]} flex items-center justify-center text-3xl mb-4`}
              >
                ✨
              </div>
              <h2 className="text-xl font-medium text-white mb-2">SloughGPT</h2>
              <p className="text-zinc-500 text-sm mb-6">Start a conversation</p>
              <div className="flex flex-wrap justify-center gap-2 max-w-md">
                {['Explain quantum', 'Write code', 'What is ML?', 'Help me create'].map((example, i) => (
                  <button
                    key={i}
                    onClick={() => setInput(example)}
                    className="px-3 py-1.5 bg-white/5 hover:bg-white/10 border border-white/5 hover:border-white/10 text-zinc-400 rounded-full text-xs transition-all"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`group max-w-[85%] rounded-2xl px-4 py-2.5 ${
                  msg.role === 'user'
                    ? `bg-gradient-to-r ${themeColors[theme]} text-white`
                    : 'bg-white/5 text-zinc-100'
                }`}
              >
                <div className="text-sm whitespace-pre-wrap">{msg.content}</div>

                {msg.role === 'assistant' && (
                  <div className="flex gap-3 mt-1 pt-1 border-t border-white/10 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => copyToClipboard(msg.content)}
                      className="text-xs text-zinc-500 hover:text-zinc-300"
                    >
                      Copy
                    </button>
                    <button
                      onClick={() => retryAssistantMessage(msg.id)}
                      className="text-xs text-zinc-500 hover:text-zinc-300"
                    >
                      Retry
                    </button>
                  </div>
                )}
                {msg.role === 'user' && (
                  <div className="flex gap-3 mt-1 pt-1 border-t border-white/10 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => editFromUserMessage(msg.id)}
                      className="text-xs text-zinc-100/80 hover:text-white"
                    >
                      Edit & resend
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white/5 rounded-2xl px-4 py-2.5">
                <div className="flex gap-1">
                  <span className="w-1.5 h-1.5 bg-zinc-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-1.5 h-1.5 bg-zinc-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-1.5 h-1.5 bg-zinc-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="border-t border-white/5 pt-3">
          <div className="flex gap-2">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey && !e.metaKey) {
                  e.preventDefault()
                  sendMessage()
                }
              }}
              placeholder="Message..."
              rows={1}
              className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-1 focus:ring-[var(--primary)] resize-none"
            />
            <button
              onClick={sendMessage}
              disabled={isLoading || !input.trim()}
              className={`p-2.5 bg-gradient-to-r ${themeColors[theme]} hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
