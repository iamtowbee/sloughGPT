'use client'

import { useEffect, useMemo, useRef, useState } from 'react'

import { api } from '@/lib/api'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { Textarea } from '@/components/ui/textarea'

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

function ChevronDownIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
    </svg>
  )
}

function SendIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
    </svg>
  )
}

export default function ChatPage() {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [availableModels, setAvailableModels] = useState<Array<{ id: string; name: string; source?: string }>>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const activeSession = sessions.find((s) => s.id === activeSessionId) ?? null
  const messages = useMemo(
    () => activeSession?.messages ?? [],
    [activeSession],
  )
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
        // malformed storage
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

  const selectedModelLabel = availableModels.find((m) => m.id === selectedModel)?.name || selectedModel

  return (
    <div className="flex h-full min-h-0 gap-0 md:gap-2">
      <aside className="flex w-72 shrink-0 flex-col border-r border-border bg-card/40 py-3 pr-3">
        <Button type="button" variant="secondary" className="mb-3 w-full border-dashed" onClick={startNewConversation}>
          + New chat
        </Button>
        <Input
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search chats…"
          className="mb-3 bg-muted/30"
          aria-label="Search chats"
        />
        <div className="flex flex-1 flex-col space-y-1 overflow-y-auto">
          {filteredSessions.map((session) => (
            <div
              key={session.id}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  setActiveSessionId(session.id)
                }
              }}
              className={`group cursor-pointer border px-2 py-2 transition-colors duration-200 ease-smooth ${
                session.id === activeSessionId
                  ? 'border-primary/40 bg-primary/10'
                  : 'border-transparent hover:bg-muted/50'
              }`}
              onClick={() => setActiveSessionId(session.id)}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="truncate text-sm text-foreground">{session.title}</div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-xs opacity-0 transition-opacity group-hover:opacity-100"
                  onClick={(e) => {
                    e.stopPropagation()
                    deleteSession(session.id)
                  }}
                >
                  Delete
                </Button>
              </div>
            </div>
          ))}
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col px-1 md:px-2">
        <div className="flex items-center justify-between border-b border-border py-3">
          <div className="flex flex-wrap items-center gap-2 md:gap-3">
            <h1 className="text-lg font-semibold text-foreground">{activeSession?.title ?? 'Chat'}</h1>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="secondary" size="sm" className="gap-2 font-normal">
                  <span className="max-w-[140px] truncate">{selectedModelLabel}</span>
                  <ChevronDownIcon className="h-3 w-3 shrink-0 opacity-70" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="max-h-64 overflow-y-auto">
                {availableModels.length === 0 && (
                  <div className="px-2 py-2 text-sm text-muted-foreground">Loading models…</div>
                )}
                {availableModels.map((model) => (
                  <DropdownMenuItem
                    key={model.id}
                    onClick={() => updateActiveSession((s) => ({ ...s, selectedModel: model.id }))}
                    className={selectedModel === model.id ? 'bg-primary/10' : ''}
                  >
                    <div>
                      <div className="font-medium">{model.name}</div>
                      <div className="text-xs text-muted-foreground">{model.source || 'local'}</div>
                    </div>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <Button type="button" variant="outline" size="sm" onClick={() => setShowSettings(true)}>
              Generation settings
            </Button>
          </div>
        </div>

        <Dialog open={showSettings} onOpenChange={setShowSettings}>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>Generation settings</DialogTitle>
              <DialogDescription>Applied to this chat only. Uses the HTTP inference API.</DialogDescription>
            </DialogHeader>
            <div className="grid grid-cols-2 gap-4 py-2">
              <div className="space-y-2">
                <Label htmlFor="chat-temp">Temperature</Label>
                <Input
                  id="chat-temp"
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
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="chat-max">Max tokens</Label>
                <Input
                  id="chat-max"
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
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="chat-topp">Top P</Label>
                <Input
                  id="chat-topp"
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
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="chat-topk">Top K</Label>
                <Input
                  id="chat-topk"
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
                />
              </div>
            </div>
            <DialogFooter>
              <Button type="button" onClick={() => setShowSettings(false)}>
                Done
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <div className="flex flex-1 flex-col space-y-3 overflow-y-auto py-4">
          {messages.length === 0 && (
            <div className="flex h-full flex-col items-center justify-center py-20 text-center">
              <div className="mb-4 flex h-16 w-16 items-center justify-center border border-primary/30 bg-primary/10 font-mono text-xl font-semibold text-primary">
                S
              </div>
              <h2 className="mb-2 text-xl font-medium text-foreground">SloughGPT</h2>
              <p className="mb-6 text-sm text-muted-foreground">Start a conversation</p>
              <div className="flex max-w-md flex-wrap justify-center gap-2">
                {['Explain quantum', 'Write code', 'What is ML?', 'Help me create'].map((example, i) => (
                  <Button
                    type="button"
                    key={i}
                    variant="secondary"
                    size="sm"
                    className="text-xs"
                    onClick={() => setInput(example)}
                  >
                    {example}
                  </Button>
                ))}
              </div>
            </div>
          )}
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`group max-w-[85%] border px-4 py-2.5 transition-colors duration-200 ease-smooth ${
                  msg.role === 'user'
                    ? 'border-primary/40 bg-primary text-primary-foreground'
                    : 'border-border bg-muted/40 text-foreground'
                }`}
              >
                <div className="whitespace-pre-wrap text-sm">{msg.content}</div>

                {msg.role === 'assistant' && (
                  <div className="mt-1 flex gap-3 border-t border-border pt-1 opacity-0 transition-opacity duration-200 group-hover:opacity-100">
                    <button
                      type="button"
                      onClick={() => copyToClipboard(msg.content)}
                      className="text-xs text-muted-foreground hover:text-foreground"
                    >
                      Copy
                    </button>
                    <button
                      type="button"
                      onClick={() => retryAssistantMessage(msg.id)}
                      className="text-xs text-muted-foreground hover:text-foreground"
                    >
                      Retry
                    </button>
                  </div>
                )}
                {msg.role === 'user' && (
                  <div className="mt-1 flex gap-3 border-t border-primary-foreground/25 pt-1 opacity-0 transition-opacity duration-200 group-hover:opacity-100">
                    <button
                      type="button"
                      onClick={() => editFromUserMessage(msg.id)}
                      className="text-xs text-primary-foreground/90 hover:text-primary-foreground"
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
              <div className="flex gap-1 border border-border bg-muted/40 px-4 py-3">
                <span
                  className="h-1.5 w-1.5 animate-bounce bg-primary"
                  style={{ animationDelay: '0ms' }}
                />
                <span
                  className="h-1.5 w-1.5 animate-bounce bg-primary"
                  style={{ animationDelay: '150ms' }}
                />
                <span
                  className="h-1.5 w-1.5 animate-bounce bg-primary"
                  style={{ animationDelay: '300ms' }}
                />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <Separator className="mt-1" />

        <div className="pt-3">
          <div className="flex gap-2">
            <Textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey && !e.metaKey) {
                  e.preventDefault()
                  sendMessage()
                }
              }}
              placeholder="Message…"
              rows={2}
              className="min-h-[52px] flex-1 resize-none bg-muted/30"
            />
            <Button
              type="button"
              size="icon"
              className="h-auto shrink-0 self-end"
              onClick={sendMessage}
              disabled={isLoading || !input.trim()}
              aria-label="Send message"
            >
              <SendIcon className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
