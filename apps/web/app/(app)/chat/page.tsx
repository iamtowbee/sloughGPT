'use client'

import { useEffect, useMemo, useRef, useState } from 'react'

import { useApiHealth } from '@/hooks/useApiHealth'
import { api } from '@/lib/api'
import { revealTypingSequence } from '@/lib/chat-reveal'
import { devDebug } from '@/lib/dev-log'
import { AppRouteHeader } from '@/components/AppRouteHeader'
import { InferenceRuntimeToolbar, InferenceStatusBar } from '@/components/InferenceStatusBar'
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
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/cn'
import { MessageBubble } from '@sloughgpt/strui'

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
const CHAT_RAIL_EXPANDED_KEY = 'sloughgpt_chat_rail_expanded_v1'

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

/** Paper-plane outline — avoid trailing stem reading like a warning glyph. */
function SendIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M12 19l9 2-9-18-9 18 9-2z"
      />
    </svg>
  )
}

/** Collapsed rail — open full conversation list (desktop). */
function ConversationsRailIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
      />
    </svg>
  )
}

function ChevronLeftIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
    </svg>
  )
}

/** Leading control in composer (reference: chat apps use + for attachments / actions). */
function PlusIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v14M5 12h14" />
    </svg>
  )
}

/** Generation / sampling controls — icon-only in header (dialog still has full labels). */
function SlidersIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 21v-7M4 10V3M12 21v-9M12 3v3M20 21v-5M20 8V3" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 21h6M15 10h6M3 15h6" />
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
  const [mobileSessionsOpen, setMobileSessionsOpen] = useState(false)
  const [chatRailExpanded, setChatRailExpanded] = useState(false)
  const [availableModels, setAvailableModels] = useState<Array<{ id: string; name: string; source?: string }>>([])
  const { state: apiHealth, refresh: refreshHealth } = useApiHealth()
  const [modelsCatalogError, setModelsCatalogError] = useState(false)
  const [modelsCatalogLoading, setModelsCatalogLoading] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const activeSession = sessions.find((s) => s.id === activeSessionId) ?? null
  const messages = useMemo(
    () => activeSession?.messages ?? [],
    [activeSession],
  )
  const selectedModel = activeSession?.selectedModel ?? 'gpt2'
  const settings = activeSession?.settings ?? defaultSettings

  const canInfer = useMemo(() => {
    if (apiHealth === null) return false
    if (apiHealth === 'offline') return false
    return apiHealth.model_loaded
  }, [apiHealth])

  const sendBlockedReason =
    apiHealth === null
      ? 'Waiting for API status…'
      : apiHealth === 'offline'
        ? 'Cannot reach the API'
        : !apiHealth.model_loaded
          ? 'Load weights in the API (Models page or autoload)'
          : undefined

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
    try {
      if (localStorage.getItem(CHAT_RAIL_EXPANDED_KEY) === '1') {
        setChatRailExpanded(true)
      }
    } catch {
      /* ignore */
    }
  }, [])

  useEffect(() => {
    try {
      localStorage.setItem(CHAT_RAIL_EXPANDED_KEY, chatRailExpanded ? '1' : '0')
    } catch {
      /* ignore */
    }
  }, [chatRailExpanded])

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
      setModelsCatalogLoading(true)
      setModelsCatalogError(false)
      try {
        const models = await api.getModels()
        setAvailableModels(models.map((m) => ({ id: m.id, name: m.name, source: m.type })))
      } catch {
        setAvailableModels([])
        setModelsCatalogError(true)
      } finally {
        setModelsCatalogLoading(false)
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
    setMobileSessionsOpen(false)
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

  const generateForPrompt = async (
    prompt: string,
    options?: {
      skipUserAppend?: boolean
      chatMessagesOverride?: Array<{ role: string; content: string }>
    },
  ) => {
    if (!activeSession) return
    const skipUserAppend = options?.skipUserAppend ?? false

    const chatMessages: Array<{ role: string; content: string }> =
      options?.chatMessagesOverride ??
      (skipUserAppend
        ? activeSession.messages.map((m) => ({ role: m.role, content: m.content }))
        : [...activeSession.messages.map((m) => ({ role: m.role, content: m.content })), { role: 'user', content: prompt }])

    if (!skipUserAppend) {
      const userMessage: Message = {
        id: Date.now().toString(),
        role: 'user',
        content: prompt,
        timestamp: new Date(),
      }
      updateActiveSession((session) => ({ ...session, messages: [...session.messages, userMessage] }))
      upsertSessionTitle(prompt)
      setInput('')
    }

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

    const revealAssistantText = async (fullContent: string, delayMs: number) => {
      for (const partial of revealTypingSequence(fullContent, 3)) {
        updateActiveSession((session) => ({
          ...session,
          messages: session.messages.map((m) =>
            m.id === assistantId ? { ...m, content: partial } : m,
          ),
        }))
        await new Promise((r) => setTimeout(r, delayMs))
      }
    }

    const streamViaChat = (): Promise<boolean> =>
      new Promise((resolve) => {
        let gotToken = false
        api.chatStream(
          {
            messages: chatMessages,
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
      if (await streamViaChat()) {
        setIsLoading(false)
        return
      }
    } catch (err) {
      devDebug('Chat streaming failed, falling back to non-stream generation:', err)
    }

    try {
      const data = await api.chat({
        messages: chatMessages,
        model: selectedModel,
        max_new_tokens: settings.maxNewTokens,
        temperature: settings.temperature,
        top_p: settings.topP,
        top_k: settings.topK,
      })
      const fullContent = data.text || ''
      await revealAssistantText(fullContent, 10)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Generation failed'
      const fullContent = `Could not generate a reply. ${message}\n\nIf the API is running, load a model (Models page or POST /models/load) or ensure the server finished startup autoload (SLOUGHGPT_AUTOLOAD_MODEL, default gpt2).`
      await revealAssistantText(fullContent, 4)
    }

    setIsLoading(false)
  }

  const sendMessage = async () => {
    if (!input.trim() || isLoading || !canInfer) return
    await generateForPrompt(input.trim())
  }

  const retryAssistantMessage = async (assistantMessageId: string) => {
    if (!activeSession || isLoading || !canInfer) return
    const idx = activeSession.messages.findIndex((m) => m.id === assistantMessageId && m.role === 'assistant')
    if (idx <= 0) return
    const prompt = [...activeSession.messages]
      .slice(0, idx)
      .reverse()
      .find((m) => m.role === 'user')?.content
    if (!prompt) return
    const historyForApi = activeSession.messages
      .slice(0, idx)
      .map((m) => ({ role: m.role, content: m.content }))
    updateActiveSession((session) => ({ ...session, messages: session.messages.slice(0, idx) }))
    await generateForPrompt(prompt, { skipUserAppend: true, chatMessagesOverride: historyForApi })
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

  const pickSession = (id: string) => {
    setActiveSessionId(id)
    setMobileSessionsOpen(false)
  }

  const sessionRowClass = (active: boolean) =>
    cn(
      'group relative flex w-full min-h-[2.25rem] cursor-pointer items-center gap-2 border-0 px-3 py-2 text-left text-sm transition-colors duration-200 ease-smooth',
      active
        ? 'bg-primary/[0.11] font-medium text-primary shadow-[inset_3px_0_0_0] shadow-primary dark:bg-primary/[0.09]'
        : 'text-foreground/78 hover:bg-secondary/70 hover:text-foreground dark:text-muted-foreground',
    )

  const renderSessionsPanel = (opts?: { hideTitle?: boolean }) => (
    <nav className="flex min-h-0 flex-1 flex-col" aria-label="Chat history">
      {!opts?.hideTitle && (
        <p className="mb-2 px-3 font-mono text-[10px] uppercase tracking-wider text-foreground/48 dark:text-muted-foreground">
          Conversations
        </p>
      )}
      <button
        type="button"
        onClick={startNewConversation}
        className="mb-2 w-full text-left text-sm text-primary underline-offset-2 transition-colors hover:text-primary/90 hover:underline"
      >
        + New chat
      </button>
      <label className="sr-only" htmlFor="chat-session-search">
        Search chats
      </label>
      <input
        id="chat-session-search"
        type="search"
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        placeholder="Search…"
        className="mb-2 w-full border-0 border-b border-border/25 bg-transparent px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/75 outline-none ring-0 transition-colors focus:border-primary/35"
      />
      <ul className="flex min-h-0 flex-1 flex-col divide-y divide-border/10 overflow-y-auto overscroll-contain dark:divide-border/15">
        {filteredSessions.length === 0 ? (
          <li className="list-none px-3 py-6 text-xs leading-relaxed text-muted-foreground">
            {searchQuery.trim()
              ? 'No chats match this search.'
              : 'No conversations yet — use + New chat above.'}
          </li>
        ) : (
          filteredSessions.map((session) => {
          const active = session.id === activeSessionId
          return (
            <li key={session.id} className="group">
              <div className="flex min-h-[2.25rem] items-stretch">
                <button
                  type="button"
                  className={cn('min-w-0 flex-1 truncate text-left', sessionRowClass(active))}
                  onClick={() => pickSession(session.id)}
                >
                  {session.title}
                </button>
                <button
                  type="button"
                  className="flex shrink-0 items-center px-2 font-mono text-base leading-none text-muted-foreground opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100"
                  title="Remove chat"
                  onClick={(e) => {
                    e.preventDefault()
                    e.stopPropagation()
                    deleteSession(session.id)
                  }}
                >
                  ×
                </button>
              </div>
            </li>
          )
        })
        )}
      </ul>
    </nav>
  )

  const sessionsPanel = renderSessionsPanel()

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden md:flex-row">
      {/* Desktop: minified by default; expand for full conversation list */}
      <aside
        className={cn(
          'sl-chat-rail hidden min-h-0 shrink-0 flex-col overflow-hidden border-r border-border/30 md:flex',
          chatRailExpanded ? 'w-[var(--sidebar-width)] p-2' : 'w-11 items-stretch p-1',
        )}
        aria-label="Conversations"
      >
        {chatRailExpanded ? (
          <div className="flex min-h-0 flex-1 flex-col">
            <div className="mb-1 flex shrink-0 items-center justify-between gap-2 px-1">
              <span className="font-mono text-[10px] uppercase tracking-wider text-foreground/48">
                Conversations
              </span>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-8 w-8 shrink-0"
                aria-label="Minimize conversation list"
                title="Minimize"
                onClick={() => setChatRailExpanded(false)}
              >
                <ChevronLeftIcon className="h-4 w-4" />
              </Button>
            </div>
            {renderSessionsPanel({ hideTitle: true })}
          </div>
        ) : (
          <div className="flex min-h-0 flex-1 flex-col items-center pt-2">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-9 w-9 shrink-0 text-foreground/70 hover:text-foreground"
              aria-label="Open conversation list"
              title="Conversations"
              onClick={() => setChatRailExpanded(true)}
            >
              <ConversationsRailIcon className="h-5 w-5" />
            </Button>
          </div>
        )}
      </aside>

      {/* Main column: toolbar, status, thread, and composer share one max-width column */}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
        <div className="mx-auto flex min-h-0 w-full max-w-[var(--chat-thread-max)] flex-1 flex-col overflow-hidden px-3 sm:px-4 md:px-6">
        <AppRouteHeader
          className="shrink-0 pt-3 pb-1"
          left={
            <>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="shrink-0 md:hidden"
                onClick={() => setMobileSessionsOpen(true)}
                aria-expanded={mobileSessionsOpen}
              >
                Chats
              </Button>
              <h1 className="min-w-0 max-w-[min(100%,14rem)] truncate text-base font-semibold tracking-tight text-foreground sm:max-w-[min(100%,18rem)]">
                {activeSession?.title ?? 'Chat'}
              </h1>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="secondary"
                    size="sm"
                    className="max-w-[min(100%,18ch)] gap-1.5 font-normal"
                    title="Catalog model for this chat (inference uses API runtime on the right)"
                  >
                    <span className="truncate">{selectedModelLabel}</span>
                    <ChevronDownIcon className="h-3 w-3 shrink-0 opacity-70" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="max-h-64 overflow-y-auto">
                  {modelsCatalogLoading && (
                    <div className="px-2 py-2 text-sm text-muted-foreground">Loading catalog…</div>
                  )}
                  {!modelsCatalogLoading && modelsCatalogError && (
                    <div className="px-2 py-2 text-sm text-destructive">Could not list models (API error).</div>
                  )}
                  {!modelsCatalogLoading && !modelsCatalogError && availableModels.length === 0 && (
                    <div className="px-2 py-2 text-sm text-muted-foreground">No models in catalog.</div>
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
              <Button
                type="button"
                variant="outline"
                size="icon"
                className="h-8 w-8 shrink-0"
                onClick={() => setShowSettings(true)}
                title="Generation settings"
                aria-label="Generation settings"
              >
                <SlidersIcon className="h-4 w-4" />
              </Button>
            </>
          }
          right={<InferenceRuntimeToolbar health={apiHealth} onRefresh={refreshHealth} />}
        />

        <div className="shrink-0 pb-2">
          <InferenceStatusBar health={apiHealth} selectedCatalogId={selectedModel} />
        </div>

        <Dialog open={mobileSessionsOpen} onOpenChange={setMobileSessionsOpen}>
          <DialogContent className="flex max-h-[min(90dvh,32rem)] flex-col gap-0 overflow-hidden border-border/80 p-0 sm:max-w-md">
            <DialogHeader className="shrink-0 border-b border-border/70 px-4 py-3 text-left">
              <DialogTitle className="font-mono text-[10px] uppercase tracking-wider text-foreground/55">
                Conversations
              </DialogTitle>
              <DialogDescription className="text-xs text-muted-foreground">
                Search, switch, or start a new chat.
              </DialogDescription>
            </DialogHeader>
            <div className="flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain px-2 py-3">
              {sessionsPanel}
            </div>
          </DialogContent>
        </Dialog>

        <Dialog open={showSettings} onOpenChange={setShowSettings}>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>Generation settings</DialogTitle>
              <DialogDescription>
                Applied to this chat only. Generation calls the API; the loaded weights are those reported on the chat header (not only this dropdown).
              </DialogDescription>
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

        <div className="sl-chat-thread flex min-h-0 flex-1 flex-col gap-0 overflow-y-auto overscroll-contain px-3 pt-3 pb-4 sm:px-5 sm:pt-4">
            {messages.length === 0 && (
              <div className="flex min-h-[min(50dvh,28rem)] flex-col items-center justify-center px-2 py-8 text-center sm:py-16">
                <div className="mb-5 flex aspect-square w-[4.5rem] shrink-0 items-center justify-center border border-primary/35 bg-gradient-to-br from-primary/12 to-accent/20 font-mono text-2xl font-semibold text-primary shadow-[inset_0_1px_0_rgba(255,255,255,0.35)] dark:shadow-none">
                  S
                </div>
                <h2 className="mb-1.5 text-xl font-semibold tracking-tight text-foreground">SloughGPT</h2>
                <p className="mb-8 max-w-sm text-sm leading-relaxed text-muted-foreground">
                  Start a conversation — prompts use the loaded model shown in the status bar.
                </p>
                <div className="flex max-w-lg flex-wrap justify-center gap-2">
                  {['Explain quantum', 'Write code', 'What is ML?', 'Help me create'].map((example, i) => (
                    <Button
                      type="button"
                      key={i}
                      variant="secondary"
                      size="sm"
                      className="text-xs"
                      disabled={!canInfer}
                      title={!canInfer ? sendBlockedReason : undefined}
                      onClick={() => setInput(example)}
                    >
                      {example}
                    </Button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((msg) =>
              msg.role === 'assistant' ? (
                <div key={msg.id} className="mb-4 flex w-full justify-start sm:mb-5">
                  <MessageBubble role="assistant" variant="transcript" className="group">
                    <div className="whitespace-pre-wrap break-words">{msg.content}</div>
                    <div className="mt-2 flex gap-3 border-t border-border/60 pt-2 opacity-0 transition-opacity duration-200 group-hover:opacity-100">
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
                  </MessageBubble>
                </div>
              ) : (
                <div key={msg.id} className="mb-4 flex w-full justify-end sm:mb-5">
                  <div className="group max-w-[min(100%,var(--chat-thread-max))] border border-primary/35 bg-primary px-4 py-2.5 text-primary-foreground shadow-sm transition-colors duration-200 ease-smooth">
                    <div className="whitespace-pre-wrap break-words text-sm leading-relaxed">{msg.content}</div>
                    <div className="mt-1.5 flex gap-3 border-t border-primary-foreground/25 pt-1.5 opacity-0 transition-opacity duration-200 group-hover:opacity-100">
                      <button
                        type="button"
                        onClick={() => editFromUserMessage(msg.id)}
                        className="text-xs text-primary-foreground/90 hover:text-primary-foreground"
                      >
                        Edit & resend
                      </button>
                    </div>
                  </div>
                </div>
              ),
            )}
            {isLoading && (
              <div className="mb-4 flex w-full justify-start sm:mb-5" aria-busy>
                <div className="flex gap-1.5 py-3 text-muted-foreground" aria-label="Assistant is responding">
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary/80 [animation-delay:0s]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary/80 [animation-delay:0.15s]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary/80 [animation-delay:0.3s]" />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="shrink-0 border-t border-border/80 bg-background/90 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-3 backdrop-blur-md supports-[backdrop-filter]:bg-background/75">
            {/* Composer shell — pill well + lead action + send (see docs/design/references/chat-composer-reference-*.png) */}
            <div className="rounded-2xl border border-border/90 bg-card/95 p-2 shadow-sm ring-1 ring-border/15 dark:bg-card/90 dark:ring-border/25">
              <div className="flex items-end gap-1.5 sm:gap-2">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-11 w-11 shrink-0 rounded-full text-muted-foreground hover:text-foreground"
                  aria-label="Composer actions"
                  title="Attachments and shortcuts — coming soon"
                  disabled
                >
                  <PlusIcon className="h-5 w-5" />
                </Button>
                <Textarea
                  data-testid="chat-message-input"
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
                  title={!canInfer ? sendBlockedReason : undefined}
                  className="min-h-[2.75rem] flex-1 resize-none border-0 bg-transparent px-1 py-2.5 shadow-none focus-visible:ring-0 focus-visible:ring-offset-0"
                />
                <Button
                  type="button"
                  data-testid="chat-send-button"
                  size="icon"
                  className="h-11 w-11 shrink-0 rounded-full"
                  onClick={sendMessage}
                  disabled={isLoading || !input.trim() || !canInfer}
                  title={!canInfer ? sendBlockedReason : undefined}
                  aria-label="Send message"
                >
                  <SendIcon className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <p className="mt-2.5 text-center text-[11px] leading-relaxed text-muted-foreground/85">
              SloughGPT calls your API; outputs may be wrong. Verify important answers.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
