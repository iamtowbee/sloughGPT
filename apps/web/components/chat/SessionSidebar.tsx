'use client'

import { useState, useEffect } from 'react'
import { cn } from '@/lib/cn'
import { Button } from '@/components/ui/button'
import type { ChatMessage } from './ChatMessages'

interface ChatSession {
  id: string
  name: string
  messages: ChatMessage[]
  createdAt: string
  updatedAt: string
}

interface SessionSidebarProps {
  isOpen: boolean
  onClose: () => void
  currentSessionId: string
  onLoadSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
  onNewChat: () => void
}

function ChatIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
    </svg>
  )
}

function TrashIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
    </svg>
  )
}

function PlusIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
    </svg>
  )
}

function CloseIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  )
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`
  return date.toLocaleDateString()
}

function SearchIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
    </svg>
  )
}

function DownloadIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
    </svg>
  )
}

function TrashAllIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
    </svg>
  )
}

export function SessionSidebar({
  isOpen,
  onClose,
  currentSessionId,
  onLoadSession,
  onDeleteSession,
  onNewChat,
}: SessionSidebarProps) {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)

  const STORAGE_KEY = 'sloughgpt_chat_sessions'

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      setSessions(JSON.parse(stored))
    }
  }, [currentSessionId])

  const filteredSessions = sessions.filter(s =>
    s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.messages.some(m => m.content.toLowerCase().includes(searchQuery.toLowerCase()))
  )

  const handleDelete = (sessionId: string) => {
    if (deleteConfirm === sessionId) {
      onDeleteSession(sessionId)
      setSessions(prev => prev.filter(s => s.id !== sessionId))
      setDeleteConfirm(null)
    } else {
      setDeleteConfirm(sessionId)
      setTimeout(() => setDeleteConfirm(null), 3000)
    }
  }

  const handleExportChat = (session: ChatSession, format: 'md' | 'json') => {
    let content: string
    let filename: string
    let mimeType: string

    if (format === 'md') {
      content = `# ${session.name}\n\n${session.messages.map(m => {
        const role = m.role === 'user' ? '**User**' : '**Assistant**'
        return `${role}:\n${m.content}\n`
      }).join('\n')}`
      filename = `${session.name.replace(/[^a-z0-9]/gi, '_')}.md`
      mimeType = 'text/markdown'
    } else {
      content = JSON.stringify({ name: session.name, messages: session.messages }, null, 2)
      filename = `${session.name.replace(/[^a-z0-9]/gi, '_')}.json`
      mimeType = 'application/json'
    }

    const blob = new Blob([content], { type: mimeType })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleClearAll = () => {
    if (confirm('Clear all sessions? This cannot be undone.')) {
      localStorage.removeItem(STORAGE_KEY)
      setSessions([])
      onNewChat()
    }
  }

  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/30 z-40 sm:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={cn(
          "fixed top-0 left-0 h-full w-72 bg-background border-r border-border/50 z-50 transform transition-transform duration-200 ease-in-out flex flex-col",
          isOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="flex items-center justify-between p-4 border-b border-border/50">
          <h2 className="font-semibold text-sm">Sessions</h2>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-secondary"
            aria-label="Close sidebar"
          >
            <CloseIcon className="h-4 w-4 text-muted-foreground" />
          </button>
        </div>

        <div className="p-3 space-y-2">
          <Button
            onClick={onNewChat}
            className="w-full justify-start text-xs"
            size="sm"
          >
            <PlusIcon className="h-4 w-4 mr-2" />
            New Chat
          </Button>

          <div className="relative">
            <SearchIcon className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search sessions..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 text-xs rounded-md border border-input bg-background focus:outline-none focus:ring-1 focus:ring-primary/50"
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-2 pb-2">
          {filteredSessions.length === 0 ? (
            <p className="text-xs text-muted-foreground text-center py-8 px-4">
              {searchQuery ? 'No matching sessions' : 'No sessions yet'}
            </p>
          ) : (
            <div className="space-y-1">
              {filteredSessions.map((session) => (
                <div
                  key={session.id}
                  className={cn(
                    "group relative rounded-lg p-2.5 transition-colors cursor-pointer",
                    session.id === currentSessionId
                      ? "bg-secondary"
                      : "hover:bg-secondary/50"
                  )}
                  onClick={() => {
                    if (session.id !== currentSessionId) {
                      onLoadSession(session.id)
                    }
                  }}
                >
                  <div className="flex items-start gap-2">
                    <ChatIcon className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium truncate">{session.name}</p>
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        {session.messages.length} messages · {formatDate(session.updatedAt)}
                      </p>
                    </div>
                  </div>

                  <div className="absolute top-1.5 right-1.5 opacity-0 group-hover:opacity-100 transition-opacity flex gap-0.5">
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        handleExportChat(session, 'md')
                      }}
                      className="p-1 rounded hover:bg-background/80"
                      title="Export markdown"
                    >
                      <DownloadIcon className="h-3 w-3 text-muted-foreground" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        handleDelete(session.id)
                      }}
                      className={cn(
                        "p-1 rounded transition-colors",
                        deleteConfirm === session.id
                          ? "bg-destructive text-destructive-foreground"
                          : "hover:bg-background/80"
                      )}
                      title={deleteConfirm === session.id ? "Click again to confirm" : "Delete"}
                    >
                      <TrashIcon className="h-3 w-3" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {sessions.length > 0 && (
          <div className="p-3 border-t border-border/50">
            <Button
              onClick={handleClearAll}
              variant="outline"
              className="w-full justify-start text-xs text-destructive hover:text-destructive"
              size="sm"
            >
              <TrashAllIcon className="h-4 w-4 mr-2" />
              Clear All Sessions
            </Button>
          </div>
        )}
      </aside>
    </>
  )
}
