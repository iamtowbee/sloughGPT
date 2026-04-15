'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { ModelStatusBar } from '@/components/InferenceStatusBar'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import { cn } from '@/lib/cn'

interface ChatHeaderProps {
  health: ApiHealthSnapshot
  showSettings: boolean
  showSidebar: boolean
  onToggleSettings: () => void
  onToggleSidebar: () => void
  onNewChat: () => void
  sessionCount?: number
  model?: string
  onModelChange?: (model: string) => void
  models?: string[]
}

function PlusIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
    </svg>
  )
}

function MenuIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
    </svg>
  )
}

export function ChatHeader({ health, showSettings, showSidebar, onToggleSettings, onToggleSidebar, onNewChat, model, onModelChange, models = [] }: ChatHeaderProps) {
  const [showModelList, setShowModelList] = useState(false)

  return (
    <header className="shrink-0 border-b border-border/50 px-3 py-2.5 sm:px-4 sm:py-3">
      <div className="mx-auto flex max-w-2xl items-center justify-between">
        <div className="flex items-center gap-2 sm:gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleSidebar}
            className={cn(
              "p-1.5",
              showSidebar && "bg-secondary"
            )}
            title="Toggle sidebar"
          >
            <MenuIcon className="h-4 w-4" />
          </Button>
          <span className="text-sm font-semibold text-foreground sm:text-base">Chat</span>
          <ModelStatusBar health={health} />
        </div>
        <div className="flex items-center gap-1">
          {model && onModelChange && models.length > 0 && (
            <div className="relative">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowModelList(!showModelList)}
                className="text-xs sm:text-sm max-w-[100px] sm:max-w-[150px] truncate"
                title={model}
              >
                {model}
                <svg className="w-3 h-3 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </Button>
              {showModelList && (
                <div className="absolute right-0 top-full mt-1 z-50 bg-background border border-border rounded-md shadow-lg min-w-[120px] max-h-[200px] overflow-auto">
                  {models.map((m) => (
                    <button
                      key={m}
                      onClick={() => {
                        onModelChange(m)
                        setShowModelList(false)
                      }}
                      className={cn(
                        "w-full text-left px-3 py-2 text-sm hover:bg-muted transition-colors",
                        m === model && "bg-muted font-medium"
                      )}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={onNewChat}
            className="p-1.5"
            title="New chat (Ctrl+N)"
          >
            <PlusIcon className="h-4 w-4" />
          </Button>
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={onToggleSettings}
            className="text-xs sm:text-sm"
          >
            {showSettings ? 'Hide' : 'Settings'}
          </Button>
        </div>
      </div>
    </header>
  )
}
