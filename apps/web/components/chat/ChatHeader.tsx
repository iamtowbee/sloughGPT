'use client'

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

export function ChatHeader({ health, showSettings, showSidebar, onToggleSettings, onToggleSidebar, onNewChat, sessionCount }: ChatHeaderProps) {
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
          <Button
            variant="ghost"
            size="sm"
            onClick={onNewChat}
            className={cn(
              "text-xs sm:text-sm hover:opacity-80 active:opacity-70",
              sessionCount !== undefined && sessionCount > 0 && "hidden sm:flex"
            )}
            title="New chat (Ctrl+N)"
          >
            <PlusIcon className="h-4 w-4 sm:mr-1" />
            <span className="hidden sm:inline">New</span>
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
