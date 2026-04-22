'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { ModelStatusBar } from '@/components/InferenceStatusBar'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import { cn } from '@/lib/cn'
import { getAgentName, AGENTS } from '@/lib/agents'
import type { AgentId } from '@/lib/agents'

interface ChatHeaderProps {
  health: ApiHealthSnapshot
  showSettings?: boolean
  showSidebar?: boolean
  onToggleSettings?: () => void
  onToggleSidebar?: () => void
  onNewChat?: () => void
  sessionCount?: number
  model?: string
  onModelChange?: (model: string) => void
  models?: string[]
  agent?: string
  onAgentChange?: (agent: string) => void
  temperature?: number
  maxTokens?: number
  onTemperatureChange?: (t: number) => void
  onMaxTokensChange?: (t: number) => void
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

function GearIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.807 2.885 2.165a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.807 3.31-2.165 2.885a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.807-2.885-2.165a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.807-3.31 2.165-2.885a1.724 1.724 0 002.572-1.065z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  )
}

export function ChatHeader({ 
  health, 
  showSettings, 
  showSidebar, 
  onToggleSettings, 
  onToggleSidebar, 
  onNewChat, 
  model, 
  onModelChange, 
  models = [],
  agent = 'general',
  onAgentChange,
  temperature,
  maxTokens,
  onTemperatureChange,
  onMaxTokensChange,
}: ChatHeaderProps) {
  const [showModelList, setShowModelList] = useState(false)
  const [showAgentList, setShowAgentList] = useState(false)
  const [showSettingsMenu, setShowSettingsMenu] = useState(false)
  const settingsMenuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (settingsMenuRef.current && !settingsMenuRef.current.contains(e.target as Node)) {
        setShowSettingsMenu(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

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
        <div className="flex items-center gap-1.5">
          {agent && onAgentChange && (
            <div className="relative">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowAgentList(!showAgentList)}
                className="text-xs sm:text-sm max-w-[80px] sm:max-w-[100px] truncate border border-border/50 hover:bg-secondary/80 hover:border-border"
                title={getAgentName(agent)}
              >
                {getAgentName(agent)}
                <svg className="w-3 h-3 ml-1 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </Button>
              {showAgentList && (
                <div className="absolute right-0 top-full mt-1.5 z-50 bg-background border border-border/60 rounded-lg shadow-xl min-w-[160px] max-h-[240px] overflow-auto p-1">
                  {(Object.keys(AGENTS) as AgentId[]).map((id) => (
                    <button
                      key={id}
                      onClick={() => {
                        onAgentChange(id)
                        setShowAgentList(false)
                      }}
                      className={cn(
                        "w-full text-left px-3 py-2.5 text-sm rounded-md hover:bg-secondary/80 transition-colors",
                        id === agent && "bg-secondary font-medium text-primary"
                      )}
                    >
                      {getAgentName(id)}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          {model && onModelChange && models.length > 0 && (
            <div className="relative">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowModelList(!showModelList)}
                className="text-xs sm:text-sm max-w-[100px] sm:max-w-[150px] truncate border border-border/50 hover:bg-secondary/80 hover:border-border"
                title={model}
              >
                {model}
                <svg className="w-3 h-3 ml-1 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </Button>
              {showModelList && (
                <div className="absolute right-0 top-full mt-1.5 z-50 bg-background border border-border/60 rounded-lg shadow-xl min-w-[140px] max-h-[240px] overflow-auto p-1">
                  {models.map((m) => (
                    <button
                      key={m}
                      onClick={() => {
                        onModelChange(m)
                        setShowModelList(false)
                      }}
                      className={cn(
                        "w-full text-left px-3 py-2.5 text-sm rounded-md hover:bg-secondary/80 transition-colors",
                        m === model && "bg-secondary font-medium text-primary"
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
          <div className="relative" ref={settingsMenuRef}>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowSettingsMenu(!showSettingsMenu)}
              className={cn(
                "p-1.5",
                showSettingsMenu && "bg-secondary"
              )}
              title="Settings"
            >
              <GearIcon className="h-4 w-4" />
            </Button>
            {showSettingsMenu && (
              <div className="absolute right-0 top-full mt-1.5 z-50 bg-background border border-border/60 rounded-lg shadow-xl min-w-[200px] p-1">
                {onTemperatureChange && onMaxTokensChange && (
                  <>
                    <div className="px-3 py-2 text-xs font-medium text-muted-foreground">Settings</div>
                    <div className="px-3 py-2 flex items-center justify-between">
                      <span className="text-sm">Temperature</span>
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => onTemperatureChange(Math.max(0, (temperature || 0.8) - 0.1))}
                          disabled={(temperature || 0.8) <= 0}
                          className="w-6 h-6 flex items-center justify-center rounded border border-border/50 hover:bg-muted disabled:opacity-30 text-xs"
                        >
                          −
                        </button>
                        <span className="w-8 text-center text-muted-foreground text-sm">{(temperature || 0.8).toFixed(1)}</span>
                        <button
                          onClick={() => onTemperatureChange(Math.min(2, (temperature || 0.8) + 0.1))}
                          disabled={(temperature || 0.8) >= 2}
                          className="w-6 h-6 flex items-center justify-center rounded border border-border/50 hover:bg-muted disabled:opacity-30 text-xs"
                        >
                          +
                        </button>
                      </div>
                    </div>
                    <div className="px-3 py-2 flex items-center justify-between">
                      <span className="text-sm">Max tokens</span>
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => onMaxTokensChange(Math.max(1, (maxTokens || 100) - 50))}
                          disabled={(maxTokens || 100) <= 1}
                          className="w-6 h-6 flex items-center justify-center rounded border border-border/50 hover:bg-muted disabled:opacity-30 text-xs"
                        >
                          −
                        </button>
                        <span className="w-8 text-center text-muted-foreground text-sm">{maxTokens || 100}</span>
                        <button
                          onClick={() => onMaxTokensChange(Math.min(4096, (maxTokens || 100) + 50))}
                          disabled={(maxTokens || 100) >= 4096}
                          className="w-6 h-6 flex items-center justify-center rounded border border-border/50 hover:bg-muted disabled:opacity-30 text-xs"
                        >
                          +
                        </button>
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}