'use client'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/cn'

interface ChatSettingsProps {
  isOpen: boolean
  model: string
  temperature: number
  maxTokens: number
  onModelChange: (value: string) => void
  onTemperatureChange: (value: number) => void
  onMaxTokensChange: (value: number) => void
  onClear: () => void
  hasMessages: boolean
}

export function ChatSettings({
  isOpen,
  model,
  temperature,
  maxTokens,
  onModelChange,
  onTemperatureChange,
  onMaxTokensChange,
  onClear,
  hasMessages,
}: ChatSettingsProps) {
  return (
    <section 
      className={cn(
        "shrink-0 border-b border-border/40 bg-muted/20 px-3 py-2 transition-all duration-200 sm:px-4",
        isOpen ? "max-h-40 opacity-100" : "max-h-0 border-transparent p-0 opacity-0 overflow-hidden"
      )}
    >
      <div className="mx-auto flex max-w-2xl flex-wrap items-center gap-x-5 gap-y-2 text-sm">
        <label className="flex items-center gap-2">
          <span className="text-muted-foreground whitespace-nowrap font-medium">Model</span>
          <select
            value={model}
            onChange={(e) => onModelChange(e.target.value)}
            className="rounded-md border border-border/60 bg-background px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary/50 transition-all cursor-pointer hover:border-border"
          >
            <option value="gpt2">gpt2</option>
          </select>
        </label>
        
        <label className="flex items-center gap-2.5">
          <span className="text-muted-foreground whitespace-nowrap font-medium">Temp</span>
          <input
            type="number"
            value={temperature}
            onChange={(e) => onTemperatureChange(Number(e.target.value))}
            step="0.1"
            min="0"
            max="2"
            className="w-16 rounded-lg border border-border/60 bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary/50 transition-all"
          />
        </label>
        
        <label className="flex items-center gap-1.5 sm:gap-2">
          <span className="text-muted-foreground whitespace-nowrap">Max</span>
          <input
            type="number"
            value={maxTokens}
            onChange={(e) => onMaxTokensChange(Number(e.target.value))}
            min="1"
            max="1000"
            className="w-14 rounded-md border bg-background px-2 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-primary/50 sm:w-16"
          />
        </label>
        
        <Button 
          variant="outline" 
          size="sm" 
          onClick={onClear}
          className="text-xs hover:opacity-80 active:opacity-70 disabled:opacity-50"
          disabled={!hasMessages}
        >
          Clear
        </Button>
        
        <span className="text-[10px] text-muted-foreground/50 hidden sm:inline">
          Esc to close
        </span>
      </div>
    </section>
  )
}
