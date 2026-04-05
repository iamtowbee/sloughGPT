import { useState } from 'react'
import type { Meta, StoryObj } from '@storybook/react'
import { Button } from '@/components/ui/button'
import { ChatThread } from './chat-thread'
import { CodeSnippet } from './code-snippet'
import { EmptyState } from './empty-state'
import { MessageBubble } from './message-bubble'
import { PromptComposer } from './prompt-composer'
import { TokenMeter } from './token-meter'
import { ToolCallCard } from './tool-call-card'
import { TypingIndicator } from './typing-indicator'

const meta = {
  title: 'AI/ChatShell',
  parameters: {
    layout: 'fullscreen',
  },
} satisfies Meta

export default meta

function ShellDemo() {
  const [value, setValue] = useState('')
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; text: string }[]>([])
  const [busy, setBusy] = useState(false)

  const send = () => {
    const t = value.trim()
    if (!t) return
    setMessages((m) => [...m, { role: 'user', text: t }])
    setValue('')
    setBusy(true)
    window.setTimeout(() => {
      setMessages((m) => [...m, { role: 'assistant', text: 'Acknowledged. (demo reply)' }])
      setBusy(false)
    }, 600)
  }

  return (
    <div className="str-min-h-screen flex flex-col bg-background">
      <header className="str-safe-top str-safe-x flex flex-wrap items-center justify-between gap-2 border-b border-border bg-card/80 px-3 py-3 backdrop-blur-sm sm:px-4">
        <span className="text-sm font-semibold text-foreground">Assistant</span>
        <TokenMeter total={8420} contextLimit={128000} />
      </header>

      <main className="flex min-h-0 flex-1 flex-col">
        {messages.length === 0 ? (
          <EmptyState
            title="New session"
            description="Safe areas and touch targets follow PWA + mobile patterns."
          >
            <Button type="button" variant="secondary" className="str-touch-target w-full" onClick={() => {}}>
              Browse models
            </Button>
          </EmptyState>
        ) : (
          <ChatThread className="flex-1">
            {messages.map((msg, i) => (
              <MessageBubble key={i} role={msg.role}>
                {msg.text}
              </MessageBubble>
            ))}
            {busy ? <TypingIndicator /> : null}
            <ToolCallCard
              name="list_checkpoints"
              argsPreview='{"dir": "data/experiments/run_01"}'
              state="ok"
            />
            <CodeSnippet>{'loss = criterion(logits, labels)'}</CodeSnippet>
          </ChatThread>
        )}
        <PromptComposer
          value={value}
          onValueChange={setValue}
          onSubmit={send}
          busy={busy}
          placeholder="Message…"
        />
      </main>
    </div>
  )
}

export const Desktop: StoryObj = {
  render: () => <ShellDemo />,
}

export const IPhone: StoryObj = {
  ...Desktop,
  parameters: {
    viewport: {
      defaultViewport: 'iphone12',
    },
  },
}
