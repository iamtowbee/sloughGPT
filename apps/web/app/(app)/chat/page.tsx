'use client'

import { useEffect, useState, useRef } from 'react'
import { useApiHealth } from '@/hooks/useApiHealth'
import { api } from '@/lib/api'
import { AppRouteHeader } from '@/components/AppRouteHeader'
import { Button } from '@/components/ui/button'
import { InferenceRuntimeToolbar } from '@/components/InferenceStatusBar'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [model, setModel] = useState('gpt2')
  const [temp, setTemp] = useState(0.8)
  const [maxTokens, setMaxTokens] = useState(200)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { state: health } = useApiHealth()

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async () => {
    if (!input.trim() || loading) return
    
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    }
    
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const response = await api.chat({
        messages: [...messages, userMessage].map(m => ({ role: m.role, content: m.content })),
        model,
        max_new_tokens: maxTokens,
        temperature: temp,
      })
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.text || 'No response',
      }
      setMessages(prev => [...prev, assistantMessage])
    } catch (err) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error: ${err instanceof Error ? err.message : 'Unknown error'}`,
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const clearChat = () => setMessages([])

  return (
    <div className="flex h-full flex-col">
      <AppRouteHeader
        left={
          <div className="flex items-center gap-2">
            <span className="font-medium text-sm">Chat</span>
          </div>
        }
        right={
          <div className="flex items-center gap-2">
            <InferenceRuntimeToolbar health={health} onRefresh={() => {}} />
            <Button variant="ghost" size="sm" onClick={() => setShowSettings(!showSettings)}>
              Settings
            </Button>
          </div>
        }
      />

      {showSettings && (
        <div className="border-b border-border/50 px-4 py-3 bg-muted/30">
          <div className="flex flex-wrap gap-4 items-center text-sm">
            <label className="flex items-center gap-2">
              <span className="text-muted-foreground">Model:</span>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="rounded border bg-background px-2 py-1 text-xs"
              >
                <option value="gpt2">gpt2</option>
              </select>
            </label>
            <label className="flex items-center gap-2">
              <span className="text-muted-foreground">Temp:</span>
              <input
                type="number"
                value={temp}
                onChange={(e) => setTemp(Number(e.target.value))}
                step="0.1"
                min="0"
                max="2"
                className="w-16 rounded border bg-background px-2 py-1 text-xs"
              />
            </label>
            <label className="flex items-center gap-2">
              <span className="text-muted-foreground">Max:</span>
              <input
                type="number"
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                min="1"
                max="1000"
                className="w-16 rounded border bg-background px-2 py-1 text-xs"
              />
            </label>
            <Button variant="ghost" size="sm" onClick={clearChat}>
              Clear
            </Button>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto flex flex-col justify-end">
        <div className="mx-auto w-full max-w-2xl px-4 py-4">
        {messages.length === 0 && (
          <div className="flex h-full items-center justify-center text-muted-foreground text-sm">
            Start a conversation
          </div>
        )}
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`mb-4 flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm ${
                msg.role === 'user'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-foreground'
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="mb-4 flex justify-start">
            <div className="rounded-2xl bg-muted px-4 py-2.5 text-sm text-muted-foreground">
              Thinking...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="border-t border-border/50 px-4 py-3">
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                sendMessage()
              }
            }}
            placeholder="Type a message..."
            className="flex-1 resize-none rounded-xl border border-input bg-background px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
            rows={1}
          />
          <Button onClick={sendMessage} disabled={loading || !input.trim()}>
            Send
          </Button>
        </div>
      </div>
    </div>
  )
}
