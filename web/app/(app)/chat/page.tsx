'use client'

import { useState, useEffect, useRef } from 'react'
import { useTheme } from '@/components/ThemeProvider'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

interface ModelInfo {
  id: string
  name: string
  source?: string
}

export default function ChatPage() {
  const { theme } = useTheme()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [selectedModel, setSelectedModel] = useState('gpt2')
  const [showModelSelector, setShowModelSelector] = useState(false)
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const saved = localStorage.getItem('sloughgpt_messages')
    if (saved) {
      try {
        const msgs = JSON.parse(saved)
        setMessages(msgs.map((m: any) => ({ ...m, timestamp: new Date(m.timestamp) })))
      } catch {}
    }
  }, [])

  useEffect(() => {
    fetchModels()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('sloughgpt_messages', JSON.stringify(messages))
    }
  }, [messages])

  const fetchModels = async () => {
    try {
      const res = await fetch('http://localhost:8000/models')
      if (res.ok) {
        const data = await res.json()
        setAvailableModels(data.models || [])
      }
    } catch {
      setAvailableModels([{ id: 'gpt2', name: 'GPT-2', source: 'huggingface' }])
    }
  }

  const startNewConversation = () => {
    setMessages([])
    localStorage.removeItem('sloughgpt_messages')
  }

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return
    
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    }
    
    setMessages(prev => [...prev, userMessage])
    const prompt = input
    setInput('')
    setIsLoading(true)
    
    const assistantId = (Date.now() + 1).toString()
    const assistantMessage: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    }
    setMessages(prev => [...prev, assistantMessage])
    
    // Try streaming endpoint first, fall back to non-streaming
    try {
      const response = await fetch('http://localhost:8000/generate/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          max_new_tokens: 200,
          temperature: 0.8,
          top_p: 0.9,
        })
      })
      
      if (response.ok && response.body) {
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          
          const chunk = decoder.decode(value)
          const lines = chunk.split('\n').filter(l => l.startsWith('data: '))
          
          for (const line of lines) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.token) {
                setMessages(prev => prev.map(m => 
                  m.id === assistantId 
                    ? { ...m, content: m.content + data.token } 
                    : m
                ))
              }
              if (data.done) break
            } catch {}
          }
        }
      } else {
        throw new Error(`HTTP ${response.status}`)
      }
    } catch (err) {
      console.log('Streaming failed, trying non-streaming:', err)
      
      try {
        const response = await fetch('http://localhost:8000/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt,
            max_new_tokens: 200,
            temperature: 0.8,
            top_p: 0.9,
          })
        })
        
        if (response.ok) {
          const data = await response.json()
          const fullContent = data.text || data.generated_text || ''
          // Stream it locally
          for (let i = 0; i <= fullContent.length; i += 3) {
            setMessages(prev => prev.map(m => 
              m.id === assistantId ? { ...m, content: fullContent.slice(0, i) } : m
            ))
            await new Promise(r => setTimeout(r, 10))
          }
        } else {
          throw new Error(`HTTP ${response.status}`)
        }
      } catch {
        const demoResponses = [
          `Hey! I'm SloughGPT. Start the API server for real AI responses.`,
          `Interesting question! Let me think about that...\n\nThis is a demo response.`,
          `Got it! I can help with:\n\n• Writing code\n• Answering questions\n• Creative writing\n• And more!`,
        ]
        const fullContent = demoResponses[Math.floor(Math.random() * demoResponses.length)]
        for (let i = 0; i <= fullContent.length; i += 3) {
          setMessages(prev => prev.map(m => 
            m.id === assistantId ? { ...m, content: fullContent.slice(0, i) } : m
          ))
          await new Promise(r => setTimeout(r, 15))
        }
      }
    }
    
    setIsLoading(false)
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

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
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between py-3 border-b border-white/5">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold text-white">Chat</h1>
          <div className="relative">
            <button
              onClick={() => setShowModelSelector(!showModelSelector)}
              className="flex items-center gap-2 px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-zinc-300"
            >
              <span>{availableModels.find(m => m.id === selectedModel)?.name || selectedModel}</span>
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {showModelSelector && (
              <div className="absolute top-full left-0 mt-1 bg-[#1a1a2e] border border-white/10 rounded-lg shadow-xl py-1 z-50 min-w-[180px] max-h-64 overflow-y-auto">
                {availableModels.length === 0 && (
                  <div className="px-3 py-2 text-sm text-zinc-500">Loading models...</div>
                )}
                {availableModels.map(model => (
                  <button
                    key={model.id}
                    onClick={() => { setSelectedModel(model.id); setShowModelSelector(false) }}
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
        </div>
        <button
          onClick={startNewConversation}
          className="p-2 hover:bg-white/5 rounded-lg text-zinc-400"
          title="New chat"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
        </button>
      </div>
      
      {/* Messages */}
      <div className="flex-1 overflow-y-auto py-4 space-y-3">
        {messages.length === 0 && (
          <div className="flex-1 flex flex-col items-center justify-center h-full text-center py-20">
            <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${themeColors[theme]} flex items-center justify-center text-3xl mb-4`}>
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
            <div className={`group max-w-[85%] rounded-2xl px-4 py-2.5 ${
              msg.role === 'user'
                ? `bg-gradient-to-r ${themeColors[theme]} text-white`
                : 'bg-white/5 text-zinc-100'
            }`}>
              <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
              
              {msg.role === 'assistant' && (
                <div className="flex gap-3 mt-1 pt-1 border-t border-white/10 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button onClick={() => copyToClipboard(msg.content)} className="text-xs text-zinc-500 hover:text-zinc-300">
                    Copy
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
      
      {/* Input */}
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
  )
}
