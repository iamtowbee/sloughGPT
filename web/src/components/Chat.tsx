import React, { useState, useEffect, useRef } from 'react'
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Spinner
} from '@base-ui/react'
import { useStore, ChatMessage } from '../store'
import { api } from '../utils/api'

interface ChatInterfaceProps {}

export const ChatInterface: React.FC<ChatInterfaceProps> = () => {
  const [inputValue, setInputValue] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  
  const { 
    messages, 
    addMessage, 
    selectedModel,
    currentConversationId,
    setCurrentConversationId,
    conversations,
    setConversations,
    models
  } = useStore()

  useEffect(() => {
    loadConversations()
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const loadConversations = async () => {
    try {
      const response = await api.listConversations()
      if (response.data) {
        setConversations(response.data.conversations)
      }
    } catch (err) {
      console.error('Failed to load conversations:', err)
    }
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const handleNewConversation = async () => {
    try {
      const response = await api.createConversation()
      if (response.data) {
        setConversations([response.data, ...conversations])
        setCurrentConversationId(response.data.id)
      }
    } catch (err) {
      console.error('Failed to create conversation:', err)
    }
  }

  const handleSwitchConversation = async (convId: string) => {
    setCurrentConversationId(convId)
    try {
      const response = await api.getConversation(convId)
      if (response.data) {
        // Load messages for this conversation
        console.log('Loaded conversation:', response.data.messages.length, 'messages')
      }
    } catch (err) {
      console.error('Failed to load conversation:', err)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputValue.trim() || isSending) return
    
    setIsSending(true)
    setError(null)
    
    const userMessage = inputValue.trim()
    setInputValue('')
    
    // Add user message immediately
    addMessage({
      role: 'user',
      content: userMessage,
      model: selectedModel
    })
    
    try {
      const response = await api.sendChatMessage(
        userMessage,
        selectedModel,
        currentConversationId
      )
      
      if (response.data) {
        addMessage({
          role: 'assistant',
          content: response.data.message.content,
          model: response.data.model
        })
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message')
      
      // Add error message
      addMessage({
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        model: selectedModel
      })
    } finally {
      setIsSending(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <Card className="mb-4">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-xl font-semibold">Chat</span>
              <select
                value={currentConversationId}
                onChange={(e) => handleSwitchConversation(e.target.value)}
                className="px-2 py-1 text-sm border border-slate-300 rounded"
              >
                {conversations.length > 0 ? (
                  conversations.map(conv => (
                    <option key={conv.id} value={conv.id}>
                      {conv.name} ({conv.message_count} messages)
                    </option>
                  ))
                ) : (
                  <option value="conv_default">Default</option>
                )}
              </select>
            </div>
            <Button 
              variant="outline" 
              size="sm"
              onClick={handleNewConversation}
            >
              + New Chat
            </Button>
          </CardTitle>
        </CardHeader>
        
        <CardDescription className="px-4 pb-2 text-sm text-slate-500">
          Current model: {models.find(m => m.id === selectedModel)?.name || selectedModel}
        </CardDescription>
      </Card>
      
      {/* Messages */}
      <div className="flex-1 overflow-y-auto mb-4 px-4" style={{ maxHeight: 'calc(100vh - 280px)' }}>
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-slate-400">
            <div className="text-center">
              <div className="text-4xl mb-4">💬</div>
              <p>Start a conversation by typing a message below</p>
              <p className="text-sm mt-2">Select a model from the dropdown</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <ChatMessageComponent key={message.id} message={message} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
      
      {/* Error message */}
      {error && (
        <div className="mx-4 mb-2 px-4 py-2 bg-red-50 text-red-600 rounded-lg text-sm">
          {error}
        </div>
      )}
      
      {/* Input */}
      <Card className="mb-4">
        <CardContent className="p-4">
          <form onSubmit={handleSubmit} className="flex items-end gap-2">
            <div className="flex-1">
              <textarea
                placeholder="Type your message here... (press Enter to send)"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={isSending}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-slate-100"
                rows={2}
                style={{ minHeight: '60px' }}
              />
            </div>
            
            <div className="flex flex-col gap-2">
              <select
                value={selectedModel}
                onChange={(e) => useStore.getState().setSelectedModel(e.target.value)}
                className="px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={isSending}
              >
                {models.map(m => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
              
              <Button
                type='submit'
                disabled={isSending || !inputValue.trim()}
                className="h-10 px-4 rounded-lg bg-blue-500 text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSending ? <Spinner className="h-4 w-4" /> : 'Send'}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}

interface ChatMessageProps {
  message: ChatMessage
}

const ChatMessageComponent: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user'
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      {!isUser && (
        <div className="mr-3 flex-shrink-0">
          <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center text-xl">
            🤖
          </div>
        </div>
      )}
      
      <div 
        className={`max-w-md px-4 py-3 rounded-lg ${
          isUser 
            ? 'bg-blue-500 text-white' 
            : 'bg-gray-100 text-gray-900'
        }`}
      >
        <div className="text-xs mb-1 opacity-75">
          {isUser ? 'You' : 'Assistant'}
          {message.model && (
            <span className="ml-2">· {message.model}</span>
          )}
        </div>
        
        <div className="text-sm whitespace-pre-wrap">
          {message.content}
        </div>
        
        <div className="text-xs opacity-50 mt-2">
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>
      
      {isUser && (
        <div className="ml-3 flex-shrink-0">
          <div className="w-10 h-10 rounded-full bg-slate-200 flex items-center justify-center text-xl">
            👤
          </div>
        </div>
      )}
    </div>
  )
}

export default ChatInterface
