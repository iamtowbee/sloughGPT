import { api } from './api'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  model?: string
}

export const chatService = {
  async sendMessage(
    message: string,
    model?: string,
    conversationId?: string
  ): Promise<ChatMessage> {
    const response = await api.sendChatMessage(message, model, conversationId)
    
    if (response.error) {
      throw new Error(response.error)
    }
    
    return {
      id: Date.now().toString(),
      role: 'assistant',
      content: response.data?.message.content || '',
      timestamp: new Date(response.data?.message.timestamp || ''),
      model: response.data?.model
    }
  },
  
  async getConversation(conversationId: string): Promise<ChatMessage[]> {
    const response = await api.getConversation(conversationId)
    
    if (response.error) {
      throw new Error(response.error)
    }
    
    return (response.data?.messages || []).map((msg: any, index: number) => ({
      id: `${conversationId}-${index}`,
      role: msg.role as 'user' | 'assistant',
      content: msg.content,
      timestamp: new Date(msg.timestamp)
    }))
  }
}

export const modelService = {
  async getModels() {
    const response = await api.getModels()
    
    if (response.error) {
      throw new Error(response.error)
    }
    
    return response.data?.models || []
  }
}

export const healthService = {
  async check() {
    const response = await api.healthCheck()
    
    return {
      healthy: !response.error,
      status: response.data?.status || 'unknown',
      timestamp: response.data?.timestamp
    }
  }
}

export default { chatService, modelService, healthService }