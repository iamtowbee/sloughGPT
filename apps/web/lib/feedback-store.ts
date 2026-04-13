'use client'

import { create } from 'zustand'
import { api, type FeedbackStats, type UserAdapterStats, type WorkflowStatus } from './api'

interface FeedbackState {
  stats: FeedbackStats | null
  adapterStats: UserAdapterStats | null
  workflowStatus: WorkflowStatus | null
  isLoading: boolean
  error: string | null
  
  // Actions
  recordFeedback: (params: {
    userMessage: string
    assistantResponse: string
    rating: 'thumbs_up' | 'thumbs_down'
    conversationId?: string
    qualityScore?: number
    userId?: string
  }) => Promise<boolean>
  
  fetchStats: () => Promise<void>
  fetchAdapterStats: () => Promise<void>
  fetchWorkflowStatus: () => Promise<void>
  triggerWorkflowAction: (action: 'aggregate' | 'prune' | 'export') => Promise<boolean>
  reset: () => void
}

export const useFeedbackStore = create<FeedbackState>()((set, get) => ({
  stats: null,
  adapterStats: null,
  workflowStatus: null,
  isLoading: false,
  error: null,

  recordFeedback: async (params) => {
    set({ isLoading: true, error: null })
    try {
      await api.recordFeedback(params)
      await get().fetchStats()
      await get().fetchAdapterStats()
      set({ isLoading: false })
      return true
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Failed to record feedback'
      set({ isLoading: false, error })
      return false
    }
  },

  fetchStats: async () => {
    try {
      const stats = await api.getFeedbackStats()
      set({ stats })
    } catch (err) {
      console.error('Failed to fetch feedback stats:', err)
    }
  },

  fetchAdapterStats: async () => {
    try {
      const adapterStats = await api.getUserAdapters()
      set({ adapterStats })
    } catch (err) {
      console.error('Failed to fetch adapter stats:', err)
    }
  },

  fetchWorkflowStatus: async () => {
    try {
      const workflowStatus = await api.getWorkflowStatus()
      set({ workflowStatus })
    } catch (err) {
      console.error('Failed to fetch workflow status:', err)
    }
  },

  triggerWorkflowAction: async (action) => {
    set({ isLoading: true, error: null })
    try {
      await api.triggerWorkflowAction(action)
      await get().fetchAdapterStats()
      await get().fetchWorkflowStatus()
      set({ isLoading: false })
      return true
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Failed to trigger action'
      set({ isLoading: false, error })
      return false
    }
  },

  reset: () => {
    set({
      stats: null,
      adapterStats: null,
      workflowStatus: null,
      isLoading: false,
      error: null,
    })
  },
}))
