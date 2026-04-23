'use client'

import { useState, useEffect, useCallback } from 'react'
import { api } from '@/lib/api'

export interface KnowledgeItem {
  id: string
  content: string
  category: string
  tags: string[]
  created_at: string
  usage_count: number
}

interface UseKnowledgeResult {
  items: KnowledgeItem[]
  loading: boolean
  add: (content: string, category?: string) => Promise<void>
  remove: (id: string) => Promise<void>
  search: (query: string) => Promise<KnowledgeItem[]>
  refresh: () => Promise<void>
  stats: {
    total: number
    categories: number
    total_usage: number
  }
}

export function useKnowledge(): UseKnowledgeResult {
  const [items, setItems] = useState<KnowledgeItem[]>([])
  const [loading, setLoading] = useState(true)

  const fetchItems = useCallback(async () => {
    try {
      const data = await api.getKnowledge()
      setItems(data.items || [])
    } catch (err) {
      console.error('Failed to fetch knowledge:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchItems()
  }, [fetchItems])

  const add = useCallback(async (content: string, category: string = 'general') => {
    if (!content.trim()) return

    try {
      const data = await api.addKnowledge(content, category)
      setItems(prev => [...prev, data.item])
    } catch (err) {
      console.error('Failed to add knowledge:', err)
    }
  }, [])

  const remove = useCallback(async (id: string) => {
    try {
      await api.deleteKnowledge(id)
      setItems(prev => prev.filter(item => item.id !== id))
    } catch (err) {
      console.error('Failed to delete knowledge:', err)
    }
  }, [])

  const search = useCallback(async (query: string): Promise<KnowledgeItem[]> => {
    try {
      const data = await api.searchKnowledge(query)
      return data.results || []
    } catch (err) {
      console.error('Failed to search knowledge:', err)
    }
    return []
  }, [])

  const categories = new Set(items.map(i => i.category))
  const totalUsage = items.reduce((sum, i) => sum + (i.usage_count || 0), 0)

  return {
    items,
    loading,
    add,
    remove,
    search,
    refresh: fetchItems,
    stats: {
      total: items.length,
      categories: categories.size,
      total_usage: totalUsage,
    },
  }
}