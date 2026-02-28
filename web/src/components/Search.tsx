import React, { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'

interface SearchResult {
  id: string
  type: 'conversation' | 'dataset' | 'model' | 'training'
  title: string
  description: string
  path: string
}

interface SearchModalProps {
  isOpen: boolean
  onClose: () => void
}

export const SearchModal: React.FC<SearchModalProps> = ({ isOpen, onClose }) => {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100)
      setQuery('')
      setResults([])
      setSelectedIndex(0)
    }
  }, [isOpen])

  useEffect(() => {
    // Mock search results - in real app would call API
    const mockResults: SearchResult[] = query.length > 0 ? [
      {
        id: '1',
        type: 'conversation',
        title: 'General Chat',
        description: 'Default conversation',
        path: '/chat'
      },
      {
        id: '2',
        type: 'dataset',
        title: 'Training Data',
        description: 'Custom training dataset',
        path: '/datasets'
      },
      {
        id: '3',
        type: 'model',
        title: 'GPT-4',
        description: 'OpenAI GPT-4 model',
        path: '/models'
      },
      {
        id: '4',
        type: 'training',
        title: 'Current Training Job',
        description: 'Training nanogpt model',
        path: '/training'
      }
    ].filter(r => 
      r.title.toLowerCase().includes(query.toLowerCase()) ||
      r.description.toLowerCase().includes(query.toLowerCase())
    ) : []

    setResults(mockResults)
    setSelectedIndex(0)
  }, [query])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        setSelectedIndex(i => Math.min(i + 1, results.length - 1))
        break
      case 'ArrowUp':
        e.preventDefault()
        setSelectedIndex(i => Math.max(i - 1, 0))
        break
      case 'Enter':
        e.preventDefault()
        if (results[selectedIndex]) {
          navigate(results[selectedIndex].path)
          onClose()
        }
        break
      case 'Escape':
        onClose()
        break
    }
  }

  const getTypeIcon = (type: string) => {
    const icons: Record<string, string> = {
      conversation: '💬',
      dataset: '📊',
      model: '🤖',
      training: '🧠'
    }
    return icons[type] || '📄'
  }

  const getTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      conversation: 'bg-blue-100 text-blue-800',
      dataset: 'bg-green-100 text-green-800',
      model: 'bg-purple-100 text-purple-800',
      training: 'bg-orange-100 text-orange-800'
    }
    return colors[type] || 'bg-gray-100 text-gray-800'
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-24">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="relative w-full max-w-xl bg-white dark:bg-slate-800 rounded-xl shadow-2xl overflow-hidden">
        {/* Search Input */}
        <div className="p-4 border-b border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-3">
            <span className="text-xl">🔍</span>
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search conversations, datasets, models..."
              className="flex-1 bg-transparent text-lg outline-none placeholder:text-slate-400 dark:text-white"
            />
            <kbd className="px-2 py-1 text-xs bg-slate-100 dark:bg-slate-700 rounded">
              ESC
            </kbd>
          </div>
        </div>

        {/* Results */}
        <div className="max-h-96 overflow-y-auto">
          {results.length === 0 ? (
            <div className="p-8 text-center text-slate-500">
              {query ? 'No results found' : 'Start typing to search...'}
            </div>
          ) : (
            <ul>
              {results.map((result, index) => (
                <li key={result.id}>
                  <button
                    onClick={() => {
                      navigate(result.path)
                      onClose()
                    }}
                    className={`w-full flex items-center gap-3 px-4 py-3 text-left transition-colors ${
                      index === selectedIndex
                        ? 'bg-blue-50 dark:bg-blue-900/20'
                        : 'hover:bg-slate-50 dark:hover:bg-slate-700/50'
                    }`}
                  >
                    <span className="text-xl">{getTypeIcon(result.type)}</span>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium truncate dark:text-white">
                        {result.title}
                      </p>
                      <p className="text-sm text-slate-500 truncate">
                        {result.description}
                      </p>
                    </div>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${getTypeColor(result.type)}`}>
                      {result.type}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
          <div className="flex items-center gap-4 text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-white dark:bg-slate-700 rounded">↑</kbd>
              <kbd className="px-1.5 py-0.5 bg-white dark:bg-slate-700 rounded">↓</kbd>
              to navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-white dark:bg-slate-700 rounded">↵</kbd>
              to select
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export const SearchButton: React.FC<{ onClick: () => void }> = ({ onClick }) => {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-500 bg-slate-100 dark:bg-slate-800 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
    >
      <span>🔍</span>
      <span>Search...</span>
      <kbd className="px-1.5 py-0.5 text-xs bg-white dark:bg-slate-600 rounded">⌘K</kbd>
    </button>
  )
}

export default SearchModal
