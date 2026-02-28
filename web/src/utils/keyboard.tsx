import { useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'

interface KeyboardShortcut {
  key: string
  ctrl?: boolean
  shift?: boolean
  alt?: boolean
  action: () => void
  description: string
}

export const useKeyboardShortcuts = (shortcuts: KeyboardShortcut[]) => {
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    for (const shortcut of shortcuts) {
      const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase()
      const ctrlMatch = shortcut.ctrl ? (event.ctrlKey || event.metaKey) : true
      const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey
      const altMatch = shortcut.alt ? event.altKey : !event.altKey

      if (keyMatch && ctrlMatch && shiftMatch && altMatch) {
        event.preventDefault()
        shortcut.action()
        break
      }
    }
  }, [shortcuts])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])
}

export const useGlobalShortcuts = () => {
  const navigate = useNavigate()

  const shortcuts: KeyboardShortcut[] = [
    {
      key: '1',
      action: () => navigate('/'),
      description: 'Go to Home'
    },
    {
      key: '2',
      action: () => navigate('/chat'),
      description: 'Go to Chat'
    },
    {
      key: '3',
      action: () => navigate('/datasets'),
      description: 'Go to Datasets'
    },
    {
      key: '4',
      action: () => navigate('/models'),
      description: 'Go to Models'
    },
    {
      key: '5',
      action: () => navigate('/training'),
      description: 'Go to Training'
    },
    {
      key: '6',
      action: () => navigate('/monitoring'),
      description: 'Go to Monitoring'
    },
    {
      key: 's',
      ctrl: true,
      action: () => {
        // Save action - could be extended
        console.log('Save shortcut triggered')
      },
      description: 'Save (Ctrl+S)'
    },
    {
      key: 'n',
      ctrl: true,
      action: () => {
        // New conversation
        navigate('/chat')
      },
      description: 'New Chat (Ctrl+N)'
    },
    {
      key: 'k',
      ctrl: true,
      action: () => {
        // Focus search
        const searchInput = document.querySelector('input[type="search"]') as HTMLInputElement
        searchInput?.focus()
      },
      description: 'Search (Ctrl+K)'
    },
    {
      key: '/',
      action: () => {
        // Show help
        alert('Keyboard Shortcuts:\n\n1 - Home\n2 - Chat\n3 - Datasets\n4 - Models\n5 - Training\n6 - Monitoring\nCtrl+N - New Chat\nCtrl+K - Search\nCtrl+/ - Show Shortcuts')
      },
      description: 'Show Shortcuts'
    }
  ]

  useKeyboardShortcuts(shortcuts)
}

export const KeyboardShortcutsHelp: React.FC = () => {
  const shortcuts = [
    { key: '1', description: 'Home' },
    { key: '2', description: 'Chat' },
    { key: '3', description: 'Datasets' },
    { key: '4', description: 'Models' },
    { key: '5', description: 'Training' },
    { key: '6', description: 'Monitoring' },
    { key: 'Ctrl+N', description: 'New Chat' },
    { key: 'Ctrl+K', description: 'Search' },
    { key: 'Ctrl+S', description: 'Save' },
    { key: '?', description: 'Show Shortcuts' }
  ]

  return (
    <div className="p-4 bg-slate-100 rounded-lg">
      <h3 className="font-semibold mb-2">Keyboard Shortcuts</h3>
      <div className="grid grid-cols-2 gap-2 text-sm">
        {shortcuts.map((s) => (
          <div key={s.key} className="flex justify-between">
            <kbd className="px-2 py-1 bg-white rounded border border-slate-300 text-xs font-mono">
              {s.key}
            </kbd>
            <span className="text-slate-600">{s.description}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default useGlobalShortcuts
