'use client'

import { useState, useEffect } from 'react'

interface Plugin {
  id: string
  name: string
  description: string
  version: string
  author: string
  icon: string
  enabled: boolean
  category: 'search' | 'code' | 'data' | 'integration' | 'utility'
}

const DEFAULT_PLUGINS: Plugin[] = [
  {
    id: 'web-search',
    name: 'Web Search',
    description: 'Search the web for current information',
    version: '1.0.0',
    author: 'SloughGPT',
    icon: '🔍',
    enabled: true,
    category: 'search'
  },
  {
    id: 'code-executor',
    name: 'Code Executor',
    description: 'Execute Python and JavaScript code',
    version: '1.0.0',
    author: 'SloughGPT',
    icon: '💻',
    enabled: true,
    category: 'code'
  },
  {
    id: 'file-reader',
    name: 'File Reader',
    description: 'Read and analyze uploaded files',
    version: '1.0.0',
    author: 'SloughGPT',
    icon: '📄',
    enabled: true,
    category: 'data'
  },
  {
    id: 'image-gen',
    name: 'Image Generation',
    description: 'Generate images from text descriptions',
    version: '1.0.0',
    author: 'SloughGPT',
    icon: '🎨',
    enabled: false,
    category: 'integration'
  },
  {
    id: 'github',
    name: 'GitHub Integration',
    description: 'Connect to GitHub for repository operations',
    version: '1.0.0',
    author: 'SloughGPT',
    icon: '🐙',
    enabled: false,
    category: 'integration'
  },
  {
    id: 'translate',
    name: 'Translator',
    description: 'Translate text between languages',
    version: '1.0.0',
    author: 'SloughGPT',
    icon: '🌐',
    enabled: false,
    category: 'utility'
  }
]

const CATEGORIES = [
  { id: 'all', name: 'All', icon: '📦' },
  { id: 'search', name: 'Search', icon: '🔍' },
  { id: 'code', name: 'Code', icon: '💻' },
  { id: 'data', name: 'Data', icon: '📊' },
  { id: 'integration', name: 'Integrations', icon: '🔗' },
  { id: 'utility', name: 'Utilities', icon: '🛠️' },
]

export default function PluginsPage() {
  const [plugins, setPlugins] = useState<Plugin[]>([])
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [showInstall, setShowInstall] = useState(false)
  const [installUrl, setInstallUrl] = useState('')
  
  useEffect(() => {
    const saved = localStorage.getItem('sloughgpt_plugins')
    if (saved) {
      try {
        setPlugins(JSON.parse(saved))
      } catch {
        setPlugins(DEFAULT_PLUGINS)
      }
    } else {
      setPlugins(DEFAULT_PLUGINS)
    }
  }, [])
  
  const savePlugins = (newPlugins: Plugin[]) => {
    setPlugins(newPlugins)
    localStorage.setItem('sloughgpt_plugins', JSON.stringify(newPlugins))
  }
  
  const togglePlugin = (id: string) => {
    savePlugins(plugins.map(p => 
      p.id === id ? { ...p, enabled: !p.enabled } : p
    ))
  }
  
  const installPlugin = () => {
    if (!installUrl.trim()) return
    
    const newPlugin: Plugin = {
      id: `plugin-${Date.now()}`,
      name: 'Custom Plugin',
      description: 'Installed from URL',
      version: '1.0.0',
      author: 'Unknown',
      icon: '🔌',
      enabled: true,
      category: 'integration'
    }
    
    savePlugins([...plugins, newPlugin])
    setInstallUrl('')
    setShowInstall(false)
  }
  
  const uninstallPlugin = (id: string) => {
    savePlugins(plugins.filter(p => p.id !== id))
  }
  
  const filteredPlugins = selectedCategory === 'all' 
    ? plugins 
    : plugins.filter(p => p.category === selectedCategory)
  
  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 dark:text-white">Plugins</h1>
          <p className="text-slate-500">Extend SloughGPT with plugins</p>
        </div>
        <button
          onClick={() => setShowInstall(true)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Install Plugin
        </button>
      </div>
      
      {/* Categories */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {CATEGORIES.map(cat => (
          <button
            key={cat.id}
            onClick={() => setSelectedCategory(cat.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg whitespace-nowrap transition-colors ${
              selectedCategory === cat.id
                ? 'bg-blue-600 text-white'
                : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
            }`}
          >
            <span>{cat.icon}</span>
            <span>{cat.name}</span>
          </button>
        ))}
      </div>
      
      {/* Plugins Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredPlugins.map(plugin => (
          <div
            key={plugin.id}
            className={`bg-white dark:bg-slate-800 rounded-xl border p-4 transition-all ${
              plugin.enabled 
                ? 'border-blue-200 dark:border-blue-800 shadow-sm' 
                : 'border-slate-200 dark:border-slate-700 opacity-60'
            }`}
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className="text-2xl">{plugin.icon}</div>
                <div>
                  <h3 className="font-semibold text-slate-800 dark:text-white">{plugin.name}</h3>
                  <p className="text-xs text-slate-500">v{plugin.version} by {plugin.author}</p>
                </div>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={plugin.enabled}
                  onChange={() => togglePlugin(plugin.id)}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-slate-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-slate-600 peer-checked:bg-blue-600"></div>
              </label>
            </div>
            
            <p className="text-sm text-slate-600 dark:text-slate-300 mb-4">{plugin.description}</p>
            
            <div className="flex items-center justify-between">
              <span className="text-xs px-2 py-1 bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 rounded">
                {plugin.category}
              </span>
              <button
                onClick={() => uninstallPlugin(plugin.id)}
                className="text-xs text-red-500 hover:text-red-600"
              >
                Uninstall
              </button>
            </div>
          </div>
        ))}
      </div>
      
      {/* Install Modal */}
      {showInstall && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowInstall(false)}>
          <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 max-w-md w-full mx-4" onClick={e => e.stopPropagation()}>
            <h2 className="text-xl font-bold text-slate-800 dark:text-white mb-4">Install Plugin</h2>
            <p className="text-sm text-slate-500 mb-4">Enter a plugin URL or npm package name to install.</p>
            <input
              type="text"
              value={installUrl}
              onChange={e => setInstallUrl(e.target.value)}
              placeholder="https://example.com/plugin.js or @scope/plugin"
              className="w-full px-4 py-2 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-800 dark:text-white mb-4"
            />
            <div className="flex gap-2">
              <button
                onClick={installPlugin}
                className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium"
              >
                Install
              </button>
              <button
                onClick={() => setShowInstall(false)}
                className="px-4 py-2 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg font-medium"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* MCP Protocol Info */}
      <div className="mt-8 p-6 bg-blue-50 dark:bg-blue-900/20 rounded-xl">
        <div className="flex items-start gap-4">
          <div className="text-3xl">🔌</div>
          <div>
            <h3 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">Model Context Protocol (MCP)</h3>
            <p className="text-sm text-blue-700 dark:text-blue-400 mb-3">
              SloughGPT supports the Model Context Protocol for seamless plugin integration. 
              Build your own plugins using the MCP SDK to extend AI capabilities.
            </p>
            <a 
              href="https://modelcontextprotocol.io" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-sm text-blue-600 dark:text-blue-300 hover:underline"
            >
              Learn more about MCP →
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}
