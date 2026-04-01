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
    icon: 'S',
    enabled: true,
    category: 'search',
  },
  {
    id: 'code-executor',
    name: 'Code Executor',
    description: 'Execute Python and JavaScript code',
    version: '1.0.0',
    author: 'SloughGPT',
    icon: 'C',
    enabled: true,
    category: 'code',
  },
  {
    id: 'file-reader',
    name: 'File Reader',
    description: 'Read and analyze uploaded files',
    version: '1.0.0',
    author: 'SloughGPT',
    icon: 'F',
    enabled: true,
    category: 'data',
  },
  {
    id: 'image-gen',
    name: 'Image Generation',
    description: 'Generate images from text descriptions',
    version: '1.0.0',
    author: 'SloughGPT',
    icon: 'I',
    enabled: false,
    category: 'integration',
  },
  {
    id: 'github',
    name: 'GitHub Integration',
    description: 'Connect to GitHub for repository operations',
    version: '1.0.0',
    author: 'SloughGPT',
    icon: 'G',
    enabled: false,
    category: 'integration',
  },
  {
    id: 'translate',
    name: 'Translator',
    description: 'Translate text between languages',
    version: '1.0.0',
    author: 'SloughGPT',
    icon: 'T',
    enabled: false,
    category: 'utility',
  },
]

const CATEGORIES = [
  { id: 'all', name: 'All' },
  { id: 'search', name: 'Search' },
  { id: 'code', name: 'Code' },
  { id: 'data', name: 'Data' },
  { id: 'integration', name: 'Integrations' },
  { id: 'utility', name: 'Utilities' },
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
    savePlugins(plugins.map((p) => (p.id === id ? { ...p, enabled: !p.enabled } : p)))
  }

  const installPlugin = () => {
    if (!installUrl.trim()) return
    const newPlugin: Plugin = {
      id: `plugin-${Date.now()}`,
      name: 'Custom Plugin',
      description: 'Installed from URL',
      version: '1.0.0',
      author: 'Unknown',
      icon: 'P',
      enabled: true,
      category: 'integration',
    }
    savePlugins([...plugins, newPlugin])
    setInstallUrl('')
    setShowInstall(false)
  }

  const uninstallPlugin = (id: string) => {
    savePlugins(plugins.filter((p) => p.id !== id))
  }

  const filteredPlugins = selectedCategory === 'all' ? plugins : plugins.filter((p) => p.category === selectedCategory)

  return (
    <div className="sl-page max-w-6xl mx-auto">
      <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="sl-h1">Plugins</h1>
          <p className="text-muted-foreground text-sm">Extend SloughGPT with plugins</p>
        </div>
        <button type="button" onClick={() => setShowInstall(true)} className="sl-btn-primary rounded-lg px-4 py-2">
          Install Plugin
        </button>
      </div>

      <div className="mb-6 flex gap-2 overflow-x-auto pb-2">
        {CATEGORIES.map((cat) => (
          <button
            key={cat.id}
            type="button"
            onClick={() => setSelectedCategory(cat.id)}
            className={`whitespace-nowrap rounded-lg border px-4 py-2 text-sm font-medium transition-colors ${
              selectedCategory === cat.id
                ? 'border-primary/40 bg-primary/15 text-primary'
                : 'border-border bg-muted/40 text-muted-foreground hover:text-foreground'
            }`}
          >
            {cat.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {filteredPlugins.map((plugin) => (
          <div
            key={plugin.id}
            className={`sl-card p-4 transition-all ${
              plugin.enabled ? 'ring-1 ring-primary/15' : 'opacity-70'
            }`}
          >
            <div className="mb-3 flex items-start justify-between">
              <div className="flex items-center gap-3">
                <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/15 font-mono text-sm font-semibold text-primary ring-1 ring-primary/20">
                  {plugin.icon}
                </span>
                <div>
                  <h3 className="font-semibold text-foreground">{plugin.name}</h3>
                  <p className="text-xs text-muted-foreground">
                    v{plugin.version} · {plugin.author}
                  </p>
                </div>
              </div>
              <label className="relative inline-flex cursor-pointer items-center">
                <input type="checkbox" className="peer sr-only" checked={plugin.enabled} onChange={() => togglePlugin(plugin.id)} />
                <div className="peer h-6 w-11 rounded-full bg-muted after:absolute after:left-0.5 after:top-0.5 after:h-5 after:w-5 after:rounded-full after:border after:border-border after:bg-card after:transition-all peer-checked:bg-primary peer-checked:after:translate-x-5" />
              </label>
            </div>

            <p className="mb-4 text-sm text-muted-foreground">{plugin.description}</p>

            <div className="flex items-center justify-between">
              <span className="rounded-md border border-border bg-muted/50 px-2 py-1 text-xs text-muted-foreground">
                {plugin.category}
              </span>
              <button
                type="button"
                onClick={() => uninstallPlugin(plugin.id)}
                className="text-xs font-medium text-destructive hover:underline"
              >
                Uninstall
              </button>
            </div>
          </div>
        ))}
      </div>

      {showInstall && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 p-4 backdrop-blur-sm"
          onClick={() => setShowInstall(false)}
          role="presentation"
        >
          <div className="sl-card-solid w-full max-w-md border border-border p-6 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <h2 className="mb-4 text-xl font-semibold text-foreground">Install Plugin</h2>
            <p className="mb-4 text-sm text-muted-foreground">Enter a plugin URL or npm package name to install.</p>
            <input
              type="text"
              value={installUrl}
              onChange={(e) => setInstallUrl(e.target.value)}
              placeholder="https://example.com/plugin.js or @scope/plugin"
              className="sl-input mb-4"
            />
            <div className="flex gap-2">
              <button type="button" onClick={installPlugin} className="flex-1 sl-btn-primary rounded-lg py-2">
                Install
              </button>
              <button type="button" onClick={() => setShowInstall(false)} className="sl-btn-secondary rounded-lg px-4 py-2">
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="sl-card mt-8 border-primary/15 bg-primary/5 p-6">
        <h3 className="mb-2 font-semibold text-foreground">Model Context Protocol (MCP)</h3>
        <p className="mb-3 text-sm text-muted-foreground">
          SloughGPT supports MCP for plugin integration. Build extensions with the MCP SDK.
        </p>
        <a
          href="https://modelcontextprotocol.io"
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm font-medium text-primary hover:underline"
        >
          Learn more about MCP →
        </a>
      </div>
    </div>
  )
}
