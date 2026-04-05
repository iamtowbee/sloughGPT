'use client'

import { useState, useEffect } from 'react'

import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'

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
    <div className="sl-page mx-auto max-w-6xl">
      <AppRouteHeader
        className="mb-6 items-start"
        left={
          <AppRouteHeaderLead
            title="Plugins"
            subtitle="Extend SloughGPT with plugins (local demo state)"
          />
        }
        right={
          <Button type="button" onClick={() => setShowInstall(true)}>
            Install plugin
          </Button>
        }
      />

      <div className="mb-6 inline-flex flex-wrap gap-0 border border-border bg-muted/30 p-0.5">
        {CATEGORIES.map((cat) => (
          <Button
            key={cat.id}
            type="button"
            variant={selectedCategory === cat.id ? 'default' : 'ghost'}
            size="sm"
            className="rounded-none px-4"
            onClick={() => setSelectedCategory(cat.id)}
          >
            {cat.name}
          </Button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {filteredPlugins.map((plugin) => (
          <Card
            key={plugin.id}
            className={`transition-all duration-200 ${plugin.enabled ? 'border-primary/20' : 'opacity-80'}`}
          >
            <CardHeader className="pb-2">
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-3">
                  <span className="flex h-10 w-10 items-center justify-center border border-primary/25 bg-primary/10 font-mono text-sm font-semibold text-primary">
                    {plugin.icon}
                  </span>
                  <div>
                    <CardTitle className="text-base">{plugin.name}</CardTitle>
                    <CardDescription>
                      v{plugin.version} · {plugin.author}
                    </CardDescription>
                  </div>
                </div>
                <Switch checked={plugin.enabled} onCheckedChange={() => togglePlugin(plugin.id)} />
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">{plugin.description}</p>
            </CardContent>
            <CardFooter className="flex justify-between border-t border-border pt-4">
              <span className="border border-border bg-muted/50 px-2 py-1 text-xs text-muted-foreground">
                {plugin.category}
              </span>
              <Button type="button" variant="ghost" size="sm" className="text-destructive" onClick={() => uninstallPlugin(plugin.id)}>
                Uninstall
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>

      <Dialog open={showInstall} onOpenChange={setShowInstall}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Install plugin</DialogTitle>
            <DialogDescription>Enter a plugin URL or npm package name (demo only — not wired to a registry).</DialogDescription>
          </DialogHeader>
          <div className="space-y-2 py-2">
            <Label htmlFor="install-url">Source</Label>
            <Input
              id="install-url"
              value={installUrl}
              onChange={(e) => setInstallUrl(e.target.value)}
              placeholder="https://example.com/plugin.js or @scope/plugin"
            />
          </div>
          <DialogFooter>
            <Button type="button" variant="secondary" onClick={() => setShowInstall(false)}>
              Cancel
            </Button>
            <Button type="button" onClick={installPlugin}>
              Install
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Card className="mt-8 border-primary/15 bg-primary/5">
        <CardHeader>
          <CardTitle className="text-base">Model Context Protocol (MCP)</CardTitle>
          <CardDescription>
            SloughGPT can integrate with MCP-style extensions. See upstream docs for production wiring.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <a
            href="https://modelcontextprotocol.io"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-medium text-primary hover:underline"
          >
            Learn more about MCP →
          </a>
        </CardContent>
      </Card>
    </div>
  )
}
