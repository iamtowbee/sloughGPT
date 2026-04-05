'use client'

import { useState, useEffect } from 'react'

import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { PUBLIC_API_URL } from '@/lib/config'

interface Settings {
  apiUrl: string
  hfToken: string
  defaultModel: string
  defaultTemp: number
  defaultMaxTokens: number
  theme: 'dark' | 'light' | 'system'
  streaming: boolean
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings>({
    apiUrl: PUBLIC_API_URL,
    hfToken: '',
    defaultModel: 'gpt2',
    defaultTemp: 0.8,
    defaultMaxTokens: 200,
    theme: 'light',
    streaming: true,
  })
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    const stored = localStorage.getItem('sloughgpt_settings')
    if (stored) {
      try {
        setSettings((s) => ({ ...s, ...JSON.parse(stored) }))
      } catch {
        /* ignore */
      }
    }
  }, [])

  const saveSettings = () => {
    localStorage.setItem('sloughgpt_settings', JSON.stringify(settings))
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const clearChat = () => {
    localStorage.removeItem('sloughgpt_messages')
    localStorage.removeItem('sloughgpt_chat_sessions_v1')
    localStorage.removeItem('sloughgpt_active_chat_v1')
  }

  return (
    <div className="sl-page mx-auto max-w-2xl">
      <AppRouteHeader
        className="mb-6 items-center"
        left={<AppRouteHeaderLead title="Settings" />}
        right={saved ? <span className="text-sm font-medium text-success">Saved!</span> : undefined}
      />

      <Tabs defaultValue="connection" className="space-y-6">
        <TabsList className="w-full justify-start">
          <TabsTrigger value="connection">Connection</TabsTrigger>
          <TabsTrigger value="defaults">Model defaults</TabsTrigger>
          <TabsTrigger value="data">Data</TabsTrigger>
        </TabsList>

        <TabsContent value="connection" className="space-y-6 focus-visible:outline-none">
          <Card>
            <CardHeader>
              <CardTitle>API</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="api-url">API URL</Label>
                <Input
                  id="api-url"
                  type="text"
                  value={settings.apiUrl}
                  onChange={(e) => setSettings({ ...settings, apiUrl: e.target.value })}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="hf-token">Hugging Face token</Label>
                <Input
                  id="hf-token"
                  type="password"
                  value={settings.hfToken}
                  onChange={(e) => setSettings({ ...settings, hfToken: e.target.value })}
                  placeholder="hf_…"
                />
              </div>
              <div className="flex items-center justify-between gap-4 border border-border p-3">
                <div>
                  <p className="text-sm font-medium text-foreground">Streaming</p>
                  <p className="text-xs text-muted-foreground">Stream tokens as they generate (client preference)</p>
                </div>
                <Switch
                  checked={settings.streaming}
                  onCheckedChange={(streaming) => setSettings({ ...settings, streaming })}
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="defaults" className="space-y-6 focus-visible:outline-none">
          <Card>
            <CardHeader>
              <CardTitle>Defaults</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="def-model">Default model</Label>
                <select
                  id="def-model"
                  value={settings.defaultModel}
                  onChange={(e) => setSettings({ ...settings, defaultModel: e.target.value })}
                  className="sl-input"
                >
                  <option value="gpt2">GPT-2</option>
                  <option value="sloughgpt">SloughGPT</option>
                  <option value="hf/gpt2-medium">GPT-2 Medium</option>
                </select>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="def-temp">Temperature — {settings.defaultTemp}</Label>
                  <input
                    id="def-temp"
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={settings.defaultTemp}
                    onChange={(e) => setSettings({ ...settings, defaultTemp: parseFloat(e.target.value) })}
                    className="h-2 w-full cursor-pointer accent-primary"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="def-max">Max tokens — {settings.defaultMaxTokens}</Label>
                  <input
                    id="def-max"
                    type="range"
                    min="50"
                    max="1000"
                    step="50"
                    value={settings.defaultMaxTokens}
                    onChange={(e) => setSettings({ ...settings, defaultMaxTokens: parseInt(e.target.value, 10) })}
                    className="h-2 w-full cursor-pointer accent-primary"
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="data" className="space-y-6 focus-visible:outline-none">
          <Card>
            <CardHeader>
              <CardTitle>Chat data</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Clears browser-stored chat sessions. This cannot be undone.
              </p>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button type="button" variant="destructive" className="w-full sm:w-auto">
                    Clear chat history
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Clear all chat history?</AlertDialogTitle>
                    <AlertDialogDescription>
                      Removes saved conversations from this browser (localStorage).
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction
                      onClick={clearChat}
                      className="bg-destructive text-destructive-foreground hover:opacity-90"
                    >
                      Clear
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <Separator className="my-8" />

      <Button type="button" className="w-full" onClick={saveSettings}>
        Save settings
      </Button>
    </div>
  )
}
