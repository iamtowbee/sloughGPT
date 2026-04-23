'use client'

import { useState } from 'react'

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
import { Textarea } from '@/components/ui/textarea'
import { useSettings, useUpdateSettings } from '@/lib/store'

export default function SettingsPage() {
  const settings = useSettings()
  const updateSettings = useUpdateSettings()
  const [saved, setSaved] = useState(false)

  const saveSettings = () => {
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const clearChat = () => {
    localStorage.removeItem('sloughgpt_messages')
    localStorage.removeItem('sloughgpt_chat_sessions')
    localStorage.removeItem('sloughgpt_current_session')
    alert('Chat history cleared!')
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
          <TabsTrigger value="context">Context</TabsTrigger>
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
                  onChange={(e) => updateSettings({ apiUrl: e.target.value })}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="hf-token">Hugging Face token</Label>
                <Input
                  id="hf-token"
                  type="password"
                  value={settings.hfToken}
                  onChange={(e) => updateSettings({ hfToken: e.target.value })}
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
                  onCheckedChange={(streaming) => updateSettings({ streaming })}
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
                  onChange={(e) => updateSettings({ defaultModel: e.target.value })}
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
                    onChange={(e) => updateSettings({ defaultTemp: parseFloat(e.target.value) })}
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
                    onChange={(e) => updateSettings({ defaultMaxTokens: parseInt(e.target.value, 10) })}
                    className="h-2 w-full cursor-pointer accent-primary"
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="context" className="space-y-6 focus-visible:outline-none">
          <Card>
            <CardHeader>
              <CardTitle>Custom Context</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Add facts, instructions, or context that will be included with every prompt.
              </p>
              <Textarea
                className="min-h-[150px]"
                placeholder="e.g., You are a helpful coding assistant. Keep responses concise..."
                value={settings.customContext}
                onChange={(e) => updateSettings({ customContext: e.target.value })}
              />
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