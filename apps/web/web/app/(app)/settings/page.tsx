'use client'

import { useState, useEffect } from 'react'

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
    theme: 'dark',
    streaming: true,
  })
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    const stored = localStorage.getItem('sloughgpt_settings')
    if (stored) {
      setSettings(JSON.parse(stored))
    }
  }, [])

  const saveSettings = () => {
    localStorage.setItem('sloughgpt_settings', JSON.stringify(settings))
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const clearChat = () => {
    if (confirm('Clear all chat history?')) {
      localStorage.removeItem('sloughgpt_messages')
    }
  }

  return (
    <div className="sl-page max-w-2xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="sl-h1">Settings</h1>
        {saved && <span className="text-success text-sm font-medium">Saved!</span>}
      </div>

      <div className="space-y-6">
        <div className="sl-card p-6">
          <h2 className="sl-h2 mb-4">API Configuration</h2>
          <div className="space-y-4">
            <div>
              <label className="sl-label normal-case tracking-normal">API URL</label>
              <input
                type="text"
                value={settings.apiUrl}
                onChange={(e) => setSettings({ ...settings, apiUrl: e.target.value })}
                className="sl-input"
              />
            </div>
            <div>
              <label className="sl-label normal-case tracking-normal">HuggingFace Token</label>
              <input
                type="password"
                value={settings.hfToken}
                onChange={(e) => setSettings({ ...settings, hfToken: e.target.value })}
                placeholder="hf_..."
                className="sl-input"
              />
            </div>
          </div>
        </div>

        <div className="sl-card p-6">
          <h2 className="sl-h2 mb-4">Model Defaults</h2>
          <div className="space-y-4">
            <div>
              <label className="sl-label normal-case tracking-normal">Default Model</label>
              <select
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
              <div>
                <label className="sl-label normal-case tracking-normal">Temperature: {settings.defaultTemp}</label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={settings.defaultTemp}
                  onChange={(e) => setSettings({ ...settings, defaultTemp: parseFloat(e.target.value) })}
                  className="w-full accent-primary"
                />
              </div>
              <div>
                <label className="sl-label normal-case tracking-normal">Max Tokens: {settings.defaultMaxTokens}</label>
                <input
                  type="range"
                  min="50"
                  max="1000"
                  step="50"
                  value={settings.defaultMaxTokens}
                  onChange={(e) => setSettings({ ...settings, defaultMaxTokens: parseInt(e.target.value) })}
                  className="w-full accent-primary"
                />
              </div>
            </div>
          </div>
        </div>

        <div className="sl-card p-6">
          <h2 className="sl-h2 mb-4">UI Settings</h2>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-foreground font-medium">Streaming Responses</p>
              <p className="text-sm text-muted-foreground">Stream tokens as they generate</p>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={settings.streaming}
              onClick={() => setSettings({ ...settings, streaming: !settings.streaming })}
              className={`w-12 h-6 rounded-full transition-colors relative ${
                settings.streaming ? 'bg-primary' : 'bg-muted'
              }`}
            >
              <span
                className={`absolute top-0.5 w-5 h-5 rounded-full bg-primary-foreground shadow transition-transform ${
                  settings.streaming ? 'translate-x-6' : 'translate-x-0.5'
                }`}
              />
            </button>
          </div>
        </div>

        <div className="sl-card p-6">
          <h2 className="sl-h2 mb-4">Data</h2>
          <div className="space-y-3">
            <button
              type="button"
              onClick={clearChat}
              className="w-full rounded-lg border border-destructive/40 bg-destructive/10 text-destructive hover:bg-destructive/20 px-4 py-2 text-left text-sm font-medium"
            >
              Clear Chat History
            </button>
            <p className="text-sm text-muted-foreground">
              Stored in localStorage. Clearing will remove all saved conversations.
            </p>
          </div>
        </div>

        <button type="button" onClick={saveSettings} className="w-full sl-btn-primary py-3 rounded-lg">
          Save Settings
        </button>
      </div>
    </div>
  )
}
