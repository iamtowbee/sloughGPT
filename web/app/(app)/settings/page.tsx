'use client'

import { useState, useEffect } from 'react'

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
    apiUrl: 'http://localhost:8000',
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
    <div className="p-6 max-w-2xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold text-white">Settings</h1>
        {saved && (
          <span className="text-green-400 text-sm">Saved!</span>
        )}
      </div>

      <div className="space-y-6">
        <div className="bg-white/5 border border-white/10 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">API Configuration</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-zinc-400 mb-1">API URL</label>
              <input
                type="text"
                value={settings.apiUrl}
                onChange={(e) => setSettings({ ...settings, apiUrl: e.target.value })}
                className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-white"
              />
            </div>
            <div>
              <label className="block text-sm text-zinc-400 mb-1">HuggingFace Token</label>
              <input
                type="password"
                value={settings.hfToken}
                onChange={(e) => setSettings({ ...settings, hfToken: e.target.value })}
                placeholder="hf_..."
                className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-white"
              />
            </div>
          </div>
        </div>

        <div className="bg-white/5 border border-white/10 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Model Defaults</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-zinc-400 mb-1">Default Model</label>
              <select
                value={settings.defaultModel}
                onChange={(e) => setSettings({ ...settings, defaultModel: e.target.value })}
                className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-white"
              >
                <option value="gpt2">GPT-2</option>
                <option value="sloughgpt">SloughGPT</option>
                <option value="hf/gpt2-medium">GPT-2 Medium</option>
              </select>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-zinc-400 mb-1">Temperature: {settings.defaultTemp}</label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={settings.defaultTemp}
                  onChange={(e) => setSettings({ ...settings, defaultTemp: parseFloat(e.target.value) })}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm text-zinc-400 mb-1">Max Tokens: {settings.defaultMaxTokens}</label>
                <input
                  type="range"
                  min="50"
                  max="1000"
                  step="50"
                  value={settings.defaultMaxTokens}
                  onChange={(e) => setSettings({ ...settings, defaultMaxTokens: parseInt(e.target.value) })}
                  className="w-full"
                />
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white/5 border border-white/10 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">UI Settings</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-white">Streaming Responses</p>
                <p className="text-sm text-zinc-500">Stream tokens as they generate</p>
              </div>
              <button
                onClick={() => setSettings({ ...settings, streaming: !settings.streaming })}
                className={`w-12 h-6 rounded-full transition-colors ${
                  settings.streaming ? 'bg-blue-600' : 'bg-white/10'
                }`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                  settings.streaming ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </div>
          </div>
        </div>

        <div className="bg-white/5 border border-white/10 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Data</h2>
          <div className="space-y-3">
            <button
              onClick={clearChat}
              className="w-full bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded-lg px-4 py-2 text-left"
            >
              Clear Chat History
            </button>
            <p className="text-sm text-zinc-500">
              Stored in localStorage. Clearing will remove all saved conversations.
            </p>
          </div>
        </div>

        <button
          onClick={saveSettings}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-4 py-3 font-medium"
        >
          Save Settings
        </button>
      </div>
    </div>
  )
}
