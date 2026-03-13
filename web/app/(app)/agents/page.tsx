'use client'

import { useState, useEffect } from 'react'
import { useTheme } from '@/components/ThemeProvider'

interface Agent {
  id: string
  name: string
  description: string
  instructions: string
  tools: string[]
  avatar?: string
}

const DEFAULT_AGENTS: Agent[] = [
  {
    id: 'coder',
    name: 'Code Assistant',
    description: 'Expert programmer',
    instructions: 'You are a helpful coding assistant.',
    tools: ['code_execution', 'file_search'],
    avatar: '💻',
  },
  {
    id: 'writer',
    name: 'Writer',
    description: 'Creative writing',
    instructions: 'You are a creative writing assistant.',
    tools: ['web_search'],
    avatar: '✍️',
  },
  {
    id: 'researcher',
    name: 'Researcher',
    description: 'Research assistant',
    instructions: 'You are a research assistant.',
    tools: ['web_search', 'citation'],
    avatar: '🔬',
  },
]

export default function AgentsPage() {
  const { theme } = useTheme()
  const [agents, setAgents] = useState<Agent[]>([])
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null)
  const [isEditing, setIsEditing] = useState(false)
  
  const [editForm, setEditForm] = useState({
    name: '',
    description: '',
    instructions: '',
  })

  const themeColors: Record<string, string> = {
    blue: 'from-blue-500 to-cyan-400',
    purple: 'from-purple-500 to-pink-400',
    pink: 'from-pink-500 to-rose-400',
    red: 'from-red-500 to-orange-400',
    orange: 'from-orange-500 to-yellow-400',
    green: 'from-green-500 to-emerald-400',
    teal: 'from-teal-500 to-cyan-400',
  }

  useEffect(() => {
    const saved = localStorage.getItem('sloughgpt_agents')
    if (saved) {
      try {
        setAgents(JSON.parse(saved))
      } catch {
        setAgents(DEFAULT_AGENTS)
      }
    } else {
      setAgents(DEFAULT_AGENTS)
    }
  }, [])

  const saveAgents = (newAgents: Agent[]) => {
    setAgents(newAgents)
    localStorage.setItem('sloughgpt_agents', JSON.stringify(newAgents))
  }

  const createAgent = () => {
    const newAgent: Agent = {
      id: `agent-${Date.now()}`,
      name: editForm.name || 'New Agent',
      description: editForm.description || 'Custom agent',
      instructions: editForm.instructions || 'You are a helpful assistant.',
      tools: [],
      avatar: '🤖',
    }
    saveAgents([...agents, newAgent])
    setEditForm({ name: '', description: '', instructions: '' })
    setSelectedAgent(newAgent)
  }

  const deleteAgent = (id: string) => {
    saveAgents(agents.filter(a => a.id !== id))
    if (selectedAgent?.id === id) {
      setSelectedAgent(null)
    }
  }

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between py-3 border-b border-white/5">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold text-white">Agents</h1>
          <span className="text-xs text-zinc-500">{agents.length} agents</span>
        </div>
        <button
          onClick={() => { setIsEditing(true); setSelectedAgent(null) }}
          className={`px-3 py-1.5 bg-gradient-to-r ${themeColors[theme]} text-white text-sm rounded-lg`}
        >
          + New Agent
        </button>
      </div>

      <div className="flex-1 flex gap-4 py-4 overflow-hidden">
        {/* Agents List */}
        <div className="w-64 space-y-2 overflow-y-auto">
          {agents.map(agent => (
            <button
              key={agent.id}
              onClick={() => { setSelectedAgent(agent); setIsEditing(false) }}
              className={`w-full text-left p-3 rounded-xl transition-all ${
                selectedAgent?.id === agent.id 
                  ? `bg-gradient-to-r ${themeColors[theme]} bg-opacity-20 border border-white/10` 
                  : 'bg-white/5 hover:bg-white/10 border border-transparent'
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="text-xl">{agent.avatar}</span>
                <div>
                  <div className="font-medium text-white text-sm">{agent.name}</div>
                  <div className="text-xs text-zinc-500">{agent.description}</div>
                </div>
              </div>
            </button>
          ))}
        </div>

        {/* Agent Details / Create */}
        <div className="flex-1 bg-white/5 rounded-xl border border-white/5 p-4 overflow-y-auto">
          {isEditing && !selectedAgent ? (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold text-white">Create New Agent</h2>
              <div>
                <label className="block text-xs text-zinc-500 mb-1">Name</label>
                <input
                  type="text"
                  value={editForm.name}
                  onChange={e => setEditForm({ ...editForm, name: e.target.value })}
                  placeholder="My Agent"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-zinc-500 mb-1">Description</label>
                <input
                  type="text"
                  value={editForm.description}
                  onChange={e => setEditForm({ ...editForm, description: e.target.value })}
                  placeholder="What does this agent do?"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-zinc-500 mb-1">Instructions</label>
                <textarea
                  value={editForm.instructions}
                  onChange={e => setEditForm({ ...editForm, instructions: e.target.value })}
                  placeholder="Agent behavior instructions..."
                  rows={4}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm resize-none"
                />
              </div>
              <div className="flex gap-2">
                <button
                  onClick={createAgent}
                  className={`px-4 py-2 bg-gradient-to-r ${themeColors[theme]} text-white text-sm rounded-lg`}
                >
                  Create
                </button>
                <button
                  onClick={() => setIsEditing(false)}
                  className="px-4 py-2 bg-white/10 text-white text-sm rounded-lg"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : selectedAgent ? (
            <div>
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <span className="text-3xl">{selectedAgent.avatar}</span>
                  <div>
                    <h2 className="text-lg font-semibold text-white">{selectedAgent.name}</h2>
                    <p className="text-sm text-zinc-500">{selectedAgent.description}</p>
                  </div>
                </div>
                <button
                  onClick={() => deleteAgent(selectedAgent.id)}
                  className="p-2 text-zinc-500 hover:text-red-400"
                >
                  🗑️
                </button>
              </div>
              <div className="mb-4">
                <h3 className="text-xs text-zinc-500 mb-2">Instructions</h3>
                <p className="text-sm text-zinc-300 bg-white/5 p-3 rounded-lg">{selectedAgent.instructions}</p>
              </div>
              <div>
                <h3 className="text-xs text-zinc-500 mb-2">Tools</h3>
                <div className="flex flex-wrap gap-2">
                  {selectedAgent.tools.map(tool => (
                    <span key={tool} className={`px-2 py-1 bg-gradient-to-r ${themeColors[theme]} bg-opacity-20 text-xs rounded`}>
                      {tool}
                    </span>
                  ))}
                  {selectedAgent.tools.length === 0 && (
                    <span className="text-xs text-zinc-500">No tools enabled</span>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-zinc-500">
              <div className="text-center">
                <div className="text-4xl mb-2">🤖</div>
                <p>Select an agent or create new</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
