'use client'

import { useState, useEffect } from 'react'

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
    avatar: 'C',
  },
  {
    id: 'writer',
    name: 'Writer',
    description: 'Creative writing',
    instructions: 'You are a creative writing assistant.',
    tools: ['web_search'],
    avatar: 'W',
  },
  {
    id: 'researcher',
    name: 'Researcher',
    description: 'Research assistant',
    instructions: 'You are a research assistant.',
    tools: ['web_search', 'citation'],
    avatar: 'R',
  },
]

function AgentAvatar({ label }: { label: string }) {
  const ch = (label && label[0]) || 'A'
  return (
    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/20 text-primary text-sm font-semibold font-mono ring-1 ring-primary/30">
      {ch.toUpperCase()}
    </span>
  )
}

export default function AgentsPage() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null)
  const [isEditing, setIsEditing] = useState(false)

  const [editForm, setEditForm] = useState({
    name: '',
    description: '',
    instructions: '',
  })

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
    const label = editForm.name || 'New Agent'
    const newAgent: Agent = {
      id: `agent-${Date.now()}`,
      name: label,
      description: editForm.description || 'Custom agent',
      instructions: editForm.instructions || 'You are a helpful assistant.',
      tools: [],
      avatar: label[0] || 'A',
    }
    saveAgents([...agents, newAgent])
    setEditForm({ name: '', description: '', instructions: '' })
    setSelectedAgent(newAgent)
  }

  const deleteAgent = (id: string) => {
    saveAgents(agents.filter((a) => a.id !== id))
    if (selectedAgent?.id === id) {
      setSelectedAgent(null)
    }
  }

  return (
    <div className="flex h-[calc(100vh-0px)] flex-col px-2">
      <div className="flex items-center justify-between border-b border-border py-3">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold text-foreground">Agents</h1>
          <span className="text-xs text-muted-foreground font-mono">{agents.length} agents</span>
        </div>
        <button
          type="button"
          onClick={() => {
            setIsEditing(true)
            setSelectedAgent(null)
          }}
          className="sl-btn-primary rounded-lg px-3 py-1.5 text-sm"
        >
          + New Agent
        </button>
      </div>

      <div className="flex flex-1 gap-4 overflow-hidden py-4">
        <div className="w-64 shrink-0 space-y-2 overflow-y-auto">
          {agents.map((agent) => (
            <button
              key={agent.id}
              type="button"
              onClick={() => {
                setSelectedAgent(agent)
                setIsEditing(false)
              }}
              className={`w-full rounded-xl border p-3 text-left transition-colors ${
                selectedAgent?.id === agent.id
                  ? 'border-primary/40 bg-primary/10'
                  : 'border-transparent bg-muted/30 hover:bg-muted/50'
              }`}
            >
              <div className="flex items-center gap-2">
                <AgentAvatar label={agent.avatar || agent.name} />
                <div>
                  <div className="text-sm font-medium text-foreground">{agent.name}</div>
                  <div className="text-xs text-muted-foreground line-clamp-1">{agent.description}</div>
                </div>
              </div>
            </button>
          ))}
        </div>

        <div className="sl-card-solid min-w-0 flex-1 overflow-y-auto border border-border p-4">
          {isEditing && !selectedAgent ? (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold text-foreground">Create New Agent</h2>
              <div>
                <label className="sl-label normal-case tracking-normal">Name</label>
                <input
                  type="text"
                  value={editForm.name}
                  onChange={(e) => setEditForm({ ...editForm, name: e.target.value })}
                  placeholder="My Agent"
                  className="sl-input"
                />
              </div>
              <div>
                <label className="sl-label normal-case tracking-normal">Description</label>
                <input
                  type="text"
                  value={editForm.description}
                  onChange={(e) => setEditForm({ ...editForm, description: e.target.value })}
                  placeholder="What does this agent do?"
                  className="sl-input"
                />
              </div>
              <div>
                <label className="sl-label normal-case tracking-normal">Instructions</label>
                <textarea
                  value={editForm.instructions}
                  onChange={(e) => setEditForm({ ...editForm, instructions: e.target.value })}
                  placeholder="Agent behavior instructions..."
                  rows={4}
                  className="sl-input resize-none"
                />
              </div>
              <div className="flex gap-2">
                <button type="button" onClick={createAgent} className="sl-btn-primary rounded-lg px-4 py-2 text-sm">
                  Create
                </button>
                <button
                  type="button"
                  onClick={() => setIsEditing(false)}
                  className="sl-btn-secondary rounded-lg px-4 py-2 text-sm"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : selectedAgent ? (
            <div>
              <div className="mb-4 flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <AgentAvatar label={selectedAgent.avatar || selectedAgent.name} />
                  <div>
                    <h2 className="text-lg font-semibold text-foreground">{selectedAgent.name}</h2>
                    <p className="text-sm text-muted-foreground">{selectedAgent.description}</p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => deleteAgent(selectedAgent.id)}
                  className="sl-btn-ghost p-2 text-destructive hover:text-destructive"
                  aria-label="Delete agent"
                >
                  <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
              <div className="mb-4">
                <h3 className="mb-2 text-xs font-mono uppercase tracking-wider text-muted-foreground">Instructions</h3>
                <p className="rounded-lg border border-border bg-muted/30 p-3 text-sm text-foreground">{selectedAgent.instructions}</p>
              </div>
              <div>
                <h3 className="mb-2 text-xs font-mono uppercase tracking-wider text-muted-foreground">Tools</h3>
                <div className="flex flex-wrap gap-2">
                  {selectedAgent.tools.map((tool) => (
                    <span key={tool} className="rounded-md border border-primary/25 bg-primary/10 px-2 py-1 text-xs text-primary">
                      {tool}
                    </span>
                  ))}
                  {selectedAgent.tools.length === 0 && (
                    <span className="text-xs text-muted-foreground">No tools enabled</span>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex h-full items-center justify-center text-muted-foreground">
              <div className="text-center">
                <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-muted font-mono text-xl text-primary">
                  A
                </div>
                <p className="text-sm">Select an agent or create new</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
