'use client'

import { useState, useEffect } from 'react'

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
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { Textarea } from '@/components/ui/textarea'

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
    <span className="flex h-9 w-9 shrink-0 items-center justify-center border border-primary/30 bg-primary/15 font-mono text-sm font-semibold text-primary">
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
    setIsEditing(false)
  }

  const deleteAgent = (id: string) => {
    saveAgents(agents.filter((a) => a.id !== id))
    if (selectedAgent?.id === id) {
      setSelectedAgent(null)
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain px-2">
      <AppRouteHeader
        className="border-b border-border py-3"
        left={
          <AppRouteHeaderLead
            title={
              <div className="flex min-w-0 items-center gap-3">
                <h1 className="text-lg font-semibold text-foreground">Agents</h1>
                <span className="font-mono text-xs text-muted-foreground">{agents.length} agents</span>
              </div>
            }
          />
        }
        right={
          <Button
            type="button"
            size="sm"
            onClick={() => {
              setIsEditing(true)
              setSelectedAgent(null)
            }}
          >
            + New agent
          </Button>
        }
      />

      <div className="flex flex-1 gap-4 overflow-hidden py-4">
        <div className="w-64 shrink-0 space-y-2 overflow-y-auto">
          {agents.map((agent) => (
            <Button
              key={agent.id}
              type="button"
              variant={selectedAgent?.id === agent.id ? 'secondary' : 'ghost'}
              className={`h-auto w-full justify-start border px-3 py-3 ${
                selectedAgent?.id === agent.id ? 'border-primary/30 bg-primary/10' : 'border-transparent'
              }`}
              onClick={() => {
                setSelectedAgent(agent)
                setIsEditing(false)
              }}
            >
              <div className="flex items-center gap-2 text-left">
                <AgentAvatar label={agent.avatar || agent.name} />
                <div>
                  <div className="text-sm font-medium text-foreground">{agent.name}</div>
                  <div className="line-clamp-1 text-xs text-muted-foreground">{agent.description}</div>
                </div>
              </div>
            </Button>
          ))}
        </div>

        <Card className="min-w-0 flex-1 overflow-y-auto">
          <CardContent className="p-6">
            {isEditing && !selectedAgent ? (
              <div className="space-y-4">
                <CardTitle className="text-lg">Create agent</CardTitle>
                <div className="space-y-2">
                  <Label htmlFor="agent-name">Name</Label>
                  <Input
                    id="agent-name"
                    value={editForm.name}
                    onChange={(e) => setEditForm({ ...editForm, name: e.target.value })}
                    placeholder="My agent"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="agent-desc">Description</Label>
                  <Input
                    id="agent-desc"
                    value={editForm.description}
                    onChange={(e) => setEditForm({ ...editForm, description: e.target.value })}
                    placeholder="What does this agent do?"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="agent-inst">Instructions</Label>
                  <Textarea
                    id="agent-inst"
                    value={editForm.instructions}
                    onChange={(e) => setEditForm({ ...editForm, instructions: e.target.value })}
                    placeholder="Agent behavior instructions…"
                    rows={4}
                  />
                </div>
                <div className="flex gap-2">
                  <Button type="button" onClick={createAgent}>
                    Create
                  </Button>
                  <Button type="button" variant="secondary" onClick={() => setIsEditing(false)}>
                    Cancel
                  </Button>
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
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button type="button" variant="ghost" size="icon" className="text-destructive" aria-label="Delete agent">
                        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                          />
                        </svg>
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Delete agent?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This removes <strong>{selectedAgent.name}</strong> from local storage.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          className="bg-destructive text-destructive-foreground hover:opacity-90"
                          onClick={() => deleteAgent(selectedAgent.id)}
                        >
                          Delete
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
                <Separator className="my-4" />
                <div className="mb-4">
                  <h3 className="mb-2 font-mono text-xs uppercase tracking-wider text-muted-foreground">Instructions</h3>
                  <p className="border border-border bg-muted/30 p-3 text-sm text-foreground">{selectedAgent.instructions}</p>
                </div>
                <div>
                  <h3 className="mb-2 font-mono text-xs uppercase tracking-wider text-muted-foreground">Tools</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedAgent.tools.map((tool) => (
                      <span
                        key={tool}
                        className="border border-primary/25 bg-primary/10 px-2 py-1 text-xs text-primary"
                      >
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
              <div className="flex h-full min-h-[200px] items-center justify-center text-muted-foreground">
                <div className="text-center">
                  <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center border border-border bg-muted font-mono text-xl text-primary">
                    A
                  </div>
                  <p className="text-sm">Select an agent or create a new one</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
