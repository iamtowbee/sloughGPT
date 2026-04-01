'use client'

import { useState, useEffect } from 'react'

import { api, Experiment, Run } from '@/lib/api'

function statusBadge(status: string) {
  if (status === 'running') return 'bg-primary/20 text-primary'
  if (status === 'completed') return 'bg-success/20 text-success'
  return 'bg-destructive/20 text-destructive'
}

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedExp, setSelectedExp] = useState<string | null>(null)
  const [runs, setRuns] = useState<Run[]>([])

  useEffect(() => {
    api
      .getExperiments()
      .then(setExperiments)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (selectedExp) {
      api.getRuns(selectedExp).then(setRuns).catch(console.error)
    }
  }, [selectedExp])

  return (
    <div className="sl-page max-w-6xl mx-auto">
      <h1 className="sl-h1 mb-6">Experiments</h1>

      {loading ? (
        <div className="py-8 text-center text-muted-foreground">Loading...</div>
      ) : experiments.length === 0 ? (
        <div className="py-8 text-center text-muted-foreground">No experiments yet. Start a training job to create one.</div>
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="sl-card-solid overflow-hidden">
            <div className="border-b border-border p-4">
              <h2 className="font-semibold text-foreground">Experiments</h2>
            </div>
            <div className="divide-y divide-border">
              {experiments.map((exp) => (
                <button
                  key={exp.id}
                  type="button"
                  onClick={() => setSelectedExp(exp.id)}
                  className={`w-full p-4 text-left transition-colors hover:bg-muted/40 ${
                    selectedExp === exp.id ? 'bg-primary/10' : ''
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <h3 className="font-medium text-foreground">{exp.name}</h3>
                      <p className="font-mono text-xs text-muted-foreground">{exp.id}</p>
                    </div>
                    <span className={`shrink-0 rounded px-2 py-1 text-xs font-medium ${statusBadge(exp.status)}`}>
                      {exp.status}
                    </span>
                  </div>
                  <div className="mt-2 text-sm text-muted-foreground">
                    Created: {new Date(exp.created_at).toLocaleString()}
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="sl-card-solid overflow-hidden">
            <div className="border-b border-border p-4">
              <h2 className="font-semibold text-foreground">Runs</h2>
            </div>
            {selectedExp ? (
              runs.length > 0 ? (
                <div className="divide-y divide-border">
                  {runs.map((run) => (
                    <div key={run.id} className="p-4">
                      <div className="mb-2 flex items-start justify-between">
                        <h3 className="font-medium font-mono text-sm text-foreground">{run.id}</h3>
                        <span className={`rounded px-2 py-1 text-xs font-medium ${statusBadge(run.status)}`}>
                          {run.status}
                        </span>
                      </div>

                      {Object.keys(run.metrics).length > 0 && (
                        <div className="mt-3 grid grid-cols-2 gap-2">
                          {Object.entries(run.metrics).map(([key, value]) => (
                            <div key={key} className="text-sm">
                              <span className="text-muted-foreground">{key}: </span>
                              <span className="font-medium text-foreground">
                                {typeof value === 'number' ? value.toFixed(4) : String(value)}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}

                      {run.params && Object.keys(run.params).length > 0 && (
                        <div className="mt-3 text-sm">
                          <p className="mb-1 text-muted-foreground">Parameters:</p>
                          <div className="flex flex-wrap gap-2">
                            {Object.entries(run.params).map(([key, value]) => (
                              <span key={key} className="rounded border border-border bg-muted/50 px-2 py-1 text-xs text-foreground">
                                {key}: {String(value)}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-4 text-center text-muted-foreground">No runs yet</div>
              )
            ) : (
              <div className="p-4 text-center text-muted-foreground">Select an experiment to view runs</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
