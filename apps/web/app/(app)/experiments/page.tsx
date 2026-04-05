'use client'

import { useState, useEffect } from 'react'

import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { api, Experiment, Run } from '@/lib/api'
import { devDebug } from '@/lib/dev-log'

function statusClass(status: string) {
  if (status === 'running') return 'border-primary/30 bg-primary/10 text-primary'
  if (status === 'completed') return 'border-success/30 bg-success/10 text-success'
  return 'border-destructive/30 bg-destructive/10 text-destructive'
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
      .catch((e) => devDebug('getExperiments:', e))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (selectedExp) {
      api.getRuns(selectedExp).then(setRuns).catch((e) => devDebug('getRuns:', e))
    }
  }, [selectedExp])

  return (
    <div className="sl-page mx-auto max-w-6xl">
      <AppRouteHeader className="mb-6 items-start" left={<AppRouteHeaderLead title="Experiments" />} />

      {loading ? (
        <p className="py-8 text-center text-muted-foreground">Loading…</p>
      ) : experiments.length === 0 ? (
        <Card className="border-dashed">
          <CardContent className="py-12 text-center text-muted-foreground">
            No experiments yet. Start a training job to create one.
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Card className="overflow-hidden">
            <CardHeader className="border-b border-border">
              <CardTitle className="text-base">Experiments</CardTitle>
              <CardDescription>Select an experiment to load runs</CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <div className="divide-y divide-border">
                {experiments.map((exp) => (
                  <button
                    key={exp.id}
                    type="button"
                    onClick={() => setSelectedExp(exp.id)}
                    className={`w-full p-4 text-left transition-colors duration-200 hover:bg-muted/40 ${
                      selectedExp === exp.id ? 'bg-primary/10' : ''
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <h3 className="font-medium text-foreground">{exp.name}</h3>
                        <p className="font-mono text-xs text-muted-foreground">{exp.id}</p>
                      </div>
                      <span className={`shrink-0 border px-2 py-1 text-xs font-medium ${statusClass(exp.status)}`}>
                        {exp.status}
                      </span>
                    </div>
                    <p className="mt-2 text-sm text-muted-foreground">
                      Created: {new Date(exp.created_at).toLocaleString()}
                    </p>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="overflow-hidden">
            <CardHeader className="border-b border-border">
              <CardTitle className="text-base">Runs</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {selectedExp ? (
                runs.length > 0 ? (
                  <div className="divide-y divide-border">
                    {runs.map((run) => (
                      <div key={run.id} className="p-4">
                        <div className="mb-2 flex items-start justify-between">
                          <h3 className="font-mono text-sm font-medium text-foreground">{run.id}</h3>
                          <span className={`border px-2 py-1 text-xs font-medium ${statusClass(run.status)}`}>
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
                            <p className="mb-1 text-muted-foreground">Parameters</p>
                            <div className="flex flex-wrap gap-2">
                              {Object.entries(run.params).map(([key, value]) => (
                                <span
                                  key={key}
                                  className="border border-border bg-muted/50 px-2 py-1 text-xs text-foreground"
                                >
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
                  <p className="p-6 text-center text-muted-foreground">No runs yet</p>
                )
              ) : (
                <p className="p-6 text-center text-muted-foreground">Select an experiment to view runs</p>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
