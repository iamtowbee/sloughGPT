'use client'

import { useState, useEffect } from 'react'
import { api, Experiment, Run } from '@/lib/api'

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedExp, setSelectedExp] = useState<string | null>(null)
  const [runs, setRuns] = useState<Run[]>([])

  useEffect(() => {
    api.getExperiments()
      .then(setExperiments)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (selectedExp) {
      api.getRuns(selectedExp)
        .then(setRuns)
        .catch(console.error)
    }
  }, [selectedExp])

  return (
    <div>
      <h1 className="text-3xl font-bold text-slate-800 dark:text-white mb-6">Experiments</h1>

      {loading ? (
        <div className="text-center py-8 text-slate-500">Loading...</div>
      ) : experiments.length === 0 ? (
        <div className="text-center py-8 text-slate-500">
          No experiments yet. Start a training job to create one.
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
            <div className="p-4 border-b border-slate-200 dark:border-slate-700">
              <h2 className="font-semibold text-slate-800 dark:text-white">Experiments</h2>
            </div>
            <div className="divide-y divide-slate-200 dark:divide-slate-700">
              {experiments.map((exp) => (
                <div
                  key={exp.id}
                  onClick={() => setSelectedExp(exp.id)}
                  className={`p-4 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-700 ${
                    selectedExp === exp.id ? 'bg-blue-50 dark:bg-blue-900/20' : ''
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-medium text-slate-800 dark:text-white">{exp.name}</h3>
                      <p className="text-sm text-slate-500">{exp.id}</p>
                    </div>
                    <span className={`px-2 py-1 rounded text-xs ${
                      exp.status === 'running' ? 'bg-blue-100 text-blue-700' :
                      exp.status === 'completed' ? 'bg-green-100 text-green-700' :
                      'bg-red-100 text-red-700'
                    }`}>
                      {exp.status}
                    </span>
                  </div>
                  <div className="mt-2 text-sm text-slate-500">
                    Created: {new Date(exp.created_at).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
            <div className="p-4 border-b border-slate-200 dark:border-slate-700">
              <h2 className="font-semibold text-slate-800 dark:text-white">Runs</h2>
            </div>
            {selectedExp ? (
              runs.length > 0 ? (
                <div className="divide-y divide-slate-200 dark:divide-slate-700">
                  {runs.map((run) => (
                    <div key={run.id} className="p-4">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <h3 className="font-medium text-slate-800 dark:text-white">{run.id}</h3>
                        </div>
                        <span className={`px-2 py-1 rounded text-xs ${
                          run.status === 'running' ? 'bg-blue-100 text-blue-700' :
                          run.status === 'completed' ? 'bg-green-100 text-green-700' :
                          'bg-red-100 text-red-700'
                        }`}>
                          {run.status}
                        </span>
                      </div>
                      
                      {Object.keys(run.metrics).length > 0 && (
                        <div className="mt-3 grid grid-cols-2 gap-2">
                          {Object.entries(run.metrics).map(([key, value]) => (
                            <div key={key} className="text-sm">
                              <span className="text-slate-500">{key}: </span>
                              <span className="text-slate-800 dark:text-white font-medium">
                                {typeof value === 'number' ? value.toFixed(4) : String(value)}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                      
                      {run.params && Object.keys(run.params).length > 0 && (
                        <div className="mt-3 text-sm">
                          <p className="text-slate-500 mb-1">Parameters:</p>
                          <div className="flex flex-wrap gap-2">
                            {Object.entries(run.params).map(([key, value]) => (
                              <span key={key} className="bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded text-xs">
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
                <div className="p-4 text-center text-slate-500">No runs yet</div>
              )
            ) : (
              <div className="p-4 text-center text-slate-500">Select an experiment to view runs</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
