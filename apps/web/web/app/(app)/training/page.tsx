'use client'

import { useState, useEffect } from 'react'
import { api, TrainingJob, TrainResolveResponse } from '@/lib/api'

type CorpusMode = 'folder' | 'manifest' | 'ref'

const initialForm = {
  name: '',
  model: 'gpt2',
  corpusMode: 'folder' as CorpusMode,
  dataset: 'openwebtext',
  manifest_uri: '',
  ref_dataset_id: '',
  ref_version: '',
  ref_manifest_uri: '',
  epochs: 3,
  batch_size: 8,
  learning_rate: 1e-5,
  n_embed: 128,
  n_layer: 4,
  n_head: 4,
  block_size: 128,
  maxStepsInput: '',
}

export default function TrainingPage() {
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [loading, setLoading] = useState(true)
  const [showModal, setShowModal] = useState(false)
  const [newJob, setNewJob] = useState(initialForm)
  const [starting, setStarting] = useState(false)
  const [resolving, setResolving] = useState(false)
  const [resolveResult, setResolveResult] = useState<TrainResolveResponse | null>(null)
  const [resolveError, setResolveError] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const fetchJobs = async () => {
    try {
      const data = await api.getTrainingJobs()
      setJobs(data)
    } catch (error) {
      console.error('Failed to fetch jobs:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchJobs()
    const interval = setInterval(fetchJobs, 5000)
    return () => clearInterval(interval)
  }, [])

  const buildResolveBody = () => {
    if (newJob.corpusMode === 'folder') {
      return { dataset: newJob.dataset }
    }
    if (newJob.corpusMode === 'manifest') {
      return { manifest_uri: newJob.manifest_uri.trim() }
    }
    return {
      dataset_ref: {
        dataset_id: newJob.ref_dataset_id.trim(),
        version: newJob.ref_version.trim(),
        manifest_uri: newJob.ref_manifest_uri.trim(),
      },
    }
  }

  const buildStartPayload = () => {
    const ms = newJob.maxStepsInput.trim()
    const base = {
      name: newJob.name,
      model: newJob.model,
      epochs: newJob.epochs,
      batch_size: newJob.batch_size,
      learning_rate: newJob.learning_rate,
      n_embed: newJob.n_embed,
      n_layer: newJob.n_layer,
      n_head: newJob.n_head,
      block_size: newJob.block_size,
      ...(ms !== '' && !Number.isNaN(parseInt(ms, 10)) ? { max_steps: Math.max(1, parseInt(ms, 10)) } : {}),
    }
    if (newJob.corpusMode === 'folder') {
      return { ...base, dataset: newJob.dataset }
    }
    if (newJob.corpusMode === 'manifest') {
      return { ...base, manifest_uri: newJob.manifest_uri.trim() }
    }
    return {
      ...base,
      dataset_ref: {
        dataset_id: newJob.ref_dataset_id.trim(),
        version: newJob.ref_version.trim(),
        manifest_uri: newJob.ref_manifest_uri.trim(),
      },
    }
  }

  const previewResolution = async () => {
    setResolving(true)
    setResolveError(null)
    setResolveResult(null)
    try {
      const body = buildResolveBody()
      const r = await api.resolveTrainingData(body)
      setResolveResult(r)
    } catch (e) {
      setResolveError(e instanceof Error ? e.message : 'Resolution failed')
    } finally {
      setResolving(false)
    }
  }

  const startTraining = async () => {
    if (!newJob.name.trim()) return

    setStarting(true)
    setResolveError(null)
    try {
      await api.startTraining(buildStartPayload())
      setShowModal(false)
      setNewJob(initialForm)
      setResolveResult(null)
      fetchJobs()
    } catch (error) {
      console.error('Failed to start training:', error)
      setResolveError(error instanceof Error ? error.message : 'Start failed')
    } finally {
      setStarting(false)
    }
  }

  const openModal = () => {
    setResolveResult(null)
    setResolveError(null)
    setShowAdvanced(false)
    setShowModal(true)
  }

  return (
    <div className="sl-page mx-auto max-w-5xl">
      <div className="flex justify-between items-center mb-6">
        <h1 className="sl-h1">Training</h1>
        <button type="button" onClick={openModal} className="sl-btn-primary rounded-lg px-4 py-2">
          New Training Job
        </button>
      </div>

      {loading ? (
        <div className="text-center py-8 text-muted-foreground">Loading...</div>
      ) : jobs.length === 0 ? (
        <div className="text-center py-8 text-muted-foreground">No training jobs. Create one to get started.</div>
      ) : (
        <div className="sl-card-solid overflow-hidden">
          <div className="p-4 border-b border-border">
            <h2 className="font-semibold text-foreground">Training Jobs</h2>
          </div>

          <div className="divide-y divide-border">
            {jobs.map((job) => (
              <div key={job.id} className="p-4">
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h3 className="font-medium text-foreground">{job.name}</h3>
                    <p className="text-sm text-muted-foreground">
                      {job.model} • {job.dataset}
                      {job.data_source && (
                        <span className="ml-2 text-xs uppercase tracking-wide text-muted-foreground">
                          ({job.data_source})
                        </span>
                      )}
                    </p>
                    {job.data_path && (
                      <p className="text-xs text-muted-foreground mt-1 font-mono break-all">
                        {job.data_path}
                      </p>
                    )}
                  </div>
                  <span
                    className={`px-2 py-1 rounded text-xs font-medium ${
                      job.status === 'running'
                        ? 'bg-primary/20 text-primary'
                        : job.status === 'completed'
                          ? 'bg-success/20 text-success'
                          : job.status === 'pending'
                            ? 'bg-warning/20 text-warning'
                            : job.status === 'failed'
                              ? 'bg-destructive/20 text-destructive'
                              : 'bg-muted text-muted-foreground'
                    }`}
                  >
                    {job.status}
                  </span>
                </div>

                <div className="mt-3">
                  <div className="flex justify-between text-sm text-muted-foreground mb-1">
                    <span>Progress {job.current_epoch ? `(Epoch ${job.current_epoch}/${job.epochs})` : ''}</span>
                    <span>{job.progress}%</span>
                  </div>
                  <div className="w-full bg-muted rounded-full h-2">
                    <div
                      className="bg-primary h-2 rounded-full transition-all"
                      style={{ width: `${job.progress}%` }}
                    />
                  </div>
                </div>

                {job.loss != null && Number.isFinite(job.loss) && (
                  <div className="flex gap-4 mt-3 text-sm text-muted-foreground">
                    <span>Loss: {job.loss.toFixed(4)}</span>
                  </div>
                )}
                {job.checkpoint && (
                  <p className="text-xs text-muted-foreground mt-2 font-mono break-all">Checkpoint: {job.checkpoint}</p>
                )}
                {job.error && (
                  <p className="text-xs text-warning mt-2">{job.error}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {showModal && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="sl-card-solid p-6 w-full max-w-lg max-h-[90vh] overflow-y-auto border border-border shadow-lg">
            <h2 className="text-xl font-bold text-foreground mb-4">New Training Job</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-muted-foreground mb-1">
                  Job Name
                </label>
                <input
                  type="text"
                  value={newJob.name}
                  onChange={(e) => setNewJob({ ...newJob, name: e.target.value })}
                  placeholder="My Fine-tune"
                  className="sl-input"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-muted-foreground mb-1">
                  Base Model
                </label>
                <select
                  value={newJob.model}
                  onChange={(e) => setNewJob({ ...newJob, model: e.target.value })}
                  className="sl-input"
                >
                  <option value="gpt2">GPT-2</option>
                  <option value="llama-2-7b">Llama-2-7B</option>
                  <option value="mistral-7b">Mistral-7B</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-muted-foreground mb-1">
                  Training corpus
                </label>
                <select
                  value={newJob.corpusMode}
                  onChange={(e) =>
                    setNewJob({
                      ...newJob,
                      corpusMode: e.target.value as CorpusMode,
                      manifest_uri: '',
                      ref_dataset_id: '',
                      ref_version: '',
                      ref_manifest_uri: '',
                    })
                  }
                  className="sl-input"
                >
                  <option value="folder">Folder: datasets/&lt;name&gt;/input.txt</option>
                  <option value="manifest">v1 manifest (manifest_uri)</option>
                  <option value="ref">v1 dataset_ref (id + version + manifest)</option>
                </select>
              </div>

              {newJob.corpusMode === 'folder' && (
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1">
                    Dataset folder name
                  </label>
                  <select
                    value={newJob.dataset}
                    onChange={(e) => setNewJob({ ...newJob, dataset: e.target.value })}
                    className="sl-input"
                  >
                    <option value="openwebtext">OpenWebText</option>
                    <option value="wikitext-103">WikiText-103</option>
                    <option value="code-search-net">CodeSearchNet</option>
                  </select>
                  <p className="text-xs text-muted-foreground mt-1">API cwd must contain datasets/&lt;name&gt;/input.txt</p>
                </div>
              )}

              {newJob.corpusMode === 'manifest' && (
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1">
                    Path to dataset_manifest.json
                  </label>
                  <input
                    type="text"
                    value={newJob.manifest_uri}
                    onChange={(e) => setNewJob({ ...newJob, manifest_uri: e.target.value })}
                    placeholder="datasets/my_run/dataset_manifest.json"
                    className="sl-input font-mono text-sm"
                  />
                </div>
              )}

              {newJob.corpusMode === 'ref' && (
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-muted-foreground mb-1">
                      dataset_id (must match manifest)
                    </label>
                    <input
                      type="text"
                      value={newJob.ref_dataset_id}
                      onChange={(e) => setNewJob({ ...newJob, ref_dataset_id: e.target.value })}
                      className="sl-input font-mono text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-muted-foreground mb-1">
                      version (must match manifest)
                    </label>
                    <input
                      type="text"
                      value={newJob.ref_version}
                      onChange={(e) => setNewJob({ ...newJob, ref_version: e.target.value })}
                      className="sl-input font-mono text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-muted-foreground mb-1">
                      manifest_uri
                    </label>
                    <input
                      type="text"
                      value={newJob.ref_manifest_uri}
                      onChange={(e) => setNewJob({ ...newJob, ref_manifest_uri: e.target.value })}
                      className="sl-input font-mono text-sm"
                    />
                  </div>
                </div>
              )}

              <div className="flex flex-wrap gap-2 items-center">
                <button
                  type="button"
                  onClick={previewResolution}
                  disabled={resolving}
                  className="text-sm sl-btn-secondary rounded-lg px-3 py-2"
                >
                  {resolving ? 'Checking…' : 'Preview resolution'}
                </button>
                {resolveResult && (
                  <span className="text-xs text-success">OK → {resolveResult.data_path}</span>
                )}
              </div>
              {resolveError && (
                <p className="text-sm text-destructive">{resolveError}</p>
              )}
              {resolveResult && (
                <pre className="text-xs bg-muted border border-border p-3 rounded-lg overflow-x-auto text-foreground">
                  {JSON.stringify(resolveResult, null, 2)}
                </pre>
              )}

              <button
                type="button"
                onClick={() => setShowAdvanced((v) => !v)}
                className="text-sm text-primary hover:underline"
              >
                {showAdvanced ? 'Hide' : 'Show'} advanced model size
              </button>

              {showAdvanced && (
                <div className="grid grid-cols-2 gap-3 p-3 rounded-lg border border-border">
                  <div>
                    <label className="block text-xs font-medium text-muted-foreground mb-1">n_embed</label>
                    <input
                      type="number"
                      min={32}
                      value={newJob.n_embed}
                      onChange={(e) => setNewJob({ ...newJob, n_embed: parseInt(e.target.value, 10) || 128 })}
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-muted-foreground mb-1">n_layer</label>
                    <input
                      type="number"
                      min={1}
                      value={newJob.n_layer}
                      onChange={(e) => setNewJob({ ...newJob, n_layer: parseInt(e.target.value, 10) || 4 })}
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-muted-foreground mb-1">n_head</label>
                    <input
                      type="number"
                      min={1}
                      value={newJob.n_head}
                      onChange={(e) => setNewJob({ ...newJob, n_head: parseInt(e.target.value, 10) || 4 })}
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-muted-foreground mb-1">block_size</label>
                    <input
                      type="number"
                      min={8}
                      value={newJob.block_size}
                      onChange={(e) => setNewJob({ ...newJob, block_size: parseInt(e.target.value, 10) || 128 })}
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div className="col-span-2">
                    <label className="block text-xs font-medium text-muted-foreground mb-1">
                      max_steps (optional, caps training steps)
                    </label>
                    <input
                      type="text"
                      inputMode="numeric"
                      placeholder="e.g. 100"
                      value={newJob.maxStepsInput}
                      onChange={(e) => setNewJob({ ...newJob, maxStepsInput: e.target.value })}
                      className="sl-input py-1 text-sm font-mono"
                    />
                  </div>
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-muted-foreground mb-1">Epochs</label>
                <input
                  type="number"
                  value={newJob.epochs}
                  onChange={(e) => setNewJob({ ...newJob, epochs: parseInt(e.target.value, 10) })}
                  min={1}
                  max={100}
                  className="sl-input"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-muted-foreground mb-1">Batch Size</label>
                <input
                  type="number"
                  value={newJob.batch_size}
                  onChange={(e) => setNewJob({ ...newJob, batch_size: parseInt(e.target.value, 10) })}
                  min={1}
                  max={128}
                  className="sl-input"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-muted-foreground mb-1">Learning Rate</label>
                <input
                  type="number"
                  step="0.000001"
                  value={newJob.learning_rate}
                  onChange={(e) => setNewJob({ ...newJob, learning_rate: parseFloat(e.target.value) })}
                  className="sl-input"
                />
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                type="button"
                onClick={() => setShowModal(false)}
                className="flex-1 sl-btn-secondary rounded-lg px-4 py-2"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={startTraining}
                disabled={starting || !newJob.name.trim()}
                className="flex-1 sl-btn-primary rounded-lg px-4 py-2"
              >
                {starting ? 'Starting...' : 'Start Training'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
