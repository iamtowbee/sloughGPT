'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { FoldSection, JobStatus, ProgressBar, StatCard, KpiGrid } from '@/components/strui'
import { Button } from '@/components/ui/button'
import { Select } from '@/components/ui/select'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { inferenceHealthLabel, useApiHealth } from '@/hooks/useApiHealth'
import { api, TrainingJob, TrainResolveResponse, type Dataset } from '@/lib/api'
import { PUBLIC_API_URL } from '@/lib/config'
import { devDebug } from '@/lib/dev-log'
import {
  TRAINING_API_DEFAULTS,
  type TrainingMixedPrecisionDtype,
} from '@/lib/training-defaults'
import { trainingJobStatusToStrui } from '@/lib/training-status'
import { ToastContainer, type Toast } from '@/components/chat/Toast'

type CorpusMode = 'folder' | 'manifest' | 'ref'

const initialForm = {
  name: '',
  model: 'sloughgpt',
  corpusMode: 'folder' as CorpusMode,
  /** Default to bundled demo; API cwd must contain datasets/<name>/input.txt */
  dataset: 'shakespeare',
  manifest_uri: '',
  ref_dataset_id: '',
  ref_version: '',
  ref_manifest_uri: '',
  ...TRAINING_API_DEFAULTS,
  maxStepsInput: '',
  /** Sent in JSON only when non-empty (server / trainer default otherwise). */
  device: '',
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
  const { state: health, refresh: refreshHealth } = useApiHealth()

  // Toast notifications
  const [toasts, setToasts] = useState<Toast[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loadingDatasets, setLoadingDatasets] = useState(false)
  const [exportFormats, setExportFormats] = useState<{ id: string; name: string; description: string }[]>([])

  // Fetch export formats
  const fetchExportFormats = useCallback(async () => {
    try {
      const data = await api.getExportFormats()
      setExportFormats(Object.entries(data.formats).map(([id, desc]) => ({
        id,
        name: id.toUpperCase(),
        description: desc as string,
      })))
    } catch (err) {
      devDebug('Failed to fetch export formats:', err)
    }
  }, [])

  const addToast = useCallback((message: string, type: Toast['type'] = 'info') => {
    const id = Date.now().toString()
    setToasts(prev => [...prev, { id, message, type }])
  }, [])

  const dismissToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  const apiHealthLabel = useMemo(() => inferenceHealthLabel(health), [health])

  const healthToneClass = useMemo(() => {
    if (health === null) return 'text-muted-foreground'
    if (health === 'offline') return 'text-destructive'
    if (health.model_loaded) return 'text-success'
    return 'text-warning'
  }, [health])

  const fetchJobs = useCallback(async () => {
    try {
      const data = await api.getTrainingJobs()
      setJobs(data)
    } catch (error) {
      devDebug('Failed to fetch jobs:', error)
    } finally {
      setLoading(false)
    }
  }, [])

  // Training state control
  const [trainingState, setTrainingState] = useState<{
    state: string
    is_running: boolean
    is_paused: boolean
    is_idle: boolean
    current_job_id: string | null
    current_job_name: string | null
    can_start: boolean
    can_pause: boolean
    can_resume: boolean
    can_stop: boolean
    total_jobs: number
    completed_jobs: number
  } | null>(null)
  const [controlLoading, setControlLoading] = useState(false)

  // Fetch training state
  const fetchTrainingState = useCallback(async () => {
    try {
      const state = await api.getTrainingStatus()
      setTrainingState(state)
    } catch (error) {
      devDebug('Failed to fetch training state:', error)
    }
  }, [])

  // Training control handlers
  const handlePause = useCallback(async () => {
    setControlLoading(true)
    try {
      const result = await api.pauseTraining()
      addToast(result.message, result.success ? 'info' : 'error')
      void fetchTrainingState()
    } catch (error) {
      addToast('Failed to pause training', 'error')
    } finally {
      setControlLoading(false)
    }
  }, [fetchTrainingState])

  const handleResume = useCallback(async () => {
    setControlLoading(true)
    try {
      const result = await api.resumeTraining()
      addToast(result.message, result.success ? 'info' : 'error')
      void fetchTrainingState()
    } catch (error) {
      addToast('Failed to resume training', 'error')
    } finally {
      setControlLoading(false)
    }
  }, [fetchTrainingState])

  const handleStop = useCallback(async () => {
    if (!confirm('Stop current training job?')) return
    setControlLoading(true)
    try {
      const result = await api.stopTraining()
      addToast(result.message, result.success ? 'info' : 'error')
      void fetchTrainingState()
    } catch (error) {
      addToast('Failed to stop training', 'error')
    } finally {
      setControlLoading(false)
    }
  }, [fetchTrainingState])

  const fetchDatasets = useCallback(async () => {
    setLoadingDatasets(true)
    try {
      const data = await api.getDatasets()
      setDatasets(data)
    } catch (error) {
      devDebug('Failed to fetch datasets:', error)
    } finally {
      setLoadingDatasets(false)
    }
  }, [])

  const [perplexityText, setPerplexityText] = useState('')
  const [perplexityResult, setPerplexityResult] = useState<{ perplexity: number; text_length: number } | null>(null)
  const [perplexityLoading, setPerplexityLoading] = useState(false)

  const calculatePerplexity = async () => {
    if (!perplexityText.trim()) return
    setPerplexityLoading(true)
    setPerplexityResult(null)
    try {
      const result = await api.calculatePerplexity(perplexityText)
      setPerplexityResult(result)
    } catch (error) {
      addToast('Perplexity calculation failed', 'error')
    } finally {
      setPerplexityLoading(false)
    }
  }

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'r' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        void fetchJobs()
        void refreshHealth()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [fetchJobs, refreshHealth])

  useEffect(() => {
    void fetchDatasets()
    void fetchTrainingState()
    void fetchExportFormats()
  }, [fetchDatasets, fetchTrainingState, fetchExportFormats])

  useEffect(() => {
    const running = jobs.some((j) => j.status === 'running')
    const ms = running ? 5000 : 15000
    const id = setInterval(() => void fetchJobs(), ms)
    return () => clearInterval(id)
  }, [jobs, fetchJobs])

  useEffect(() => {
    const interval = setInterval(() => void fetchTrainingState(), 10000)
    return () => clearInterval(interval)
  }, [fetchTrainingState])

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
      log_interval: newJob.log_interval,
      eval_interval: newJob.eval_interval,
      dropout: newJob.dropout,
      weight_decay: newJob.weight_decay,
      gradient_accumulation_steps: newJob.gradient_accumulation_steps,
      max_grad_norm: newJob.max_grad_norm,
      use_mixed_precision: newJob.use_mixed_precision,
      mixed_precision_dtype: newJob.mixed_precision_dtype,
      warmup_steps: newJob.warmup_steps,
      min_lr: newJob.min_lr,
      scheduler: newJob.scheduler,
      use_lora: newJob.use_lora,
      lora_rank: newJob.lora_rank,
      lora_alpha: newJob.lora_alpha,
      checkpoint_dir: newJob.checkpoint_dir,
      checkpoint_interval: newJob.checkpoint_interval,
      save_best_only: newJob.save_best_only,
      max_checkpoints: newJob.max_checkpoints,
      ...(ms !== '' && !Number.isNaN(parseInt(ms, 10)) ? { max_steps: Math.max(1, parseInt(ms, 10)) } : {}),
      ...(newJob.device.trim() !== '' ? { device: newJob.device.trim() } : {}),
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
      addToast(`Training "${newJob.name}" started`, 'success')
      fetchJobs()
    } catch (error) {
      devDebug('Failed to start training:', error)
      addToast(error instanceof Error ? error.message : 'Start failed', 'error')
      setResolveError(error instanceof Error ? error.message : 'Start failed')
    } finally {
      setStarting(false)
    }
  }

  const openModal = () => {
    setResolveResult(null)
    setResolveError(null)
    setShowModal(true)
    void fetchDatasets()
  }

  const deleteJob = async (jobId: string, jobName: string) => {
    if (!confirm(`Delete training job "${jobName}"?`)) return
    try {
      await api.deleteTrainingJob(jobId)
      addToast(`Deleted job "${jobName}"`, 'success')
      fetchJobs()
    } catch (error) {
      addToast(error instanceof Error ? error.message : 'Delete failed', 'error')
    }
  }

  return (
    <div className="sl-page mx-auto max-w-7xl">
      <AppRouteHeader
        className="mb-6 justify-between"
        left={
          <AppRouteHeaderLead
            title="Training"
            subtitle={
              <>
                API:{' '}
                <span className={healthToneClass} data-testid="training-api-status">
                  {apiHealthLabel}
                </span>
              </>
            }
          />
        }
        right={
          <div className="flex flex-wrap justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              size="icon"
              onClick={() => {
                void fetchJobs()
              }}
              title="Refresh jobs"
              className="h-8 w-8"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </Button>
            <Button type="button" onClick={openModal}>
              + New Training Job
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={async () => {
                if (!confirm('Start self-training? This runs the model talking to itself.')) return
                try {
                  const res = await api.startSelfTrain({})
                  if (res.ok) addToast('Self-training started', 'success')
                  else addToast('Failed to start', 'error')
                } catch (e) {
                  addToast('Failed to start self-training', 'error')
                }
              }}
            >
              Self-Train
            </Button>
          </div>
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <RecoveryPanel jobs={jobs} addToast={addToast} fetchJobs={fetchJobs} />

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Training Jobs</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              {loading ? (
                <div className="text-center py-8 text-muted-foreground">Loading...</div>
              ) : jobs.length === 0 ? (
                <div className="text-center py-8">
                  <p className="text-muted-foreground mb-2">No training jobs yet</p>
                  <p className="text-sm text-muted-foreground">
                    Click <strong>+ New Training Job</strong> to start
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {jobs.map((job) => (
                    <div key={job.id} className="p-4 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <h3 className="font-medium">{job.name}</h3>
                          <p className="text-sm text-muted-foreground">
                            {job.model} • {job.dataset}
                          </p>
                        </div>
                        <JobStatus status={trainingJobStatusToStrui(job.status)} />
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center gap-3">
                          <div className="flex-1">
                            <ProgressBar 
                              value={job.progress || 0} 
                              indeterminate={job.status === 'running' && (job.progress || 0) === 0}
                            />
                          </div>
                          <span className="text-sm font-mono text-foreground">
                            {job.progress || 0}%
                          </span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-muted-foreground">
                            Step {job.global_step || 0}
                          </span>
                          {job.status === 'running' && (
                            <span className="flex items-center gap-1">
                              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                              Training
                            </span>
                          )}
                          {job.eval_loss !== undefined && job.status === 'completed' && (
                            <span className="text-muted-foreground">
                              Loss: {job.eval_loss.toFixed(4)}
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex gap-2 mt-3">
                        {job.status === 'completed' && job.checkpoint && (
                          <ExportDropdown jobId={job.id} checkpoint={job.checkpoint} addToast={addToast} />
                        )}
                        {job.status !== 'running' && job.status !== 'pending' && (
                          <Button size="sm" variant="ghost" onClick={() => deleteJob(job.id, job.name)} className="text-destructive/60 hover:text-destructive">
                            Delete
                          </Button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <FoldSection heading="Evaluate">
            <div className="space-y-3">
              <textarea
                className="sl-input w-full h-24 resize-none text-sm font-mono"
                placeholder="Enter text to evaluate..."
                value={perplexityText}
                onChange={(e) => setPerplexityText(e.target.value)}
              />
              <div className="flex items-center justify-between">
                <Button
                  size="sm"
                  onClick={calculatePerplexity}
                  disabled={perplexityLoading || !perplexityText.trim()}
                >
                  {perplexityLoading ? 'Calculating...' : 'Calculate Perplexity'}
                </Button>
                {perplexityResult && (
                  <div className="text-sm">
                    <span className="text-muted-foreground">Perplexity: </span>
                    <span className="font-mono font-medium">{perplexityResult.perplexity.toFixed(4)}</span>
                  </div>
                )}
              </div>
            </div>
          </FoldSection>
        </div>

        <div className="space-y-6">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Datasets</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="space-y-2">
                {datasets.map((ds) => (
                  <div key={ds.id} className="flex items-center justify-between py-2 px-3 rounded-md bg-muted/30 hover:bg-muted/50 transition-colors">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-sm truncate">{ds.name}</span>
                        <span className="text-xs text-muted-foreground">{ds.size}</span>
                      </div>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        if (confirm(`Delete dataset "${ds.name}"?\n\nThis will permanently remove the dataset. This cannot be undone.`)) {
                          void api.deleteDataset(ds.id).then(() => {
                            addToast(`Deleted "${ds.name}"`, 'success')
                            fetchDatasets()
                          }).catch((err) => {
                            addToast(`Failed to delete: ${err.message}`, 'error')
                          })
                        }
                      }}
                      className="text-destructive/60 hover:text-destructive shrink-0"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </Button>
                  </div>
                ))}
                {datasets.length === 0 && (
                  <p className="text-sm text-muted-foreground text-center py-4">No datasets found</p>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Webhooks</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <WebhookManager addToast={addToast} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Conversation Data</CardTitle>
              <CardDescription>Train from your chats</CardDescription>
            </CardHeader>
            <CardContent className="pt-0 space-y-4">
              <ConversationDataSection addToast={addToast} />
            </CardContent>
          </Card>
        </div>
      </div>

      <Dialog
        open={showModal}
        onOpenChange={(open) => {
          setShowModal(open)
          if (!open) {
            setResolveResult(null)
            setResolveError(null)
          }
        }}
      >
        <DialogContent className="w-[90vw] max-w-2xl">
          <DialogHeader>
            <DialogTitle>New Training Job</DialogTitle>
            <DialogDescription>
              Start a char-level SloughGPTModel job on the API host. Corpus must resolve as{' '}
              <span className="font-mono text-foreground/90">datasets/&lt;name&gt;/input.txt</span> from the
              server working directory.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6">
              <div className="bg-muted/50 rounded-lg p-4 space-y-3">
                <h3 className="text-sm font-semibold text-foreground">Basic Info</h3>
                <div>
                  <label className="block sl-text-body font-medium text-foreground mb-1">
                    Job Name <span className="text-destructive">*</span>
                  </label>
                  <input
                    type="text"
                    value={newJob.name}
                    onChange={(e) => setNewJob({ ...newJob, name: e.target.value })}
                    placeholder="e.g., Shakespeare Fine-tune"
                    className="sl-input"
                  />
                </div>

                <div>
                  <label className="block sl-text-body font-medium text-foreground mb-1">
                    Model Label
                  </label>
                  <input
                    type="text"
                    value={newJob.model}
                    onChange={(e) => setNewJob({ ...newJob, model: e.target.value })}
                    placeholder="sloughgpt"
                    className="sl-input font-mono text-sm"
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    For display only. All jobs train <span className="font-medium text-foreground">SloughGPTModel</span> (char-level).
                  </p>
                </div>
              </div>

              <div className="bg-muted/50 rounded-lg p-4 space-y-3">
                <h3 className="text-sm font-semibold text-foreground">Training Data</h3>
                <div>
                  <label className="block sl-text-body font-medium text-foreground mb-1">
                    Data Source
                  </label>
                  <Select
                    value={newJob.corpusMode}
                    onValueChange={(val) =>
                      setNewJob({
                        ...newJob,
                        corpusMode: val as CorpusMode,
                        manifest_uri: '',
                        ref_dataset_id: '',
                        ref_version: '',
                        ref_manifest_uri: '',
                      })
                    }
                    options={[
                      { value: 'folder', label: 'Local folder (datasets/)' },
                      { value: 'manifest', label: 'v1 manifest file' },
                      { value: 'ref', label: 'Versioned dataset (id + version)' },
                    ]}
                  />
                </div>

                {newJob.corpusMode === 'folder' && (
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label className="block text-sm font-medium text-foreground">Dataset</label>
                      <Button
                        type="button"
                        onClick={() => void fetchDatasets()}
                        disabled={loadingDatasets}
                        variant="ghost"
                        size="sm"
                        className="h-6 text-xs"
                      >
                        {loadingDatasets ? 'Loading...' : '↻'}
                      </Button>
                    </div>
                    {datasets.length > 0 ? (
                      <Select
                        value={newJob.dataset}
                        onValueChange={(val) => setNewJob({ ...newJob, dataset: val })}
                        options={datasets.map((ds) => ({
                          value: ds.id,
                          label: `${ds.name} (${ds.size})`,
                        }))}
                      />
                    ) : (
                      <div className="sl-input py-2 text-sm text-muted-foreground">
                        No datasets found
                      </div>
                    )}
                  </div>
                )}
              </div>

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
                <Button
                  type="button"
                  onClick={previewResolution}
                  disabled={resolving}
                  variant="secondary"
                  size="sm"
                >
                  {resolving ? 'Checking…' : 'Preview resolution'}
                </Button>
                {resolveResult && (
                  <span className="text-xs text-success">OK → {resolveResult.data_path}</span>
                )}
              </div>
              {resolveError && (
                <p className="text-sm text-destructive">{resolveError}</p>
              )}
              {resolveResult && (
                <pre className="text-xs bg-muted border border-border p-3 rounded-none overflow-x-auto text-foreground">
                  {JSON.stringify(resolveResult, null, 2)}
                </pre>
              )}

              <FoldSection heading="Advanced">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">Embed</label>
                    <input
                      type="number"
                      min={32}
                      value={newJob.n_embed}
                      onChange={(e) =>
                        setNewJob({
                          ...newJob,
                          n_embed: parseInt(e.target.value, 10) || TRAINING_API_DEFAULTS.n_embed,
                        })
                      }
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">Layers</label>
                    <input
                      type="number"
                      min={1}
                      value={newJob.n_layer}
                      onChange={(e) =>
                        setNewJob({
                          ...newJob,
                          n_layer: parseInt(e.target.value, 10) || TRAINING_API_DEFAULTS.n_layer,
                        })
                      }
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">Heads</label>
                    <input
                      type="number"
                      min={1}
                      value={newJob.n_head}
                      onChange={(e) =>
                        setNewJob({
                          ...newJob,
                          n_head: parseInt(e.target.value, 10) || TRAINING_API_DEFAULTS.n_head,
                        })
                      }
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">Block Size</label>
                    <input
                      type="number"
                      min={8}
                      value={newJob.block_size}
                      onChange={(e) =>
                        setNewJob({
                          ...newJob,
                          block_size: parseInt(e.target.value, 10) || TRAINING_API_DEFAULTS.block_size,
                        })
                      }
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div className="col-span-2">
                    <label className="block sl-text-caption text-muted-foreground mb-1">
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
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">
                      Log Every
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={50000}
                      value={newJob.log_interval}
                      onChange={(e) =>
                        setNewJob({
                          ...newJob,
                          log_interval: parseInt(e.target.value, 10) || TRAINING_API_DEFAULTS.log_interval,
                        })
                      }
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">
                      Eval Every
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={1000000}
                      value={newJob.eval_interval}
                      onChange={(e) =>
                        setNewJob({
                          ...newJob,
                          eval_interval: parseInt(e.target.value, 10) || TRAINING_API_DEFAULTS.eval_interval,
                        })
                      }
                      className="sl-input py-1 text-sm"
                    />
                  </div>

                  <div className="col-span-2 border-t border-border pt-3 mt-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                    Optimizer
                  </div>
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">dropout</label>
                    <input
                      type="number"
                      min={0}
                      max={0.9}
                      step={0.05}
                      value={newJob.dropout}
                      onChange={(e) => {
                        const v = parseFloat(e.target.value)
                        setNewJob({
                          ...newJob,
                          dropout: Number.isFinite(v) ? v : TRAINING_API_DEFAULTS.dropout,
                        })
                      }}
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">weight_decay</label>
                    <input
                      type="number"
                      min={0}
                      step={0.001}
                      value={newJob.weight_decay}
                      onChange={(e) => {
                        const v = parseFloat(e.target.value)
                        setNewJob({
                          ...newJob,
                          weight_decay: Number.isFinite(v) ? v : TRAINING_API_DEFAULTS.weight_decay,
                        })
                      }}
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">max_grad_norm</label>
                    <input
                      type="number"
                      min={0}
                      step={0.1}
                      value={newJob.max_grad_norm}
                      onChange={(e) => {
                        const v = parseFloat(e.target.value)
                        setNewJob({
                          ...newJob,
                          max_grad_norm: Number.isFinite(v) ? v : TRAINING_API_DEFAULTS.max_grad_norm,
                        })
                      }}
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">grad_accum_steps</label>
                    <input
                      type="number"
                      min={1}
                      value={newJob.gradient_accumulation_steps}
                      onChange={(e) =>
                        setNewJob({
                          ...newJob,
                          gradient_accumulation_steps:
                            parseInt(e.target.value, 10) || TRAINING_API_DEFAULTS.gradient_accumulation_steps,
                        })
                      }
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">warmup_steps</label>
                    <input
                      type="number"
                      min={0}
                      value={newJob.warmup_steps}
                      onChange={(e) => {
                        const v = parseInt(e.target.value, 10)
                        setNewJob({
                          ...newJob,
                          warmup_steps: Number.isFinite(v) && v >= 0 ? v : TRAINING_API_DEFAULTS.warmup_steps,
                        })
                      }}
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">min_lr</label>
                    <input
                      type="number"
                      step="any"
                      value={newJob.min_lr}
                      onChange={(e) => {
                        const v = parseFloat(e.target.value)
                        setNewJob({
                          ...newJob,
                          min_lr: Number.isFinite(v) ? v : TRAINING_API_DEFAULTS.min_lr,
                        })
                      }}
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div className="col-span-2">
                    <label className="block sl-text-caption text-muted-foreground mb-1">scheduler</label>
                    <input
                      type="text"
                      value={newJob.scheduler}
                      onChange={(e) =>
                        setNewJob({ ...newJob, scheduler: e.target.value.trim() || 'cosine' })
                      }
                      className="sl-input py-1 text-sm font-mono"
                      placeholder="cosine"
                    />
                  </div>
                  <label className="col-span-2 flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      className="rounded border-border"
                      checked={newJob.use_mixed_precision}
                      onChange={(e) =>
                        setNewJob({ ...newJob, use_mixed_precision: e.target.checked })
                      }
                    />
                    <span className="text-xs text-muted-foreground">Mixed precision</span>
                  </label>
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">AMP dtype</label>
                    <select
                      value={newJob.mixed_precision_dtype}
                      onChange={(e) =>
                        setNewJob({
                          ...newJob,
                          mixed_precision_dtype: e.target.value as TrainingMixedPrecisionDtype,
                        })
                      }
                      disabled={!newJob.use_mixed_precision}
                      className="sl-input py-1 text-sm"
                    >
                      <option value="bf16">bf16</option>
                      <option value="fp16">fp16</option>
                    </select>
                  </div>
                  <div className="col-span-2">
                    <label className="block sl-text-caption text-muted-foreground mb-1">
                      device (training host; empty = default)
                    </label>
                    <input
                      type="text"
                      value={newJob.device}
                      onChange={(e) => setNewJob({ ...newJob, device: e.target.value })}
                      className="sl-input py-1 text-sm font-mono"
                      placeholder="cuda, cpu, mps…"
                    />
                  </div>
                  <label className="col-span-2 flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      className="rounded border-border"
                      checked={newJob.use_lora}
                      onChange={(e) => setNewJob({ ...newJob, use_lora: e.target.checked })}
                    />
                    <span className="text-xs text-muted-foreground">LoRA</span>
                  </label>
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">lora_rank</label>
                    <input
                      type="number"
                      min={1}
                      value={newJob.lora_rank}
                      disabled={!newJob.use_lora}
                      onChange={(e) =>
                        setNewJob({
                          ...newJob,
                          lora_rank: parseInt(e.target.value, 10) || TRAINING_API_DEFAULTS.lora_rank,
                        })
                      }
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">lora_alpha</label>
                    <input
                      type="number"
                      min={1}
                      value={newJob.lora_alpha}
                      disabled={!newJob.use_lora}
                      onChange={(e) =>
                        setNewJob({
                          ...newJob,
                          lora_alpha: parseInt(e.target.value, 10) || TRAINING_API_DEFAULTS.lora_alpha,
                        })
                      }
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div className="col-span-2">
                    <label className="block sl-text-caption text-muted-foreground mb-1">checkpoint_dir</label>
                    <input
                      type="text"
                      value={newJob.checkpoint_dir}
                      onChange={(e) =>
                        setNewJob({
                          ...newJob,
                          checkpoint_dir: e.target.value.trim() || TRAINING_API_DEFAULTS.checkpoint_dir,
                        })
                      }
                      className="sl-input py-1 text-sm font-mono"
                    />
                    <p className="text-xs text-muted-foreground mt-1">
                      Periodic <span className="font-mono text-foreground/80">step_*.pt</span> here
                      includes <span className="font-mono text-foreground/80">stoi</span> /{' '}
                      <span className="font-mono text-foreground/80">itos</span> for{' '}
                      <span className="font-mono text-foreground/80">cli.py eval</span>. See repo{' '}
                      <span className="font-mono text-foreground/80">docs/policies/CONTRIBUTING.md</span>{' '}
                      (Checkpoint vocabulary).
                    </p>
                  </div>
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">
                      checkpoint_interval
                    </label>
                    <input
                      type="number"
                      min={1}
                      value={newJob.checkpoint_interval}
                      onChange={(e) =>
                        setNewJob({
                          ...newJob,
                          checkpoint_interval:
                            parseInt(e.target.value, 10) || TRAINING_API_DEFAULTS.checkpoint_interval,
                        })
                      }
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block sl-text-caption text-muted-foreground mb-1">max_checkpoints</label>
                    <input
                      type="number"
                      min={1}
                      value={newJob.max_checkpoints}
                      onChange={(e) =>
                        setNewJob({
                          ...newJob,
                          max_checkpoints:
                            parseInt(e.target.value, 10) || TRAINING_API_DEFAULTS.max_checkpoints,
                        })
                      }
                      className="sl-input py-1 text-sm"
                    />
                  </div>
                  <label className="col-span-2 flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      className="rounded border-border"
                      checked={newJob.save_best_only}
                      onChange={(e) =>
                        setNewJob({ ...newJob, save_best_only: e.target.checked })
                      }
                    />
                    <span className="text-xs text-muted-foreground">save_best_only</span>
                  </label>
                </div>
              </FoldSection>

              <div className="bg-muted/50 rounded-lg p-4 space-y-3">
                <h3 className="text-sm font-semibold text-foreground">Parameters</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block sl-text-body font-medium text-foreground mb-1">Epochs</label>
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
                    <label className="block sl-text-body font-medium text-foreground mb-1">Batch</label>
                    <input
                      type="number"
                      value={newJob.batch_size}
                      onChange={(e) => setNewJob({ ...newJob, batch_size: parseInt(e.target.value, 10) })}
                      min={1}
                      max={128}
                      className="sl-input"
                    />
                    <p className="text-xs text-muted-foreground mt-1">Samples/batch</p>
                  </div>

                  <div>
                    <label className="block sl-text-body font-medium text-foreground mb-1">LR</label>
                    <input
                      type="number"
                      step="0.000001"
                      value={newJob.learning_rate}
                      onChange={(e) => setNewJob({ ...newJob, learning_rate: parseFloat(e.target.value) })}
                      className="sl-input"
                    />
                  </div>
                </div>
              </div>

              {/* Training Preview Summary */}
              <div className={`rounded-lg p-4 space-y-2 ${
                !newJob.name.trim() 
                  ? 'bg-amber-500/10 border border-amber-500/30' 
                  : 'bg-green-500/10 border border-green-500/30'
              }`}>
                <h4 className={`text-sm font-semibold flex items-center gap-2 ${
                  !newJob.name.trim() ? 'text-amber-600' : 'text-green-600'
                }`}>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                  {newJob.name.trim() ? 'Ready to Train' : 'Missing Job Name'}
                </h4>
                {newJob.name.trim() ? (
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                    <span className="text-muted-foreground">Job:</span>
                    <span className="font-medium">{newJob.name}</span>
                    <span className="text-muted-foreground">Dataset:</span>
                    <span className="font-medium">{newJob.dataset}</span>
                    <span className="text-muted-foreground">Epochs:</span>
                    <span className="font-medium">{newJob.epochs}</span>
                    <span className="text-muted-foreground">Batch:</span>
                    <span className="font-medium">{newJob.batch_size}</span>
                    <span className="text-muted-foreground">LR:</span>
                    <span className="font-medium">{newJob.learning_rate}</span>
                  </div>
                ) : (
                  <p className="text-xs text-amber-600/80">
                    Please enter a job name above to start training.
                  </p>
                )}
              </div>
            </div>

            <DialogFooter className="mt-4 gap-3 sm:gap-3">
              <Button
                type="button"
                variant="secondary"
                className="flex-1 sm:flex-none"
                onClick={() => setShowModal(false)}
              >
                Cancel
              </Button>
              <Button
                type="button"
                className="flex-1 sm:flex-none bg-green-600 hover:bg-green-700"
                onClick={startTraining}
                disabled={starting || !newJob.name.trim()}
              >
                {starting ? 'Starting...' : 'Start Training'}
              </Button>
            </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Toast Notifications */}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </div>
  )
}

// ===== RecoveryPanel Component =====

interface RecoveryPanelProps {
  jobs: TrainingJob[]
  addToast: (message: string, type?: 'info' | 'success' | 'error') => void
  fetchJobs: () => void
}

function RecoveryPanel({ addToast, fetchJobs }: RecoveryPanelProps) {
  const [expanded, setExpanded] = useState(false)
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState<{
    pending: number
    running: number
    completed: number
    failed: number
    total: number
    crashed_jobs: number
    recoverable_jobs: number
  } | null>(null)
  const [recoverableJobs, setRecoverableJobs] = useState<Array<{
    id: string
    name: string
    status: string
    progress: number
    config: Record<string, unknown>
    checkpoint_path?: string
  }>>([])

  const checkRecovery = useCallback(async () => {
    try {
      const data = await api.getRecoveryStats()
      setStats(data)
      const recoverable = await api.getRecoverableJobs()
      setRecoverableJobs(recoverable.jobs)
    } catch (error) {
      devDebug('Failed to check recovery:', error)
    }
  }, [])

  useEffect(() => {
    void checkRecovery()
    const interval = setInterval(() => void checkRecovery(), 30000)
    return () => clearInterval(interval)
  }, [checkRecovery])

  const handleRecover = async (jobId: string) => {
    if (!confirm('Recover this job? It will restart from the last checkpoint.')) return
    setLoading(true)
    try {
      const result = await api.recoverJob(jobId)
      addToast(result.message, result.status === 'recovered' ? 'success' : 'info')
      void fetchJobs()
    } catch (error) {
      addToast(error instanceof Error ? error.message : 'Recovery failed', 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleAbandon = async (jobId: string) => {
    if (!confirm('Abandon this job permanently? This cannot be undone.')) return
    setLoading(true)
    try {
      const result = await api.abandonJob(jobId)
      addToast(result.message, 'success')
      void fetchJobs()
      void checkRecovery()
    } catch (error) {
      addToast(error instanceof Error ? error.message : 'Abandon failed', 'error')
    } finally {
      setLoading(false)
    }
  }

  const crashedCount = stats?.crashed_jobs ?? 0

  if (crashedCount === 0) return null

  return (
    <FoldSection heading="Job Recovery">
      <div className="flex items-center justify-between p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
        <div>
          <p className="text-sm font-medium text-amber-700 dark:text-amber-300">
            {crashedCount > 0 ? `${crashedCount} job(s) may have crashed` : 'Interrupted jobs detected'}
          </p>
          <p className="text-xs text-amber-700 dark:text-amber-300 mt-1">
            Server may have stopped unexpectedly.
          </p>
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? 'Hide' : 'Show'}
        </Button>
      </div>

      {expanded && (
        <div className="mt-3 space-y-2">
          {recoverableJobs.length > 0 ? (
            recoverableJobs.map(job => (
              <div key={job.id} className="flex items-center justify-between p-3 bg-muted/30 rounded-md">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-amber-500" />
                    <span className="font-medium text-sm truncate">{job.name || job.id}</span>
                  </div>
                  <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                    <span>Progress: {job.progress}%</span>
                    {job.checkpoint_path && (
                      <span className="truncate">Checkpoint: {job.checkpoint_path.split('/').pop()}</span>
                    )}
                  </div>
                </div>
                <div className="flex gap-2 ml-2">
                  <Button
                    size="sm"
                    variant="default"
                    onClick={() => void handleRecover(job.id)}
                    disabled={loading}
                    className="bg-green-600 hover:bg-green-700"
                  >
                    Recover
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => void handleAbandon(job.id)}
                    className="text-destructive/60 hover:text-destructive"
                  >
                    Abandon
                  </Button>
                </div>
              </div>
            ))
          ) : (
            <p className="text-sm text-muted-foreground text-center py-4">
              No recoverable jobs found.
            </p>
          )}
        </div>
      )}
    </FoldSection>
  )
}

// ===== WebhookManager Component =====

interface WebhookManagerProps {
  addToast: (message: string, type?: 'info' | 'success' | 'error') => void
}

function WebhookManager({ addToast }: WebhookManagerProps) {
  const [webhooks, setWebhooks] = useState<Array<{
    id: string
    url: string
    events: string[]
    description: string
    is_active: boolean
    created_at: string
  }>>([])
  const [showAdd, setShowAdd] = useState(false)
  const [newWebhook, setNewWebhook] = useState({
    url: '',
    description: '',
    events: ['training.completed'],
  })
  const [loading, setLoading] = useState(false)
  const [availableEvents, setAvailableEvents] = useState<string[]>(['training.completed'])

  const fetchWebhooks = useCallback(async () => {
    try {
      const data = await api.getWebhooks()
      setWebhooks(data.webhooks)
      setAvailableEvents(data.available_events)
    } catch (error) {
      devDebug('Failed to fetch webhooks:', error)
    }
  }, [])

  const fetchStats = useCallback(async () => {
    try {
      await api.getWebhookStats()
    } catch (error) {
      devDebug('Failed to fetch webhook stats:', error)
    }
  }, [])

  useEffect(() => {
    void fetchWebhooks()
    void fetchStats()
  }, [fetchWebhooks, fetchStats])

  const handleAddWebhook = async () => {
    if (!newWebhook.url.trim()) return
    setLoading(true)
    try {
      await api.registerWebhook({
        url: newWebhook.url,
        events: newWebhook.events,
        description: newWebhook.description,
      })
      addToast('Webhook registered', 'success')
      setShowAdd(false)
      setNewWebhook({ url: '', description: '', events: ['training.completed'] })
      void fetchWebhooks()
    } catch (error) {
      addToast(error instanceof Error ? error.message : 'Failed to register webhook', 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteWebhook = async (id: string) => {
    if (!confirm('Delete this webhook?')) return
    try {
      await api.deleteWebhook(id)
      addToast('Webhook deleted', 'success')
      void fetchWebhooks()
    } catch (error) {
      addToast(error instanceof Error ? error.message : 'Delete failed', 'error')
    }
  }

  const handleTestWebhook = async (url: string) => {
    try {
      await api.testWebhook(url)
      addToast('Test sent', 'success')
    } catch (error) {
      addToast(error instanceof Error ? error.message : 'Test failed', 'error')
    }
  }

  const toggleEvent = (event: string) => {
    setNewWebhook(prev => ({
      ...prev,
      events: prev.events.includes(event)
        ? prev.events.filter(e => e !== event)
        : [...prev.events, event],
    }))
  }

  return (
    <div className="space-y-3">
      {webhooks.length > 0 ? (
        <div className="space-y-2">
          {webhooks.map(webhook => (
            <div key={webhook.id} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${webhook.is_active ? 'bg-green-500' : 'bg-gray-400'}`} />
                  <span className="font-medium text-sm truncate">{webhook.url}</span>
                </div>
                <div className="flex gap-1 mt-1 flex-wrap">
                  {webhook.events.map(event => (
                    <span key={event} className="text-xs bg-muted px-2 py-0.5 rounded">
                      {event.split('.')[1]}
                    </span>
                  ))}
                </div>
              </div>
              <div className="flex gap-1 ml-2">
                <Button size="sm" variant="ghost" onClick={() => handleTestWebhook(webhook.url)}>
                  Test
                </Button>
                <Button size="sm" variant="ghost" onClick={() => handleDeleteWebhook(webhook.id)} className="text-destructive/60 hover:text-destructive">
                  ✕
                </Button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground text-center py-4">No webhooks configured</p>
      )}

      {showAdd ? (
        <div className="p-4 border border-border rounded-lg space-y-3">
          <input
            type="url"
            value={newWebhook.url}
            onChange={e => setNewWebhook(prev => ({ ...prev, url: e.target.value }))}
            placeholder="https://example.com/webhook"
            className="sl-input w-full"
          />
          <input
            type="text"
            value={newWebhook.description}
            onChange={e => setNewWebhook(prev => ({ ...prev, description: e.target.value }))}
            placeholder="Description (optional)"
            className="sl-input w-full"
          />
          <div className="flex gap-2 flex-wrap">
            {availableEvents.map(event => (
              <label key={event} className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={newWebhook.events.includes(event)}
                  onChange={() => toggleEvent(event)}
                />
                {event.split('.')[1]}
              </label>
            ))}
          </div>
          <div className="flex gap-2">
            <Button size="sm" onClick={handleAddWebhook} disabled={loading || !newWebhook.url.trim()}>
              {loading ? 'Adding...' : 'Add Webhook'}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setShowAdd(false)}>
              Cancel
            </Button>
          </div>
        </div>
      ) : (
        <Button size="sm" onClick={() => setShowAdd(true)}>
          + Add Webhook
        </Button>
      )}
    </div>
  )
}

// ===== ExportDropdown Component =====

interface ExportDropdownProps {
  jobId: string
  checkpoint: string
  addToast: (message: string, type?: 'info' | 'success' | 'error') => void
}

function ExportDropdown({ jobId, checkpoint, addToast }: ExportDropdownProps) {
  const [format, setFormat] = useState('pt')
  const [exporting, setExporting] = useState(false)

  const handleExport = async () => {
    setExporting(true)
    try {
      const blob = await api.exportTrainingJob(jobId)
      const filename = checkpoint.split('/').pop() || `model.${format}`
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      link.click()
      URL.revokeObjectURL(url)
      addToast(`Downloaded ${filename}`, 'success')
    } catch (err) {
      addToast(`Export failed: ${err instanceof Error ? err.message : 'Unknown error'}`, 'error')
    } finally {
      setExporting(false)
    }
  }

  const formats = [
    { id: 'pt', name: 'PyTorch (.pt)', desc: 'Standard PyTorch format' },
    { id: 'sou', name: 'Soul (.sou)', desc: 'SloughGPT with personality' },
    { id: 'safetensors', name: 'SafeTensors', desc: 'Safe, memory-mapped' },
    { id: 'onnx', name: 'ONNX (.onnx)', desc: 'Cross-platform inference' },
    { id: 'gguf', name: 'GGUF', desc: 'Mobile/embedded (llama.cpp)' },
  ]

  return (
    <div className="flex gap-2 items-center">
      <select
        value={format}
        onChange={(e) => setFormat(e.target.value)}
        className="sl-input py-1 px-2 text-xs h-8"
      >
        {formats.map((f) => (
          <option key={f.id} value={f.id}>
            {f.name}
          </option>
        ))}
      </select>
      <Button
        size="sm"
        variant="outline"
        onClick={handleExport}
        disabled={exporting}
      >
        {exporting ? 'Exporting...' : 'Export'}
      </Button>
    </div>
  )
}

// ===== ConversationDataSection Component =====

function ConversationDataSection({ addToast }: { addToast: (message: string, type?: 'info' | 'success' | 'error') => void }) {
  const [stats, setStats] = useState<{
    total_pairs: number
    positive_pairs: number
    negative_pairs: number
    neutral_pairs: number
    unused_pairs: number
  } | null>(null)
  const [loading, setLoading] = useState(true)
  const [exporting, setExporting] = useState(false)
  const [strategy, setStrategy] = useState<'balanced' | 'weighted' | 'simple'>('balanced')
  const [targetCount, setTargetCount] = useState(100)

  const fetchStats = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.getTrainingStatus()
      setStats({
        total_pairs: res.total_jobs,
        positive_pairs: res.completed_jobs,
        negative_pairs: res.running_jobs?.length || 0,
        neutral_pairs: res.failed_jobs || 0,
        unused_pairs: 0,
      })
    } catch (err) {
      devDebug('Failed to fetch training stats:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchStats()
  }, [fetchStats])

  const handleExport = useCallback(async () => {
    setExporting(true)
    try {
      const data = await api.exportFeedbackPairs(0, targetCount)
      console.log('Export response:', data)
      
      if (data.error) {
        addToast(data.error, 'error')
      } else {
        addToast(`Exported ${data.pairs_count || 0} pairs to ${data.filepath || 'file'}`, 'success')
        void fetchStats()
      }
    } catch (err) {
      console.error('Export error:', err)
      addToast('Export failed: ' + (err instanceof Error ? err.message : 'Unknown error'), 'error')
    } finally {
      setExporting(false)
    }
  }, [targetCount, addToast, fetchStats])

  return (
    <div className="space-y-4">
      {loading ? (
        <div className="text-center py-4 text-muted-foreground">Loading...</div>
      ) : stats ? (
        <>
          <div className="grid grid-cols-2 gap-2">
            <div className="p-3 rounded-lg bg-green-500/10 border border-green-500/30">
              <div className="text-2xl font-bold text-green-600">{stats.positive_pairs}</div>
              <div className="text-xs text-muted-foreground">Positive (👍)</div>
            </div>
            <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30">
              <div className="text-2xl font-bold text-red-600">{stats.negative_pairs}</div>
              <div className="text-xs text-muted-foreground">Negative (👎)</div>
            </div>
            <div className="p-3 rounded-lg bg-muted">
              <div className="text-2xl font-bold">{stats.neutral_pairs}</div>
              <div className="text-xs text-muted-foreground">Neutral</div>
            </div>
            <div className="p-3 rounded-lg bg-muted">
              <div className="text-2xl font-bold">{stats.total_pairs}</div>
              <div className="text-xs text-muted-foreground">Total Pairs</div>
            </div>
          </div>

          <div className="space-y-3">
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Sampling Strategy</label>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value as any)}
                className="sl-input py-1.5 text-sm"
              >
                <option value="balanced">Balanced (equal +/ /neutral)</option>
                <option value="weighted">Weighted (edge cases)</option>
                <option value="simple">Simple (filter by quality)</option>
              </select>
            </div>

            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Target Count</label>
              <input
                type="number"
                value={targetCount}
                onChange={(e) => setTargetCount(parseInt(e.target.value) || 100)}
                min={10}
                max={1000}
                className="sl-input py-1.5 text-sm"
              />
            </div>

            <Button
              onClick={handleExport}
              disabled={exporting || stats.total_pairs < 5}
              className="w-full"
              size="sm"
            >
              {exporting ? 'Exporting...' : `Export ${targetCount} Pairs for Training`}
            </Button>

            {stats.total_pairs < 5 && (
              <p className="text-xs text-amber-600">
                Need at least 5 conversation pairs to export. Chat more to build your training data!
              </p>
            )}
          </div>
        </>
      ) : (
        <div className="text-center py-4 text-muted-foreground">
          No training data yet. Start chatting to build your dataset.
        </div>
      )}
    </div>
  )
}
