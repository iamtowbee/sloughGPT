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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { inferenceHealthLabel, useApiHealth } from '@/hooks/useApiHealth'
import { api, TrainingJob, TrainResolveResponse, type TrainingStats, type UserAdapterStats, type WorkflowStatus, type Dataset } from '@/lib/api'
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

  // Feedback training data state
  const [trainingStats, setTrainingStats] = useState<TrainingStats | null>(null)
  const [exporting, setExporting] = useState<string | null>(null)
  const [exportResult, setExportResult] = useState<string | null>(null)
  const [trainResult, setTrainResult] = useState<{ status: string; job_id?: string; samples?: number; message?: string } | null>(null)

  // Workflow & adapter state
  const [workflowStatus, setWorkflowStatus] = useState<WorkflowStatus | null>(null)
  const [adapterStats, setAdapterStats] = useState<UserAdapterStats | null>(null)
  const [adminAction, setAdminAction] = useState<string | null>(null)

  // Toast notifications
  const [toasts, setToasts] = useState<Toast[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loadingDatasets, setLoadingDatasets] = useState(false)

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

  useEffect(() => {
    void fetchJobs()
    // Fetch feedback training stats
    api.getTrainingStats().then(setTrainingStats).catch(() => setTrainingStats(null))
    // Fetch workflow status
    api.getWorkflowStatus().then(setWorkflowStatus).catch(() => setWorkflowStatus(null))
    // Fetch adapter stats
    api.getUserAdapters().then(setAdapterStats).catch(() => setAdapterStats(null))
  }, [fetchJobs])

  const handleExport = useCallback(async (format: 'dpo' | 'sft') => {
    setExporting(format)
    setExportResult(null)
    try {
      const result = await api.exportTrainingData(format)
      setExportResult(`Exported ${result.count} ${format.toUpperCase()} examples to ${result.filepath}`)
      // Refresh stats
      const stats = await api.getTrainingStats()
      setTrainingStats(stats)
    } catch (error) {
      setExportResult('Export failed')
      devDebug('Export failed:', error)
    } finally {
      setExporting(null)
    }
  }, [])

  const handleWorkflowAction = useCallback(async (action: 'aggregate' | 'prune' | 'export' | 'start' | 'stop') => {
    setAdminAction(action)
    try {
      if (action === 'start') {
        await api.startWorkflow()
      } else if (action === 'stop') {
        await api.stopWorkflow()
      } else {
        await api.triggerWorkflowAction(action)
      }
      // Refresh status
      const [wf, adapters] = await Promise.all([
        api.getWorkflowStatus(),
        api.getUserAdapters()
      ])
      setWorkflowStatus(wf)
      setAdapterStats(adapters)
    } catch (error) {
      devDebug('Workflow action failed:', error)
    } finally {
      setAdminAction(null)
    }
  }, [])

  const handleTrainFromFeedback = useCallback(async () => {
    setAdminAction('train')
    setTrainResult(null)
    try {
      const result = await api.trainFromFeedback()
      setTrainResult(result)
      // Refresh jobs to show new training job
      await fetchJobs()
    } catch (error) {
      devDebug('Train from feedback failed:', error)
      setTrainResult({ status: 'error', message: 'Training failed to start' })
    } finally {
      setAdminAction(null)
    }
  }, [fetchJobs])

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
    const running = jobs.some((j) => j.status === 'running')
    const ms = running ? 2000 : 8000
    const id = setInterval(() => void fetchJobs(), ms)
    return () => clearInterval(id)
  }, [jobs, fetchJobs])

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

  return (
    <div className="sl-page mx-auto max-w-5xl">
      <AppRouteHeader
        className="mb-6 items-start"
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
              variant="secondary"
              size="sm"
              onClick={() => {
                void fetchJobs()
                void refreshHealth()
              }}
            >
              Refresh
            </Button>
            <Button type="button" onClick={openModal}>
              New Training Job
            </Button>
          </div>
        }
      />

      {/* Feedback Training Data Section */}
      {trainingStats && (
        <FoldSection heading="Feedback Training Data">
          <div className="space-y-4">
            <KpiGrid>
              <StatCard label="Conversations" value={trainingStats.total_conversations} />
              <StatCard label="Thumbs Up" value={trainingStats.thumbs_up} hint="Positive feedback" />
              <StatCard label="Thumbs Down" value={trainingStats.thumbs_down} hint="Negative feedback" />
              <StatCard label="DPO Pairs" value={trainingStats.available_dpo_pairs} />
            </KpiGrid>

            {exportResult && (
              <div className="p-2 text-sm bg-muted rounded-md font-mono break-all">
                {exportResult}
              </div>
            )}

            <div className="flex gap-2">
              <Button
                size="sm"
                onClick={() => handleExport('dpo')}
                disabled={exporting !== null || trainingStats.available_dpo_pairs === 0}
              >
                {exporting === 'dpo' ? 'Exporting...' : 'Export DPO'}
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => handleExport('sft')}
                disabled={exporting !== null || trainingStats.available_sft_examples === 0}
              >
                {exporting === 'sft' ? 'Exporting...' : 'Export SFT'}
              </Button>
            </div>
          </div>
        </FoldSection>
      )}

      {/* Workflow & Adapters Management Section */}
      {(workflowStatus || adapterStats) && (
        <FoldSection heading="LLM Adaptation System">
          <div className="space-y-4">
            <KpiGrid>
              {adapterStats && (
                <>
                  <StatCard label="User Adapters" value={adapterStats.total_users} hint={typeof adapterStats.total_size_mb === 'number' ? `${adapterStats.total_size_mb.toFixed(2)} MB` : '—'} />
                  <StatCard 
                    label="Quality Adapters" 
                    value={adapterStats.auto_management?.quality_adapters_count ?? 0} 
                    hint={`Threshold: ${adapterStats.auto_management?.aggregate_threshold ?? 50}`} 
                  />
                </>
              )}
              {workflowStatus && (
                <StatCard 
                  label="Workflow" 
                  value={workflowStatus.running ? 'Active' : 'Stopped'} 
                  hint={`Aggregations: ${workflowStatus.stats.aggregations_performed}`}
                />
              )}
            </KpiGrid>

            <div className="flex flex-wrap gap-2">
              {workflowStatus && (
                <Button
                  size="sm"
                  variant={workflowStatus.running ? 'destructive' : 'default'}
                  onClick={() => handleWorkflowAction(workflowStatus.running ? 'stop' : 'start')}
                  disabled={adminAction !== null}
                >
                  {adminAction === (workflowStatus.running ? 'stop' : 'start') 
                    ? 'Processing...' 
                    : workflowStatus.running ? 'Stop Workflow' : 'Start Workflow'}
                </Button>
              )}
              <Button
                size="sm"
                variant="secondary"
                onClick={() => handleWorkflowAction('aggregate')}
                disabled={adminAction !== null}
              >
                {adminAction === 'aggregate' ? 'Aggregating...' : 'Aggregate Adapters'}
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => handleWorkflowAction('prune')}
                disabled={adminAction !== null}
              >
                {adminAction === 'prune' ? 'Pruning...' : 'Prune Low Quality'}
              </Button>
              <Button
                size="sm"
                variant="default"
                onClick={handleTrainFromFeedback}
                disabled={adminAction !== null || !trainingStats || trainingStats.available_sft_examples === 0}
                className="bg-green-600 hover:bg-green-700"
              >
                {adminAction === 'train' ? 'Training...' : 'Train from Feedback'}
              </Button>
            </div>
            {trainResult && (
              <div className={`p-2 rounded-md text-sm ${trainResult.status === 'started' ? 'bg-green-500/20 text-green-700' : 'bg-muted'}`}>
                {trainResult.message}
                {trainResult.job_id && <span className="ml-2 font-mono text-xs">Job: {trainResult.job_id}</span>}
                {trainResult.samples && <span className="ml-2">({trainResult.samples} samples)</span>}
              </div>
            )}
          </div>
        </FoldSection>
      )}

      {loading ? (
        <div className="text-center py-8 text-muted-foreground">Loading...</div>
      ) : jobs.length === 0 ? (
        <Card className="mx-auto max-w-md border-dashed">
          <CardHeader className="text-center">
            <CardTitle className="text-base">No training jobs yet</CardTitle>
            <CardDescription className="text-balance">
              Start the FastAPI app from the repo root, pick a dataset folder under{' '}
              <span className="font-mono text-foreground/90">datasets/</span> (e.g.{' '}
              <span className="font-mono">shakespeare</span>), then use <strong>New Training Job</strong>.
            </CardDescription>
          </CardHeader>
          <CardContent className="text-center">
            <p className="font-mono text-xs text-muted-foreground break-all">python3 apps/api/server/main.py</p>
          </CardContent>
        </Card>
      ) : (
        <Card className="overflow-hidden p-0">
          <CardHeader className="border-b border-border py-4">
            <CardTitle className="text-base">Training jobs</CardTitle>
          </CardHeader>

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
                  <JobStatus status={trainingJobStatusToStrui(job.status)} />
                </div>

                <div className="mt-3">
                  <div className="flex justify-between text-sm text-muted-foreground mb-1">
                    <span>
                      {job.current_epoch && job.epochs
                        ? `Epoch ${job.current_epoch}/${job.epochs}`
                        : 'Progress'}
                      {job.global_step != null && job.status === 'running' && (
                        <span className="ml-1.5 font-mono text-xs">· step {job.global_step}</span>
                      )}
                    </span>
                    <span>{job.progress}%</span>
                  </div>
                  <ProgressBar
                    value={job.progress}
                    max={100}
                    indeterminate={
                      job.status === 'pending' ||
                      (job.status === 'running' && job.progress === 0)
                    }
                  />
                </div>

                {(job.train_loss != null && Number.isFinite(job.train_loss)) ||
                (job.eval_loss != null && Number.isFinite(job.eval_loss)) ? (
                  <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 text-sm text-muted-foreground">
                    {job.train_loss != null && Number.isFinite(job.train_loss) && (
                      <span>Train loss: {job.train_loss.toFixed(4)}</span>
                    )}
                    {job.eval_loss != null && Number.isFinite(job.eval_loss) && (
                      <span>Eval loss: {job.eval_loss.toFixed(4)}</span>
                    )}
                  </div>
                ) : null}

                {job.loss != null && Number.isFinite(job.loss) && job.status === 'completed' && (
                  <div className="flex gap-4 mt-3 text-sm text-muted-foreground">
                    <span>Best eval loss: {job.loss.toFixed(4)}</span>
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
        </Card>
      )}

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
        <DialogContent className="max-w-lg">
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
                  <label className="block text-sm font-medium text-foreground mb-1">
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
                  <label className="block text-sm font-medium text-foreground mb-1">
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
                  <label className="block text-sm font-medium text-foreground mb-1">
                    Data Source
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
                    <option value="folder">Local folder (datasets/)</option>
                    <option value="manifest">v1 manifest file</option>
                    <option value="ref">Versioned dataset (id + version)</option>
                  </select>
                </div>

                {newJob.corpusMode === 'folder' && (
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <label className="block text-sm font-medium text-foreground">
                        Dataset <span className="text-destructive">*</span>
                      </label>
                      <button
                        type="button"
                        onClick={() => void fetchDatasets()}
                        className="text-xs text-muted-foreground hover:text-foreground"
                        disabled={loadingDatasets}
                      >
                        {loadingDatasets ? 'Loading...' : 'Refresh'}
                      </button>
                    </div>
                    {datasets.length > 0 ? (
                      <select
                        value={newJob.dataset}
                        onChange={(e) => setNewJob({ ...newJob, dataset: e.target.value })}
                        className="sl-input"
                      >
                        {datasets.map((ds) => (
                          <option key={ds.id} value={ds.id}>
                            {ds.name} ({ds.size}) - {ds.type}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <div className="sl-input py-2 text-sm text-muted-foreground">
                        No datasets found. Server may not be running from repo root.
                      </div>
                    )}
                    {datasets.find(d => d.id === newJob.dataset) && (
                      <p className="text-xs text-muted-foreground mt-1">
                        Path: <span className="font-mono">{datasets.find(d => d.id === newJob.dataset)?.path}</span>
                      </p>
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

              <FoldSection heading="Advanced (model dimensions, loop, trainer)">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-medium text-muted-foreground mb-1">n_embed</label>
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">n_layer</label>
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">n_head</label>
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">block_size</label>
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
                  <div>
                    <label className="block text-xs font-medium text-muted-foreground mb-1">
                      log_interval (progress / train loss UI)
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">
                      eval_interval (eval loss)
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

                  <div className="col-span-2 border-t border-border pt-3 mt-1 text-xs font-semibold text-muted-foreground">
                    Trainer (API / cli train --api parity)
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-muted-foreground mb-1">dropout</label>
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">weight_decay</label>
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">max_grad_norm</label>
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">grad_accum_steps</label>
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">warmup_steps</label>
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">min_lr</label>
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">scheduler</label>
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">AMP dtype</label>
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">lora_rank</label>
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">lora_alpha</label>
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">checkpoint_dir</label>
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">
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
                    <label className="block text-xs font-medium text-muted-foreground mb-1">max_checkpoints</label>
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
                <h3 className="text-sm font-semibold text-foreground">Training Parameters</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-1">Epochs</label>
                    <input
                      type="number"
                      value={newJob.epochs}
                      onChange={(e) => setNewJob({ ...newJob, epochs: parseInt(e.target.value, 10) })}
                      min={1}
                      max={100}
                      className="sl-input"
                    />
                    <p className="text-xs text-muted-foreground mt-1">Training passes</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-foreground mb-1">Batch Size</label>
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
                    <label className="block text-sm font-medium text-foreground mb-1">Learning Rate</label>
                    <input
                      type="number"
                      step="0.000001"
                      value={newJob.learning_rate}
                      onChange={(e) => setNewJob({ ...newJob, learning_rate: parseFloat(e.target.value) })}
                      className="sl-input"
                    />
                    <p className="text-xs text-muted-foreground mt-1">Step size</p>
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
                    <span className="text-muted-foreground">Model:</span>
                    <span className="font-medium">{newJob.model}</span>
                    <span className="text-muted-foreground">Dataset:</span>
                    <span className="font-medium">{newJob.dataset}</span>
                    <span className="text-muted-foreground">Epochs:</span>
                    <span className="font-medium">{newJob.epochs}</span>
                    <span className="text-muted-foreground">Batch Size:</span>
                    <span className="font-medium">{newJob.batch_size}</span>
                    <span className="text-muted-foreground">Learning Rate:</span>
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
