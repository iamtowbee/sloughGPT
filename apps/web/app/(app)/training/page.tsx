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
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { inferenceHealthLabel, useApiHealth } from '@/hooks/useApiHealth'
import { api, TrainingJob, TrainResolveResponse } from '@/lib/api'
import { devDebug } from '@/lib/dev-log'
import {
  TRAINING_API_DEFAULTS,
  type TrainingMixedPrecisionDtype,
} from '@/lib/training-defaults'

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
  const [showAdvanced, setShowAdvanced] = useState(false)
  const { state: health, refresh: refreshHealth } = useApiHealth()

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
  }, [fetchJobs])

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
      fetchJobs()
    } catch (error) {
      devDebug('Failed to start training:', error)
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
                  <div className="w-full bg-muted rounded-none h-2">
                    <div
                      className="bg-primary h-2 rounded-none transition-all duration-200 ease-smooth"
                      style={{ width: `${job.progress}%` }}
                    />
                  </div>
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
            setShowAdvanced(false)
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
                  Model label
                </label>
                <input
                  type="text"
                  value={newJob.model}
                  onChange={(e) => setNewJob({ ...newJob, model: e.target.value })}
                  placeholder="sloughgpt"
                  className="sl-input font-mono text-sm"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  For your records and the job card only. The API always trains{' '}
                  <span className="font-mono text-foreground/90">SloughGPTModel</span> (char-level on the resolved corpus).
                </p>
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
                    <option value="shakespeare">Shakespeare (bundled demo)</option>
                    <option value="demo">demo</option>
                    <option value="summary">summary</option>
                    <option value="openwebtext">openwebtext (add data first)</option>
                    <option value="wikitext-103">wikitext-103 (add data first)</option>
                    <option value="code-search-net">code-search-net (add data first)</option>
                  </select>
                  <p className="text-xs text-muted-foreground mt-1">
                    Server resolves <span className="font-mono text-foreground/80">datasets/&lt;name&gt;/input.txt</span>{' '}
                    from the API process working directory (run the API from the repo root).
                  </p>
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

              <button
                type="button"
                onClick={() => setShowAdvanced((v) => !v)}
                className="text-sm text-primary hover:underline"
              >
                {showAdvanced ? 'Hide' : 'Show'} advanced (model, loop, trainer)
              </button>

              {showAdvanced && (
                <div className="grid grid-cols-2 gap-3 p-3 rounded-none border border-border">
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

            <DialogFooter className="mt-2 gap-3 sm:gap-3">
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
                className="flex-1 sm:flex-none"
                onClick={startTraining}
                disabled={starting || !newJob.name.trim()}
              >
                {starting ? 'Starting...' : 'Start Training'}
              </Button>
            </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
