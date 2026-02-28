import React, { useState, useEffect } from 'react'
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Spinner,
  Badge
} from '@base-ui/react'
import { useStore, TrainingJob, TrainingConfig } from '../store'
import { api } from '../utils/api'

interface TrainingMetrics {
  job_id: string
  chart_data: {
    epochs: number[]
    loss: { train: number[]; validation: number[] }
    learning_rate: number[]
  }
  stats: {
    total_epochs: number
    best_train_loss: number | null
    best_val_loss: number | null
    avg_train_loss: number | null
    avg_val_loss: number | null
    final_learning_rate: number | null
  }
}

export const Training: React.FC = () => {
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [showCreateDialog, setShowCreateDialog] = useState(false)
  const [selectedJob, setSelectedJob] = useState<TrainingJob | null>(null)
  const [jobMetrics, setJobMetrics] = useState<TrainingMetrics | null>(null)
  const [activeTab, setActiveTab] = useState<'jobs' | 'create' | 'details'>('jobs')
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [config, setConfig] = useState<TrainingConfig>({
    dataset_name: '',
    model_id: 'nanogpt',
    epochs: 3,
    batch_size: 8,
    learning_rate: 0.0001,
    vocab_size: 500,
    n_embed: 128,
    n_layer: 3,
    n_head: 4,
    optimizer: 'adam',
    scheduler: 'cosine',
    validation_split: 0.1,
    early_stopping_patience: 5,
    save_checkpoint_every: 1,
    gradient_clip: 1.0,
    warmup_steps: 100,
    weight_decay: 0.01
  })
  
  const { datasets, models, setTrainingJobs, addTrainingJob, updateTrainingJob } = useStore()

  useEffect(() => {
    loadTrainingJobs()
  }, [])

  useEffect(() => {
    // Poll for job updates
    const interval = setInterval(() => {
      loadTrainingJobs()
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  const loadTrainingJobs = async () => {
    try {
      const response = await api.listTrainingJobs()
      if (response.data) {
        setJobs(response.data.jobs)
        setTrainingJobs(response.data.jobs)
      }
    } catch (error) {
      console.error('Error loading training jobs:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleCreateJob = async () => {
    if (!config.dataset_name) {
      alert('Please select a dataset')
      return
    }

    try {
      const response = await api.createTrainingJob(config)
      if (response.data) {
        addTrainingJob(response.data)
        setActiveTab('jobs')
        loadTrainingJobs()
      }
    } catch (error) {
      console.error('Error creating training job:', error)
      alert('Error creating training job')
    }
  }

  const handleCancelJob = async (jobId: string) => {
    if (!confirm('Are you sure you want to cancel this training job?')) return

    try {
      await api.cancelTrainingJob(jobId)
      loadTrainingJobs()
    } catch (error) {
      console.error('Error cancelling training job:', error)
    }
  }

  const handleViewDetails = async (job: TrainingJob) => {
    setSelectedJob(job)
    setActiveTab('details')
    try {
      const response = await api.getTrainingJob(job.id)
      if (response.data) {
        setSelectedJob(response.data)
      }
      const metricsRes = await api.getTrainingLogs(job.id)
      if (metricsRes.data) {
        setJobMetrics(metricsRes.data as unknown as TrainingMetrics)
      }
    } catch (error) {
      console.error('Error loading job details:', error)
    }
  }

  const handleRestartJob = async (jobId: string) => {
    try {
      const response = await api.restartTrainingJob(jobId)
      if (response.data) {
        alert(`Job restarted! New job ID: ${response.data.new_job_id}`)
        loadTrainingJobs()
        setActiveTab('jobs')
      }
    } catch (error) {
      console.error('Error restarting job:', error)
      alert('Error restarting job')
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800'
      case 'running': return 'bg-blue-100 text-blue-800'
      case 'pending': return 'bg-yellow-100 text-yellow-800'
      case 'failed': return 'bg-red-100 text-red-800'
      case 'cancelled': return 'bg-gray-100 text-gray-800'
      case 'early_stopped': return 'bg-orange-100 text-orange-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return '✓'
      case 'running': return '⟳'
      case 'pending': return '⏳'
      case 'failed': return '✕'
      case 'cancelled': return '⊘'
      case 'early_stopped': return '⏹'
      default: return '?'
    }
  }

  const renderJobCard = (job: TrainingJob) => (
    <Card 
      key={job.id} 
      className="mb-4 cursor-pointer hover:shadow-md transition-shadow"
      onClick={() => handleViewDetails(job)}
    >
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-lg">{getStatusIcon(job.status)}</span>
            <span className="font-medium">{job.dataset_name}</span>
          </div>
          <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(job.status)}`}>
            {job.status}
          </span>
        </CardTitle>
        <CardDescription className="flex items-center gap-4">
          <span>Model: {job.model_id}</span>
          <span>•</span>
          <span>Epochs: {job.total_epochs}</span>
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-slate-500">Progress</span>
              <span className="font-medium">{job.progress?.toFixed(1) ?? 0}%</span>
            </div>
            <div className="w-full bg-slate-200 rounded-full h-2.5">
              <div 
                className={`h-2.5 rounded-full transition-all ${
                  job.status === 'completed' ? 'bg-green-500' : 
                  job.status === 'running' ? 'bg-blue-500 animate-pulse' : 
                  'bg-slate-400'
                }`}
                style={{ width: `${job.progress ?? 0}%` }}
              />
            </div>
          </div>
          
          <div className="grid grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-slate-500 block text-xs">Epoch</span>
              <span className="font-medium">{job.current_epoch ?? 0}/{job.total_epochs}</span>
            </div>
            <div>
              <span className="text-slate-500 block text-xs">Loss</span>
              <span className="font-medium">{(job.loss ?? 0).toFixed(4)}</span>
            </div>
            <div>
              <span className="text-slate-500 block text-xs">Val Loss</span>
              <span className="font-medium">{(job.val_loss ?? 0).toFixed(4)}</span>
            </div>
            <div>
              <span className="text-slate-500 block text-xs">Best</span>
              <span className="font-medium text-green-600">{(job.best_loss ?? 0).toFixed(4)}</span>
            </div>
          </div>
          
          {job.status === 'running' && (
            <Button 
              variant="outline" 
              size="sm"
              onClick={(e) => {
                e.stopPropagation()
                handleCancelJob(job.id)
              }}
              className="w-full border-red-200 text-red-600 hover:bg-red-50"
            >
              Cancel Training
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )

  const filteredJobs = statusFilter 
    ? jobs.filter(j => j.status === statusFilter)
    : jobs

  return (
    <div className="space-y-6">
      {/* Header with Tabs */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-xl font-semibold">Training Jobs</CardTitle>
            <div className="flex gap-2">
              <Button 
                variant={activeTab === 'jobs' ? 'solid' : 'outline'} 
                size="sm"
                onClick={() => setActiveTab('jobs')}
              >
                Jobs
              </Button>
              <Button 
                variant={activeTab === 'create' ? 'solid' : 'outline'} 
                size="sm"
                onClick={() => setActiveTab('create')}
              >
                New Job
              </Button>
            </div>
          </div>
          <CardDescription>
            Manage and monitor model training jobs
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Job Details View */}
      {activeTab === 'details' && selectedJob && (
        <div className="space-y-4">
          <Button variant="outline" size="sm" onClick={() => setActiveTab('jobs')}>
            ← Back to Jobs
          </Button>
          
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>{selectedJob.dataset_name}</CardTitle>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(selectedJob.status)}`}>
                  {selectedJob.status}
                </span>
              </div>
              <CardDescription>
                Model: {selectedJob.model_id} • Epochs: {selectedJob.total_epochs}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-slate-50 p-4 rounded-lg">
                  <div className="text-slate-500 text-sm">Progress</div>
                  <div className="text-2xl font-bold">{selectedJob.progress?.toFixed(1) ?? 0}%</div>
                </div>
                <div className="bg-slate-50 p-4 rounded-lg">
                  <div className="text-slate-500 text-sm">Current Loss</div>
                  <div className="text-2xl font-bold">{(selectedJob.loss ?? 0).toFixed(4)}</div>
                </div>
                <div className="bg-slate-50 p-4 rounded-lg">
                  <div className="text-slate-500 text-sm">Val Loss</div>
                  <div className="text-2xl font-bold">{(selectedJob.val_loss ?? 0).toFixed(4)}</div>
                </div>
                <div className="bg-slate-50 p-4 rounded-lg">
                  <div className="text-slate-500 text-sm">Best Loss</div>
                  <div className="text-2xl font-bold text-green-600">{(selectedJob.best_loss ?? 0).toFixed(4)}</div>
                </div>
              </div>

              {jobMetrics && (
                <div className="space-y-4">
                  <h4 className="font-semibold">Training Metrics</h4>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div>
                      <div className="text-slate-500 text-sm">Best Train Loss</div>
                      <div className="font-medium">{jobMetrics.stats.best_train_loss?.toFixed(4) ?? '-'}</div>
                    </div>
                    <div>
                      <div className="text-slate-500 text-sm">Best Val Loss</div>
                      <div className="font-medium">{jobMetrics.stats.best_val_loss?.toFixed(4) ?? '-'}</div>
                    </div>
                    <div>
                      <div className="text-slate-500 text-sm">Total Steps</div>
                      <div className="font-medium">{selectedJob.total_steps ?? 0}</div>
                    </div>
                    <div>
                      <div className="text-slate-500 text-sm">Optimizer</div>
                      <div className="font-medium">{selectedJob.optimizer ?? 'adam'}</div>
                    </div>
                    <div>
                      <div className="text-slate-500 text-sm">Scheduler</div>
                      <div className="font-medium">{selectedJob.scheduler ?? 'none'}</div>
                    </div>
                    <div>
                      <div className="text-slate-500 text-sm">Learning Rate</div>
                      <div className="font-medium">{(selectedJob.learning_rate ?? 0).toFixed(6)}</div>
                    </div>
                  </div>

                  {/* Loss Chart Visualization */}
                  {jobMetrics.chart_data && jobMetrics.chart_data.epochs.length > 0 && (
                    <div className="mt-6">
                      <h5 className="font-medium mb-2">Loss Over Epochs</h5>
                      <div className="h-48 flex items-end gap-2">
                        {jobMetrics.chart_data.loss.train.map((loss, i) => (
                          <div key={i} className="flex-1 flex flex-col items-center gap-1">
                            <div 
                              className="w-full bg-blue-500 rounded-t"
                              style={{ height: `${Math.min(100, loss * 20)}px` }}
                              title={`Train: ${loss.toFixed(4)}`}
                            />
                            <div 
                              className="w-full bg-orange-400 rounded-t"
                              style={{ height: `${Math.min(100, (jobMetrics.chart_data.loss.validation[i] ?? 0) * 20)}px` }}
                              title={`Val: ${jobMetrics.chart_data.loss.validation[i]?.toFixed(4)}`}
                            />
                            <span className="text-xs text-slate-500">E{i + 1}</span>
                          </div>
                        ))}
                      </div>
                      <div className="flex gap-4 mt-2 text-xs">
                        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-blue-500 rounded"></span> Train</span>
                        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-orange-400 rounded"></span> Validation</span>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Checkpoints */}
              {selectedJob.checkpoints && selectedJob.checkpoints.length > 0 && (
                <div className="mt-6">
                  <h4 className="font-semibold mb-2">Checkpoints ({selectedJob.checkpoints.length})</h4>
                  <div className="space-y-2">
                    {selectedJob.checkpoints.map((cp, i) => (
                      <div key={i} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg text-sm">
                        <span>Epoch {cp.epoch}</span>
                        <span>Loss: {(cp.loss ?? 0).toFixed(4)}</span>
                        <span className="text-slate-500">{cp.path}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {selectedJob.status === 'running' && (
                <Button 
                  variant="outline" 
                  className="mt-4 border-red-200 text-red-600 hover:bg-red-50"
                  onClick={() => handleCancelJob(selectedJob.id)}
                >
                  Cancel Training
                </Button>
              )}

              {(selectedJob.status === 'completed' || selectedJob.status === 'cancelled' || selectedJob.status === 'early_stopped' || selectedJob.status === 'failed') && (
                <Button 
                  className="mt-4"
                  onClick={() => handleRestartJob(selectedJob.id)}
                >
                  Restart Job
                </Button>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Job List View */}
      {activeTab === 'jobs' && (
        <>
          {/* Filter */}
          <div className="flex gap-2">
            <Button 
              variant={statusFilter === '' ? 'solid' : 'outline'} 
              size="sm"
              onClick={() => setStatusFilter('')}
            >
              All ({jobs.length})
            </Button>
            <Button 
              variant={statusFilter === 'running' ? 'solid' : 'outline'} 
              size="sm"
              onClick={() => setStatusFilter('running')}
            >
              Running ({jobs.filter(j => j.status === 'running').length})
            </Button>
            <Button 
              variant={statusFilter === 'completed' ? 'solid' : 'outline'} 
              size="sm"
              onClick={() => setStatusFilter('completed')}
            >
              Completed ({jobs.filter(j => j.status === 'completed').length})
            </Button>
          </div>

          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <Spinner className="h-8 w-8 text-blue-500" size="8" />
            </div>
          ) : filteredJobs.length === 0 ? (
            <Card className="h-64 flex items-center justify-center text-slate-400">
              <div className="text-center">
                <div className="text-4xl mb-4">🧠</div>
                <p>No training jobs found</p>
                <p className="text-sm">Create your first training job to get started</p>
              </div>
            </Card>
          ) : (
            <div>
              {filteredJobs.map(renderJobCard)}
            </div>
          )}
        </>
      )}

      {/* Create Job View */}
      {activeTab === 'create' && (
        <Card className="max-h-[90vh] overflow-y-auto">
          <CardHeader>
            <CardTitle>Create Training Job</CardTitle>
            <CardDescription>
              Configure your training parameters
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {/* Basic Settings */}
              <div>
                <h4 className="font-medium text-slate-700 mb-3">Basic Settings</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">Dataset</label>
                    <select
                      value={config.dataset_name}
                      onChange={(e) => setConfig({ ...config, dataset_name: e.target.value })}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    >
                      <option value="">Select a dataset</option>
                      {datasets.map(ds => (
                        <option key={ds.name} value={ds.name}>{ds.name}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Model</label>
                    <select
                      value={config.model_id}
                      onChange={(e) => setConfig({ ...config, model_id: e.target.value })}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    >
                      {models.map(m => (
                        <option key={m.id} value={m.id}>{m.name} ({m.provider})</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* Training Parameters */}
              <div>
                <h4 className="font-medium text-slate-700 mb-3">Training Parameters</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">Epochs</label>
                    <input
                      type="number"
                      value={config.epochs}
                      onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                      min="1"
                      max="100"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Batch Size</label>
                    <input
                      type="number"
                      value={config.batch_size}
                      onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                      min="1"
                      max="256"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Learning Rate</label>
                    <input
                      type="number"
                      step="0.0001"
                      value={config.learning_rate}
                      onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Weight Decay</label>
                    <input
                      type="number"
                      step="0.01"
                      value={config.weight_decay}
                      onChange={(e) => setConfig({ ...config, weight_decay: parseFloat(e.target.value) })}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                      min="0"
                      max="0.5"
                    />
                  </div>
                </div>
              </div>

              {/* Model Architecture */}
              <div>
                <h4 className="font-medium text-slate-700 mb-3">Model Architecture</h4>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">Vocab Size</label>
                    <input
                      type="number"
                      value={config.vocab_size}
                      onChange={(e) => setConfig({ ...config, vocab_size: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Embed Dim</label>
                    <input
                      type="number"
                      value={config.n_embed}
                      onChange={(e) => setConfig({ ...config, n_embed: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Layers</label>
                    <input
                      type="number"
                      value={config.n_layer}
                      onChange={(e) => setConfig({ ...config, n_layer: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Heads</label>
                    <input
                      type="number"
                      value={config.n_head}
                      onChange={(e) => setConfig({ ...config, n_head: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Grad Clip</label>
                    <input
                      type="number"
                      step="0.1"
                      value={config.gradient_clip}
                      onChange={(e) => setConfig({ ...config, gradient_clip: parseFloat(e.target.value) })}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    />
                  </div>
                </div>
              </div>

              {/* Advanced Settings */}
              <div>
                <h4 className="font-medium text-slate-700 mb-3">Advanced Settings</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">Optimizer</label>
                    <select
                      value={config.optimizer}
                      onChange={(e) => setConfig({ ...config, optimizer: e.target.value })}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    >
                      <option value="adam">Adam</option>
                      <option value="sgd">SGD</option>
                      <option value="adamw">AdamW</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Scheduler</label>
                    <select
                      value={config.scheduler}
                      onChange={(e) => setConfig({ ...config, scheduler: e.target.value })}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                    >
                      <option value="cosine">Cosine</option>
                      <option value="step">Step</option>
                      <option value="exponential">Exponential</option>
                      <option value="None">None</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Early Stopping</label>
                    <input
                      type="number"
                      value={config.early_stopping_patience}
                      onChange={(e) => setConfig({ ...config, early_stopping_patience: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                      min="0"
                      max="20"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Save Every</label>
                    <input
                      type="number"
                      value={config.save_checkpoint_every}
                      onChange={(e) => setConfig({ ...config, save_checkpoint_every: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                      min="1"
                      max="10"
                    />
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
          <div className="flex justify-end gap-2 p-4 border-t">
            <Button variant="outline" onClick={() => setActiveTab('jobs')}>
              Cancel
            </Button>
            <Button onClick={handleCreateJob}>
              Start Training
            </Button>
          </div>
        </Card>
      )}
    </div>
  )
}

export default Training
