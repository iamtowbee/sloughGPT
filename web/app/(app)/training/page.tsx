'use client'

import { useState, useEffect } from 'react'
import { api, TrainingJob } from '@/lib/api'

export default function TrainingPage() {
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [loading, setLoading] = useState(true)
  const [showModal, setShowModal] = useState(false)
  const [newJob, setNewJob] = useState({ 
    name: '', 
    model: 'gpt2', 
    dataset: 'openwebtext',
    epochs: 3,
    batch_size: 8,
    learning_rate: 1e-5
  })
  const [starting, setStarting] = useState(false)

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

  const startTraining = async () => {
    if (!newJob.name.trim()) return
    
    setStarting(true)
    try {
      await api.startTraining({
        name: newJob.name,
        model: newJob.model,
        dataset: newJob.dataset,
        epochs: newJob.epochs,
        batch_size: newJob.batch_size,
        learning_rate: newJob.learning_rate,
      })
      setShowModal(false)
      setNewJob({ name: '', model: 'gpt2', dataset: 'openwebtext', epochs: 3, batch_size: 8, learning_rate: 1e-5 })
      fetchJobs()
    } catch (error) {
      console.error('Failed to start training:', error)
    } finally {
      setStarting(false)
    }
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-slate-800 dark:text-white">Training</h1>
        <button 
          onClick={() => setShowModal(true)}
          className="bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-4 py-2"
        >
          New Training Job
        </button>
      </div>

      {loading ? (
        <div className="text-center py-8 text-slate-500">Loading...</div>
      ) : jobs.length === 0 ? (
        <div className="text-center py-8 text-slate-500">No training jobs. Create one to get started.</div>
      ) : (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
          <div className="p-4 border-b border-slate-200 dark:border-slate-700">
            <h2 className="font-semibold text-slate-800 dark:text-white">Training Jobs</h2>
          </div>
          
          <div className="divide-y divide-slate-200 dark:divide-slate-700">
            {jobs.map((job) => (
              <div key={job.id} className="p-4">
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h3 className="font-medium text-slate-800 dark:text-white">{job.name}</h3>
                    <p className="text-sm text-slate-500">{job.model} • {job.dataset}</p>
                  </div>
                  <span className={`px-2 py-1 rounded text-xs ${
                    job.status === 'running' ? 'bg-blue-100 text-blue-700' :
                    job.status === 'completed' ? 'bg-green-100 text-green-700' :
                    job.status === 'pending' ? 'bg-yellow-100 text-yellow-700' :
                    'bg-red-100 text-red-700'
                  }`}>
                    {job.status}
                  </span>
                </div>
                
                <div className="mt-3">
                  <div className="flex justify-between text-sm text-slate-500 mb-1">
                    <span>Progress {job.current_epoch ? `(Epoch ${job.current_epoch}/${job.epochs})` : ''}</span>
                    <span>{job.progress}%</span>
                  </div>
                  <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all"
                      style={{ width: `${job.progress}%` }}
                    />
                  </div>
                </div>
                
                {job.loss !== undefined && job.loss !== null && (
                  <div className="flex gap-4 mt-3 text-sm text-slate-500">
                    <span>Loss: {job.loss.toFixed(4)}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {showModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-slate-800 rounded-xl p-6 w-full max-w-md">
            <h2 className="text-xl font-bold text-slate-800 dark:text-white mb-4">New Training Job</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-600 dark:text-slate-300 mb-1">
                  Job Name
                </label>
                <input
                  type="text"
                  value={newJob.name}
                  onChange={(e) => setNewJob({ ...newJob, name: e.target.value })}
                  placeholder="My Fine-tune"
                  className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg px-4 py-2 text-slate-800 dark:text-white"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-slate-600 dark:text-slate-300 mb-1">
                  Base Model
                </label>
                <select
                  value={newJob.model}
                  onChange={(e) => setNewJob({ ...newJob, model: e.target.value })}
                  className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg px-4 py-2 text-slate-800 dark:text-white"
                >
                  <option value="gpt2">GPT-2</option>
                  <option value="llama-2-7b">Llama-2-7B</option>
                  <option value="mistral-7b">Mistral-7B</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-600 dark:text-slate-300 mb-1">
                  Dataset
                </label>
                <select
                  value={newJob.dataset}
                  onChange={(e) => setNewJob({ ...newJob, dataset: e.target.value })}
                  className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg px-4 py-2 text-slate-800 dark:text-white"
                >
                  <option value="openwebtext">OpenWebText</option>
                  <option value="wikitext-103">WikiText-103</option>
                  <option value="code-search-net">CodeSearchNet</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-slate-600 dark:text-slate-300 mb-1">
                  Epochs
                </label>
                <input
                  type="number"
                  value={newJob.epochs}
                  onChange={(e) => setNewJob({ ...newJob, epochs: parseInt(e.target.value) })}
                  min={1}
                  max={100}
                  className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg px-4 py-2 text-slate-800 dark:text-white"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-600 dark:text-slate-300 mb-1">
                  Batch Size
                </label>
                <input
                  type="number"
                  value={newJob.batch_size}
                  onChange={(e) => setNewJob({ ...newJob, batch_size: parseInt(e.target.value) })}
                  min={1}
                  max={128}
                  className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg px-4 py-2 text-slate-800 dark:text-white"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-600 dark:text-slate-300 mb-1">
                  Learning Rate
                </label>
                <input
                  type="number"
                  step="1e-6"
                  value={newJob.learning_rate}
                  onChange={(e) => setNewJob({ ...newJob, learning_rate: parseFloat(e.target.value) })}
                  className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg px-4 py-2 text-slate-800 dark:text-white"
                />
              </div>
            </div>
            
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowModal(false)}
                className="flex-1 bg-slate-200 dark:bg-slate-700 text-slate-800 dark:text-white rounded-lg px-4 py-2"
              >
                Cancel
              </button>
              <button
                onClick={startTraining}
                disabled={starting || !newJob.name.trim()}
                className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white rounded-lg px-4 py-2"
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
