'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'

import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { inferenceHealthLabel, useApiHealth } from '@/hooks/useApiHealth'
import { api, type Dataset } from '@/lib/api'
import { devDebug } from '@/lib/dev-log'
import { DatasetImportModal } from '@/components/DatasetImportModal'
import { DatasetPreview } from '@/components/DatasetPreview'
import { cn } from '@/lib/cn'

function DatasetCardSkeleton() {
  return (
    <div className="animate-pulse rounded-xl bg-card/50 border border-border/50 overflow-hidden">
      <div className="p-5">
        <div className="flex items-start justify-between mb-3">
          <div className="h-6 w-32 bg-muted rounded"></div>
          <div className="h-5 w-16 bg-muted rounded-full"></div>
        </div>
        <div className="h-4 w-48 bg-muted rounded mb-2"></div>
        <div className="flex gap-2">
          <div className="h-5 w-16 bg-muted rounded-full"></div>
          <div className="h-5 w-20 bg-muted rounded-full"></div>
        </div>
      </div>
      <div className="px-5 py-3 border-t border-border/50 flex justify-between items-center bg-muted/30">
        <div className="h-4 w-20 bg-muted rounded"></div>
        <div className="flex gap-2">
          <div className="h-8 w-16 bg-muted rounded-lg"></div>
          <div className="h-8 w-16 bg-muted rounded-lg"></div>
        </div>
      </div>
    </div>
  )
}

function DatasetIcon({ type }: { type: string }) {
  const lowerType = (type || '').toLowerCase()
  if (lowerType.includes('hf') || lowerType.includes('hugging')) {
    return (
      <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-yellow-500 to-orange-500 flex items-center justify-center shadow-sm">
        <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 0L1.5 6v12L12 24l10.5-6V6L12 0zm0 2.25l8.25 4.5v9L12 20.75l-8.25-4.5v-9L12 2.25z"/>
        </svg>
      </div>
    )
  }
  if (lowerType.includes('json')) {
    return (
      <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center shadow-sm">
        <span className="text-white font-bold text-xs">JSON</span>
      </div>
    )
  }
  if (lowerType.includes('csv') || lowerType.includes('tabular')) {
    return (
      <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center shadow-sm">
        <span className="text-white font-bold text-xs">CSV</span>
      </div>
    )
  }
  return (
    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center shadow-sm">
      <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
      </svg>
    </div>
  )
}

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [importModalOpen, setImportModalOpen] = useState(false)
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [editingDataset, setEditingDataset] = useState<{ id: string; name: string; description: string; tags: string }>({ id: '', name: '', description: '', tags: '' })
  const [editModalOpen, setEditModalOpen] = useState(false)
  const [savingId, setSavingId] = useState<string | null>(null)
  const [versionsModalOpen, setVersionsModalOpen] = useState(false)
  const [versionsDatasetId, setVersionsDatasetId] = useState<string | null>(null)
  const [versions, setVersions] = useState<Array<{ version_id: string; created_at: string; description: string }>>([])
  const [loadingVersions, setLoadingVersions] = useState(false)
  const [createVersionDesc, setCreateVersionDesc] = useState('')
  const [batchModalOpen, setBatchModalOpen] = useState(false)
  const [batchSources, setBatchSources] = useState<Array<{ type: string; url: string; name: string }>>([{ type: 'url', url: '', name: '' }])
  const [batchResults, setBatchResults] = useState<{ successful: number; failed: number; results: Array<{ dataset_id: string; success: boolean; message: string }> } | null>(null)
  const { state: health, refresh: refreshHealth } = useApiHealth()

  const apiHealthLabel = useMemo(() => inferenceHealthLabel(health), [health])

  const healthToneClass = useMemo(() => {
    if (health === null) return 'text-muted-foreground'
    if (health === 'offline') return 'text-destructive'
    if (health.model_loaded) return 'text-success'
    return 'text-warning'
  }, [health])

  useEffect(() => {
    fetchDatasets()
  }, [])

  const fetchDatasets = async (query = '') => {
    setLoading(true)
    try {
      const rows = await api.getDatasets(query)
      setDatasets(rows)
    } catch (err) {
      devDebug('Failed to fetch datasets:', err)
      setDatasets([])
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    void fetchDatasets(searchQuery)
  }

  const handleImportComplete = () => {
    void fetchDatasets()
  }

  const handleViewDataset = (datasetId: string) => {
    setSelectedDataset(datasetId)
  }

  const handleDeleteDataset = async (datasetId: string) => {
    setDeletingId(datasetId)
    try {
      await api.deleteDataset(datasetId)
      setDatasets((prev) => prev.filter((d) => d.id !== datasetId))
      if (selectedDataset === datasetId) {
        setSelectedDataset(null)
      }
    } catch (err) {
      devDebug('Failed to delete dataset:', err)
      alert('Failed to delete dataset. Please try again.')
    } finally {
      setDeletingId(null)
    }
  }

  const handleExportDataset = async (datasetId: string, format: string) => {
    try {
      const result = await api.exportDataset(datasetId, format)
      const blob = new Blob([result.content], { type: 'text/plain' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${datasetId}.${format}`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      devDebug('Failed to export dataset:', err)
      alert('Failed to export dataset. Please try again.')
    }
  }

  const handleEditDataset = (dataset: Dataset) => {
    setEditingDataset({ id: dataset.id, name: dataset.name, description: '', tags: '' })
    setEditModalOpen(true)
  }

  const handleSaveEdit = async () => {
    if (!editingDataset.id) return
    setSavingId(editingDataset.id)
    try {
      await api.updateDataset(editingDataset.id, {
        name: editingDataset.name,
        description: editingDataset.description,
        tags: editingDataset.tags ? editingDataset.tags.split(',').map(t => t.trim()).filter(Boolean) : undefined,
      })
      void fetchDatasets()
      setEditModalOpen(false)
    } catch (err) {
      devDebug('Failed to update dataset:', err)
      alert('Failed to update dataset. Please try again.')
    } finally {
      setSavingId(null)
    }
  }

  const handleViewVersions = async (datasetId: string) => {
    setVersionsDatasetId(datasetId)
    setVersionsModalOpen(true)
    setLoadingVersions(true)
    try {
      const data = await api.listDatasetVersions(datasetId)
      setVersions(data.versions)
    } catch (err) {
      devDebug('Failed to load versions:', err)
      setVersions([])
    } finally {
      setLoadingVersions(false)
    }
  }

  const handleCreateVersion = async () => {
    if (!versionsDatasetId) return
    setSavingId(versionsDatasetId)
    try {
      await api.createDatasetVersion(versionsDatasetId, createVersionDesc)
      setCreateVersionDesc('')
      void handleViewVersions(versionsDatasetId)
    } catch (err) {
      devDebug('Failed to create version:', err)
      alert('Failed to create version.')
    } finally {
      setSavingId(null)
    }
  }

  const handleRollback = async (versionId: string) => {
    if (!versionsDatasetId) return
    if (!confirm('Rollback to this version? Current data will be replaced.')) return
    try {
      await api.rollbackDataset(versionsDatasetId, versionId)
      alert('Dataset rolled back successfully.')
      setVersionsModalOpen(false)
      void fetchDatasets()
    } catch (err) {
      devDebug('Failed to rollback:', err)
      alert('Failed to rollback.')
    }
  }

  const handleBatchImport = async () => {
    setSavingId('batch')
    setBatchResults(null)
    try {
      const sources = batchSources.filter(s => s.url.trim()).map(s => ({
        type: s.type as 'url' | 'local' | 'github',
        url: s.url,
        name: s.name || undefined,
      }))
      if (sources.length === 0) {
        alert('Please add at least one URL to import.')
        return
      }
      const result = await api.batchImport(sources)
      setBatchResults(result)
      if (result.successful > 0) {
        void fetchDatasets()
      }
    } catch (err) {
      devDebug('Batch import failed:', err)
      alert('Batch import failed.')
    } finally {
      setSavingId(null)
    }
  }

  const handleAddBatchSource = () => {
    setBatchSources(prev => [...prev, { type: 'url', url: '', name: '' }])
  }

  const handleRemoveBatchSource = (index: number) => {
    setBatchSources(prev => prev.filter((_, i) => i !== index))
  }

  return (
    <div className="sl-page mx-auto max-w-6xl">
      <AppRouteHeader
        className="mb-6 items-start"
        left={
          <AppRouteHeaderLead
            title="Datasets"
            subtitle={
              <>
                API:{' '}
                <span className={healthToneClass} data-testid="datasets-api-status">
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
              size="sm"
              onClick={() => setImportModalOpen(true)}
            >
              Import Dataset
            </Button>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={() => { setBatchModalOpen(true); setBatchResults(null); }}
            >
              Batch Import
            </Button>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={() => {
                void fetchDatasets()
                void refreshHealth()
              }}
            >
              Refresh
            </Button>
            <Button type="button" size="sm" variant="outline" asChild>
              <Link href="/training">Start training</Link>
            </Button>
          </div>
        }
      />

      <Tabs value={selectedDataset ? 'preview' : 'list'} onValueChange={(v) => v === 'list' && setSelectedDataset(null)}>
        <TabsList className="mb-4 w-full justify-start">
          <TabsTrigger value="list">All Datasets</TabsTrigger>
          {selectedDataset && (
            <TabsTrigger value="preview">{selectedDataset} Preview</TabsTrigger>
          )}
        </TabsList>

        <form onSubmit={handleSearch} className="mb-6 flex gap-2">
          <Input
            type="search"
            placeholder="Search datasets..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="max-w-xs"
          />
          <Button type="submit" variant="outline" size="sm">
            Search
          </Button>
          {searchQuery && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => {
                setSearchQuery('')
                void fetchDatasets()
              }}
            >
              Clear
            </Button>
          )}
        </form>

        <TabsContent value="list">
          {loading ? (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <DatasetCardSkeleton key={i} />
              ))}
            </div>
          ) : datasets.length === 0 ? (
            <Card className="border-dashed">
              <CardContent className="py-12 text-center">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-muted/50 flex items-center justify-center">
                  <svg className="w-8 h-8 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                  </svg>
                </div>
                <p className="text-muted-foreground mb-4">No datasets found.</p>
                <Button type="button" onClick={() => setImportModalOpen(true)}>
                  Import your first dataset
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
              {datasets.map((dataset) => (
                <Card
                  key={dataset.id}
                  className={cn(
                    "group transition-all duration-300 hover:shadow-lg hover:-translate-y-1",
                    "border-border/60 hover:border-primary/30"
                  )}
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-start gap-3">
                      <DatasetIcon type={dataset.type} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between gap-2">
                          <CardTitle className="text-base truncate">{dataset.name}</CardTitle>
                          <span className="shrink-0 px-2 py-0.5 text-xs font-medium rounded-full border border-border/60 bg-muted/50 text-muted-foreground">
                            {dataset.type}
                          </span>
                        </div>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="pt-0">
                    {dataset.path && (
                      <p className="truncate font-mono text-xs text-muted-foreground mb-3" title={dataset.path}>
                        {dataset.path}
                      </p>
                    )}
                    {dataset.samples && (
                      <div className="flex items-center gap-3 text-xs text-muted-foreground">
                        <span className="flex items-center gap-1">
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                          {dataset.samples.toLocaleString()} samples
                        </span>
                      </div>
                    )}
                  </CardContent>
                  <CardFooter className="flex justify-between items-center border-t border-border/50 pt-4">
                    <span className="text-sm font-medium text-muted-foreground">{dataset.size}</span>
                    <div className="flex gap-1">
                        <Button type="button" variant="ghost" size="sm" onClick={() => handleViewDataset(dataset.id)}>
                          View
                        </Button>
                        <Button type="button" variant="ghost" size="sm" onClick={() => handleEditDataset(dataset)}>
                          Edit
                        </Button>
                        <Button type="button" variant="ghost" size="sm" onClick={() => handleViewVersions(dataset.id)}>
                          Versions
                        </Button>
                        <div className="relative group">
                        <Button type="button" variant="ghost" size="sm">
                          Export
                        </Button>
                        <div className="absolute right-0 top-full z-10 hidden min-w-[120px] rounded-md border border-border bg-background py-1 shadow-lg group-hover:block">
                          <button
                            type="button"
                            className="w-full px-4 py-2 text-left sl-text-body hover:bg-muted"
                            onClick={() => handleExportDataset(dataset.id, 'json')}
                          >
                            JSON
                          </button>
                          <button
                            type="button"
                            className="w-full px-4 py-2 text-left sl-text-body hover:bg-muted"
                            onClick={() => handleExportDataset(dataset.id, 'jsonl')}
                          >
                            JSONL
                          </button>
                          <button
                            type="button"
                            className="w-full px-4 py-2 text-left sl-text-body hover:bg-muted"
                            onClick={() => handleExportDataset(dataset.id, 'csv')}
                          >
                            CSV
                          </button>
                          <button
                            type="button"
                            className="w-full px-4 py-2 text-left sl-text-body hover:bg-muted"
                            onClick={() => handleExportDataset(dataset.id, 'txt')}
                          >
                            TXT
                          </button>
                        </div>
                      </div>
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button type="button" variant="ghost" size="sm" className="text-destructive hover:text-destructive">
                            Delete
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>Delete dataset?</AlertDialogTitle>
                            <AlertDialogDescription>
                              This will permanently delete &quot;{dataset.name}&quot;. This action cannot be undone.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>Cancel</AlertDialogCancel>
                            <AlertDialogAction
                              onClick={() => void handleDeleteDataset(dataset.id)}
                              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                              disabled={deletingId === dataset.id}
                            >
                              {deletingId === dataset.id ? 'Deleting…' : 'Delete'}
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
                  </CardFooter>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        {selectedDataset && (
          <TabsContent value="preview">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setSelectedDataset(null)}
              className="mb-4"
            >
              ← Back to datasets
            </Button>
            <DatasetPreview
              datasetId={selectedDataset}
              onUseForTraining={() => {
                window.location.href = `/training?dataset=${encodeURIComponent(selectedDataset)}`
              }}
            />
          </TabsContent>
        )}
      </Tabs>

      <Card className="mt-8">
        <CardHeader>
          <CardTitle className="text-base">CLI</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1 font-mono text-sm text-muted-foreground">
          <p>python3 cli.py data validate datasets/my_data/</p>
          <p>python3 cli.py data stats datasets/my_data/</p>
          <p>python3 cli.py data split datasets/my_data/ --train 0.9</p>
        </CardContent>
      </Card>

      <DatasetImportModal
        open={importModalOpen}
        onOpenChange={setImportModalOpen}
        onImportComplete={handleImportComplete}
      />

      <Dialog open={editModalOpen} onOpenChange={setEditModalOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Dataset</DialogTitle>
            <DialogDescription>
              Update dataset name, description, and tags.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
              <label className="block text-sm font-medium mb-1">Name</label>
              <Input
                value={editingDataset.name}
                onChange={(e) => setEditingDataset(prev => ({ ...prev, name: e.target.value }))}
                placeholder="Dataset name"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Description</label>
              <Input
                value={editingDataset.description}
                onChange={(e) => setEditingDataset(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Optional description"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Tags (comma-separated)</label>
              <Input
                value={editingDataset.tags}
                onChange={(e) => setEditingDataset(prev => ({ ...prev, tags: e.target.value }))}
                placeholder="e.g., text, shakespeare, training"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditModalOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleSaveEdit} disabled={savingId !== null}>
              {savingId ? 'Saving...' : 'Save Changes'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={versionsModalOpen} onOpenChange={setVersionsModalOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Dataset Versions</DialogTitle>
            <DialogDescription>
              Create snapshots to save versions of your dataset. You can rollback to any previous version.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="flex gap-2">
              <Input
                value={createVersionDesc}
                onChange={(e) => setCreateVersionDesc(e.target.value)}
                placeholder="Version description (optional)"
                className="flex-1"
              />
              <Button onClick={handleCreateVersion} disabled={savingId !== null}>
                {savingId ? 'Creating...' : 'Create Snapshot'}
              </Button>
            </div>
            <div className="max-h-[300px] overflow-auto space-y-2">
              {loadingVersions ? (
                <p className="text-center text-muted-foreground py-4">Loading versions...</p>
              ) : versions.length === 0 ? (
                <p className="text-center text-muted-foreground py-4">No versions yet. Create a snapshot to save your data.</p>
              ) : (
                versions.map((v) => (
                  <div key={v.version_id} className="flex items-center justify-between p-3 rounded-lg bg-muted/30">
                    <div>
                      <p className="font-medium text-sm">{v.version_id}</p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(v.created_at).toLocaleString()} • {v.description}
                      </p>
                    </div>
                    <Button size="sm" variant="outline" onClick={() => handleRollback(v.version_id)}>
                      Rollback
                    </Button>
                  </div>
                ))
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={batchModalOpen} onOpenChange={setBatchModalOpen}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-auto">
          <DialogHeader>
            <DialogTitle>Batch Import</DialogTitle>
            <DialogDescription>
              Import multiple datasets at once by URLs. Up to 20 imports per batch.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            {batchSources.map((source, index) => (
              <div key={index} className="flex gap-2 items-start">
                <select
                  value={source.type}
                  onChange={(e) => {
                    const newSources = [...batchSources]
                    newSources[index].type = e.target.value
                    setBatchSources(newSources)
                  }}
                  className="sl-input py-2 px-2 w-28"
                >
                  <option value="url">URL</option>
                  <option value="local">Local</option>
                  <option value="github">GitHub</option>
                </select>
                <Input
                  value={source.url}
                  onChange={(e) => {
                    const newSources = [...batchSources]
                    newSources[index].url = e.target.value
                    setBatchSources(newSources)
                  }}
                  placeholder={source.type === 'github' ? 'owner/repo/path' : 'https://...'}
                  className="flex-1"
                />
                <Input
                  value={source.name}
                  onChange={(e) => {
                    const newSources = [...batchSources]
                    newSources[index].name = e.target.value
                    setBatchSources(newSources)
                  }}
                  placeholder="Name (optional)"
                  className="w-32"
                />
                <Button variant="ghost" size="sm" onClick={() => handleRemoveBatchSource(index)}>
                  ✕
                </Button>
              </div>
            ))}
            <Button variant="outline" size="sm" onClick={handleAddBatchSource}>
              + Add Source
            </Button>

            {batchResults && (
              <div className="mt-4 p-4 rounded-lg bg-muted/30">
                <h4 className="font-medium mb-2">Results</h4>
                <p className="text-sm">
                  <span className="text-green-600 font-medium">{batchResults.successful}</span> successful,{' '}
                  <span className="text-red-600 font-medium">{batchResults.failed}</span> failed
                </p>
                {batchResults.results.filter(r => r.success).map((r) => (
                  <p key={r.dataset_id} className="text-sm text-green-600">
                    ✓ {r.dataset_id}
                  </p>
                ))}
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setBatchModalOpen(false)}>
              Close
            </Button>
            <Button onClick={handleBatchImport} disabled={savingId !== null}>
              {savingId ? 'Importing...' : 'Import All'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
