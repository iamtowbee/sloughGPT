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
import { DatasetCombineModal } from '@/components/DatasetCombineModal'
import { DatasetPreview } from '@/components/DatasetPreview'

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [importModalOpen, setImportModalOpen] = useState(false)
  const [combineModalOpen, setCombineModalOpen] = useState(false)
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [editingDataset, setEditingDataset] = useState<{ id: string; name: string; description: string; tags: string }>({ id: '', name: '', description: '', tags: '' })
  const [editModalOpen, setEditModalOpen] = useState(false)
  const [savingId, setSavingId] = useState<string | null>(null)
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
              variant="outline"
              size="sm"
              onClick={() => setCombineModalOpen(true)}
              disabled={datasets.length < 2}
            >
              Combine ({datasets.length})
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

        <form onSubmit={handleSearch} className="mb-4 flex gap-2">
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
            <div className="py-12 text-center text-muted-foreground">Loading datasets…</div>
          ) : datasets.length === 0 ? (
            <Card className="border-dashed">
              <CardContent className="py-12 text-center text-muted-foreground">
                <p className="mb-4">No datasets found.</p>
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
                  className="transition-colors duration-200 ease-smooth hover:border-primary/25"
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-start justify-between gap-2">
                      <CardTitle className="text-base">{dataset.name}</CardTitle>
                      <span className="shrink-0 border border-border bg-muted/50 px-2 py-0.5 font-mono text-xs text-muted-foreground">
                        {dataset.type}
                      </span>
                    </div>
                  </CardHeader>
                  <CardContent className="pt-0">
                    {dataset.path && (
                      <p className="truncate font-mono text-xs text-muted-foreground">{dataset.path}</p>
                    )}
                  </CardContent>
                  <CardFooter className="flex justify-between border-t border-border pt-4">
                    <span className="text-sm font-medium text-chart-3">{dataset.size}</span>
                      <div className="flex gap-1">
                        <Button type="button" variant="ghost" size="sm" onClick={() => handleViewDataset(dataset.id)}>
                          View
                        </Button>
                        <Button type="button" variant="ghost" size="sm" onClick={() => handleEditDataset(dataset)}>
                          Edit
                        </Button>
                        <div className="relative group">
                        <Button type="button" variant="ghost" size="sm">
                          Export
                        </Button>
                        <div className="absolute right-0 top-full z-10 hidden min-w-[120px] rounded-md border bg-background py-1 shadow-lg group-hover:block">
                          <button
                            type="button"
                            className="w-full px-4 py-2 text-left text-sm hover:bg-muted"
                            onClick={() => handleExportDataset(dataset.id, 'json')}
                          >
                            JSON
                          </button>
                          <button
                            type="button"
                            className="w-full px-4 py-2 text-left text-sm hover:bg-muted"
                            onClick={() => handleExportDataset(dataset.id, 'jsonl')}
                          >
                            JSONL
                          </button>
                          <button
                            type="button"
                            className="w-full px-4 py-2 text-left text-sm hover:bg-muted"
                            onClick={() => handleExportDataset(dataset.id, 'csv')}
                          >
                            CSV
                          </button>
                          <button
                            type="button"
                            className="w-full px-4 py-2 text-left text-sm hover:bg-muted"
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
      <DatasetCombineModal
        open={combineModalOpen}
        onOpenChange={setCombineModalOpen}
        datasets={datasets}
        onCombineComplete={handleImportComplete}
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
    </div>
  )
}
