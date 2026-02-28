import React, { useState, useEffect } from 'react'
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Spinner
} from '@base-ui/react'
import { useStore, Dataset } from '../store'
import { api } from '../utils/api'

interface DatasetsProps {}

export const Datasets: React.FC<DatasetsProps> = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [showCreateDialog, setShowCreateDialog] = useState(false)
  const [newDatasetName, setNewDatasetName] = useState('')
  const [newDatasetContent, setNewDatasetContent] = useState('')
  const [newDatasetDescription, setNewDatasetDescription] = useState('')
  
  const { setDatasets: updateStoreDatasets, datasets: storeDatasets } = useStore()

  useEffect(() => {
    loadDatasets()
  }, [])

  const loadDatasets = async () => {
    setIsLoading(true)
    try {
      const response = await api.listDatasets()
      if (response.data) {
        setDatasets(response.data.datasets)
        updateStoreDatasets(response.data.datasets)
      }
    } catch (error) {
      console.error('Error loading datasets:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleCreateDataset = async () => {
    if (!newDatasetName.trim() || !newDatasetContent.trim()) {
      alert('Please provide both name and content for the dataset')
      return
    }

    try {
      const response = await api.createDataset(
        newDatasetName,
        newDatasetContent,
        newDatasetDescription || undefined
      )
      
      if (response.data) {
        setDatasets(prev => [response.data!, ...prev])
        updateStoreDatasets([response.data!, ...storeDatasets])
        setShowCreateDialog(false)
        setNewDatasetName('')
        setNewDatasetContent('')
        setNewDatasetDescription('')
        alert('Dataset created successfully!')
      }
    } catch (error) {
      console.error('Error creating dataset:', error)
      alert('Error creating dataset')
    }
  }

  const handleDeleteDataset = async (name: string) => {
    if (!confirm(`Are you sure you want to delete dataset "${name}"?`)) return

    try {
      const response = await api.deleteDataset(name)
      if (!response.error) {
        setDatasets(prev => prev.filter(d => d.name !== name))
        updateStoreDatasets(storeDatasets.filter(d => d.name !== name))
        alert('Dataset deleted successfully!')
      }
    } catch (error) {
      console.error('Error deleting dataset:', error)
      alert('Error deleting dataset')
    }
  }

  const handleDatasetInfo = (dataset: Dataset) => {
    alert(`Dataset: ${dataset.name}

Path: ${dataset.path}
Size: ${(dataset.size || 0).toLocaleString()} bytes
Created: ${dataset.created_at}
Updated: ${dataset.updated_at}

Files:
${dataset.has_meta ? '✓ Metadata' : '✗ Metadata'}
${dataset.has_train ? '✓ Training data' : '✗ Training data'}
${dataset.has_val ? '✓ Validation data' : '✗ Validation data'}

Description: ${dataset.description || 'No description'}`)
  }

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const renderDatasetCard = (dataset: Dataset) => (
    <Card key={dataset.name} className="hover:shadow-md transition-shadow">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between">
          <span className="font-medium">{dataset.name}</span>
          <div className="flex gap-1">
            {dataset.has_train && (
              <span className="px-2 py-0.5 bg-green-100 text-green-800 text-xs rounded-full">Train</span>
            )}
            {dataset.has_val && (
              <span className="px-2 py-0.5 bg-blue-100 text-blue-800 text-xs rounded-full">Val</span>
            )}
            {dataset.has_meta && (
              <span className="px-2 py-0.5 bg-purple-100 text-purple-800 text-xs rounded-full">Meta</span>
            )}
          </div>
        </CardTitle>
        <CardDescription className="text-sm">
          {dataset.description || 'No description'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-between text-sm text-slate-500">
          <span>{formatSize(dataset.size || 0)}</span>
          <div className="flex gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleDatasetInfo(dataset)}
            >
              ℹ️ Info
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleDeleteDataset(dataset.name)}
              className="text-red-500 hover:text-red-700"
            >
              🗑️ Delete
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="text-xl font-semibold">Datasets</span>
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => setShowCreateDialog(true)}
            >
              + Create Dataset
            </Button>
          </CardTitle>
          <CardDescription>
            Manage and organize your training datasets
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Dataset List */}
      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <Spinner className="h-8 w-8 text-blue-500" size="8" />
        </div>
      ) : datasets.length === 0 ? (
        <Card className="h-64 flex items-center justify-center text-slate-400">
          <div className="text-center">
            <div className="text-4xl mb-4">📊</div>
            <p>No datasets found</p>
            <p className="text-sm">Create your first dataset to get started</p>
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {datasets.map(renderDatasetCard)}
        </div>
      )}

      {/* Create Dialog */}
      {showCreateDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <Card className="w-full max-w-lg mx-4 max-h-[90vh] overflow-y-auto">
            <CardHeader>
              <CardTitle>Create New Dataset</CardTitle>
              <CardDescription>
                Create a new dataset for training your models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Dataset Name *</label>
                  <input
                    type="text"
                    value={newDatasetName}
                    onChange={(e) => setNewDatasetName(e.target.value)}
                    placeholder="my-dataset"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">Description</label>
                  <input
                    type="text"
                    value={newDatasetDescription}
                    onChange={(e) => setNewDatasetDescription(e.target.value)}
                    placeholder="Optional description"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">Content *</label>
                  <textarea
                    value={newDatasetContent}
                    onChange={(e) => setNewDatasetContent(e.target.value)}
                    placeholder="Enter training text content..."
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg resize-none"
                    rows={8}
                  />
                </div>
              </div>
            </CardContent>
            <div className="flex justify-end gap-2 p-4 border-t">
              <Button 
                variant="outline" 
                onClick={() => setShowCreateDialog(false)}
              >
                Cancel
              </Button>
              <Button 
                onClick={handleCreateDataset}
                disabled={!newDatasetName.trim() || !newDatasetContent.trim()}
              >
                Create Dataset
              </Button>
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}

export default Datasets
