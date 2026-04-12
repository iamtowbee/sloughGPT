'use client'

import { useState } from 'react'
import { api, type Dataset } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'

interface DatasetCombineModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  datasets: Dataset[]
  onCombineComplete: () => void
}

export function DatasetCombineModal({
  open,
  onOpenChange,
  datasets,
  onCombineComplete,
}: DatasetCombineModalProps) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [combinedName, setCombinedName] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  const resetForm = () => {
    setSelectedIds(new Set())
    setCombinedName('')
    setError(null)
    setSuccess(null)
  }

  const handleOpenChange = (newOpen: boolean) => {
    if (!newOpen) {
      resetForm()
    }
    onOpenChange(newOpen)
  }

  const toggleDataset = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const selectAll = () => {
    setSelectedIds(new Set(datasets.map((d) => d.id)))
  }

  const selectNone = () => {
    setSelectedIds(new Set())
  }

  const handleCombine = async () => {
    if (selectedIds.size < 2) {
      setError('Select at least 2 datasets to combine')
      return
    }
    if (!combinedName.trim()) {
      setError('Enter a name for the combined dataset')
      return
    }

    setLoading(true)
    setError(null)
    setSuccess(null)

    try {
      const result = await api.combineDatasets(Array.from(selectedIds), combinedName.trim())
      setSuccess(result.message || `Created combined dataset: ${result.dataset_id}`)
      setTimeout(() => {
        resetForm()
        onCombineComplete()
        onOpenChange(false)
      }, 1500)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Combine failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Combine Datasets</DialogTitle>
          <DialogDescription>
            Merge multiple datasets into one. Select 2 or more datasets to combine.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="flex gap-2">
            <Button type="button" variant="outline" size="sm" onClick={selectAll}>
              Select All
            </Button>
            <Button type="button" variant="outline" size="sm" onClick={selectNone}>
              Select None
            </Button>
            <span className="flex items-center text-sm text-muted-foreground">
              {selectedIds.size} selected
            </span>
          </div>

          <div className="max-h-60 space-y-2 overflow-y-auto rounded-md border p-2">
            {datasets.map((dataset) => (
              <label
                key={dataset.id}
                className="flex cursor-pointer items-center gap-3 rounded-md p-2 hover:bg-muted"
              >
                <Checkbox
                  checked={selectedIds.has(dataset.id)}
                  onCheckedChange={() => toggleDataset(dataset.id)}
                />
                <div className="flex-1">
                  <div className="font-medium">{dataset.name}</div>
                  <div className="text-xs text-muted-foreground">
                    {dataset.type} • {dataset.size}
                    {dataset.samples > 0 && ` • ${dataset.samples} samples`}
                  </div>
                </div>
              </label>
            ))}
          </div>

          <div>
            <Label htmlFor="combined-name">Combined Dataset Name</Label>
            <Input
              id="combined-name"
              placeholder="my-combined-dataset"
              value={combinedName}
              onChange={(e) => setCombinedName(e.target.value)}
              className="mt-1"
            />
          </div>

          {error && (
            <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
              {error}
            </div>
          )}

          {success && (
            <div className="rounded-md bg-green-500/10 p-3 text-sm text-green-600">
              {success}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => handleOpenChange(false)}>
            Cancel
          </Button>
          <Button
            type="button"
            onClick={handleCombine}
            disabled={loading || selectedIds.size < 2}
          >
            {loading ? 'Combining...' : `Combine ${selectedIds.size} Datasets`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
