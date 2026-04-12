'use client'

import { useState } from 'react'
import { api, type ImportSource, type GitHubRepo } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Textarea } from '@/components/ui/textarea'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'

interface DatasetImportModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onImportComplete: () => void
}

type SourceOption = {
  value: ImportSource
  label: string
  description: string
}

const SOURCE_OPTIONS: SourceOption[] = [
  { value: 'github', label: 'GitHub', description: 'Clone a repository' },
  { value: 'huggingface', label: 'HuggingFace', description: 'Download from HF Hub' },
  { value: 'kaggle', label: 'Kaggle', description: 'Download from Kaggle' },
  { value: 'url', label: 'URL', description: 'Download from a URL' },
  { value: 'local', label: 'Local', description: 'Import from local files' },
]

const DEFAULT_EXTENSIONS = ['.py', '.js', '.ts', '.md', '.txt', '.json']

export function DatasetImportModal({
  open,
  onOpenChange,
  onImportComplete,
}: DatasetImportModalProps) {
  const [source, setSource] = useState<ImportSource>('github')
  const [url, setUrl] = useState('')
  const [name, setName] = useState('')
  const [datasetId, setDatasetId] = useState('')
  const [kaggleDataset, setKaggleDataset] = useState('')
  const [path, setPath] = useState('')
  const [extensions, setExtensions] = useState<string[]>(DEFAULT_EXTENSIONS)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [searchResults, setSearchResults] = useState<GitHubRepo[]>([])
  const [searching, setSearching] = useState(false)

  const resetForm = () => {
    setUrl('')
    setName('')
    setDatasetId('')
    setPath('')
    setExtensions(DEFAULT_EXTENSIONS)
    setError(null)
    setSuccess(null)
    setSearchResults([])
  }

  const handleOpenChange = (newOpen: boolean) => {
    if (!newOpen) {
      resetForm()
    }
    onOpenChange(newOpen)
  }

  const handleSearch = async () => {
    if (!url.trim()) return
    setSearching(true)
    setError(null)
    try {
      const result = await api.searchGitHubRepos(url.trim())
      setSearchResults(result.repos || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed')
    } finally {
      setSearching(false)
    }
  }

  const handleImport = async () => {
    setLoading(true)
    setError(null)
    setSuccess(null)

    try {
      let result

      switch (source) {
        case 'github':
          if (!url.trim()) {
            throw new Error('GitHub URL is required')
          }
          const repoName = name.trim() || url.split('/').pop()?.replace('.git', '') || 'dataset'
          result = await api.importFromGitHub({
            url: url.trim(),
            name: repoName,
            extensions,
          })
          break

        case 'huggingface':
          if (!datasetId.trim()) {
            throw new Error('HuggingFace dataset ID is required')
          }
          result = await api.importFromHuggingFace({
            dataset_id: datasetId.trim(),
            name: name.trim() || undefined,
          })
          break

        case 'url':
          if (!url.trim()) {
            throw new Error('URL is required')
          }
          if (!name.trim()) {
            throw new Error('Dataset name is required')
          }
          result = await api.importFromURL({
            url: url.trim(),
            name: name.trim(),
          })
          break

        case 'local':
          if (!path.trim()) {
            throw new Error('Local path is required')
          }
          if (!name.trim()) {
            throw new Error('Dataset name is required')
          }
          result = await api.importFromLocal({
            path: path.trim(),
            name: name.trim(),
            extensions,
          })
          break

        case 'kaggle':
          if (!kaggleDataset.trim()) {
            throw new Error('Kaggle dataset ID is required (e.g., zillow/zecon)')
          }
          result = await api.importFromKaggle({
            dataset: kaggleDataset.trim(),
            name: name.trim() || undefined,
          })
          break
      }

      setSuccess(result.message)
      setTimeout(() => {
        resetForm()
        onImportComplete()
        onOpenChange(false)
      }, 1500)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Import failed')
    } finally {
      setLoading(false)
    }
  }

  const toggleExtension = (ext: string) => {
    setExtensions((prev) =>
      prev.includes(ext) ? prev.filter((e) => e !== ext) : [...prev, ext]
    )
  }

  const selectRepo = (repo: GitHubRepo) => {
    setUrl(repo.url)
    setName(repo.name)
    setSearchResults([])
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Import Dataset</DialogTitle>
          <DialogDescription>
            Import training data from various sources
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {SOURCE_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => setSource(option.value)}
                className={`rounded-md border p-3 text-left transition-colors ${
                  source === option.value
                    ? 'border-primary bg-primary/10'
                    : 'border-border hover:border-primary/50'
                }`}
              >
                <div className="font-medium">{option.label}</div>
                <div className="text-xs text-muted-foreground">{option.description}</div>
              </button>
            ))}
          </div>

          {source === 'github' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="github-url">GitHub Repository URL</Label>
                <Input
                  id="github-url"
                  placeholder="https://github.com/user/repo"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  className="mt-1"
                />
              </div>

              {searchResults.length > 0 && (
                <div className="max-h-40 overflow-y-auto rounded-md border">
                  {searchResults.map((repo) => (
                    <button
                      key={repo.id}
                      type="button"
                      onClick={() => selectRepo(repo)}
                      className="flex w-full items-center justify-between border-b px-3 py-2 text-left last:border-b-0 hover:bg-muted"
                    >
                      <div>
                        <div className="font-medium">{repo.name}</div>
                        <div className="text-xs text-muted-foreground">
                          {repo.full_name}
                        </div>
                      </div>
                      <Badge variant="secondary">{repo.stars} stars</Badge>
                    </button>
                  ))}
                </div>
              )}

              <Button type="button" variant="outline" size="sm" onClick={handleSearch} disabled={searching}>
                {searching ? 'Searching...' : 'Search Repos'}
              </Button>
            </div>
          )}

          {source === 'huggingface' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="hf-id">HuggingFace Dataset ID</Label>
                <Input
                  id="hf-id"
                  placeholder="username/dataset-name"
                  value={datasetId}
                  onChange={(e) => setDatasetId(e.target.value)}
                  className="mt-1"
                />
              </div>
            </div>
          )}

          {source === 'kaggle' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="kaggle-id">Kaggle Dataset ID</Label>
                <Input
                  id="kaggle-id"
                  placeholder="username/dataset-name"
                  value={kaggleDataset}
                  onChange={(e) => setKaggleDataset(e.target.value)}
                  className="mt-1"
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Requires Kaggle CLI installed and authenticated
              </p>
            </div>
          )}

          {source === 'url' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="url-input">URL</Label>
                <Input
                  id="url-input"
                  placeholder="https://example.com/data.txt"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  className="mt-1"
                />
              </div>
            </div>
          )}

          {source === 'local' && (
            <div className="space-y-3">
              <div>
                <Label htmlFor="local-path">Path to file or directory</Label>
                <Input
                  id="local-path"
                  placeholder="/path/to/data"
                  value={path}
                  onChange={(e) => setPath(e.target.value)}
                  className="mt-1"
                />
              </div>
            </div>
          )}

          {(source === 'github' || source === 'local') && (
            <div>
              <Label>File Types (for code repos)</Label>
              <div className="mt-2 flex flex-wrap gap-2">
                {['.py', '.js', '.ts', '.md', '.txt', '.json', '.yaml', '.csv'].map((ext) => (
                  <button
                    key={ext}
                    type="button"
                    onClick={() => toggleExtension(ext)}
                    className={`rounded-md px-2 py-1 text-sm ${
                      extensions.includes(ext)
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted hover:bg-muted/80'
                    }`}
                  >
                    {ext}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div>
            <Label htmlFor="name">Dataset Name</Label>
            <Input
              id="name"
              placeholder="my-dataset"
              value={name}
              onChange={(e) => setName(e.target.value)}
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
          <Button type="button" onClick={handleImport} disabled={loading}>
            {loading ? 'Importing...' : 'Import'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
