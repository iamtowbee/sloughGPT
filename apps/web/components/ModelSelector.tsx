'use client'

import { useModels, useLocalModels, useHuggingFaceModels } from '@/contexts/ModelContext'
import { cn } from '@/lib/cn'
import { Select } from '@/components/ui/select'
import { Button } from '@/components/ui/button'

function Spinner({ className }: { className?: string }) {
  return (
    <svg
      className={cn("animate-spin h-4 w-4", className)}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  )
}

interface ModelSelectorProps {
  value?: string
  onValueChange?: (modelId: string) => void
  filter?: 'all' | 'local' | 'huggingface'
  showLoadButton?: boolean
  className?: string
  placeholder?: string
}

export function ModelSelector({
  value,
  onValueChange,
  filter = 'all',
  showLoadButton = true,
  className,
  placeholder = 'Select a model',
}: ModelSelectorProps) {
  const { models, loading, loadingModelId, loadModel, currentModel, isModelLoaded } = useModels()
  const localModels = useLocalModels()
  const hfModels = useHuggingFaceModels()

  const filteredModels = filter === 'all' ? models
    : filter === 'local' ? localModels
    : hfModels

  const displayModels = filteredModels.length > 0 ? filteredModels : models

  const options = displayModels.map(m => ({
    value: m.id,
    label: m.name,
  }))

  const handleLoadModel = async (modelId: string) => {
    const result = await loadModel(modelId, { mode: 'local' })
    if (result.success && onValueChange) {
      onValueChange(modelId)
    }
  }

  return (
    <div className={cn("flex gap-2", className)}>
      <Select
        value={value || currentModel || ''}
        onValueChange={onValueChange || (() => {})}
        options={options}
        placeholder={placeholder}
        className="w-[200px]"
      />

      {showLoadButton && value && (
        <Button
          size="sm"
          onClick={() => handleLoadModel(value)}
          disabled={loading || value === currentModel}
          className="gap-1.5"
        >
          {loading && loadingModelId === value ? (
            <>
              <Spinner className="h-3 w-3" />
              Loading...
            </>
          ) : value === currentModel && isModelLoaded ? (
            'Loaded'
          ) : (
            'Load'
          )}
        </Button>
      )}
    </div>
  )
}

interface ModelCardProps {
  model: {
    id: string
    name: string
    type: string
    sizeMb?: number
    params?: string
    description?: string
    tags?: string[]
  }
  isActive?: boolean
  isLoading?: boolean
  onLoad?: () => void
  onSelect?: () => void
  className?: string
}

export function ModelCard({
  model,
  isActive = false,
  isLoading = false,
  onLoad,
  onSelect,
  className,
}: ModelCardProps) {
  return (
    <div
      className={cn(
        "p-4 rounded-lg border transition-all cursor-pointer",
        isActive
          ? "border-primary bg-primary/5"
          : "border-border hover:border-primary/50 hover:bg-muted/50",
        className
      )}
      onClick={onSelect}
    >
      <div className="flex items-start justify-between mb-2">
        <h3 className="font-medium">{model.name}</h3>
        <span className={cn(
          "text-xs px-2 py-0.5 rounded-full",
          model.type === 'local'
            ? "bg-green-500/10 text-green-600 dark:text-green-400"
            : "bg-blue-500/10 text-blue-600 dark:text-blue-400"
        )}>
          {model.type}
        </span>
      </div>

      {model.description && (
        <p className="text-sm text-muted-foreground mb-2 line-clamp-2">
          {model.description}
        </p>
      )}

      <div className="flex items-center gap-3 text-xs text-muted-foreground mb-3">
        {model.sizeMb && (
          <span>{model.sizeMb < 1 ? `${(model.sizeMb * 1024).toFixed(0)} KB` : `${model.sizeMb.toFixed(1)} MB`}</span>
        )}
        {model.params && <span>{model.params}</span>}
      </div>

      {model.tags && model.tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-3">
          {model.tags.slice(0, 4).map((tag) => (
            <span
              key={tag}
              className="text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      <div className="flex justify-end">
        {isActive ? (
          <span className="text-sm text-green-600 dark:text-green-400 font-medium">
            Active
          </span>
        ) : onLoad ? (
          <Button
            size="sm"
            variant={isLoading ? "secondary" : "default"}
            onClick={(e) => {
              e.stopPropagation()
              onLoad()
            }}
            disabled={isLoading}
          >
            {isLoading ? 'Loading...' : 'Load'}
          </Button>
        ) : null}
      </div>
    </div>
  )
}