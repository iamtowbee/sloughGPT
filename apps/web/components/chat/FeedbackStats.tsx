'use client'

import { useFeedbackStore } from '@/lib/feedback-store'

export function FeedbackStats() {
  const { stats, adapterStats, workflowStatus, isLoading, triggerWorkflowAction } = useFeedbackStore()

  if (!stats && !adapterStats) {
    return null
  }

  return (
    <div className="bg-muted/50 rounded-lg p-4 text-sm">
      <h3 className="font-semibold mb-3">Feedback System</h3>
      
      {stats && (
        <div className="space-y-2 mb-4">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Thumbs Up:</span>
            <span className="font-medium text-green-600">{stats.db_stats.thumbs_up}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Thumbs Down:</span>
            <span className="font-medium text-red-600">{stats.db_stats.thumbs_down}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Ratio:</span>
            <span className="font-medium">{(stats.db_stats.ratio * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}

      {adapterStats && (
        <div className="space-y-2 mb-4 pt-3 border-t">
          <div className="flex justify-between">
            <span className="text-muted-foreground">User Adapters:</span>
            <span className="font-medium">{adapterStats.total_users}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Total Size:</span>
            <span className="font-medium">{adapterStats.total_size_mb.toFixed(2)} MB</span>
          </div>
          {adapterStats.auto_management && (
            <div className="flex justify-between">
              <span className="text-muted-foreground">Quality Adapters:</span>
              <span className="font-medium">{adapterStats.auto_management.quality_adapters_count}</span>
            </div>
          )}
        </div>
      )}

      {workflowStatus && (
        <div className="space-y-2 pt-3 border-t">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Workflow:</span>
            <span className={`font-medium ${workflowStatus.running ? 'text-green-600' : 'text-gray-500'}`}>
              {workflowStatus.running ? 'Running' : 'Stopped'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Aggregations:</span>
            <span className="font-medium">{workflowStatus.stats.aggregations_performed}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Prunes:</span>
            <span className="font-medium">{workflowStatus.stats.prunes_performed}</span>
          </div>
        </div>
      )}

      <div className="flex gap-2 mt-4 pt-3 border-t">
        <button
          onClick={() => triggerWorkflowAction('aggregate')}
          disabled={isLoading}
          className="flex-1 px-2 py-1 text-xs bg-primary text-primary-foreground rounded hover:opacity-90 disabled:opacity-50"
        >
          Aggregate
        </button>
        <button
          onClick={() => triggerWorkflowAction('prune')}
          disabled={isLoading}
          className="flex-1 px-2 py-1 text-xs bg-secondary rounded hover:opacity-90 disabled:opacity-50"
        >
          Prune
        </button>
        <button
          onClick={() => triggerWorkflowAction('export')}
          disabled={isLoading}
          className="flex-1 px-2 py-1 text-xs bg-secondary rounded hover:opacity-90 disabled:opacity-50"
        >
          Export
        </button>
      </div>
    </div>
  )
}
