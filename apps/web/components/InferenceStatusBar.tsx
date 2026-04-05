'use client'

import Link from 'next/link'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import { catalogIdMatchesRuntime } from '@/lib/inference-display'

type Props = {
  health: ApiHealthSnapshot
  selectedCatalogId: string
}

/**
 * Offline / no-weights / catalog-vs-runtime alerts below the chat toolbar.
 * Runtime badges and refresh live in the toolbar (`InferenceRuntimeToolbar`).
 */
export function InferenceStatusBar({ health, selectedCatalogId }: Props) {
  const mismatch =
    health !== null &&
    health !== 'offline' &&
    health.model_loaded &&
    !catalogIdMatchesRuntime(selectedCatalogId, health.model_type)

  const showOffline = health === 'offline'
  const showNoWeights =
    health !== null && health !== 'offline' && !health.model_loaded
  const showMismatch = mismatch

  if (!showOffline && !showNoWeights && !showMismatch) {
    return null
  }

  return (
    <div className="space-y-2 border-b border-border pb-2">
      {showOffline ? (
        <Card className="border-destructive/30 bg-muted/20">
          <CardContent className="p-3 pt-3 text-xs text-muted-foreground">
            Cannot reach the API. Start it from the repo root (
            <code className="font-mono text-xs">python3 apps/api/server/main.py</code>) and ensure{' '}
            <code className="font-mono text-xs">NEXT_PUBLIC_API_URL</code> matches.
          </CardContent>
        </Card>
      ) : null}

      {showNoWeights ? (
        <Card className="border-warning/40 bg-warning/5">
          <CardContent className="p-3 pt-3 text-xs text-muted-foreground">
            No weights loaded in the API process yet.{' '}
            <Link href="/models" className="text-primary underline-offset-2 hover:underline">
              Load a model
            </Link>{' '}
            or wait for server autoload (<code className="font-mono text-xs">SLOUGHGPT_AUTOLOAD_MODEL</code>).
          </CardContent>
        </Card>
      ) : null}

      {showMismatch ? (
        <Card className="border-border bg-muted/15">
          <CardContent className="flex flex-col gap-1 p-3 pt-3 text-xs text-muted-foreground">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline">Catalog ≠ runtime</Badge>
              <span>
                This chat&apos;s dropdown is <span className="font-mono text-foreground">{selectedCatalogId}</span>;{' '}
                generation uses the API&apos;s loaded weights (
                <span className="font-mono text-foreground">{health.model_type}</span>).
              </span>
            </div>
          </CardContent>
        </Card>
      ) : null}
    </div>
  )
}

type ToolbarProps = {
  health: ApiHealthSnapshot
  onRefresh: () => void
}

function RefreshIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
      />
    </svg>
  )
}

/** Compact API runtime row: badges + refresh (use beside model catalog dropdown). */
export function InferenceRuntimeToolbar({ health, onRefresh }: ToolbarProps) {
  return (
    <div className="flex flex-wrap items-center justify-end gap-1.5 sm:justify-start">
      {health === null ? (
        <Badge variant="outline" className="font-normal text-muted-foreground">
          API status…
        </Badge>
      ) : health === 'offline' ? (
        <Badge variant="destructive">Disconnected</Badge>
      ) : (
        <>
          <Badge variant={health.model_loaded ? 'success' : 'warning'}>
            {health.model_loaded ? 'Weights loaded' : 'No weights'}
          </Badge>
          <Badge
            variant="outline"
            className="max-w-[min(100%,14rem)] truncate font-mono text-xs font-normal sm:max-w-[min(100%,18rem)]"
            title="Loaded in API process (inference uses this)"
          >
            {health.model_type}
          </Badge>
        </>
      )}
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className="h-8 w-8 shrink-0"
        onClick={() => void onRefresh()}
        aria-label="Refresh API status"
      >
        <RefreshIcon className="h-4 w-4" />
      </Button>
    </div>
  )
}
