'use client'

import type { ReactNode } from 'react'
import Link from 'next/link'

import { Button } from '@/components/ui/button'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import { catalogIdMatchesRuntime } from '@/lib/inference-display'
import { cn } from '@/lib/cn'

type Props = {
  health: ApiHealthSnapshot
  selectedCatalogId: string
}

function InlineCode({ children }: { children: ReactNode }) {
  return <code className="sl-chat-inline-code">{children}</code>
}

function runtimeModelLabel(health: ApiHealthSnapshot): string {
  if (health === null || health === 'offline') return 'unknown'
  return String(health.model_type ?? '').trim() || 'unknown'
}

function CheckCircleIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
      />
    </svg>
  )
}

function AlertTriangleIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
      />
    </svg>
  )
}

/** Blocked / unreachable — reads clearly at 16px in header toolbars. */
function OfflineIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <circle cx="12" cy="12" r="9" strokeWidth={2} />
      <path strokeLinecap="round" strokeWidth={2} d="M7 7l10 10" />
    </svg>
  )
}

function DotsPulseIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="currentColor" viewBox="0 0 24 24" aria-hidden>
      <circle className="animate-pulse opacity-60" cx="6" cy="12" r="2" />
      <circle className="animate-pulse [animation-delay:150ms]" cx="12" cy="12" r="2" />
      <circle className="animate-pulse [animation-delay:300ms]" cx="18" cy="12" r="2" />
    </svg>
  )
}

function CpuIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
      />
    </svg>
  )
}

/**
 * Offline / no-weights / catalog-vs-runtime notices below the chat toolbar.
 * Styled as compact shell notes (see `.sl-chat-toolbar-note` in globals.css).
 * Runtime controls live in the toolbar (`InferenceRuntimeToolbar`).
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
    <div className="flex flex-col gap-1.5" role="status">
      {showOffline ? (
        <div className="sl-chat-toolbar-note sl-chat-toolbar-note--err">
          <p className="sl-chat-toolbar-note__label">API unreachable</p>
          <p className="text-xs leading-snug text-muted-foreground">
            Start the server from the repo root (
            <InlineCode>python3 apps/api/server/main.py</InlineCode>) and ensure{' '}
            <InlineCode>NEXT_PUBLIC_API_URL</InlineCode> matches.
          </p>
        </div>
      ) : null}

      {showNoWeights ? (
        <div className="sl-chat-toolbar-note sl-chat-toolbar-note--warn">
          <p className="sl-chat-toolbar-note__label">No weights loaded</p>
          <p className="text-xs leading-snug text-muted-foreground">
            Load weights in the API before chatting.{' '}
            <Link href="/models" className="text-primary underline-offset-2 hover:underline">
              Open models
            </Link>{' '}
            or wait for autoload (<InlineCode>SLOUGHGPT_AUTOLOAD_MODEL</InlineCode>).
          </p>
        </div>
      ) : null}

      {showMismatch ? (
        <div className="sl-chat-toolbar-note sl-chat-toolbar-note--hint flex flex-row items-start gap-2.5">
          <AlertTriangleIcon className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden />
          <p className="min-w-0 text-xs leading-snug text-muted-foreground">
            <span className="font-medium text-foreground/85">Catalog ≠ runtime.</span>{' '}
            Chat selection <InlineCode>{selectedCatalogId}</InlineCode>
            <span className="text-foreground/35"> · </span>
            API <InlineCode>{runtimeModelLabel(health)}</InlineCode>.
          </p>
        </div>
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

/** Simple status indicator for chat header. */
export function InferenceRuntimeToolbar({ health, onRefresh }: ToolbarProps) {
  if (health === null) return <span className="text-xs text-muted-foreground/50">...</span>
  if (health === 'offline') return <span className="text-xs text-destructive">Offline</span>
  if (health.model_loaded) return <span className="text-xs text-success/70">{health.model_type}</span>
  return <span className="text-xs text-warning/70">No model</span>
}
