'use client'

import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { PUBLIC_API_URL } from '@/lib/config'
import { apiDocsCurlExamplesBlock } from '@/lib/api-docs/curl-examples'
import { API_DOC_ENDPOINTS } from '@/lib/api-docs/endpoints'
import type { ApiDocEndpoint } from '@/lib/api-docs/types'

const METHOD_COLORS: Record<string, string> = {
  GET: 'border border-success/30 bg-success/10 text-success',
  POST: 'border border-primary/30 bg-primary/10 text-primary',
  PUT: 'border border-warning/30 bg-warning/10 text-warning',
  DELETE: 'border border-destructive/30 bg-destructive/10 text-destructive',
  WS: 'border border-chart-4/30 bg-chart-4/10 text-chart-4',
}

function EndpointRow({
  ep,
  expanded,
  onToggle,
}: {
  ep: ApiDocEndpoint
  expanded: boolean
  onToggle: () => void
}) {
  const colorClass = METHOD_COLORS[ep.method] ?? METHOD_COLORS.GET

  return (
    <Card className="overflow-hidden p-0">
      <Button
        type="button"
        variant="ghost"
        onClick={onToggle}
        className="h-auto w-full justify-start gap-3 rounded-none p-4 text-left font-normal hover:bg-muted/50"
      >
        <span className={`px-2 py-0.5 font-mono text-xs font-bold ${colorClass}`}>{ep.method}</span>
        <code className="font-mono text-sm text-foreground">{ep.path}</code>
        <span className="flex-1 text-sm text-muted-foreground">{ep.description}</span>
        <svg
          className={`h-4 w-4 shrink-0 text-muted-foreground transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </Button>

      {expanded && ep.body && (
        <div className="border-t border-border px-4 pb-4">
          <p className="mb-2 mt-3 text-sm text-muted-foreground">Request body</p>
          <div className="overflow-hidden border border-border bg-muted/30">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">Field</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">Type</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">Required</th>
                </tr>
              </thead>
              <tbody>
                {ep.body.map((field) => (
                  <tr key={field.field} className="border-b border-border/60 last:border-0">
                    <td className="px-3 py-2 font-mono text-primary">{field.field}</td>
                    <td className="px-3 py-2 text-foreground">{field.type}</td>
                    <td className="px-3 py-2">
                      {field.required ? (
                        <span className="font-medium text-destructive">Yes</span>
                      ) : (
                        <span className="text-muted-foreground">No</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Card>
  )
}

export default function ApiDocsPage() {
  const [expandedPath, setExpandedPath] = useState<string | null>(null)

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <h1 className="sl-h1 mb-2">API documentation</h1>
      <p className="mb-2 text-muted-foreground">
        Base URL: <code className="sl-code">{PUBLIC_API_URL}</code>
      </p>
      <p className="mb-6 text-sm text-muted-foreground">
        Interactive OpenAPI:{' '}
        <a className="font-medium text-primary hover:underline" href={`${PUBLIC_API_URL}/docs`} target="_blank" rel="noreferrer">
          {PUBLIC_API_URL}/docs
        </a>
      </p>

      <div className="space-y-3">
        {API_DOC_ENDPOINTS.map((ep) => (
          <EndpointRow
            key={`${ep.method}:${ep.path}`}
            ep={ep}
            expanded={expandedPath === ep.path}
            onToggle={() => setExpandedPath(expandedPath === ep.path ? null : ep.path)}
          />
        ))}
      </div>

      <Separator className="my-8" />

      <Card>
        <CardContent className="pt-6">
          <h2 className="sl-h2 mb-3">Quick examples</h2>
          <pre className="overflow-x-auto border border-border bg-muted/40 p-3 font-mono text-sm text-foreground">
            {apiDocsCurlExamplesBlock(PUBLIC_API_URL)}
          </pre>
        </CardContent>
      </Card>
    </div>
  )
}
