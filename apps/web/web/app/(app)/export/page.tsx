'use client'

import { useState, useEffect } from 'react'

import { api, ExportResult } from '@/lib/api'

export default function ExportPage() {
  const [loading, setLoading] = useState(false)
  const [formats, setFormats] = useState<Record<string, string>>({})
  const [outputPath, setOutputPath] = useState('models/exported')
  const [format, setFormat] = useState('sou')
  const [includeTokenizer, setIncludeTokenizer] = useState(true)
  const [result, setResult] = useState<ExportResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.getExportFormats().then((r) => setFormats(r.formats)).catch(console.error)
  }, [])

  const exportModel = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.exportModel(outputPath, format, includeTokenizer)
      if (res.error) {
        setError(res.error)
      } else {
        setResult(res)
      }
    } catch {
      setError('Failed to export model')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="sl-page max-w-4xl mx-auto">
      <h1 className="sl-h1 mb-6">Export Model</h1>

      <div className="sl-card mb-6 p-6">
        <h2 className="sl-h2 mb-4">Export Options</h2>

        <div className="mb-4 grid gap-4">
          <div>
            <label className="sl-label normal-case tracking-normal">Output Path</label>
            <input type="text" value={outputPath} onChange={(e) => setOutputPath(e.target.value)} className="sl-input" />
          </div>

          <div>
            <span className="sl-label normal-case tracking-normal">Format</span>
            <div className="grid grid-cols-2 gap-2 md:grid-cols-3">
              {Object.entries(formats).map(([key, desc]) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setFormat(key)}
                  className={`rounded-lg border p-3 text-left transition-colors ${
                    format === key ? 'border-primary bg-primary/10' : 'border-border bg-muted/30 hover:bg-muted/50'
                  }`}
                >
                  <span className="font-medium text-foreground">{key.toUpperCase()}</span>
                  <span className="mt-1 block text-xs text-muted-foreground">{desc}</span>
                </button>
              ))}
            </div>
          </div>

          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={includeTokenizer}
              onChange={(e) => setIncludeTokenizer(e.target.checked)}
              className="h-4 w-4 rounded border-border accent-primary"
            />
            <span className="text-sm font-medium text-foreground">Include tokenizer</span>
          </label>
        </div>

        <button type="button" onClick={exportModel} disabled={loading} className="sl-btn-primary rounded-lg px-4 py-2">
          {loading ? 'Exporting...' : 'Export Model'}
        </button>

        {error && <p className="mt-4 text-sm text-destructive">{error}</p>}
      </div>

      {result && !result.error && (
        <div className="sl-card p-6">
          <h2 className="sl-h2 mb-4">Export Complete</h2>

          <div className="rounded-lg border border-success/30 bg-success/10 p-4 text-sm">
            <p className="font-medium text-success">Status: {result.status}</p>
            <p className="mt-1 text-muted-foreground">Format: {result.format}</p>
          </div>

          <div className="mt-4">
            <p className="mb-2 text-sm font-medium text-foreground">Exported Files:</p>
            <ul className="space-y-1">
              {Object.entries(result.files || {}).map(([key, filePath]) => (
                <li key={key} className="text-sm text-muted-foreground">
                  {key}: <code className="sl-code">{String(filePath)}</code>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}
