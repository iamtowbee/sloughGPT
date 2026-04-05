'use client'

import { useState, useEffect } from 'react'

import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { api, ExportResult } from '@/lib/api'
import { devDebug } from '@/lib/dev-log'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'

export default function ExportPage() {
  const [loading, setLoading] = useState(false)
  const [formats, setFormats] = useState<Record<string, string>>({})
  const [outputPath, setOutputPath] = useState('models/exported')
  const [format, setFormat] = useState('sou')
  const [includeTokenizer, setIncludeTokenizer] = useState(true)
  const [result, setResult] = useState<ExportResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .getExportFormats()
      .then((r) => setFormats(r.formats))
      .catch((e) => devDebug('getExportFormats:', e))
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
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader className="mb-6 items-start" left={<AppRouteHeaderLead title="Export model" />} />

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Export options</CardTitle>
          <CardDescription>Calls the API export endpoint with the selected format.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="export-path">Output path</Label>
            <Input id="export-path" value={outputPath} onChange={(e) => setOutputPath(e.target.value)} />
          </div>

          <div>
            <Label className="mb-2 block">Format</Label>
            <div className="grid grid-cols-2 gap-2 md:grid-cols-3">
              {Object.entries(formats).map(([key, desc]) => (
                <Button
                  key={key}
                  type="button"
                  variant={format === key ? 'default' : 'outline'}
                  className="h-auto flex-col items-start gap-1 py-3"
                  onClick={() => setFormat(key)}
                >
                  <span className="font-medium">{key.toUpperCase()}</span>
                  <span className="text-left text-xs font-normal text-muted-foreground">{desc}</span>
                </Button>
              ))}
            </div>
          </div>

          <div className="flex items-center justify-between gap-4 border border-border p-3">
            <Label htmlFor="tok-switch" className="cursor-pointer text-foreground">
              Include tokenizer
            </Label>
            <Switch
              id="tok-switch"
              checked={includeTokenizer}
              onCheckedChange={setIncludeTokenizer}
            />
          </div>

          <Button type="button" onClick={exportModel} disabled={loading}>
            {loading ? 'Exporting…' : 'Export model'}
          </Button>

          {error && <p className="text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>

      {result && !result.error && (
        <Card>
          <CardHeader>
            <CardTitle>Export complete</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="border border-success/30 bg-success/10 p-4 text-sm">
              <p className="font-medium text-success">Status: {result.status}</p>
              <p className="mt-1 text-muted-foreground">Format: {result.format}</p>
            </div>
            <div>
              <p className="mb-2 text-sm font-medium text-foreground">Exported files</p>
              <ul className="space-y-1">
                {Object.entries(result.files || {}).map(([key, filePath]) => (
                  <li key={key} className="text-sm text-muted-foreground">
                    {key}: <code className="sl-code">{String(filePath)}</code>
                  </li>
                ))}
              </ul>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
