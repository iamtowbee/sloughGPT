'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { api, type Dataset } from '@/lib/api'
import { devDebug } from '@/lib/dev-log'

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDatasets()
  }, [])

  const fetchDatasets = async () => {
    setLoading(true)
    try {
      const rows = await api.getDatasets()
      setDatasets(rows)
    } catch (err) {
      devDebug('Failed to fetch datasets:', err)
      setDatasets([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="sl-page mx-auto max-w-6xl">
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <h1 className="sl-h1">Datasets</h1>
        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="secondary" size="sm" onClick={() => void fetchDatasets()}>
            Refresh
          </Button>
          <Button type="button" size="sm" variant="outline" asChild>
            <Link href="/training">Start training</Link>
          </Button>
        </div>
      </div>

      {loading ? (
        <div className="py-12 text-center text-muted-foreground">Loading datasets…</div>
      ) : datasets.length === 0 ? (
        <Card className="border-dashed">
          <CardContent className="py-12 text-center text-muted-foreground">
            No datasets found. Run the API from the repo root so <span className="font-mono">datasets/</span> is
            visible.
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {datasets.map((dataset) => (
            <Card
              key={dataset.id}
              className="transition-colors duration-200 ease-smooth hover:border-primary/25"
            >
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between gap-2">
                  <CardTitle className="text-base">{dataset.name}</CardTitle>
                  <span className="shrink-0 border border-border bg-muted/50 px-2 py-0.5 font-mono text-xs text-muted-foreground">
                    {dataset.type}
                  </span>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                {dataset.path && (
                  <p className="truncate font-mono text-xs text-muted-foreground">{dataset.path}</p>
                )}
              </CardContent>
              <CardFooter className="flex justify-between border-t border-border pt-4">
                <span className="text-sm font-medium text-chart-3">{dataset.size}</span>
                <Button type="button" variant="ghost" size="sm">
                  View
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      )}

      <Card className="mt-8">
        <CardHeader>
          <CardTitle className="text-base">CLI</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1 font-mono text-sm text-muted-foreground">
          <p>python3 cli.py data validate datasets/my_data/</p>
          <p>python3 cli.py data stats datasets/my_data/</p>
          <p>python3 cli.py data split datasets/my_data/ --train 0.9</p>
        </CardContent>
      </Card>
    </div>
  )
}
