'use client'

import { useState } from 'react'
import { api } from '@/lib/api'

export default function ExportPage() {
  const [loading, setLoading] = useState(false)
  const [formats, setFormats] = useState<Record<string, string>>({})
  const [outputPath, setOutputPath] = useState('models/exported')
  const [format, setFormat] = useState('sou')
  const [includeTokenizer, setIncludeTokenizer] = useState(true)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.getExportFormats().then(setFormats).catch(console.error)
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
    } catch (e) {
      setError('Failed to export model')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Export Model</h1>
      
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h2 className="text-lg font-semibold mb-4">Export Options</h2>
        
        <div className="grid gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium mb-1">Output Path</label>
            <input
              type="text"
              value={outputPath}
              onChange={(e) => setOutputPath(e.target.value)}
              className="w-full px-3 py-2 border rounded-lg"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">Format</label>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {Object.entries(formats).map(([key, desc]) => (
                <button
                  key={key}
                  onClick={() => setFormat(key)}
                  className={`p-3 rounded-lg border text-left ${
                    format === key ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                  }`}
                >
                  <span className="font-medium">{key.toUpperCase()}</span>
                  <span className="block text-xs text-gray-500 mt-1">{desc}</span>
                </button>
              ))}
            </div>
          </div>
          
          <div>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={includeTokenizer}
                onChange={(e) => setIncludeTokenizer(e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm font-medium">Include tokenizer</span>
            </label>
          </div>
        </div>
        
        <button
          onClick={exportModel}
          disabled={loading}
          className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
        >
          {loading ? 'Exporting...' : 'Export Model'}
        </button>
        
        {error && (
          <p className="mt-4 text-red-600">{error}</p>
        )}
      </div>

      {result && !result.error && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Export Complete</h2>
          
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <p className="text-green-800 font-medium">Status: {result.status}</p>
            <p className="text-green-600 mt-1">Format: {result.format}</p>
          </div>
          
          <div className="mt-4">
            <p className="text-sm font-medium mb-2">Exported Files:</p>
            <ul className="space-y-1">
              {Object.entries(result.files || {}).map(([key, path]) => (
                <li key={key} className="text-sm text-gray-600">
                  {key}: <code className="bg-gray-100 px-2 py-1 rounded">{path}</code>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}
