'use client'

import { useState } from 'react'

interface CodeExecutionResult {
  output: string
  error?: string
  executionTime?: number
}

export function CodeSandbox({
  code,
  language = 'python',
  onClose,
}: {
  code: string
  language?: string
  onClose?: () => void
}) {
  const [output, setOutput] = useState<string>('')
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const runCode = async () => {
    setIsRunning(true)
    setError(null)
    setOutput('')

    try {
      if (language === 'javascript' || language === 'js') {
        const logs: string[] = []
        const customConsole = {
          log: (...args: unknown[]) => logs.push(args.map(String).join(' ')),
          error: (...args: unknown[]) => logs.push('Error: ' + args.map(String).join(' ')),
          warn: (...args: unknown[]) => logs.push('Warning: ' + args.map(String).join(' ')),
        }

        try {
          const fn = new Function('console', code)
          fn(customConsole)
          setOutput(logs.join('\n') || 'Code executed successfully (no output)')
        } catch (e: unknown) {
          setError(e instanceof Error ? e.message : String(e))
        }
      } else if (language === 'python' || language === 'py') {
        const response = await fetch('https://emkc.org/api/v2/piston/execute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            language: 'python',
            version: '3.10',
            files: [{ content: code }],
          }),
        })

        const result = await response.json()

        if (result.run?.output) {
          setOutput(result.run.output)
        } else if (result.run?.stderr) {
          setError(result.run.stderr)
        } else {
          setOutput('Code executed successfully (no output)')
        }
      } else {
        setOutput(`Language "${language}" execution not supported in browser. Supported: Python, JavaScript`)
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to execute code')
    }

    setIsRunning(false)
  }

  return (
    <div className="my-4 overflow-hidden rounded-xl border border-border bg-card shadow-sm">
      <div className="flex items-center justify-between border-b border-border bg-muted/50 px-4 py-2">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5">
            <div className="h-3 w-3 rounded-full bg-destructive" />
            <div className="h-3 w-3 rounded-full bg-warning" />
            <div className="h-3 w-3 rounded-full bg-success" />
          </div>
          <span className="ml-2 font-mono text-sm text-muted-foreground">{language}</span>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={runCode}
            disabled={isRunning}
            className="sl-btn-primary flex items-center gap-1 rounded-md px-3 py-1 text-sm"
          >
            {isRunning ? (
              <>
                <span className="h-3 w-3 animate-spin rounded-full border-2 border-primary-foreground/30 border-t-primary-foreground" />
                Running...
              </>
            ) : (
              <>
                <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20" aria-hidden>
                  <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                </svg>
                Run
              </>
            )}
          </button>
          {onClose && (
            <button type="button" onClick={onClose} className="sl-btn-secondary rounded-md px-3 py-1 text-sm">
              Close
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 divide-y divide-border md:grid-cols-2 md:divide-x md:divide-y-0">
        <div className="p-4">
          <pre className="whitespace-pre-wrap font-mono text-sm text-foreground">{code}</pre>
        </div>
        <div className="bg-muted/30 p-4">
          <div className="mb-2 font-mono text-xs uppercase tracking-wider text-muted-foreground">Output</div>
          {error ? (
            <pre className="whitespace-pre-wrap font-mono text-sm text-destructive">{error}</pre>
          ) : output ? (
            <pre className="whitespace-pre-wrap font-mono text-sm text-success">{output}</pre>
          ) : (
            <span className="text-sm text-muted-foreground">Click Run to execute...</span>
          )}
        </div>
      </div>
    </div>
  )
}

export function detectAndExecuteCode(content: string): { hasCode: boolean; code?: string; language?: string } {
  const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g
  const match = codeBlockRegex.exec(content)

  if (match) {
    const language = match[1] || 'text'
    const code = match[2].trim()
    return { hasCode: true, code, language }
  }

  return { hasCode: false }
}
