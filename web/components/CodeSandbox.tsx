'use client'

import { useState, useRef, useEffect } from 'react'

interface CodeExecutionResult {
  output: string
  error?: string
  executionTime?: number
}

export function CodeSandbox({ code, language = 'python', onClose }: { 
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
    
    const startTime = performance.now()
    
    try {
      if (language === 'javascript' || language === 'js') {
        const logs: string[] = []
        const customConsole = {
          log: (...args: any[]) => logs.push(args.map(String).join(' ')),
          error: (...args: any[]) => logs.push('Error: ' + args.map(String).join(' ')),
          warn: (...args: any[]) => logs.push('Warning: ' + args.map(String).join(' ')),
        }
        
        try {
          const fn = new Function('console', code)
          fn(customConsole)
          setOutput(logs.join('\n') || 'Code executed successfully (no output)')
        } catch (e: any) {
          setError(e.message)
        }
      } else if (language === 'python' || language === 'py') {
        const response = await fetch('https://emkc.org/api/v2/piston/execute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            language: 'python',
            version: '3.10',
            files: [{ content: code }]
          })
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
    } catch (e: any) {
      setError(e.message || 'Failed to execute code')
    }
    
    const endTime = performance.now()
    setIsRunning(false)
  }
  
  return (
    <div className="bg-slate-900 rounded-xl overflow-hidden my-4">
      <div className="flex items-center justify-between px-4 py-2 bg-slate-800">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <div className="w-3 h-3 rounded-full bg-green-500" />
          </div>
          <span className="text-slate-400 text-sm ml-2">{language}</span>
        </div>
        <div className="flex gap-2">
          <button
            onClick={runCode}
            disabled={isRunning}
            className="px-3 py-1 bg-green-600 hover:bg-green-700 disabled:bg-green-800 text-white text-sm rounded flex items-center gap-1"
          >
            {isRunning ? (
              <>
                <span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Running...
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                </svg>
                Run
              </>
            )}
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded"
            >
              Close
            </button>
          )}
        </div>
      </div>
      
      <div className="grid grid-cols-2 divide-x divide-slate-700">
        <div className="p-4">
          <pre className="text-sm text-slate-200 font-mono whitespace-pre-wrap">{code}</pre>
        </div>
        <div className="p-4 bg-slate-950">
          <div className="text-xs text-slate-500 mb-2">Output:</div>
          {error ? (
            <pre className="text-sm text-red-400 font-mono whitespace-pre-wrap">{error}</pre>
          ) : output ? (
            <pre className="text-sm text-green-400 font-mono whitespace-pre-wrap">{output}</pre>
          ) : (
            <span className="text-slate-500 text-sm">Click Run to execute...</span>
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
