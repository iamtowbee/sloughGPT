/**
 * External API wrappers - third-party services
 */

import HttpClient from './http-client'

const PISTON_API_URL = 'https://emkc.org/api/v2/piston'

export interface PistonExecuteRequest {
  language: string
  version: string
  files: Array<{ content: string }>
}

export interface PistonExecuteResponse {
  run?: {
    stdout: string
    stderr: string
    output: string
    code: number
    signal: string | null
  }
  compile?: {
    stdout: string
    stderr: string
    output: string
    code: number
    signal: string | null
  }
}

export interface PistonRuntime {
  language: string
  version: string
  aliases: string[]
}

const pistonClient = new HttpClient(PISTON_API_URL)

export async function executeCode(
  code: string,
  language: string = 'python',
  version: string = '3.10'
): Promise<{ output: string; error?: string }> {
  try {
    const response = await pistonClient.post<PistonExecuteResponse>('/execute', {
      language,
      version,
      files: [{ content: code }],
    })

    if (response.run?.output) {
      return { output: response.run.output }
    } else if (response.run?.stderr) {
      return { output: '', error: response.run.stderr }
    }
    return { output: 'Code executed successfully (no output)' }
  } catch (err) {
    if (err instanceof Error) {
      return { output: '', error: `HTTP ${err.message}` }
    }
    return { output: '', error: 'Failed to execute code' }
  }
}

export async function getPistonRuntimes(): Promise<PistonRuntime[]> {
  try {
    const response = await pistonClient.get<PistonRuntime[]>('/runtimes')
    return response
  } catch {
    return []
  }
}