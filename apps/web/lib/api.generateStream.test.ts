import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./auth', () => ({
  useAuthStore: {
    getState: () => ({ token: null as string | null }),
  },
}))

vi.mock('./config', () => ({
  PUBLIC_API_URL: 'http://127.0.0.1:9',
}))

import { api } from './api'

function sseResponse(chunks: string[]) {
  return new ReadableStream({
    start(controller) {
      const enc = new TextEncoder()
      for (const c of chunks) {
        controller.enqueue(enc.encode(c))
      }
      controller.close()
    },
  })
}

describe('api.generateStream', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('invokes onToken for each data frame and onDone when stream completes', async () => {
    const tokens: string[] = []
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: sseResponse([
        'data: {"token":"Hel","done":false}\n\n',
        'data: {"token":"lo","done":true}\n\n',
      ]),
    } as Response)

    await new Promise<void>((resolve) => {
      api.generateStream({ prompt: 'hi' }, (t) => tokens.push(t), () => resolve())
    })

    expect(tokens).toEqual(['Hel', 'lo'])
  })

  it('POSTs /inference/generate/stream with inference payload (max_new_tokens alias)', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: sseResponse(['data: {"token":"x","done":true}\n\n']),
    } as Response)

    await new Promise<void>((resolve) => {
      api.generateStream(
        { prompt: 'p', max_tokens: 42, temperature: 0.5, top_p: 0.7, top_k: 10 },
        () => {},
        () => resolve(),
      )
    })

    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:9/inference/generate/stream',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          prompt: 'p',
          max_new_tokens: 42,
          temperature: 0.5,
          top_p: 0.7,
          top_k: 10,
        }),
      }),
    )
  })

  it('calls onDone without tokens when response is not ok', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 503,
      body: null,
    } as Response)

    const tokens: string[] = []
    await new Promise<void>((resolve) => {
      api.generateStream({ prompt: 'x' }, (t) => tokens.push(t), () => resolve())
    })
    expect(tokens).toEqual([])
  })

  it('stops on SSE error payload', async () => {
    const tokens: string[] = []
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: sseResponse([
        'data: {"token":"a","done":false}\n\n',
        'data: {"error":"boom"}\n\n',
        'data: {"token":"b","done":false}\n\n',
      ]),
    } as Response)

    await new Promise<void>((resolve) => {
      api.generateStream({ prompt: 'x' }, (t) => tokens.push(t), () => resolve())
    })

    expect(tokens).toEqual(['a'])
  })

  it('abort rejects fetch and still calls onDone', async () => {
    vi.mocked(fetch).mockImplementation((_url, init) => {
      return new Promise((_resolve, reject) => {
        const signal = init?.signal as AbortSignal
        if (signal.aborted) {
          reject(new DOMException('Aborted', 'AbortError'))
          return
        }
        signal.addEventListener('abort', () => {
          reject(new DOMException('Aborted', 'AbortError'))
        })
      })
    })

    await new Promise<void>((resolve) => {
      const cancel = api.generateStream({ prompt: 'x' }, () => {}, () => resolve())
      cancel()
    })
  })
})
