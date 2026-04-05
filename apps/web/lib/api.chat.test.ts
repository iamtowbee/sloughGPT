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

describe('api.chatStream', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('POSTs /chat/stream with messages and generation fields', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      body: sseResponse(['data: {"token":"ok","done":true}\n\n']),
    } as Response)

    await new Promise<void>((resolve) => {
      api.chatStream(
        {
          messages: [{ role: 'user', content: 'hi' }],
          max_new_tokens: 40,
          temperature: 0.5,
          top_p: 0.7,
          top_k: 10,
          model: 'gpt2',
        },
        () => {},
        () => resolve(),
      )
    })

    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:9/chat/stream',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          messages: [{ role: 'user', content: 'hi' }],
          max_new_tokens: 40,
          temperature: 0.5,
          top_p: 0.7,
          top_k: 10,
          model: 'gpt2',
        }),
      }),
    )
  })
})

describe('api.chat', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('POSTs /chat with chat payload', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({
        text: 'hello',
        model: 'gpt2-engine',
        tokens_generated: 2,
      }),
    } as Response)

    const out = await api.chat({
      messages: [{ role: 'user', content: 'yo' }],
      max_new_tokens: 50,
    })

    expect(out.text).toBe('hello')
    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:9/chat',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          messages: [{ role: 'user', content: 'yo' }],
          max_new_tokens: 50,
          temperature: 0.8,
          top_p: 0.9,
          top_k: 50,
        }),
      }),
    )
  })
})
