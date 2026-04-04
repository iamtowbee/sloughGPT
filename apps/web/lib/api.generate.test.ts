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

describe('api.generate', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('throws when server returns 200 with error field (no model loaded)', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ error: 'Model not loaded', text: '' }),
    } as Response)

    await expect(api.generate({ prompt: 'hello' })).rejects.toThrow('Model not loaded')
  })

  it('throws when server returns 200 with empty text and no error string', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ text: '   ' }),
    } as Response)

    await expect(api.generate({ prompt: 'hello' })).rejects.toThrow(/no text/i)
  })

  it('returns text when server returns a non-empty body', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({
        text: 'hi there',
        model: 'gpt2-engine',
        tokens_generated: 2,
      }),
    } as Response)

    const out = await api.generate({ prompt: 'hello' })
    expect(out.text).toBe('hi there')
    expect(out.model).toBe('gpt2-engine')
    expect(out.tokens_generated).toBe(2)
  })

  it('throws on HTTP error responses', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({ error: 'Service unavailable' }),
    } as Response)

    await expect(api.generate({ prompt: 'x' })).rejects.toThrow('Service unavailable')
  })
})
