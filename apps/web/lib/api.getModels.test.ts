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

describe('api.getModels', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('GET /models and maps rows to UI Model shape', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({
        models: [
          { id: 'm1', name: 'Alpha', source: 'huggingface', size_mb: 512 },
          { name: 'unnamed-id', size_mb: 1.5 },
        ],
      }),
    } as Response)

    const rows = await api.getModels()

    expect(fetch).toHaveBeenCalledWith('http://127.0.0.1:9/models')
    expect(rows).toEqual([
      { id: 'm1', name: 'Alpha', size: '512 MB', type: 'huggingface' },
      { id: 'unnamed-id', name: 'unnamed-id', size: '1.5 MB', type: 'unknown' },
    ])
  })

  it('handles missing models array', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({}),
    } as Response)

    const rows = await api.getModels()
    expect(rows).toEqual([])
  })

  it('throws when GET /models is not ok', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({}),
    } as Response)

    await expect(api.getModels()).rejects.toThrow('503')
  })
})
