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
      {
        id: 'm1',
        name: 'Alpha',
        size: '512.00 MB',
        type: 'huggingface',
        size_mb: 512,
      },
      {
        id: 'unnamed-id',
        name: 'unnamed-id',
        size: '1.50 MB',
        type: 'unknown',
        size_mb: 1.5,
      },
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

  it('maps description and string-only tags', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({
        models: [
          {
            id: 't1',
            name: 'Tagged',
            source: 'local',
            size_mb: 8,
            description: 'A model',
            tags: ['gen', 99, 'small'],
          },
        ],
      }),
    } as Response)

    const rows = await api.getModels()
    expect(rows[0]).toMatchObject({
      id: 't1',
      description: 'A model',
      tags: ['gen', 'small'],
      type: 'local',
    })
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
