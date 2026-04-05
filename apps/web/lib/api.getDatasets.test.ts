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

describe('api.getDatasets', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('GET /datasets and maps rows', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({
        datasets: [
          { id: 'ds1', name: 'shakespeare', path: 'datasets/shakespeare', type: 'text', size_kb: 12.3 },
        ],
      }),
    } as Response)

    const rows = await api.getDatasets()

    expect(fetch).toHaveBeenCalledWith('http://127.0.0.1:9/datasets')
    expect(rows).toEqual([
      {
        id: 'ds1',
        name: 'shakespeare',
        size: '12.3 KB',
        samples: 0,
        type: 'text',
        path: 'datasets/shakespeare',
      },
    ])
  })

  it('throws when GET /datasets is not ok', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 502,
      json: async () => ({}),
    } as Response)

    await expect(api.getDatasets()).rejects.toThrow('502')
  })
})
