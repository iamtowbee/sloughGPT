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

describe('api.loadModel', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('POSTs /models/load with model_id and default mode/device', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'loaded', model: 'gpt2', model_type: 'gpt2' }),
    } as Response)

    await api.loadModel('gpt2')

    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:9/models/load',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: 'gpt2', mode: 'local', device: 'auto' }),
      }),
    )
  })

  it('throws when server returns status error', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'error', error: 'CUDA OOM' }),
    } as Response)

    await expect(api.loadModel('huge')).rejects.toThrow('CUDA OOM')
  })
})
