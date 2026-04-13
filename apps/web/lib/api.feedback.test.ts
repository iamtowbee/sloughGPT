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

describe('api.recordFeedback', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('POSTs to /feedback/record with feedback data', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({
        status: 'recorded',
        feedback_id: 'test-123',
        stats: { db_stats: { feedback_total: 1 } },
      }),
    } as Response)

    const result = await api.recordFeedback({
      userMessage: 'Hello?',
      assistantResponse: 'Hi there!',
      rating: 'thumbs_up',
      userId: 'user-1',
    })

    expect(result.status).toBe('recorded')
    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:9/feedback/record',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          user_message: 'Hello?',
          assistant_response: 'Hi there!',
          rating: 'thumbs_up',
          user_id: 'user-1',
        }),
      }),
    )
  })

  it('handles thumbs_down rating', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'recorded', feedback_id: 'test-456' }),
    } as Response)

    await api.recordFeedback({
      userMessage: 'What is 2+2?',
      assistantResponse: '5',
      rating: 'thumbs_down',
    })

    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:9/feedback/record',
      expect.objectContaining({
        body: expect.stringContaining('thumbs_down'),
      }),
    )
  })
})

describe('api.getFeedbackStats', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('calls /meta-weights/stats endpoint', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({
        db_stats: { feedback_total: 10, thumbs_up: 7, thumbs_down: 3 },
        current_weights: { temperature: 0.8 },
      }),
    } as Response)

    await api.getFeedbackStats()

    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:9/meta-weights/stats',
      expect.any(Object),
    )
  })
})

describe('api.getUserAdapters', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('calls /user-adapters endpoint', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ stats: { total_users: 5, total_size_mb: 0.25 } }),
    } as Response)

    await api.getUserAdapters()

    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:9/user-adapters',
      expect.any(Object),
    )
  })
})

describe('api.getWorkflowStatus', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('GETs /workflow/status', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({
        running: true,
        stats: { workflow_runs: 10 },
      }),
    } as Response)

    const result = await api.getWorkflowStatus()

    expect(result.running).toBe(true)
    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:9/workflow/status',
      expect.objectContaining({ headers: expect.any(Object) }),
    )
  })
})

describe('api.getTrainingStats', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('calls /feedback-stats/training endpoint', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({
        total_conversations: 20,
        thumbs_up: 15,
        thumbs_down: 5,
        available_dpo_pairs: 3,
        available_sft_examples: 12,
      }),
    } as Response)

    await api.getTrainingStats()

    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:9/feedback-stats/training',
      expect.any(Object),
    )
  })
})

describe('api.exportTrainingData', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('POSTs to /feedback/export-training with format', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({
        status: 'exported',
        filepath: '/data/dpo_123.jsonl',
        count: 10,
      }),
    } as Response)

    const result = await api.exportTrainingData('dpo')

    expect(result.count).toBe(10)
    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:9/feedback/export-training',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ format: 'dpo' }),
      }),
    )
  })
})
