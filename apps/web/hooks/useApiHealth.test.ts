import { describe, expect, it } from 'vitest'

import { inferenceHealthLabel } from './useApiHealth'

describe('inferenceHealthLabel', () => {
  it('covers loading, offline, ready, and idle API states', () => {
    expect(inferenceHealthLabel(null)).toBe('checking...')
    expect(inferenceHealthLabel('offline')).toBe('disconnected')
    expect(
      inferenceHealthLabel({
        status: 'ok',
        model_loaded: true,
        model_type: 'gpt2',
      }),
    ).toBe('inference ready · gpt2')
    expect(
      inferenceHealthLabel({
        status: 'ok',
        model_loaded: false,
        model_type: 'none',
      }),
    ).toBe('connected · no weights (none)')
  })
})
