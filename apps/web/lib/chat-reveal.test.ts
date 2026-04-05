import { describe, expect, it } from 'vitest'

import { revealTypingSequence } from './chat-reveal'

describe('revealTypingSequence', () => {
  it('ends on full text when length is not a multiple of chunkSize', () => {
    expect(revealTypingSequence('abcd', 3)).toEqual(['', 'abc', 'abcd'])
  })

  it('does not duplicate when the loop already ends on full text', () => {
    expect(revealTypingSequence('abc', 3)).toEqual(['', 'abc'])
  })

  it('handles empty string', () => {
    expect(revealTypingSequence('', 3)).toEqual([''])
  })

  it('handles single character', () => {
    expect(revealTypingSequence('x', 3)).toEqual(['', 'x'])
  })

  it('rejects non-positive chunkSize', () => {
    expect(() => revealTypingSequence('hi', 0)).toThrow(RangeError)
  })
})
