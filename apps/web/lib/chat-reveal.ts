/**
 * Chunked “typing” steps for assistant text. Steps by `chunkSize` code units; the last
 * value is always the full string (covers tails not divisible by `chunkSize`).
 */
export function revealTypingSequence(fullContent: string, chunkSize: number): string[] {
  if (chunkSize < 1) {
    throw new RangeError('chunkSize must be >= 1')
  }
  const out: string[] = []
  for (let i = 0; i <= fullContent.length; i += chunkSize) {
    out.push(fullContent.slice(0, i))
  }
  if (out[out.length - 1] !== fullContent) {
    out.push(fullContent)
  }
  return out
}
