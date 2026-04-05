import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { EmptyCard } from './empty-card'

describe('EmptyCard', () => {
  it('renders dashed card with title', () => {
    const html = renderToStaticMarkup(<EmptyCard title="No rows" />)
    expect(html).toContain('border-dashed')
    expect(html).toContain('No rows')
  })

  it('renders description and children', () => {
    const html = renderToStaticMarkup(
      <EmptyCard title="T" description="D">
        <p>Child</p>
      </EmptyCard>,
    )
    expect(html).toContain('D')
    expect(html).toContain('Child')
  })
})
