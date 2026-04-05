import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { ListRow } from './list-row'

describe('ListRow', () => {
  it('renders as type=button with title', () => {
    const html = renderToStaticMarkup(<ListRow title="Conversation A" />)
    expect(html).toContain('type="button"')
    expect(html).toContain('Conversation A')
  })

  it('renders description and trailing', () => {
    const html = renderToStaticMarkup(
      <ListRow title="T" description="Updated" trailing="›" />,
    )
    expect(html).toContain('Updated')
    expect(html).toContain('›')
  })
})
