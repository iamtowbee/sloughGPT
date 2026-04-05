import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { AppRouteHeader, AppRouteHeaderLead } from './AppRouteHeader'

describe('AppRouteHeader', () => {
  it('renders left and right slots', () => {
    const html = renderToStaticMarkup(
      <AppRouteHeader left={<span data-testid="L">Left</span>} right={<span data-testid="R">Right</span>} />,
    )
    expect(html).toContain('Left')
    expect(html).toContain('Right')
  })

  it('omits right column when right is undefined', () => {
    const html = renderToStaticMarkup(<AppRouteHeader left={<span>Only</span>} />)
    expect(html).toContain('Only')
    expect(html).not.toContain('justify-end')
  })

  it('merges className onto header', () => {
    const html = renderToStaticMarkup(
      <AppRouteHeader className="items-start border-b" left={<span>x</span>} />,
    )
    expect(html).toContain('items-start')
    expect(html).toContain('border-b')
  })
})

describe('AppRouteHeaderLead', () => {
  it('wraps title and subtitle', () => {
    const html = renderToStaticMarkup(
      <AppRouteHeaderLead title="Models" subtitle={<>API: ok</>} />,
    )
    expect(html).toContain('Models')
    expect(html).toContain('API: ok')
    expect(html).toContain('sl-h1')
  })

  it('renders children after subtitle', () => {
    const html = renderToStaticMarkup(
      <AppRouteHeaderLead title="Monitoring" subtitle="Inference: gpt2">
        <p className="note">Host note</p>
      </AppRouteHeaderLead>,
    )
    expect(html).toContain('Host note')
    expect(html).toContain('Inference: gpt2')
  })
})
