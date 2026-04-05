import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { PromptComposer } from '@sloughgpt/strui'

/**
 * Contract for /chat + Cypress: data-testids must stay stable on the strui composer shell.
 */
describe('Chat PromptComposer E2E contract', () => {
  it('exposes chat-message-input and chat-send-button when wired like the chat page', () => {
    const html = renderToStaticMarkup(
      <PromptComposer
        value="ok"
        onValueChange={() => {}}
        onSubmit={() => {}}
        textAreaProps={{ 'data-testid': 'chat-message-input' }}
        sendButtonProps={{ 'data-testid': 'chat-send-button' }}
      />,
    )
    expect(html).toContain('data-testid="chat-message-input"')
    expect(html).toContain('data-testid="chat-send-button"')
  })
})
