/**
 * Chat UI + mocked inference API (paths match `NEXT_PUBLIC_API_URL`, default http://localhost:8000).
 */
describe('Chat page — inference contract (mocked API)', () => {
  const api = 'http://localhost:8000'

  beforeEach(() => {
    cy.intercept('GET', `${api}/models`, { fixture: 'models-list.json' }).as('getModels')
    cy.intercept('GET', `${api}/health`, {
      statusCode: 200,
      body: { status: 'healthy', model_type: 'gpt2', model_loaded: true },
    }).as('health')
    cy.intercept('POST', `${api}/chat/stream`, {
      statusCode: 503,
      body: 'no sse in e2e',
    }).as('chatStreamFail')
    cy.intercept(
      { method: 'POST', url: /\/chat$/ },
      {
        statusCode: 200,
        body: {
          text: 'Mock reply: integration test body.',
          model: 'gpt2-engine',
          tokens_generated: 5,
        },
      },
    ).as('chat')
  })

  it('sends a user message and shows an assistant reply', () => {
    cy.visit('/chat/')

    // Chat gates send until /health reports model_loaded; wait for mocks before typing.
    cy.wait(['@health', '@getModels'], { timeout: 30_000 })

    cy.get('[data-testid="chat-message-input"]', { timeout: 30_000 }).should('be.visible').type('Hello from Cypress')
    cy.get('[data-testid="chat-message-input"]').should('have.value', 'Hello from Cypress')
    cy.get('[data-testid="chat-send-button"]').should('not.be.disabled').click()

    cy.wait('@chat', { timeout: 30_000 })
    cy.get('.whitespace-pre-wrap').last().should('include.text', 'Mock reply: integration test body.')
  })
})
