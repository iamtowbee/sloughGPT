/** Models page + mocked `GET /models` / `GET /health` (no Python server). */
describe('Models page — API wiring (mocked)', () => {
  const apiBase = 'http://localhost:8000'

  beforeEach(() => {
    cy.intercept('GET', `${apiBase}/health`, {
      statusCode: 200,
      body: { status: 'healthy', model_type: 'gpt2' },
    })
    cy.intercept('GET', `${apiBase}/models`, { fixture: 'models-list.json' }).as('models')
  })

  it('lists models from GET /models', () => {
    cy.visit('/models')
    cy.wait('@models')
    cy.contains('GPT-2').should('be.visible')
    cy.contains('Fixture model').should('be.visible')
  })
})
