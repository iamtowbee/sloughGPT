/** Models page + mocked `GET /models` / `GET /health` (no Python server). */

describe('Models page — API wiring (mocked)', () => {
  beforeEach(() => {
    // Match list endpoint only (not /models/hf). Use handler form so Cypress matches reliably.
    cy.intercept('GET', 'http://localhost:8000/models', (req) => {
      req.reply({ fixture: 'models-list.json' })
    }).as('models')
    cy.intercept('GET', 'http://localhost:8000/health', {
      statusCode: 200,
      body: { status: 'healthy', model_type: 'gpt2', model_loaded: false },
    })
  })

  it('lists models from GET /models', () => {
    cy.visit('/models/')
    cy.wait('@models', { timeout: 20_000 })
    cy.contains('GPT-2').should('be.visible')
    cy.contains('Fixture model').should('be.visible')
  })

  it('shows inference-ready API line when health reports weights loaded', () => {
    cy.intercept('GET', 'http://localhost:8000/health', {
      statusCode: 200,
      body: { status: 'healthy', model_type: 'gpt2', model_loaded: true },
    })
    cy.visit('/models/')
    cy.wait('@models', { timeout: 20_000 })
    cy.get('[data-testid="models-api-status"]').should('contain.text', 'inference ready')
  })

  it('shows disconnected when health fails', () => {
    cy.intercept('GET', 'http://localhost:8000/health', { forceNetworkError: true })
    cy.visit('/models/')
    cy.wait('@models', { timeout: 20_000 })
    cy.get('[data-testid="models-api-status"]').should('contain.text', 'disconnected')
  })
})
