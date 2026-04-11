/**
 * Home page - dashboard overview
 */
describe('Home page - dashboard', () => {
  const api = 'http://localhost:8000'

  beforeEach(() => {
    cy.intercept('GET', `${api}/health`, {
      statusCode: 200,
      body: { status: 'healthy', model_type: 'gpt2', model_loaded: false },
    }).as('health')
    cy.intercept('GET', `${api}/models`, { fixture: 'models-list.json' }).as('models')
    cy.intercept('GET', `${api}/datasets`, { body: { datasets: [] } }).as('datasets')
  })

  it('displays the dashboard with all feature cards', () => {
    cy.visit('/')
    cy.wait(['@health', '@models', '@datasets'])

    // Check title
    cy.contains('SloughGPT').should('be.visible')

    // Check feature cards
    cy.contains('Chat').should('be.visible')
    cy.contains('Models').should('be.visible')
    cy.contains('Training').should('be.visible')
    cy.contains('Datasets').should('be.visible')
    cy.contains('Monitor').should('be.visible')
    cy.contains('API Docs').should('be.visible')
  })

  it('shows API status indicator', () => {
    cy.visit('/')
    cy.wait(['@health'])

    cy.contains('API').should('be.visible')
  })

  it('links to chat page', () => {
    cy.visit('/')
    cy.wait('@health')
    cy.contains('Chat').closest('a').should('have.attr', 'href', '/chat')
  })
})
