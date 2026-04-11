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

  it('displays the dashboard title', () => {
    cy.visit('/')
    cy.wait(['@health', '@models', '@datasets'])

    cy.contains('h1', 'SloughGPT').should('be.visible')
  })

  it('shows API status card', () => {
    cy.visit('/')
    cy.wait(['@health'])

    cy.contains('API status').should('be.visible')
  })

  it('shows quick actions section', () => {
    cy.visit('/')
    cy.wait(['@health', '@models', '@datasets'])

    cy.contains('Quick actions').should('be.visible')
  })
})
