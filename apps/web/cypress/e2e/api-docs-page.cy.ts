/**
 * API Docs page - endpoint documentation
 */
describe('API Docs page', () => {
  const api = 'http://localhost:8000'

  beforeEach(() => {
    cy.intercept('GET', `${api}/health`, {
      statusCode: 200,
      body: { status: 'healthy' },
    }).as('health')
  })

  it('loads the API docs page', () => {
    cy.visit('/api-docs')
    cy.wait('@health')

    cy.contains('h1', 'API documentation').should('be.visible')
    cy.contains('Base URL').should('be.visible')
  })

  it('shows endpoint documentation', () => {
    cy.visit('/api-docs')
    cy.wait('@health')

    cy.contains('Quick examples').should('be.visible')
  })
})
