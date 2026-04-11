/**
 * API Docs page - Swagger/OpenAPI documentation
 */
describe('API Docs page', () => {
  const api = 'http://localhost:8000'

  beforeEach(() => {
    cy.intercept('GET', `${api}/health`, {
      statusCode: 200,
      body: { status: 'healthy' },
    }).as('health')
    cy.intercept('GET', `${api}/openapi.json`, {
      statusCode: 200,
      body: {
        openapi: '3.0.0',
        info: { title: 'SloughGPT API' },
        paths: {},
      },
    }).as('openapi')
  })

  it('loads the API docs page', () => {
    cy.visit('/api-docs')
    cy.wait('@health')

    cy.contains('API').should('be.visible')
    cy.contains('Docs').should('be.visible')
  })
})
