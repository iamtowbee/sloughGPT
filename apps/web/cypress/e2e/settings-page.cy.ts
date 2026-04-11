/**
 * Settings page - configuration
 */
describe('Settings page', () => {
  const api = 'http://localhost:8000'

  beforeEach(() => {
    cy.intercept('GET', `${api}/health`, {
      statusCode: 200,
      body: { status: 'healthy' },
    }).as('health')
  })

  it('loads the settings page', () => {
    cy.visit('/settings')
    cy.wait('@health')

    cy.contains('Settings').should('be.visible')
  })
})
