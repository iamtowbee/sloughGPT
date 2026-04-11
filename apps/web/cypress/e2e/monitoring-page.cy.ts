/**
 * Monitoring page - system metrics
 */
describe('Monitoring page', () => {
  const api = 'http://localhost:8000'

  beforeEach(() => {
    cy.intercept('GET', `${api}/health`, {
      statusCode: 200,
      body: { status: 'healthy' },
    }).as('health')
    cy.intercept('GET', `${api}/metrics`, {
      statusCode: 200,
      body: {
        cpu_percent: 25,
        memory_percent: 50,
        active_connections: 0,
      },
    }).as('metrics')
  })

  it('loads the monitoring page', () => {
    cy.visit('/monitoring')
    cy.wait(['@health', '@metrics'])

    cy.contains('Monitoring').should('be.visible')
  })

  it('shows system metrics', () => {
    cy.visit('/monitoring')
    cy.wait(['@health', '@metrics'])

    cy.contains('CPU').should('be.visible')
    cy.contains('Memory').should('be.visible')
  })
})
