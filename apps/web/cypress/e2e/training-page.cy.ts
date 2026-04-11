/**
 * Training page - training jobs management
 */
describe('Training page', () => {
  const api = 'http://localhost:8000'

  beforeEach(() => {
    cy.intercept('GET', `${api}/health`, {
      statusCode: 200,
      body: { status: 'healthy' },
    }).as('health')
    cy.intercept('GET', `${api}/training/jobs`, {
      statusCode: 200,
      body: { jobs: [] },
    }).as('jobs')
  })

  it('loads the training page', () => {
    cy.visit('/training')
    cy.wait(['@health', '@jobs'])

    cy.contains('Training').should('be.visible')
  })

  it('shows start training button', () => {
    cy.visit('/training')
    cy.wait('@health')

    cy.contains('Start').should('be.visible')
  })
})
