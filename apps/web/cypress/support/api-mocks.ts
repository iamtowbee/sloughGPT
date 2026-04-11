/**
 * Global API intercepts for e2e tests
 * This file is loaded in cypress/support/e2e.ts
 */

const api = 'http://localhost:8000'

Cypress.Commands.add('mockHealth', (overrides = {}) => {
  cy.intercept('GET', `${api}/health`, {
    statusCode: 200,
    body: {
      status: 'healthy',
      model_type: 'gpt2',
      model_loaded: false,
      ...overrides,
    },
  }).as('health')
})

Cypress.Commands.add('mockModels', () => {
  cy.intercept('GET', `${api}/models`, {
    statusCode: 200,
    body: { models: [] },
  }).as('models')
})

Cypress.Commands.add('mockDatasets', () => {
  cy.intercept('GET', `${api}/datasets`, {
    statusCode: 200,
    body: { datasets: [] },
  }).as('datasets')
})

Cypress.Commands.add('mockAll', () => {
  cy.mockHealth()
  cy.mockModels()
  cy.mockDatasets()
})
