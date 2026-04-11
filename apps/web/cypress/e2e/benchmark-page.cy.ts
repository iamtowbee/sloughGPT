/**
 * Benchmark page - model performance comparison
 */
describe('Benchmark page', () => {
  const api = 'http://localhost:8000'

  beforeEach(() => {
    cy.intercept('GET', `${api}/health`, {
      statusCode: 200,
      body: { status: 'healthy' },
    }).as('health')
    cy.intercept('POST', `${api}/benchmark/run`, {
      statusCode: 200,
      body: {
        model_name: 'gpt2',
        num_parameters: 124000000,
        memory_mb: 500,
        inference_time_ms: 100,
        throughput_tokens_per_sec: 50,
        latency_p50_ms: 95,
        latency_p95_ms: 150,
        latency_p99_ms: 200,
      },
    }).as('benchmark')
  })

  it('loads the benchmark page', () => {
    cy.visit('/benchmark')
    cy.wait('@health')

    cy.contains('Benchmark').should('be.visible')
  })

  it('shows benchmark form', () => {
    cy.visit('/benchmark')
    cy.wait('@health')

    cy.contains('Run Benchmark').should('be.visible')
  })
})
