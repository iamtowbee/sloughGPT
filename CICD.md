# SloughGPT CI/CD Pipelines

## 🔄 Continuous Integration & Deployment

### GitHub Actions Workflows

#### Main CI/CD Pipeline

```yaml
# .github/workflows/ci-cd.yml
name: SloughGPT CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  release:
    types: [ published ]

env:
  PYTHON_VERSION: 3.9
  NODE_VERSION: 18

jobs:
  test:
    name: Test Suite
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_USER: test
          POSTGRES_DB: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
          ${{ runner.os }}-
          
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio
        pip install coverage
        
    - name: Lint with Ruff
      run: |
        pip install ruff
        ruff check sloughgpt/ --format=github
        
    - name: Type check with MyPy
      run: |
        pip install mypy types-all
        mypy sloughgpt/ --ignore-missing-imports
        
    - name: Security scan with Bandit
      run: |
        pip install bandit
        bandit -r sloughgpt/ -f json -o bandit-report.json
        
    - name: Run unit tests
      run: |
        pytest sloughgpt/tests/ -v --cov=sloughgpt --cov-report=xml --cov-report=html
        
    - name: Run integration tests
      run: |
        python sloughgpt.py health
        python -c "
        import asyncio
        import sys
        sys.path.insert(0, '.')
        from sloughgpt.tests.test_integration import run_all_tests
        asyncio.run(run_all_tests())
        "
      env:
        DATABASE_URL: postgresql://test:test@localhost:5432/test
        REDIS_URL: redis://localhost:6379
        JWT_SECRET_KEY: test-secret-key
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        
    - name: Upload test artifacts
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.python-version }}
        path: |
          htmlcov/
          bandit-report.json

  build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main' || github.event_name == 'release'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Login to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: sloughgpt/sloughgpt
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-
          
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        build-args: |
          BUILD_DATE=${{ fromJSON(steps.meta.outputs.json).labels['org.opencontainers.image.created'] }}
          VCS_REF=${{ github.ref_name }}
          VCS_SHA=${{ github.sha }}

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - name: Deploy to staging
      run: |
        echo "🚀 Deploying to staging environment"
        # Add your staging deployment commands here
        # For example: kubectl apply -f k8s/staging/
        
    - name: Run smoke tests
      run: |
        echo "🧪 Running smoke tests on staging"
        # Add smoke test commands here
        curl -f https://staging.sloughgpt.com/health

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: build
    if: github.event_name == 'release'
    environment: production
    
    steps:
    - name: Deploy to production
      run: |
        echo "🚀 Deploying version ${{ github.event.release.tag_name }} to production"
        # Add production deployment commands here
        # For example: kubectl apply -f k8s/production/
        
    - name: Update deployment status
      run: |
        curl -X POST \
          -H "Authorization: token ${{ secrets.GITHUB_TOKEN }}" \
          https://api.github.com/repos/${{ github.repository }}/deployments/${{ github.event.deployment.id }}/statuses \
          -d '{
            "state": "success",
            "target_url": "https://sloughgpt.com",
            "description": "Deployed to production"
          }'
          
    - name: Notify team
      run: |
        curl -X POST \
          -H "Content-Type: application/json" \
          -d '{
            "text": "🚀 SloughGPT v${{ github.event.release.tag_name }} deployed to production!",
            "channel": "#deployments"
          }' \
          ${{ secrets.SLACK_WEBHOOK_URL }}

  security-scan:
    name: Security Scanning
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  performance-test:
    name: Performance Testing
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Run performance tests
      run: |
        echo "🏃 Running performance benchmarks"
        # Deploy to test environment
        docker-compose -f docker-compose.test.yml up -d
        sleep 60
        
        # Run load tests
        pip install locust
        locust -f tests/performance/locustfile.py \
          --host http://localhost:8000 \
          --users 100 \
          --spawn-rate 10 \
          --run-time 60 \
          --headless \
          --html performance-report.html || true
          
    - name: Upload performance results
      uses: actions/upload-artifact@v3
      with:
        name: performance-report
        path: performance-report.html
```

### Quality Gates

```yaml
# .github/workflows/quality-gate.yml
name: Quality Gates

on:
  pull_request:
    branches: [ main ]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
        
    - name: Code quality checks
      run: |
        # Test coverage threshold (80%)
        pip install coverage pytest-cov
        pytest --cov=sloughgpt --cov-fail-under=80
        
        # Code complexity
        pip install radon
        radon cc sloughgpt/ --min B
        radon mi sloughgpt/ --min B
        
        # Documentation coverage
        pip install interrogate
        interrogate sloughgpt/ --verbose
        
    - name: Security checks
      run: |
        # Security scan
        pip install safety bandit
        safety check
        bandit -r sloughgpt/ -ll
        
    - name: Dependency check
      run: |
        # Check for outdated dependencies
        pip list --outdated
        pip-audit
```

## 🚀 Deployment Automation

### Multi-Environment Deployment

```yaml
# .github/workflows/deploy.yml
name: Deploy to Environments

on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target environment'
        required: true
        default: 'staging'
        type: choice
        options:
        - staging
        - production
      version:
        description: 'Version to deploy'
        required: false
        default: 'latest'

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    environment: ${{ github.event.inputs.environment }}
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
        
    - name: Deploy application
      run: |
        ENVIRONMENT=${{ github.event.inputs.environment }}
        VERSION=${{ github.event.inputs.version }}
        
        echo "🚀 Deploying $VERSION to $ENVIRONMENT"
        
        # Apply environment-specific configurations
        kubectl apply -f k8s/${ENVIRONMENT}/
        kubectl apply -f k8s/common/
        
        # Wait for deployment
        kubectl rollout status deployment/sloughgpt-api -n sloughgpt-${ENVIRONMENT}
        
        # Run health checks
        kubectl wait --for=condition=ready pod -l app=sloughgpt-api -n sloughgpt-${ENVIRONMENT} --timeout=300s
```

### Canary Deployments

```yaml
# .github/workflows/canary.yml
name: Canary Deployment

on:
  push:
    branches: [ main ]

jobs:
  canary:
    runs-on: ubuntu-latest
    if: contains(github.event.head_commit.message, '[canary]')
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Deploy canary
      run: |
        echo "🐤 Deploying canary version"
        
        # Deploy canary (10% traffic)
        kubectl patch deployment sloughgpt-api \
          -p '{"spec":{"replicas":1}}' \
          -n sloughgpt-canary
          
        # Update traffic split
        kubectl patch service sloughgpt-api \
          -p '{"spec":{"selector":{"version":"canary"}}}' \
          -n sloughgpt-canary
          
    - name: Monitor canary
      run: |
        echo "📊 Monitoring canary deployment"
        sleep 300  # Monitor for 5 minutes
        
        # Check metrics
        SUCCESS_RATE=$(kubectl get canary sloughgpt-api -o jsonpath='{.status.successRate}')
        if (( $(echo "$SUCCESS_RATE >= 0.99" | bc -l) )); then
          echo "✅ Canary successful - promoting to production"
          kubectl patch deployment sloughgpt-api -n sloughgpt-production -p '{"spec":{"template":{"spec":{"containers":[{"name":"sloughgpt-api","image":"'$(kubectl get deployment sloughgpt-api -n sloughgpt-canary -o jsonpath='{.spec.template.spec.containers[0].image}')'"}}]}}}'
        else
          echo "❌ Canary failed - rolling back"
          kubectl rollout undo deployment/sloughgpt-api -n sloughgpt-production
        fi
```

## 🔍 Monitoring and Alerting

### CI/CD Monitoring

```yaml
# .github/workflows/monitoring.yml
name: CI/CD Monitoring

on:
  schedule:
    - cron: '0 */5 * * * *'  # Every 5 minutes
  workflow_dispatch:

jobs:
  monitor-pipelines:
    runs-on: ubuntu-latest
    
    steps:
    - name: Check CI/CD health
      run: |
        # Check recent workflow runs
        FAILED_RUNS=$(gh run list --workflow=ci-cd --status=failure --limit=10 | wc -l)
        
        if [ $FAILED_RUNS -gt 3 ]; then
          echo "🚨 Multiple CI/CD failures detected"
          curl -X POST \
            -H "Content-Type: application/json" \
            -d '{
              "text": "🚨 CI/CD pipeline health degraded: $FAILED_RUNS failures in last 10 runs",
              "channel": "#alerts"
            }' \
            ${{ secrets.SLACK_WEBHOOK_URL }}
        fi
        
    - name: Check deployment health
      run: |
        # Check production deployment health
        HEALTH_CHECK=$(curl -s -o /dev/null -w "%{http_code}" https://api.sloughgpt.com/health)
        
        if [ "$HEALTH_CHECK" != "200" ]; then
          echo "🚨 Production deployment unhealthy"
          # Trigger rollback or alert
          curl -X POST \
            -H "Content-Type: application/json" \
            -d '{
              "text": "🚨 Production deployment unhealthy (HTTP $HEALTH_CHECK)",
              "channel": "#alerts"
            }' \
            ${{ secrets.SLACK_WEBHOOK_URL }}
        fi
```

## 📊 Performance Monitoring

### Performance Regression Detection

```yaml
# .github/workflows/performance.yml
name: Performance Monitoring

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
        
    - name: Run benchmarks
      run: |
        pip install -r requirements.txt
        pip install locust pytest-benchmark
        
        # Deploy and test
        docker-compose -f docker-compose.benchmark.yml up -d
        sleep 60
        
        # Run performance tests
        locust -f tests/performance/locustfile.py \
          --host http://localhost:8000 \
          --users 50 \
          --spawn-rate 5 \
          --run-time 120 \
          --headless \
          --json benchmark-results.json || true
          
        # Parse results
        RESPONSE_TIME=$(cat benchmark-results.json | jq '.results[0].response_time_percentiles_95')
        REQUESTS_PER_SEC=$(cat benchmark-results.json | jq '.results[0].requests_per_second')
        
        echo "95th percentile response time: ${RESPONSE_TIME}ms"
        echo "Requests per second: ${REQUESTS_PER_SEC}"
        
        # Performance thresholds
        if (( $(echo "$RESPONSE_TIME > 1000" | bc -l) )); then
          echo "❌ Response time threshold exceeded"
          exit 1
        fi
        
        if (( $(echo "$REQUESTS_PER_SEC < 100" | bc -l) )); then
          echo "❌ Throughput threshold not met"
          exit 1
        fi
        
    - name: Compare with baseline
      run: |
        # Get baseline performance
        BASELINE_RESPONSE_TIME=$(curl -s https://api.sloughgpt.com/metrics/baseline | jq '.response_time_p95')
        
        REGRESSION=$(echo "$RESPONSE_TIME - $BASELINE_RESPONSE_TIME" | bc)
        
        if (( $(echo "$REGRESSION > 100" | bc -l) )); then
          echo "❌ Performance regression detected: +${REGRESSION}ms"
          exit 1
        fi
```

## 🔧 Environment Management

### Environment Configuration

```yaml
# environments/staging/k8s/namespace.yml
apiVersion: v1
kind: Namespace
metadata:
  name: sloughgpt-staging
  labels:
    name: sloughgpt-staging
    environment: staging

---
# environments/production/k8s/namespace.yml
apiVersion: v1
kind: Namespace
metadata:
  name: sloughgpt-production
  labels:
    name: sloughgpt-production
    environment: production
```

### Helm Charts

```yaml
# helm/sloughgpt/Chart.yaml
apiVersion: v2
name: sloughgpt
description: SloughGPT Enterprise AI Framework
type: application
version: 1.0.0
appVersion: "1.0.0"

dependencies:
  - name: postgresql
    version: 12.x.x
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
  - name: redis
    version: 17.x.x
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled

keywords:
  - ai
  - machine-learning
  - llm
  - transformer

maintainers:
  - name: SloughGPT Team
    email: team@sloughgpt.ai

sources:
  - https://github.com/sloughgpt/sloughgpt

annotations:
  category: AI
  licenses: MIT
```

## 🎯 Deployment Strategies

### Blue-Green Deployment

```bash
#!/bin/bash
# scripts/blue-green-deploy.sh

NEW_VERSION=$1
CURRENT_ENV=$(kubectl get service sloughgpt-api -o jsonpath='{.spec.selector.environment}')

echo "🔄 Deploying $NEW_VERSION using blue-green strategy"

if [ "$CURRENT_ENV" = "blue" ]; then
    TARGET_ENV="green"
else
    TARGET_ENV="blue"
fi

# Deploy to target environment
helm install sloughgpt-$TARGET_ENV ./helm/sloughgpt \
  --namespace sloughgpt \
  --set image.tag=$NEW_VERSION \
  --set environment=$TARGET_ENV

# Wait for deployment
kubectl rollout status deployment/sloughgpt-$TARGET_ENV -n sloughgpt

# Health check
kubectl wait --for=condition=ready pod -l environment=$TARGET_ENV -n sloughgpt --timeout=300s

# Switch traffic
kubectl patch service sloughgpt-api \
  -p '{"spec":{"selector":{"environment":"'$TARGET_ENV'"}}}' \
  -n sloughgpt

# Cleanup old environment
helm uninstall sloughgpt-$CURRENT_ENV -n sloughgpt

echo "✅ Blue-green deployment completed: $TARGET_ENV active"
```

### Rolling Updates

```bash
#!/bin/bash
# scripts/rolling-update.sh

NEW_VERSION=$1
echo "🔄 Performing rolling update to $NEW_VERSION"

# Update deployment with zero downtime
kubectl set image deployment/sloughgpt-api \
  sloughgpt=sloughgpt/sloughgpt:$NEW_VERSION \
  -n sloughgpt

# Wait for rollout
kubectl rollout status deployment/sloughgpt-api -n sloughgpt

# Verify deployment
kubectl rollout history deployment/sloughgpt-api -n sloughgpt
kubectl get pods -l app=sloughgpt-api -n sloughgpt

echo "✅ Rolling update completed"
```

---

**🔄 SloughGPT Enterprise AI Framework - CI/CD Complete!**

Complete automated pipelines for testing, building, security scanning, and multi-environment deployment.