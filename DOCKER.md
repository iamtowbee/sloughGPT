# SloughGPT Docker Configuration

## 🐳 Docker Images

### Official Images

```bash
# Pull the latest official image
docker pull sloughgpt/sloughgpt:latest

# Pull specific version
docker pull sloughgpt/sloughgpt:v1.0.0

# Pull development version
docker pull sloughgpt/sloughgpt:dev
```

## 📋 Dockerfiles

### Multi-stage Production Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash sloughgpt

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY sloughgpt/ ./sloughgpt/
COPY sloughgpt.py .
COPY *.md ./
COPY requirements.txt .

# Create necessary directories
RUN mkdir -p data logs models checkpoints && \
    chown -R sloughgpt:sloughgpt /app

# Switch to non-root user
USER sloughgpt

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python3", "sloughgpt.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

### Development Dockerfile

```dockerfile
# Dockerfile.dev
FROM python:3.9

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs models checkpoints

# Expose ports
EXPOSE 8000 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command (development mode)
CMD ["python3", "sloughgpt.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

## 🐳 Docker Compose

### Complete Development Environment

```yaml
# docker-compose.yml
version: '3.8'

services:
  # SloughGPT API Server
  sloughgpt-api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    container_name: sloughgpt-api
    ports:
      - "8000:8000"
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://sloughgpt:password@postgres:5432/sloughgpt
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET_KEY=your-256-bit-secret-key-here
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    depends_on:
      - postgres
      - redis
    networks:
      - sloughgpt-network
    restart: unless-stopped

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: sloughgpt-postgres
    environment:
      POSTGRES_DB: sloughgpt
      POSTGRES_USER: sloughgpt
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - sloughgpt-network
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: sloughgpt-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - sloughgpt-network
    restart: unless-stopped

  # Nginx Reverse Proxy (Optional)
  nginx:
    image: nginx:alpine
    container_name: sloughgpt-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - sloughgpt-api
    networks:
      - sloughgpt-network
    restart: unless-stopped

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: sloughgpt-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - sloughgpt-network
    restart: unless-stopped

  # Grafana Dashboard
  grafana:
    image: grafana/grafana:latest
    container_name: sloughgpt-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    networks:
      - sloughgpt-network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  sloughgpt-network:
    driver: bridge
```

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  sloughgpt-api:
    image: sloughgpt/sloughgpt:latest
    container_name: sloughgpt-api-prod
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - LOG_LEVEL=WARNING
      - ENABLE_METRICS=true
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 1G
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.prod.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - sloughgpt-api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

## 🔧 Docker Commands

### Build Images

```bash
# Build development image
docker build -f Dockerfile.dev -t sloughgpt:dev .

# Build production image
docker build -t sloughgpt:latest .

# Build with build args
docker build \
  --build-arg PYTHON_VERSION=3.9 \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  -t sloughgpt:custom .
```

### Run Containers

```bash
# Run with default settings
docker run -d \
  --name sloughgpt \
  -p 8000:8000 \
  -p 8080:8080 \
  sloughgpt:latest

# Run with environment variables
docker run -d \
  --name sloughgpt-prod \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  -e JWT_SECRET_KEY=your-secret \
  -v $(pwd)/data:/app/data \
  sloughgpt:latest

# Run with mounted volumes
docker run -d \
  --name sloughgpt-dev \
  -p 8000:8000 \
  -p 8080:8080 \
  -v $(pwd)/sloughgpt:/app/sloughgpt \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  sloughgpt:dev
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# Start with custom compose file
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose logs -f sloughgpt-api

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build -d
```

## 🐳 Multi-Architecture Builds

### Build for Different Architectures

```bash
# Build for AMD64
docker build --platform linux/amd64 -t sloughgpt:amd64 .

# Build for ARM64
docker build --platform linux/arm64 -t sloughgpt:arm64 .

# Create multi-architecture image
docker buildx build --platform linux/amd64,linux/arm64 -t sloughgpt:multi .
```

### Docker Compose with Platform Selection

```yaml
# docker-compose.platform.yml
version: '3.8'

services:
  sloughgpt-api:
    platform: linux/amd64  # or linux/arm64
    build:
      context: .
      dockerfile: Dockerfile
    # ... rest of configuration
```

## 🔒 Security Best Practices

### Production Security

```dockerfile
# Production Dockerfile security
FROM python:3.9-slim

# Use non-root user
RUN useradd --create-home --shell /bin/bash sloughgpt

# Minimal base image
FROM python:3.9-slim as base

# Security scanning
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y security-updates && \
    rm -rf /var/lib/apt/lists/*

# Switch to non-root user early
USER sloughgpt

# Read-only filesystem where possible
VOLUME ["/app/data", "/app/logs", "/app/models"]

# Limit attack surface
EXPOSE 8000 8080

# Health checks
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')"
```

### Environment Variables for Security

```bash
# Don't pass secrets in Dockerfile
# Use environment variables or secrets management

# Build with build args
docker build --build-arg SECRET_KEY=$SECRET_KEY -t sloughgpt .

# Use Docker secrets (Docker Swarm)
docker service create \
  --secret jwt_secret \
  sloughgpt

# Use Kubernetes secrets
kubectl create secret generic sloughgpt-secrets \
  --from-literal=jwt-secret=$JWT_SECRET
```

## 🚀 Deployment Scripts

### Automated Deployment

```bash
#!/bin/bash
# deploy.sh

set -e

IMAGE_TAG=${1:-latest}
ENVIRONMENT=${2:-production}

echo "🚀 Deploying SloughGPT $IMAGE_TAG to $ENVIRONMENT"

# Build and push
docker build -t sloughgpt/sloughgpt:$IMAGE_TAG .
docker push sloughgpt/sloughgpt:$IMAGE_TAG

# Deploy to staging
if [ "$ENVIRONMENT" = "staging" ]; then
    docker-compose -f docker-compose.staging.yml up -d
fi

# Deploy to production
if [ "$ENVIRONMENT" = "production" ]; then
    docker-compose -f docker-compose.prod.yml up -d
fi

# Health check
sleep 30
curl -f http://localhost:8000/health || exit 1

echo "✅ Deployment successful!"
```

## 📊 Monitoring and Logging

### Logging Configuration

```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  sloughgpt-api:
    image: sloughgpt/sloughgpt:latest
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    environment:
      - LOG_LEVEL=INFO
      - ENABLE_FILE_LOGGING=true
    volumes:
      - ./logs:/app/logs
```

### Metrics Collection

```bash
# Monitor Docker containers
docker stats sloughgpt-api

# View resource usage
docker stats --no-stream sloughgpt-api

# Check container health
docker inspect sloughgpt-api | jq '.[0].State.Health'
```

## 🎯 Quick Start with Docker

```bash
# Clone and setup
git clone https://github.com/sloughgpt/sloughgpt.git
cd sloughgpt

# Start development environment
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Check health
curl http://localhost:8000/health

# View logs
docker-compose logs -f

# Access services
echo "🌐 API Server: http://localhost:8000"
echo "📊 Admin Dashboard: http://localhost:8080"
echo "📈 Grafana: http://localhost:3000 (admin/admin)"
echo "🔍 Prometheus: http://localhost:9090"
```

---

**🐳 SloughGPT Enterprise AI Framework - Docker Ready!**