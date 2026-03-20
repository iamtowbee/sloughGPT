# SloughGPT Environment Variables

Complete reference for all environment variables used in SloughGPT.

## Quick Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SLAUGHGPT_API_KEY` | Yes | - | API key for authentication |
| `SLAUGHGPT_JWT_SECRET` | Yes | - | Secret for JWT token signing |
| `SLAUGHGPT_ENV` | No | `development` | Environment mode |
| `SLAUGHGPT_HOST` | No | `0.0.0.0` | Server host |
| `SLAUGHGPT_PORT` | No | `8000` | Server port |

---

## Authentication

### SLAUGHGPT_API_KEY
**Required in production**

API key for authenticating requests.

```bash
# Generate a secure key
openssl rand -hex 32

# Set in .env
SLAUGHGPT_API_KEY=your-generated-key-here
```

### SLAUGHGPT_API_KEYS
**Optional**

Comma-separated list of multiple valid API keys.

```bash
SLAUGHGPT_API_KEYS=key1,key2,key3
```

### SLAUGHGPT_JWT_SECRET
**Required in production**

Secret key for signing JWT tokens.

```bash
# Generate a secure secret
openssl rand -hex 64

# Set in .env
SLAUGHGPT_JWT_SECRET=your-64-character-secret
```

### JWT_ALGORITHM
**Optional**

JWT signing algorithm.

```bash
JWT_ALGORITHM=HS256  # Default
```

### JWT_EXPIRATION_HOURS
**Optional**

JWT token expiration time in hours.

```bash
JWT_EXPIRATION_HOURS=24  # Default
```

---

## Security

### RATE_LIMIT_REQUESTS_PER_MINUTE
**Optional**

Maximum requests per minute per client.

```bash
RATE_LIMIT_REQUESTS_PER_MINUTE=60  # Default
```

### RATE_LIMIT_BURST_SIZE
**Optional**

Burst size for rate limiting.

```bash
RATE_LIMIT_BURST_SIZE=10  # Default
```

### ENABLE_AUTH
**Optional**

Enable authentication.

```bash
ENABLE_AUTH=true  # Default
```

### CORS_ORIGINS
**Optional**

Comma-separated list of allowed CORS origins.

```bash
CORS_ORIGINS=http://localhost:3000,https://example.com
```

---

## Server Configuration

### SLAUGHGPT_ENV
**Optional**

Environment mode.

```bash
SLAUGHGPT_ENV=development  # or production
```

### SLAUGHGPT_HOST
**Optional**

Server bind address.

```bash
SLAUGHGPT_HOST=0.0.0.0  # Default
```

### SLAUGHGPT_PORT
**Optional**

Server port.

```bash
SLAUGHGPT_PORT=8000  # Default
```

### WORKERS
**Optional**

Number of worker processes.

```bash
WORKERS=4  # Default
```

### REQUEST_TIMEOUT
**Optional**

Request timeout in seconds.

```bash
REQUEST_TIMEOUT=30  # Default
```

---

## Model Configuration

### MODEL_PATH
**Optional**

Path to model files.

```bash
MODEL_PATH=/app/models
```

### MODEL_CONTEXT_SIZE
**Optional**

Model context/window size.

```bash
MODEL_CONTEXT_SIZE=2048  # Default
```

### MODEL_BATCH_SIZE
**Optional**

Default batch size.

```bash
MODEL_BATCH_SIZE=8  # Default
```

### MODEL_MAX_TOKENS
**Optional**

Maximum tokens to generate.

```bash
MODEL_MAX_TOKENS=1024  # Default
```

---

## Performance

### MAX_CONCURRENT_REQUESTS
**Optional**

Maximum concurrent requests.

```bash
MAX_CONCURRENT_REQUESTS=100  # Default
```

### ENABLE_REQUEST_BATCHING
**Optional**

Enable request batching.

```bash
ENABLE_REQUEST_BATCHING=true  # Default
```

### BATCH_SIZE
**Optional**

Batch size for processing.

```bash
BATCH_SIZE=4  # Default
```

### BATCH_TIMEOUT
**Optional**

Batch timeout in seconds.

```bash
BATCH_TIMEOUT=0.1  # Default
```

---

## GPU Configuration

### CUDA_VISIBLE_DEVICES
**Optional**

CUDA device IDs to use.

```bash
CUDA_VISIBLE_DEVICES=0,1  # Use GPU 0 and 1
```

### GPU_ENABLED
**Optional**

Enable GPU support.

```bash
GPU_ENABLED=true  # Default
```

---

## Monitoring

### PROMETHEUS_ENABLED
**Optional**

Enable Prometheus metrics.

```bash
PROMETHEUS_ENABLED=true  # Default
```

### GRAFANA_ENABLED
**Optional**

Enable Grafana dashboards.

```bash
GRAFANA_ENABLED=true  # Default
```

---

## Logging

### LOG_LEVEL
**Optional**

Logging level.

```bash
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### LOG_FORMAT
**Optional**

Log format.

```bash
LOG_FORMAT=json  # or text
```

---

## Example .env File

```bash
# Security (REQUIRED in production)
SLAUGHGPT_API_KEY=generate-a-secure-32-char-key
SLAUGHGPT_JWT_SECRET=generate-a-secure-64-char-secret
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST_SIZE=10

# Server
SLAUGHGPT_ENV=production
SLAUGHGPT_HOST=0.0.0.0
SLAUGHGPT_PORT=8000
WORKERS=4

# Model
MODEL_PATH=/app/models
MODEL_CONTEXT_SIZE=2048
MODEL_BATCH_SIZE=8

# Performance
MAX_CONCURRENT_REQUESTS=100
ENABLE_REQUEST_BATCHING=true

# GPU
CUDA_VISIBLE_DEVICES=0

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```
