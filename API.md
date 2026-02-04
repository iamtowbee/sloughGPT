# SloughGPT API Documentation

## Overview

SloughGPT provides a comprehensive REST API for training, deploying, and managing AI models with enterprise-grade features including authentication, cost tracking, and real-time monitoring.

**Base URL**: `https://api.sloughgpt.com`
**API Version**: v1
**Documentation Version**: 1.0.0

## Authentication

SloughGPT uses JWT-based authentication with API key support for programmatic access.

### Bearer Token Authentication

```http
Authorization: Bearer <your_jwt_token>
```

### API Key Authentication

```http
X-API-Key: <your_api_key>
```

### Authentication Endpoints

#### Register User
```http
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password",
  "name": "John Doe"
}
```

#### Login
```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password"
}
```

#### Refresh Token
```http
POST /auth/refresh
Authorization: Bearer <refresh_token>
```

#### Create API Key
```http
POST /auth/api-keys
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "name": "Production API Key",
  "permissions": ["read", "write", "delete"],
  "expires_days": 365
}
```

## Core API Endpoints

### Model Management

#### List Models
```http
GET /models
Authorization: Bearer <token>

# Response
{
  "models": [
    {
      "id": "model_123",
      "name": "gpt2-medium",
      "type": "language_model",
      "status": "active",
      "created_at": "2024-01-15T10:00:00Z",
      "size": "1.5GB",
      "accuracy": 0.89
    }
  ],
  "total": 1,
  "page": 1,
  "per_page": 20
}
```

#### Create Model
```http
POST /models
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "custom-gpt2",
  "type": "language_model",
  "config": {
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "vocab_size": 50257
  },
  "description": "Custom trained model"
}
```

#### Get Model Details
```http
GET /models/{model_id}
Authorization: Bearer <token>

# Response
{
  "id": "model_123",
  "name": "gpt2-medium",
  "type": "language_model",
  "status": "active",
  "config": {
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "num_hidden_layers": 24
  },
  "metrics": {
    "accuracy": 0.89,
    "inference_time_ms": 45,
    "memory_usage_mb": 2048
  },
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-20T15:30:00Z"
}
```

#### Delete Model
```http
DELETE /models/{model_id}
Authorization: Bearer <token>
```

### Text Generation

#### Generate Text
```http
POST /generate
Authorization: Bearer <token>
Content-Type: application/json

{
  "model_id": "model_123",
  "prompt": "Explain artificial intelligence",
  "max_tokens": 150,
  "temperature": 0.7,
  "top_p": 0.9,
  "stop_sequences": ["\n", "###"],
  "stream": false
}
```

#### Streaming Generation
```http
POST /generate
Authorization: Bearer <token>
Content-Type: application/json

{
  "model_id": "model_123",
  "prompt": "Write a story about AI",
  "max_tokens": 500,
  "temperature": 0.8,
  "stream": true
}
```

#### Batch Generation
```http
POST /generate/batch
Authorization: Bearer <token>
Content-Type: application/json

{
  "model_id": "model_123",
  "prompts": [
    "What is machine learning?",
    "Explain neural networks",
    "How does AI work?"
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

### Model Training

#### Start Training Job
```http
POST /training/jobs
Authorization: Bearer <token>
Content-Type: application/json

{
  "model_id": "model_123",
  "dataset_path": "s3://my-dataset/training-data.json",
  "config": {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "warmup_steps": 1000,
    "save_steps": 500
  },
  "resources": {
    "gpu_type": "V100",
    "gpu_count": 2,
    "cpu_cores": 8,
    "memory_gb": 32
  }
}
```

#### Get Training Status
```http
GET /training/jobs/{job_id}
Authorization: Bearer <token>

# Response
{
  "id": "job_456",
  "status": "training",
  "progress": 0.45,
  "current_epoch": 4,
  "total_epochs": 10,
  "metrics": {
    "loss": 0.234,
    "accuracy": 0.89,
    "perplexity": 15.2
  },
  "resources_used": {
    "gpu_hours": 12.5,
    "cost_usd": 45.67
  },
  "estimated_completion": "2024-02-01T14:30:00Z"
}
```

#### Stop Training Job
```http
POST /training/jobs/{job_id}/stop
Authorization: Bearer <token>
```

### Fine-tuning

#### Create Fine-tuning Job
```http
POST /fine-tuning/jobs
Authorization: Bearer <token>
Content-Type: application/json

{
  "base_model_id": "model_123",
  "training_data": [
    {"prompt": "Question", "completion": "Answer"},
    {"prompt": "Input", "completion": "Output"}
  ],
  "config": {
    "batch_size": 8,
    "learning_rate": 5e-5,
    "num_epochs": 3
  },
  "hyperparameters": {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1
  }
}
```

### Embeddings

#### Generate Embeddings
```http
POST /embeddings
Authorization: Bearer <token>
Content-Type: application/json

{
  "model_id": "embedding_model_789",
  "texts": [
    "Hello world",
    "Artificial intelligence",
    "Machine learning"
  ],
  "dimension": 768
}

# Response
{
  "embeddings": [
    [0.1234, -0.5678, 0.9012, ...],
    [-0.2345, 0.6789, -0.1234, ...],
    [0.3456, -0.7890, 0.2345, ...]
  ],
  "model_id": "embedding_model_789",
  "dimension": 768
}
```

### Vector Search

#### Search Similar Vectors
```http
POST /search/vector
Authorization: Bearer <token>
Content-Type: application/json

{
  "query_vector": [0.1234, -0.5678, 0.9012, ...],
  "top_k": 10,
  "similarity_threshold": 0.7,
  "filters": {
    "category": "technology",
    "date_range": "2024-01-01:2024-02-01"
  }
}
```

### Cost Management

#### Get Cost Statistics
```http
GET /cost/stats
Authorization: Bearer <token>
?start_date=2024-01-01&end_date=2024-01-31&granularity=daily

# Response
{
  "total_cost": 1234.56,
  "breakdown": {
    "inference": 890.12,
    "training": 234.45,
    "storage": 45.67,
    "network": 64.32
  },
  "daily_stats": [
    {
      "date": "2024-01-01",
      "cost": 45.67,
      "requests": 1234,
      "tokens": 123456
    }
  ],
  "budget_status": {
    "monthly_budget": 5000.00,
    "spent": 1234.56,
    "remaining": 3765.44,
    "percentage_used": 24.69
  }
}
```

#### Set Budget
```http
POST /cost/budget
Authorization: Bearer <token>
Content-Type: application/json

{
  "monthly_budget": 10000.00,
  "alert_threshold": 0.8,
  "alert_emails": ["admin@company.com"]
}
```

### Usage Analytics

#### Get Usage Metrics
```http
GET /analytics/usage
Authorization: Bearer <token>
?start_date=2024-01-01&end_date=2024-01-31&group_by=model

# Response
{
  "total_requests": 50000,
  "total_tokens": 5000000,
  "average_response_time_ms": 85,
  "success_rate": 0.995,
  "breakdown": [
    {
      "model_id": "model_123",
      "requests": 30000,
      "tokens": 3000000,
      "avg_response_time": 75,
      "cost": 789.45
    }
  ]
}
```

## Real-time Features

### WebSocket Connection

```javascript
// Connect to real-time updates
const ws = new WebSocket('wss://api.sloughgpt.com/ws');

// Authenticate
ws.send(JSON.stringify({
  type: 'auth',
  token: 'your_jwt_token'
}));

// Subscribe to events
ws.send(JSON.stringify({
  type: 'subscribe',
  channels: ['training', 'cost', 'usage']
}));

// Receive updates
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};
```

### Event Types

- `training_started` - Training job started
- `training_completed` - Training job completed
- `cost_alert` - Cost threshold exceeded
- `model_deployed` - Model successfully deployed
- `system_maintenance` - System maintenance notification

## Error Handling

### Standard Error Format

```json
{
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "Parameter 'temperature' must be between 0 and 2",
    "details": {
      "parameter": "temperature",
      "value": 3.0,
      "allowed_range": [0.0, 2.0]
    },
    "request_id": "req_123456789"
  }
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_PARAMETER` | Invalid request parameter | 400 |
| `UNAUTHORIZED` | Authentication failed | 401 |
| `FORBIDDEN` | Insufficient permissions | 403 |
| `NOT_FOUND` | Resource not found | 404 |
| `RATE_LIMITED` | Rate limit exceeded | 429 |
| `MODEL_BUSY` | Model is currently busy | 503 |
| `INSUFFICIENT_FUNDS` | Insufficient budget | 402 |
| `QUOTA_EXCEEDED` | Usage quota exceeded | 429 |
| `INTERNAL_ERROR` | Internal server error | 500 |

## Rate Limiting

### Rate Limits

- **Free Tier**: 100 requests/minute, 10,000 tokens/day
- **Pro Tier**: 1,000 requests/minute, 100,000 tokens/day  
- **Enterprise**: 10,000 requests/minute, 1,000,000 tokens/day

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
X-RateLimit-Retry-After: 60
```

## SDK Examples

### Python SDK

```python
from sloughgpt import SloughGPTClient

# Initialize client
client = SloughGPTClient(
    api_key="your_api_key",
    base_url="https://api.sloughgpt.com"
)

# Generate text
response = client.generate(
    model_id="gpt2-medium",
    prompt="Explain quantum computing",
    max_tokens=150,
    temperature=0.7
)

print(response.text)

# Train a model
job = client.training.create_job(
    model_id="custom-model",
    dataset_path="data/training.json",
    config={"epochs": 5, "batch_size": 16}
)

# Monitor training
status = client.training.get_status(job.id)
print(f"Progress: {status.progress:.1%}")
```

### JavaScript SDK

```javascript
import { SloughGPTClient } from '@sloughgpt/client';

const client = new SloughGPTClient({
  apiKey: 'your_api_key',
  baseURL: 'https://api.sloughgpt.com'
});

// Generate text
const response = await client.generate({
  modelId: 'gpt2-medium',
  prompt: 'Write a poem about AI',
  maxTokens: 100,
  temperature: 0.8
});

console.log(response.text);

// Stream generation
const stream = await client.generate({
  modelId: 'gpt2-medium',
  prompt: 'Tell me a story',
  stream: true
});

for await (const chunk of stream) {
  process.stdout.write(chunk.text);
}
```

## API Versioning

### Version Strategy

- URL Path Versioning: `/v1/generate`, `/v2/generate`
- Backward compatibility maintained for 12 months
- Deprecation warnings sent 6 months before removal

### Version Headers

```http
API-Version: 2024-01
Accept: application/vnd.sloughgpt.v1+json
```

## Testing

### Test Environment

- **Base URL**: `https://api-test.sloughgpt.com`
- **Authentication**: Use test API keys
- **Rate Limits**: Relaxed limits for testing
- **Data**: Isolated from production

### Testing Examples

```bash
# Health check
curl https://api-test.sloughgpt.com/health

# Generate text with test API key
curl -X POST https://api-test.sloughgpt.com/generate \
  -H "X-API-Key: test_api_key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

## Support

### Documentation

- [Full API Reference](https://docs.sloughgpt.com/api)
- [SDK Documentation](https://docs.sloughgpt.com/sdks)
- [Code Examples](https://github.com/sloughgpt/examples)

### Contact

- **Email**: api-support@sloughgpt.ai
- **Discord**: [SloughGPT Community](https://discord.gg/sloughgpt)
- **Issues**: [GitHub Issues](https://github.com/sloughgpt/sloughgpt/issues)

---

**🚀 SloughGPT API - Enterprise AI Made Simple**

Built with ❤️ by the SloughGPT Team