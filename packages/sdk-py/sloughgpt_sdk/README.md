# SloughGPT Python SDK

A comprehensive Python client library for the SloughGPT API. Build AI-powered applications with full SaaS infrastructure out of the box.

## Features

| Category | Features |
|----------|----------|
| **Core** | Sync/Async HTTP client, streaming, batch processing |
| **Auth** | API keys, tiers, quotas, rotation |
| **SaaS** | Webhooks, billing, subscriptions, usage analytics |
| **Tools** | Caching, benchmarks, profiling, model registry |
| **CLI** | Full command-line interface |

## Installation

```bash
pip install sloughgpt-sdk
```

For all features:
```bash
pip install sloughgpt-sdk[all]
```

## Development (monorepo)

When developing inside the **sloughGPT** repository, install from the **repo root** (`pip install -e ".[dev]"`) and run **`python3 -m pytest tests/test_sdk.py`** (same checks as CI job **`sdk-test-py`** in **`.github/workflows/ci_cd.yml`**).

## Quick Start

```python
from sloughgpt_sdk import SloughGPTClient, ChatMessage

client = SloughGPTClient(base_url="http://localhost:8000")

# Text generation
result = client.generate("Hello!")
print(result.generated_text)

# Chat
result = client.chat([ChatMessage.user("Hi!")])
print(result.message.content)
```

## API Key Management

```python
from sloughgpt_sdk import APIKeyManager, KeyTier

manager = APIKeyManager()

# Create key
key, data = manager.create_key(
    name="My App",
    tier=KeyTier.PRO,
    quota_daily=10000
)

# Validate key
is_valid, reason, key_data = manager.validate_key(key)

# Record usage
manager.record_usage(key, requests_count=1)

# Rotate key
new_key, new_data = manager.rotate_key(data.key_id)
```

## Simple Tracking

```python
# Track metrics during training
with client.track("training-v1") as t:
    for epoch in range(10):
        acc = train()
        t.log("accuracy", acc)
        t.next_step()
```

## Model Registry

```python
from sloughgpt_sdk import ModelRegistry, ModelStatus

registry = ModelRegistry()

# Register model
registry.register(
    id="gpt2-large",
    name="GPT-2 Large",
    version="1.0",
    path="/models/gpt2-large",
    tags=["stable", "gpu"]
)

# Record metrics
registry.record_request("gpt2-large", latency_ms=50, tokens=100)

# Get best model
best = registry.get_best_model("latency")
```

## Caching

```python
from sloughgpt_sdk import InMemoryCache

cache = InMemoryCache(ttl=3600)  # 1 hour
cache.set("key", "value")
value = cache.get("key")
```

## Webhooks

```python
from sloughgpt_sdk import WebhookManager, WebhookEvent

wh = WebhookManager()
wh.create_webhook(
    url="https://myapp.com/webhook",
    events=[WebhookEvent.KEY_CREATED, WebhookEvent.QUOTA_EXCEEDED]
)
```

## Billing

```python
from sloughgpt_sdk import BillingManager, BillingCycle

billing = BillingManager()

# Create customer & subscription
customer = billing.create_customer("user@example.com", "User")
subscription = billing.create_subscription(
    customer["id"],
    plan_id="pro",
    billing_cycle=BillingCycle.MONTHLY
)
```

## Usage Dashboard

```python
from sloughgpt_sdk import UsageDashboard

dashboard = UsageDashboard()

# Record request
dashboard.record_request("sk_xxx", "cus_xxx", tokens=100)

# Get metrics
metrics = dashboard.get_metrics("7d")
report = dashboard.generate_report("30d")
```

## Benchmarks

```python
from sloughgpt_sdk import Benchmark

bench = Benchmark()
result = bench.run("My operation", lambda: do_work(), iterations=1000)
print(f"{result.ops_per_second} ops/sec")
```

## CLI Tool

```bash
# Generate text
sloughgpt-cli generate "Hello world"

# Chat
sloughgpt-cli chat "What is Python?"

# API Keys
sloughgpt-cli key create --name "My App" --tier pro
sloughgpt-cli key list

# Registry
sloughgpt-cli registry register --id gpt2 --name "GPT-2" --version 1.0 --path /models/gpt2
sloughgpt-cli registry list
sloughgpt-cli registry best --criteria latency

# Metrics
sloughgpt-cli metrics
```

## All SDK Modules

```python
from sloughgpt_sdk import (
    # Core
    SloughGPTClient,
    AsyncSloughGPTClient,
    
    # Models
    GenerateRequest, GenerationResult,
    ChatMessage, ChatRequest, ChatResult,
    BatchRequest, BatchResult,
    ModelInfo, DatasetInfo,
    HealthStatus, SystemInfo, MetricsData,
    
    # Auth
    APIKeyManager, APIKey, KeyTier,
    
    # Webhooks
    WebhookManager, Webhook, WebhookEvent,
    
    # Billing
    BillingManager, Plan, Subscription, Invoice,
    
    # Dashboard
    UsageDashboard, DashboardMetrics,
    
    # Registry
    ModelRegistry, ModelSelector, ModelStatus, ModelTag,
)
```

## Subscription Tiers

| Tier | Rate Limit | Daily | Monthly |
|------|-----------|-------|---------|
| Free | 60/min | 100 | 1,000 |
| Starter | 120/min | 1,000 | 10,000 |
| Pro | 300/min | 10,000 | 100,000 |
| Enterprise | 1000/min | 100,000 | 1,000,000 |

## Error Handling

```python
from sloughgpt_sdk import SloughGPTClient
from sloughgpt_sdk.exceptions import APIError, RateLimitError

try:
    result = client.generate("Hello")
except RateLimitError:
    print("Rate limited")
except APIError as e:
    print(f"API error: {e.message}")
```

## License

MIT License
