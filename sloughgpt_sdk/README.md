# SloughGPT Python SDK

A comprehensive Python client library for the SloughGPT API with support for text generation, chat completions, streaming, caching, and more.

## Features

- **Synchronous & Async Clients** - Both sync and async API clients
- **Streaming Support** - Real-time token streaming via HTTP or WebSocket
- **Response Caching** - In-memory and disk-based caching with TTL
- **Type Safety** - Full type hints and dataclass models
- **Error Handling** - Custom exceptions for different error types
- **CLI Tool** - Command-line interface for quick interactions
- **Retry Logic** - Automatic retries with exponential backoff

## Installation

```bash
pip install sloughgpt-sdk
```

For async support:
```bash
pip install sloughgpt-sdk[async]
```

For WebSocket streaming:
```bash
pip install sloughgpt-sdk[websocket]
```

For all features:
```bash
pip install sloughgpt-sdk[all]
```

## Quick Start

### Basic Usage

```python
from sloughgpt_sdk import SloughGPTClient, ChatMessage

# Create client
client = SloughGPTClient(base_url="http://localhost:8000")

# Check health
health = client.health()
print(f"API Status: {health.status}")

# Generate text
result = client.generate("Hello, how are you?")
print(result.generated_text)

# Chat completion
chat_result = client.chat([
    ChatMessage.user("Hello!"),
    ChatMessage.assistant("Hi! How can I help you?"),
    ChatMessage.user("Tell me a joke"),
])
print(chat_result.message.content)

# Batch generation
batch_result = client.batch_generate(["Hello", "Hi there", "Greetings"])
for r in batch_result.results:
    print(r.generated_text)

# Streaming
for token in client.generate_stream("Once upon a time"):
    print(token, end="", flush=True)
```

### Using Context Manager

```python
from sloughgpt_sdk import SloughGPTClient

with SloughGPTClient(base_url="http://localhost:8000") as client:
    result = client.generate("Hello!")
    print(result.generated_text)
```

## API Reference

### SloughGPTClient

#### Initialization

```python
from sloughgpt_sdk import SloughGPTClient

client = SloughGPTClient(
    base_url="http://localhost:8000",  # API base URL
    api_key="your-api-key",            # Optional API key
    timeout=30,                        # Request timeout
    verify_ssl=True,                   # SSL verification
    headers={"Custom-Header": "value"} # Custom headers
)
```

#### Health & Status

```python
# Check API health
health = client.health()
print(f"Status: {health.status}")
print(f"Version: {health.version}")
print(f"Model Loaded: {health.model_loaded}")

# Liveness probe
liveness = client.liveness()

# Readiness probe
readiness = client.readiness()

# System information
info = client.info()
print(f"PyTorch: {info.pytorch_version}")
print(f"CUDA Available: {info.cuda_available}")
```

#### Text Generation

```python
# Basic generation
result = client.generate(
    prompt="The capital of France is",
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
print(result.generated_text)

# Streaming
for token in client.generate_stream("Once upon a time"):
    print(token, end="", flush=True)
print()

# With options
result = client.generate(
    prompt="Write a haiku",
    max_new_tokens=100,
    temperature=0.7,
    top_k=40,
    top_p=0.95,
    repetition_penalty=1.1,
    num_beams=1,
    early_stopping=True
)
```

#### Chat Completions

```python
from sloughgpt_sdk import ChatMessage

# Single message
result = client.chat([ChatMessage.user("Hello!")])

# Multi-turn conversation
messages = [
    ChatMessage.system("You are a helpful assistant."),
    ChatMessage.user("What is Python?"),
]
result = client.chat(messages)
print(result.message.content)

# With options
result = client.chat(
    messages=[ChatMessage.user("Explain AI")],
    model="gpt2",
    temperature=0.7,
    max_new_tokens=200
)
```

#### Batch Processing

```python
prompts = [
    "Hello, how are you?",
    "What is machine learning?",
    "Tell me a fact about space."
]

batch_result = client.batch_generate(prompts, max_new_tokens=50)
print(f"Processed: {batch_result.total_prompts}")
print(f"Successful: {batch_result.successful}")

for r in batch_result.results:
    print(f"Prompt: {r.prompt}")
    print(f"Response: {r.generated_text}")
```

#### Model Management

```python
# List models
models = client.list_models()
for model in models:
    print(f"{model.id}: {model.source or 'unknown'}")

# Get specific model
model = client.get_model("gpt2")
print(f"Name: {model.name}")
print(f"Size: {model.size_mb} MB")

# Load model
client.load_model("gpt2-medium")

# List HuggingFace models
hf_models = client.list_hf_models(query="gpt", limit=10)
```

#### Dataset Management

```python
# List datasets
datasets = client.list_datasets()
for ds in datasets:
    print(f"{ds.id}: {ds.description}")

# Get specific dataset
dataset = client.get_dataset("openwebtext")
```

#### Metrics & Monitoring

```python
# Get metrics
metrics = client.metrics()
print(f"Total Requests: {metrics.requests_total}")
print(f"Cache Hits: {metrics.cache_hits}")
print(f"Avg Response Time: {metrics.avg_response_time_ms}ms")

# Prometheus format
prometheus_metrics = client.metrics_prometheus()
```

#### Authentication

```python
# Login
auth_result = client.login(username="admin", password="password")
access_token = auth_result.get("access_token")

# Refresh token
new_token = client.refresh_token()
```

## WebSocket Streaming

```python
from sloughgpt_sdk.websocket import WebSocketClient

def on_token(message):
    print(message.data, end="", flush=True)

def on_complete(message):
    print("\nGeneration complete!")

ws = WebSocketClient("http://localhost:8000")
ws.connect()
ws.on("token", on_token)
ws.on("complete", on_complete)
ws.send_generate("Hello, how are you?")
ws.close()
```

## Caching

```python
from sloughgpt_sdk.cache import InMemoryCache, DiskCache, cached

# In-memory cache
cache = InMemoryCache(ttl=3600)  # 1 hour TTL
cache.set("key", "value")
value = cache.get("key")
print(cache.stats)

# Disk cache
disk_cache = DiskCache(cache_dir="./cache", ttl=86400)

# Decorator caching
@cached(ttl=3600, cache=cache)
def expensive_operation(param):
    return do_work(param)

# Clear cache
cache.clear()
```

## Error Handling

```python
from sloughgpt_sdk import SloughGPTClient
from sloughgpt_sdk.exceptions import APIError, RateLimitError, ValidationError

client = SloughGPTClient()

try:
    result = client.generate("Hello")
except APIError as e:
    print(f"API Error: {e.message}")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except ValidationError as e:
    print(f"Validation failed: {e.message}")
```

## CLI Usage

```bash
# Check health
sloughgpt-cli health

# Generate text
sloughgpt-cli generate "Hello, how are you?"

# Stream generation
sloughgpt-cli generate "Once upon a time" --stream

# Chat
sloughgpt-cli chat "What is Python?"

# List models
sloughgpt-cli models

# Get metrics
sloughgpt-cli metrics

# JSON output
sloughgpt-cli health --json

# Custom API URL
sloughgpt-cli --url http://custom:8000 health
```

## Async Usage

```python
import asyncio
from sloughgpt_sdk import AsyncSloughGPTClient

async def main():
    async with AsyncSloughGPTClient() as client:
        result = await client.generate("Hello!")
        print(result.generated_text)
        
        # List models
        models = await client.list_models()
        for model in models:
            print(model.id)

asyncio.run(main())
```

## Models

All API response models are available:

- `GenerateRequest` - Text generation request
- `GenerationResult` - Text generation result
- `ChatMessage` - Chat message (user/assistant/system)
- `ChatRequest` - Chat completion request
- `ChatResult` - Chat completion result
- `BatchRequest` - Batch generation request
- `BatchResult` - Batch generation result
- `ModelInfo` - Model information
- `DatasetInfo` - Dataset information
- `HealthStatus` - Health check status
- `SystemInfo` - System information
- `MetricsData` - API metrics

## Configuration

Environment variables:

- `SLOUGHGPT_API_URL` - API base URL
- `SLOUGHGPT_API_KEY` - API key for authentication

## License

MIT License
