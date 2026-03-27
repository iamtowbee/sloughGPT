# SloughGPT

Self-hosted LLM infrastructure with local model training, inference, and experimentation.

## Features

- **Local Model Training** - Train SloughGPTModel from scratch or fine-tune HuggingFace models
- **Soul Engine** - Every model IS a soul (.sou format) with personality and cognitive capabilities
- **Production Inference** - High-performance inference engine with streaming
- **GGUF Export** - Mobile deployment via llama.rn (iOS/Android)
- **ONNX Export** - Cross-platform deployment (server, web, mobile)
- **Quantization** - FP16, INT8, INT4, Q4_K_M, Q5_K_M support
- **Experiment Tracking** - MLflow-style metrics and parameter logging
- **Federated Learning** - Privacy-preserving distributed training
- **Benchmarking** - Performance metrics and model comparison
- **Optimizations** - Mixed precision, gradient checkpointing, torch.compile, Flash Attention
- **API Security** - JWT auth, rate limiting, input validation, audit logging
- **Batch Processing** - Process up to 50 prompts in one request
- **Response Caching** - TTL-based caching with hit/miss stats
- **Kubernetes Ready** - Helm charts, health probes, Prometheus metrics
- **Docker Ready** - Docker Compose with API, GPU, monitoring stacks
- **TypeScript SDK** - Full-featured SDK with webhooks, billing, caching

## Quick Start

### CLI Commands
```bash
# Quick train + generate (auto-optimized)
python3 cli.py quick --steps 100 --prompt "Hello world"

# Benchmark inference
python3 cli.py benchmark -m gpt2 -d mps

# Check optimizations
python3 cli.py optimize

# System info
python3 cli.py system

# Generate text
python3 cli.py generate "Hello world"

# Interactive chat (starts the API with uvicorn if it is not already running)
python3 cli.py chat
```

### Start Server
```bash
# CPU mode (default for Intel Macs)
python3 apps/api/server/main.py

# With uvicorn
python3 -m uvicorn apps.api.server.main:app --port 8000
```

### Docker
```bash
# Start with Docker
./docker-manage.sh start

# Development mode
./docker-manage.sh dev

# GPU mode
./docker-manage.sh gpu
```

## GPU Support

| Hardware | Support | Speed |
|----------|---------|-------|
| NVIDIA GPU (CUDA) | ✅ Full | Fast |
| Apple Silicon (MPS) | ✅ Full | Good |
| AMD GPU (ROCm) | ✅ Linux only | Good |
| Intel Mac + AMD | ❌ CPU only | Slow |

## Industry Standard Optimizations

```python
from domains.training.optimized_trainer import Presets

# Auto-detect best settings
config = Presets.auto()

# Or specific hardware
config = Presets.high_end_gpu()   # A100, H100, RTX 4090
config = Presets.apple_silicon()  # M1/M2/M3
config = Presets.cpu_only()        # CPU training
```

Optimizations:
- Mixed Precision (FP16/BF16) - 2-3x speedup
- Gradient Checkpointing - 50% memory savings
- torch.compile - 1.5-2x speedup
- Flash Attention (NVIDIA/AMD) - 2-4x speedup

## API Endpoints

### Health & Status
```bash
# Basic health
curl http://localhost:8000/health

# Liveness probe (Kubernetes)
curl http://localhost:8000/health/live

# Readiness probe (Kubernetes)
curl http://localhost:8000/health/ready

# Detailed health
curl http://localhost:8000/health/detailed
```

### Authentication
```bash
# Create JWT token from API key
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your-api-key"}'

# Verify token
curl -X POST http://localhost:8000/auth/verify \
  -H "Authorization: Bearer <token>"

# Refresh token
curl -X POST http://localhost:8000/auth/refresh \
  -H "Authorization: Bearer <token>"
```

### Rate Limiting
```bash
# Check rate limit status
curl http://localhost:8000/rate-limit/status

# Check your current usage
curl http://localhost:8000/rate-limit/check
```

### Caching
```bash
# Cache statistics
curl http://localhost:8000/cache/stats

# Clear cache
curl -X DELETE http://localhost:8000/cache
```

### Metrics
```bash
# JSON metrics
curl http://localhost:8000/metrics

# Prometheus format
curl http://localhost:8000/metrics/prometheus

# Security audit logs
curl http://localhost:8000/security/audit
```

### Inference
```bash
# Generate text
curl -X POST http://localhost:8000/inference/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_new_tokens": 50}'

# Streaming (SSE)
curl -X POST http://localhost:8000/inference/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_new_tokens": 100}'

# Batch processing (up to 50 prompts)
curl -X POST http://localhost:8000/inference/batch \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hello", "Hi there", "Good morning"], "max_new_tokens": 50}'

# Inference stats
curl http://localhost:8000/inference/stats
```

### Training
```bash
# Start training
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"dataset": "shakespeare", "epochs": 5, "batch_size": 32}'

# List training jobs
curl http://localhost:8000/training/jobs
```

### Experiments
```bash
# Create experiment
curl -X POST "http://localhost:8000/experiments?name=test&description=Testing"

# Log metrics
curl -X POST "http://localhost:8000/experiments/{id}/log_metric?metric_name=loss&value=2.5&step=0"
```

### Benchmarking
```bash
# Run benchmark
curl -X POST http://localhost:8000/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_new_tokens": 20}'

# Compare quantization levels
curl http://localhost:8000/benchmark/compare
```

### Model Export
```bash
# Export model (SafeTensors, Torch, ONNX, GGUF, .sou)
curl -X POST http://localhost:8000/model/export \
  -H "Content-Type: application/json" \
  -d '{"output_path": "models/exported", "format": "safetensors"}'

# List formats
curl http://localhost:8000/model/export/formats

# CLI export examples
python3 cli.py export models/slough.pt --format safetensors
python3 cli.py export models/slough.pt --format gguf_q4_k_m  # Mobile
python3 cli.py export models/slough.pt --format onnx         # Cross-platform
```

### Soul Engine
```bash
# Load a soul (.sou model)
curl -X POST http://localhost:8000/load-soul \
  -H "Content-Type: application/json" \
  -d '{"soul_path": "models/slough.sou"}'

# Get current soul profile
curl http://localhost:8000/soul

# Generate with soul (uses personality + reasoning)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "use_soul": true}'
```

## Architecture

```
SloughGPT/
├── domains/
│   ├── core/               # Soul Engine & cognitive modules
│   │   ├── soul.py         # SoulEngine - THE core model wrapper
│   │   ├── reasoning.py    # Cognitive reasoning engine
│   │   └── cognitive/      # LearningOptimizer, CreativityEngine
│   ├── models/             # Model interface & SloughGPTModel
│   │   ├── __init__.py    # OUR OWN architecture (RoPE, SwiGLU, RMSNorm)
│   │   └── external.py    # External model support (HF, Ollama, GGUF)
│   ├── inference/          # Inference engine
│   │   ├── engine.py       # Production inference
│   │   ├── quantization.py  # FP16/INT8/INT4, Q4/Q5/Q8
│   │   ├── optimizations.py # KV cache, batching
│   │   └── sou_format.py   # .sou model format
│   ├── training/           # Training infrastructure
│   │   ├── train_pipeline.py
│   │   ├── optimized_trainer.py  # Optimized training
│   │   ├── onnx_export.py       # ONNX export
│   │   ├── gguf_export.py       # GGUF export (15+ architectures)
│   │   ├── huggingface/    # HF model support
│   │   └── lora.py         # LoRA/QLoRA fine-tuning
│   └── ml_infrastructure/
│       ├── experiment_tracker.py
│       └── benchmarking.py
├── server/
│   └── main.py              # FastAPI server (100+ endpoints)
├── sloughgpt_sdk/           # Python SDK
├── typescript-sdk/          # TypeScript SDK
├── cli.py                    # CLI commands
├── setup.sh                  # Setup script
├── docker-compose.yml        # Docker deployment
├── k8s/                      # Kubernetes manifests
├── helm/                     # Helm charts
└── tests/                   # Unit tests
```

## Supported Models

| Model | Size | Context | Recommended |
|-------|------|---------|-------------|
| GPT-2 | 124M | 1K | FP16 |
| GPT-2 Medium | 355M | 1K | FP16 |
| GPT-2 Large | 774M | 1K | FP16 |
| Phi-2 | 2.7B | 2K | Q4_K |
| Mistral 7B | 7.3B | 32K | Q4_K |
| LLaMA-2 7B | 7B | 4K | Q4_K |
| Qwen2 1.5B | 1.5B | 32K | Q4_K |
| Gemma 2B | 2B | 8K | Q4_K |

## Installation

```bash
pip install torch transformers fastapi uvicorn
```

## Development

```bash
# Run server
python3 apps/api/server/main.py

# Train a model
python3 packages/core-py/domains/training/train_pipeline.py --data datasets/shakespeare/input.txt --epochs 5
```

## License

MIT
