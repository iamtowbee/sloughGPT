# SloughGPT Quick Start Guide

## Get Started in 5 Minutes

### 1. Install
```bash
git clone https://github.com/iamtowbee/sloughGPT.git
cd sloughGPT
pip install torch transformers fastapi uvicorn pydantic pytest
```

### 2. Quick Training (CLI)
```bash
python3 cli.py quick --steps 100 --prompt "Hello world"
```

### 3. Start API Server
```bash
python3 apps/api/server/main.py
# Access at http://localhost:8000/docs
```

---

## CLI Commands

### Training
```bash
# Quick train + generate (auto-optimized)
python3 cli.py quick --steps 100 --prompt "The future is"

# Custom model config
python3 cli.py quick --epochs 3 --batch 64 --embed 256 --layers 6

# CPU only (no optimizations)
python3 cli.py quick --no-optimize
```

### Inference
```bash
# Generate text
python3 cli.py generate "Hello world" --max-tokens 100

# Interactive chat (auto-start API if needed)
python3 cli.py chat

# Interactive chat with model preload (recommended)
python3 cli.py chat --auto-model gpt2

# HuggingFace model
python3 cli.py hf-serve gpt2

# Download model
python3 cli.py hf-download gpt2
```

### Benchmarking
```bash
# Benchmark inference
python3 cli.py benchmark -m gpt2 -d mps -t latency

# Full benchmark suite
python3 cli.py benchmark -m gpt2 -d mps -t all

# Check GPU optimizations
python3 cli.py optimize
```

### Model Export
```bash
# Export to different formats
python3 cli.py export gpt2 --format onnx
python3 cli.py export gpt2 --format safetensors

# With quantization
python3 cli.py export gpt2 --quantize int8
```

### System
```bash
# System info
python3 cli.py system

# Health check
python3 cli.py health

# Docker management
python3 cli.py docker status
python3 cli.py docker logs

# Environment check
python3 cli.py config check

# Configuration validation
python3 cli.py config validate

# Generate secrets
python3 cli.py config generate --type all

# Statistics
python3 cli.py stats

# List models
python3 cli.py models

# List datasets
python3 cli.py datasets
```

### API Management
```bash
# Check API status
python3 cli.py api-status

# Test API endpoints
python3 cli.py api-test

# Test authentication
python3 cli.py api-auth

# Compare models
python3 cli.py compare
```

---

## API Endpoints

### Health Check
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
# Create JWT token
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

### Batch Processing
```bash
# Batch generation (up to 50 prompts)
curl -X POST http://localhost:8000/inference/batch \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hello", "Hi"], "max_new_tokens": 20}'
```

### Generate Text
```bash
curl -X POST http://localhost:8000/inference/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_new_tokens": 50}'
```

### Streaming Generation
```bash
curl -X POST http://localhost:8000/inference/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_new_tokens": 100}'
```

### Training
```bash
curl -X POST http://localhost:8000/train \
  -d "dataset=shakespeare&epochs=5&batch_size=32"
```

### Benchmarking
```bash
curl -X POST http://localhost:8000/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt2", "num_runs": 10}'
```

---

## GPU Support

| Hardware | Speed | Command |
|----------|-------|---------|
| NVIDIA GPU | Fast | `--device cuda` |
| Apple Silicon (M1/M2/M3) | Good | `--device mps` |
| AMD GPU (Linux + ROCm) | Good | `--device cuda` |
| Intel Mac AMD GPU | ❌ | Use CPU |
| CPU | Slow | `--device cpu` |

### Verify GPU
```bash
python3 cli.py optimize
```

---

## Docker Deployment

```bash
# Start API server
docker compose up -d api

# Development mode
docker compose --profile dev up -d

# GPU mode (NVIDIA)
docker compose --profile gpu up -d

# Stop
docker compose down
```

## Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace sloughgpt

# Apply manifests
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -n sloughgpt

# View logs
kubectl logs -n sloughgpt -l app=sloughgpt-api

# Helm chart
helm install sloughgpt ./helm/sloughgpt -n sloughgpt
```

---

## Optimization Presets

```python
from domains.training.optimized_trainer import Presets

# Auto-detect best settings
config = Presets.auto()

# Specific hardware
Presets.high_end_gpu()   # A100, H100, RTX 4090
Presets.mid_range_gpu()  # RTX 3060, V100
Presets.apple_silicon()  # M1/M2/M3
Presets.cpu_only()       # CPU training
```

### Speedup Estimates

| Optimization | Speedup | Memory |
|-------------|---------|--------|
| FP16 | 2-3x | -50% |
| torch.compile | 1.5-2x | +10% |
| Flash Attention | 2-4x | -20% |
| **Combined** | **3-6x** | **-60%** |

---

## Troubleshooting

### macOS PyTorch hangs?
```bash
# Add to ~/.zshrc or ~/.bashrc
export DYLD_INSERT_LIBRARIES=""
```

### Docker not running?
```bash
open -a Docker
```

### Out of memory?
```bash
# Smaller batch size
python3 cli.py quick --batch 8

# Or use quantization
python3 cli.py export model --quantize int8
```

---

## File Structure

```
SloughGPT/
├── domains/
│   ├── inference/          # Inference engine
│   │   ├── engine.py       # Production inference
│   │   ├── quantization.py  # FP16/INT8/INT4
│   │   ├── optimizations.py # KV cache, batching
│   │   └── sou_format.py   # .sou model format
│   ├── training/            # Training
│   │   ├── train_pipeline.py
│   │   ├── optimized_trainer.py  # Optimized training
│   │   ├── models/nanogpt.py
│   │   ├── huggingface/     # HuggingFace integration
│   │   └── lora.py         # LoRA fine-tuning
│   └── ml_infrastructure/  # Infrastructure
│       ├── benchmarking.py
│       └── experiment_tracker.py
├── apps/api/server/main.py  # FastAPI server
├── cli.py                    # CLI commands
├── infra/k8s/               # Kubernetes, Helm, Grafana assets
├── tests/                   # Unit tests (100+ tests)
├── datasets/                # Training data
├── infra/docker/docker-compose.yml  # Docker deployment
└── sloughgpt_colab.ipynb   # Colab notebook
```

---

## Next Steps

1. **Run the notebook**: `jupyter notebook sloughgpt_colab.ipynb` (in Colab, try `python3 cli.py chat --auto-model gpt2` after setup)
2. **Try different datasets**: Shakespeare, TinyStories, custom
3. **Explore model architecture**: Section 5 in the notebook
4. **Deploy with Docker**: See Docker section above
5. **Read the docs**: `README.md`, `API.md`, `TODO.md`

---

## Links

- **GitHub**: https://github.com/iamtowbee/sloughGPT
- **API Docs**: http://localhost:8000/docs
- **Tests**: `python3 -m pytest tests/`
