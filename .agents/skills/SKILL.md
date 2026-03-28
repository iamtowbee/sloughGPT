# SloughGPT - Enterprise AI Framework

## Overview

SloughGPT is an enterprise-grade AI framework with production-ready ML infrastructure for training, fine-tuning, and deploying large language models.

## Project Structure

```
/Users/mac/sloughGPT/
├── apps/
│   ├── api/server/          # FastAPI app (main.py), API requirements.txt
│   ├── web/web/             # Next.js 14 UI (port 3000)
│   └── cli/                 # cli.py (repo-root launcher may wrap this)
├── packages/
│   ├── core-py/domains/     # Python domains (training, inference, models, …)
│   ├── sdk-py/              # Python SDK
│   ├── sdk-ts/typescript-sdk/  # TypeScript SDK (npm package root)
│   └── standards/           # SloughGPT Standard v1 docs
├── infra/docker/            # docker-compose and deployment assets
├── tests/                   # pytest suites
├── requirements.txt         # Root Python deps (install first)
└── TODO.md                  # Roadmap notes
```

## Key Features

### Training
- **LR Schedulers**: Cosine, warmup, OneCycle, cyclic, polynomial
- **Mixed Precision**: FP32, FP16, BF16 with GradScaler
- **Distributed**: DDP + FSDP
- **Memory**: Activation/gradient checkpointing, flash attention
- **Fine-tuning**: LoRA, QLoRA, LoRA+, IA3
- **RLHF**: PPO training with reward model

### Efficiency (Low-End Devices)
- **Quantization**: INT4, INT8, FP16
- **Pruning**: Magnitude, gradient, structured, lottery ticket
- **Distillation**: Temperature-based, feature-based, progressive
- **Optimizations**: KV cache, CPU optimizations, dynamic batching

### Deployment
- **Web UI**: Next.js 14 with chat, training, models, experiments
- **API**: FastAPI with auth, streaming, WebSocket
- **CLI**: Full command-line interface with shell completions
- **Models**: 14 custom architectures + HuggingFace integration

## Commands

```bash
# Editable install (domains + apps.cli; dev extras: ruff, pytest, sloughgpt CLI)
pip install -e ".[dev]"

# Start API server
cd apps/api/server && python3 main.py

# Start web UI
cd apps/web/web && npm run dev

# Run tests
python3 -m pytest tests/ -q

# CLI (repo root)
python3 apps/cli/cli.py --help
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Server health |
| `/auth/login` | Login (query params) |
| `/auth/register` | Register (query params) |
| `/models` | List models |
| `/inference/generate` | Run inference |
| `/training/start` | Start training job |
| `/training/jobs` | List jobs |
| `/experiments` | List experiments |

## Technology Stack

- **Backend**: FastAPI, SQLite, WebSocket
- **Frontend**: Next.js 14, Tailwind CSS, Zustand
- **ML**: PyTorch, Transformers
- **Testing**: pytest (`tests/`)

## Important Notes

1. **No downloads required** - Everything runs locally
2. **Auth uses query params** - Not JSON body (for /auth/login, /auth/register)
3. **Run `python3 -m pytest tests/`** - full suite is the source of truth for pass count
4. **Quantization reduces memory**: 7B model = 14GB (FP16) → 3.3GB (INT4)
5. **Multi-modal is disabled** - Focus on core LLM training

## Testing

```bash
# Run all tests
python3 tests/run_tests.py

# Run specific test suite
python3 tests/ml/test_infrastructure.py
python3 tests/cli/test_cli.py
python3 tests/web/test_frontend.py
```

## Environment

- Python 3.9+ (use `python3` command)
- Node.js for web
- SQLite for data (server/database.sqlite)

## Gaps Filled (Recent)

- RLHF/PPO training
- Model pruning
- Knowledge distillation  
- INT4/INT8 quantization
- AWQ/GPTQ support
- KV cache optimization
- CPU-specific optimizations
