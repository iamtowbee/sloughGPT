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
Two drivers share **`SloughGPTModel`** (checkpoints differ):
- **`train_sloughgpt.py`** (repo root): **`train_sloughgpt()`** — char-level file dataset, exports, **`--resume`**; Colab-aligned.
- **`SloughGPTTrainer`** (`domains.training.train_pipeline`): **`cli.py train`** (local), **`POST /training/start`**, **`examples/quick_train.py`**.

CI: **`tests/test_checkpoint_utils.py`**, **`tests/test_train_sloughgpt_*.py`**, **`tests/test_sloughgpt_trainer_smoke.py`**, **`train_sloughgpt.py --help`**.

Shared checkpoint I/O: **`packages/core-py/domains/training/checkpoint_utils.py`** (`normalize_raw_checkpoint`, `load_sloughgpt_from_checkpoint`, …) used by **`train_sloughgpt.py`**, **`SloughGPTTrainer`** loads, **`cli.py generate`**, **`ModelLoader._load_pt`**, and **`scripts/export_to_gguf.py`** for consistent `.pt` parsing.

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
python3 -m pip install -e ".[dev]"

# Start API server
cd apps/api/server && python3 main.py

# Start web UI
cd apps/web/web && npm run dev

# Run tests
python3 -m pytest tests/ -q
# or (uses .venv when present): ./run.sh python3 -m pytest tests/ -q

# CLI (repo root)
python3 cli.py --help
python3 apps/cli/cli.py --help

# Char-level trainer script (repo root)
python3 train_sloughgpt.py --help
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
# Full suite (from repo root)
python3 -m pytest tests/ -q

# Optional: path checks + same ruff smoke as CI (prints parity commands for ci_cd.yml jobs)
./verify.sh
```

- **Python CI subset:** `.github/workflows/reusable-ci-core.yml` (`workflow_call`): ruff smoke includes **`train_sloughgpt.py`**; pytest includes training smoke tests (see **CONTRIBUTING.md**).
- **Also in `ci_cd.yml`:** `test-web`, `test-sdk-ts`, `sdk-test-py`, `standards-schemas` (run `python3 scripts/validate_standards_schemas.py`; `jsonschema` is in `python3 -m pip install -e ".[dev]"`).

## Environment

- Python 3.9+ (use `python3` command)
- Node.js **20** for web / TS SDK (repo root **`.nvmrc`**; matches CI `setup-node`)
- SQLite for data (server/database.sqlite)

## Gaps Filled (Recent)

- RLHF/PPO training
- Model pruning
- Knowledge distillation  
- INT4/INT8 quantization
- AWQ/GPTQ support
- KV cache optimization
- CPU-specific optimizations
