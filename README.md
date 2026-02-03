
# SloGPT

![SloGPT](assets/slogpt.jpg)

A modular transformer training framework with advanced dataset management and multi-modal capabilities. Built for modern ML workflows with support for distributed training, state-aware models, and real-time dataset editing.

**Key Features:**
- Multi-dataset training with dynamic mixing ratios
- State-aware transformer models with special token handling
- Real-time dataset editing via FastAPI + OpenWebUI
- Distributed training (DDP) with Mac MPS support
- Automated fine-tuning scheduler
- Codebase-to-dataset pipeline
- Comprehensive monitoring with Wandb integration

## install

```bash
# Core dependencies
pip install torch numpy transformers datasets tiktoken wandb tqdm

# For development
pip install -r requirements-dev.txt

# Setup pre-commit hooks (optional)
pre-commit install
```

Core Dependencies:
- [pytorch](https://pytorch.org) - Training framework with MPS/CUDA support
- [numpy](https://numpy.org/install/) - Numerical operations
- [transformers](https://huggingface.co/transformers/) - Model checkpoint loading
- [datasets](https://huggingface.co/datasets/) - Dataset utilities
- [tiktoken](https://github.com/openai/tiktoken) - Fast BPE tokenization
- [wandb](https://wandb.ai) - Experiment tracking (enabled by default)
- [tqdm](https://tqdm.github.io/) - Progress bars

## quick start

**Fast training preset:**
```bash
python train.py config/train_fast.py
```

**Multi-dataset training:**
```bash
python train.py dataset=multi datasets='{"webtext": 0.7, "code": 0.3}'
```

**Mac MPS training:**
```bash
python train.py --device=mps --compile=False
```

**Chat with your trained model:**
```bash
python chat.py --out_dir=out-mydata
```

**Training dashboard:**
```bash
python train_ui.py
```

## intelligent dataset extraction

**Simple lmtrain integration using existing infrastructure:**
```bash
# One-step: web search + create dataset
python simple_lmtrain.py --query "react hooks examples" --dataset_name react_hooks

# Convert existing lmtrain output to dataset
python simple_lmtrain.py --input ../lmtrain/output.jsonl --dataset_name existing_data

# With language filter and limits
python simple_lmtrain.py --query "machine learning" --dataset_name ml_python --language python --max_examples 50
```

**Clean Workflow:**
1. 🔍 **lmtrain**: LLM-powered web search and intelligent code extraction
2. 🔄 **simple_lmtrain.py**: Converts lmtrain JSONL to text format
3. 📋 **mydata/prepare.py**: Reuses existing text processor (no code duplication!)
4. 🚀 **SloGPT**: Ready for training with `python train.py --dataset=<name>`

**Key Benefits:**
- 🤖 **LLM-powered intelligent crawling** from lmtrain
- ♻️ **Reuses existing code** - no custom prepare.py needed
- 🎯 **Works with infrastructure** - leverages existing mydata/prepare.py
- 📦 **Standard format** - creates proper train.bin/val.bin + meta.pkl
- 🚀 **Zero-friction** - immediate training compatibility

**Why Use Existing mydata/prepare.py:**
- ✅ **Perfect for text input** - lmtrain output is just text
- ✅ **Battle-tested** - works reliably for custom datasets
- ✅ **Character encoding** - ideal for code and mixed content
- ✅ **Simple maintenance** - one prepare.py to maintain, not dozens

**Direct repo_obtainer usage:**
```bash
# Clone and index repository
python repo_obtainer.py index --source https://github.com/user/repo.git

# Export to dataset format  
python repo_obtainer.py export --source /path/to/repo --format dataset

# Export local directory
python repo_obtainer.py export --source ./my_project --format dataset
```

## dataset editor (FastAPI + OpenWebUI)

Use OpenWebUI as the interface and call dataset update endpoints via tools.

Repository layout (monorepo):
- `packages/core/` contains core training, models, services, configs, notebooks
- `packages/apps/` contains view-layer apps and assets
- `datasets/` stores all dataset folders (with a `data/` symlink for legacy scripts)
- `runs/` stores training outputs (with `out*` symlinks for legacy scripts)
- `docs/` contains architecture and structure notes
- `tests/` contains test scaffolding (symlink to `packages/core/tests`)
- `config/` is a symlink to `packages/core/src/configs`

Entrypoints (wrappers):
- Training: `python train.py`
- Sampling: `python sample.py`
- Chat: `python chat.py`
- API server: `python api_server.py`
- Training UI: `python train_ui.py`
- Dataset extraction: `python enhanced_dataset_extractor.py`

Quick training preset:
- `python train.py config/train_fast.py` for faster iteration

Monorepo metadata:
- Root `pyproject.toml` points to `packages/core` and `packages/apps`.

Migration:
- See `docs/MIGRATION.md` for legacy path mapping.

Changelog:
- `CHANGELOG.md`

Contributing:
- `CONTRIBUTING.md`

Code of Conduct:
- `CODE_OF_CONDUCT.md`

Security:
- `SECURITY.md`

Release:
- `RELEASE.md`

Roadmap:
- `docs/ROADMAP.md`

Porting:
- `docs/PORTING_ASM_WASM.md`

OpenWebUI wrapper:
- `docs/OPENWEBUI_INTEGRATION.md`

OpenWebUI fork:
- `integrations/openwebui/README.md`

Chat logging and fine-tune buffer:
```sh
SLO_CHAT_LOG=1 python api_server.py --out_dir=out-mydata
python chat_buffer_to_dataset.py --input runs/chat_buffer.jsonl --dataset chatbuffer
```

Fine-tune scheduler:
```sh
SLO_CHAT_LOG=1 SLO_TUNE_EVERY=100 python api_server.py --out_dir=out-mydata
python finetune_scheduler.py --dataset chatbuffer --out_dir out-mydata --device mps
```

State bins:
- `train_state.bin` / `val_state.bin` contain signed int16 state values for special tokens.

Codebase data structure:
- `docs/DATA_STRUCTURE.md`

Makefile shortcuts:
- `make setup`
- `make train`
- `make sample`
- `make chat`
- `make api`
- `make train-ui`
- `make lint`
- `make format`

CI:
- `.github/workflows/ci.yml` runs `ruff check .`
- CI installs `packages/core` and `packages/apps` and runs pytest
- CI runs on Python 3.9–3.11

Pre-commit:
```sh
pip install pre-commit
pre-commit install
```

Tests:
```sh
make test
```

Dev dependencies:
```sh
pip install -r requirements-dev.txt
```

### Backend (FastAPI)

```sh
python api_server.py --out_dir=out-mydata --port=8000
```

Dataset endpoints:
- `GET /dataset/status?dataset=mydata`
- `POST /dataset/update_text` (JSON body: `{ "text": "...", "mode": "replace|append", "dataset": "mydata" }`)
- `POST /dataset/upload` (multipart form: `file`, `mode`, `dataset`)

These endpoints update `data/<dataset>/input.txt` and re-run `prepare.py`.

### OpenWebUI

1. Start OpenWebUI and connect to `http://localhost:8000/v1` as a model provider.
2. Add the API as a tool using the OpenAPI spec at `http://localhost:8000/openapi.json`.
3. Call `dataset_update_text` or `dataset_upload` from the tool panel.

## training dashboard (Gradio)

Run a lightweight training dashboard with real-time logs and loss curves:

```sh
python train_ui.py
```

Use the UI to start/stop training, view output logs, and visualize loss over time.

## advanced training

**Multi-GPU training:**
```bash
torchrun --standalone --nproc_per_node=8 train.py
```

**Multi-node training:**
```bash
# Master node
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=IP_ADDRESS --master_port=1234 train.py
# Worker node  
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=IP_ADDRESS --master_port=1234 train.py
```

**State-aware training:**
```bash
python train.py --dataset your_dataset --use_state_head
```

**Resume training:**
```bash
python train.py --init_from=resume
```

**From GPT-2 checkpoint:**
```bash
python train.py --init_from=gpt2 --dataset your_dataset
```

## fine-tuning

**From GPT-2 checkpoint:**
```bash
python train.py --init_from=gpt2 --dataset your_dataset --learning_rate=5e-5 --max_iters=1000
```

**Automated fine-tuning scheduler:**
```bash
# Enable chat logging and auto-tune every 100 messages
SLO_CHAT_LOG=1 SLO_TUNE_EVERY=100 python api_server.py --out_dir=out-mydata
python finetune_scheduler.py --dataset chatbuffer --out_dir out-mydata --device mps
```

**Chat buffer to dataset:**
```bash
SLO_CHAT_LOG=1 python api_server.py --out_dir=out-mydata
python chat_buffer_to_dataset.py --input runs/chat_buffer.jsonl --dataset chatbuffer
```

## sampling & inference

**From your trained model:**
```bash
python sample.py --out_dir=out-mydata --start="Your prompt here" --num_samples=3
```

**From GPT-2 checkpoint:**
```bash
python sample.py --init_from=gpt2-xl --start="What is the answer to life?" --num_samples=5
```

**Interactive chat:**
```bash
python chat.py --out_dir=out-mydata
```

**State-aware sampling:**
```bash
python sample.py --out_dir=out-mydata --use_state_head --start="Special token prompt"
```

## performance optimization

**Benchmarking:**
```bash
python bench.py  # Profile training loop performance
```

**Compilation:**
- CUDA: Uses `torch.compile()` by default for 2-3x speedup
- MPS: Compilation not supported, uses eager execution
- CPU: Set `--compile=False` for compatibility

**Memory optimization:**
- Gradient checkpointing for large models
- Automatic mixed precision (AMP) on CUDA
- Efficient memory-mapped datasets

## architecture

**Modern transformer features:**
- Rotary Position Embeddings (RoPE)
- SwiGLU activation function  
- RMSNorm normalization
- State-aware heads for special tokens

**Model variants:**
- Classic GPT-2 style architecture
- LLaMA-style modern architecture
- Hybrid configurations supported

## troubleshooting

**Common issues:**
- CUDA memory: Reduce `batch_size` or use gradient accumulation
- MPS compilation: Add `--compile=False --device=mps`
- Import errors: Ensure all dependencies installed with `pip install -r requirements-dev.txt`

**Performance tips:**
- Use `--compile=True` on CUDA for 2-3x speedup
- Enable MPS on Apple Silicon (`--device=mps`)
- Use multi-dataset mode for varied training data

## development

**Linting & formatting:**
```bash
make lint     # Run ruff checks
make format   # Auto-format code
```

**Testing:**
```bash
make test     # Run pytest
```

**Development setup:**
```bash
make setup    # Install dev dependencies and pre-commit hooks
```
