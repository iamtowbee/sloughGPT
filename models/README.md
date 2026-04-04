# `models/` — local artifacts (mostly gitignored)

This directory holds **weights**, **Soul (`.sou`)** bundles, exports, and occasional **Hugging Face–style tokenizer** files. Large binaries are listed in **`.gitignore`**; small sidecars like **`.sou.meta.json`** may still be tracked.

## Char-level SloughGPT vs other formats

- **Native char LM:** `.pt` checkpoints (especially trainer **`step_*.pt`**) and some **`.meta.json`** sidecars embed **`stoi` / `itos` / `chars`** for fair **`cli.py eval`**. See **`docs/policies/CONTRIBUTING.md`** (*Checkpoint vocabulary*).
- **Soul (`.sou`):** self-contained identity + weights; sidecar **`.sou.meta.json`** mirrors the Soul JSON for inspection. New exports serialize non-finite losses as JSON **`null`** (not `Infinity`).
- **GPT‑2 BPE files** (`vocab.json`, `merges.txt`, `tokenizer_config.json`, `special_tokens_map.json`): subword tokenizer for HF-style models, **not** the char vocabulary used by the SloughGPT char trainer. Prefer a subfolder (e.g. `tokenizers/gpt2/`) if you keep them here.
- **Extensionless zips** (e.g. API dumps): rename to **`.zip`** so tooling and docs stay clear.

## Suggested layout (optional)

| Area | Purpose |
|------|---------|
| `exported/` | Staged release builds (GGUF, safetensors, `.sou`, …) |
| `registry/models.json` | App/registry metadata (not a filesystem index) |
| `*/` per run | Colab smoke, local demos—keep or prune by age |

## CLI resolution

**`cli.py generate`** (local) resolves candidates under **`models/`** (e.g. **`sloughgpt.sou`**, newest **`*.sou`**, then **`sloughgpt_finetuned.pt`**) per **`apps/cli/cli.py`** (`_local_soul_candidate_paths`).
