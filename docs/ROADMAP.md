# sloughGPT roadmap (brief)

## Current state
- Text-only GPT for conversational inference.
- Training via dataset folders that produce `train.bin`/`val.bin`.
- Single dataset per training run.

## Near-term goals (next 1–3 iterations)
1. **Multi-dataset training**
   - Allow selecting multiple datasets at once.
   - Support dataset mixing ratios.
   - Produce combined binaries or stream batches from multiple bins.

2. **Dataset management**
   - Standardize dataset metadata (`meta.pkl`) across all datasets.
   - Add a simple registry (name, path, vocab, size).
   - Make dataset preparation consistent for each dataset type.

3. **Better inference UX**
   - Stable chat UI and API serving.
   - Quick switches between checkpoints/datasets.
   - Clear model info panel (dataset, vocab size, block size).

4. **Learning loop (safe)**
   - Log chats to buffer (JSONL).
   - Periodic batch fine-tune.
   - Hold-out eval to prevent skew.

## Longer-term goals
- **Adaptive learning loop**: opt-in logging of conversations, periodic fine-tuning.
- **Evaluation pipeline**: basic metrics per dataset (loss, perplexity).
- **Model variants**: allow larger configs for accuracy, smaller for speed.

## Open decisions
- How to merge vocabularies across datasets (shared vs per-dataset).
- Whether to build a combined bin or sample from multiple bins at runtime.
