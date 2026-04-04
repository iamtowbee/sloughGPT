# `models/exported/`

Drop **release-style exports** here (GGUF, safetensors, `.sou`, large `.pt`, zips). Contents are **gitignored** so local builds do not dirty `git status`; only this README and `.gitkeep` stay tracked.

- Prefer dated or named subfolders (`2026-04-04-gpt2-export/`, `release-candidate/`) so runs do not overwrite each other.
- For char-LM checkpoints and **`.sou.meta.json`** semantics, see **`../README.md`** and **`docs/policies/CONTRIBUTING.md`** (*Checkpoint vocabulary*).
