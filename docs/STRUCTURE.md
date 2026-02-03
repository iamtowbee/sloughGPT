# Monorepo structure

This repository is organized as a lightweight monorepo:

- `packages/core/` — training, model code, services, configs, notebooks
- `packages/apps/` — view-layer apps and UI entrypoints
- `datasets/` — all datasets (legacy `data/` symlink)
- `runs/` — training outputs (legacy `out*` symlinks)
- `tests/` — symlink to `packages/core/tests`
- `docs/policies/` — changelog, contributing, security, release notes
- `config/` — symlink to `packages/core/src/configs`

## Package installs (optional)

You can install packages in editable mode:

```sh
pip install -e packages/core
pip install -e packages/apps
```

Root wrappers (`train.py`, `api_server.py`, etc.) still work without installation.
