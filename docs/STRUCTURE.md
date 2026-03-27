# Monorepo structure

Lightweight layout:

- `apps/` — API server (`apps/api/server/`), CLI (`apps/cli/`), web UI (`apps/web/web/`)
- `packages/core-py/` — domains and shared Python core (`domains/`, `utils/`)
- `packages/sdk-py/sloughgpt_sdk/` — Python API client SDK
- `packages/standards/` — standards docs and JSON schemas
- `tests/` — pytest suite (root `pyproject.toml` `testpaths`)
- `infra/docker/` — Dockerfiles and Compose file for local runs

Install the core package from the repository root:

```sh
pip install -e .
```

See `docs/REPO_STRUCTURE_MIGRATION.md` for the full move map.
