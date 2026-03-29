# Monorepo structure

Lightweight layout:

- `apps/` — runnable apps (see **[apps/README.md](../apps/README.md)**): API (`apps/api/server/`), CLI (`apps/cli/`), web (`apps/web/web/`)
- `packages/core-py/` — domains and shared Python core (`domains/`, `utils/`); see **[packages/core-py/README.md](../packages/core-py/README.md)**
- `packages/sdk-py/sloughgpt_sdk/` — Python API client SDK (**[README.md](../packages/sdk-py/sloughgpt_sdk/README.md)**)
- `packages/sdk-ts/typescript-sdk/` — TypeScript SDK (npm root; **[README.md](../packages/sdk-ts/typescript-sdk/README.md)**)
- `packages/standards/` — standards docs and JSON schemas; see **[packages/standards/README.md](../packages/standards/README.md)**
- `tests/` — pytest suite (root `pyproject.toml` `testpaths`)
- `infra/docker/` — Dockerfiles and Compose file for local runs
- `run.sh` — optional wrapper: prepends `.venv/bin` to `PATH`, then runs the rest (e.g. `./run.sh python3 -m pytest tests/ -q`)
- `.cursor/rules/`, `.agents/skills/` — editor/agent guidance (not part of the installable Python package). Entry **`AGENTS.md`** at the repo root links here.

Install the package from the repository root (dev extras include **ruff**, **pytest**, **jsonschema**, and the **`sloughgpt`** CLI entry):

```sh
pip install -e ".[dev]"
```

Minimal install without dev tools: `pip install -e .`

See `docs/REPO_STRUCTURE_MIGRATION.md` for the full move map.

Contributor and security policies: **[CONTRIBUTING.md](../CONTRIBUTING.md)** and **[SECURITY.md](../SECURITY.md)** at the repo root (symlinks to `docs/policies/`).

CI: **`.github/workflows/reusable-ci-core.yml`** (Python ruff smoke + core pytest), **`ci_cd.yml`** (Python SDK **`sdk-test-py`**, TypeScript SDK **`test-sdk-ts`**, Next.js **`test-web`**, **`standards-schemas`**, Docker images, benchmarks), **`publish.yml`** (PyPI). Local parity commands are in **CONTRIBUTING.md**.
