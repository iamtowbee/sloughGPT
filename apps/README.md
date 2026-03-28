## Apps

Runnable services and application entrypoints.

- `api/` — FastAPI server and routers (`api/server/main.py`; see **`api/README.md`**).
- `web/web/` — Next.js frontend (app router under `app/`).
- `cli/` — CLI (`sloughgpt` entrypoint via `pyproject.toml`; see **`cli/README.md`**).

### Quick start (repo root)

Use **Node 20** locally if you can: **`nvm use`** / **`fnm use`** reads **`.nvmrc`** at the repo root (same as **`test-web`** / **`test-sdk-ts`** in **`ci_cd.yml`**).

1. **`./verify.sh`** — checks core paths; runs the same ruff smoke as CI when `ruff` is installed (use `pip install -e ".[dev]"` if needed). Also prints **`ci_cd.yml`** parity commands (**`test-web`**, **`test-sdk-ts`**, **`sdk-test-py`**).
2. **API:** `python3 apps/api/server/main.py`, or `cd apps/api/server && python3 -m uvicorn main:app --reload --port 8000`.
3. **Web:** `cd apps/web/web && npm run dev`. Before pushing web changes, **`npm ci && npm run lint && npm run typecheck`** (matches CI job **`test-web`** in **`.github/workflows/ci_cd.yml`**).

**Docker:** `docker compose -f infra/docker/docker-compose.yml up -d api` (see **QUICKSTART.md** / **docs/DEPLOYMENT.md** for profiles).
