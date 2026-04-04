## Apps

Runnable services and application entrypoints.

- `api/` — FastAPI server and routers (`api/server/main.py`; see **`api/README.md`**). HTTP training uses **`SloughGPTTrainer`** **`step_*.pt`** charset semantics — **`docs/policies/CONTRIBUTING.md`** (*Checkpoint vocabulary*).
- `web/` — Next.js frontend (**`app/(app)/`** routes under **`app/`**).
- `cli/` — CLI (`sloughgpt` entrypoint via `pyproject.toml`; see **`cli/README.md`** for **`cli.py train`** export naming (**`--save-stem`**) and **`cli.py generate`** local `.sou` resolution).
- `tui/` — placeholder for a future interactive terminal UI; plan: **`docs/plans/tui-cli-port.md`**.

### Quick start (repo root)

Use **Node 20** locally if you can: **`nvm use`** / **`fnm use`** reads **`.nvmrc`** at the repo root (same as **`test-web`** / **`test-sdk-ts`** in **`ci_cd.yml`**).

1. **`./verify.sh`** — checks core paths; runs the same ruff smoke as CI when `ruff` is installed (use `python3 -m pip install -e ".[dev]"` if needed); runs **`apps/web`** **`npm run ci`** when **`node_modules`** exists. Also prints **`ci_cd.yml`** parity commands (**`test-web`**, **`test-sdk-ts`**, **`sdk-test-py`**, **`standards-schemas`**).
2. **API:** `python3 apps/api/server/main.py`, or `cd apps/api/server && python3 -m uvicorn main:app --reload --port 8000`.
3. **Web:** `cd apps/web && npm run dev`. Before pushing web changes, **`npm ci && npm run ci`** (matches CI job **`test-web`** in **`.github/workflows/ci_cd.yml`**).

**Docker:** `docker compose -f infra/docker/docker-compose.yml up -d api` (see **QUICKSTART.md** / **docs/DEPLOYMENT.md** for profiles).
