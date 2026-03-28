# Contributing

Thanks for helping improve SloughGPT.

## Quick start

1. **Branch** from `main` with a descriptive name.
2. **Install** from the repo root: `pip install -e ".[dev]"` (see **QUICKSTART.md**).
3. **Validate**:
   - `./verify.sh` and `python3 -m pytest tests/` (Python; CI subset in **`.github/workflows/reusable-ci-core.yml`**).
   - If you touch **`apps/web/web/`**: `cd apps/web/web && npm ci && npm run lint && npm run typecheck` (see **`.github/workflows/ci_cd.yml`** job `test-web`).
4. **Open a PR** with a clear description of intent and scope (see **`.github/pull_request_template.md`**).

## Issues

Use **`.github/ISSUE_TEMPLATE/`** when filing bugs or feature requests (blank issues stay enabled).

## Structure

See **docs/STRUCTURE.md** and **docs/MIGRATION.md** for layout and migration history.
