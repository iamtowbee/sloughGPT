# Contributing

Thanks for helping improve SloughGPT.

## Quick start

1. **Branch** from `main` with a descriptive name.
2. **Install** from the repo root: `pip install -e ".[dev]"` (see **QUICKSTART.md**).
3. **Validate**:
   - `./verify.sh` and `python3 -m pytest tests/` (Python; CI subset in **`.github/workflows/reusable-ci-core.yml`**). With a **`.venv`**, you can use **`./run.sh python3 -m pytest tests/`** so the same interpreter is on **`PATH`**.
   - If you touch **`apps/web/web/`**: `cd apps/web/web && npm ci && npm run lint && npm run typecheck` (job **`test-web`**).
   - If you touch **`packages/sdk-ts/typescript-sdk/`**: `cd packages/sdk-ts/typescript-sdk && npm ci && npm run lint && npm run build && npm test` (job **`test-sdk-ts`**).
   - If you touch **`packages/sdk-py/sloughgpt_sdk/`**: `python3 -m pytest tests/test_sdk.py -q` (job **`sdk-test-py`**).
   - If you change **`packages/standards/`** or schemas: `python3 scripts/validate_standards_schemas.py` (**`jsonschema`** is in **`pip install -e ".[dev]"`**; otherwise `pip install jsonschema`). Job **`standards-schemas`**.
4. **Open a PR** with a clear description of intent and scope (see **`.github/pull_request_template.md`**).

## Issues

Use **`.github/ISSUE_TEMPLATE/`** when filing bugs or feature requests (blank issues stay enabled).

## Structure

See **docs/STRUCTURE.md** and **docs/MIGRATION.md** for layout and migration history.
