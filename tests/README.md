## Tests

Pytest collection lives under this directory (see **`pytest.ini`** at the repo root).

- **Top-level** — `test_*.py` modules by area (API, SDK, training, inference, security, etc.).
- **`server/`** — additional HTTP/API coverage (`test_server_api.py`).
- **Markers** — `unit`, `integration`, `e2e`, `slow` (see `pytest.ini`).

From the repo root: `python3 -m pytest tests/` (full suite) or the subset used in **`.github/workflows/reusable-ci-core.yml`**. Optional: `./verify.sh` for path checks and the same ruff smoke as CI.
