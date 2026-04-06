# TUI (Phase 1 — shell building blocks)

**Implemented:** session + read-only HTTP adapter; **stub** CLI entry (no Textual yet).

| Module | Role |
|--------|------|
| `session.py` | `TuiSession`, `discover_repo_root()` |
| `adapters/http_api.py` | `fetch_health()` → `GET /health` |
| `app.py` | `python3 -m apps.tui` argparse stub |

- **Roadmap:** [docs/plans/tui-cli-port.md](../../docs/plans/tui-cli-port.md)
- **Tests (TDD):** `tests/test_tui_phase1.py`
- **Main CLI:** [apps/cli/](../cli/) — `sloughgpt` / `python3 cli.py`

Later: Textual screens, train/export adapters, optional **`[project.optional-dependencies] tui`** in **`pyproject.toml`**.
