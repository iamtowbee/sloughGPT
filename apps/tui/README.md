# TUI (Phase 1 — shell building blocks)

**Implemented:** session, HTTP JSON GET helpers, local repo scan (same idea as `cli.py status`), and a **stub** CLI (no Textual yet).

| Module | Role |
|--------|------|
| `session.py` | `TuiSession`, `discover_repo_root()` |
| `adapters/http_api.py` | `fetch_health`, `fetch_metrics`, `fetch_health_detailed` |
| `adapters/local_status.py` | `scan_local_repo()` → `LocalStatusSnapshot` |
| `app.py` | `python3 -m apps.tui` CLI probes |

**CLI (examples):**

```bash
python3 -m apps.tui --local-status
python3 -m apps.tui --local-status --repo-root /path/to/sloughGPT
python3 -m apps.tui --api-health --host 127.0.0.1 --port 8000
python3 -m apps.tui --api-metrics
python3 -m apps.tui --api-health-detailed
```

- **Roadmap:** [docs/plans/tui-cli-port.md](../../docs/plans/tui-cli-port.md)
- **Tests (TDD):** `tests/test_tui_phase1.py`
- **Main CLI:** [apps/cli/](../cli/) — `sloughgpt` / `python3 cli.py`

Later: Textual screens, train/export adapters, optional **`[project.optional-dependencies] tui`** in **`pyproject.toml`**.
