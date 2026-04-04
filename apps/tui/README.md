# TUI (planned)

Interactive terminal UI for SloughGPT is **not implemented** in this folder yet.

- **Roadmap and codebase map:** [docs/plans/tui-cli-port.md](../../docs/plans/tui-cli-port.md)
- **Today’s CLI:** [apps/cli/](../cli/) — `sloughgpt` / `python3 cli.py`

When work starts, this directory should hold the Textual (or similar) app, **adapters** that call existing `cmd_*` / `domains` / HTTP APIs, and optional **`[project.optional-dependencies] tui`** in **`pyproject.toml`**.
