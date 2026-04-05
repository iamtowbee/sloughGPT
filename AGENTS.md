# Agents

- **Repo map & commands:** [.agents/skills/SKILL.md](.agents/skills/SKILL.md). **API + web** in one terminal: repo root **`npm install && npm run dev:stack`**, or **`./scripts/dev-stack.sh`** / **`make dev-stack`** — [QUICKSTART.md](QUICKSTART.md).
- **Colab notebook:** root [`sloughgpt_colab.ipynb`](sloughgpt_colab.ipynb); full execute [`scripts/run_colab_notebook_smoke.sh`](scripts/run_colab_notebook_smoke.sh) or **`make colab-smoke`** (`--help`, **`make help`**); regression **`make colab-test`** / [`tests/test_sloughgpt_colab_notebook.py`](tests/test_sloughgpt_colab_notebook.py) — [README.md](README.md) (*Google Colab*). **§11** cognitive (SM-2 + SCAMPER) must remain **one** code cell (single **`_asyncio_run`**). One pytest subtest needs **`bash`** on **`PATH`** (skipped otherwise; see [CONTRIBUTING.md](CONTRIBUTING.md)).
- **Contributing & CI parity:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Layout:** [docs/STRUCTURE.md](docs/STRUCTURE.md); runtime dirs (**`data/experiments`**, **`data/features`**, **`data/tuning`**, **`data/vector_store`**) — [data/README.md](data/README.md)
- **CLI → TUI roadmap (planning):** [docs/plans/tui-cli-port.md](docs/plans/tui-cli-port.md)
