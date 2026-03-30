## CLI

`apps/cli/` hosts the **`sloughgpt`** console entrypoint (**`pyproject.toml`** → **`apps.cli.cli:main`**). The repo root **`cli.py`** forwards to the same **`main`**.

```bash
python3 -m pip install -e ".[dev]"   # or python3 -m pip install -e .
sloughgpt --help
# or: python3 cli.py --help
```

See **QUICKSTART.md** for common commands and **CONTRIBUTING.md** for validation.
