## Packages

Shared libraries consumed by apps.

- `core-py/` — Python shared domain logic (`domains` package on the import path when the repo is installed).
- `sdk-py/` — Python SDK (`sloughgpt_sdk`).
- `sdk-ts/typescript-sdk/` — TypeScript SDK (npm package).
- `standards/` — contracts/schemas.

Install from the **repository root** so imports resolve: `pip install -e ".[dev]"` (see **pyproject.toml**). Layout overview: **docs/STRUCTURE.md**.

For **npm** work (`sdk-ts/typescript-sdk/` and **`apps/web/web/`**), use the **Node** version in **`.nvmrc`** at the repo root (`nvm use` / `fnm use`; matches CI).
