## Repository Structure Migration

This repository has been moved to a layered monorepo layout:

- `apps/` runnable applications (`api`, `web`, `cli`)
- `packages/` shared libraries (`core-py`, `sdk-py`, `sdk-ts`, `standards`)
- `infra/` docker and kubernetes assets

### Compatibility links currently retained

To preserve backward compatibility during rollout, these root-level paths are currently symlinked to canonical locations:

- `server` -> `apps/api/server`
- `web` -> `apps/web/web`
- `sloughgpt` -> `apps/cli/sloughgpt`
- `domains` -> `packages/core-py/domains`
- `sloughgpt_sdk` -> `packages/sdk-py/sloughgpt_sdk`
- `typescript-sdk` -> `packages/sdk-ts/typescript-sdk`
- `standards` -> `packages/standards/standards`
- docker/k8s/helm/grafana/prometheus and root Dockerfiles -> `infra/*`

### Canonical paths moving forward

- API server: `apps/api/server/main.py`
- Web app: `apps/web/web`
- CLI: `apps/cli/cli.py`
- Python SDK: `packages/sdk-py/sloughgpt_sdk`
- TS SDK: `packages/sdk-ts/typescript-sdk`

These compatibility links can be removed once downstream tooling and docs stop referencing legacy root paths.
