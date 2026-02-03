# Migration notes

## Legacy paths

Legacy paths still work via symlinks and thin wrapper scripts:

- `data/` → `datasets/`
- `out*` → `runs/*`
- root scripts (e.g. `train.py`) → `bin/*` wrappers

## Monorepo layout

- `packages/core/src/` — models, controllers, services, configs, notebooks
- `packages/apps/apps/` — UI entrypoints

If you have custom scripts, update imports to:

```py
from packages.core.src.models.model import Slo
```
