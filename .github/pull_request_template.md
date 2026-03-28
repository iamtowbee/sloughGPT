## Summary

What does this change and why?

## Checklist

- [ ] Installed from repo root as needed: `pip install -e ".[dev]"` (see **QUICKSTART.md**)
- [ ] `./verify.sh` passes (path checks + ruff smoke when `ruff` is available)
- [ ] `python3 -m pytest tests/` passes (or the subset you touched; CI uses **`.github/workflows/reusable-ci-core.yml`**)
- [ ] If **`apps/web/web/`** changed: `cd apps/web/web && npm ci && npm run lint && npm run typecheck`
- [ ] If **`packages/sdk-ts/typescript-sdk/`** changed: `cd packages/sdk-ts/typescript-sdk && npm ci && npm run lint && npm run build && npm test`
- [ ] If **`packages/sdk-py/sloughgpt_sdk/`** changed: `python3 -m pytest tests/test_sdk.py -q`

## Notes

Optional: related issues, rollout notes, breaking changes.
