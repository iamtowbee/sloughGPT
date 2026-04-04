## API

`apps/api/server/` contains the FastAPI app (**`main.py`**) and API-specific **`requirements.txt`**.

```bash
# From repo root
python3 apps/api/server/main.py

# Or with reload
cd apps/api/server && python3 -m uvicorn main:app --reload --port 8000
```

Training routes and shared schemas live alongside **`main.py`** (e.g. **`training/`**). Native trainer **`step_*.pt`** on the API host embeds **`stoi` / `itos` / `chars`** for **`cli.py eval`**; see **`docs/policies/CONTRIBUTING.md`** (*Checkpoint vocabulary*). See **QUICKSTART.md** and **docs/AI_SOFTWARE_ENGINEERING.md** for boundaries.
