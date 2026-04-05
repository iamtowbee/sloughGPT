#!/usr/bin/env bash
# Start FastAPI (default :8000) and Next.js dev (:3000). Ctrl+C stops both.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cleanup() {
  if [[ -n "${API_PID:-}" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting API: python3 apps/api/server/main.py"
python3 apps/api/server/main.py &
API_PID=$!

echo "Starting web: npm run dev (apps/web)"
(cd "$ROOT/apps/web" && npm run dev)
