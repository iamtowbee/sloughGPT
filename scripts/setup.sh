#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "🚀 Setting up SloughGPT Development Environment"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📌 Python version: $python_version"

# Check Node.js version
node_version=$(node --version 2>&1)
echo "📌 Node.js version: $node_version"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
python3 -m pip install -r requirements.txt
python3 -m pip install -r apps/api/server/requirements.txt
python3 -m pip install -e ".[dev]"

# Install Node.js dependencies (web + TS SDK; use .nvmrc / Node 20 if you can)
echo "📦 Installing Node.js dependencies..."
( cd apps/web/web && npm install )
( cd packages/sdk-ts/typescript-sdk && npm install )

# Create .env files if they don't exist
echo "📝 Setting up environment files..."

if [ ! -f ".env" ]; then
    cp .env.example .env 2>/dev/null || echo "API_URL=http://localhost:8000" > .env
    echo "Created .env at repo root"
fi

if [ ! -f "apps/web/web/.env.local" ]; then
    cp apps/web/web/.env.example apps/web/web/.env.local 2>/dev/null || echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > apps/web/web/.env.local
    echo "Created apps/web/web/.env.local"
fi

echo "✅ Setup complete!"
echo ""
echo "Optional sanity check:"
echo "  ./verify.sh"
echo ""
echo "To start development:"
echo "  Terminal 1: cd apps/api/server && python3 main.py"
echo "  Terminal 2: cd apps/web/web && npm run dev"
echo ""
echo "Or use Docker:"
echo "  docker compose -f infra/docker/docker-compose.yml up -d api"
echo ""
echo "Before PRs that touch apps/web/web, run the same checks as CI (test-web):"
echo "  cd apps/web/web && npm ci && npm run lint && npm run typecheck"
echo ""
echo "Before PRs that touch packages/sdk-ts/typescript-sdk (test-sdk-ts):"
echo "  cd packages/sdk-ts/typescript-sdk && npm ci && npm run lint && npm run build && npm test"
echo ""
echo "Before PRs that touch packages/sdk-py/sloughgpt_sdk (sdk-test-py):"
echo "  python3 -m pytest tests/test_sdk.py -q"
echo ""
echo "Before PRs that touch packages/standards (standards-schemas):"
echo "  python3 scripts/validate_standards_schemas.py"
echo "  (jsonschema: already in pip install -e \".[dev]\", or: pip install jsonschema)"
