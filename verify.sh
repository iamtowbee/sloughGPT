#!/bin/bash
# Quick verification script for SloughGPT (monorepo paths)

echo "🔍 Verifying SloughGPT installation..."

# Check Python
echo "Checking Python..."
python3 --version

# Check Node (optional for web)
echo "Checking Node..."
if command -v node &>/dev/null; then
    node --version
else
    echo "(node not found — skip web checks)"
fi

echo ""
echo "Checking required files..."

files=(
    "apps/api/server/main.py"
    "packages/core-py/domains/ui/api_server.py"
    "apps/web/web/package.json"
    "apps/web/web/app/(app)/page.tsx"
    "apps/web/web/app/(app)/chat/page.tsx"
    "pyproject.toml"
)

all_found=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file NOT FOUND"
        all_found=false
    fi
done

if [ "$all_found" = true ]; then
    echo ""
    echo "✅ Core paths present!"
    echo ""
    if python3 -m ruff --version &>/dev/null; then
        echo "Ruff smoke (same rules as CI)..."
        python3 -m ruff check tests/ apps/cli/ apps/api/server/ --select E9,F63,F7,F82 || {
            echo "❌ Ruff smoke failed"
            exit 1
        }
        echo "✓ Ruff smoke passed"
        echo ""
    else
        echo "(Optional: pip install ruff to run the CI lint smoke check locally.)"
        echo ""
    fi
    echo "To start SloughGPT:"
    echo "  1. API:  python3 apps/api/server/main.py"
    echo "     or:  cd apps/api/server && python3 -m uvicorn main:app --reload --port 8000"
    echo "  2. Web: cd apps/web/web && npm run dev"
    echo ""
    echo "Docker: docker compose -f infra/docker/docker-compose.yml up -d api"
    echo ""
    echo "Then open the web dev URL (often http://localhost:3000)"
    echo ""
    echo "Web CI parity (job test-web in .github/workflows/ci_cd.yml):"
    echo "  cd apps/web/web && npm ci && npm run lint && npm run typecheck"
    echo ""
    echo "TypeScript SDK CI parity (job test-sdk-ts):"
    echo "  cd packages/sdk-ts/typescript-sdk && npm ci && npm run lint && npm run build && npm test"
    echo ""
    echo "Python SDK CI parity (job sdk-test-py):"
    echo "  python3 -m pytest tests/test_sdk.py -q"
    echo ""
    echo "Standards CI parity (job standards-schemas):"
    echo "  python3 scripts/validate_standards_schemas.py"
    echo "  (jsonschema: pip install -e \".[dev]\" or pip install jsonschema)"
else
    echo ""
    echo "❌ Some files are missing!"
    exit 1
fi
