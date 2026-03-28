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
pip install -r requirements.txt
pip install -r apps/api/server/requirements.txt
pip install -e ".[dev]"

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
( cd apps/web/web && npm install )

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
echo "To start development:"
echo "  Terminal 1: cd apps/api/server && python3 main.py"
echo "  Terminal 2: cd apps/web/web && npm run dev"
echo ""
echo "Or use Docker:"
echo "  docker compose -f infra/docker/docker-compose.yml up"
