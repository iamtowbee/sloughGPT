#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "🚀 Starting SloughGPT..."

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "🐳 Using Docker..."
    docker compose -f "$ROOT/infra/docker/docker-compose.yml" up
else
    echo "🐍 Using local Python..."
    
    # Start API server
    echo "Starting API server on port 8000..."
    ( cd "$ROOT/apps/api/server" && python3 main.py ) &
    API_PID=$!
    
    # Start Web server
    echo "Starting Web server on port 3000..."
    ( cd "$ROOT/apps/web/web" && npm run dev ) &
    WEB_PID=$!
    
    echo "✅ SloughGPT is running!"
    echo "  API:   http://localhost:8000"
    echo "  Web:   http://localhost:3000"
    echo ""
    echo "Press Ctrl+C to stop..."
    
    trap "kill $API_PID $WEB_PID 2>/dev/null" EXIT
    
    wait
fi
