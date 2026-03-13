#!/bin/bash
set -e

echo "🚀 Starting SloughGPT..."

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "🐳 Using Docker..."
    docker-compose up
else
    echo "🐍 Using local Python..."
    
    # Start API server
    echo "Starting API server on port 8000..."
    cd server
    python main.py &
    API_PID=$!
    cd ..
    
    # Start Web server
    echo "Starting Web server on port 3000..."
    cd web
    npm run dev &
    WEB_PID=$!
    cd ..
    
    echo "✅ SloughGPT is running!"
    echo "  API:   http://localhost:8000"
    echo "  Web:   http://localhost:3000"
    echo ""
    echo "Press Ctrl+C to stop..."
    
    trap "kill $API_PID $WEB_PID 2>/dev/null" EXIT
    
    wait
fi
