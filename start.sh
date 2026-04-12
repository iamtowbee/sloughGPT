#!/bin/bash
# Start both SloughGPT API and Web servers

cd /Users/mac/sloughGPT

echo "=== Starting SloughGPT ==="

# Kill existing servers
pkill -f "python3 apps/api/server/main.py" 2>/dev/null
pkill -f "next dev" 2>/dev/null
sleep 2

# Start API server
echo "Starting API server on port 8000..."
nohup python3 apps/api/server/main.py > /tmp/sloughgpt-api.log 2>&1 &
API_PID=$!
echo "API PID: $API_PID"

# Wait for API to start
sleep 5

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "API server: OK"
else
    echo "API server: FAILED"
    tail -20 /tmp/sloughgpt-api.log
fi

# Start Web server
echo "Starting Web server on port 3000..."
cd apps/web
nohup npm run dev > /tmp/sloughgpt-web.log 2>&1 &
WEB_PID=$!
echo "Web PID: $WEB_PID"

sleep 5

# Check if Web is running
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "Web server: OK"
else
    echo "Web server: FAILED"
    tail -20 /tmp/sloughgpt-web.log
fi

echo ""
echo "=== SloughGPT is running ==="
echo "API:  http://localhost:8000"
echo "Web:  http://localhost:3000"
echo "Docs: http://localhost:3000/api-docs"
echo ""
echo "Logs:"
echo "  API: tail -f /tmp/sloughgpt-api.log"
echo "  Web: tail -f /tmp/sloughgpt-web.log"
