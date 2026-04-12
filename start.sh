#!/bin/bash
# Start SloughGPT with unified logging
# Shows API, Web, and Core Engine logs in one terminal

cd /Users/mac/sloughGPT

# Default model path (use absolute path)
DEFAULT_MODEL="/Users/mac/models/llama3.2-1b-q8_0.gguf"
export SLOUGHGPT_MODEL_PATH="${SLOUGHGPT_MODEL_PATH:-$DEFAULT_MODEL}"

echo "=== Starting SloughGPT ==="
echo "Model: $SLOUGHGPT_MODEL_PATH"

# Kill existing servers
pkill -f "unified-log-server" 2>/dev/null
pkill -f "python3 apps/api/server" 2>/dev/null
pkill -f "next dev" 2>/dev/null
sleep 1

# Start unified log server
echo "Starting unified log server on port 9999..."
nohup /usr/bin/python3 scripts/unified-log-server.py > /tmp/sloughgpt-log-server.log 2>&1 &
LOG_PID=$!
sleep 1

# Function to send logs to unified server
send_log() {
    local source=$1
    local level=$2
    shift 2
    local message="$@"
    curl -s -X POST http://localhost:9999/log \
        -H "Content-Type: application/json" \
        -d "{\"source\":\"$source\",\"level\":\"$level\",\"message\":\"$message\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)\"}" > /dev/null 2>&1
}

# Start API server with model path
echo "Starting API server..."
send_log "api" "INFO" "🚀 Starting API server on port 8000..."
nohup /bin/bash -c "export SLOUGHGPT_MODEL_PATH='$SLOUGHGPT_MODEL_PATH'; /usr/bin/python3 apps/api/server/simple_server.py" > >(while IFS= read -r line; do send_log "api" "INFO" "$line"; done) 2>&1 &
API_PID=$!
echo "API PID: $API_PID"

# Wait for API
for i in {1..15}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        send_log "api" "INFO" "✅ API server ready on port 8000"
        break
    fi
    sleep 1
done

# Start Web server
echo "Starting Web server..."
send_log "web" "INFO" "🚀 Starting Web server on port 3000..."
cd apps/web
nohup npm run dev > >(while IFS= read -r line; do send_log "web" "INFO" "$line"; done) 2>&1 &
WEB_PID=$!
cd ..
echo "Web PID: $WEB_PID"

# Wait for Web
for i in {1..15}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        send_log "web" "INFO" "✅ Web server ready on port 3000"
        break
    fi
    sleep 1
done

send_log "system" "INFO" "🎉 SloughGPT is running!"
send_log "system" "INFO" "   API:  http://localhost:8000"
send_log "system" "INFO" "   Web:  http://localhost:3000"
send_log "system" "INFO" "   Logs: http://localhost:9999 (unified)"

echo ""
echo "=== SloughGPT is running ==="
echo "API:  http://localhost:8000"
echo "Web:  http://localhost:3000"
echo "Logs: http://localhost:9999"
echo ""
echo "Unified log stream:"
echo "────────────────────────────────────────"
tail -f /tmp/sloughgpt-unified.log &
TAIL_PID=$!
echo ""

# Cleanup function
cleanup() {
    send_log "system" "INFO" "🛑 Shutting down SloughGPT..."
    kill $TAIL_PID 2>/dev/null
    pkill -f "unified-log-server" 2>/dev/null
    pkill -f "python3 apps/api/server" 2>/dev/null
    pkill -f "next dev" 2>/dev/null
    echo "Servers stopped"
}

trap cleanup EXIT INT TERM
wait
