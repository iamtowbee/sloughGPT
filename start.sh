#!/bin/bash

# SloughGPT Startup Script
# Starts both the API server and Web UI

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting SloughGPT...${NC}"

# Function to check if port is in use
check_port() {
    if lsof -i:$1 > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Start API Server
echo -e "${YELLOW}📡 Starting API Server on port 8000...${NC}"
cd "$(dirname "$0")/.."

if check_port 8000; then
    echo -e "${GREEN}✓ API Server already running on port 8000${NC}"
else
    python -m uvicorn domains.ui.api_server:app --reload --port 8000 &
    API_PID=$!
    echo -e "${GREEN}✓ API Server started (PID: $API_PID)${NC}"
fi

# Wait for API to be ready
echo -e "${YELLOW}⏳ Waiting for API to be ready...${NC}"
sleep 3

# Start Web UI
echo -e "${YELLOW}🌐 Starting Web UI on port 3000...${NC}"
cd "$(dirname "$0")"

if check_port 3000; then
    echo -e "${GREEN}✓ Web UI already running on port 3000${NC}"
else
    npm run dev &
    WEB_PID=$!
    echo -e "${GREEN}✓ Web UI started (PID: $WEB_PID)${NC}"
fi

echo ""
echo -e "${GREEN}🎉 SloughGPT is now running!${NC}"
echo -e "${BLUE}   API:   http://localhost:8000${NC}"
echo -e "${BLUE}   UI:    http://localhost:3000${NC}"
echo -e "${BLUE}   Docs:  http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Wait for interrupt
trap "echo -e '\n${YELLOW}🛑 Stopping SloughGPT...${NC}'; kill $API_PID $WEB_PID 2>/dev/null; exit 0" INT TERM

wait
