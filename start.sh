#!/bin/bash

# SloughGPT - One command to start everything

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 Starting SloughGPT...${NC}"

# Check and install dependencies
echo -e "${YELLOW}📦 Checking dependencies...${NC}"
pip install -q torch transformers fastapi uvicorn pydantic 2>/dev/null

# Start API server
echo -e "${YELLOW}🔌 Starting API server (port 8000)...${NC}"
cd "$(dirname "$0")/server"
python3 main.py &
API_PID=$!

cd ..

# Start Web UI
echo -e "${YELLOW}🌐 Starting Web UI (port 3000)...${NC}"
cd web
npm run dev &
WEB_PID=$!

echo ""
echo -e "${GREEN}✅ SloughGPT is ready!${NC}"
echo "   Web UI:  http://localhost:3000"
echo "   API:     http://localhost:8000"
echo "   Docs:    http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

trap "kill $API_PID $WEB_PID 2>/dev/null; exit" INT TERM
wait
