#!/bin/bash
# Quick verification script for SloughGPT

echo "🔍 Verifying SloughGPT installation..."

# Check Python
echo "Checking Python..."
python --version

# Check Node
echo "Checking Node..."
node --version

# Check if required files exist
echo ""
echo "Checking required files..."

files=(
    "domains/ui/api_server.py"
    "web/package.json"
    "web/src/App.tsx"
    "web/src/main.tsx"
    "web/src/store/index.ts"
    "web/src/utils/api.ts"
    "web/src/components/Chat.tsx"
    "web/src/components/Datasets.tsx"
    "web/src/components/Models.tsx"
    "web/src/components/Training.tsx"
    "web/src/components/Monitoring.tsx"
    "web/src/components/Home.tsx"
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
    echo "✅ All files present!"
    echo ""
    echo "To start SloughGPT:"
    echo "1. Terminal 1: python -m uvicorn:app --reload --port 8000"
    echo domains.ui.api_server "2. Terminal 2: cd web && npm run dev"
    echo ""
    echo "Then open http://localhost:3000"
else
    echo ""
    echo "❌ Some files are missing!"
    exit 1
fi
