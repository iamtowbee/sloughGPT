#!/bin/bash
set -e

echo "🚀 Setting up SloughGPT Development Environment"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📌 Python version: $python_version"

# Check Node.js version
node_version=$(node --version 2>&1)
echo "📌 Node.js version: $node_version"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
cd server
pip install -r requirements.txt
cd ..

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
cd web
npm install
cd ..

# Create .env files if they don't exist
echo "📝 Setting up environment files..."

if [ ! -f "server/.env" ]; then
    cp server/.env.example server/.env 2>/dev/null || echo "API_URL=http://localhost:8000" > server/.env
    echo "Created server/.env"
fi

if [ ! -f "web/.env.local" ]; then
    cp web/.env.example web/.env.local 2>/dev/null || echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > web/.env.local
    echo "Created web/.env.local"
fi

echo "✅ Setup complete!"
echo ""
echo "To start development:"
echo "  Terminal 1: cd server && python main.py"
echo "  Terminal 2: cd web && npm run dev"
echo ""
echo "Or use Docker:"
echo "  docker-compose up"
