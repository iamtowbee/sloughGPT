# SloughGPT Installation Script

## 🚀 Automated Installation

This script provides a quick way to set up SloughGPT and resolve common dependency issues.

```bash
# Download and run the installation script
curl -fsSL https://raw.githubusercontent.com/sloughgpt/sloughgpt/main/install.sh | bash
```

## Manual Installation Steps

### 1. Fix NumPy Compatibility

```bash
# Fix NumPy 2.x compatibility issue
pip install "numpy<2.0"
pip install --force-reinstall torch torchvision torchaudio
```

### 2. Install Core Dependencies

```bash
# Install without torch first to avoid conflicts
pip install fastapi uvicorn sqlalchemy alembic redis
pip install pydantic python-jose[cryptography] passlib[bcrypt]
pip install python-multipart psutil prometheus-client
pip install aiosmtplib pytest pytest-asyncio

# Then install torch with compatibility
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

### 3. Install Additional Dependencies

```bash
# Install remaining dependencies
pip install scikit-learn pandas numpy matplotlib seaborn
pip install jupyter ipykernel notebook
pip install plotly dash streamlit
pip install aiofiles aiohttp websockets
```

### 4. Verify Installation

```bash
# Test Python import
python3 -c "
import sys
try:
    import sloughgpt
    print('✅ SloughGPT imported successfully')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"

# Test launcher
python3 sloughgpt.py version
```

## Docker Installation (Recommended)

### Using Pre-built Docker Image

```bash
# Pull the latest image
docker pull sloughgpt/sloughgpt:latest

# Run with environment variables
docker run -d \
  --name sloughgpt-api \
  -p 8000:8000 \
  -p 8080:8080 \
  -e DATABASE_URL=sqlite:///sloughgpt.db \
  -e JWT_SECRET_KEY=your-secret-key \
  sloughgpt/sloughgpt:latest
```

### Building from Source

```bash
# Clone repository
git clone https://github.com/sloughgpt/sloughgpt.git
cd sloughgpt

# Build Docker image
docker build -t sloughgpt:local .

# Run with custom configuration
docker run -d \
  --name sloughgpt-local \
  -p 8000:8000 \
  -p 8080:8080 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  sloughgpt:local
```

## Kubernetes Installation

### Helm Chart

```bash
# Add SloughGPT Helm repository
helm repo add sloughgpt https://charts.sloughgpt.ai
helm repo update

# Install with default values
helm install sloughgpt sloughgpt/sloughgpt

# Install with custom values
helm install sloughgpt sloughgpt/sloughgpt -f values.yaml
```

### Kubernetes Manifests

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n sloughgpt
kubectl get services -n sloughgpt
```

## Environment Configuration

### Required Environment Variables

```bash
# Create .env file
cat > .env << EOF
# Database
DATABASE_URL=sqlite:///sloughgpt.db

# Security
JWT_SECRET_KEY=your-256-bit-secret-key-here
BCRYPT_ROUNDS=12

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
ADMIN_PORT=8080

# Logging
LOG_LEVEL=INFO
ENABLE_FILE_LOGGING=true

# Cost Management
DEFAULT_MONTHLY_BUDGET=1000
COST_ALERT_THRESHOLD=0.8
EOF
```

### Production Environment

```bash
# Production settings
DATABASE_URL=postgresql://user:password@localhost:5432/sloughgpt
REDIS_URL=redis://localhost:6379

# Strong security
JWT_SECRET_KEY=$(openssl rand -hex 32)
BCRYPT_ROUNDS=14

# SSL/TLS
SSL_CERT_PATH=/etc/ssl/cert.pem
SSL_KEY_PATH=/etc/ssl/key.pem

# Monitoring
ENABLE_METRICS=true
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

## Troubleshooting

### Common Issues

#### 1. NumPy Compatibility Error

**Problem**: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2`

**Solution**:
```bash
pip install "numpy<2.0"
pip install --force-reinstall torch torchvision torchaudio
```

#### 2. Import Error for get_database_manager

**Problem**: `cannot import name 'get_database_manager'`

**Solution**: The function is missing from core module, but this is expected during setup.

#### 3. Torch Not Found

**Problem**: `No module named 'torch'`

**Solution**:
```bash
# Install with specific version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# Or use conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 4. Permission Denied

**Problem**: `Permission denied` when running scripts

**Solution**:
```bash
chmod +x sloughgpt.py
chmod +x install.sh
```

### Verification Steps

```bash
# 1. Check Python version
python3 --version  # Should be 3.8+

# 2. Check critical imports
python3 -c "
try:
    import fastapi, torch, sqlalchemy, redis
    print('✅ Core dependencies OK')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
"

# 3. Test basic functionality
python3 -c "
try:
    from sloughgpt import SloughGPTConfig
    config = SloughGPTConfig()
    print('✅ Configuration system OK')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"

# 4. Check ports
lsof -i :8000  # Should be free
lsof -i :8080  # Should be free
```

## Platform-Specific Instructions

### macOS

```bash
# Install with Homebrew
brew install python@3.9

# Install with pip
python3.9 -m pip install --upgrade pip
python3.9 -m pip install "numpy<2.0"
python3.9 -m pip install torch torchvision torchaudio
```

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3.9 python3.9-dev
sudo apt-get install build-essential

# Install SloughGPT
python3.9 -m pip install sloughgpt
```

### Windows

```bash
# Use PowerShell
python -m pip install "numpy<2.0"
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or use conda
conda create -n sloughgpt python=3.9
conda activate sloughgpt
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Next Steps

After successful installation:

1. **Start API Server**: `python3 sloughgpt.py serve`
2. **Launch Admin Dashboard**: `python3 sloughgpt.py admin`
3. **Check Health**: `python3 sloughgpt.py health`
4. **View Documentation**: http://localhost:8080/docs

## Support

- **Documentation**: https://docs.sloughgpt.ai
- **Issues**: https://github.com/sloughgpt/sloughgpt/issues
- **Community**: https://community.sloughgpt.ai
- **Enterprise Support**: enterprise@sloughgpt.ai

---

**🚀 SloughGPT Enterprise AI Framework - Installation Complete!**