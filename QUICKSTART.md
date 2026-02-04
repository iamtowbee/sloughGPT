# SloughGPT Quick Start Guide

## 🚀 Get Started in 5 Minutes

SloughGPT is an enterprise AI framework that enables you to train, deploy, and manage AI models with production-ready infrastructure.

### Prerequisites

- Python 3.8+
- 8GB+ RAM recommended
- GPU optional but recommended for training

### Installation

```bash
# Clone the repository
git clone https://github.com/sloughgpt/sloughgpt.git
cd sloughgpt

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 sloughgpt.py health
```

## 🎯 Quick Start Commands

### 1. Start API Server

```bash
# Start the main API server
python3 sloughgpt.py serve

# Custom host and port
python3 sloughgpt.py serve --host 0.0.0.0 --port 8080
```

Access API at: http://127.0.0.1:8000

### 2. Launch Admin Dashboard

```bash
# Start the admin dashboard
python3 sloughgpt.py admin

# Custom port
python3 sloughgpt.py admin --port 9000
```

Access Dashboard at: http://127.0.0.1:8080

### 3. Train Your First Model

```bash
# Create model configuration
cat > model_config.json << EOF
{
  "model_name": "my-gpt2",
  "hidden_size": 768,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "vocab_size": 50257
}
EOF

# Train the model
python3 sloughgpt.py train --config model_config.json --data ./data/my_data.json
```

### 4. Check System Health

```bash
# Run comprehensive diagnostics
python3 sloughgpt.py health

# Show package information
python3 sloughgpt.py info

# Show version
python3 sloughgpt.py version
```

## 📊 Quick API Usage

### Generate Text

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain artificial intelligence",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Check API Health

```bash
curl http://127.0.0.1:8000/health
```

### API Documentation

Visit http://127.0.0.1:8000/docs for interactive API documentation.

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/sloughgpt

# Security
JWT_SECRET_KEY=your-secret-key-here

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Cost Management
DEFAULT_MONTHLY_BUDGET=1000
COST_ALERT_THRESHOLD=0.8
```

### Configuration File

Create `sloughgpt_config.json`:

```json
{
  "model_config": {
    "model_name": "gpt2-medium",
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "num_hidden_layers": 24
  },
  "learning_config": {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 10
  },
  "database_config": {
    "database_url": "sqlite:///sloughgpt.db"
  },
  "security_config": {
    "jwt_secret_key": "your-256-bit-secret"
  }
}
```

## 🎯 Next Steps

### 1. Explore the Admin Dashboard
- Monitor system health and performance
- Manage users and API keys
- Track costs and usage analytics
- Configure alerts and notifications

### 2. Train Custom Models
- Prepare your training data
- Configure model parameters
- Monitor training progress
- Deploy trained models

### 3. Build Applications
- Use the REST API for your apps
- Integrate with Python SDK
- Deploy to production with Kubernetes
- Monitor and optimize performance

### 4. Scale to Production
- Set up PostgreSQL and Redis
- Deploy with Docker/Kubernetes
- Configure SSL/TLS certificates
- Implement monitoring and alerting

## 📚 Learn More

- **Full Documentation**: [README.md](README.md)
- **API Reference**: [API.md](API.md)
- **Deployment Guide**: [DEPLOYMENT.md]
- **Enterprise Features**: [ENTERPRISE_SHOWCASE.md](ENTERPRISE_SHOWCASE.md)

## 🆘 Need Help?

- **Documentation**: https://docs.sloughgpt.ai
- **Community**: https://community.sloughgpt.ai
- **Issues**: https://github.com/sloughgpt/sloughgpt/issues
- **Enterprise Support**: enterprise@sloughgpt.ai

---

**🚀 SloughGPT Enterprise AI Framework - Start Building Today!**

*Your journey to enterprise AI begins here.*