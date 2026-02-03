# SloughGPT WebUI

A comprehensive web interface for the SloughGPT project with multiple deployment options.

## 🚀 Quick Start

### Option 1: Enhanced WebUI (Recommended)
```bash
python3 launch_webui.py --mode enhanced
```

### Option 2: Simple WebUI
```bash
python3 launch_webui.py --mode simple
```

### Option 3: Cerebro WebUI (Advanced)
```bash
python3 launch_webui.py --mode cerebro
```

## 📋 Available WebUI Modes

### Enhanced WebUI (`enhanced`)
- **Features**: Modern UI, real-time chat, model selection, conversation history
- **Best for**: Development and testing
- **Port**: 8080 (default)
- **Access**: http://localhost:8080

### Simple WebUI (`simple`)
- **Features**: Basic HTML interface, API endpoints
- **Best for**: Quick testing and demos
- **Port**: 8080 (default)
- **Access**: http://localhost:8080

### Cerebro WebUI (`cerebro`)
- **Features**: Full OpenWebUI integration, advanced features
- **Best for**: Production use
- **Port**: 8080 (default)
- **Access**: http://localhost:8080

## ⚙️ Configuration

### Command Line Options
```bash
python3 launch_webui.py --mode enhanced --port 8080 --host 0.0.0.0
```

- `--mode`: WebUI mode (`simple`, `enhanced`, `cerebro`)
- `--port`: Port number (default: 8080)
- `--host`: Host address (default: 0.0.0.0)

### Environment Variables
- `PORT`: Server port
- `HOST`: Server host
- `PYTHONPATH`: Python module path

## 📡 API Endpoints

### Common Endpoints
- `GET /api/health` - Health check
- `GET /api/models` - Available models
- `GET /api/status` - System status
- `GET /docs` - API documentation

### Enhanced WebUI Endpoints
- `POST /api/chat` - Chat with AI
- `GET /api/conversations` - List conversations
- `GET /api/conversations/{id}` - Get conversation

## 🛠️ Development

### File Structure
```
sloughgpt/
├── launch_webui.py          # Unified launcher
├── simple_webui.py          # Simple web interface
├── enhanced_webui.py        # Enhanced web interface
├── webui.py                 # Original webui launcher
└── packages/apps/apps/cerebro/  # Cerebro WebUI
    └── open_webui/
        ├── main.py
        └── config.py
```

### Dependencies
- FastAPI
- Uvicorn
- Pydantic
- Python 3.9+

## 🔧 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   export PYTHONPATH=/path/to/sloughgpt:$PYTHONPATH
   ```

2. **Port Already in Use**
   ```bash
   python3 launch_webui.py --port 8081
   ```

3. **Missing Dependencies**
   ```bash
   pip3 install fastapi uvicorn pydantic
   ```

### Health Check
```bash
curl http://localhost:8080/api/health
```

## 📊 Features Comparison

| Feature | Simple | Enhanced | Cerebro |
|---------|--------|----------|---------|
| Modern UI | ❌ | ✅ | ✅ |
| Real-time Chat | ❌ | ✅ | ✅ |
| Model Selection | ❌ | ✅ | ✅ |
| Conversation History | ❌ | ✅ | ✅ |
| API Documentation | ✅ | ✅ | ✅ |
| File Upload | ❌ | ❌ | ✅ |
| User Management | ❌ | ❌ | ✅ |
| Database Support | ❌ | ❌ | ✅ |

## 🚀 Production Deployment

### Docker (Coming Soon)
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
CMD ["python3", "launch_webui.py", "--mode", "cerebro"]
```

### Systemd Service
```ini
[Unit]
Description=SloughGPT WebUI
After=network.target

[Service]
Type=simple
User=webui
WorkingDirectory=/opt/sloughgpt
ExecStart=/usr/bin/python3 launch_webui.py --mode cerebro
Restart=always

[Install]
WantedBy=multi-user.target
```

## 📝 License

This project is part of SloughGPT and follows the same licensing terms.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `python3 launch_webui.py --mode enhanced`
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Open an issue on GitHub
- Join the community discussions