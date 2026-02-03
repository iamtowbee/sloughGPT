#!/usr/bin/env python3
"""
Simple FastAPI webui to test basic functionality
"""

import os
import sys
from pathlib import Path

# Add parent directories to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="SloughGPT WebUI", description="Simple web interface for SloughGPT")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SloughGPT WebUI</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .status { padding: 20px; margin: 20px 0; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
            .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            .btn:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 SloughGPT WebUI</h1>
            
            <div class="status success">
                <strong>✅ Status:</strong> WebUI is running successfully!
            </div>
            
            <div class="status info">
                <strong>📊 Info:</strong> This is a basic web interface for the SloughGPT project.
            </div>
            
            <div class="section">
                <h3>🔧 Available Features</h3>
                <ul>
                    <li>✅ FastAPI backend server</li>
                    <li>✅ CORS enabled</li>
                    <li>✅ Static file serving</li>
                    <li>✅ HTML interface</li>
                    <li>✅ Health check endpoint</li>
                    <li>✅ Model listing endpoint</li>
                    <li>✅ Mock chat endpoint</li>
                    <li>🔄 OpenAI-compatible API endpoints (coming)</li>
                    <li>🔄 Model management interface (coming)</li>
                    <li>🔄 Real AI integration (coming)</li>
                </ul>
            </div>
            
            <div class="section">
                <h3>💬 Quick Chat Test</h3>
                <div style="margin: 10px 0;">
                    <input type="text" id="chatInput" placeholder="Type a message..." style="width: 70%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                    <button class="btn" onclick="sendChat()">Send</button>
                </div>
                <div id="chatResponse" style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 4px; min-height: 50px; display: none;"></div>
            </div>
            
            <div class="section">
                <h3>📡 API Endpoints</h3>
                <ul>
                    <li><a href="/api/health">GET /api/health</a> - Health check</li>
                    <li><a href="/api/models">GET /api/models</a> - Available models</li>
                    <li><a href="/api/status">GET /api/status</a> - System status</li>
                    <li><a href="/api/chat">GET /api/chat</a> - Mock chat endpoint</li>
                    <li><a href="/docs">GET /docs</a> - API documentation</li>
                </ul>
            </div>
            
            <div class="section">
                <h3>🚀 Quick Actions</h3>
                <button class="btn" onclick="location.href='/api/health'">Check Health</button>
                <button class="btn" onclick="location.href='/docs'">View API Docs</button>
                <button class="btn" onclick="location.reload()">Refresh Page</button>
        </div>
        
        <script>
            async function sendChat() {
                const input = document.getElementById('chatInput');
                const response = document.getElementById('chatResponse');
                const message = input.value.trim();
                
                if (!message) return;
                
                response.style.display = 'block';
                response.innerHTML = '<em>Thinking...</em>';
                
                try {
                    const res = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: message})
                    });
                    
                    const data = await res.json();
                    response.innerHTML = `
                        <strong>Response:</strong> ${data.response}<br>
                        <small>Model: ${data.model} | Time: ${data.timestamp}</small>
                    `;
                    input.value = '';
                } catch (error) {
                    response.innerHTML = '<em>Error: Could not get response</em>';
                }
            }
            
            // Allow Enter key to send message
            document.getElementById('chatInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') sendChat();
            });
        </script>
    </body>
    </html>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "SloughGPT WebUI is running",
        "version": "0.1.0"
    }

@app.get("/api/models")
async def list_models():
    """List available models"""
    # This would integrate with the actual model system
    return {
        "models": [
            {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "provider": "openai",
                "status": "available"
            },
            {
                "id": "claude-3-sonnet",
                "name": "Claude 3 Sonnet", 
                "provider": "anthropic",
                "status": "available"
            }
        ],
        "total": 2
    }

@app.get("/api/status")
async def system_status():
    """System status information"""
    return {
        "webui": {
            "status": "running",
            "version": "0.1.0"
        },
        "features": {
            "api": "enabled",
            "static_files": "enabled", 
            "cors": "enabled",
            "documentation": "enabled",
            "chat": "coming soon",
            "model_management": "coming soon",
            "file_upload": "coming soon"
        },
        "uptime": "just started",
        "project_files": {
            "webui_py": "✅ Available",
            "simple_webui_py": "✅ Running",
            "cerebro_webui": "⚠️ Complex dependencies",
            "api_server": "✅ Available"
        }
    }

@app.get("/api/chat")
async def chat_endpoint(message: str, model: str = "gpt-3.5-turbo"):
    """Simple chat endpoint (mock)"""
    # Mock response - would integrate with actual AI models
    return {
        "message": f"You said: {message}",
        "response": f"This is a mock response from {model}. Real AI integration coming soon!",
        "model": model,
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.post("/api/chat")
async def chat_post_endpoint(data: dict):
    """POST chat endpoint"""
    message = data.get("message", "")
    model = data.get("model", "gpt-3.5-turbo")
    
    return {
        "message": message,
        "response": f"This is a mock response from {model}. Real AI integration coming soon!",
        "model": model,
        "timestamp": "2024-01-01T00:00:00Z"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting SloughGPT WebUI...")
    print(f"📍 Server will be available at: http://{host}:{port}")
    print(f"📚 API docs at: http://{host}:{port}/docs")
    print(f"❤️  Health check: http://{host}:{port}/api/health")
    
    uvicorn.run(app, host=host, port=port)