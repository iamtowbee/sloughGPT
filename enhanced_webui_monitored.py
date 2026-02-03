#!/usr/bin/env python3
"""
Enhanced WebUI with more features for SloughGPT
Including monitoring and logging capabilities
"""

import os
import sys
import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

# Add parent directories to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('sloughgpt_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('sloughgpt_request_duration_seconds', 'Request duration')
ACTIVE_CONVERSATIONS = Gauge('sloughgpt_active_conversations', 'Number of active conversations')
API_MODELS_AVAILABLE = Gauge('sloughgpt_available_models', 'Number of available models')

app = FastAPI(title="SloughGPT Enhanced WebUI", description="Enhanced web interface for SloughGPT")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class ChatMessage(BaseModel):
    message: str
    model: str = "gpt-3.5-turbo"
    conversation_id: Optional[str] = None

class ModelConfig(BaseModel):
    id: str
    name: str
    provider: str
    status: str = "available"
    description: Optional[str] = None

# In-memory storage (would use database in production)
conversations: Dict[str, List[Dict]] = {}
models: List[ModelConfig] = [
    ModelConfig(
        id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        provider="OpenAI",
        status="available",
        description="Fast and efficient for most tasks"
    ),
    ModelConfig(
        id="gpt-4",
        name="GPT-4",
        provider="OpenAI", 
        status="available",
        description="Most capable model for complex tasks"
    ),
    ModelConfig(
        id="claude-3-sonnet",
        name="Claude 3 Sonnet",
        provider="Anthropic",
        status="available",
        description="Balanced performance and safety"
    ),
    ModelConfig(
        id="llama-2-7b",
        name="Llama 2 7B",
        provider="Meta",
        status="local",
        description="Open source model for local deployment"
    )
]

# Update metrics
API_MODELS_AVAILABLE.set(len(models))

# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    process_time = time.time() - start_time
    REQUEST_DURATION.observe(process_time)
    
    # Log request
    logger.info(
        f"Method: {request.method} | Path: {request.url.path} | "
        f"Status: {response.status_code} | Duration: {process_time:.3f}s"
    )
    
    # Update request counter
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        # Update active conversations metric
        ACTIVE_CONVERSATIONS.set(len(conversations))
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return Response("Error generating metrics", status_code=500)

@app.get("/logs")
async def get_logs(lines: int = 100):
    """Get application logs (for monitoring)"""
    try:
        log_file = Path('/app/logs/app.log')
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_lines = f.readlines()[-lines:]
            return {"logs": log_lines, "lines": len(log_lines)}
        return {"logs": [], "lines": 0}
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return Response("Error reading logs", status_code=500)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Enhanced HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SloughGPT Enhanced WebUI</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                min-height: 100vh; 
            }
            .header { 
                background: rgba(255,255,255,0.1); 
                backdrop-filter: blur(10px); 
                padding: 20px; 
                text-align: center; 
                color: white; 
                border-bottom: 1px solid rgba(255,255,255,0.1); 
            }
            .container { 
                max-width: 1200px; 
                margin: 20px auto; 
                display: grid; 
                grid-template-columns: 300px 1fr; 
                gap: 20px; 
                padding: 0 20px; 
            }
            .sidebar { 
                background: rgba(255,255,255,0.95); 
                border-radius: 15px; 
                padding: 20px; 
                height: fit-content; 
                box-shadow: 0 8px 32px rgba(0,0,0,0.1); 
            }
            .main { 
                background: rgba(255,255,255,0.95); 
                border-radius: 15px; 
                padding: 30px; 
                box-shadow: 0 8px 32px rgba(0,0,0,0.1); 
            }
            .chat-container { height: 500px; display: flex; flex-direction: column; }
            .chat-messages { 
                flex: 1; 
                overflow-y: auto; 
                padding: 20px; 
                background: #f8f9fa; 
                border-radius: 10px; 
                margin-bottom: 20px; 
            }
            .chat-input { display: flex; gap: 10px; }
            .chat-input input { 
                flex: 1; 
                padding: 12px; 
                border: 2px solid #e9ecef; 
                border-radius: 8px; 
                font-size: 14px; 
            }
            .chat-input button { 
                padding: 12px 24px; 
                background: #007bff; 
                color: white; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer; 
                font-weight: 600; 
            }
            .chat-input button:hover { background: #0056b3; }
            .message { margin: 10px 0; padding: 12px; border-radius: 8px; max-width: 80%; }
            .user-message { background: #007bff; color: white; margin-left: auto; }
            .ai-message { background: #e9ecef; color: #333; }
            .model-selector { margin: 20px 0; }
            .model-selector select { 
                width: 100%; 
                padding: 10px; 
                border: 2px solid #e9ecef; 
                border-radius: 8px; 
                font-size: 14px; 
            }
            .status-card { 
                background: #d4edda; 
                color: #155724; 
                padding: 15px; 
                border-radius: 8px; 
                margin: 10px 0; 
                border: 1px solid #c3e6cb; 
            }
            .feature-list { list-style: none; padding: 0; }
            .feature-list li { padding: 8px 0; border-bottom: 1px solid #eee; }
            .feature-list li:before { content: "✅ "; margin-right: 8px; }
            .btn { 
                background: #007bff; 
                color: white; 
                padding: 10px 20px; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer; 
                margin: 5px; 
                text-decoration: none; 
                display: inline-block; 
                font-weight: 600; 
            }
            .btn:hover { background: #0056b3; }
            .btn-secondary { background: #6c757d; }
            .btn-secondary:hover { background: #545b62; }
            h1, h2, h3 { color: #333; margin-top: 0; }
            .header h1 { color: white; font-size: 2.5em; margin: 0; font-weight: 700; }
            .header p { color: rgba(255,255,255,0.8); margin: 10px 0 0 0; font-size: 1.1em; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 SloughGPT Enhanced WebUI</h1>
            <p>Advanced AI interface with real-time chat and model management</p>
        </div>
        
        <div class="container">
            <div class="sidebar">
                <h3>🤖 Model Selection</h3>
                <div class="model-selector sloughgpt-model-selector">
                    <select id="modelSelect" data-cy="model-selector" aria-label="Select a model">
                        <option value="gpt-3.5-turbo">GPT-3.5 Turbo (Fast)</option>
                        <option value="gpt-4">GPT-4 (Capable)</option>
                        <option value="claude-3-sonnet">Claude 3 Sonnet (Balanced)</option>
                        <option value="llama-2-7b">Llama 2 7B (Local)</option>
                    </select>
                </div>
                
                <h3>📊 System Status</h3>
                <div class="status-card">
                    <strong>✅ WebUI Running</strong><br>
                    <small>Port: 8080 | Version: 0.2.0</small>
                </div>
                
                <h3>🔧 Features</h3>
                <ul class="feature-list">
                    <li>Real-time chat interface</li>
                    <li>Multiple model support</li>
                    <li>Conversation history</li>
                    <li>Health monitoring</li>
                    <li>API documentation</li>
                    <li>CORS enabled</li>
                </ul>
                
                <h3>🔗 Quick Links</h3>
                <a href="/api/health" class="btn">Health Check</a>
                <a href="/api/models" class="btn">Models API</a>
                <a href="/docs" class="btn btn-secondary">API Docs</a>
                <a href="/metrics" class="btn btn-secondary">Metrics</a>
            </div>
            
            <div class="main">
                <h2>💬 Chat Interface</h2>
                <div class="chat-container">
                    <div class="chat-messages" id="chatMessages">
                        <div class="message ai-message">
                            👋 Welcome to SloughGPT Enhanced WebUI! Select a model and start.
                            This is a mock interface - real AI integration coming soon!
                        </div>
                    </div>
                    <div class="chat-input">
                        <input type="text" id="chatInput" data-cy="chat-input" 
                               placeholder="Type here..." onkeypress="handleKeyPress(event)">
                        <button type="submit" onclick="sendMessage()" data-cy="send-button">
                            Send
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let conversationId = null;
            
            async function sendMessage() {
                const input = document.getElementById('chatInput');
                const messages = document.getElementById('chatMessages');
                const modelSelect = document.getElementById('modelSelect');
                const message = input.value.trim();
                
                if (!message) return;
                
                // Add user message
                const userMsg = document.createElement('div');
                userMsg.className = 'message user-message';
                userMsg.textContent = message;
                messages.appendChild(userMsg);
                
                // Clear input
                input.value = '';
                
                // Add thinking message
                const thinkingMsg = document.createElement('div');
                thinkingMsg.className = 'message ai-message';
                thinkingMsg.innerHTML = '🤔 Thinking...';
                messages.appendChild(thinkingMsg);
                
                // Scroll to bottom
                messages.scrollTop = messages.scrollHeight;
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            message: message,
                            model: modelSelect.value,
                            conversation_id: conversationId
                        })
                    });
                    
                    const data = await response.json();
                    
                    // Update thinking message with response
                    thinkingMsg.innerHTML = `
                        <strong>${data.model}:</strong> ${data.response}<br>
                        <small style="opacity: 0.7;">${data.timestamp}</small>
                    `;
                    
                    // Update conversation ID
                    if (data.conversation_id) {
                        conversationId = data.conversation_id;
                    }
                    
                } catch (error) {
                    thinkingMsg.innerHTML = '❌ Error: Could not get response from server';
                }
                
                // Scroll to bottom
                messages.scrollTop = messages.scrollHeight;
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
            
            // Focus input on load
            document.getElementById('chatInput').focus();
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "SloughGPT Enhanced WebUI is running",
        "version": "0.2.0",
        "features": {
            "chat": "enabled",
            "models": "enabled",
            "conversations": "enabled",
            "cors": "enabled"
        },
        "monitoring": {
            "metrics": "enabled",
            "logging": "enabled",
            "prometheus": "enabled"
        }
    }

@app.get("/api/models")
async def list_models():
    """List available models"""
    return {
        "models": [model.dict() for model in models],
        "total": len(models),
        "default": "gpt-3.5-turbo"
    }

@app.get("/api/status")
async def system_status():
    """System status information"""
    return {
        "webui": {
            "status": "running",
            "version": "0.2.0",
            "uptime": "just started"
        },
        "features": {
            "api": "enabled",
            "chat": "enabled",
            "models": "enabled", 
            "conversations": "enabled",
            "cors": "enabled",
            "documentation": "enabled",
            "monitoring": "enabled"
        },
        "statistics": {
            "total_models": len(models),
            "active_conversations": len(conversations),
            "available_models": len([m for m in models if m.status == "available"])
        }
    }

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage):
    """Enhanced chat endpoint with conversation support"""
    # Log chat request
    logger.info(f"Chat request - Model: {message.model}, Message length: {len(message.message)}")
    
    # Generate or use conversation ID
    conv_id = message.conversation_id or f"conv_{len(conversations) + 1}"
    
    # Initialize conversation if new
    if conv_id not in conversations:
        conversations[conv_id] = []
        logger.info(f"Created new conversation: {conv_id}")
    
    # Add user message to conversation
    conversations[conv_id].append({
        "role": "user",
        "content": message.message,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # Mock AI response (would integrate with real AI models)
    ai_response = (
        f"This is an enhanced mock response from {message.model}. "
        f"Your message was: '{message.message}'. Conversation ID: {conv_id}"
    )
    
    # Add AI response to conversation
    conversations[conv_id].append({
        "role": "assistant", 
        "content": ai_response,
        "model": message.model,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    logger.info(f"Chat response generated - Conversation: {conv_id}")
    
    return {
        "message": message.message,
        "response": ai_response,
        "model": message.model,
        "conversation_id": conv_id,
        "timestamp": datetime.utcnow().isoformat(),
        "conversation_length": len(conversations[conv_id])
    }

@app.get("/api/conversations")
async def list_conversations():
    """List all conversations"""
    return {
        "conversations": [
            {
                "id": conv_id,
                "message_count": len(messages),
                "last_message": messages[-1] if messages else None
            }
            for conv_id, messages in conversations.items()
        ],
        "total": len(conversations)
    }

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get specific conversation"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id],
        "message_count": len(conversations[conversation_id])
    }

@app.get("/api/models/sloughgpt")
async def sloughgpt_models():
    """SloughGPT specific models endpoint"""
    return {
        "models": [model.dict() for model in models],
        "total": len(models),
        "default": "gpt-3.5-turbo",
        "provider": "SloughGPT Enhanced WebUI"
    }

@app.get("/api/status/sloughgpt")
async def sloughgpt_status():
    """SloughGPT specific status endpoint"""
    return {
        "status": "operational",
        "webui": {
            "status": "running",
            "version": "0.2.0",
            "uptime": "just started"
        },
        "models": {
            "total": len(models),
            "available": len([m for m in models if m.status == "available"]),
            "local": len([m for m in models if m.status == "local"])
        },
        "features": {
            "api": "enabled",
            "chat": "enabled", 
            "models": "enabled",
            "conversations": "enabled",
            "cors": "enabled",
            "documentation": "enabled",
            "sloughgpt_branding": "enabled",
            "monitoring": "enabled"
        }
    }

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    log_dir = Path('/app/logs')
    log_dir.mkdir(exist_ok=True)
    
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info("🚀 Starting SloughGPT Enhanced WebUI...")
    logger.info(f"📍 Server: http://{host}:{port}")
    logger.info(f"📚 API Docs: http://{host}:{port}/docs")
    logger.info(f"❤️  Health: http://{host}:{port}/api/health")
    logger.info(f"🤖 Models: http://{host}:{port}/api/models")
    logger.info(f"💬 Chat: http://{host}:{port}/")
    logger.info(f"📊 Metrics: http://{host}:{port}/metrics")
    
    uvicorn.run(app, host=host, port=port)