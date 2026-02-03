#!/usr/bin/env python3
"""
SloughGPT Production Deployment
API Server, Web UI, and Real-world Integration
"""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import time
import json
import uuid
from typing import Optional, List, Dict, Any
import aiofiles
from pathlib import Path

from sloughgpt_integrated import SloughGPTIntegrated, SystemMode
from sloughgpt_neural_network import ModelConfig

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize SloughGPT
config = ModelConfig(vocab_size=10000, d_model=512, n_heads=8, n_layers=6)
sloughgpt = SloughGPTIntegrated(config)

# Initialize FastAPI
app = FastAPI(
    title="SloughGPT Production API",
    description="Custom GPT system with continuous learning and cognitive capabilities",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class PromptRequest(BaseModel):
    prompt: str
    mode: str = "adaptive"
    context: Optional[Dict[str, Any]] = None

class FeedbackRequest(BaseModel):
    response_id: str
    rating: float
    helpfulness: Optional[float] = None
    accuracy: Optional[float] = None
    creativity: Optional[float] = None
    coherence: Optional[float] = None
    thinking_mode: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: float

class ChatSession(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    created_at: float
    last_activity: float

# In-memory storage (would use database in production)
chat_sessions: {}
active_responses: {}

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Main Web UI"""
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SloughGPT - Custom AI System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .title {
            color: white;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            color: rgba(255,255,255,0.9);
            font-size: 1.1em;
            margin-bottom: 5px;
        }
        
        .status {
            background: rgba(255,255,255,0.1);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            display: inline-block;
        }
        
        .main-content {
            flex: 1;
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .chat-panel {
            flex: 1;
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            min-height: 300px;
            max-height: 400px;
        }
        
        .chat-input {
            padding: 20px;
            border-top: 1px solid rgba(0,0,0,0.1);
        }
        
        .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        #prompt {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #667eea;
            border-radius: 8px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        #prompt:focus {
            border-color: #764ba2;
            box-shadow: 0 0 0 3px rgba(118,75,162,0.1);
        }
        
        .mode-select {
            padding: 12px 16px;
            border: 2px solid #667eea;
            border-radius: 8px;
            background: white;
            font-size: 14px;
            cursor: pointer;
        }
        
        #send {
            padding: 12px 24px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        #send:hover {
            background: #764ba2;
        }
        
        .sidebar {
            width: 250px;
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            padding: 20px;
        }
        
        .sidebar h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        
        .system-info {
            font-size: 0.9em;
            line-height: 1.6;
        }
        
        .info-item {
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(0,0,0,0.1);
        }
        
        .info-label {
            color: #666;
            font-weight: bold;
        }
        
        .info-value {
            color: #333;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 12px;
            position: relative;
        }
        
        .message.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 20%;
            text-align: right;
        }
        
        .message.assistant {
            background: rgba(102,126,234,0.1);
            border: 1px solid rgba(102,126,234,0.2);
            margin-right: 20%;
        }
        
        .message-content {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .message-time {
            font-size: 0.7em;
            opacity: 0.7;
            margin-top: 5px;
        }
        
        .feedback-section {
            margin-top: 5px;
            padding: 8px;
            background: rgba(0,0,0,0.05);
            border-radius: 8px;
            font-size: 0.8em;
        }
        
        .feedback-btn {
            margin: 0 2px;
            padding: 4px 8px;
            border: 1px solid #667eea;
            border-radius: 4px;
            background: white;
            color: #667eea;
            cursor: pointer;
            font-size: 0.8em;
        }
        
        .feedback-btn:hover {
            background: #667eea;
            color: white;
        }
        
        .typing-indicator {
            color: #667eea;
            font-style: italic;
            padding: 10px 20px;
            display: none;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .thinking {
            animation: pulse 1.5s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1 class="title">SloughGPT</h1>
            <div class="subtitle">Custom AI System with Continuous Learning</div>
            <div class="status" id="systemStatus">Initializing...</div>
        </header>
        
        <div class="main-content">
            <div class="chat-panel">
                <div class="chat-messages" id="chatMessages">
                    <div class="message assistant">
                        <div class="message-content">Hello! I'm SloughGPT, your custom AI assistant with reasoning, creativity, and continuous learning capabilities. How can I help you today?</div>
                        <div class="message-time">Just now</div>
                    </div>
                </div>
                
                <div class="chat-input">
                    <div class="input-group">
                        <input type="text" id="prompt" placeholder="Ask me anything..." autofocus>
                        <select id="mode" class="mode-select">
                            <option value="adaptive">Adaptive</option>
                            <option value="cognitive">Cognitive</option>
                            <option value="generation">Generation</option>
                            <option value="learning">Learning</option>
                        </select>
                        <button id="send" onclick="sendPrompt()">Send</button>
                    </div>
                    <div class="typing-indicator" id="typingIndicator">🧠 SloughGPT is thinking...</div>
                </div>
            </div>
            
            <div class="sidebar">
                <h3>🧠 System Status</h3>
                <div class="system-info" id="systemInfo">
                    <div class="info-item">
                        <div class="info-label">Status:</div>
                        <div class="info-value" id="statusInfo">Active</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Interactions:</div>
                        <div class="info-value" id="interactionsInfo">0</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Avg Confidence:</div>
                        <div class="info-value" id="confidenceInfo">0.00</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Learning Rate:</div>
                        <div class="info-value" id="learningRateInfo">0.0001</div>
                    </div>
                </div>
                
                <h3>🎓 Learning</h3>
                <div class="system-info">
                    <div class="info-item">
                        <div class="info-label">Total Experiences:</div>
                        <div class="info-value" id="experiencesInfo">0</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Success Rate:</div>
                        <div class="info-value" id="successRateInfo">0.00</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentSessionId = 'session_' + Date.now();
        let messages = [];
        let activeResponse = null;

        async function updateSystemStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                
                document.getElementById('statusInfo').textContent = 'Active';
                document.getElementById('interactionsInfo').textContent = status.performance_metrics.total_interactions;
                document.getElementById('confidenceInfo').textContent = status.performance_metrics.average_confidence.toFixed(3);
                document.getElementById('learningRateInfo').textContent = status.performance_metrics.learning_rate.toFixed(6);
                document.getElementById('experiencesInfo').textContent = status.learning_system.experience_buffer_size;
                document.getElementById('successRateInfo').textContent = (status.learning_system.learning_metrics.success_rate * 100).toFixed(1) + '%';
            } catch (error) {
                console.error('Error updating status:', error);
            }
        }

        async function sendPrompt() {
            const prompt = document.getElementById('prompt').value.trim();
            const mode = document.getElementById('mode').value;
            
            if (!prompt) return;
            
            // Add user message
            addMessage('user', prompt);
            document.getElementById('prompt').value = '';
            
            // Show typing indicator
            document.getElementById('typingIndicator').style.display = 'block';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        mode: mode
                    })
                });
                
                const data = await response.json();
                
                // Hide typing indicator
                document.getElementById('typingIndicator').style.display = 'none';
                
                // Add assistant response
                addMessage('assistant', data.neural_response, data);
                activeResponse = data;
                
                // Update status
                await updateSystemStatus();
                
            } catch (error) {
                document.getElementById('typingIndicator').style.display = 'none';
                addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
                console.error('Error:', error);
            }
        }

        function addMessage(role, content, metadata = null) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            let messageHTML = `<div class="message-content">${content}</div>`;
            messageHTML += `<div class="message-time">${new Date().toLocaleTimeString()}</div>`;
            
            // Add feedback for assistant messages
            if (role === 'assistant' && metadata) {
                messageHTML += `
                    <div class="feedback-section">
                        Was this helpful?
                        <button class="feedback-btn" onclick="sendFeedback(${metadata.response_id}, 1)">👍</button>
                        <button class="feedback-btn" onclick="sendFeedback(${metadata.response_id}, 0)">👎</button>
                        <button class="feedback-btn" onclick="sendFeedback(${metadata.response_id}, 0.5)">🤔</button>
                    </div>
                `;
            }
            
            messageDiv.innerHTML = messageHTML;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        async function sendFeedback(responseId, rating) {
            try {
                await fetch('/api/feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        response_id: responseId,
                        rating: rating
                    })
                });
                
                // Update the feedback section
                console.log('Feedback sent for response:', responseId);
                
                // Update system status after feedback
                setTimeout(updateSystemStatus, 1000);
                
            } catch (error) {
                console.error('Error sending feedback:', error);
            }
        }

        // Enter key to send
        document.getElementById('prompt').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendPrompt();
            }
        });

        // Update system status on load
        window.addEventListener('load', function() {
            updateSystemStatus();
            // Update status every 5 seconds
            setInterval(updateSystemStatus, 5000);
        });
    </script>
</body>
</html>
    """
    
    return HTMLResponse(content=html_content)

@app.post("/api/chat")
async def chat(request: PromptRequest):
    """Main chat endpoint"""
    try:
        # Convert string mode to enum
        mode_map = {
            "adaptive": SystemMode.ADAPTIVE,
            "cognitive": SystemMode.COGNITIVE,
            "generation": SystemMode.GENERATION,
            "learning": SystemMode.LEARNING
        }
        
        mode = mode_map.get(request.mode, SystemMode.ADAPTIVE)
        
        # Process with SloughGPT
        response = await sloughgpt.process_prompt(
            request.prompt,
            mode=mode,
            context=request.context
        )
        
        # Store response for feedback
        response_id = str(id(response))
        active_responses[response_id] = response
        
        return JSONResponse({
            "response_id": response_id,
            "prompt": request.prompt,
            "mode": request.mode,
            "neural_response": response.neural_response,
            "cognitive_analysis": {
                "thinking_modes": response.metadata.get("thinking_modes_used", []),
                "reasoning_steps": response.metadata.get("reasoning_steps", 0),
                "creative_ideas": response.metadata.get("creative_ideas", 0)
            },
            "confidence": response.confidence,
            "processing_time": response.processing_time,
            "timestamp": response.timestamp
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
async def feedback(request: FeedbackRequest):
    """Learning feedback endpoint"""
    try:
        result = await sloughgpt.learn_from_feedback(
            request.response_id,
            {
                "rating": request.rating,
                "helpfulness": request.helpfulness,
                "accuracy": request.accuracy,
                "creativity": request.creativity,
                "coherence": request.coherence,
                "thinking_mode": request.thinking_mode
            }
        )
        
        logger.info(f"Feedback received: {request.response_id}, rating: {request.rating}")
        
        return JSONResponse({
            "status": "success",
            "message": "Learning completed",
            "learning_updates": result.get("learning_updates", {}),
            "new_cognitive_params": result.get("new_cognitive_params", {})
        })
        
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def status():
    """System status endpoint"""
    try:
        system_status = sloughgpt.get_system_status()
        return JSONResponse(system_status)
    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def models():
    """Available models endpoint"""
    return JSONResponse({
        "models": [
            {
                "id": "sloughgpt-v1",
                "name": "SloughGPT v1.0",
                "description": "Custom GPT with continuous learning and cognitive capabilities",
                "parameters": sum(p.numel() for p in sloughgpt.neural_model.parameters()),
                "config": {
                    "vocab_size": sloughgpt.config.vocab_size,
                    "d_model": sloughgpt.config.d_model,
                    "n_heads": sloughgpt.config.n_heads,
                    "n_layers": sloughgpt.config.n_layers
                },
                "capabilities": [
                    "text_generation",
                    "reasoning", 
                    "creativity",
                    "continuous_learning",
                    "cognitive_analysis"
                ]
            }
        ]
    })

@app.get("/health")
async def health():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "components": {
            "neural_model": "active",
            "learning_system": "active", 
            "cognitive_system": "active"
        }
    })

if __name__ == "__main__":
    # Save learning state on shutdown
    import atexit
    atexit.register(lambda: sloughgpt._save_learning_state())
    
    print("🚀 SloughGPT Production Server Starting")
    print("=" * 50)
    print("✅ Custom Neural Network: Active")
    print("✅ Learning System: Active") 
    print("✅ Cognitive Integration: Active")
    print("✅ Web UI: Available")
    print("✅ API Endpoints: Active")
    print()
    print("🌐 Server will be available at:")
    print("   http://localhost:8000")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )