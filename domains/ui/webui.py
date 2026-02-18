"""
Enhanced WebUI - Ported from recovered enhanced_webui.py
FastAPI-based web interface for SloughGPT
"""

from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime


app = FastAPI(title="SloughGPT Enhanced WebUI", description="Enhanced web interface for SloughGPT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


conversations: Dict[str, List[Dict]] = {}
models: List[ModelConfig] = [
    ModelConfig(id="gpt-3.5-turbo", name="GPT-3.5 Turbo", provider="OpenAI", description="Fast, cost-effective"),
    ModelConfig(id="gpt-4", name="GPT-4", provider="OpenAI", description="Most capable"),
    ModelConfig(id="claude-3", name="Claude 3", provider="Anthropic", description="Helpful assistant"),
]


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "SloughGPT Enhanced WebUI", "version": "2.0.0"}


@app.get("/chat/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "messages": conversations[conversation_id]}


@app.post("/chat")
async def chat(message: ChatMessage):
    """Send a chat message."""
    conversation_id = message.conversation_id or f"conv_{datetime.now().timestamp()}"
    
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    conversations[conversation_id].append({
        "role": "user",
        "content": message.message,
        "timestamp": datetime.now().isoformat()
    })
    
    response = {
        "role": "assistant",
        "content": f"Echo: {message.message}",
        "timestamp": datetime.now().isoformat()
    }
    
    conversations[conversation_id].append(response)
    
    return {
        "conversation_id": conversation_id,
        "message": response,
        "model": message.model
    }


@app.get("/models")
async def list_models():
    """List available models."""
    return {"models": [m.model_dump() for m in models]}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


__all__ = ["app"]
