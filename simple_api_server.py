"""Simple working API server for testing"""

import time
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SloGPT API",
    description="Simple API for SloGPT model inference",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    messages: List[dict]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str
    usage: dict

# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SloGPT API Server",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": True,
        "version": "1.0.0"
    }

@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    """Chat endpoint"""
    try:
        # Mock response for now
        response = "This is a mock response from SLO model"
        return ChatResponse(
            response=response,
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/model")
async def get_models():
    """Get available models"""
    return {
        "models": [
            {
                "id": "slo-gpt",
                "name": "SLO GPT Model",
                "description": "SLO transformer model for chat"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )