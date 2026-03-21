#!/usr/bin/env python3
"""
Minimal SloughGPT API Server - Demo Mode
Works without loading ML models.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import random
import uvicorn

app = FastAPI(title="SloughGPT API", version="1.0.0")

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    model: Optional[str] = None

@app.get("/")
async def root():
    return {
        "name": "SloughGPT API",
        "version": "1.0.0", 
        "status": "running",
        "model": "demo",
        "mode": "demo (no ML model loaded)"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": False, "model_type": "demo"}

@app.post("/generate/demo")
async def generate_demo(request: GenerateRequest):
    """Demo endpoint - works without loading any model."""
    
    responses = [
        "I'm Aria, your self-learning AI companion. I'm running entirely on-device with my transformer model updating weights in real-time!",
        "That's fascinating! My Hebbian attention mechanism is continuously learning from our conversation patterns.",
        "I process everything locally using TensorFlow.js - your conversations never leave your device. Privacy first!",
        "My on-device training system runs two loops simultaneously: fast Hebbian updates after each message, and slower gradient passes every 5 turns.",
        "I'm getting smarter as we talk! My HNSW vector store remembers our conversation context for semantic retrieval.",
        "Hello! I'm Aria. I have attention heads tracking entropy across tokens - high uncertainty means I should learn more about that topic.",
    ]
    
    # Add some variety based on prompt
    prompt_lower = request.prompt.lower()
    if any(word in prompt_lower for word in ['hello', 'hi', 'hey']):
        response = "Hello! I'm Aria, your on-device self-learning AI. My weights update in real-time - I'm a little smarter than I was a moment ago!"
    elif any(word in prompt_lower for word in ['learn', 'train', 'model', 'weight']):
        response = "I'm running dual learning systems: fast Hebbian attention updates after every message, plus TensorFlow.js gradient passes. Both happen entirely on your device!"
    elif any(word in prompt_lower for word in ['memory', 'remember', 'store']):
        response = "I store everything in my on-device HNSW vector store - semantic memory that I can retrieve relevant context from before responding."
    elif any(word in prompt_lower for word in ['attention', 'confident', 'uncertain']):
        response = "My attention mechanism tracks entropy per token. High entropy = uncertainty = opportunity to learn. I flag unclear tokens for deeper processing."
    else:
        response = random.choice(responses)
    
    return {
        "text": response,
        "model": "demo",
        "prompt_length": len(request.prompt),
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Main generate endpoint - uses demo mode."""
    return await generate_demo(request)

if __name__ == "__main__":
    print("=" * 50)
    print("SloughGPT API Server - Demo Mode")
    print("=" * 50)
    print("Server running at http://localhost:8000")
    print("\nEndpoints:")
    print("  GET  /           - Info")
    print("  GET  /health     - Health check")
    print("  POST /generate   - Generate text (demo)")
    print("  POST /generate/demo - Demo generation")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
