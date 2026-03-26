#!/usr/bin/env python3
"""
Minimal SloughGPT Server with GPT-2
Simple server that loads GPT-2 for text generation.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import asyncio
import threading
import time

app = FastAPI(title="SloughGPT", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
model = None
tokenizer = None
model_type = "gpt2"
model_loaded = False
load_error = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: int = 100
    temperature: float = 0.8


def load_gpt2():
    """Load GPT-2 model."""
    global model, tokenizer, model_loaded, load_error
    
    print("Loading GPT-2 from HuggingFace...")
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        model_loaded = True
        print("GPT-2 loaded successfully!")
    except Exception as e:
        load_error = str(e)
        print(f"Failed to load GPT-2: {e}")


# Load model in background thread
def background_load():
    time.sleep(1)  # Give server time to start
    load_gpt2()

threading.Thread(target=background_load, daemon=True).start()


@app.get("/")
async def root():
    return {
        "name": "SloughGPT API",
        "version": "1.0.0",
        "model": model_type,
        "model_loaded": model_loaded,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_type": model_type,
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text from prompt."""
    if not model_loaded:
        return {
            "text": f"[Demo mode] {request.prompt[:50]}... (model loading...)",
            "model": model_type,
        }
    
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"text": text, "model": model_type}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/completions")
async def chat(request: ChatRequest):
    """Chat completion endpoint."""
    # Combine messages into prompt
    prompt = ""
    for msg in request.messages:
        role = msg.role.upper()
        if role == "USER":
            prompt += f"User: {msg.content}\n"
        elif role == "ASSISTANT":
            prompt += f"Assistant: {msg.content}\n"
    prompt += "Assistant:"
    
    if not model_loaded:
        return {
            "message": {"role": "assistant", "content": f"[Demo mode] Hi! (model loading...)"},
            "model": model_type,
        }
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant response
        response = text.split("Assistant:")[-1].strip()
        
        return {
            "message": {"role": "assistant", "content": response},
            "model": model_type,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load")
async def load_model():
    """Load or reload model."""
    global model_loaded
    
    if model_loaded:
        return {"status": "already_loaded", "model": model_type}
    
    load_gpt2()
    
    if model_loaded:
        return {"status": "loaded", "model": model_type}
    else:
        return {"status": "error", "error": load_error}


if __name__ == "__main__":
    import uvicorn
    import socket

    def find_available_port(start=8000, max_attempts=10):
        for port in range(start, start + max_attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("", port))
                sock.close()
                return port
            except OSError:
                continue
        raise RuntimeError(f"Could not find available port")

    print("=" * 50)
    print("SloughGPT Server with GPT-2")
    print("=" * 50)
    print("Endpoints:")
    print("  GET  /           - Server info")
    print("  GET  /health     - Health check")
    print("  POST /generate   - Text generation")
    print("  POST /chat/completions - Chat")
    print("  POST /load       - Load/reload model")
    print("=" * 50)
    port = find_available_port()
    print(f"Starting on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
    uvicorn.run(app, host="0.0.0.0", port=8000)
