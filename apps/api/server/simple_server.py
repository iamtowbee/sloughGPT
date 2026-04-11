#!/usr/bin/env python3
"""
SloughGPT Inference Server
Fast server using Ollama for GPU-accelerated inference.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
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

# Configuration
OLLAMA_URL = os.environ.get("SLOUGHGPT_OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("SLOUGHGPT_OLLAMA_MODEL", "llama3.2:1b")

# Global state
model_type = "ollama"
model_loaded = False
load_error = None


def check_ollama():
    """Check if Ollama is available."""
    global model_loaded, load_error
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m["name"] for m in models]
            print(f"Ollama available with models: {model_names}")
            if DEFAULT_MODEL in model_names:
                model_loaded = True
                print(f"Default model '{DEFAULT_MODEL}' ready")
            else:
                print(f"Warning: Default model '{DEFAULT_MODEL}' not found")
        else:
            load_error = f"Ollama returned status {resp.status_code}"
            print(f"Error: {load_error}")
    except Exception as e:
        load_error = str(e)
        print(f"Ollama not available: {e}")


# Check Ollama at startup
check_ollama()


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
    """Load GPT-2 model (fallback if Ollama unavailable)."""
    global model, tokenizer, model_loaded, load_error

    print("Ollama not available, falling back to GPT-2...")
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        tokenizer = GPT2LMHeadModel.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        model_loaded = True
        model_type = "gpt2"
        print("GPT-2 loaded successfully!")
    except Exception as e:
        load_error = str(e)
        print(f"Failed to load GPT-2: {e}")


# Fallback to GPT-2 if Ollama not available
if not model_loaded:
    load_gpt2()


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
            "text": f"[Error] Ollama not available and GPT-2 failed to load: {load_error}",
            "model": model_type,
        }

    if model_type == "ollama":
        return await ollama_generate(request)
    else:
        return await gpt2_generate(request)


async def ollama_generate(request: GenerateRequest):
    """Generate using Ollama API."""
    try:
        payload = {
            "model": DEFAULT_MODEL,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "num_predict": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
            },
        }

        start = time.time()
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        elapsed = time.time() - start

        if resp.status_code == 200:
            data = resp.json()
            tokens = data.get("eval_count", 0)
            tps = tokens / elapsed if elapsed > 0 else 0
            return {
                "text": data.get("response", ""),
                "model": f"ollama/{DEFAULT_MODEL}",
                "tokens_per_second": tps,
                "latency_ms": int(elapsed * 1000),
            }
        else:
            raise HTTPException(status_code=500, detail=resp.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def gpt2_generate(request: GenerateRequest):
    """Generate using GPT-2 (fallback)."""
    import torch

    global model, tokenizer

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
    if not model_loaded:
        return {
            "message": {
                "role": "assistant",
                "content": f"[Error] Model not available: {load_error}",
            },
            "model": model_type,
        }

    if model_type == "ollama":
        return await ollama_chat(request)
    else:
        return await gpt2_chat(request)


async def ollama_chat(request: ChatRequest):
    """Chat using Ollama."""
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": request.messages[-1].content if request.messages else "",
        "stream": False,
        "options": {
            "num_predict": request.max_new_tokens,
            "temperature": request.temperature,
        },
    }

    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    if resp.status_code == 200:
        data = resp.json()
        return {
            "message": {"role": "assistant", "content": data.get("response", "")},
            "model": f"ollama/{DEFAULT_MODEL}",
        }
    else:
        raise HTTPException(status_code=500, detail=resp.text)


async def gpt2_chat(request: ChatRequest):
    """Chat using GPT-2."""
    import torch

    global model, tokenizer

    prompt = ""
    for msg in request.messages:
        role = msg.role.upper()
        if role == "USER":
            prompt += f"User: {msg.content}\n"
        elif role == "ASSISTANT":
            prompt += f"Assistant: {msg.content}\n"
    prompt += "Assistant:"

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
        response = text.split("Assistant:")[-1].strip()

        return {
            "message": {"role": "assistant", "content": response},
            "model": model_type,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load")
async def load_model():
    """Reload/check Ollama connection."""
    global model_loaded, load_error
    check_ollama()

    if model_loaded:
        return {
            "status": "loaded",
            "model": model_type,
            "backend": "ollama",
            "ollama_model": DEFAULT_MODEL,
        }
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
    print("SloughGPT Inference Server")
    print("=" * 50)
    print(f"Backend: {model_type}")
    print(f"Ollama: {OLLAMA_URL}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Status: {'Ready' if model_loaded else f'Error: {load_error}'}")
    print("=" * 50)
    print("Endpoints:")
    print("  GET  /           - Server info")
    print("  GET  /health     - Health check")
    print("  POST /generate   - Text generation")
    print("  POST /chat/completions - Chat")
    print("  POST /load       - Reload model")
    print("=" * 50)
    port = find_available_port()
    print(f"Starting on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
