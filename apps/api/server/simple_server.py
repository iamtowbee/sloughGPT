#!/usr/bin/env python3
"""
SloughGPT Inference Server
Auto-selects best available inference backend.
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
USE_OLLAMA = os.environ.get("SLOUGHGPT_USE_OLLAMA", "auto").lower()

# Global state
model_type = "none"
backend_type = "none"
model_loaded = False
load_error = None


def init_backend():
    """Initialize the best available inference backend."""
    global model_type, backend_type, model_loaded, load_error

    # Check for Ollama
    if USE_OLLAMA in ("auto", "1", "true", "yes", ""):
        if check_ollama():
            backend_type = "ollama"
            model_type = f"ollama/{DEFAULT_MODEL}"
            model_loaded = True
            return

    # Check GPU and use llama.cpp
    try:
        from domains.inference.llama_engine import detect_gpu, auto_select_backend

        gpu = detect_gpu()
        if gpu:
            print(f"GPU: {gpu.name} ({gpu.vram_mb:.0f}MB) - {gpu.reason}")

        n_gpu_layers = auto_select_backend(1.5)
        if n_gpu_layers > 0:
            backend_type = "llama.cpp-gpu"
        else:
            backend_type = "llama.cpp-cpu"

        model_type = backend_type
        model_loaded = True
        print(f"Using {backend_type} for inference")
        return
    except Exception as e:
        print(f"llama.cpp init failed: {e}")

    # Fallback to PyTorch GPT-2
    init_torch()


def check_ollama() -> bool:
    """Check if Ollama is available."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m["name"] for m in models]
            print(f"Ollama: {model_names}")
            return True
    except:
        pass
    return False


def init_torch():
    """Initialize PyTorch fallback."""
    global model_type, backend_type, model_loaded, load_error

    print("Using PyTorch GPT-2...")
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        backend_type = "pytorch"
        model_type = "gpt2"
        model_loaded = True
        print("GPT-2 ready")
    except Exception as e:
        load_error = str(e)
        print(f"GPT-2 failed: {e}")


# Initialize at startup
init_backend()


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


@app.get("/")
async def root():
    return {
        "name": "SloughGPT Inference Server",
        "version": "1.0.0",
        "backend": backend_type,
        "model": model_type,
        "model_loaded": model_loaded,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model_loaded else "error",
        "backend": backend_type,
        "model": model_type,
        "model_loaded": model_loaded,
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text from prompt."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    if backend_type == "ollama":
        return await ollama_generate(request)
    elif backend_type in ("llama.cpp-gpu", "llama.cpp-cpu"):
        return await llama_cpp_generate(request)
    else:
        return await torch_generate(request)


async def ollama_generate(request: GenerateRequest):
    """Generate using Ollama."""
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


async def llama_cpp_generate(request: GenerateRequest):
    """Generate using llama.cpp."""
    try:
        from domains.inference.llama_engine import AutoInferenceEngine

        engine = AutoInferenceEngine()
        result = engine.benchmark(request.prompt, request.max_new_tokens)
        return {
            "text": result.get("text", ""),
            "model": "llama.cpp",
            "tokens_per_second": result.get("tokens_per_second", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def torch_generate(request: GenerateRequest):
    """Generate using PyTorch GPT-2."""
    import torch

    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()

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
        return {"text": text, "model": "gpt2"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/completions")
async def chat(request: ChatRequest):
    """Chat completion endpoint."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    prompt = request.messages[-1].content if request.messages else ""
    gen_req = GenerateRequest(
        prompt=prompt, max_new_tokens=request.max_new_tokens, temperature=request.temperature
    )

    return await generate(gen_req)


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("SloughGPT Inference Server")
    print("=" * 50)
    print(f"Backend: {backend_type}")
    print(f"Model: {model_type}")
    print(f"Status: {'Ready' if model_loaded else f'Error: {load_error}'}")
    print("=" * 50)
    print("Endpoints:")
    print("  GET  /health    - Health check")
    print("  POST /generate  - Text generation")
    print("  POST /chat/completions - Chat")
    print("=" * 50)

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
