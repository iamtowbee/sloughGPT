#!/usr/bin/env python3
"""
SloughGPT Inference Server
Uses llama.cpp with smart CPU/GPU selection.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
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
MODEL_PATH = os.environ.get("SLOUGHGPT_MODEL_PATH", "")

# Global state
backend_type = "none"
gpu_layers = 0
model_loaded = False
load_error = None


def init_backend():
    """Initialize llama.cpp with smart CPU/GPU selection."""
    global backend_type, gpu_layers, model_loaded, load_error

    try:
        from domains.inference.llama_engine import detect_gpu, auto_select_backend

        gpu = detect_gpu()
        if gpu:
            print(f"GPU: {gpu.name} ({gpu.vram_mb:.0f}MB)")
            print(f"  Reason: {gpu.reason}")

        gpu_layers = auto_select_backend(1.5)

        if gpu_layers > 0:
            backend_type = "llama.cpp-gpu"
        else:
            backend_type = "llama.cpp-cpu"

        model_loaded = True
        print(f"Backend: {backend_type} (n_gpu_layers={gpu_layers})")

    except Exception as e:
        load_error = str(e)
        print(f"llama.cpp failed: {e}")
        init_torch()


def init_torch():
    """Initialize PyTorch fallback."""
    global backend_type, model_loaded, load_error

    print("Using PyTorch GPT-2...")
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        backend_type = "pytorch"
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


@app.get("/")
async def root():
    return {
        "name": "SloughGPT Inference Server",
        "version": "1.0.0",
        "backend": backend_type,
        "gpu_layers": gpu_layers,
        "model_path": MODEL_PATH,
        "model_loaded": model_loaded,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model_loaded else "error",
        "backend": backend_type,
        "gpu_layers": gpu_layers,
        "model_loaded": model_loaded,
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text from prompt."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    if backend_type.startswith("llama.cpp"):
        return await llama_cpp_generate(request)
    else:
        return await torch_generate(request)


async def llama_cpp_generate(request: GenerateRequest):
    """Generate using llama.cpp."""
    try:
        from domains.inference.llama_engine import LlamaInferenceConfig, LlamaInferenceEngine

        if not MODEL_PATH:
            raise HTTPException(status_code=500, detail="SLOUGHGPT_MODEL_PATH not set")

        config = LlamaInferenceConfig(
            model_path=MODEL_PATH,
            n_gpu_layers=gpu_layers,
        )
        engine = LlamaInferenceEngine(config)
        result = engine.benchmark(request.prompt, request.max_new_tokens)

        return {
            "text": result.get("text_preview", ""),
            "model": backend_type,
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


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("SloughGPT Inference Server")
    print("=" * 50)
    print(f"Backend: {backend_type}")
    print(f"GPU layers: {gpu_layers}")
    print(f"Model path: {MODEL_PATH or '(not set)'}")
    print(f"Status: {'Ready' if model_loaded else f'Error: {load_error}'}")
    print("=" * 50)
    print("Endpoints:")
    print("  GET  /health    - Health check")
    print("  POST /generate  - Text generation")
    print("=" * 50)
    print("Environment:")
    print("  SLOUGHGPT_MODEL_PATH - Path to GGUF model")
    print("  SLOUGHGPT_FORCE_GPU=1 - Force GPU (if available)")
    print("  SLOUGHGPT_FORCE_CPU=1 - Force CPU")
    print("=" * 50)

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
