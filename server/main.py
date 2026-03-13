#!/usr/bin/env python3
"""
SloughGPT Model Server
FastAPI server for model inference with HuggingFace fallback.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
import json

app = FastAPI(
    title="SloughGPT API",
    description="SloughGPT Model Inference API with HuggingFace models",
    version="1.0.0",
    docs_url="/docs",
)

# Global model
model = None
tokenizer = None
model_type = "none"
checkpoint = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.0
    seed: Optional[int] = None


def load_model():
    """Load model - prefers local, falls back to HuggingFace."""
    global model, tokenizer, model_type, checkpoint
    
    # Try loading local model first
    local_model_path = "models/sloughgpt.pt"
    if os.path.exists(local_model_path):
        print(f"Loading local model from {local_model_path}...")
        try:
            checkpoint = torch.load(local_model_path, weights_only=False, map_location='cpu')
            model = checkpoint.get('model', checkpoint)
            model_type = "nanogpt"
            print("Local NanoGPT model loaded!")
            return
        except Exception as e:
            print(f"Failed to load local model: {e}")
    
    # Fall back to HuggingFace GPT-2
    print("Loading GPT-2 from HuggingFace...")
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model_type = "gpt2"
        print("GPT-2 loaded successfully!")
    except Exception as e:
        print(f"Failed to load GPT-2: {e}")
        model_type = "none"


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/")
async def root():
    return {
        "name": "SloughGPT API",
        "version": "1.0.0",
        "status": "running",
        "model": model_type
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model_type
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    if model is None:
        # Return demo response if no model
        return {
            "text": f"Demo response to: {request.prompt[:50]}... (No model loaded)",
            "model": model_type
        }
    
    if model_type == "gpt2":
        inputs = tokenizer(request.prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                do_sample=True,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"text": text, "model": model_type}
    
    if model_type == "nanogpt":
        # Generate using local NanoGPT
        import numpy as np
        stoi = checkpoint.get('stoi', {})
        itos = checkpoint.get('itos', {})
        
        idx = torch.tensor([[stoi.get(c, 0) for c in request.prompt]], dtype=torch.long)
        
        model.eval()
        with torch.no_grad():
            for _ in range(request.max_new_tokens):
                idx_cond = idx[:, -128:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / request.temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)
        
        generated = ''.join([itos.get(i, '') for i in idx[0].tolist()])
        text = generated[len(request.prompt):]
        return {"text": text, "model": model_type}
    
    return {"text": "Model type not supported", "model": model_type}


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    if model is None or tokenizer is None:
        async def demo_stream():
            demo = f"Demo streaming response to: {request.prompt}..."
            for char in demo:
                yield f"data: {json.dumps({'token': char})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(demo_stream(), media_type="text/event-stream")
    
    async def stream():
        if model_type == "gpt2":
            inputs = tokenizer(request.prompt, return_tensors="pt")
            
            from transformers import GenerationMixin
            from typing import Iterator
            
            # Stream tokens
            with torch.no_grad():
                for i in range(request.max_new_tokens):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        do_sample=True,
                        return_dict_in_generate=True,
                    )
                    token = tokenizer.decode(outputs.sequences[0][-1])
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    inputs = tokenizer(token, return_tensors="pt")
                    
                    if i >= request.max_new_tokens - 1:
                        break
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
