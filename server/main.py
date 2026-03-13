#!/usr/bin/env python3
"""
SloughGPT Model Server
FastAPI server for model inference with HuggingFace fallback.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
import json
import asyncio

app = FastAPI(
    title="SloughGPT API",
    description="SloughGPT Model Inference API with HuggingFace models",
    version="1.0.0",
    docs_url="/docs",
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

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
    personality: Optional[str] = None


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
        "model": model_type,
        "endpoints": {
            "generate": "/generate (POST)",
            "generate_stream": "/generate/stream (POST)",
            "generate_ws": "/ws/generate (WebSocket)",
            "personalities": "/personalities (GET)",
            "models": "/models (GET)",
            "datasets": "/datasets (GET)",
            "info": "/info (GET)",
        }
    }


@app.get("/info")
async def info():
    """Get detailed server info."""
    import torch
    
    info = {
        "api_version": "1.0.0",
        "model": {
            "type": model_type,
            "loaded": model is not None,
        },
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if checkpoint:
        info["model"].update({
            "vocab_size": len(checkpoint.get('stoi', {})),
            "chars": len(checkpoint.get('chars', [])),
        })
    
    if torch.cuda.is_available():
        info["cuda"] = {
            "device": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    
    return info


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


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket endpoint for real-time text generation."""
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            prompt = request_data.get("prompt", "")
            max_tokens = request_data.get("max_tokens", 100)
            temperature = request_data.get("temperature", 0.8)
            
            # Send status
            await websocket.send_json({"status": "generating", "prompt": prompt})
            
            if model_type == "nanogpt" and checkpoint:
                stoi = checkpoint.get('stoi', {})
                itos = checkpoint.get('itos', {})
                
                idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)
                
                model.eval()
                generated = ""
                
                with torch.no_grad():
                    for _ in range(max_tokens):
                        idx_cond = idx[:, -128:]
                        logits, _ = model(idx_cond)
                        logits = logits[:, -1, :] / temperature
                        probs = torch.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        idx = torch.cat([idx, idx_next], dim=1)
                        
                        char = itos.get(idx_next.item(), '')
                        generated += char
                        
                        # Stream each character
                        await websocket.send_json({"token": char, "generated": generated})
                        
                        if len(generated) > max_tokens:
                            break
                
                await websocket.send_json({"status": "done", "text": generated})
            
            elif model_type == "gpt2" and tokenizer:
                inputs = tokenizer(prompt, return_tensors="pt")
                
                model.eval()
                generated = ""
                
                with torch.no_grad():
                    for _ in range(max_tokens):
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=1,
                            temperature=temperature,
                            do_sample=True,
                            return_dict_in_generate=True,
                        )
                        token = tokenizer.decode(outputs.sequences[0][-1])
                        generated += token
                        
                        await websocket.send_json({"token": token, "generated": generated})
                        
                        inputs = tokenizer(token, return_tensors="pt")
                
                await websocket.send_json({"status": "done", "text": generated})
            
            else:
                # Demo mode
                demo_text = f"Demo response to: {prompt}"
                for char in demo_text:
                    await websocket.send_json({"token": char, "generated": char})
                    await asyncio.sleep(0.05)
                await websocket.send_json({"status": "done", "text": demo_text})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await websocket.send_json({"status": "error", "error": str(e)})
        manager.disconnect(websocket)


@app.get("/personalities")
async def list_personalities():
    """List available personalities."""
    try:
        from domains.ai_personality import PERSONALITIES, PersonalityType
        return {
            "personalities": [
                {
                    "type": ptype.value,
                    "name": p.name,
                    "description": p.description,
                    "traits": p.traits
                }
                for ptype, p in PERSONALITIES.items()
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/models")
async def list_models():
    """List available models."""
    from pathlib import Path
    
    models_dir = Path("models")
    models = []
    
    if models_dir.exists():
        for m in models_dir.glob("*.pt"):
            size = m.stat().st_size / (1024 * 1024)  # MB
            models.append({
                "name": m.name,
                "path": str(m),
                "size_mb": round(size, 2)
            })
    
    return {"models": models}


@app.get("/datasets")
async def list_datasets():
    """List available datasets."""
    from pathlib import Path
    
    datasets_dir = Path("datasets")
    datasets = []
    
    if datasets_dir.exists():
        for d in datasets_dir.iterdir():
            if d.is_dir():
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                datasets.append({
                    "name": d.name,
                    "path": str(d),
                    "size_kb": round(size / 1024, 2)
                })
    
    return {"datasets": datasets}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
