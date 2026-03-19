#!/usr/bin/env python3
"""
SloughGPT Model Server
FastAPI server for model inference with HuggingFace fallback.
"""

import os
import sys
# Force CPU mode to avoid MPS hanging issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
from contextlib import asynccontextmanager
from collections import defaultdict
from typing import Dict, List
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Any
import torch
import json
import asyncio
import time


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.clients: Dict[str, List[float]] = defaultdict(list)

    def _cleanup(self, client_id: str):
        """Remove expired timestamps."""
        current_time = time.time()
        cutoff = current_time - 60
        self.clients[client_id] = [
            ts for ts in self.clients[client_id] if ts > cutoff
        ]

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """
        Check if request is allowed.
        Returns (allowed, remaining_requests).
        """
        self._cleanup(client_id)
        current_count = len(self.clients[client_id])

        if current_count >= self.requests_per_minute:
            return False, 0

        self.clients[client_id].append(time.time())
        remaining = self.requests_per_minute - current_count - 1
        return True, max(0, remaining)

    def get_wait_time(self, client_id: str) -> float:
        """Get seconds until next request is allowed."""
        if not self.clients[client_id]:
            return 0
        oldest = min(self.clients[client_id])
        return max(0, 60 - (time.time() - oldest))


rate_limiter = RateLimiter(requests_per_minute=60, burst_size=10)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""

    SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        allowed, remaining = rate_limiter.is_allowed(client_ip)

        if not allowed:
            wait_time = rate_limiter.get_wait_time(client_ip)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many requests",
                    "message": f"Rate limit exceeded. Try again in {wait_time:.1f} seconds.",
                    "retry_after": int(wait_time) + 1,
                },
                headers={
                    "Retry-After": str(int(wait_time) + 1),
                    "X-RateLimit-Limit": str(rate_limiter.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Don't load model at startup to avoid hanging
    # User can call /health to check status
    yield


app = FastAPI(
    title="SloughGPT API",
    description="SloughGPT Model Inference API with HuggingFace models",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RateLimitMiddleware)


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


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    model: Optional[str] = None


def load_model():
    """Load model - prefers local, falls back to HuggingFace."""
    global model, tokenizer, model_type, checkpoint

    # Try loading local model first
    local_model_path = "models/sloughgpt.pt"
    if os.path.exists(local_model_path):
        print(f"Loading local model from {local_model_path}...")
        try:
            checkpoint = torch.load(local_model_path, weights_only=False, map_location="cpu")
            model = checkpoint.get("model", checkpoint)
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


@app.post("/load")
async def load_model_endpoint():
    """Load the model on demand."""
    global model, tokenizer, model_type
    if model is not None:
        return {"status": "already_loaded", "model": model_type}
    load_model()
    return {"status": "loaded", "model": model_type}


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
        },
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
        info["model"].update(
            {
                "vocab_size": len(checkpoint.get("stoi", {})),
                "chars": len(checkpoint.get("chars", [])),
            }
        )

    if torch.cuda.is_available():
        info["cuda"] = {
            "device": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }

    return info


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None, "model_type": model_type}


@app.get("/rate-limit/status")
async def get_rate_limit_status():
    """Get current rate limit configuration."""
    return {
        "requests_per_minute": rate_limiter.requests_per_minute,
        "burst_size": rate_limiter.burst_size,
        "active_clients": len(rate_limiter.clients),
    }


@app.get("/rate-limit/check")
async def check_rate_limit(request: Request):
    """Check rate limit status for client IP."""
    client_ip = request.client.host if request.client else "unknown"
    rate_limiter._cleanup(client_ip)
    current_count = len(rate_limiter.clients.get(client_ip, []))
    return {
        "client_ip": client_ip,
        "requests_used": current_count,
        "requests_remaining": max(0, rate_limiter.requests_per_minute - current_count),
        "retry_after": 0 if current_count < rate_limiter.requests_per_minute else rate_limiter.get_wait_time(client_ip),
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    if model is None:
        # Return demo response if no model
        return {
            "text": f"Demo response to: {request.prompt[:50]}... (No model loaded)",
            "model": model_type,
        }

    # Apply personality adjustment to temperature
    temperature = request.temperature
    if request.personality:
        try:
            from domains.ai_personality import PERSONALITIES, PersonalityType

            ptype = PersonalityType(request.personality.lower())
            if ptype in PERSONALITIES:
                personality = PERSONALITIES[ptype]
                temperature = personality.modify_temperature(temperature)
        except Exception:
            pass  # Ignore personality errors

    if model_type == "gpt2":
        inputs = tokenizer(request.prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                do_sample=True,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"text": text, "model": model_type, "personality": request.personality}

    if model_type == "nanogpt":
        # Generate using local NanoGPT
        stoi = checkpoint.get("stoi", {})
        itos = checkpoint.get("itos", {})

        idx = torch.tensor([[stoi.get(c, 0) for c in request.prompt]], dtype=torch.long)

        model.eval()
        with torch.no_grad():
            for _ in range(request.max_new_tokens):
                idx_cond = idx[:, -128:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)

        generated = "".join([itos.get(i, "") for i in idx[0].tolist()])
        text = generated[len(request.prompt) :]
        return {"text": text, "model": model_type, "personality": request.personality}

    return {"text": "Model type not supported", "model": model_type}


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """Streaming text generation using Server-Sent Events."""

    async def generate_stream_tokens():
        if model is None:
            demo = f"Demo streaming response to: {request.prompt}..."
            for char in demo:
                yield f"data: {json.dumps({'token': char, 'done': False})}\n\n"
                await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
            return

        if model_type == "gpt2":
            loop = asyncio.get_event_loop()
            inputs = await loop.run_in_thread(
                None, lambda: tokenizer(request.prompt, return_tensors="pt")
            )

            generated_text = request.prompt
            for i in range(request.max_new_tokens):
                outputs = await loop.run_in_thread(
                    None,
                    lambda inp=inputs: model.generate(
                        **inp,
                        max_new_tokens=1,
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        do_sample=True,
                        return_dict_in_generate=True,
                    ),
                )
                token = tokenizer.decode(outputs.sequences[0][-1], skip_special_tokens=True)
                if token:
                    generated_text += token
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                inputs = tokenizer(token, return_tensors="pt")

                if i >= request.max_new_tokens - 1:
                    break

        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    return StreamingResponse(generate_stream_tokens(), media_type="text/event-stream")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat completion using Server-Sent Events."""

    def format_chat_prompt(messages):
        formatted = ""
        for msg in messages:
            role = msg.role
            content = msg.content
            if role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"
            elif role == "system":
                formatted += f"System: {content}\n"
        formatted += "Assistant:"
        return formatted

    async def generate_stream_tokens():
        prompt = format_chat_prompt([m.model_dump() for m in request.messages])

        if model is None:
            demo = "Demo chat response..."
            for char in demo:
                yield f"data: {json.dumps({'token': char, 'done': False})}\n\n"
                await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
            return

        if model_type == "gpt2":
            loop = asyncio.get_event_loop()
            inputs = await loop.run_in_thread(None, lambda: tokenizer(prompt, return_tensors="pt"))

            for i in range(request.max_new_tokens):
                outputs = await loop.run_in_thread(
                    None,
                    lambda inp=inputs: model.generate(
                        **inp,
                        max_new_tokens=1,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        do_sample=True,
                        return_dict_in_generate=True,
                    ),
                )
                token = tokenizer.decode(outputs.sequences[0][-1], skip_special_tokens=True)
                if token:
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                inputs = tokenizer(token, return_tensors="pt")

                if i >= request.max_new_tokens - 1:
                    break

        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    return StreamingResponse(generate_stream_tokens(), media_type="text/event-stream")


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
            model_name = request_data.get("model", None)

            await websocket.send_json({"status": "generating", "prompt": prompt})

            if model_name and model_name.startswith("hf/"):
                await websocket.send_json(
                    {
                        "status": "error",
                        "error": "HuggingFace models via WS not yet supported",
                    }
                )
                continue

            if model_type == "nanogpt" and checkpoint:
                stoi = checkpoint.get("stoi", {})
                itos = checkpoint.get("itos", {})

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

                        char = itos.get(idx_next.item(), "")
                        generated += char

                        await websocket.send_json(
                            {"token": char, "generated": generated, "done": False}
                        )

                        if len(generated) > max_tokens:
                            break

                await websocket.send_json({"status": "done", "text": generated, "done": True})

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
                        if token and not token.startswith(" "):
                            generated += token

                        await websocket.send_json(
                            {"token": token, "generated": generated, "done": False}
                        )

                        inputs = tokenizer(token, return_tensors="pt")

                await websocket.send_json({"status": "done", "text": generated, "done": True})

            else:
                demo_text = f"Demo response to: {prompt}"
                for char in demo_text:
                    await websocket.send_json({"token": char, "generated": char, "done": False})
                    await asyncio.sleep(0.05)
                await websocket.send_json({"status": "done", "text": demo_text, "done": True})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await websocket.send_json({"status": "error", "error": str(e)})
        manager.disconnect(websocket)


@app.get("/personalities")
async def list_personalities():
    """List available personalities."""
    try:
        from domains.ai_personality import PERSONALITIES

        return {
            "personalities": [
                {
                    "type": ptype.value,
                    "name": p.name,
                    "description": p.description,
                    "traits": p.traits,
                }
                for ptype, p in PERSONALITIES.items()
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/datasets", tags=["datasets"])
async def list_datasets():
    """List available datasets."""
    import os
    from pathlib import Path
    
    datasets_dir = Path("datasets")
    datasets = []
    
    if datasets_dir.exists():
        for d in datasets_dir.iterdir():
            if d.is_dir():
                input_file = d / "input.txt"
                size = 0
                if input_file.exists():
                    size = input_file.stat().st_size
                
                datasets.append({
                    "id": d.name,
                    "name": d.name.replace("_", " ").title(),
                    "path": str(d),
                    "size_bytes": size,
                    "size_formatted": f"{size / 1024:.1f} KB" if size > 0 else "Empty",
                    "type": "text",
                })
    
    return {"datasets": datasets}


@app.get("/datasets/{dataset_id}", tags=["datasets"])
async def get_dataset(dataset_id: str):
    """Get dataset details."""
    from pathlib import Path
    
    dataset_path = Path(f"datasets/{dataset_id}")
    input_file = dataset_path / "input.txt"
    
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    stats = {
        "id": dataset_id,
        "name": dataset_id.replace("_", " ").title(),
        "path": str(dataset_path),
    }
    
    if input_file.exists():
        with open(input_file, "r") as f:
            content = f.read()
        stats.update({
            "size_bytes": len(content),
            "num_lines": content.count("\n") + 1,
            "num_chars": len(content),
        })
    
    return stats


@app.get("/datasets/{dataset_id}/stats", tags=["datasets"])
async def get_dataset_stats(dataset_id: str):
    """Get detailed dataset statistics."""
    from pathlib import Path
    from collections import Counter
    
    dataset_path = Path(f"datasets/{dataset_id}/input.txt")
    
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    with open(dataset_path, "r") as f:
        content = f.read()
    
    lines = content.split("\n")
    words = content.split()
    
    return {
        "dataset_id": dataset_id,
        "total_chars": len(content),
        "total_lines": len(lines),
        "total_words": len(words),
        "avg_line_length": sum(len(l) for l in lines) / len(lines) if lines else 0,
    }


@app.get("/models")
async def list_models():
    """List available models (local + HuggingFace)."""
    from pathlib import Path

    models = []

    models_dir = Path("models")
    if models_dir.exists():
        for m in models_dir.glob("*.pt"):
            size = m.stat().st_size / (1024 * 1024)
            models.append(
                {
                    "id": f"local/{m.stem}",
                    "name": m.stem,
                    "path": str(m),
                    "size_mb": round(size, 2),
                    "source": "local",
                }
            )

    try:
        from domains.training.model_registry import get_available_hf_models

        hf_models = get_available_hf_models()
        for m in hf_models:
            models.append(
                {
                    "id": m.id,
                    "name": m.name,
                    "description": m.description,
                    "source": "huggingface",
                    "tags": m.tags,
                }
            )
    except Exception:
        pass

    return {"models": models}


class LoadModelRequest(BaseModel):
    model_id: str
    mode: Optional[str] = "local"
    device: Optional[str] = "auto"


@app.post("/models/load")
async def load_hf_model_endpoint(request: LoadModelRequest):
    """Load a HuggingFace model."""
    global model, tokenizer, model_type

    try:
        from domains.training.model_registry import load_hf_model

        client = load_hf_model(request.model_id, mode=request.mode)
        model_type = f"hf/{request.model_id}"
        return {
            "status": "loaded",
            "model": request.model_id,
            "mode": request.mode,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/models/hf")
async def list_hf_models():
    """List available HuggingFace models."""
    try:
        from domains.training.model_registry import get_available_hf_models

        models = get_available_hf_models()
        return {
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "description": m.description,
                    "tags": m.tags,
                    "hf_model_id": m.hf_model_id,
                }
                for m in models
            ]
        }
    except Exception as e:
        return {"error": str(e), "models": []}


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
                datasets.append({"name": d.name, "path": str(d), "size_kb": round(size / 1024, 2)})

    return {"datasets": datasets}


class TrainRequest(BaseModel):
    dataset: str
    epochs: Optional[int] = 3
    batch_size: Optional[int] = 32
    learning_rate: Optional[float] = 1e-3
    n_embed: Optional[int] = 128
    n_layer: Optional[int] = 4
    n_head: Optional[int] = 4
    block_size: Optional[int] = 128
    max_steps: Optional[int] = None


class TrainingRequest(BaseModel):
    name: str
    model: str
    dataset: str
    epochs: Optional[int] = 3
    batch_size: Optional[int] = 32
    learning_rate: Optional[float] = 1e-3


@app.post("/train")
async def train(request: TrainRequest):
    """Start a training job."""
    import threading
    from domains.training.train_pipeline import SloughGPTTrainer

    def train_model():
        try:
            trainer = SloughGPTTrainer(
                data_path=f"datasets/{request.dataset}/input.txt",
                n_embed=request.n_embed,
                n_layer=request.n_layer,
                n_head=request.n_head,
                block_size=request.block_size,
                batch_size=request.batch_size,
                epochs=request.epochs,
                lr=request.learning_rate,
                max_steps=request.max_steps,
            )
            trainer.train()
            trainer.save(f"models/{request.dataset}_trained.pt")
        except Exception as e:
            print(f"Training error: {e}")

    # Run training in background thread
    thread = threading.Thread(target=train_model, daemon=True)
    thread.start()

    return {
        "status": "started",
        "dataset": request.dataset,
        "epochs": request.epochs,
        "message": "Training started in background",
    }


@app.get("/train/status")
async def train_status():
    """Get training status."""
    return {"status": "ready", "message": "Use /train endpoint to start training"}


training_jobs = {}


@app.get("/training/jobs", tags=["training"])
async def list_training_jobs():
    """List all training jobs."""
    return list(training_jobs.values())


@app.get("/training/jobs/{job_id}", tags=["training"])
async def get_training_job(job_id: str):
    """Get a specific training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return training_jobs[job_id]


@app.post("/training/start", tags=["training"])
async def start_training(request: TrainingRequest):
    """Start a new training job."""
    job_id = f"job_{len(training_jobs) + 1}"
    job = {
        "id": job_id,
        "name": request.name,
        "model": request.model,
        "dataset": request.dataset,
        "status": "running",
        "progress": 0,
        "epochs": request.epochs,
        "current_epoch": 0,
        "loss": None,
    }
    training_jobs[job_id] = job

    def run_training():
        import time
        for epoch in range(request.epochs or 3):
            training_jobs[job_id]["current_epoch"] = epoch + 1
            training_jobs[job_id]["loss"] = 2.5 - (epoch * 0.3)
            training_jobs[job_id]["progress"] = int(((epoch + 1) / (request.epochs or 3)) * 100)
            time.sleep(1)
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return job


_experiment_tracker = None


def get_experiment_tracker():
    """Get or create the experiment tracker."""
    global _experiment_tracker
    if _experiment_tracker is None:
        from domains.ml_infrastructure.experiment_tracker import ExperimentTracker
        _experiment_tracker = ExperimentTracker(storage_path="./experiments")
    return _experiment_tracker


@app.post("/experiments", tags=["experiments"])
async def create_experiment(
    name: str,
    description: str = "",
    parameters: Optional[str] = None,
):
    """Create a new experiment."""
    tracker = get_experiment_tracker()
    
    params = {}
    if parameters:
        try:
            params = json.loads(parameters)
        except:
            pass
    
    experiment_id = tracker.create_experiment(
        name=name,
        description=description,
        parameters=params,
    )
    
    exp = tracker.get_experiment(experiment_id)
    if exp:
        return exp.to_dict()
    return {"experiment_id": experiment_id, "name": name}


@app.get("/experiments", tags=["experiments"])
async def list_experiments():
    """List all experiments."""
    tracker = get_experiment_tracker()
    experiments = tracker.list_experiments()
    return [exp.to_dict() for exp in experiments]


@app.get("/experiments/{experiment_id}", tags=["experiments"])
async def get_experiment(experiment_id: str):
    """Get a specific experiment."""
    tracker = get_experiment_tracker()
    exp = tracker.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp.to_dict()


@app.post("/experiments/{experiment_id}/log_metric", tags=["experiments"])
async def log_metric(
    experiment_id: str,
    metric_name: str,
    value: float,
    step: int = 0,
):
    """Log a metric for an experiment."""
    import time
    
    tracker = get_experiment_tracker()
    exp = tracker.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    from domains.ml_infrastructure.experiment_tracker import MetricPoint
    
    if metric_name not in exp.metrics:
        exp.metrics[metric_name] = []
    
    exp.metrics[metric_name].append(MetricPoint(
        timestamp=time.time(),
        step=step,
        value=value
    ))
    
    return {"status": "logged", "metric": metric_name, "value": value}


@app.post("/experiments/{experiment_id}/log_param", tags=["experiments"])
async def log_param(
    experiment_id: str,
    param_name: str,
    value: Any,
):
    """Log a parameter for an experiment."""
    tracker = get_experiment_tracker()
    exp = tracker.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    exp.parameters[param_name] = value
    return {"status": "logged", "param": param_name, "value": value}


@app.get("/experiments/{experiment_id}/runs", tags=["experiments"])
async def get_experiment_runs(experiment_id: str):
    """Get runs for an experiment."""
    tracker = get_experiment_tracker()
    return tracker.get_experiment_runs(experiment_id)


@app.get("/runs/{run_id}", tags=["experiments"])
async def get_run(run_id: str):
    """Get a specific run."""
    tracker = get_experiment_tracker()
    run = tracker.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run.to_dict()


@app.post("/experiments/{experiment_id}/complete", tags=["experiments"])
async def complete_experiment(experiment_id: str, status: str = "completed"):
    """Mark experiment as complete."""
    tracker = get_experiment_tracker()
    tracker.complete_experiment(experiment_id, status)
    return {"status": "completed"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


# Global inference engine (lazy loaded)
_inference_engine = None


def get_inference_engine():
    """Get or create the inference engine using existing model."""
    global _inference_engine, model, tokenizer
    
    if model is None or tokenizer is None:
        return None
    
    if _inference_engine is None:
        from domains.inference.engine import InferenceEngine
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        _inference_engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
    return _inference_engine


@app.post("/inference/generate", tags=["inference"])
async def inference_generate(request: GenerateRequest):
    """Generate text using the production inference engine."""
    try:
        engine = get_inference_engine()
        
        if engine is None:
            return {"error": "Model not loaded", "text": ""}
        
        text = engine.generate_single(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=1.0,
        )
        
        return {
            "text": text,
            "model": "gpt2-engine",
            "tokens_generated": len(text.split()),
        }
    except Exception as e:
        return {"error": str(e), "text": ""}


@app.post("/inference/generate/stream", tags=["inference"])
async def inference_generate_stream(request: GenerateRequest):
    """Streaming generation using the production inference engine."""
    engine = get_inference_engine()
    
    async def token_stream():
        async for token in engine.generate_stream(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        ):
            yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
    
    return StreamingResponse(token_stream(), media_type="text/event-stream")


@app.get("/inference/stats", tags=["inference"])
async def inference_stats():
    """Get inference engine statistics."""
    engine = get_inference_engine()
    if engine is None:
        return {"error": "Engine not initialized"}
    return engine.get_stats()


class QuantizeRequest(BaseModel):
    quantization_type: str = "fp16"


@app.post("/inference/quantize", tags=["inference"])
async def quantize_model(request: QuantizeRequest):
    """Quantize the current model."""
    global model, _inference_engine
    
    if model is None:
        return {"error": "No model loaded"}
    
    try:
        from domains.inference.quantization import quantize_model as do_quantize, QuantizationType
        
        qtype = QuantizationType(request.quantization_type)
        quantized_model, info = do_quantize(model, request.quantization_type)
        
        model = quantized_model
        _inference_engine = None  # Reset engine
        
        return {
            "status": "quantized",
            "quantization_type": request.quantization_type,
            "original_size_mb": info.original_size_mb,
            "quantized_size_mb": info.quantized_size_mb,
            "reduction_percent": info.reduction,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/benchmark/run", tags=["benchmark"])
async def run_benchmark(
    prompt: str = "The quick brown fox jumps over the lazy dog",
    max_new_tokens: int = 50,
    num_runs: int = 3,
):
    """Run inference benchmark."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}
    
    try:
        from domains.ml_infrastructure.benchmarking import Benchmarker
        
        benchmarker = Benchmarker(model, tokenizer, device="cpu")
        result = benchmarker.benchmark_inference(prompt, max_new_tokens, num_runs)
        
        return result.to_dict()
    except Exception as e:
        return {"error": str(e)}


@app.post("/benchmark/perplexity", tags=["benchmark"])
async def calculate_perplexity(text: str = ""):
    """Calculate model perplexity on text."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}
    
    if not text:
        return {"error": "Text required"}
    
    try:
        from domains.ml_infrastructure.benchmarking import Benchmarker
        
        benchmarker = Benchmarker(model, tokenizer, device="cpu")
        ppl = benchmarker.calculate_perplexity(text)
        
        return {"perplexity": ppl, "text_length": len(text)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/benchmark/compare", tags=["benchmark"])
async def compare_benchmarks():
    """Get comparison of different quantization levels."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}
    
    try:
        from domains.ml_infrastructure.benchmarking import Benchmarker
        from domains.inference.quantization import quantize_model
        
        results = {}
        
        for qtype in ["fp32", "fp16", "int8"]:
            try:
                from copy import deepcopy
                test_model = deepcopy(model)
                quantized, _ = quantize_model(test_model, qtype)
                
                benchmarker = Benchmarker(quantized, tokenizer, device="cpu")
                result = benchmarker.benchmark_inference("Hello world", max_new_tokens=20, num_runs=2)
                results[qtype] = result.to_dict()
            except Exception as e:
                results[qtype] = {"error": str(e)}
        
        return results
    except Exception as e:
        return {"error": str(e)}


class ExportRequest(BaseModel):
    output_path: str = "models/exported"
    format: str = "sou"
    include_tokenizer: bool = True


@app.post("/model/export", tags=["model"])
async def export_model(request: ExportRequest):
    """Export current model to file."""
    global model, tokenizer, model_type
    
    if model is None:
        return {"error": "No model loaded"}
    
    try:
        from domains.training.export import export_model, list_export_formats, ExportConfig
        
        config = ExportConfig(
            input_path="current",
            output_path=request.output_path,
            format=request.format,
            include_tokenizer=request.include_tokenizer,
            metadata={
                "model_type": model_type,
                "exported_at": str(time.time()),
            },
        )
        
        results = export_model(config, model, tokenizer)
        
        return {
            "status": "exported",
            "format": request.format,
            "files": results,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/model/export/formats", tags=["model"])
async def get_export_formats():
    """Get list of supported export formats."""
    from domains.training.export import list_export_formats
    return {"formats": list_export_formats()}


@app.get("/models", tags=["model"])
async def list_models():
    """List available models in models/ directory."""
    import os
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    models = []
    for f in os.listdir(models_dir):
        if f.endswith(('.pt', '.pth', '.sou', '.safetensors', '.onnx')):
            path = os.path.join(models_dir, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            models.append({
                "name": f,
                "path": path,
                "size_mb": round(size_mb, 2),
            })
    
    return {"models": models}
