#!/usr/bin/env python3
"""
SloughGPT Model Server
FastAPI server for model inference with HuggingFace fallback.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
import json
import asyncio


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
        prompt = format_chat_prompt([m.dict() for m in request.messages])

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
    # Simple status - could be enhanced with proper job tracking
    return {"status": "ready", "message": "Use /train endpoint to start training"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
