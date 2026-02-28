"""
SloughGPT API Server - Simplified Version
"""

import os
import sys
import psutil
import time
import uuid
import json
import hashlib
import asyncio
from pathlib import Path
from typing import Optional


# Environment configuration
class Config:
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    RATE_LIMIT: int = int(os.getenv("RATE_LIMIT", "60"))
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "30"))
    MAX_PAYLOAD_SIZE: int = int(os.getenv("MAX_PAYLOAD_SIZE", "10485760"))  # 10MB
    
    @classmethod
    def to_dict(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and not callable(v) and k != 'to_dict'}


config = Config()
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
from enum import Enum
import threading
from functools import lru_cache

from fastapi import FastAPI, HTTPException, WebSocket, Request, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator


# Rate limiting
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        with self.lock:
            self.requests[client_id] = [t for t in self.requests[client_id] if t > minute_ago]
            if len(self.requests[client_id]) >= self.requests_per_minute:
                return False
            self.requests[client_id].append(now)
            return True
    
    def get_remaining(self, client_id: str) -> int:
        now = time.time()
        minute_ago = now - 60
        with self.lock:
            self.requests[client_id] = [t for t in self.requests[client_id] if t > minute_ago]
            return max(0, self.requests_per_minute - len(self.requests[client_id]))


# Simple in-memory cache
class Cache:
    def __init__(self, ttl: int = 60):
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[any]:
        with self.lock:
            if key in self.cache:
                if time.time() - self.timestamps[key] < self.ttl:
                    return self.cache[key]
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value: any):
        with self.lock:
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()


rate_limiter = RateLimiter(requests_per_minute=60)
cache = Cache(ttl=30)

# Usage statistics
class UsageStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_requests = 0
        self.endpoint_calls = defaultdict(int)
        self.errors = 0
        self.start_time = time.time()
    
    def record_request(self, endpoint: str):
        with self.lock:
            self.total_requests += 1
            self.endpoint_calls[endpoint] += 1
    
    def record_error(self):
        with self.lock:
            self.errors += 1
    
    def get_stats(self):
        with self.lock:
            return {
                "total_requests": self.total_requests,
                "errors": self.errors,
                "endpoint_calls": dict(self.endpoint_calls),
                "uptime_seconds": time.time() - self.start_time,
                "requests_per_minute": self.total_requests / max(1, (time.time() - self.start_time) / 60)
            }


usage_stats = UsageStats()


# API Version
API_VERSION = "2.0.0"
API_PREFIX = "/api/v1"


class APIVersion(str, Enum):
    V1 = "v1"


app = FastAPI(
    title="SloughGPT API", 
    version=API_VERSION,
    description="Enterprise AI Framework API with models, training, and real-time capabilities",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    docs=dict(
        description="SloughGPT API Documentation - Enterprise AI Framework"
    )
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# GZip compression
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Custom error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    usage_stats.record_error()
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": str(request.url.path)
        }
    )


@app.on_event("startup")
async def startup_event():
    pass  # WebSocket broadcasting can be enabled here when needed


# Request logging and rate limiting middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    
    client_id = request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_id):
        usage_stats.record_error()
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded", "retry_after": 60, "request_id": request_id}
        )
    
    start_time = time.time()
    status_code = 200
    try:
        response = await call_next(request)
        status_code = response.status_code
        if status_code >= 400:
            usage_stats.record_error()
        remaining = rate_limiter.get_remaining(client_id)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-RateLimit-Limit"] = "60"
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
    except Exception:
        usage_stats.record_error()
        raise
    finally:
        duration = time.time() - start_time
        usage_stats.record_request(request.url.path)
        print(f"{request.method} {request.url.path} - {status_code} - {duration:.3f}s")


# API Key Authentication
API_KEYS = {
    "sk-test-1234567890abcdef": {"name": "test-key", "rate_limit": 100},
    "sk-prod-abcdef1234567890": {"name": "production", "rate_limit": 1000},
}

# Get API key from header
def get_api_key(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]
    return None


# Middleware for API key verification (optional)
@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    # Skip auth for docs, health, and WebSocket
    if request.url.path in ["/docs", "/redoc", "/openapi.json", "/health", "/"] or request.url.path.startswith("/ws"):
        return await call_next(request)
    
    api_key = get_api_key(request)
    if api_key and api_key in API_KEYS:
        request.state.api_key = api_key
        request.state.key_info = API_KEYS[api_key]
    
    return await call_next(request)


# Security: Input sanitization
import re
import html

def sanitize_input(text: str) -> str:
    if not text:
        return text
    text = html.escape(text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return text


def sanitize_dict(data: dict) -> dict:
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, str):
            sanitized[key] = sanitize_input(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value)
        elif isinstance(value, list):
            sanitized[key] = [sanitize_input(v) if isinstance(v, str) else v for v in value]
        else:
            sanitized[key] = value
    return sanitized


# Pydantic Models
VALID_MODELS = ["gpt-3.5-turbo", "gpt-4", "claude-3", "nanogpt", "llama-2-7b", "mixtral-8x7b"]

class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    model: str = "gpt-3.5-turbo"
    conversation_id: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=32000)
    
    @field_validator('message', mode='before')
    @classmethod
    def sanitize_message(cls, v):
        if isinstance(v, str):
            return sanitize_input(v)
        return v
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if v not in VALID_MODELS:
            raise ValueError(f'Invalid model. Must be one of: {", ".join(VALID_MODELS)}')
        return v


class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, pattern=r'^[a-zA-Z0-9_-]+$')
    content: str = Field(..., min_length=1)
    description: Optional[str] = Field(default=None, max_length=500)


class TrainingConfig(BaseModel):
    dataset_name: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=8, ge=1, le=256)
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-2)
    vocab_size: int = Field(default=500, ge=100, le=100000)
    n_embed: int = Field(default=128, ge=32, le=2048)
    n_layer: int = Field(default=3, ge=1, le=24)
    n_head: int = Field(default=4, ge=1, le=16)
    optimizer: str = Field(default="adam", pattern="^(adam|sgd|adamw)$")
    scheduler: Optional[str] = Field(default="cosine", pattern="^(cosine|step|exponential|None)$")
    validation_split: float = Field(default=0.1, ge=0.0, le=0.5)
    early_stopping_patience: int = Field(default=5, ge=0, le=20)
    save_checkpoint_every: int = Field(default=1, ge=1, le=10)
    gradient_clip: float = Field(default=1.0, ge=0.1, le=10.0)
    warmup_steps: int = Field(default=100, ge=0, le=1000)
    weight_decay: float = Field(default=0.01, ge=0.0, le=0.5)
    # CPU optimization parameters
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=8)
    use_mixed_precision: bool = Field(default=False)
    num_workers: int = Field(default=1, ge=1, le=4)


# Storage
class Storage:
    def __init__(self):
        self.conversations = defaultdict(list)
        self.conversation_metadata = {}
        self.datasets = {}
        self.training_jobs = {}
        self.start_time = time.time()
        self.datasets_dir = Path("datasets")
        self.datasets_dir.mkdir(exist_ok=True)
        self._load_datasets()
    
    def _load_datasets(self):
        if not self.datasets_dir.exists():
            return
        for item in self.datasets_dir.iterdir():
            if item.is_dir():
                self.datasets[item.name] = {
                    "name": item.name,
                    "path": str(item),
                    "size": sum(f.stat().st_size for f in item.rglob("*") if f.is_file()),
                    "created_at": datetime.fromtimestamp(item.stat().st_ctime).isoformat(),
                    "updated_at": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                    "has_train": (item / "train.bin").exists(),
                    "has_val": (item / "val.bin").exists(),
                    "has_meta": (item / "meta.json").exists(),
                    "description": ""
                }
    
    def create_conversation(self, name=None):
        conv_id = f"conv_{uuid.uuid4().hex[:8]}"
        self.conversations[conv_id] = []
        self.conversation_metadata[conv_id] = {
            "id": conv_id,
            "name": name or f"Conversation {len(self.conversation_metadata) + 1}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        return self.conversation_metadata[conv_id]
    
    def list_conversations(self):
        return [
            {**meta, "message_count": len(self.conversations[conv_id])}
            for conv_id, meta in self.conversation_metadata.items()
        ]
    
    def add_message(self, conv_id, role, content):
        msg = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        self.conversations[conv_id].append(msg)
        return msg
    
    def create_dataset(self, name, content, description=None):
        path = self.datasets_dir / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "input.txt").write_text(content)
        (path / "meta.json").write_text('{"description": ""}')
        
        ds = {
            "name": name,
            "path": str(path),
            "size": len(content),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "has_train": False,
            "has_val": False,
            "has_meta": True,
            "description": description or ""
        }
        self.datasets[name] = ds
        return ds
    
    def delete_dataset(self, name):
        if name not in self.datasets:
            return False
        import shutil
        shutil.rmtree(self.datasets_dir / name)
        del self.datasets[name]
        return True
    
    def create_training_job(self, config):
        job_id = f"train_{uuid.uuid4().hex[:8]}"
        
        # Get dataset size for realistic simulation
        dataset_info = self.datasets.get(config.dataset_name, {})
        dataset_size = dataset_info.get("size", 1000)
        
        job = {
            "id": job_id,
            "status": "pending",
            "dataset_name": config.dataset_name,
            "model_id": config.model_id,
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": config.epochs,
            "loss": 0.0,
            "val_loss": None,
            "best_loss": float('inf'),
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "optimizer": config.optimizer,
            "scheduler": config.scheduler,
            "gradient_clip": config.gradient_clip,
            "weight_decay": config.weight_decay,
            "warmup_steps": config.warmup_steps,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "use_mixed_precision": config.use_mixed_precision,
            "num_workers": config.num_workers,
            "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
            "steps_per_epoch": max(10, dataset_size // config.batch_size),
            "total_steps": 0,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "metrics_history": [],
            "checkpoints": [],
            "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_counter": 0,
            "save_checkpoint_every": config.save_checkpoint_every,
            "epochs_without_improvement": 0
        }
        self.training_jobs[job_id] = job
        # Simulate training in background
        thread = threading.Thread(target=self._simulate_training, args=(job_id,))
        thread.daemon = True
        thread.start()
        return job
    
    def _simulate_training(self, job_id):
        import random
        import math
        
        job = self.training_jobs.get(job_id)
        if not job:
            return
        
        job["status"] = "running"
        job["started_at"] = datetime.now().isoformat()
        
        # Training configuration
        initial_lr = job.get("learning_rate", 1e-4)
        scheduler = job.get("scheduler", "cosine")
        total_epochs = job["total_epochs"]
        batch_size = job.get("batch_size", 8)
        gradient_clip = job.get("gradient_clip", 1.0)
        
        # CPU optimization parameters
        gradient_accumulation_steps = job.get("gradient_accumulation_steps", 1)
        use_mixed_precision = job.get("use_mixed_precision", False)
        
        # Calculate effective batch size
        effective_batch_size = batch_size * gradient_accumulation_steps
        
        # Simulate dataset size based on config
        base_steps = 100
        steps_per_epoch = max(10, base_steps // (effective_batch_size // 8))
        
        # Initialize training metrics
        current_loss = 3.5
        best_loss = float('inf')
        momentum = 0.9
        velocity = 0
        
        for epoch in range(1, total_epochs + 1):
            # Check if cancelled
            if job.get("status") == "cancelled":
                return
            
            # Calculate learning rate based on scheduler with warmup
            warmup_epochs = job.get("warmup_steps", 0) // max(1, steps_per_epoch)
            if epoch <= warmup_epochs:
                # Linear warmup
                lr = initial_lr * epoch / max(1, warmup_epochs)
            elif scheduler == "cosine":
                # Cosine annealing with warmup
                progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
                lr = initial_lr * 0.5 * (1 + math.cos(math.pi * progress))
            elif scheduler == "step":
                lr = initial_lr * (0.5 ** ((epoch - warmup_epochs) // 2))
            elif scheduler == "exponential":
                lr = initial_lr * (0.9 ** (epoch - warmup_epochs))
            else:
                lr = initial_lr
            
            job["learning_rate"] = lr
            
            # Simulate steps with gradient accumulation
            epoch_loss = 0
            accumulated_loss = 0
            
            for step in range(steps_per_epoch):
                # Simulate loss with SGD/Adam dynamics
                # Use momentum for more realistic loss curves
                grad = current_loss * 0.1 + random.uniform(-0.05, 0.05)
                
                # Apply gradient clipping
                grad = max(-gradient_clip, min(gradient_clip, grad))
                
                # Apply momentum (Adam-like)
                velocity = momentum * velocity + (1 - momentum) * grad
                
                # Update loss with learning rate
                current_loss = max(0.01, current_loss - lr * velocity)
                
                # Accumulate for gradient accumulation
                accumulated_loss += current_loss
                
                # Update weights every gradient_accumulation_steps
                if (step + 1) % gradient_accumulation_steps == 0:
                    epoch_loss += accumulated_loss / gradient_accumulation_steps
                    accumulated_loss = 0
            
            # Average loss for epoch
            avg_loss = epoch_loss / steps_per_epoch
            job["loss"] = round(avg_loss, 4)
            job["total_steps"] = epoch * steps_per_epoch
            
            # Simulate validation loss with overfitting gap
            overfit_factor = random.uniform(1.1, 1.4)
            job["val_loss"] = round(avg_loss * overfit_factor, 4)
            
            # Track best loss for early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                job["best_loss"] = round(best_loss, 4)
                job["epochs_without_improvement"] = 0
            else:
                job["epochs_without_improvement"] = job.get("epochs_without_improvement", 0) + 1
            
            # Early stopping check
            patience = job.get("early_stopping_patience", 5)
            if job.get("epochs_without_improvement", 0) >= patience and patience > 0:
                job["status"] = "early_stopped"
                job["completed_at"] = datetime.now().isoformat()
                job["error"] = f"Early stopping triggered after {patience} epochs without improvement"
                return
            
            # Save checkpoint
            save_every = job.get("save_checkpoint_every", 1)
            if epoch % save_every == 0 or epoch == total_epochs:
                checkpoint = {
                    "epoch": epoch,
                    "loss": job["loss"],
                    "val_loss": job["val_loss"],
                    "best_loss": job["best_loss"],
                    "learning_rate": lr,
                    "timestamp": datetime.now().isoformat(),
                    "path": f"checkpoints/{job_id}/epoch_{epoch}.pt"
                }
                job["checkpoints"].append(checkpoint)
            
            # Update progress
            job["current_epoch"] = epoch
            job["progress"] = round((epoch / total_epochs) * 100, 1)
            
            # Store metrics history
            job["metrics_history"].append({
                "epoch": epoch,
                "loss": job["loss"],
                "val_loss": job["val_loss"],
                "learning_rate": job["learning_rate"],
                "timestamp": datetime.now().isoformat()
            })
            
            # Simulate epoch time (faster for larger batch sizes)
            time.sleep(max(0.1, 0.5 - (effective_batch_size / 1000)))
        
        job["status"] = "completed"
        job["progress"] = 100.0
        job["completed_at"] = datetime.now().isoformat()
    
    def get_metrics(self):
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net = psutil.net_io_counters()
        return {
            "cpu_percent": psutil.cpu_percent(0.1),
            "memory_percent": mem.percent,
            "memory_used_mb": mem.used / 1024**2,
            "memory_total_mb": mem.total / 1024**2,
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / 1024**3,
            "disk_total_gb": disk.total / 1024**3,
            "network_sent_mb": net.bytes_sent / 1024**2,
            "network_recv_mb": net.bytes_recv / 1024**2,
            "timestamp": datetime.now().isoformat()
        }


storage = Storage()

# API Versioning with routers
from fastapi import APIRouter

v1_router = APIRouter(prefix="/api/v1")


# Versioned endpoints
@v1_router.get("/health")
async def v1_health():
    return {
        "status": "healthy",
        "version": API_VERSION,
        "uptime_seconds": time.time() - storage.start_time
    }


@v1_router.get("/models")
async def v1_models():
    cached = cache.get("models")
    if cached is not None:
        return cached
    models = [
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "OpenAI", "status": "available"},
        {"id": "gpt-4", "name": "GPT-4", "provider": "OpenAI", "status": "available"},
        {"id": "claude-3", "name": "Claude 3", "provider": "Anthropic", "status": "available"},
        {"id": "nanogpt", "name": "NanoGPT", "provider": "Local", "status": "available"},
        {"id": "llama-2-7b", "name": "Llama 2 7B", "provider": "Meta", "status": "available"},
        {"id": "mixtral-8x7b", "name": "Mixtral 8x7B", "provider": "Mistral", "status": "available"}
    ]
    result = {"models": models}
    cache.set("models", result)
    return result


@v1_router.post("/chat")
async def v1_chat(req: ChatMessageRequest):
    conv_id = req.conversation_id or "conv_default"
    storage.add_message(conv_id, "user", req.message)
    response = f"Echo: {req.message}\n\n(Model: {req.model}, Temp: {req.temperature})"
    storage.add_message(conv_id, "assistant", response)
    return {
        "conversation_id": conv_id,
        "message": {"role": "assistant", "content": response},
        "model": req.model
    }


app.include_router(v1_router)


# Endpoints
@app.get("/")
async def root():
    return {"message": "SloughGPT API", "version": "2.0.0", "docs": "/docs"}


@app.get("/health", tags=["Health"])
async def health():
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    return {
        "status": "healthy",
        "version": API_VERSION,
        "uptime_seconds": time.time() - storage.start_time,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "healthy",
            "database": "healthy",
            "cache": "healthy",
            "model_service": "healthy"
        },
        "resources": {
            "memory_percent": round(mem.percent, 1),
            "memory_available_mb": round(mem.available / 1024**2, 1),
            "disk_percent": round(disk.percent, 1),
            "disk_free_gb": round(disk.free / 1024**3, 1)
        }
    }


@app.get("/metrics", tags=["System"])
async def metrics():
    cached = cache.get("metrics")
    if cached is not None:
        return cached
    result = storage.get_metrics()
    cache.set("metrics", result)
    return result


@app.get("/info")
async def info():
    return {
        "python_version": "3.10+",
        "platform": os.name,
        "cpu_count": psutil.cpu_count(),
        "total_memory_gb": psutil.virtual_memory().total / 1024**3,
        "disk_total_gb": psutil.disk_usage('/').total / 1024**3,
        "conversations_count": len(storage.conversations),
        "datasets_count": len(storage.datasets),
        "training_jobs_count": len(storage.training_jobs)
    }


@app.get("/models", tags=["Models"])
async def list_models():
    cached = cache.get("models")
    if cached is not None:
        return cached
    models = [
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "OpenAI", "status": "available", "description": "Fast, cost-effective", "context_length": 4096},
        {"id": "gpt-4", "name": "GPT-4", "provider": "OpenAI", "status": "available", "description": "Most capable", "context_length": 8192},
        {"id": "claude-3", "name": "Claude 3", "provider": "Anthropic", "status": "available", "description": "Helpful assistant", "context_length": 200000},
        {"id": "nanogpt", "name": "NanoGPT", "provider": "Local", "status": "available", "description": "Custom trained", "context_length": 512},
        {"id": "llama-2-7b", "name": "Llama 2 7B", "provider": "Meta", "status": "available", "description": "Open source", "context_length": 4096},
        {"id": "mixtral-8x7b", "name": "Mixtral 8x7B", "provider": "Mistral", "status": "available", "description": "MoE model", "context_length": 32000}
    ]
    result = {"models": models}
    cache.set("models", result)
    return result


MODELS_DETAIL = {
    "gpt-3.5-turbo": {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "OpenAI", "context_length": 4096, "pricing": {"prompt": 0.0015, "completion": 0.002}, "capabilities": ["chat", "completion"]},
    "gpt-4": {"id": "gpt-4", "name": "GPT-4", "provider": "OpenAI", "context_length": 8192, "pricing": {"prompt": 0.03, "completion": 0.06}, "capabilities": ["chat", "completion", "tools"]},
    "claude-3": {"id": "claude-3", "name": "Claude 3", "provider": "Anthropic", "context_length": 200000, "pricing": {"prompt": 0.015, "completion": 0.075}, "capabilities": ["chat", "completion", "vision"]},
    "nanogpt": {"id": "nanogpt", "name": "NanoGPT", "provider": "Local", "context_length": 512, "pricing": {"prompt": 0, "completion": 0}, "capabilities": ["completion"]},
    "llama-2-7b": {"id": "llama-2-7b", "name": "Llama 2 7B", "provider": "Meta", "context_length": 4096, "pricing": {"prompt": 0, "completion": 0}, "capabilities": ["chat", "completion"]},
    "mixtral-8x7b": {"id": "mixtral-8x7b", "name": "Mixtral 8x7B", "provider": "Mistral", "context_length": 32000, "pricing": {"prompt": 0, "completion": 0}, "capabilities": ["chat", "completion"]},
}


@app.get("/models/{model_id}", tags=["Models"])
async def get_model(model_id: str):
    if model_id not in MODELS_DETAIL:
        raise HTTPException(404, "Model not found")
    return MODELS_DETAIL[model_id]


@app.get("/datasets", tags=["Datasets"])
async def list_datasets(
    page: int = 1, 
    limit: int = 10,
    search: Optional[str] = None,
    has_train: Optional[bool] = None,
    has_val: Optional[bool] = None
):
    all_datasets = list(storage.datasets.values())
    
    # Filter
    if search:
        search_lower = search.lower()
        all_datasets = [d for d in all_datasets if search_lower in d.get("name", "").lower()]
    if has_train is not None:
        all_datasets = [d for d in all_datasets if d.get("has_train") == has_train]
    if has_val is not None:
        all_datasets = [d for d in all_datasets if d.get("has_val") == has_val]
    
    total = len(all_datasets)
    start = (page - 1) * limit
    end = start + limit
    items = all_datasets[start:end]
    return {
        "datasets": items,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (total + limit - 1) // limit
        }
    }


@app.post("/datasets", tags=["Datasets"])
async def create_dataset(data: DatasetCreate):
    if data.name in storage.datasets:
        raise HTTPException(400, "Dataset already exists")
    return storage.create_dataset(data.name, data.content, data.description)


@app.delete("/datasets/{name}")
async def delete_dataset(name: str):
    if not storage.delete_dataset(name):
        raise HTTPException(404, "Dataset not found")
    return {"status": "deleted", "name": name}


@app.get("/datasets/{name}", tags=["Datasets"])
async def get_dataset(name: str):
    if name not in storage.datasets:
        raise HTTPException(404, "Dataset not found")
    dataset = storage.datasets[name]
    # Add metadata
    path = Path(dataset["path"])
    files = []
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file():
                files.append({"name": f.name, "size": f.stat().st_size})
    return {
        **dataset,
        "files": files
    }


# Export/Import endpoints
@app.get("/datasets/{name}/export", tags=["Datasets"])
async def export_dataset(name: str):
    if name not in storage.datasets:
        raise HTTPException(404, "Dataset not found")
    dataset = storage.datasets[name]
    path = Path(dataset["path"])
    if not path.exists():
        raise HTTPException(404, "Dataset path not found")
    # Return as JSON for simplicity
    import json
    data = {"name": name, "files": []}
    for f in path.rglob("*"):
        if f.is_file():
            data["files"].append({"name": str(f.relative_to(path)), "size": f.stat().st_size})
    return data


class DatasetImport(BaseModel):
    name: str
    data: Dict[str, str]


@app.post("/datasets/import", tags=["Datasets"])
async def import_dataset(req: DatasetImport):
    path = storage.datasets_dir / req.name
    path.mkdir(parents=True, exist_ok=True)
    for filename, content in req.data.items():
        (path / filename).write_text(content)
    storage._load_datasets()
    return {"status": "imported", "name": req.name}


# Webhook management
webhooks: Dict[str, Dict] = {}


@app.get("/webhooks", tags=["Webhooks"])
async def list_webhooks():
    return {"webhooks": list(webhooks.values())}


class WebhookCreate(BaseModel):
    url: str
    events: List[str]
    name: Optional[str] = None


@app.post("/webhooks", tags=["Webhooks"])
async def create_webhook(req: WebhookCreate):
    import secrets
    webhook_id = secrets.token_hex(8)
    webhook = {
        "id": webhook_id,
        "url": req.url,
        "events": req.events,
        "name": req.name or f"Webhook {webhook_id[:8]}",
        "active": True
    }
    webhooks[webhook_id] = webhook
    return webhook


@app.delete("/webhooks/{webhook_id}", tags=["Webhooks"])
async def delete_webhook(webhook_id: str):
    if webhook_id not in webhooks:
        raise HTTPException(404, "Webhook not found")
    del webhooks[webhook_id]
    return {"status": "deleted", "webhook_id": webhook_id}


@app.post("/webhooks/{webhook_id}/test", tags=["Webhooks"])
async def test_webhook(webhook_id: str):
    if webhook_id not in webhooks:
        raise HTTPException(404, "Webhook not found")
    return {"status": "sent", "webhook_id": webhook_id}


@app.post("/conversations")
async def create_conversation():
    return storage.create_conversation()


@app.get("/conversations")
async def list_conversations():
    return {"conversations": storage.list_conversations()}


@app.post("/chat", tags=["Chat"])
async def chat(req: ChatMessageRequest):
    conv_id = req.conversation_id or "conv_default"
    storage.add_message(conv_id, "user", req.message)
    response = f"Echo: {req.message}\n\n(Model: {req.model}, Temp: {req.temperature})"
    storage.add_message(conv_id, "assistant", response)
    return {
        "conversation_id": conv_id,
        "message": {"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()},
        "model": req.model,
        "usage": {"prompt_tokens": len(req.message.split()), "completion_tokens": len(response.split()), "total_tokens": len(req.message.split()) + len(response.split())}
    }


@app.get("/training", tags=["Training"])
async def list_training(status: Optional[str] = None):
    jobs = list(storage.training_jobs.values())
    if status:
        jobs = [j for j in jobs if j.get("status") == status]
    return {"jobs": jobs, "total": len(jobs)}


@app.get("/training/recommendations", tags=["Training"])
async def get_training_recommendations():
    """Get training recommendations based on system state"""
    import random
    
    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent()
    disk = psutil.disk_usage('/')
    
    recommendations = []
    
    # Memory-based recommendations
    if mem.percent > 80:
        recommendations.append({
            "type": "warning",
            "category": "memory",
            "message": "High memory usage detected. Consider reducing batch size.",
            "suggested_batch_size": max(4, 32 - int((mem.percent - 80) * 2))
        })
    
    # CPU-based recommendations
    if cpu_percent > 70:
        recommendations.append({
            "type": "info",
            "category": "cpu",
            "message": "High CPU usage. Consider reducing model size or using gradient accumulation.",
            "suggested_batch_size": max(8, 64 - int((cpu_percent - 70) * 2))
        })
    
    # Disk-based recommendations
    if disk.percent > 90:
        recommendations.append({
            "type": "warning",
            "category": "disk",
            "message": "Low disk space. Consider deleting old checkpoints.",
            "action": "cleanup_checkpoints"
        })
    
    # Default recommendations
    if not recommendations:
        recommendations.append({
            "type": "success",
            "category": "system",
            "message": "System is running optimally for training.",
            "suggested_batch_size": 32,
            "suggested_lr": 0.001
        })
    
    return {
        "recommendations": recommendations,
        "system_state": {
            "memory_percent": round(mem.percent, 1),
            "cpu_percent": round(cpu_percent, 1),
            "disk_percent": round(disk.percent, 1)
        }
    }


@app.get("/training/compare", tags=["Training"])
async def compare_training(job_ids: str):
    """Compare multiple training jobs"""
    ids = job_ids.split(",")
    jobs = []
    for job_id in ids:
        job = storage.training_jobs.get(job_id.strip())
        if job:
            jobs.append({
                "id": job["id"],
                "dataset_name": job["dataset_name"],
                "model_id": job.get("model_id"),
                "status": job["status"],
                "total_epochs": job.get("total_epochs"),
                "final_loss": job.get("loss"),
                "best_loss": job.get("best_loss"),
                "val_loss": job.get("val_loss"),
                "optimizer": job.get("optimizer"),
                "scheduler": job.get("scheduler"),
                "learning_rate": job.get("learning_rate"),
                "completed_at": job.get("completed_at"),
                "duration_seconds": (
                    datetime.fromisoformat(job["completed_at"]).timestamp() - 
                    datetime.fromisoformat(job["started_at"]).timestamp()
                ) if job.get("started_at") and job.get("completed_at") else None
            })
    
    return {"jobs": jobs, "total": len(jobs)}


@app.post("/training", tags=["Training"])
async def create_training(config: TrainingConfig):
    if config.dataset_name not in storage.datasets:
        raise HTTPException(400, "Dataset not found")
    return storage.create_training_job(config)


@app.get("/training/{job_id}", tags=["Training"])
async def get_training(job_id: str):
    job = storage.training_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Training job not found")
    return job


@app.delete("/training/{job_id}", tags=["Training"])
async def cancel_training(job_id: str):
    job = storage.training_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Training job not found")
    if job["status"] == "completed":
        raise HTTPException(400, "Cannot cancel completed job")
    job["status"] = "cancelled"
    job["error"] = "Cancelled by user"
    return {"status": "cancelled", "job_id": job_id}


@app.post("/training/{job_id}/restart", tags=["Training"])
async def restart_training(job_id: str):
    """Restart a completed, failed, or cancelled training job"""
    job = storage.training_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Training job not found")
    
    if job["status"] == "running":
        raise HTTPException(400, "Job is already running")
    
    # Get initial LR (ensure minimum of 1e-4)
    initial_lr = max(job.get("learning_rate", 1e-4), 1e-4)
    
    # Create new job with same config
    config = TrainingConfig(
        dataset_name=job["dataset_name"],
        model_id=job.get("model_id", "nanogpt"),
        epochs=job.get("total_epochs", 3),
        batch_size=job.get("batch_size", 8),
        learning_rate=initial_lr,
        optimizer=job.get("optimizer", "adam"),
        scheduler=job.get("scheduler", "cosine"),
        early_stopping_patience=job.get("early_stopping_patience", 5),
        save_checkpoint_every=job.get("save_checkpoint_every", 1),
        gradient_clip=job.get("gradient_clip", 1.0),
        weight_decay=job.get("weight_decay", 0.01)
    )
    
    new_job = storage.create_training_job(config)
    return {
        "new_job_id": new_job["id"],
        "original_job_id": job_id,
        "status": "restarted"
    }


@app.get("/training/{job_id}/logs", tags=["Training"])
async def get_training_logs(job_id: str):
    job = storage.training_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Training job not found")
    
    # Return actual metrics history
    logs = job.get("metrics_history", [])
    
    return {
        "job_id": job_id,
        "logs": logs,
        "summary": {
            "initial_loss": logs[0]["loss"] if logs else None,
            "final_loss": logs[-1]["loss"] if logs else None,
            "best_loss": job.get("best_loss"),
            "total_steps": job.get("total_steps", 0),
            "learning_rate_final": job.get("learning_rate")
        }
    }


@app.get("/training/{job_id}/metrics", tags=["Training"])
async def get_training_metrics(job_id: str):
    """Get training metrics formatted for visualization charts"""
    job = storage.training_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Training job not found")
    
    history = job.get("metrics_history", [])
    
    # Format data for charts
    epochs = [h["epoch"] for h in history]
    train_loss = [h["loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    learning_rates = [h["learning_rate"] for h in history]
    
    return {
        "job_id": job_id,
        "chart_data": {
            "epochs": epochs,
            "loss": {
                "train": train_loss,
                "validation": val_loss
            },
            "learning_rate": learning_rates
        },
        "stats": {
            "total_epochs": job.get("total_epochs", 0),
            "best_train_loss": min(train_loss) if train_loss else None,
            "best_val_loss": min(val_loss) if val_loss else None,
            "avg_train_loss": sum(train_loss) / len(train_loss) if train_loss else None,
            "avg_val_loss": sum(val_loss) / len(val_loss) if val_loss else None,
            "final_learning_rate": job.get("learning_rate")
        }
    }


@app.get("/training/{job_id}/export", tags=["Training"])
async def export_training_job(job_id: str):
    """Export training job details and metrics"""
    job = storage.training_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Training job not found")
    
    return {
        "job": job,
        "config": {
            "dataset_name": job["dataset_name"],
            "model_id": job.get("model_id"),
            "epochs": job.get("total_epochs"),
            "batch_size": job.get("batch_size"),
            "learning_rate": job.get("learning_rate"),
            "optimizer": job.get("optimizer"),
            "scheduler": job.get("scheduler"),
            "gradient_clip": job.get("gradient_clip"),
            "weight_decay": job.get("weight_decay")
        },
        "metrics": {
            "final_loss": job.get("loss"),
            "best_loss": job.get("best_loss"),
            "val_loss": job.get("val_loss"),
            "total_steps": job.get("total_steps")
        },
        "checkpoints": job.get("checkpoints", []),
        "history": job.get("metrics_history", [])
    }


# Training history storage
training_history: Dict[str, List[Dict]] = defaultdict(list)


@app.get("/training/{job_id}/history", tags=["Training"])
async def get_training_history(job_id: str):
    if job_id not in storage.training_jobs:
        raise HTTPException(404, "Training job not found")
    return {"job_id": job_id, "history": training_history.get(job_id, [])}


@app.post("/generate")
async def generate(prompt: str, model: str = "nanogpt", max_length: int = 100, temperature: float = 0.8):
    return {
        "text": f"Generated: {prompt[:50]}... (length: {max_length})",
        "model": model,
        "tokens_generated": max_length,
        "processing_time_ms": 10.5
    }


@app.get("/cache")
async def get_cache_info():
    return {
        "ttl_seconds": cache.ttl,
        "cached_keys": list(cache.cache.keys()),
        "rate_limit": {
            "requests_per_minute": rate_limiter.requests_per_minute
        }
    }


@app.delete("/cache")
async def clear_cache():
    cache.clear()
    return {"status": "cache cleared"}


# Batch processing models
class BatchGenerateRequest(BaseModel):
    prompts: List[str]
    model: str = "nanogpt"
    max_length: int = 100
    temperature: float = 0.8


class BatchChatRequest(BaseModel):
    messages: List[str]
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7


# Batch processing endpoints
@app.post("/batch/generate")
async def batch_generate(req: BatchGenerateRequest):
    results = []
    for prompt in req.prompts:
        results.append({
            "prompt": prompt,
            "text": f"Generated: {prompt[:50]}...",
            "model": req.model,
            "tokens_generated": req.max_length
        })
    return {"results": results, "total": len(results)}


@app.post("/batch/chat")
async def batch_chat(req: BatchChatRequest):
    results = []
    for message in req.messages:
        results.append({
            "message": message,
            "response": f"Echo: {message}\n(Model: {req.model})",
            "model": req.model
        })
    return {"results": results, "total": len(results)}


@app.get("/batch/status/{batch_id}")
async def get_batch_status(batch_id: str):
    return {
        "batch_id": batch_id,
        "status": "completed",
        "progress": 100,
        "completed": 10,
        "total": 10
    }


# API Key management
@app.get("/keys")
async def list_api_keys():
    return {
        "keys": [
            {"name": info["name"], "rate_limit": info["rate_limit"], "prefix": key[:10] + "..."}
            for key, info in API_KEYS.items()
        ]
    }


@app.post("/keys")
async def create_api_key(name: str, rate_limit: int = 100):
    import secrets
    key = "sk-" + secrets.token_hex(16)
    API_KEYS[key] = {"name": name, "rate_limit": rate_limit}
    return {"key": key, "name": name, "rate_limit": rate_limit}


# Usage statistics endpoint
@app.get("/stats/usage")
async def get_usage_stats():
    return usage_stats.get_stats()


# System info endpoint
@app.get("/info")
async def get_info():
    return {
        "api_version": API_VERSION,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": os.name,
        "cpu_count": psutil.cpu_count(),
        "total_memory_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "uptime_seconds": time.time() - storage.start_time,
        "features": {
            "rate_limiting": True,
            "caching": True,
            "websocket": True,
            "batch_processing": True,
            "api_keys": True
        }
    }


# Configuration endpoint
@app.get("/config")
async def get_config():
    return config.to_dict()


# Admin dashboard endpoint
@app.get("/admin/dashboard", tags=["Admin"])
async def admin_dashboard():
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    return {
        "api": {
            "version": API_VERSION,
            "status": "running"
        },
        "resources": {
            "memory": {"total_gb": round(mem.total / 1024**3, 2), "percent": mem.percent},
            "disk": {"total_gb": round(disk.total / 1024**3, 2), "percent": disk.percent},
            "cpu_count": psutil.cpu_count()
        },
        "storage": {
            "datasets_count": len(storage.datasets),
            "conversations_count": len(storage.conversations),
            "training_jobs_count": len(storage.training_jobs)
        },
        "usage": usage_stats.get_stats(),
        "webhooks_count": len(webhooks),
        "cache": {"ttl": cache.ttl, "keys": len(cache.cache)}
    }


# Model inference endpoint
class InferenceRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    stream: bool = False


@app.post("/inference", tags=["Inference"])
async def inference(req: InferenceRequest):
    if req.model not in MODELS_DETAIL:
        raise HTTPException(404, "Model not found")
    
    model_info = MODELS_DETAIL[req.model]
    
    if req.stream:
        # Return streaming response
        import random
        words = req.prompt.split()
        result_text = " ".join(words[:min(5, len(words))]) + " " + " ".join([f"word{i}" for i in range(req.max_tokens // 5)])
        return {"text": result_text, "model": req.model, "tokens": req.max_tokens}
    else:
        # Simulated inference
        result = f"Inference result for: {req.prompt[:50]}... (model: {model_info['name']})"
        return {
            "text": result,
            "model": req.model,
            "tokens": req.max_tokens,
            "usage": {
                "prompt_tokens": len(req.prompt.split()),
                "completion_tokens": req.max_tokens,
                "total_tokens": len(req.prompt.split()) + req.max_tokens
            }
        }


# Inference using trained checkpoints
class CheckpointInferenceRequest(BaseModel):
    model: str = "nanogpt"
    checkpoint_path: Optional[str] = None
    job_id: Optional[str] = None
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_length: int = Field(default=100, ge=1, le=1000)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


@app.post("/inference/checkpoint", tags=["Inference"])
async def inference_with_checkpoint(req: CheckpointInferenceRequest):
    # Get checkpoint info
    checkpoint_info = None
    checkpoint_path = None
    model_id = req.model
    epoch = None
    
    if req.job_id:
        job = storage.training_jobs.get(req.job_id)
        if not job:
            raise HTTPException(404, "Training job not found")
        if job.get("checkpoints"):
            checkpoint_info = job["checkpoints"][-1]  # Use latest checkpoint
            checkpoint_path = checkpoint_info.get("path")
            model_id = job.get("model_id", req.model)
            epoch = checkpoint_info.get("epoch")
    elif req.checkpoint_path:
        checkpoint_path = req.checkpoint_path
        model_id = req.model
        epoch = "custom"
    else:
        raise HTTPException(400, "Either job_id or checkpoint_path required")
    
    # Simulate inference with checkpoint
    import random
    words = req.prompt.split()
    
    # Generate text based on temperature
    if req.temperature < 0.5:
        # More deterministic
        generated = " ".join(words[:min(3, len(words))]) + " " + " ".join(["token"] * (req.max_length // 6))
    else:
        # More random
        generated = " ".join(words[:min(5, len(words))]) + " " + " ".join([f"w{i}" for i in range(req.max_length // 5)])
    
    return {
        "generated_text": generated,
        "checkpoint": checkpoint_path,
        "model": model_id,
        "epoch": epoch,
        "parameters": {
            "max_length": req.max_length,
            "temperature": req.temperature,
            "top_p": req.top_p
        },
        "usage": {
            "prompt_tokens": len(req.prompt.split()),
            "completion_tokens": req.max_length,
            "total_tokens": len(req.prompt.split()) + req.max_length
        }
    }


# List available checkpoints
@app.get("/checkpoints", tags=["Training"])
async def list_checkpoints():
    all_checkpoints = []
    for job_id, job in storage.training_jobs.items():
        if job.get("checkpoints"):
            for cp in job["checkpoints"]:
                all_checkpoints.append({
                    "job_id": job_id,
                    "dataset": job.get("dataset_name"),
                    "model": job.get("model_id"),
                    "epoch": cp.get("epoch"),
                    "loss": cp.get("loss"),
                    "val_loss": cp.get("val_loss"),
                    "path": cp.get("path"),
                    "timestamp": cp.get("timestamp")
                })
    return {"checkpoints": all_checkpoints, "total": len(all_checkpoints)}


# Model Evaluation
class EvaluationRequest(BaseModel):
    model: str
    checkpoint_path: Optional[str] = None
    job_id: Optional[str] = None
    test_data: List[str]
    metrics: Optional[List[str]] = None


@app.post("/evaluate", tags=["Models"])
async def evaluate_model(req: EvaluationRequest):
    """Evaluate model on test data"""
    import random
    
    # Get checkpoint info
    checkpoint_info = None
    checkpoint_path = None
    model_id = req.model
    epoch = None
    
    if req.job_id:
        job = storage.training_jobs.get(req.job_id)
        if job:
            checkpoint_info = job.get("checkpoints", [{}])[-1]
            checkpoint_path = checkpoint_info.get("path") if checkpoint_info else None
            model_id = job.get("model_id", req.model)
            epoch = checkpoint_info.get("epoch") if checkpoint_info else None
    elif req.checkpoint_path:
        checkpoint_path = req.checkpoint_path
    
    # Calculate metrics
    num_samples = len(req.test_data)
    
    # Simulated evaluation metrics
    perplexity = random.uniform(1.5, 5.0)
    accuracy = random.uniform(0.6, 0.95)
    f1_score = random.uniform(0.5, 0.9)
    
    # Per-character metrics
    char_accuracy = random.uniform(0.7, 0.98)
    edit_distance = random.uniform(0, 10)
    
    return {
        "model": model_id,
        "checkpoint": checkpoint_path,
        "epoch": epoch,
        "num_samples": num_samples,
        "metrics": {
            "perplexity": round(perplexity, 4),
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1_score, 4),
            "char_accuracy": round(char_accuracy, 4),
            "avg_edit_distance": round(edit_distance, 2)
        },
        "evaluation_time_seconds": round(random.uniform(0.5, 5.0), 2)
    }


from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = str(uuid.uuid4())
    await manager.connect(client_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await manager.send_message(client_id, {"type": "pong", "timestamp": time.time()})
            elif message.get("type") == "subscribe_training":
                await manager.send_message(client_id, {"type": "subscribed", "job_id": message.get("job_id")})
            elif message.get("type") == "unsubscribe":
                pass
    except WebSocketDisconnect:
        manager.disconnect(client_id)


__all__ = ["app"]
