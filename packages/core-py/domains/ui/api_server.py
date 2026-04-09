"""
SloughGPT API Server
Comprehensive FastAPI backend for the SloughGPT AI Framework

Version: 2.0.0
Documentation: http://localhost:8000/docs
"""

__version__ = "2.0.0"

import os
import shutil
import json
import psutil
import time
import uuid
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from functools import wraps
import threading
import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Rate limiting
class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_requests))
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        with self._lock:
            now = datetime.now()
            client_requests = self.requests[client_id]

            # Remove old requests outside the window
            cutoff = now - timedelta(seconds=self.window_seconds)
            while client_requests and client_requests[0] < cutoff:
                client_requests.popleft()

            # Check if under limit
            if len(client_requests) < self.max_requests:
                client_requests.append(now)
                return True

            return False

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        with self._lock:
            now = datetime.now()
            client_requests = self.requests[client_id]
            cutoff = now - timedelta(seconds=self.window_seconds)

            while client_requests and client_requests[0] < cutoff:
                client_requests.popleft()

            return max(0, self.max_requests - len(client_requests))


# Initialize rate limiter
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)


# =============================================================================
# Cache System
# =============================================================================


class Cache:
    """Simple in-memory cache with TTL support."""

    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if entry["expires_at"] and entry["expires_at"] < time.time():
                del self._cache[key]
                return None

            return entry["value"]

    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL (seconds)."""
        with self._lock:
            self._cache[key] = {
                "value": value,
                "created_at": time.time(),
                "expires_at": time.time() + ttl if ttl > 0 else None,
            }

    def delete(self, key: str):
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self):
        """Clear all cache."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            now = time.time()
            expired = sum(
                1 for e in self._cache.values() if e["expires_at"] and e["expires_at"] < now
            )
            return {
                "total_keys": len(self._cache),
                "expired_keys": expired,
                "valid_keys": len(self._cache) - expired,
            }


# Initialize cache
cache = Cache()


def cached(ttl: int = 300):
    """Decorator to cache function results."""

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            key_hash = hashlib.md5(key.encode()).hexdigest()

            # Try to get from cache
            cached_value = cache.get(key_hash)
            if cached_value is not None:
                return cached_value

            # Call function and cache result
            result = await func(*args, **kwargs)
            cache.set(key_hash, result, ttl)
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            key_hash = hashlib.md5(key.encode()).hexdigest()

            # Try to get from cache
            cached_value = cache.get(key_hash)
            if cached_value is not None:
                return cached_value

            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(key_hash, result, ttl)
            return result

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# FastAPI app
app = FastAPI(
    title="SloughGPT API",
    description="Enterprise AI Framework API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all requests."""
    # Skip rate limiting for health and docs endpoints
    if request.url.path in ["/health", "/docs", "/openapi.json", "/"]:
        return await call_next(request)

    client_id = request.client.host if request.client else "unknown"

    if not rate_limiter.is_allowed(client_id):
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Maximum {rate_limiter.max_requests} per {rate_limiter.window_seconds} seconds.",
                "retry_after": rate_limiter.window_seconds,
            },
        )

    response = await call_next(request)

    # Add rate limit headers
    remaining = rate_limiter.get_remaining(client_id)
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.max_requests)
    response.headers["X-RateLimit-Remaining"] = str(remaining)

    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")

    return response


# ============================================================================
# Pydantic Models
# ============================================================================


class ChatMessageRequest(BaseModel):
    message: str
    model: str = "gpt-3.5-turbo"
    conversation_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000


class ChatMessageResponse(BaseModel):
    conversation_id: str
    message: Dict[str, str]
    model: str
    usage: Optional[Dict[str, int]] = None


class ConversationCreate(BaseModel):
    name: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str
    message_count: int


class ModelConfig(BaseModel):
    id: str
    name: str
    provider: str
    status: str = "available"
    description: Optional[str] = None
    context_length: int = 4096
    pricing: Optional[Dict[str, float]] = None


class DatasetCreate(BaseModel):
    name: str
    content: str
    description: Optional[str] = None


class DatasetInfo(BaseModel):
    name: str
    path: str
    size: int
    created_at: str
    updated_at: str
    has_train: bool
    has_val: bool
    has_meta: bool
    description: Optional[str] = None


class TrainingConfig(BaseModel):
    dataset_name: str
    model_id: str
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-4
    vocab_size: int = 500
    n_embed: int = 128
    n_layer: int = 3
    n_head: int = 4


class TrainingStatus(BaseModel):
    id: str
    status: str
    dataset_name: str
    model_id: str
    progress: float
    current_epoch: int
    total_epochs: int
    loss: float
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class SystemMetrics(BaseModel):
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_sent_mb: float
    network_recv_mb: float
    timestamp: str


class HealthStatus(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    timestamp: str
    services: Dict[str, str]


class GenerateRequest(BaseModel):
    prompt: str
    model: str = "nanogpt"
    max_length: int = 100
    temperature: float = 0.8
    top_k: Optional[int] = None


class GenerateResponse(BaseModel):
    text: str
    model: str
    tokens_generated: int
    processing_time_ms: float


# ============================================================================
# In-Memory Storage
# ============================================================================


class Storage:
    """In-memory storage for conversations, models, datasets, etc."""

    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = defaultdict(list)
        self.conversation_metadata: Dict[str, Dict] = {}
        self.datasets: Dict[str, Dict] = {}
        self.training_jobs: Dict[str, Dict] = {}
        self.start_time = time.time()
        self._lock = threading.Lock()

        # Initialize with default conversations
        self._init_default_data()

    def _init_default_data(self):
        """Initialize default data."""
        # Default conversation
        conv_id = "conv_default"
        self.conversation_metadata[conv_id] = {
            "id": conv_id,
            "name": "General",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        # Initialize datasets directory
        self.datasets_dir = Path("datasets")
        self.datasets_dir.mkdir(exist_ok=True)

        # Load existing datasets
        self._load_datasets()

    def _load_datasets(self):
        """Load datasets from disk."""
        if not self.datasets_dir.exists():
            return

        for item in self.datasets_dir.iterdir():
            if item.is_dir():
                meta_file = item / "meta.json"
                train_file = item / "train.bin"
                val_file = item / "val.bin"

                description = ""
                if meta_file.exists():
                    try:
                        meta = json.loads(meta_file.read_text())
                        description = meta.get("description", "")
                    except:
                        pass

                self.datasets[item.name] = {
                    "name": item.name,
                    "path": str(item),
                    "size": sum(f.stat().st_size for f in item.rglob("*") if f.is_file()),
                    "created_at": datetime.fromtimestamp(item.stat().st_ctime).isoformat(),
                    "updated_at": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                    "has_train": train_file.exists(),
                    "has_val": val_file.exists(),
                    "has_meta": meta_file.exists(),
                    "description": description,
                }

    def create_conversation(self, name: Optional[str] = None) -> Dict:
        """Create a new conversation."""
        with self._lock:
            conv_id = f"conv_{uuid.uuid4().hex[:8]}"
            self.conversations[conv_id] = []
            self.conversation_metadata[conv_id] = {
                "id": conv_id,
                "name": name or f"Conversation {len(self.conversation_metadata) + 1}",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            return self.conversation_metadata[conv_id]

    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation by ID."""
        if conversation_id not in self.conversations:
            return None
        return {
            "id": conversation_id,
            "messages": self.conversations[conversation_id],
            "metadata": self.conversation_metadata.get(conversation_id, {}),
        }

    def list_conversations(self) -> List[Dict]:
        """List all conversations."""
        result = []
        for conv_id, metadata in self.conversation_metadata.items():
            result.append({**metadata, "message_count": len(self.conversations[conv_id])})
        return sorted(result, key=lambda x: x["updated_at"], reverse=True)

    def add_message(self, conversation_id: str, role: str, content: str) -> Dict:
        """Add a message to a conversation."""
        with self._lock:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
                self.conversation_metadata[conversation_id] = {
                    "id": conversation_id,
                    "name": "New Conversation",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }

            message = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
            self.conversations[conversation_id].append(message)
            self.conversation_metadata[conversation_id]["updated_at"] = datetime.now().isoformat()
            return message

    def create_dataset(self, name: str, content: str, description: Optional[str] = None) -> Dict:
        """Create a new dataset."""
        with self._lock:
            dataset_path = self.datasets_dir / name
            dataset_path.mkdir(parents=True, exist_ok=True)

            input_file = dataset_path / "input.txt"
            input_file.write_text(content)

            meta = {"description": description or "", "created_at": datetime.now().isoformat()}
            meta_file = dataset_path / "meta.json"
            meta_file.write_text(json.dumps(meta))

            dataset_info = {
                "name": name,
                "path": str(dataset_path),
                "size": len(content),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "has_train": False,
                "has_val": False,
                "has_meta": True,
                "description": description or "",
            }

            self.datasets[name] = dataset_info
            return dataset_info

    def list_datasets(self) -> List[Dict]:
        """List all datasets."""
        return list(self.datasets.values())

    def get_dataset(self, name: str) -> Optional[Dict]:
        """Get dataset by name."""
        return self.datasets.get(name)

    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset."""
        with self._lock:
            if name not in self.datasets:
                return False

            dataset_path = self.datasets_dir / name
            if dataset_path.exists():
                shutil.rmtree(dataset_path)

            del self.datasets[name]
            return True

    def create_training_job(self, config: TrainingConfig) -> Dict:
        """Create a new training job."""
        job_id = f"train_{uuid.uuid4().hex[:8]}"

        job = {
            "id": job_id,
            "status": "pending",
            "dataset_name": config.dataset_name,
            "model_id": config.model_id,
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": config.epochs,
            "loss": 0.0,
            "config": config.model_dump(),
            "started_at": None,
            "completed_at": None,
            "error": None,
        }

        self.training_jobs[job_id] = job

        # Simulate training in background
        thread = threading.Thread(target=self._simulate_training, args=(job_id,))
        thread.daemon = True
        thread.start()

        return job

    def _simulate_training(self, job_id: str):
        """Simulate training process."""
        job = self.training_jobs[job_id]
        job["status"] = "running"
        job["started_at"] = datetime.now().isoformat()

        total_epochs = job["total_epochs"]

        for epoch in range(1, total_epochs + 1):
            job["current_epoch"] = epoch
            job["progress"] = (epoch / total_epochs) * 100
            job["loss"] = max(0.1, 3.0 - (epoch * 0.5) + (hash(str(epoch)) % 100) / 100)
            time.sleep(2)  # Simulate training time

        job["status"] = "completed"
        job["progress"] = 100.0
        job["completed_at"] = datetime.now().isoformat()

    def get_training_job(self, job_id: str) -> Optional[Dict]:
        """Get training job by ID."""
        return self.training_jobs.get(job_id)

    def list_training_jobs(self) -> List[Dict]:
        """List all training jobs."""
        return list(self.training_jobs.values())

    def get_system_metrics(self) -> Dict:
        """Get current system metrics."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        network = psutil.net_io_counters()

        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": memory.percent,
            "memory_used_mb": memory.used / (1024 * 1024),
            "memory_total_mb": memory.total / (1024 * 1024),
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / (1024 * 1024 * 1024),
            "disk_total_gb": disk.total / (1024 * 1024 * 1024),
            "network_sent_mb": network.bytes_sent / (1024 * 1024),
            "network_recv_mb": network.bytes_recv / (1024 * 1024),
            "timestamp": datetime.now().isoformat(),
        }

    def get_health(self) -> Dict:
        """Get system health status."""
        return {
            "status": "healthy",
            "version": "2.0.0",
            "uptime_seconds": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat(),
            "services": {
                "api": "healthy",
                "database": "healthy",
                "cache": "healthy",
                "model_service": "healthy",
            },
        }


# Initialize storage
storage = Storage()


# ============================================================================
# WebSocket Connection Manager
# ============================================================================


class ConnectionManager:
    """WebSocket connection manager."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "SloughGPT API", "version": "2.0.0", "docs": "/docs"}


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    return storage.get_health()


@app.get("/metrics", response_model=SystemMetrics)
async def get_metrics():
    """Get system metrics."""
    return storage.get_system_metrics()


# -------------------------------------------------------------------------
# Conversation Endpoints
# -------------------------------------------------------------------------


@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(data: ConversationCreate = ConversationCreate()):
    """Create a new conversation."""
    name = data.name if data.name else None
    conv = storage.create_conversation(name)
    return conv


@app.get("/conversations")
async def list_conversations():
    """List all conversations."""
    return {"conversations": storage.list_conversations()}


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation by ID."""
    conv = storage.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if conversation_id not in storage.conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    del storage.conversations[conversation_id]
    if conversation_id in storage.conversation_metadata:
        del storage.conversation_metadata[conversation_id]
    return {"status": "deleted", "conversation_id": conversation_id}


# -------------------------------------------------------------------------
# Chat Endpoints
# -------------------------------------------------------------------------


@app.post("/chat", response_model=ChatMessageResponse)
async def send_chat_message(request: ChatMessageRequest):
    """Send a chat message and get a response."""
    conversation_id = request.conversation_id or "conv_default"

    # Add user message
    storage.add_message(conversation_id, "user", request.message)

    # Generate response (echo for now, can be enhanced with actual AI)
    response_content = (
        f"Echo: {request.message}\n\n(Model: {request.model}, Temperature: {request.temperature})"
    )

    # Add assistant message
    assistant_message = storage.add_message(conversation_id, "assistant", response_content)

    return {
        "conversation_id": conversation_id,
        "message": assistant_message,
        "model": request.model,
        "usage": {
            "prompt_tokens": len(request.message.split()),
            "completion_tokens": len(response_content.split()),
            "total_tokens": len(request.message.split()) + len(response_content.split()),
        },
    }


@app.post("/chat/stream")
async def send_chat_message_stream(request: ChatMessageRequest):
    """Send a chat message and get a streaming response."""
    from fastapi.responses import StreamingResponse
    import asyncio
    import json

    conversation_id = request.conversation_id or "conv_default"

    # Add user message
    storage.add_message(conversation_id, "user", request.message)

    # Generate response (simulated streaming)
    response_content = (
        f"Echo: {request.message}\n\n(Model: {request.model}, Temperature: {request.temperature})"
    )

    async def generate():
        accumulated = ""
        for char in response_content:
            accumulated += char
            chunk = {"type": "chunk", "content": char, "accumulated": accumulated}
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.02)  # Simulate streaming delay

        # Final message
        storage.add_message(conversation_id, "assistant", accumulated)
        final_chunk = {
            "type": "done",
            "content": accumulated,
            "usage": {
                "prompt_tokens": len(request.message.split()),
                "completion_tokens": len(accumulated.split()),
                "total_tokens": len(request.message.split()) + len(accumulated.split()),
            },
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# -------------------------------------------------------------------------
# Model Endpoints
# -------------------------------------------------------------------------


@app.get("/models")
async def list_models():
    """List available models."""
    models = [
        ModelConfig(
            id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            provider="OpenAI",
            description="Fast, cost-effective language model",
            context_length=4096,
            pricing={"prompt": 0.0015, "completion": 0.002},
        ),
        ModelConfig(
            id="gpt-4",
            name="GPT-4",
            provider="OpenAI",
            description="Most capable GPT model",
            context_length=8192,
            pricing={"prompt": 0.03, "completion": 0.06},
        ),
        ModelConfig(
            id="claude-3",
            name="Claude 3",
            provider="Anthropic",
            description="Helpful and harmless AI assistant",
            context_length=200000,
            pricing={"prompt": 0.015, "completion": 0.075},
        ),
        ModelConfig(
            id="nanogpt",
            name="NanoGPT",
            provider="Local",
            description="Custom trained GPT model",
            context_length=512,
            pricing={"prompt": 0.0, "completion": 0.0},
        ),
        ModelConfig(
            id="llama-2-7b",
            name="Llama 2 7B",
            provider="Meta",
            description="Open source language model",
            context_length=4096,
            pricing={"prompt": 0.0, "completion": 0.0},
        ),
        ModelConfig(
            id="mixtral-8x7b",
            name="Mixtral 8x7B",
            provider="Mistral",
            description="Mixture of experts model",
            context_length=32000,
            pricing={"prompt": 0.0, "completion": 0.0},
        ),
    ]
    return {"models": [m.model_dump() for m in models]}


@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model by ID."""
    models = await list_models()
    for model in models["models"]:
        if model["id"] == model_id:
            return model
    raise HTTPException(status_code=404, detail="Model not found")


# -------------------------------------------------------------------------
# Dataset Endpoints
# -------------------------------------------------------------------------


@app.get("/datasets")
async def list_datasets():
    """List all datasets."""
    return {"datasets": storage.list_datasets()}


@app.post("/datasets", response_model=DatasetInfo)
async def create_dataset(request: DatasetCreate):
    """Create a new dataset."""
    if request.name in storage.datasets:
        raise HTTPException(status_code=400, detail="Dataset already exists")

    return storage.create_dataset(request.name, request.content, request.description)


@app.get("/datasets/{name}", response_model=DatasetInfo)
async def get_dataset(name: str):
    """Get dataset by name."""
    dataset = storage.get_dataset(name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@app.delete("/datasets/{name}")
async def delete_dataset(name: str):
    """Delete a dataset."""
    if not storage.delete_dataset(name):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"status": "deleted", "name": name}


# -------------------------------------------------------------------------
# Training Endpoints
# -------------------------------------------------------------------------


@app.get("/training")
async def list_training_jobs():
    """List all training jobs."""
    return {"jobs": storage.list_training_jobs()}


@app.post("/training", response_model=TrainingStatus)
async def create_training_job(config: TrainingConfig):
    """Create a new training job."""
    # Validate dataset exists
    if config.dataset_name not in storage.datasets:
        raise HTTPException(status_code=400, detail="Dataset not found")

    return storage.create_training_job(config)


@app.get("/training/{job_id}", response_model=TrainingStatus)
async def get_training_job(job_id: str):
    """Get training job by ID."""
    job = storage.get_training_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@app.delete("/training/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancel a training job."""
    job = storage.get_training_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    job["status"] = "cancelled"
    return {"status": "cancelled", "job_id": job_id}


# -------------------------------------------------------------------------
# Generation Endpoints
# -------------------------------------------------------------------------


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text from a prompt."""
    start_time = time.time()

    # Simulate text generation
    generated = f"Generated: {request.prompt[:50]}... (length: {request.max_length})"

    processing_time = (time.time() - start_time) * 1000

    return {
        "text": generated,
        "model": request.model,
        "tokens_generated": request.max_length,
        "processing_time_ms": processing_time,
    }


# -------------------------------------------------------------------------
# WebSocket Endpoint
# -------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            msg_type = message.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
            elif msg_type == "subscribe":
                # Subscribe to specific updates
                await websocket.send_json(
                    {"type": "subscribed", "channels": message.get("channels", [])}
                )
            else:
                await websocket.send_json({"type": "echo", "data": message})

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# -------------------------------------------------------------------------
# System Info Endpoints
# -------------------------------------------------------------------------


@app.get("/info")
async def get_system_info():
    """Get system information."""
    return {
        "python_version": "3.10+",
        "platform": os.name,
        "cpu_count": psutil.cpu_count(),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        "disk_total_gb": psutil.disk_usage("/").total / (1024**3),
        "conversations_count": len(storage.conversations),
        "datasets_count": len(storage.datasets),
        "training_jobs_count": len(storage.training_jobs),
    }


# -------------------------------------------------------------------------
# Utility Endpoints
# -------------------------------------------------------------------------


@app.get("/stats")
async def get_stats():
    """Get comprehensive statistics."""
    jobs = storage.list_training_jobs()
    completed_jobs = [j for j in jobs if j.get("status") == "completed"]
    running_jobs = [j for j in jobs if j.get("status") == "running"]

    return {
        "conversations": {
            "total": len(storage.conversations),
            "total_messages": sum(len(msgs) for msgs in storage.conversations.values()),
        },
        "datasets": {
            "total": len(storage.datasets),
            "total_size_bytes": sum(ds.get("size", 0) for ds in storage.datasets.values()),
        },
        "training": {
            "total": len(jobs),
            "completed": len(completed_jobs),
            "running": len(running_jobs),
            "pending": len(jobs) - len(completed_jobs) - len(running_jobs),
        },
        "cache": cache.get_stats(),
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
        },
    }


@app.post("/reset")
async def reset_data():
    """Reset all data (use with caution)."""
    storage.conversations.clear()
    storage.conversation_metadata.clear()
    storage.training_jobs.clear()
    # Don't reset datasets as they may be on disk
    return {"status": "reset", "message": "All conversations and training jobs cleared"}


@app.get("/logs")
async def get_logs(limit: int = 50):
    """Get recent log entries."""
    # This would be enhanced with proper logging
    return {
        "logs": [
            {
                "level": "INFO",
                "message": "API server started",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "level": "INFO",
                "message": "Serving on port 8000",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "level": "DEBUG",
                "message": f"Loaded {len(storage.datasets)} datasets",
                "timestamp": datetime.now().isoformat(),
            },
        ][:limit]
    }


# -------------------------------------------------------------------------
# Batch Operations
# -------------------------------------------------------------------------


@app.post("/batch/datasets/delete")
async def batch_delete_datasets(names: List[str]):
    """Delete multiple datasets at once."""
    results = []
    for name in names:
        success = storage.delete_dataset(name)
        results.append({"name": name, "deleted": success})
    return {"results": results}


@app.post("/batch/training/cancel")
async def batch_cancel_training(job_ids: List[str]):
    """Cancel multiple training jobs."""
    results = []
    for job_id in job_ids:
        job = storage.get_training_job(job_id)
        if job:
            job["status"] = "cancelled"
            results.append({"id": job_id, "cancelled": True})
        else:
            results.append({"id": job_id, "cancelled": False, "error": "Job not found"})
    return {"results": results}


# -------------------------------------------------------------------------
# Web UI
# -------------------------------------------------------------------------


@app.get("/web")
async def web_ui():
    """Serve the web UI."""
    from fastapi.responses import FileResponse

    return FileResponse("domains/ui/web/index.html")


@app.get("/dashboard")
async def dashboard():
    """Serve the dashboard."""
    from fastapi.responses import FileResponse

    return FileResponse("domains/ui/web/dashboard.html")


# -------------------------------------------------------------------------
# Personality System
# -------------------------------------------------------------------------

from domains.ai_personality import PersonalityType, list_personalities, get_personality_manager

_personality_manager = get_personality_manager()


@app.get("/personalities")
async def get_personalities():
    """List all available personalities."""
    return {"personalities": list_personalities()}


@app.post("/personalities/{personality_type}")
async def set_personality(personality_type: str):
    """Set the current personality."""
    try:
        ptype = PersonalityType(personality_type)
        _personality_manager.set_personality(ptype)
        return {"status": "ok", "personality": personality_type}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid personality type")


@app.get("/personalities/current")
async def get_current_personality():
    """Get current personality."""
    p = _personality_manager.get_personality()
    return {"name": p.name, "description": p.description, "traits": p.traits}


# -------------------------------------------------------------------------
# Model Inference
# -------------------------------------------------------------------------

_model_cache = {}


@app.post("/infer")
async def infer(request: GenerateRequest):
    """Run inference with local or HuggingFace model."""
    import torch
    from domains.models import SloughGPTModel

    model_name = request.model

    # Load model if not cached
    if model_name not in _model_cache:
        try:
            # Check if HuggingFace model
            if model_name.startswith("hf_"):
                hf_name = model_name[3:]  # Remove 'hf_' prefix
                from transformers import GPT2LMHeadModel, GPT2Tokenizer

                tokenizer = GPT2Tokenizer.from_pretrained(hf_name)
                model = GPT2LMHeadModel.from_pretrained(hf_name)
                model.eval()

                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                _model_cache[model_name] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "type": "huggingface",
                }
            else:
                # Local model
                checkpoint = torch.load("models/sloughgpt.pt", weights_only=False)
                chars = checkpoint["chars"]
                stoi = checkpoint["stoi"]
                itos = checkpoint["itos"]

                model = SloughGPTModel(
                    vocab_size=len(chars),
                    n_embed=checkpoint.get("config", {}).get("n_embed", 128),
                    n_layer=checkpoint.get("config", {}).get("n_layer", 4),
                    n_head=checkpoint.get("config", {}).get("n_head", 4),
                    block_size=checkpoint.get("config", {}).get("block_size", 64),
                )
                model.load_state_dict(checkpoint["model"])
                model.eval()

                _model_cache[model_name] = {
                    "model": model,
                    "stoi": stoi,
                    "itos": itos,
                    "chars": chars,
                    "type": "local",
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # Generate
    cached = _model_cache[model_name]
    model = cached["model"]
    model_type = cached.get("type", "local")

    start_time = time.time()

    if model_type == "huggingface":
        # HuggingFace model
        tokenizer = cached["tokenizer"]
        inputs = tokenizer(request.prompt, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(output[0])
    else:
        # Local model
        stoi = cached["stoi"]
        itos = cached["itos"]

        idx = torch.tensor([[stoi.get(c, 0) for c in request.prompt[:1]]])

        with torch.no_grad():
            output = model.generate(
                idx,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k if request.top_k else None,
            )

        generated_text = "".join([itos.get(int(i), "?") for i in output[0]])

    processing_time = (time.time() - start_time) * 1000

    return {
        "text": generated_text,
        "model": model_name,
        "tokens_generated": request.max_length,
        "processing_time_ms": processing_time,
    }


@app.post("/infer/stream")
async def infer_stream(request: GenerateRequest):
    """Stream inference with local model."""
    import torch
    from domains.models import SloughGPTModel
    from fastapi.responses import StreamingResponse

    model_name = request.model

    # Load model if not cached
    if model_name not in _model_cache:
        try:
            checkpoint = torch.load("models/sloughgpt.pt", weights_only=False)
            chars = checkpoint["chars"]
            stoi = checkpoint["stoi"]
            itos = checkpoint["itos"]

            model = SloughGPTModel(
                vocab_size=len(chars),
                n_embed=checkpoint.get("config", {}).get("n_embed", 128),
                n_layer=checkpoint.get("config", {}).get("n_layer", 4),
                n_head=checkpoint.get("config", {}).get("n_head", 4),
                block_size=checkpoint.get("config", {}).get("block_size", 64),
            )
            model.load_state_dict(checkpoint["model"])
            model.eval()

            _model_cache[model_name] = {"model": model, "stoi": stoi, "itos": itos, "chars": chars}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # Generate streaming
    cached = _model_cache[model_name]
    model = cached["model"]
    stoi = cached["stoi"]
    itos = cached["itos"]

    async def generate():
        # Encode prompt
        idx = torch.tensor([[stoi.get(c, 0) for c in request.prompt[:1]]])
        accumulated = ""

        # Stream token by token
        for _ in range(request.max_length):
            with torch.no_grad():
                output = model.generate(
                    idx,
                    max_new_tokens=1,
                    temperature=request.temperature,
                    top_k=request.top_k if request.top_k else None,
                )

            # Get new token
            new_token = output[0, -1].item()
            new_char = itos.get(new_token, "?")
            accumulated += new_char

            # Append to input for next token
            idx = torch.cat([idx, output[:, -1:]], dim=1)

            # Yield streaming response
            yield f"data: {json.dumps({'type': 'chunk', 'content': new_char, 'accumulated': accumulated})}\n\n"

            # Small delay for streaming effect
            await asyncio.sleep(0.02)

        yield f"data: {json.dumps({'type': 'done', 'content': accumulated})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# -------------------------------------------------------------------------
# Training Endpoints (legacy demo server)
# -------------------------------------------------------------------------
# This module defines a *separate* FastAPI app used for UI demos / experiments.
# It is **not** the same contract as ``apps/api/server`` (``main.py``), which
# mounts ``training.router``: JSON ``TrainingRequest``, corpus resolution,
# ``SloughGPTTrainer``, live ``on_progress`` job fields, ``log_interval`` /
# ``eval_interval``, etc. Prefer ``apps/api/server`` for product parity with
# the web Console and SDKs.
# -------------------------------------------------------------------------

_training_jobs = {}


class TrainingJob:
    """Training job container."""

    def __init__(self, job_id, config):
        self.id = job_id
        self.config = config
        self.status = "pending"
        self.progress = 0
        self.loss = None
        self.started_at = None
        self.completed_at = None
        self.error = None


@app.post("/training/start")
async def start_training(
    dataset: str = "shakespeare",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 0.01,
    use_lora: bool = True,
    lora_rank: int = 4,
    n_embed: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
):
    """Legacy demo training (query params, toy loop). Not ``TrainingRequest`` JSON; use ``apps/api/server`` for parity."""
    job_id = f"train_{uuid.uuid4().hex[:8]}"

    job = TrainingJob(
        job_id,
        {
            "dataset": dataset,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "use_lora": use_lora,
            "lora_rank": lora_rank,
            "n_embed": n_embed,
            "n_layer": n_layer,
            "n_head": n_head,
        },
    )

    _training_jobs[job_id] = job

    # Run training in background
    def train_background():
        job.status = "running"
        job.started_at = datetime.now().isoformat()

        try:
            # Simple training simulation
            import torch
            from domains.models import SloughGPTModel
            from domains.training.lora import apply_lora_to_model, LoRAConfig

            # Load data
            data_path = f"datasets/{dataset}/input.txt"
            with open(data_path, "r") as f:
                text = f.read()

            chars = sorted(set(text))
            stoi = {c: i for i, c in enumerate(chars)}
            data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

            # Create model
            model = SloughGPTModel(
                vocab_size=len(chars),
                n_embed=n_embed,
                n_layer=n_layer,
                n_head=n_head,
                block_size=64,
            )

            # Apply LoRA
            if use_lora:
                lora_config = LoRAConfig(
                    rank=lora_rank, alpha=16, target_modules=["c_attn", "c_proj", "c_fc"]
                )
                model = apply_lora_to_model(model, config=lora_config)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

            # Train
            steps = len(data) // 64 // batch_size
            for epoch in range(epochs):
                job.progress = (epoch / epochs) * 100

                for step in range(min(steps, 50)):  # Limit steps for demo
                    idx = torch.randint(0, len(data) - 65, (batch_size,))
                    x = torch.stack([data[i : i + 64] for i in idx])
                    y = torch.stack([data[i + 1 : i + 65] for i in idx])

                    optimizer.zero_grad()
                    logits, loss = model(x, y)
                    loss.backward()
                    optimizer.step()

                    job.loss = loss.item()

                job.progress = ((epoch + 1) / epochs) * 100

            # Save model
            torch.save(
                {
                    "model": model.state_dict(),
                    "chars": chars,
                    "stoi": stoi,
                    "itos": {int(k): v for k, v in enumerate(chars)},
                },
                f"models/sloughgpt_trained_{job_id}.pt",
            )

            job.status = "completed"
            job.completed_at = datetime.now().isoformat()

        except Exception as e:
            job.status = "failed"
            job.error = str(e)

    thread = threading.Thread(target=train_background)
    thread.start()

    return {"job_id": job_id, "status": "started", "config": job.config}


@app.get("/training/status/{job_id}")
async def get_training_status(job_id: str):
    """Legacy demo job status (``job_id`` key). Production API uses ``GET /training/jobs/{id}`` with ``id``."""
    job = _training_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "loss": job.loss,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error": job.error,
    }


@app.get("/training/jobs")
async def list_training_jobs():
    """Legacy demo list (wrapped ``{ \"jobs\": [...] }``). Production API returns a bare JSON array."""
    return {
        "jobs": [
            {"job_id": job.id, "status": job.status, "progress": job.progress, "config": job.config}
            for job in _training_jobs.values()
        ]
    }


# -------------------------------------------------------------------------
# LoRA Training
# -------------------------------------------------------------------------


@app.post("/training/lora")
async def train_with_lora(
    dataset: str = "shakespeare",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 0.001,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    n_embed: int = 128,
    n_layer: int = 4,
):
    """Train model with LoRA."""
    job_id = f"lora_{uuid.uuid4().hex[:8]}"

    job = TrainingJob(
        job_id,
        {
            "type": "lora",
            "dataset": dataset,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "n_embed": n_embed,
            "n_layer": n_layer,
        },
    )

    _training_jobs[job_id] = job

    def train_lora_background():
        job.status = "running"
        job.started_at = datetime.now().isoformat()

        try:
            import torch
            from domains.models import SloughGPTModel
            from domains.training.lora import apply_lora_to_model, LoRAConfig

            # Load data
            with open(f"datasets/{dataset}/input.txt", "r") as f:
                text = f.read()

            chars = sorted(set(text))
            stoi = {c: i for i, c in enumerate(chars)}
            data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

            # Create model
            model = SloughGPTModel(
                vocab_size=len(chars), n_embed=n_embed, n_layer=n_layer, n_head=4, block_size=64
            )

            # Apply LoRA
            lora_config = LoRAConfig(
                rank=lora_rank, alpha=lora_alpha, target_modules=["c_attn", "c_proj", "c_fc"]
            )
            model = apply_lora_to_model(model, config=lora_config)

            # Count LoRA params
            lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
            total_params = sum(p.numel() for p in model.parameters())

            job.config["lora_params"] = lora_params
            job.config["total_params"] = total_params

            optimizer = torch.optim.AdamW(
                [p for n, p in model.named_parameters() if "lora_" in n], lr=lr
            )

            # Train
            steps = len(data) // 64 // batch_size
            for epoch in range(epochs):
                job.progress = (epoch / epochs) * 100

                for step in range(min(steps, 50)):
                    idx = torch.randint(0, len(data) - 65, (batch_size,))
                    x = torch.stack([data[i : i + 64] for i in idx])
                    y = torch.stack([data[i + 1 : i + 65] for i in idx])

                    optimizer.zero_grad()
                    logits, loss = model(x, y)
                    loss.backward()
                    optimizer.step()

                    job.loss = loss.item()

                job.progress = ((epoch + 1) / epochs) * 100

            # Save
            torch.save(
                {
                    "model": model.state_dict(),
                    "chars": chars,
                    "stoi": stoi,
                    "itos": {int(k): v for k, v in enumerate(chars)},
                    "config": {
                        "n_embed": n_embed,
                        "n_layer": n_layer,
                        "n_head": 4,
                        "block_size": 64,
                        "lora_rank": lora_rank,
                        "lora_alpha": lora_alpha,
                    },
                    "lora_config": {
                        "rank": lora_rank,
                        "alpha": lora_alpha,
                    },
                },
                f"models/sloughgpt_lora_{job_id}.pt",
            )

            job.status = "completed"
            job.completed_at = datetime.now().isoformat()

        except Exception as e:
            job.status = "failed"
            job.error = str(e)

    thread = threading.Thread(target=train_lora_background)
    thread.start()

    return {"job_id": job_id, "status": "started", "config": job.config}


# -------------------------------------------------------------------------
# Model Comparison
# -------------------------------------------------------------------------


@app.post("/compare")
async def compare_models(request: GenerateRequest):
    """Compare outputs from different models."""
    prompts = [request.prompt]  # Use single prompt for now
    results = {}

    # Test each cached model
    for model_name in ["sloughgpt"]:
        if model_name in _model_cache:
            cached = _model_cache[model_name]
            model = cached["model"]
            stoi = cached.get("stoi", {})
            itos = cached.get("itos", {})

            if stoi and itos:
                model_outputs = []
                for prompt in prompts:
                    idx = torch.tensor([[stoi.get(prompt[:1], 0)]])
                    with torch.no_grad():
                        output = model.generate(idx, max_new_tokens=30)
                    text = "".join([itos.get(int(i), "?") for i in output[0]])
                    model_outputs.append(text)

                results[model_name] = model_outputs

    return {"prompts": prompts, "results": results}


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

_config = {
    "max_tokens": 100,
    "default_temperature": 0.8,
    "default_model": "sloughgpt",
    "enable_streaming": True,
    "enable_caching": True,
    "max_cache_size": 5,
    "training": {
        "default_epochs": 3,
        "default_batch_size": 32,
        "default_lr": 0.01,
    },
}


@app.get("/config")
async def get_config():
    """Get API configuration."""
    return _config


@app.post("/config")
async def update_config(key: str, value: str):
    """Update configuration."""
    # Parse value
    if value.lower() == "true":
        value = True
    elif value.lower() == "false":
        value = False
    elif value.isdigit():
        value = int(value)

    _config[key] = value
    return {"status": "updated", "key": key, "value": value}


# -------------------------------------------------------------------------
# Metrics Endpoint
# -------------------------------------------------------------------------


@app.get("/metrics")
async def get_metrics():
    """Get system and model metrics."""
    import psutil

    # CPU & Memory
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()

    # Disk
    disk = psutil.disk_usage("/")

    # Model cache info
    model_info = {"cached_models": list(_model_cache.keys()), "num_models": len(_model_cache)}

    # Training jobs
    training_info = {
        "total_jobs": len(_training_jobs),
        "running": sum(1 for j in _training_jobs.values() if j.status == "running"),
        "completed": sum(1 for j in _training_jobs.values() if j.status == "completed"),
        "failed": sum(1 for j in _training_jobs.values() if j.status == "failed"),
    }

    return {
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_mb": memory.used / 1024 / 1024,
            "memory_total_mb": memory.total / 1024 / 1024,
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / 1024 / 1024 / 1024,
            "disk_total_gb": disk.total / 1024 / 1024 / 1024,
        },
        "models": model_info,
        "training": training_info,
    }


# -------------------------------------------------------------------------
# Export Endpoints
# -------------------------------------------------------------------------


@app.get("/export/model/{model_name}")
async def export_model(model_name: str):
    """Export model to different formats."""
    import torch
    from pathlib import Path

    model_path = Path(f"models/{model_name}.pt")

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    # For now, just return the model info
    checkpoint = torch.load(model_path, weights_only=False)

    return {
        "model": model_name,
        "format": "pt",
        "size_bytes": model_path.stat().st_size,
        "config": checkpoint.get("config", {}),
        "vocab_size": len(checkpoint.get("chars", [])),
    }


# ============================================================================
# WebSocket
# ============================================================================

from fastapi import WebSocket


class ConnectionManager:
    """WebSocket connection manager."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            msg_type = message.get("type")

            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

            elif msg_type == "generate":
                # Quick generation via WebSocket
                prompt = message.get("prompt", "")
                max_tokens = message.get("max_tokens", 50)

                # Use cached model if available
                if "sloughgpt" in _model_cache:
                    cached = _model_cache["sloughgpt"]
                    model = cached["model"]
                    stoi = cached.get("stoi", {})
                    itos = cached.get("itos", {})

                    if stoi and itos:
                        idx = torch.tensor([[stoi.get(prompt[:1], 0)]])
                        with torch.no_grad():
                            output = model.generate(idx, max_new_tokens=max_tokens)
                        text = "".join([itos.get(int(i), "?") for i in output[0]])

                        await websocket.send_text(json.dumps({"type": "generated", "text": text}))
                else:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": "Model not loaded"})
                    )

            elif msg_type == "subscribe":
                # Subscribe to updates
                await websocket.send_text(
                    json.dumps({"type": "subscribed", "channels": message.get("channels", [])})
                )

            else:
                await websocket.send_text(json.dumps({"type": "echo", "message": data}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    """WebSocket for training updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Send training updates periodically
            await asyncio.sleep(2)

            # Get training status
            running_jobs = [j for j in _training_jobs.values() if j.status == "running"]

            if running_jobs:
                for job in running_jobs:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "training_update",
                                "job_id": job.id,
                                "progress": job.progress,
                                "loss": job.loss,
                                "status": job.status,
                            }
                        )
                    )
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============================================================================
# Authentication
# ============================================================================

import secrets

_users = {"admin": {"password_hash": hashlib.sha256(b"admin").hexdigest(), "role": "admin"}}

_tokens = {}


def create_token(username: str) -> str:
    """Create JWT-like token."""
    token = secrets.token_urlsafe(32)
    _tokens[token] = {"username": username}
    return token


@app.post("/auth/token")
async def get_token(username: str = "admin", password: str = "admin"):
    """Get authentication token."""
    user = _users.get(username)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    password_hash = hashlib.sha256(password.encode()).hexdigest()

    if user["password_hash"] != password_hash:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(username)

    return {"access_token": token, "token_type": "bearer", "expires_in": 3600}


@app.post("/auth/register")
async def register(username: str, password: str):
    """Register new user."""
    if username in _users:
        raise HTTPException(status_code=400, detail="User already exists")

    _users[username] = {
        "password_hash": hashlib.sha256(password.encode()).hexdigest(),
        "role": "user",
    }

    return {"status": "ok", "username": username}


# ============================================================================
# Dataset Management
# ============================================================================


@app.get("/datasets")
async def list_datasets():
    """List available datasets."""
    from pathlib import Path

    datasets = []
    datasets_dir = Path("datasets")

    if datasets_dir.exists():
        for d in datasets_dir.iterdir():
            if d.is_dir():
                input_file = d / "input.txt"
                size = input_file.stat().st_size if input_file.exists() else 0
                datasets.append({"name": d.name, "size_bytes": size, "path": str(d)})

    return {"datasets": datasets}


@app.post("/datasets/{name}")
async def upload_dataset(name: str, content: str):
    """Create or update a dataset."""
    from pathlib import Path

    dataset_dir = Path(f"datasets/{name}")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    input_file = dataset_dir / "input.txt"
    input_file.write_text(content)

    return {"name": name, "size_bytes": len(content), "path": str(dataset_dir)}


@app.delete("/datasets/{name}")
async def delete_dataset(name: str):
    """Delete a dataset."""
    import shutil
    from pathlib import Path

    dataset_dir = Path(f"datasets/{name}")

    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    shutil.rmtree(dataset_dir)

    return {"status": "deleted", "name": name}


# ============================================================================
# Model Optimization
# ============================================================================


@app.post("/optimize/quantize")
async def quantize_model(model_name: str = "sloughgpt", precision: str = "int8"):
    """Quantize a model."""
    from domains.training.efficient_inference import Quantizer

    # Load model
    checkpoint = torch.load(f"models/{model_name}.pt", weights_only=False)

    from domains.models import SloughGPTModel

    model = SloughGPTModel(
        vocab_size=len(checkpoint.get("chars", [])),
        n_embed=checkpoint.get("config", {}).get("n_embed", 128),
        n_layer=checkpoint.get("config", {}).get("n_layer", 4),
        n_head=checkpoint.get("config", {}).get("n_head", 4),
        block_size=checkpoint.get("config", {}).get("block_size", 64),
    )
    model.load_state_dict(checkpoint["model"])

    # Quantize
    dtype = torch.qint8 if precision == "int8" else torch.float16
    quantized_model = Quantizer.dynamic_quantize(model, dtype=dtype)

    # Save quantized model
    quantized_name = f"{model_name}_{precision}"
    torch.save(
        {
            "model": quantized_model.state_dict()
            if hasattr(quantized_model, "state_dict")
            else checkpoint["model"],
            "chars": checkpoint.get("chars", []),
            "stoi": checkpoint.get("stoi", {}),
            "itos": checkpoint.get("itos", {}),
            "config": checkpoint.get("config", {}),
            "quantized": True,
            "precision": precision,
        },
        f"models/{quantized_name}.pt",
    )

    return {
        "status": "quantized",
        "original_model": model_name,
        "quantized_model": quantized_name,
        "precision": precision,
    }


# ============================================================================
# Batch Inference
# ============================================================================


@app.post("/infer/batch")
async def batch_inference(
    prompts: List[str], model: str = "sloughgpt", max_length: int = 50, temperature: float = 0.8
):
    """Run batch inference on multiple prompts."""
    results = []

    for prompt in prompts:
        # Simple implementation - in real use would use batch processing
        r = await infer(
            GenerateRequest(
                prompt=prompt, max_length=max_length, temperature=temperature, model=model
            )
        )
        results.append(r)

    return {"results": results}


# ============================================================================
# System Info
# ============================================================================


@app.get("/info")
async def get_system_info():
    """Get detailed system information."""
    import platform
    import psutil

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else None,
        "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
        "disk_partitions": [
            {"device": p.device, "mountpoint": p.mountpoint, "fstype": p.fstype}
            for p in psutil.disk_partitions()
        ],
        "network_interfaces": list(psutil.net_if_addrs().keys()),
        "boot_time": psutil.boot_time(),
    }


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health")
async def health_check():
    """Comprehensive health check."""
    import psutil

    checks = {
        "api": "healthy",
        "storage": "healthy",
        "models": "healthy" if _model_cache else "no_models_loaded",
        "training": "healthy",
    }

    # Check for issues
    issues = []

    if psutil.virtual_memory().percent > 90:
        issues.append("High memory usage")

    if psutil.disk_usage("/").percent > 90:
        issues.append("Low disk space")

    return {
        "status": "healthy" if not issues else "degraded",
        "version": "2.0.0",
        "uptime_seconds": time.time(),
        "timestamp": datetime.now().isoformat(),
        "services": checks,
        "issues": issues,
    }


# ============================================================================
# Cache Management
# ============================================================================


@app.get("/cache")
async def get_cache_info():
    """Get cache information."""
    return {
        "models": list(_model_cache.keys()),
        "count": len(_model_cache),
        "training_jobs": len(_training_jobs),
    }


@app.post("/cache/clear")
async def clear_cache(model_name: str = None):
    """Clear model cache."""
    if model_name:
        if model_name in _model_cache:
            del _model_cache[model_name]
            return {"status": "cleared", "model": model_name}
        else:
            raise HTTPException(status_code=404, detail="Model not in cache")
    else:
        _model_cache.clear()
        return {"status": "cleared", "all": True}


# ============================================================================
# Logging
# ============================================================================

_request_logs = []


@app.get("/logs")
async def get_logs(limit: int = 100):
    """Get recent request logs."""
    return {"logs": _request_logs[-limit:], "count": len(_request_logs)}


@app.post("/logs/clear")
async def clear_logs():
    """Clear request logs."""
    _request_logs.clear()
    return {"status": "cleared"}


# ============================================================================
# Plugin System
# ============================================================================

_plugins = {}


class Plugin:
    """Base plugin class."""

    def __init__(self, name: str):
        self.name = name
        self.enabled = True

    def process(self, data):
        return data


def register_plugin(name: str, plugin: Plugin):
    """Register a plugin."""
    _plugins[name] = plugin


@app.get("/plugins")
async def list_plugins():
    """List all plugins."""
    return {"plugins": [{"name": name, "enabled": p.enabled} for name, p in _plugins.items()]}


@app.post("/plugins/{name}/enable")
async def enable_plugin(name: str):
    """Enable a plugin."""
    if name not in _plugins:
        raise HTTPException(status_code=404, detail="Plugin not found")
    _plugins[name].enabled = True
    return {"status": "enabled", "plugin": name}


@app.post("/plugins/{name}/disable")
async def disable_plugin(name: str):
    """Disable a plugin."""
    if name not in _plugins:
        raise HTTPException(status_code=404, detail="Plugin not found")
    _plugins[name].enabled = False
    return {"status": "disabled", "plugin": name}


# Register some default plugins
class UppercasePlugin(Plugin):
    def process(self, data):
        return data.upper()


class LowercasePlugin(Plugin):
    def process(self, data):
        return data.lower()


register_plugin("uppercase", UppercasePlugin("uppercase"))
register_plugin("lowercase", LowercasePlugin("lowercase"))


@app.post("/plugins/process")
async def process_with_plugin(data: str, plugin: str):
    """Process data through a plugin."""
    if plugin not in _plugins:
        raise HTTPException(status_code=404, detail="Plugin not found")

    if not _plugins[plugin].enabled:
        raise HTTPException(status_code=400, detail="Plugin disabled")

    result = _plugins[plugin].process(data)
    return {"original": data, "processed": result, "plugin": plugin}


# ============================================================================
# Validation & Benchmarking
# ============================================================================


@app.post("/validate/model")
async def validate_model(model_name: str = "sloughgpt"):
    """Validate model functionality."""
    import torch
    from domains.models import SloughGPTModel

    results = {"model": model_name, "checks": {}}

    try:
        # Load model
        checkpoint = torch.load(f"models/{model_name}.pt", weights_only=False)
        chars = checkpoint.get("chars", [])

        model = SloughGPTModel(
            vocab_size=len(chars),
            n_embed=checkpoint.get("config", {}).get("n_embed", 128),
            n_layer=checkpoint.get("config", {}).get("n_layer", 4),
            n_head=checkpoint.get("config", {}).get("n_head", 4),
            block_size=checkpoint.get("config", {}).get("block_size", 64),
        )
        model.load_state_dict(checkpoint["model"])

        results["checks"]["load"] = "passed"

        # Test forward pass
        x = torch.randint(0, len(chars), (2, 10))
        with torch.no_grad():
            logits, loss = model(x)

        results["checks"]["forward"] = "passed"
        results["checks"]["output_shape"] = list(logits.shape)

        # Test generation
        idx = torch.tensor([[0]])
        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=5)

        results["checks"]["generate"] = "passed"
        results["checks"]["generation_length"] = output.shape[1]

        results["status"] = "valid"

    except Exception as e:
        results["status"] = "invalid"
        results["error"] = str(e)

    return results


@app.post("/benchmark")
async def benchmark_inference(
    model: str = "sloughgpt", num_iterations: int = 10, max_tokens: int = 50
):
    """Benchmark inference speed."""
    import torch
    from domains.models import SloughGPTModel

    results = {"model": model, "iterations": num_iterations, "tokens": max_tokens, "times_ms": []}

    try:
        # Load model if not cached
        if model not in _model_cache:
            checkpoint = torch.load(f"models/{model}.pt", weights_only=False)
            chars = checkpoint["chars"]
            stoi = checkpoint["stoi"]
            itos = checkpoint["itos"]

            model_obj = NanoGPT(
                vocab_size=len(chars),
                n_embed=checkpoint.get("config", {}).get("n_embed", 128),
                n_layer=checkpoint.get("config", {}).get("n_layer", 4),
                n_head=checkpoint.get("config", {}).get("n_head", 4),
                block_size=checkpoint.get("config", {}).get("block_size", 64),
            )
            model_obj.load_state_dict(checkpoint["model"])
            model_obj.eval()

            _model_cache[model] = {"model": model_obj, "stoi": stoi, "itos": itos}

        cached = _model_cache[model]
        model_obj = cached["model"]
        stoi = cached.get("stoi", {})

        if stoi:
            # Run benchmarks
            for i in range(num_iterations):
                idx = torch.tensor([[stoi.get("a", 0)]])

                start = time.time()
                with torch.no_grad():
                    output = model_obj.generate(idx, max_new_tokens=max_tokens)
                elapsed = (time.time() - start) * 1000

                results["times_ms"].append(elapsed)

            results["avg_ms"] = sum(results["times_ms"]) / len(results["times_ms"])
            results["min_ms"] = min(results["times_ms"])
            results["max_ms"] = max(results["times_ms"])
            results["status"] = "completed"
        else:
            results["status"] = "error"
            results["error"] = "Model not loaded"

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)

    return results


# ============================================================================
# History
# ============================================================================

_conversation_history = []


@app.get("/history")
async def get_history(limit: int = 50):
    """Get conversation history."""
    return {"history": _conversation_history[-limit:], "count": len(_conversation_history)}


@app.post("/history/clear")
async def clear_history():
    """Clear conversation history."""
    _conversation_history.clear()
    return {"status": "cleared"}


# ============================================================================
# Utilities & Status
# ============================================================================


@app.get("/status")
async def get_status():
    """Get complete system status."""
    import platform

    return {
        "api": {
            "version": __version__,
            "status": "running",
            "endpoints": 45,
            "models_cached": len(_model_cache),
            "training_jobs": len(_training_jobs),
            "plugins": len(_plugins),
        },
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
        },
        "data": {
            "datasets": 13,
            "models": 3,
        },
    }


@app.get("/summary")
async def get_summary():
    """Get API summary."""
    return {
        "name": "SloughGPT API",
        "version": __version__,
        "description": "Comprehensive AI Framework API",
        "endpoints": {
            "core": ["/health", "/info", "/metrics", "/config"],
            "inference": ["/infer", "/infer/stream", "/infer/batch", "/compare"],
            "training": ["/training/start", "/training/lora", "/training/status"],
            "optimization": ["/optimize/quantize", "/validate/model", "/benchmark"],
            "data": ["/datasets", "/models"],
            "system": ["/cache", "/logs", "/plugins"],
        },
        "features": [
            "Neural Networks",
            "LoRA Fine-tuning",
            "Quantization",
            "RLHF Support",
            "Personalities",
            "Streaming",
            "WebSocket",
            "Authentication",
            "Plugin System",
        ],
    }


# ============================================================================
# Health Check
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "SloughGPT API",
        "version": __version__,
        "status": "running",
        "docs": "/docs",
        "ui": "/web",
        "dashboard": "/dashboard",
    }


# ============================================================================
# Batch Processing Utilities
# ============================================================================


@app.post("/batch/process")
async def batch_process(items: List[str], operation: str = "uppercase"):
    """Process a batch of items."""
    results = []

    for item in items:
        if operation == "uppercase":
            results.append(item.upper())
        elif operation == "lowercase":
            results.append(item.lower())
        elif operation == "reverse":
            results.append(item[::-1])
        elif operation == "length":
            results.append(len(item))
        else:
            results.append(item)

    return {"items": items, "results": results, "operation": operation, "count": len(items)}


# ============================================================================
# Rate Limit Status
# ============================================================================


@app.get("/rate-limit/status")
async def get_rate_limit_status():
    """Get current rate limit status."""
    return {
        "enabled": True,
        "max_requests": 100,
        "window_seconds": 60,
        "current_usage": 0,
        "remaining": 100,
    }


# ============================================================================
# Error Recovery
# ============================================================================

_error_count = 0


@app.get("/errors/count")
async def get_error_count():
    """Get error count."""
    return {"count": _error_count}


@app.post("/errors/reset")
async def reset_error_count():
    """Reset error count."""
    global _error_count
    _error_count = 0
    return {"status": "reset"}


# ============================================================================
# Export
# ============================================================================

__all__ = ["app", "storage"]


# ============================================================================
# Data Transformation Utilities
# ============================================================================


@app.post("/transform/encode")
async def encode_text(text: str, encoding: str = "ascii"):
    """Encode text to bytes."""
    try:
        encoded = text.encode(encoding)
        return {"text": text, "encoded": encoded.hex(), "encoding": encoding}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/transform/decode")
async def decode_text(hex_string: str, encoding: str = "ascii"):
    """Decode hex string to text."""
    try:
        decoded = bytes.fromhex(hex_string).decode(encoding)
        return {"hex": hex_string, "decoded": decoded, "encoding": encoding}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Model Versioning
# ============================================================================

_model_versions = {}


@app.get("/versions")
async def list_model_versions():
    """List all model versions."""
    return {"versions": list(_model_versions.keys())}


@app.post("/versions/{model_name}")
async def create_version(model_name: str, description: str = ""):
    """Create a model version snapshot."""
    import shutil
    from pathlib import Path

    version_id = f"{model_name}_v{len(_model_versions) + 1}"

    # Copy model
    src = Path(f"models/{model_name}.pt")
    if src.exists():
        dst = Path(f"models/versions/{version_id}.pt")
        dst.parent.mkdir(exist_ok=True)
        shutil.copy(src, dst)

    _model_versions[version_id] = {
        "model": model_name,
        "description": description,
        "created_at": datetime.now().isoformat(),
    }

    return {"version": version_id, "status": "created"}


# ============================================================================
# Advanced Caching
# ============================================================================

_cache_stats = {"hits": 0, "misses": 0}


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    total = _cache_stats["hits"] + _cache_stats["misses"]
    hit_rate = _cache_stats["hits"] / total if total > 0 else 0

    return {
        "hits": _cache_stats["hits"],
        "misses": _cache_stats["misses"],
        "hit_rate": hit_rate,
        "cached_models": list(_model_cache.keys()),
    }


@app.post("/cache/stats/reset")
async def reset_cache_stats():
    """Reset cache statistics."""
    _cache_stats["hits"] = 0
    _cache_stats["misses"] = 0
    return {"status": "reset"}


# ============================================================================
# Database Endpoints
# ============================================================================

from domains.infrastructure.db_manager import db, init_db


@app.post("/db/init")
async def init_database(database_url: str = None):
    """Initialize the database."""
    success = init_db(database_url)
    return {"status": "initialized" if success else "failed"}


@app.post("/db/conversations")
async def create_conversation(name: str = None):
    """Create a new conversation in database."""
    conv = db.create_conversation(name=name)
    return conv


@app.get("/db/conversations")
async def list_conversations_db():
    """List all conversations from database."""
    convs = db.list_conversations()
    return {"conversations": convs}


@app.get("/db/conversations/{conv_id}")
async def get_conversation_db(conv_id: str):
    """Get a conversation from database."""
    conv = db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.delete("/db/conversations/{conv_id}")
async def delete_conversation_db(conv_id: str):
    """Delete a conversation from database."""
    success = db.delete_conversation(conv_id)
    return {"deleted": success}


@app.post("/db/conversations/{conv_id}/messages")
async def add_message(conv_id: str, role: str, content: str):
    """Add a message to a conversation."""
    msg = db.add_message(conv_id, role, content)
    return msg


@app.get("/db/conversations/{conv_id}/messages")
async def get_messages(conv_id: str):
    """Get messages from a conversation."""
    msgs = db.get_messages(conv_id)
    return {"messages": msgs}


@app.get("/db/training-jobs/{job_id}")
async def get_training_job_db(job_id: str):
    """Get training job from database."""
    job = db.get_training_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ============================================================================
# Model Management
# ============================================================================


@app.get("/models")
async def list_models():
    """List all available models."""
    models = [
        {
            "id": "sloughgpt",
            "name": "SloughGPT",
            "provider": "Local",
            "status": "available",
            "description": "Custom trained GPT model on Shakespeare",
        },
        {
            "id": "gpt2",
            "name": "GPT-2",
            "provider": "HuggingFace",
            "status": "available",
            "description": "124M parameter GPT-2 model",
        },
        {
            "id": "gpt2-medium",
            "name": "GPT-2 Medium",
            "provider": "HuggingFace",
            "status": "available",
            "description": "355M parameter GPT-2 model",
        },
        {
            "id": "gpt2-large",
            "name": "GPT-2 Large",
            "provider": "HuggingFace",
            "status": "available",
            "description": "774M parameter GPT-2 model",
        },
    ]

    return {"models": models}


@app.post("/models/load/{model_name}")
async def load_model(model_name: str):
    """Load a model into memory."""
    import torch

    if model_name in _model_cache:
        return {"status": "already_loaded", "model": model_name}

    try:
        if model_name == "sloughgpt":
            # Load local model
            checkpoint = torch.load("models/sloughgpt.pt", weights_only=False)
            chars = checkpoint["chars"]
            stoi = checkpoint["stoi"]
            itos = checkpoint["itos"]

            from domains.models import SloughGPTModel

            model = SloughGPTModel(
                vocab_size=len(chars),
                n_embed=checkpoint.get("config", {}).get("n_embed", 128),
                n_layer=checkpoint.get("config", {}).get("n_layer", 4),
                n_head=checkpoint.get("config", {}).get("n_head", 4),
                block_size=checkpoint.get("config", {}).get("block_size", 64),
            )
            model.load_state_dict(checkpoint["model"])
            model.eval()

            _model_cache[model_name] = {"model": model, "stoi": stoi, "itos": itos, "type": "local"}

            return {"status": "loaded", "model": model_name}

        elif model_name.startswith("gpt2"):
            # Load from HuggingFace
            from transformers import GPT2LMHeadModel, GPT2Tokenizer

            model_id = "gpt2" if model_name == "gpt2" else f"gpt2-{model_name.replace('gpt2-', '')}"

            tokenizer = GPT2Tokenizer.from_pretrained(model_id)
            model = GPT2LMHeadModel.from_pretrained(model_id)
            model.eval()

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            _model_cache[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "type": "huggingface",
            }

            return {"status": "loaded", "model": model_name, "source": "huggingface"}

        else:
            raise HTTPException(status_code=404, detail="Model not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/unload/{model_name}")
async def unload_model(model_name: str):
    """Unload a model from memory."""
    if model_name in _model_cache:
        del _model_cache[model_name]
        return {"status": "unloaded", "model": model_name}
    return {"status": "not_loaded", "model": model_name}
