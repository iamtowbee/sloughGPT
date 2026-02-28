"""
SloughGPT API Server
Comprehensive FastAPI backend for the SloughGPT Web UI
"""

import os
import shutil
import json
import psutil
import time
import uuid
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
from functools import wraps
import threading
import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import logging

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
                "expires_at": time.time() + ttl if ttl > 0 else None
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
                1 for e in self._cache.values()
                if e["expires_at"] and e["expires_at"] < now
            )
            return {
                "total_keys": len(self._cache),
                "expired_keys": expired,
                "valid_keys": len(self._cache) - expired
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
    openapi_url="/openapi.json"
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
                "retry_after": rate_limiter.window_seconds
            }
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
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s"
    )
    
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
            "updated_at": datetime.now().isoformat()
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
                    "description": description
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
                "updated_at": datetime.now().isoformat()
            }
            return self.conversation_metadata[conv_id]
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation by ID."""
        if conversation_id not in self.conversations:
            return None
        return {
            "id": conversation_id,
            "messages": self.conversations[conversation_id],
            "metadata": self.conversation_metadata.get(conversation_id, {})
        }
    
    def list_conversations(self) -> List[Dict]:
        """List all conversations."""
        result = []
        for conv_id, metadata in self.conversation_metadata.items():
            result.append({
                **metadata,
                "message_count": len(self.conversations[conv_id])
            })
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
                    "updated_at": datetime.now().isoformat()
                }
            
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
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
            
            meta = {
                "description": description or "",
                "created_at": datetime.now().isoformat()
            }
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
                "description": description or ""
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
            "error": None
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
        disk = psutil.disk_usage('/')
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
            "timestamp": datetime.now().isoformat()
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
                "model_service": "healthy"
            }
        }


# Initialize storage
storage = Storage()


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """WebSocket connection manager for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SloughGPT API",
        "version": "2.0.0",
        "docs": "/docs"
    }


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
    response_content = f"Echo: {request.message}\n\n(Model: {request.model}, Temperature: {request.temperature})"
    
    # Add assistant message
    assistant_message = storage.add_message(conversation_id, "assistant", response_content)
    
    return {
        "conversation_id": conversation_id,
        "message": assistant_message,
        "model": request.model,
        "usage": {
            "prompt_tokens": len(request.message.split()),
            "completion_tokens": len(response_content.split()),
            "total_tokens": len(request.message.split()) + len(response_content.split())
        }
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
    response_content = f"Echo: {request.message}\n\n(Model: {request.model}, Temperature: {request.temperature})"
    
    async def generate():
        accumulated = ""
        for char in response_content:
            accumulated += char
            chunk = {
                "type": "chunk",
                "content": char,
                "accumulated": accumulated
            }
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
                "total_tokens": len(request.message.split()) + len(accumulated.split())
            }
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
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
            pricing={"prompt": 0.0015, "completion": 0.002}
        ),
        ModelConfig(
            id="gpt-4",
            name="GPT-4",
            provider="OpenAI",
            description="Most capable GPT model",
            context_length=8192,
            pricing={"prompt": 0.03, "completion": 0.06}
        ),
        ModelConfig(
            id="claude-3",
            name="Claude 3",
            provider="Anthropic",
            description="Helpful and harmless AI assistant",
            context_length=200000,
            pricing={"prompt": 0.015, "completion": 0.075}
        ),
        ModelConfig(
            id="nanogpt",
            name="NanoGPT",
            provider="Local",
            description="Custom trained GPT model",
            context_length=512,
            pricing={"prompt": 0.0, "completion": 0.0}
        ),
        ModelConfig(
            id="llama-2-7b",
            name="Llama 2 7B",
            provider="Meta",
            description="Open source language model",
            context_length=4096,
            pricing={"prompt": 0.0, "completion": 0.0}
        ),
        ModelConfig(
            id="mixtral-8x7b",
            name="Mixtral 8x7B",
            provider="Mistral",
            description="Mixture of experts model",
            context_length=32000,
            pricing={"prompt": 0.0, "completion": 0.0}
        )
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
        "processing_time_ms": processing_time
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
                await websocket.send_json({"type": "subscribed", "channels": message.get("channels", [])})
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
        "disk_total_gb": psutil.disk_usage('/').total / (1024**3),
        "conversations_count": len(storage.conversations),
        "datasets_count": len(storage.datasets),
        "training_jobs_count": len(storage.training_jobs)
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
            "total_messages": sum(len(msgs) for msgs in storage.conversations.values())
        },
        "datasets": {
            "total": len(storage.datasets),
            "total_size_bytes": sum(ds.get("size", 0) for ds in storage.datasets.values())
        },
        "training": {
            "total": len(jobs),
            "completed": len(completed_jobs),
            "running": len(running_jobs),
            "pending": len(jobs) - len(completed_jobs) - len(running_jobs)
        },
        "cache": cache.get_stats(),
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
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
            {"level": "INFO", "message": "API server started", "timestamp": datetime.now().isoformat()},
            {"level": "INFO", "message": f"Serving on port 8000", "timestamp": datetime.now().isoformat()},
            {"level": "DEBUG", "message": f"Loaded {len(storage.datasets)} datasets", "timestamp": datetime.now().isoformat()}
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


# ============================================================================
# Export
# ============================================================================

__all__ = ["app", "storage"]
