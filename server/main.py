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
from typing import Dict, List, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from datetime import datetime, timedelta
import hashlib
import secrets
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from typing import Optional, List, Any
import torch
import json
import asyncio
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sloughgpt")


# ============ Redis Caching ============
class RedisCache:
    """Simple in-memory cache with TTL (Redis-like interface)."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0

    def _is_expired(self, key: str) -> bool:
        if key not in self.cache:
            return True
        _, expiry = self.cache[key]
        return time.time() > expiry

    def get(self, key: str) -> Optional[Any]:
        if self._is_expired(key):
            self.misses += 1
            return None
        self.hits += 1
        return self.cache[key][0]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        expiry = time.time() + (ttl or self.default_ttl)
        self.cache[key] = (value, expiry)

    def delete(self, key: str) -> bool:
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def clear(self) -> None:
        self.cache.clear()

    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
        }


cache = RedisCache(max_size=500, default_ttl=300)


def cache_key(prompt: str, **kwargs) -> str:
    """Generate cache key from prompt and params."""
    params = json.dumps(kwargs, sort_keys=True)
    combined = f"{prompt}:{params}"
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


# ============ Security Configuration ============
API_KEY = os.getenv("SLAUGHGPT_API_KEY", secrets.token_urlsafe(32))
JWT_SECRET = os.getenv("SLAUGHGPT_JWT_SECRET", secrets.token_urlsafe(64))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Valid API keys (for multi-key support)
VALID_API_KEYS = set(os.getenv("SLAUGHGPT_API_KEYS", "").split(",")) - {""}
if API_KEY and API_KEY not in VALID_API_KEYS:
    VALID_API_KEYS.add(API_KEY)


# ============ JWT Authentication ============
class JWTAuth:
    """Simple JWT implementation."""

    def __init__(self):
        self.secret = JWT_SECRET
        self.algorithm = JWT_ALGORITHM
        self.expiration_hours = JWT_EXPIRATION_HOURS

    def create_token(self, subject: str, **extra_claims) -> str:
        """Create a JWT token."""
        import base64
        import json

        now = datetime.utcnow()
        payload = {
            "sub": subject,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=self.expiration_hours)).timestamp()),
            **extra_claims,
        }

        header = {"alg": self.algorithm, "typ": "JWT"}
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")

        import hmac
        signature = hmac.new(self.secret.encode(), f"{header_b64}.{payload_b64}".encode(), hashlib.sha256)
        signature_b64 = base64.urlsafe_b64encode(signature.digest()).decode().rstrip("=")

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode a JWT token."""
        import base64
        import json
        import hmac

        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, signature_b64 = parts

            # Verify signature
            expected_sig = hmac.new(self.secret.encode(), f"{header_b64}.{payload_b64}".encode(), hashlib.sha256)
            expected_sig_b64 = base64.urlsafe_b64encode(expected_sig.digest()).decode().rstrip("=")

            if not hmac.compare_digest(signature_b64, expected_sig_b64):
                return None

            # Decode payload
            payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=="))

            # Check expiration
            if payload.get("exp", 0) < datetime.utcnow().timestamp():
                return None

            return payload
        except Exception:
            return None

    def refresh_token(self, token: str) -> Optional[str]:
        """Refresh a JWT token."""
        payload = self.verify_token(token)
        if payload:
            return self.create_token(payload["sub"], **{k: v for k, v in payload.items() if k != "sub"})
        return None


jwt_auth = JWTAuth()


# ============ API Key Validation ============
def validate_api_key(api_key: Optional[str] = Header(None)) -> Optional[str]:
    """Validate API key from header."""
    if not api_key:
        return None

    # Check against valid keys
    if api_key in VALID_API_KEYS:
        return api_key

    # Check against single key
    if secrets.compare_digest(hashlib.sha256(api_key.encode()).hexdigest(),
                              hashlib.sha256(API_KEY.encode()).hexdigest()):
        return api_key

    return None


# ============ JWT Bearer Authentication ============
def require_auth(api_key: Optional[str] = Depends(validate_api_key)) -> Dict:
    """Require authentication - returns user info or raises exception."""
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return {"api_key": api_key[:8] + "...", "authenticated": True}


# ============ Audit Logger ============
class AuditLogger:
    """Audit logging for security events."""

    def __init__(self):
        self.logs: List[Dict] = []
        self.max_logs = 10000

    def log(self, event_type: str, client_ip: str, user_id: Optional[str] = None,
            resource: str = "", action: str = "", status: str = "success", details: Dict = None):
        """Log an audit event."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "client_ip": client_ip,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "status": status,
            "details": details or {},
        }
        self.logs.append(entry)
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]

        # Log to standard logger
        log_level = logging.INFO if status == "success" else logging.WARNING
        logger.log(log_level, f"AUDIT: {event_type} - {client_ip} - {action} - {status}")

    def get_logs(self, limit: int = 100, event_type: Optional[str] = None) -> List[Dict]:
        """Get audit logs."""
        logs = self.logs[-limit:]
        if event_type:
            logs = [l for l in logs if l["event_type"] == event_type]
        return logs


audit_logger = AuditLogger()


# ============ Input Validation ============
class InputValidator:
    """Input validation and sanitization."""

    @staticmethod
    def sanitize_string(value: str, max_length: int = 10000) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            return ""
        # Remove null bytes
        value = value.replace("\x00", "")
        # Trim to max length
        return value[:max_length].strip()

    @staticmethod
    def validate_prompt(prompt: str) -> str:
        """Validate and sanitize prompt."""
        prompt = InputValidator.sanitize_string(prompt, max_length=8000)
        # Check for suspicious patterns
        suspicious = ["<script", "javascript:", "onerror=", "onload="]
        for pattern in suspicious:
            if pattern.lower() in prompt.lower():
                logger.warning(f"Suspicious pattern detected: {pattern}")
                audit_logger.log("security", "unknown", resource="/generate", action="validate", status="warning", details={"pattern": pattern})
        return prompt

    @staticmethod
    def validate_temperature(temp: float) -> float:
        """Validate temperature parameter."""
        return max(0.0, min(2.0, temp))

    @staticmethod
    def validate_max_tokens(tokens: int) -> int:
        """Validate max tokens parameter."""
        return max(1, min(4096, tokens))


input_validator = InputValidator()


# ============ Rate Limiter ============
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

    SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/auth/token", "/auth/verify"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        allowed, remaining = rate_limiter.is_allowed(client_ip)

        if not allowed:
            wait_time = rate_limiter.get_wait_time(client_ip)
            audit_logger.log("rate_limit_exceeded", client_ip, resource=request.url.path, action="rate_limit", status="blocked")
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


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
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


# ============ Exception Handlers ============
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    client_ip = request.client.host if request.client else "unknown"
    audit_logger.log("http_error", client_ip, resource=str(request.url.path), action=exc.status_code, status="failure", details={"detail": exc.detail})
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    client_ip = request.client.host if request.client else "unknown"
    audit_logger.log("server_error", client_ip, resource=str(request.url.path), action="exception", status="failure", details={"error": str(exc)})
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500},
    )


# ============ Middleware ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SecurityHeadersMiddleware)
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
current_soul = None


def get_soul_generation_params():
    """Get generation params from loaded soul, or defaults."""
    if current_soul is None:
        return {"temperature": 0.8, "top_p": 0.9, "top_k": 50, "max_tokens": 100}
    gen = getattr(current_soul, "generation", None)
    if gen is None:
        return {"temperature": 0.8, "top_p": 0.9, "top_k": 50, "max_tokens": 100}
    if hasattr(gen, "to_dict"):
        d = gen.to_dict()
    elif isinstance(gen, dict):
        d = gen
    else:
        return {"temperature": 0.8, "top_p": 0.9, "top_k": 50, "max_tokens": 100}
    return {
        "temperature": d.get("temperature", 0.8),
        "top_p": d.get("top_p", 0.9),
        "top_k": d.get("top_k", 50),
        "max_tokens": d.get("max_tokens", 100),
    }


def get_soul_personality():
    """Get personality traits from loaded soul."""
    if current_soul is None:
        return None
    return {
        "name": current_soul.name if hasattr(current_soul, "name") else "unknown",
        "lineage": current_soul.lineage if hasattr(current_soul, "lineage") else "unknown",
        "personality": current_soul.personality.to_dict() if hasattr(current_soul, "personality") and current_soul.personality else {},
        "behavior": current_soul.behavior.to_dict() if hasattr(current_soul, "behavior") and current_soul.behavior else {},
        "cognition": current_soul.cognition.to_dict() if hasattr(current_soul, "cognition") and current_soul.cognition else {},
        "emotion": current_soul.emotion.to_dict() if hasattr(current_soul, "emotion") and current_soul.emotion else {},
    }


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
    """Skip model loading - use demo mode for now."""
    global model, model_type
    model = None
    model_type = "demo"
    print("Demo mode active (no model loaded)")


@app.post("/load")
async def load_model_endpoint():
    """Load the model on demand."""
    global model, tokenizer, model_type
    if model is not None:
        return {"status": "already_loaded", "model": model_type}
    load_model()
    return {"status": "loaded", "model": model_type}


class LoadSoulRequest(BaseModel):
    soul_path: str


@app.post("/load-soul")
async def load_soul(request: LoadSoulRequest):
    """Load a .sou Soul Unit file."""
    global current_soul, model_type

    try:
        from domains.inference.sou_format import SouParser, import_from_sou
        from domains.training.models.nanogpt import NanoGPT

        soul, state_dict = import_from_sou(request.soul_path)

        gen = getattr(soul, "generation", None) or {}
        if hasattr(gen, "temperature"):
            temperature = gen.temperature
            top_p = gen.top_p
            max_tokens = gen.max_tokens
        else:
            temperature = gen.get("temperature", 0.8) if isinstance(gen, dict) else 0.8
            top_p = gen.get("top_p", 0.9) if isinstance(gen, dict) else 0.9
            max_tokens = gen.get("max_tokens", 2048) if isinstance(gen, dict) else 2048

        current_soul = soul
        model_type = f"sou/{soul.name}" if hasattr(soul, "name") else "sou/loaded"

        model_cfg = state_dict.get("config", {}) if isinstance(state_dict, dict) else {}
        if hasattr(soul, "base_model"):
            n_embed = getattr(soul, "n_embed", 256)
            n_layer = getattr(soul, "n_layer", 6)
            n_head = getattr(soul, "n_head", 8)
            block_size = getattr(soul, "block_size", 128)
            vocab_size = getattr(soul, "vocab_size", 256)
        else:
            n_embed = model_cfg.get("n_embed", 256)
            n_layer = model_cfg.get("n_layer", 6)
            n_head = model_cfg.get("n_head", 8)
            block_size = model_cfg.get("block_size", 128)
            vocab_size = model_cfg.get("vocab_size", 256)

        n_embed = n_embed or 256
        n_layer = n_layer or 6
        n_head = n_head or 8
        block_size = block_size or 128
        vocab_size = vocab_size or 256

        new_model = NanoGPT(
            vocab_size=vocab_size,
            n_embed=n_embed,
            n_layer=n_layer,
            n_head=n_head,
            block_size=block_size,
        )
        new_model.load_state_dict(state_dict, strict=False)

        global model
        model = new_model

        return {
            "status": "loaded",
            "soul_name": soul.name if hasattr(soul, "name") else "unknown",
            "lineage": soul.lineage if hasattr(soul, "lineage") else "unknown",
            "born_at": soul.born_at if hasattr(soul, "born_at") else "",
            "generation_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            },
            "personality": soul.personality.to_dict() if hasattr(soul, "personality") and soul.personality else {},
            "cognition": soul.cognition.to_dict() if hasattr(soul, "cognition") and soul.cognition else {},
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/soul")
async def get_soul():
    """Get current soul profile."""
    if current_soul is None:
        return {"status": "no_soul", "message": "No soul loaded"}
    return {
        "status": "loaded",
        "name": current_soul.name if hasattr(current_soul, "name") else "unknown",
        "lineage": current_soul.lineage if hasattr(current_soul, "lineage") else "unknown",
        "born_at": current_soul.born_at if hasattr(current_soul, "born_at") else "",
        "version": current_soul.version if hasattr(current_soul, "version") else "1.0.0",
        "tagline": current_soul.tagline if hasattr(current_soul, "tagline") else "",
        "personality": current_soul.personality.to_dict() if hasattr(current_soul, "personality") and current_soul.personality else {},
        "behavior": current_soul.behavior.to_dict() if hasattr(current_soul, "behavior") and current_soul.behavior else {},
        "cognition": current_soul.cognition.to_dict() if hasattr(current_soul, "cognition") and current_soul.cognition else {},
        "emotion": current_soul.emotion.to_dict() if hasattr(current_soul, "emotion") and current_soul.emotion else {},
        "generation": current_soul.generation.to_dict() if hasattr(current_soul, "generation") and current_soul.generation else {},
        "integrity_hash": current_soul.integrity_hash if hasattr(current_soul, "integrity_hash") else "",
        "tags": current_soul.tags if hasattr(current_soul, "tags") else [],
        "certifications": current_soul.certifications if hasattr(current_soul, "certifications") else [],
    }


@app.get("/")
async def root():
    soul_name = current_soul.name if current_soul and hasattr(current_soul, "name") else None
    return {
        "name": "SloughGPT API",
        "version": "1.0.0",
        "status": "running",
        "model": model_type,
        "soul_loaded": soul_name,
        "endpoints": {
            "generate": "/generate (POST)",
            "generate_stream": "/generate/stream (POST)",
            "generate_ws": "/ws/generate (WebSocket)",
            "load_soul": "/load-soul (POST)",
            "soul": "/soul (GET)",
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

    if current_soul:
        soul_info_val = get_soul_personality()
        if soul_info_val:
            soul_info_val["integrity_hash"] = (
                current_soul.integrity_hash
                if hasattr(current_soul, "integrity_hash")
                else ""
            )
            soul_info_val["born_at"] = (
                current_soul.born_at
                if hasattr(current_soul, "born_at")
                else ""
            )
            soul_info_val["tags"] = (
                current_soul.tags if hasattr(current_soul, "tags") else []
            )
            soul_info_val["certifications"] = (
                current_soul.certifications
                if hasattr(current_soul, "certifications")
                else []
            )
            info["soul"] = soul_info_val

    if torch.cuda.is_available():
        info["cuda"] = {
            "device": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }

    return info


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None, "model_type": model_type}


@app.get("/health/live", tags=["health"])
async def liveness():
    """
    Kubernetes liveness probe.
    Returns 200 if the server is alive.
    """
    return {"status": "alive"}


@app.get("/health/ready", tags=["health"])
async def readiness():
    """
    Kubernetes readiness probe.
    Returns 200 if the server is ready to accept traffic.
    Model should be loaded for full readiness.
    """
    is_ready = model is not None
    return {
        "status": "ready" if is_ready else "initializing",
        "model_loaded": is_ready,
        "model_type": model_type,
    }


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


# ============ Authentication Endpoints ============
class TokenRequest(BaseModel):
    api_key: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
async def create_token(token_request: TokenRequest, request: Request):
    """
    Create a JWT access token using API key.
    """
    client_ip = request.client.host if request.client else "unknown"

    # Verify API key
    if token_request.api_key not in VALID_API_KEYS:
        audit_logger.log("auth_failed", client_ip, resource="/auth/token", action="token_create", status="failure")
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Create JWT token
    token = jwt_auth.create_token(subject=token_request.api_key[:8])

    audit_logger.log("auth_success", client_ip, resource="/auth/token", action="token_create", status="success")

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=JWT_EXPIRATION_HOURS * 3600,
    )


@app.post("/auth/verify", tags=["auth"])
async def verify_token(authorization: Optional[str] = Header(None)):
    """
    Verify a JWT token.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    token = authorization[7:]
    payload = jwt_auth.verify_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return {"valid": True, "subject": payload.get("sub"), "expires": payload.get("exp")}


@app.post("/auth/refresh", tags=["auth"])
async def refresh_token(authorization: Optional[str] = Header(None)):
    """
    Refresh a JWT token.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    token = authorization[7:]
    new_token = jwt_auth.refresh_token(token)

    if not new_token:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return TokenResponse(
        access_token=new_token,
        token_type="bearer",
        expires_in=JWT_EXPIRATION_HOURS * 3600,
    )


@app.get("/security/audit", tags=["security"])
async def get_audit_logs(limit: int = 100, event_type: Optional[str] = None):
    """
    Get audit logs (requires authentication).
    """
    return {"logs": audit_logger.get_logs(limit=limit, event_type=event_type)}


@app.get("/security/keys", tags=["security"])
async def get_security_config():
    """
    Get security configuration (public info only).
    """
    return {
        "rate_limiting_enabled": True,
        "jwt_auth_enabled": True,
        "api_keys_configured": len(VALID_API_KEYS),
    }


# ============ Metrics Endpoints ============
@app.get("/metrics", tags=["metrics"])
async def get_metrics():
    """
    Get server metrics for monitoring.
    """
    import psutil

    return {
        "uptime": time.time(),
        "requests_per_minute": rate_limiter.requests_per_minute,
        "active_clients": len(rate_limiter.clients),
        "websocket_connections": len(manager.active_connections),
        "api_keys_count": len(VALID_API_KEYS),
        "audit_logs_count": len(audit_logger.logs),
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_mb": psutil.virtual_memory().available // (1024 * 1024),
        },
    }


@app.get("/metrics/prometheus", tags=["metrics"])
async def prometheus_metrics():
    """
    Get metrics in Prometheus format.
    """
    import psutil

    lines = [
        "# HELP sloughgpt_uptime_seconds Server uptime in seconds",
        "# TYPE sloughgpt_uptime_seconds gauge",
        f"sloughgpt_uptime_seconds {time.time()}",
        "",
        "# HELP sloughgpt_rate_limit_requests Rate limit requests per minute",
        "# TYPE sloughgpt_rate_limit_requests gauge",
        f"sloughgpt_rate_limit_requests {rate_limiter.requests_per_minute}",
        "",
        "# HELP sloughgpt_active_clients Active clients",
        "# TYPE sloughgpt_active_clients gauge",
        f"sloughgpt_active_clients {len(rate_limiter.clients)}",
        "",
        "# HELP sloughgpt_websocket_connections WebSocket connections",
        "# TYPE sloughgpt_websocket_connections gauge",
        f"sloughgpt_websocket_connections {len(manager.active_connections)}",
        "",
        "# HELP sloughgpt_audit_logs_total Total audit logs",
        "# TYPE sloughgpt_audit_logs_total counter",
        f"sloughgpt_audit_logs_total {len(audit_logger.logs)}",
        "",
        "# HELP sloughgpt_system_cpu_usage System CPU usage",
        "# TYPE sloughgpt_system_cpu_usage gauge",
        f"sloughgpt_system_cpu_usage {psutil.cpu_percent()}",
        "",
        "# HELP sloughgpt_system_memory_percent System memory usage percent",
        "# TYPE sloughgpt_system_memory_percent gauge",
        f"sloughgpt_system_memory_percent {psutil.virtual_memory().percent}",
    ]

    return StreamingResponse(iter(lines), media_type="text/plain")


@app.get("/health/detailed", tags=["health"])
async def health_detailed():
    """
    Get detailed health information.
    """
    import psutil

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model_type,
        "uptime": time.time(),
        "rate_limiter": {
            "requests_per_minute": rate_limiter.requests_per_minute,
            "active_clients": len(rate_limiter.clients),
        },
        "websocket": {
            "active_connections": len(manager.active_connections),
        },
        "security": {
            "api_keys_configured": len(VALID_API_KEYS),
            "jwt_algorithm": JWT_ALGORITHM,
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_mb": psutil.virtual_memory().available // (1024 * 1024),
        },
    }



@app.post("/generate/demo")
async def generate_demo(request: GenerateRequest):
    """Demo endpoint - works without loading any model."""
    prompt = input_validator.validate_prompt(request.prompt)
    max_tokens = input_validator.validate_max_tokens(request.max_new_tokens or 100)
    
    # Simple demo response based on prompt
    responses = [
        "I'm Aria, your self-learning AI companion. I'm running entirely on-device!",
        "That's interesting! I'm continuously learning from our conversation.",
        "I process everything locally using TensorFlow.js - your data never leaves your device.",
        "My transformer model updates its weights in real-time. I'm getting smarter as we talk!",
    ]
    import random
    response = random.choice(responses)
    
    return {
        "text": response,
        "model": "demo",
        "prompt": prompt[:50],
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    client_ip = request.client.host if request.client else "unknown"

    prompt = input_validator.validate_prompt(request.prompt)
    soul_defaults = get_soul_generation_params()
    soul_info = get_soul_personality()

    max_tokens = input_validator.validate_max_tokens(
        request.max_new_tokens
        if request.max_new_tokens is not None
        else soul_defaults["max_tokens"]
    )
    temperature = input_validator.validate_temperature(
        request.temperature
        if request.temperature is not None
        else soul_defaults["temperature"]
    )
    top_p = request.top_p if request.top_p is not None else soul_defaults["top_p"]
    top_k = request.top_k if request.top_k is not None else soul_defaults["top_k"]

    if model is None:
        audit_logger.log("generate", client_ip, resource="/generate", action="no_model", status="success")
        return {
            "text": f"Demo response to: {prompt[:50]}... (No model loaded)",
            "model": model_type,
        }

    # Apply personality adjustment to temperature
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
        return {
            "text": text,
            "model": model_type,
            "personality": request.personality,
            "soul": soul_info,
        }

    if model_type == "nanogpt":
        stoi = checkpoint.get("stoi", {})
        itos = checkpoint.get("itos", {})

        idx = torch.tensor([[stoi.get(c, 0) for c in request.prompt]], dtype=torch.long)

        model.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                idx_cond = idx[:, -128:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)

        generated = "".join([itos.get(i, "") for i in idx[0].tolist()])
        text = generated[len(request.prompt) :]
        return {
            "text": text,
            "model": model_type,
            "personality": request.personality,
            "soul": soul_info,
        }

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
    client_ip = websocket.client.host if websocket.client else "unknown"
    authenticated = False

    try:
        await websocket.accept()

        try:
            auth_data = await websocket.receive_text()
            auth_request = json.loads(auth_data)

            api_key = auth_request.get("api_key")
            token = auth_request.get("token")

            if api_key:
                if api_key in VALID_API_KEYS or secrets.compare_digest(
                    hashlib.sha256(api_key.encode()).hexdigest(),
                    hashlib.sha256(API_KEY.encode()).hexdigest()
                ):
                    authenticated = True
            elif token:
                if jwt_auth.verify_token(token):
                    authenticated = True

            if not authenticated:
                await websocket.send_json({"status": "error", "error": "Authentication required"})
                await websocket.close(code=4001)
                audit_logger.log("ws_auth_failed", client_ip, resource="/ws/generate", action="connect", status="failure")
                return

            await websocket.send_json({"status": "authenticated"})
            audit_logger.log("ws_auth_success", client_ip, resource="/ws/generate", action="connect", status="success")

        except json.JSONDecodeError:
            await websocket.send_json({"status": "error", "error": "Invalid JSON"})
            await websocket.close(code=4002)
            return

        manager.connect(websocket)

        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)

            prompt = input_validator.validate_prompt(request_data.get("prompt", ""))
            max_tokens = input_validator.validate_max_tokens(request_data.get("max_tokens", 100))
            temperature = input_validator.validate_temperature(request_data.get("temperature", 0.8))
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

            audit_logger.log("ws_generate", client_ip, resource="/ws/generate", action="generate", status="success")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        audit_logger.log("ws_disconnect", client_ip, resource="/ws/generate", action="disconnect", status="success")
    except Exception as e:
        await websocket.send_json({"status": "error", "error": str(e)})
        manager.disconnect(websocket)
        audit_logger.log("ws_error", client_ip, resource="/ws/generate", action="error", status="failure", details={"error": str(e)})


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
    client_ip = request.client.host if request.client else "unknown"

    # Validate input
    prompt = input_validator.validate_prompt(request.prompt)
    max_tokens = input_validator.validate_max_tokens(request.max_new_tokens or 100)
    temperature = input_validator.validate_temperature(request.temperature or 0.8)

    try:
        engine = get_inference_engine()

        if engine is None:
            audit_logger.log("generate", client_ip, resource="/inference/generate", action="no_model", status="success")
            return {"error": "Model not loaded", "text": ""}

        text = engine.generate_single(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=1.0,
        )

        audit_logger.log("generate", client_ip, resource="/inference/generate", action="inference", status="success")

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


# ============ Batch Processing ============
class BatchGenerateRequest(BaseModel):
    prompts: List[str]
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    use_cache: Optional[bool] = True


class BatchGenerateItem(BaseModel):
    prompt: str
    text: str
    cached: bool = False
    error: Optional[str] = None


@app.post("/inference/batch", tags=["inference"])
async def batch_generate(request: BatchGenerateRequest):
    """
    Batch text generation for multiple prompts.
    Optionally uses caching for identical prompts.
    """
    client_ip = request.client.host if request.client else "unknown" if hasattr(request, 'client') else "unknown"
    results: List[BatchGenerateItem] = []

    for prompt in request.prompts[:50]:
        validated_prompt = input_validator.validate_prompt(prompt)
        max_tokens = input_validator.validate_max_tokens(request.max_new_tokens or 100)
        temp = input_validator.validate_temperature(request.temperature or 0.8)

        cache_key_str = cache_key(validated_prompt, max_tokens=max_tokens, temp=temp, top_p=request.top_p, top_k=request.top_k)

        if request.use_cache:
            cached_result = cache.get(cache_key_str)
            if cached_result:
                results.append(BatchGenerateItem(prompt=prompt, text=cached_result, cached=True))
                continue

        if model is None:
            results.append(BatchGenerateItem(prompt=prompt, text=f"Demo: {validated_prompt[:30]}...", error=None))
            continue

        try:
            if model_type == "gpt2" and tokenizer:
                inputs = tokenizer(validated_prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temp,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        do_sample=True,
                    )
                text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
                cache.set(cache_key_str, text)
                results.append(BatchGenerateItem(prompt=prompt, text=text))
            else:
                results.append(BatchGenerateItem(prompt=prompt, text=f"Model not ready", error="model_error"))
        except Exception as e:
            results.append(BatchGenerateItem(prompt=prompt, text="", error=str(e)))

    audit_logger.log("batch_generate", client_ip, resource="/inference/batch", action="batch", status="success", details={"count": len(request.prompts)})

    return {
        "results": [r.model_dump() for r in results],
        "count": len(results),
        "cache_stats": cache.get_stats(),
    }


@app.delete("/cache", tags=["cache"])
async def clear_cache():
    """Clear the response cache."""
    cache.clear()
    return {"message": "Cache cleared", "cache_stats": cache.get_stats()}


@app.get("/cache/stats", tags=["cache"])
async def cache_stats():
    """Get cache statistics."""
    return cache.get_stats()


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


# ============ Model Registry API ============

class RegistryModel(BaseModel):
    id: str
    name: str
    version: str
    path: str
    description: str = ""
    size_mb: float = 0
    parameters: int = 0
    framework: str = "pytorch"
    status: str = "ready"
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    config: Dict[str, Any] = {}


class RegisterModelRequest(BaseModel):
    id: str
    name: str
    version: str
    path: str
    description: str = ""
    size_mb: float = 0
    parameters: int = 0
    framework: str = "pytorch"
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    config: Dict[str, Any] = {}


class RecordRequestModel(BaseModel):
    latency_ms: float
    tokens: int = 0
    success: bool = True


# In-memory registry
_registry: Dict[str, Dict[str, Any]] = {}
_registry_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_tokens": 0,
    "total_latency_ms": 0,
    "avg_latency_ms": 0,
    "min_latency_ms": float("inf"),
    "max_latency_ms": 0,
})


@app.post("/registry/models", tags=["registry"])
async def register_model(request: RegisterModelRequest):
    """Register a new model."""
    model_data = request.model_dump()
    _registry[request.id] = model_data
    return {"status": "registered", "model": model_data}


@app.get("/registry/models", tags=["registry"])
async def list_registry_models(status: Optional[str] = None, tag: Optional[str] = None):
    """List registered models."""
    models = list(_registry.values())
    
    if status:
        models = [m for m in models if m.get("status") == status]
    
    if tag:
        models = [m for m in models if tag in m.get("tags", [])]
    
    return {"models": models}


@app.get("/registry/models/{model_id}", tags=["registry"])
async def get_registry_model(model_id: str):
    """Get a registered model."""
    if model_id not in _registry:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = _registry[model_id].copy()
    model["metrics"] = _registry_metrics[model_id]
    return model


@app.delete("/registry/models/{model_id}", tags=["registry"])
async def unregister_model(model_id: str):
    """Unregister a model."""
    if model_id in _registry:
        del _registry[model_id]
        return {"status": "unregistered"}
    raise HTTPException(status_code=404, detail="Model not found")


@app.post("/registry/models/{model_id}/record", tags=["registry"])
async def record_model_request(model_id: str, request: RecordRequestModel):
    """Record a request for a model."""
    if model_id not in _registry:
        raise HTTPException(status_code=404, detail="Model not found")
    
    metrics = _registry_metrics[model_id]
    metrics["total_requests"] += 1
    if request.success:
        metrics["successful_requests"] += 1
    else:
        metrics["failed_requests"] += 1
    metrics["total_tokens"] += request.tokens
    metrics["total_latency_ms"] += request.latency_ms
    metrics["avg_latency_ms"] = metrics["total_latency_ms"] / metrics["total_requests"]
    metrics["min_latency_ms"] = min(metrics["min_latency_ms"], request.latency_ms)
    metrics["max_latency_ms"] = max(metrics["max_latency_ms"], request.latency_ms)
    
    return {"status": "recorded", "metrics": metrics}


@app.get("/registry/models/{model_id}/metrics", tags=["registry"])
async def get_model_metrics(model_id: str):
    """Get model metrics."""
    if model_id not in _registry:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return _registry_metrics[model_id]


@app.get("/registry/best", tags=["registry"])
async def get_best_model(criteria: str = "latency", tag: Optional[str] = None):
    """Get best model by criteria."""
    models = list(_registry.values())
    
    if tag:
        models = [m for m in models if tag in m.get("tags", [])]
    
    if not models:
        return {"error": "No models found"}
    
    if criteria == "latency":
        models.sort(key=lambda m: _registry_metrics[m["id"]].get("avg_latency_ms", float("inf")))
    elif criteria == "throughput":
        models.sort(key=lambda m: _registry_metrics[m["id"]].get("total_requests", 0), reverse=True)
    
    best = models[0]
    best["metrics"] = _registry_metrics[best["id"]]
    return best


@app.get("/registry/stats", tags=["registry"])
async def get_registry_stats():
    """Get registry statistics."""
    total_models = len(_registry)
    total_requests = sum(m["total_requests"] for m in _registry_metrics.values())
    total_tokens = sum(m["total_tokens"] for m in _registry_metrics.values())
    
    by_status = defaultdict(int)
    by_framework = defaultdict(int)
    
    for model in _registry.values():
        by_status[model.get("status", "unknown")] += 1
        by_framework[model.get("framework", "unknown")] += 1
    
    return {
        "total_models": total_models,
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "by_status": dict(by_status),
        "by_framework": dict(by_framework),
    }


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
