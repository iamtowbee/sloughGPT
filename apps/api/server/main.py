#!/usr/bin/env python3
"""
SloughGPT Model Server
FastAPI server for model inference with HuggingFace fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Import path must exist before domains (see ``domains.torch_runtime``).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVER_ROOT = Path(__file__).resolve().parent
_CORE_PY_ROOT = _REPO_ROOT / "packages" / "core-py"

for _p in (_SERVER_ROOT, _CORE_PY_ROOT, _REPO_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from domains.torch_runtime import apply_api_process_torch_env

apply_api_process_torch_env()

from contextlib import asynccontextmanager
from collections import defaultdict
import threading
from typing import Any, Dict, List, Optional, Tuple
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from datetime import datetime, timedelta
import hashlib
import secrets
import re
import uuid

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Depends,
    Header,
    Request,
    Response,
)
from fastapi.staticfiles import StaticFiles
from federated_routes import router as federated_router
from settings import get_security_settings
from training.router import router as training_router
from training.schemas import TrainingRequest  # noqa: F401 — re-export for tests: ``from main import TrainingRequest``
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator
import torch
import json
import asyncio
import time
import logging

from domains.errors import require_non_empty_prompt, SloughGPTDomainError
from domains.ops.wandb_server import record_inference_call

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sloughgpt")

_PROCESS_START_MONOTONIC = time.monotonic()


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
_sec = get_security_settings()
API_KEY = _sec.primary_api_key
JWT_SECRET = _sec.jwt_secret
JWT_ALGORITHM = _sec.jwt_algorithm
JWT_EXPIRATION_HOURS = _sec.jwt_expiration_hours
VALID_API_KEYS = _sec.valid_api_keys


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

        signature = hmac.new(
            self.secret.encode(), f"{header_b64}.{payload_b64}".encode(), hashlib.sha256
        )
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
            expected_sig = hmac.new(
                self.secret.encode(), f"{header_b64}.{payload_b64}".encode(), hashlib.sha256
            )
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
            return self.create_token(
                payload["sub"], **{k: v for k, v in payload.items() if k != "sub"}
            )
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
    if secrets.compare_digest(
        hashlib.sha256(api_key.encode()).hexdigest(), hashlib.sha256(API_KEY.encode()).hexdigest()
    ):
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

    def log(
        self,
        event_type: str,
        client_ip: str,
        user_id: Optional[str] = None,
        resource: str = "",
        action: str = "",
        status: str = "success",
        details: Optional[Dict] = None,
    ):
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
            self.logs = self.logs[-self.max_logs :]

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
                audit_logger.log(
                    "security",
                    "unknown",
                    resource="/generate",
                    action="validate",
                    status="warning",
                    details={"pattern": pattern},
                )
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


def _first_trainable_device(module: Any) -> torch.device:
    """Device for placing tokenized inputs beside a loaded HF ``model``."""
    try:
        return next(module.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


def _inputs_to_model_device(inputs: Any, model: Any) -> Any:
    """Move tokenizer outputs (``BatchEncoding`` or ``dict``) to the module device."""
    dev = _first_trainable_device(model)
    if dev.type == "meta":
        return inputs
    if hasattr(inputs, "to"):
        return inputs.to(dev)
    if isinstance(inputs, dict):
        return {k: v.to(dev) for k, v in inputs.items()}
    return inputs


def _inference_engine_device_str(module: Any) -> str:
    """String device for ``InferenceEngine`` — match where ``module`` parameters live."""
    if module is None:
        return "cpu"
    try:
        dev = _first_trainable_device(module)
    except Exception:
        return "cpu"
    if dev.type == "meta":
        return "cpu"
    return str(dev.type)


def _format_chat_messages_for_standard(messages: List[ChatMessage]) -> str:
    formatted = ""
    for msg in messages:
        role = msg.role
        content = msg.content
        if role == "system":
            formatted += f"{content}\n\n"
        elif role == "user":
            formatted += f"User: {content}\n\n"
        elif role == "assistant":
            formatted += f"{content}\n\n"
    return formatted.strip() + "\n\n"


def _strip_assistant_prefix(text: str) -> str:
    """Strip 'Assistant:' prefix from generated text."""
    prefixes = ["Assistant:", "Assistant: ", "\nAssistant:", "\nAssistant: "]
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text.lstrip()


def _generate_core(request: GenerateRequest, client_ip: str) -> Dict[str, Any]:
    """Shared generation logic for /generate and /v1/infer."""
    prompt = require_non_empty_prompt(input_validator.validate_prompt(request.prompt))
    soul_defaults = get_soul_generation_params()
    soul_info = get_soul_personality()

    max_tokens = input_validator.validate_max_tokens(
        request.max_new_tokens
        if request.max_new_tokens is not None
        else soul_defaults["max_tokens"]
    )
    temperature = input_validator.validate_temperature(
        request.temperature if request.temperature is not None else soul_defaults["temperature"]
    )
    top_p = request.top_p if request.top_p is not None else soul_defaults["top_p"]
    top_k = request.top_k if request.top_k is not None else soul_defaults["top_k"]

    if soul_engine is not None and soul_engine.is_loaded:
        try:
            text = soul_engine.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            audit_logger.log(
                "generate", client_ip, resource="/generate", action="soul_engine", status="success"
            )
            return {
                "text": text,
                "model": model_type,
                "soul": soul_info,
            }
        except Exception as e:
            logger.error(f"SoulEngine generation failed: {e}")

    # Check for llama.cpp backend (GGUF models)
    llama_model_path = os.environ.get("SLOUGHGPT_MODEL_PATH", "").strip()
    if llama_model_path:
        try:
            from domains.inference.llama_engine import (
                detect_gpu,
                auto_select_backend,
                LlamaInferenceConfig,
                LlamaInferenceEngine,
            )

            gpu = detect_gpu()
            if gpu:
                logger.info(f"GPU: {gpu.name} - {gpu.reason}")

            n_gpu_layers = auto_select_backend(1.5)
            config = LlamaInferenceConfig(model_path=llama_model_path, n_gpu_layers=n_gpu_layers)
            engine = LlamaInferenceEngine(config)

            start_time = time.perf_counter()
            result = engine.benchmark(prompt, max_tokens)
            elapsed = time.perf_counter() - start_time

            backend_name = "llama.cpp-gpu" if n_gpu_layers > 0 else "llama.cpp-cpu"
            tps = result.get("tokens_per_second", 0)
            logger.info(f"llama.cpp backend ({backend_name}): {tps:.1f} tok/s")

            return {
                "text": result.get("text_preview", ""),
                "model": backend_name,
                "tokens_per_second": tps,
                "backend": backend_name,
                "gpu_layers": n_gpu_layers,
            }
        except Exception as e:
            logger.error(f"llama.cpp backend failed: {e}")

    if model is None:
        audit_logger.log(
            "generate", client_ip, resource="/generate", action="no_model", status="success"
        )
        return {
            "text": f"Demo response to: {prompt[:50]}... (No model loaded)",
            "model": model_type,
        }

    if model_type == "gpt2":
        inputs = tokenizer(request.prompt, return_tensors="pt")
        inputs = _inputs_to_model_device(inputs, model)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {
            "text": text,
            "model": model_type,
            "personality": request.personality,
            "soul": soul_info,
        }

    if model_type == "sloughgpt_finetuned":
        try:
            checkpoint = model
            if isinstance(checkpoint, dict):
                stoi = checkpoint.get("stoi", {})
                itos = checkpoint.get("itos", {})
                if not stoi and "stoi" in checkpoint:
                    stoi = checkpoint["stoi"]
                if not itos and "itos" in checkpoint:
                    itos = checkpoint["itos"]
                model_to_use = checkpoint.get("model", checkpoint)
                if not hasattr(model_to_use, "eval") and isinstance(model_to_use, dict):
                    pass
                idx = torch.tensor([[stoi.get(c, 0) for c in request.prompt]], dtype=torch.long)
                model_to_use.eval()
                with torch.no_grad():
                    for _ in range(max_tokens):
                        idx_cond = idx[:, -128:]
                        logits, _ = model_to_use(idx_cond)
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
        except Exception as e:
            logger.error(f"Failed to load NanoGPT model: {e}")
            audit_logger.log(
                "generate", client_ip, resource="/generate", action="model_error", status="failure"
            )
            return {
                "text": f"Demo response to: {prompt[:50]}... (Model loading failed: {str(e)[:50]})",
                "model": model_type,
                "error": str(e),
            }

    return {"text": "Model type not supported", "model": model_type}


_CLINICAL_DISCLAIMER = (
    "Not medical advice; verify with a licensed professional and institutional policy."
)


def _standard_apply_safety(
    task_type: str,
    safety_level: str,
    text: str,
    *,
    structured_output: bool = False,
) -> tuple[str, List[StandardSafetyFlagBody]]:
    """Minimal v1 hooks; replace with YAML-driven classifiers later."""
    flags: List[StandardSafetyFlagBody] = []
    clinical_tasks = {
        "clinical_assist",
        "clinical_summarization",
        "medical_rag_qa",
        "synthetic_patient_generation",
    }
    if safety_level == "clinical_strict" and task_type in clinical_tasks:
        if structured_output:
            flags.append(
                StandardSafetyFlagBody(
                    rule_id="require_disclaimer_clinical",
                    severity="info",
                    message="Clinical disclaimer must be shown in client UI (structured output; not appended to JSON).",
                )
            )
        else:
            text = f"{text.rstrip()}\n\n{_CLINICAL_DISCLAIMER}"
            flags.append(
                StandardSafetyFlagBody(
                    rule_id="require_disclaimer_clinical",
                    severity="info",
                    message="Disclaimer appended per clinical_strict policy.",
                )
            )
    if safety_level == "code_only" and task_type in ("code_completion", "code_rag_qa"):
        flags.append(
            StandardSafetyFlagBody(
                rule_id="code_policy_bundle_hint",
                severity="info",
                message="Run tests and SAST before merge.",
            )
        )
    return text, flags


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
        self.clients[client_id] = [ts for ts in self.clients[client_id] if ts > cutoff]

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


# ----- HTTP metrics for Prometheus (counters + histogram) -----
_HTTP_HIST_BOUNDS: Tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)


def _uuid_like(segment: str) -> bool:
    if len(segment) == 36 and segment.count("-") == 4:
        return all(
            len(p) == ind and all(c in "0123456789abcdefABCDEF" for c in p)
            for p, ind in zip(segment.split("-"), (8, 4, 4, 4, 12))
        )
    if len(segment) == 32 and all(c in "0123456789abcdefABCDEF" for c in segment):
        return True
    return False


def _normalize_endpoint(path: str) -> str:
    if not path or path == "/":
        return "/"
    parts: List[str] = []
    for segment in path.split("/"):
        if not segment:
            continue
        if segment.isdigit() or _uuid_like(segment):
            parts.append("{id}")
        else:
            parts.append(segment)
    return "/" + "/".join(parts[:12])


def _prometheus_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class HttpMetricsCollector:
    """In-process HTTP RED-style metrics (thread-safe for async workers)."""

    __slots__ = ("_lock", "requests_total", "dur_sum", "dur_count", "dur_buckets")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_total: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self.dur_sum: Dict[Tuple[str, str], float] = defaultdict(float)
        self.dur_count: Dict[Tuple[str, str], int] = defaultdict(int)
        n_buckets = len(_HTTP_HIST_BOUNDS) + 1
        self.dur_buckets: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0] * n_buckets)

    def observe(self, method: str, endpoint: str, status_code: int, duration_s: float) -> None:
        ep = _normalize_endpoint(endpoint)
        st = str(status_code)
        key_ct = (method.upper(), ep, st)
        key_d = (method.upper(), ep)
        with self._lock:
            self.requests_total[key_ct] += 1
            self.dur_sum[key_d] += duration_s
            self.dur_count[key_d] += 1
            row = self.dur_buckets[key_d]
            placed = False
            for i, bound in enumerate(_HTTP_HIST_BOUNDS):
                if duration_s <= bound:
                    for j in range(i, len(row)):
                        row[j] += 1
                    placed = True
                    break
            if not placed:
                row[-1] += 1

    def prometheus_lines(self) -> List[str]:
        lines: List[str] = [
            "# HELP http_requests_total Total HTTP requests",
            "# TYPE http_requests_total counter",
        ]
        with self._lock:
            totals = sorted(self.requests_total.items())
            d_keys = sorted(self.dur_sum.keys())
            sums = {k: self.dur_sum[k] for k in d_keys}
            counts = {k: self.dur_count[k] for k in d_keys}
            buckets_snapshot = {k: list(self.dur_buckets[k]) for k in d_keys}
        for (method, endpoint, status), n in totals:
            ml = _prometheus_label_value(method)
            el = _prometheus_label_value(endpoint)
            sl = _prometheus_label_value(status)
            lines.append(f'http_requests_total{{method="{ml}",endpoint="{el}",status="{sl}"}} {n}')
        lines.extend(
            [
                "",
                "# HELP http_request_duration_seconds HTTP request duration in seconds",
                "# TYPE http_request_duration_seconds histogram",
            ]
        )
        for method, endpoint in d_keys:
            ml = _prometheus_label_value(method)
            elms = _prometheus_label_value(endpoint)
            row = buckets_snapshot[(method, endpoint)]
            for bound, c in zip(_HTTP_HIST_BOUNDS, row):
                le = str(bound)
                lines.append(
                    f'http_request_duration_seconds_bucket{{method="{ml}",endpoint="{elms}",le="{le}"}} {c}'
                )
            lines.append(
                f'http_request_duration_seconds_bucket{{method="{ml}",endpoint="{elms}",le="+Inf"}} {row[-1]}'
            )
            lines.append(
                f'http_request_duration_seconds_sum{{method="{ml}",endpoint="{elms}"}} {sums[(method, endpoint)]}'
            )
            lines.append(
                f'http_request_duration_seconds_count{{method="{ml}",endpoint="{elms}"}} {counts[(method, endpoint)]}'
            )
        return lines

    def wandb_aggregate(self) -> Dict[str, float]:
        """Point-in-time totals for optional W&B server run (``domains.ops.wandb_server``)."""
        with self._lock:
            total = sum(self.requests_total.values())
            s = sum(self.dur_sum.values())
            c = sum(self.dur_count.values())
        mean_ms = (s / max(c, 1)) * 1000.0
        return {
            "server/http_requests_total": float(total),
            "server/http_latency_mean_ms": float(mean_ms),
        }


http_metrics = HttpMetricsCollector()


class PrometheusHttpMetricsMiddleware(BaseHTTPMiddleware):
    """Record request counts and latencies for /metrics/prometheus (skips scrape endpoints)."""

    SKIP_PATHS = frozenset({"/metrics", "/metrics/prometheus"})

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = time.perf_counter() - start
            http_metrics.observe(request.method, request.url.path, status_code, duration)


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
            audit_logger.log(
                "rate_limit_exceeded",
                client_ip,
                resource=request.url.path,
                action="rate_limit",
                status="blocked",
            )
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
    """Load default HF weights in-process when ``SLOUGHGPT_AUTOLOAD_MODEL`` is set (default: ``gpt2``)."""
    try:
        await asyncio.to_thread(_autoload_hf_model_at_startup)
    except Exception as e:
        logger.warning("Startup model autoload failed: %s", e)
    wandb_server_task: Optional[asyncio.Task] = None
    try:
        from domains.ops.wandb_server import start_wandb_server_background

        def _wandb_server_extra_metrics() -> Dict[str, Any]:
            """Align W&B server flush with ``GET /info`` host snapshot + API process RSS."""
            from host_metrics import sample_host_metrics_sync

            h = sample_host_metrics_sync()
            out: Dict[str, Any] = {
                "host/cpu_percent": float(h["cpu_percent"]),
                "host/memory_percent": float(h["memory_percent"]),
            }
            rss = h.get("process_rss_bytes")
            if isinstance(rss, int) and rss >= 0:
                out["server/process_rss_bytes"] = float(rss)
            return out

        wandb_server_task = await start_wandb_server_background(
            http_metrics,
            extra_metrics=_wandb_server_extra_metrics,
        )
    except Exception as e:
        logger.warning("W&B server background task did not start: %s", e)
    yield
    if wandb_server_task is not None:
        wandb_server_task.cancel()
        try:
            await wandb_server_task
        except asyncio.CancelledError:
            pass


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
    audit_logger.log(
        "http_error",
        client_ip,
        resource=str(request.url.path),
        action=str(exc.status_code),
        status="failure",
        details={"detail": exc.detail},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(SloughGPTDomainError)
async def domain_exception_handler(request: Request, exc: SloughGPTDomainError):
    """Map core ``domains`` errors to JSON (validation and other predictable failures)."""
    client_ip = request.client.host if request.client else "unknown"
    audit_logger.log(
        "domain_error",
        client_ip,
        resource=str(request.url.path),
        action=exc.code,
        status="failure",
        details={"detail": str(exc)},
    )
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": str(exc), "status_code": exc.http_status, "code": exc.code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    client_ip = request.client.host if request.client else "unknown"
    audit_logger.log(
        "server_error",
        client_ip,
        resource=str(request.url.path),
        action="exception",
        status="failure",
        details={"error": str(exc)},
    )
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
app.add_middleware(PrometheusHttpMetricsMiddleware)


# ============ Federated Learning Routes ============
app.include_router(federated_router)
app.include_router(training_router)


# Serve model files (GGUF, etc.) when repo ``models/`` exists (path stable regardless of cwd).
_models_dir = (Path(__file__).resolve().parent.parent / "models").resolve()
if _models_dir.is_dir():
    app.mount("/models", StaticFiles(directory=str(_models_dir)), name="models")


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
soul_engine = None


def get_soul_generation_params():
    """Get generation params from loaded soul, or defaults."""
    if soul_engine is not None:
        gen = soul_engine.soul.generation
        return {
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "top_k": gen.top_k,
            "max_tokens": gen.max_tokens,
        }
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
    if soul_engine is not None:
        soul = soul_engine.soul
        return {
            "name": soul.name,
            "lineage": soul.lineage,
            "personality": soul.personality.to_dict() if soul.personality else {},
            "behavior": soul.behavior.to_dict() if soul.behavior else {},
            "cognition": soul.cognition.to_dict() if soul.cognition else {},
            "emotion": soul.emotion.to_dict() if soul.emotion else {},
        }
    if current_soul is None:
        return None
    return {
        "name": current_soul.name if hasattr(current_soul, "name") else "unknown",
        "lineage": current_soul.lineage if hasattr(current_soul, "lineage") else "unknown",
        "personality": current_soul.personality.to_dict()
        if hasattr(current_soul, "personality") and current_soul.personality
        else {},
        "behavior": current_soul.behavior.to_dict()
        if hasattr(current_soul, "behavior") and current_soul.behavior
        else {},
        "cognition": current_soul.cognition.to_dict()
        if hasattr(current_soul, "cognition") and current_soul.cognition
        else {},
        "emotion": current_soul.emotion.to_dict()
        if hasattr(current_soul, "emotion") and current_soul.emotion
        else {},
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
    top_k: Optional[int] = 50
    model: Optional[str] = None


# ============ SloughGPT Standard v1 inference envelope ============
class StandardInferenceInput(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    context: Optional[str] = None


class StandardInferenceGeneration(BaseModel):
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.0
    seed: Optional[int] = None


class StandardInferenceSafety(BaseModel):
    level: str = "standard"
    policy_bundle: Optional[str] = None


class StandardInferenceRequest(BaseModel):
    """Request body for POST /v1/infer (see standards/SLOUGHGPT_STANDARD_V1.md)."""

    trace_id: Optional[str] = None
    tenant_id: Optional[str] = None
    model_id: Optional[str] = None
    task_type: str = "general_generate"
    mode: str = "generate"
    safety: Optional[StandardInferenceSafety] = None
    output_schema_ref: Optional[str] = None
    retrieval: Optional[Dict[str, Any]] = None
    input: StandardInferenceInput
    generation: Optional[StandardInferenceGeneration] = None

    @model_validator(mode="after")
    def _validate_mode_input(self) -> "StandardInferenceRequest":
        mode = self.mode
        if mode == "generate":
            if not (self.input.prompt and self.input.prompt.strip()):
                raise ValueError("input.prompt is required when mode is 'generate'")
        elif mode == "chat":
            if not self.input.messages:
                raise ValueError("input.messages is required when mode is 'chat'")
        elif mode == "structured":
            if not (self.input.prompt and self.input.prompt.strip()) and not self.input.messages:
                raise ValueError("structured mode requires input.prompt or input.messages")
        else:
            raise ValueError("mode must be 'generate', 'chat', or 'structured'")
        return self


class StandardOutputBody(BaseModel):
    text: Optional[str] = None
    structured: Optional[Dict[str, Any]] = None


class StandardUsageBody(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: float = 0.0


class StandardSafetyFlagBody(BaseModel):
    rule_id: str
    severity: str
    message: Optional[str] = None


class StandardInferenceResponse(BaseModel):
    trace_id: str
    model_id: str
    model_version: str
    task_type: str
    mode: str
    output: StandardOutputBody
    usage: StandardUsageBody
    safety_flags: List[StandardSafetyFlagBody] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)


def load_model():
    """Load the actual sloughgpt model."""
    global model, tokenizer, model_type
    try:
        from pathlib import Path
        import torch

        # Check for available models (relative to server directory)
        model_paths = [
            "../models/sloughgpt_finetuned.pt",
            "../models/sloughgpt_lora.pt",
            "../models/sloughgpt_variant.pt",
        ]

        model_path = None
        for path in model_paths:
            if Path(path).exists():
                model_path = path
                break

        if model_path is None:
            # Fallback to demo mode
            model = None
            model_type = "demo"
            print("No model found, demo mode active")
            return

        # Load the model checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location="cpu")

        # Extract model and tokenizer info
        if isinstance(checkpoint, dict):
            # Store the full checkpoint
            model = checkpoint
            model_type = "sloughgpt_finetuned"

            # Also set up tokenizer-like info for generation
            if "chars" in checkpoint and "stoi" in checkpoint and "itos" in checkpoint:
                # These are needed for the simple character-level model
                tokenizer = {
                    "chars": checkpoint["chars"],
                    "stoi": checkpoint["stoi"],
                    "itos": checkpoint["itos"],
                    "vocab_size": len(checkpoint["chars"]),
                }
                print(f"✅ Tokenizer loaded: {len(checkpoint['chars'])} characters")
            else:
                tokenizer = None

            print(f"✅ Model loaded successfully from {model_path}")
            print(f"📊 Model contains {len(checkpoint.get('model', {}))} parameters")
        else:
            # Assume the checkpoint is the model itself
            model = checkpoint
            model_type = "sloughgpt_finetuned"
            tokenizer = None
            print(f"✅ Model loaded successfully from {model_path}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        # Fallback to demo mode
        model = None
        tokenizer = None
        model_type = "demo"


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
    """Load a .sou Soul Unit file into SoulEngine."""
    global current_soul, model_type, soul_engine

    try:
        from domains.core import SoulEngine

        engine = SoulEngine(device="cpu")
        soul = engine.load_soul(request.soul_path)

        current_soul = soul
        soul_engine = engine
        model_type = f"sou/{soul.name}" if hasattr(soul, "name") else "sou/loaded"
        model = engine.model

        return {
            "status": "loaded",
            "soul_name": soul.name if hasattr(soul, "name") else "unknown",
            "lineage": soul.lineage if hasattr(soul, "lineage") else "unknown",
            "born_at": soul.born_at if hasattr(soul, "born_at") else "",
            "generation_params": {
                "temperature": soul.generation.temperature if soul.generation else 0.8,
                "top_p": soul.generation.top_p if soul.generation else 0.9,
                "max_tokens": soul.generation.max_tokens if soul.generation else 2048,
            },
            "personality": soul.personality.to_dict()
            if hasattr(soul, "personality") and soul.personality
            else {},
            "cognition": soul.cognition.to_dict()
            if hasattr(soul, "cognition") and soul.cognition
            else {},
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
        "personality": current_soul.personality.to_dict()
        if hasattr(current_soul, "personality") and current_soul.personality
        else {},
        "behavior": current_soul.behavior.to_dict()
        if hasattr(current_soul, "behavior") and current_soul.behavior
        else {},
        "cognition": current_soul.cognition.to_dict()
        if hasattr(current_soul, "cognition") and current_soul.cognition
        else {},
        "emotion": current_soul.emotion.to_dict()
        if hasattr(current_soul, "emotion") and current_soul.emotion
        else {},
        "generation": current_soul.generation.to_dict()
        if hasattr(current_soul, "generation") and current_soul.generation
        else {},
        "integrity_hash": current_soul.integrity_hash
        if hasattr(current_soul, "integrity_hash")
        else "",
        "tags": current_soul.tags if hasattr(current_soul, "tags") else [],
        "certifications": current_soul.certifications
        if hasattr(current_soul, "certifications")
        else [],
    }


@app.get("/")
async def root():
    soul_name = None
    if soul_engine is not None and soul_engine.soul:
        soul_name = soul_engine.soul.name
    elif current_soul and hasattr(current_soul, "name"):
        soul_name = current_soul.name
    return {
        "name": "SloughGPT API",
        "version": "1.0.0",
        "status": "running",
        "model": model_type,
        "soul_loaded": soul_name,
        "soul_engine_active": soul_engine is not None and soul_engine.is_loaded,
        "endpoints": {
            "generate": "/generate (POST)",
            "v1_infer": "/v1/infer (POST) — SloughGPT Standard v1 envelope",
            "generate_stream": "/generate/stream (POST)",
            "generate_ws": "/ws/generate (WebSocket)",
            "load_soul": "/load-soul (POST) - loads into SoulEngine",
            "soul": "/soul (GET)",
            "personalities": "/personalities (GET)",
            "models": "/models (GET)",
            "datasets": "/datasets (GET)",
            "train_resolve": "/train/resolve (POST) — preview manifest → data_path",
            "info": "/info (GET)",
        },
    }


@app.get("/info")
async def info():
    """Get detailed server info (includes host CPU/RAM when psutil is available)."""
    import torch

    from host_metrics import sample_host_metrics_async

    info = {
        "api_version": "1.0.0",
        "model": {
            "type": model_type,
            "loaded": model is not None,
        },
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    host = await sample_host_metrics_async()
    if host is not None:
        info["host"] = host

    if checkpoint:
        info["model"].update(
            {
                "vocab_size": len(checkpoint.get("stoi", {})),
                "chars": len(checkpoint.get("chars", [])),
            }
        )

    if soul_engine is not None and soul_engine.is_loaded:
        info["soul_engine"] = soul_engine.get_stats()
    elif current_soul:
        soul_info_val = get_soul_personality()
        if soul_info_val:
            soul_info_val["integrity_hash"] = (
                current_soul.integrity_hash if hasattr(current_soul, "integrity_hash") else ""
            )
            soul_info_val["born_at"] = (
                current_soul.born_at if hasattr(current_soul, "born_at") else ""
            )
            soul_info_val["tags"] = current_soul.tags if hasattr(current_soul, "tags") else []
            soul_info_val["certifications"] = (
                current_soul.certifications if hasattr(current_soul, "certifications") else []
            )
            info["soul"] = soul_info_val

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_b = int(props.total_memory)
        used_b = int(torch.cuda.memory_allocated(0))
        info["cuda"] = {
            "device": torch.cuda.get_device_name(0),
            "memory_total": total_b / 1e9,
            "memory_total_bytes": total_b,
            "memory_used_bytes": used_b,
            "memory_percent": round(100.0 * used_b / max(total_b, 1), 2),
        }

    return info


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model_type,
        "soul_engine_active": soul_engine is not None and soul_engine.is_loaded,
        "soul_name": soul_engine.soul.name if soul_engine and soul_engine.soul else None,
    }


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
        "retry_after": 0
        if current_count < rate_limiter.requests_per_minute
        else rate_limiter.get_wait_time(client_ip),
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
        audit_logger.log(
            "auth_failed",
            client_ip,
            resource="/auth/token",
            action="token_create",
            status="failure",
        )
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Create JWT token
    token = jwt_auth.create_token(subject=token_request.api_key[:8])

    audit_logger.log(
        "auth_success", client_ip, resource="/auth/token", action="token_create", status="success"
    )

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
        "inference": _get_inference_metrics(),
    }


def _get_inference_metrics():
    """Get inference-specific metrics from llama engine."""
    try:
        from domains.inference.llama_engine import get_inference_stats, get_memory_usage

        stats = get_inference_stats()
        memory = get_memory_usage()
        return {
            "total_requests": stats.get("total_requests", 0),
            "total_tokens": stats.get("total_tokens", 0),
            "avg_tokens_per_second": round(stats.get("avg_tokens_per_second", 0), 2),
            "cache_hits": stats.get("cache_hits", 0),
            "cached_models": stats.get("cached_models", []),
            "memory_rss_mb": round(memory.get("rss_mb", 0), 1) if "error" not in memory else None,
        }
    except Exception:
        return {"error": "inference metrics unavailable"}


@app.get("/metrics/prometheus", tags=["metrics"])
async def prometheus_metrics():
    """
    Get metrics in Prometheus format.
    """
    import psutil

    uptime_s = time.monotonic() - _PROCESS_START_MONOTONIC
    lines = [
        "# HELP sloughgpt_uptime_seconds Server uptime in seconds",
        "# TYPE sloughgpt_uptime_seconds gauge",
        f"sloughgpt_uptime_seconds {uptime_s}",
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
    lines.append("")
    lines.extend(http_metrics.prometheus_lines())

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
    prompt = require_non_empty_prompt(input_validator.validate_prompt(request.prompt))
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
async def generate(request: GenerateRequest, req: Request):
    client_ip = req.client.host if req.client else "unknown"
    return _generate_core(request, client_ip)


@app.post("/v1/infer", tags=["inference"])
async def v1_infer(
    body: StandardInferenceRequest, req: Request, response: Response
) -> StandardInferenceResponse:
    """SloughGPT Standard v1 inference envelope (see standards/SLOUGHGPT_STANDARD_V1.md)."""
    response.headers["X-SloughGPT-Standard"] = "1"
    trace_id = body.trace_id or req.headers.get("X-Trace-Id") or str(uuid.uuid4())
    client_ip = req.client.host if req.client else "unknown"
    gen = body.generation or StandardInferenceGeneration()

    if body.mode == "chat":
        base_prompt = _format_chat_messages_for_standard(body.input.messages or [])
    elif body.mode == "structured":
        if body.input.messages:
            base_prompt = _format_chat_messages_for_standard(body.input.messages)
        else:
            base_prompt = (body.input.prompt or "").strip()
        base_prompt = base_prompt.rstrip() + "\n\nRespond with valid JSON only."
    else:
        base_prompt = (body.input.prompt or "").strip()

    if body.input.context:
        base_prompt = f"Context:\n{body.input.context.strip()}\n\n{base_prompt}"

    gr = GenerateRequest(
        prompt=base_prompt,
        max_new_tokens=gen.max_new_tokens,
        temperature=gen.temperature,
        top_p=gen.top_p,
        top_k=gen.top_k,
        repetition_penalty=gen.repetition_penalty,
        seed=gen.seed,
        personality=None,
    )

    t0 = time.perf_counter()
    raw = _generate_core(gr, client_ip)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    text = raw.get("text") or ""

    safety_level = body.safety.level if body.safety else "standard"
    safety_flags: List[StandardSafetyFlagBody] = []
    structured: Optional[Dict[str, Any]] = None
    structured_ok = False
    if body.mode == "structured":
        try:
            structured = json.loads(text.strip())
            structured_ok = True
        except json.JSONDecodeError:
            safety_flags.append(
                StandardSafetyFlagBody(
                    rule_id="structured_parse_failed",
                    severity="warn",
                    message="Model output was not valid JSON.",
                )
            )

    text_out, policy_flags = _standard_apply_safety(
        body.task_type,
        safety_level,
        text,
        structured_output=structured_ok,
    )
    safety_flags.extend(policy_flags)

    if body.mode == "structured" and structured_ok:
        display_text = text
    else:
        display_text = text_out

    model_id_out = body.model_id or raw.get("model") or model_type
    audit_logger.log(
        "v1_infer", client_ip, resource="/v1/infer", action=body.task_type, status="success"
    )

    return StandardInferenceResponse(
        trace_id=trace_id,
        model_id=str(model_id_out),
        model_version=str(model_type),
        task_type=body.task_type,
        mode=body.mode,
        output=StandardOutputBody(text=display_text, structured=structured),
        usage=StandardUsageBody(latency_ms=round(latency_ms, 3)),
        safety_flags=safety_flags,
        citations=[],
    )


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """Streaming text generation using Server-Sent Events."""
    require_non_empty_prompt(input_validator.validate_prompt(request.prompt))
    max_gen = input_validator.validate_max_tokens(request.max_new_tokens or 100)
    temperature = input_validator.validate_temperature(request.temperature or 0.8)
    top_p_val = max(0.0, min(1.0, float(request.top_p if request.top_p is not None else 0.9)))

    async def generate_stream_tokens():
        # llama.cpp streaming
        llama_model_path = os.environ.get("SLOUGHGPT_MODEL_PATH", "").strip()
        if llama_model_path:
            try:
                from domains.inference.llama_engine import (
                    detect_gpu,
                    auto_select_backend,
                    LlamaInferenceConfig,
                    LlamaInferenceEngine,
                )

                gpu = detect_gpu()
                n_gpu_layers = auto_select_backend(1.5)
                config = LlamaInferenceConfig(
                    model_path=llama_model_path, n_gpu_layers=n_gpu_layers
                )

                start_time = time.perf_counter()

                for token in engine.generate_stream(request.prompt, max_tokens=max_gen):
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

                elapsed = time.perf_counter() - start_time
                yield f"data: {json.dumps({'token': '', 'done': True, 'elapsed': elapsed})}\n\n"
                return
            except Exception as e:
                logger.error(f"llama.cpp streaming failed: {e}")

        if model is None:
            demo = f"Demo streaming response to: {request.prompt}..."
            for char in demo:
                yield f"data: {json.dumps({'token': char, 'done': False})}\n\n"
                await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
            return

        if model_type == "gpt2":
            loop = asyncio.get_event_loop()

            def _tokenize():
                enc = tokenizer(request.prompt, return_tensors="pt")
                return _inputs_to_model_device(enc, model)

            inputs = await loop.run_in_executor(None, _tokenize)

            def _generate_batch():
                idx = inputs["input_ids"]
                batch_size = 8
                generated = []
                with torch.no_grad():
                    for _ in range(max_gen):
                        logits = model(idx, use_cache=True)[0]
                        logits = logits[:, -1, :]

                        if temperature > 0 and top_p_val < 1.0:
                            logits = logits / temperature
                            probs = torch.softmax(logits, dim=-1)
                            if request.top_k and request.top_k > 0:
                                k = min(request.top_k, probs.size(-1))
                                _, indices = torch.topk(probs, k)
                                mask = torch.zeros_like(probs)
                                mask.scatter_(1, indices, 1.0)
                                probs = probs * mask
                                probs = probs / probs.sum(dim=-1, keepdim=True)
                            cumsum = torch.cumsum(probs, dim=-1)
                            cumsum[..., -1:] = 1.0
                            mask = cumsum > top_p_val
                            probs[mask] = 0
                            probs = probs / probs.sum(dim=-1, keepdim=True)
                            next_tok = torch.multinomial(probs, num_samples=1)
                        else:
                            next_tok = logits.argmax(dim=-1, keepdim=True)

                        idx = torch.cat([idx, next_tok], dim=1)
                        generated.append(next_tok.item())

                        if next_tok.item() == tokenizer.eos_token_id:
                            break

                        if len(generated) >= batch_size:
                            yield generated
                            generated = []

                    if generated:
                        yield generated

            for token_ids in _generate_batch():
                token_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                if token_text:
                    yield f"data: {json.dumps({'token': token_text, 'done': False})}\n\n"

        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    return StreamingResponse(generate_stream_tokens(), media_type="text/event-stream")


@app.post("/chat", tags=["chat"])
async def chat_completion(request: ChatRequest, req: Request):
    """Non-streaming chat completion (messages → assistant text). Uses the same engine as ``/inference/generate``."""
    client_ip = req.client.host if req.client else "unknown"
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    prompt = _format_chat_messages_for_standard(request.messages)
    max_tokens = input_validator.validate_max_tokens(request.max_new_tokens or 100)
    temperature = input_validator.validate_temperature(
        request.temperature if request.temperature is not None else 0.8
    )
    top_p_val = max(0.0, min(1.0, float(request.top_p if request.top_p is not None else 0.9)))
    top_k = request.top_k if request.top_k is not None else 50
    top_k = max(1, min(500, int(top_k)))

    t0 = time.perf_counter()
    text = ""
    try:
        engine = get_inference_engine()
        if engine is None:
            audit_logger.log(
                "chat", client_ip, resource="/chat", action="no_model", status="success"
            )
            return {"error": "Model not loaded", "text": ""}

        text = engine.generate_single(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p_val,
            top_k=top_k,
            repetition_penalty=1.0,
        )
        text = _strip_assistant_prefix(text)
        audit_logger.log("chat", client_ip, resource="/chat", action="inference", status="success")
        return {
            "text": text,
            "model": "gpt2-engine",
            "tokens_generated": len(text.split()) if text else 0,
        }
    except SloughGPTDomainError:
        raise
    except Exception as e:
        return {"error": str(e), "text": ""}
    finally:
        record_inference_call(time.perf_counter() - t0, float(len(text.split()) if text else 0))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat completion using Server-Sent Events (same prompt formatting as ``POST /chat``)."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    prompt = _format_chat_messages_for_standard(request.messages)
    require_non_empty_prompt(prompt, field_name="messages")
    max_gen = input_validator.validate_max_tokens(request.max_new_tokens or 100)
    temperature = input_validator.validate_temperature(
        request.temperature if request.temperature is not None else 0.8
    )
    top_p_val = max(0.0, min(1.0, float(request.top_p if request.top_p is not None else 0.9)))
    top_k = request.top_k if request.top_k is not None else 50
    top_k = max(1, min(500, int(top_k)))

    engine = get_inference_engine()

    async def token_stream():
        if engine is None:
            yield f"data: {json.dumps({'error': 'Model not loaded', 'token': '', 'done': True})}\n\n"
            return
        prefix_stripped = False
        accumulated = ""
        async for token in engine.generate_stream(
            prompt=prompt,
            max_new_tokens=max_gen,
            temperature=temperature,
            top_p=top_p_val,
            top_k=top_k,
        ):
            if not prefix_stripped:
                accumulated += token
                if accumulated.startswith("Assistant:"):
                    accumulated = accumulated[len("Assistant:") :].lstrip()
                    prefix_stripped = True
                    if accumulated:
                        yield f"data: {json.dumps({'token': accumulated, 'done': False})}\n\n"
                elif accumulated.startswith("Assistant: "):
                    accumulated = accumulated[len("Assistant: ") :]
                    prefix_stripped = True
                    if accumulated:
                        yield f"data: {json.dumps({'token': accumulated, 'done': False})}\n\n"
            else:
                yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


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
                    hashlib.sha256(API_KEY.encode()).hexdigest(),
                ):
                    authenticated = True
            elif token:
                if jwt_auth.verify_token(token):
                    authenticated = True

            if not authenticated:
                await websocket.send_json({"status": "error", "error": "Authentication required"})
                await websocket.close(code=4001)
                audit_logger.log(
                    "ws_auth_failed",
                    client_ip,
                    resource="/ws/generate",
                    action="connect",
                    status="failure",
                )
                return

            await websocket.send_json({"status": "authenticated"})
            audit_logger.log(
                "ws_auth_success",
                client_ip,
                resource="/ws/generate",
                action="connect",
                status="success",
            )

        except json.JSONDecodeError:
            await websocket.send_json({"status": "error", "error": "Invalid JSON"})
            await websocket.close(code=4002)
            return

        manager.connect(websocket)

        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)

            try:
                prompt = require_non_empty_prompt(
                    input_validator.validate_prompt(request_data.get("prompt", ""))
                )
            except SloughGPTDomainError as e:
                await websocket.send_json({"status": "error", "error": str(e), "code": e.code})
                continue

            max_tokens = input_validator.validate_max_tokens(request_data.get("max_tokens", 100))
            temperature = input_validator.validate_temperature(request_data.get("temperature", 0.8))
            raw_top_p = request_data.get("top_p", 0.9)
            top_p_val = max(0.0, min(1.0, float(raw_top_p if raw_top_p is not None else 0.9)))
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
                inputs = _inputs_to_model_device(tokenizer(prompt, return_tensors="pt"), model)

                model.eval()
                generated = ""

                with torch.no_grad():
                    for _ in range(max_tokens):
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=1,
                            temperature=temperature,
                            top_p=top_p_val,
                            do_sample=True,
                            return_dict_in_generate=True,
                        )
                        token = tokenizer.decode(outputs.sequences[0][-1])
                        if token and not token.startswith(" "):
                            generated += token

                        await websocket.send_json(
                            {"token": token, "generated": generated, "done": False}
                        )

                        inputs = _inputs_to_model_device(
                            tokenizer(token, return_tensors="pt"), model
                        )

                await websocket.send_json({"status": "done", "text": generated, "done": True})

            else:
                demo_text = f"Demo response to: {prompt}"
                for char in demo_text:
                    await websocket.send_json({"token": char, "generated": char, "done": False})
                    await asyncio.sleep(0.05)
                await websocket.send_json({"status": "done", "text": demo_text, "done": True})

            audit_logger.log(
                "ws_generate",
                client_ip,
                resource="/ws/generate",
                action="generate",
                status="success",
            )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        audit_logger.log(
            "ws_disconnect",
            client_ip,
            resource="/ws/generate",
            action="disconnect",
            status="success",
        )
    except Exception as e:
        await websocket.send_json({"status": "error", "error": str(e)})
        manager.disconnect(websocket)
        audit_logger.log(
            "ws_error",
            client_ip,
            resource="/ws/generate",
            action="error",
            status="failure",
            details={"error": str(e)},
        )


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
async def list_datasets(q: Optional[str] = None, type: Optional[str] = None):
    """List available datasets with optional search and filter."""
    import os
    from pathlib import Path

    datasets_dir = Path("datasets")
    datasets = []

    if datasets_dir.exists():
        for d in datasets_dir.iterdir():
            if d.is_dir():
                input_file = d / "input.txt"
                corpus_file = d / "corpus.jsonl"
                size = 0
                num_samples = 0
                has_corpus = corpus_file.exists()

                if has_corpus:
                    size = corpus_file.stat().st_size
                    with open(corpus_file, "r", encoding="utf-8") as f:
                        num_samples = sum(1 for _ in f)
                elif input_file.exists():
                    size = input_file.stat().st_size

                dataset = {
                    "id": d.name,
                    "name": d.name.replace("_", " ").title(),
                    "path": str(d),
                    "size_bytes": size,
                    "size_formatted": f"{size / 1024:.1f} KB" if size > 0 else "Empty",
                    "type": "corpus" if has_corpus else "text",
                    "num_samples": num_samples,
                }

                # Apply filters
                if q:
                    q_lower = q.lower()
                    if q_lower not in d.name.lower() and q_lower not in dataset["name"].lower():
                        continue

                if type and dataset["type"] != type:
                    continue

                datasets.append(dataset)

    return {"datasets": datasets, "count": len(datasets)}


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
        stats.update(
            {
                "size_bytes": len(content),
                "num_lines": content.count("\n") + 1,
                "num_chars": len(content),
            }
        )

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


@app.get("/datasets/{dataset_id}/preview", tags=["datasets"])
async def preview_dataset(dataset_id: str, limit: int = 10):
    """Preview dataset content and stats."""
    from pathlib import Path
    import json

    dataset_path = Path(f"datasets/{dataset_id}")

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    corpus_file = dataset_path / "corpus.jsonl"
    input_file = dataset_path / "input.txt"

    samples = []
    total_chars = 0
    languages = {}

    if corpus_file.exists():
        with open(corpus_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                try:
                    record = json.loads(line)
                    samples.append(
                        {
                            "path": record.get("path", ""),
                            "language": record.get("language", "text"),
                            "content": record.get("content", "")[:500],
                            "size": record.get("size", 0),
                        }
                    )
                    total_chars += record.get("size", 0)
                    lang = record.get("language", "text")
                    languages[lang] = languages.get(lang, 0) + 1
                except json.JSONDecodeError:
                    continue
    elif input_file.exists():
        with open(input_file, "r", encoding="utf-8") as f:
            lines_content = f.readlines()[:limit]
            for i, line in enumerate(lines_content):
                samples.append(
                    {
                        "path": f"line_{i}",
                        "language": "text",
                        "content": line.strip()[:500],
                        "size": len(line),
                    }
                )
                total_chars += len(line)
    else:
        raise HTTPException(status_code=404, detail="No corpus or input file found")

    return {
        "dataset_id": dataset_id,
        "samples": samples,
        "total_samples": len(samples),
        "total_chars": total_chars,
        "languages": languages,
    }


@app.post("/datasets/{dataset_id}/validate", tags=["datasets"])
async def validate_dataset(dataset_id: str):
    """Validate a dataset for training compatibility."""
    from pathlib import Path
    import json

    dataset_path = Path(f"datasets/{dataset_id}")
    corpus_file = dataset_path / "corpus.jsonl"
    input_file = dataset_path / "input.txt"

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    validation = {
        "dataset_id": dataset_id,
        "valid": True,
        "issues": [],
        "warnings": [],
        "stats": {},
    }

    corpus_exists = corpus_file.exists()
    input_exists = input_file.exists()

    if not corpus_exists and not input_exists:
        validation["valid"] = False
        validation["issues"].append("No corpus.jsonl or input.txt found")
        return validation

    file_path = corpus_file if corpus_exists else input_file

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        non_empty_lines = [l for l in lines if l.strip()]

        stats = {
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty_lines),
            "total_chars": len(content),
            "avg_line_length": len(content) / len(lines) if lines else 0,
        }

        if corpus_exists:
            stats["num_records"] = sum(1 for _ in open(corpus_file))

            languages = {}
            for line in open(corpus_file):
                try:
                    record = json.loads(line)
                    lang = record.get("language", "unknown")
                    languages[lang] = languages.get(lang, 0) + 1
                except:
                    pass
            stats["languages"] = languages

        validation["stats"] = stats

        if len(non_empty_lines) < 10:
            validation["warnings"].append(
                f"Very small dataset: only {len(non_empty_lines)} non-empty lines"
            )

        if stats["avg_line_length"] < 5:
            validation["warnings"].append("Many very short lines detected")

    except Exception as e:
        validation["valid"] = False
        validation["issues"].append(f"Failed to read dataset: {str(e)}")

    return validation


@app.delete("/datasets/{dataset_id}", tags=["datasets"])
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    import shutil

    dataset_path = Path(f"datasets/{dataset_id}")

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        shutil.rmtree(dataset_path)
        return {"success": True, "message": f"Dataset '{dataset_id}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CombineDatasetsRequest(BaseModel):
    source_ids: List[str]
    name: str


@app.post("/datasets/combine", tags=["datasets"])
async def combine_datasets(request: CombineDatasetsRequest):
    """Combine multiple datasets into one."""
    import json

    if len(request.source_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 datasets required")

    output_path = Path(f"datasets/{request.name}")
    output_path.mkdir(parents=True, exist_ok=True)
    corpus_file = output_path / "corpus.jsonl"

    total_files = 0
    total_chars = 0

    with open(corpus_file, "w", encoding="utf-8") as out:
        for source_id in request.source_ids:
            source_path = Path(f"datasets/{source_id}")
            if not source_path.exists():
                raise HTTPException(status_code=404, detail=f"Dataset not found: {source_id}")

            corpus = source_path / "corpus.jsonl"
            if corpus.exists():
                with open(corpus, "r", encoding="utf-8") as f:
                    for line in f:
                        out.write(line)
                        total_files += 1
                        try:
                            record = json.loads(line)
                            total_chars += record.get("size", len(record.get("content", "")))
                        except:
                            pass
            else:
                input_file = source_path / "input.txt"
                if input_file.exists():
                    content = input_file.read_text()
                    record = {"content": content, "source": source_id}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_files += 1
                    total_chars += len(content)

    return {
        "success": True,
        "dataset_id": request.name,
        "message": f"Combined {len(request.source_ids)} datasets: {total_files} files, {total_chars} chars",
        "output_path": str(corpus_file),
    }


class GitHubImportRequest(BaseModel):
    url: str
    name: str
    extensions: Optional[List[str]] = None
    max_files: Optional[int] = None


class HuggingFaceImportRequest(BaseModel):
    dataset_id: str
    name: Optional[str] = None


class URLImportRequest(BaseModel):
    url: str
    name: str


class LocalImportRequest(BaseModel):
    path: str
    name: str
    extensions: Optional[List[str]] = None


@app.post("/datasets/import/github", tags=["datasets"])
async def import_from_github(request: GitHubImportRequest):
    """Import dataset from GitHub repository."""
    try:
        from domains.training.data_import import RepoImporter
        from pathlib import Path

        repo = RepoImporter()
        extensions = request.extensions or [".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml"]

        result = repo.import_from_github(
            url=request.url,
            dataset_name=request.name,
            extensions=extensions,
            max_files=request.max_files,
        )

        if result.success:
            return {
                "success": True,
                "dataset_id": result.name,
                "message": f"Imported {result.files_imported} files ({result.total_chars} chars)",
                "output_path": result.output_path,
            }
        else:
            raise HTTPException(status_code=400, detail=result.error or "Import failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/datasets/import/huggingface", tags=["datasets"])
async def import_from_huggingface(request: HuggingFaceImportRequest):
    """Import dataset from HuggingFace Hub."""
    try:
        from domains.training.data_import import HuggingFaceImporter

        hf = HuggingFaceImporter()
        name = request.name or request.dataset_id.split("/")[-1]

        result = hf.download_dataset(
            dataset_id=request.dataset_id,
            name=name,
        )

        if result.success:
            return {
                "success": True,
                "dataset_id": name,
                "message": f"Downloaded {result.files_imported} splits ({result.total_chars} chars)",
                "output_path": result.output_path,
            }
        else:
            raise HTTPException(status_code=400, detail=result.error or "Download failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/datasets/import/url", tags=["datasets"])
async def import_from_url(request: URLImportRequest):
    """Import dataset from URL."""
    try:
        from domains.training.data_import import URLImporter

        url_importer = URLImporter()
        result = url_importer.import_from_url(
            url=request.url,
            dataset_name=request.name,
        )

        if result.success:
            return {
                "success": True,
                "dataset_id": request.name,
                "message": f"Downloaded {result.total_chars} chars",
                "output_path": result.output_path,
            }
        else:
            raise HTTPException(status_code=400, detail=result.error or "Download failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/datasets/import/local", tags=["datasets"])
async def import_from_local(request: LocalImportRequest):
    """Import dataset from local file or directory."""
    try:
        from domains.training.data_import import DataImporter

        importer = DataImporter()
        extensions = request.extensions or [".py", ".js", ".ts", ".md", ".txt", ".json"]

        result = importer.import_from_local(
            path=request.path,
            name=request.name,
            extensions=extensions,
        )

        if result.success:
            return {
                "success": True,
                "dataset_id": request.name,
                "message": f"Imported {result.files_imported} files ({result.total_chars} chars)",
                "output_path": result.output_path,
            }
        else:
            raise HTTPException(status_code=400, detail=result.error or "Import failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class KaggleImportRequest(BaseModel):
    dataset: str  # e.g., "zillow/zecon"
    name: Optional[str] = None


@app.post("/datasets/import/kaggle", tags=["datasets"])
async def import_from_kaggle(request: KaggleImportRequest):
    """Import dataset from Kaggle."""
    try:
        import subprocess
        import zipfile
        import shutil

        name = request.name or request.dataset.replace("/", "_")
        output_dir = DATA_DIR / "datasets" / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Try using kaggle CLI if available
        try:
            result = subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    request.dataset,
                    "-p",
                    str(output_dir),
                    "--unzip",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Kaggle import failed: {result.stderr}. Make sure 'kaggle' CLI is installed and authenticated.",
                )
        except FileNotFoundError:
            raise HTTPException(
                status_code=400,
                detail="Kaggle CLI not found. Install with: pip install kaggle && mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/",
            )

        # Move files from subdirectory if needed
        temp_dir = output_dir / request.dataset.replace("/", "_")
        if temp_dir.exists():
            for item in temp_dir.iterdir():
                shutil.move(str(item), str(output_dir / item.name))
            temp_dir.rmdir()

        # Count files and estimate size
        file_count = sum(1 for _ in output_dir.rglob("*") if _.is_file())
        total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())

        return {
            "success": True,
            "dataset_id": name,
            "message": f"Downloaded {file_count} files ({total_size / 1024 / 1024:.1f} MB) from Kaggle",
            "output_path": str(output_dir),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CSVImportRequest(BaseModel):
    url: str  # URL to CSV file
    name: str
    delimiter: Optional[str] = ","
    encoding: Optional[str] = "utf-8"


@app.post("/datasets/import/csv", tags=["datasets"])
async def import_from_csv(request: CSVImportRequest):
    """Import dataset from CSV URL."""
    import csv
    import urllib.request

    try:
        name = request.name
        output_dir = DATA_DIR / "datasets" / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download CSV
        req = urllib.request.Request(request.url, headers={"User-Agent": "SloughGPT"})
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode(request.encoding or "utf-8")

        # Parse CSV
        lines = content.strip().split("\n")
        if not lines:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        dialect = csv.Sniffer().sniff(lines[0][:1000], delimiters=",;\t")
        reader = csv.reader(lines, dialect=dialect)
        headers = next(reader)
        rows = list(reader)

        # Save as JSONL (each row is a JSON object)
        jsonl_path = output_dir / f"{name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in rows:
                obj = {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
                f.write(json.dumps(obj) + "\n")

        # Save metadata
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source": request.url,
                    "columns": headers,
                    "rows": len(rows),
                    "delimiter": request.delimiter,
                },
                f,
                indent=2,
            )

        return {
            "success": True,
            "dataset_id": name,
            "message": f"Imported CSV with {len(rows)} rows, {len(headers)} columns",
            "output_path": str(output_dir),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets/search", tags=["datasets"])
async def search_datasets(q: str, limit: int = 20):
    """Search datasets by name or content."""
    from pathlib import Path
    import re

    datasets_dir = Path("datasets")
    results = []

    if not datasets_dir.exists():
        return {"results": [], "count": 0}

    q_lower = q.lower()
    pattern = re.compile(re.escape(q), re.IGNORECASE)

    for d in datasets_dir.iterdir():
        if not d.is_dir():
            continue

        # Match by name
        name_match = pattern.search(d.name)
        if name_match:
            results.append(
                {
                    "id": d.name,
                    "name": d.name.replace("_", " ").title(),
                    "match_type": "name",
                    "match_highlight": name_match.group(0),
                }
            )
            continue

        # Search in content
        for file_path in d.rglob("*.txt"):
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")[:10000]
                match = pattern.search(content)
                if match:
                    results.append(
                        {
                            "id": d.name,
                            "name": d.name.replace("_", " ").title(),
                            "match_type": "content",
                            "match_highlight": content[
                                max(0, match.start() - 30) : match.end() + 30
                            ],
                            "file": str(file_path.relative_to(d)),
                        }
                    )
                    break
            except Exception:
                continue

    return {"results": results[:limit], "count": len(results), "query": q}


@app.get("/datasets/search/github", tags=["datasets"])
async def search_github_repos(query: str, limit: int = 10):
    """Search GitHub repositories (uses basic API, no auth)."""
    import urllib.request
    import urllib.parse
    import json

    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.github.com/search/repositories?q={encoded_query}&sort=stars&order=desc&per_page={limit}"

        req = urllib.request.Request(url, headers={"User-Agent": "SloughGPT"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        repos = []
        for item in data.get("items", [])[:limit]:
            repos.append(
                {
                    "id": item.get("full_name"),
                    "name": item.get("name"),
                    "full_name": item.get("full_name"),
                    "description": item.get("description"),
                    "stars": item.get("stargazers_count", 0),
                    "url": item.get("html_url"),
                    "language": item.get("language"),
                }
            )

        return {"repos": repos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DatasetExportRequest(BaseModel):
    format: str = "json"
    include_metadata: bool = True


@app.post("/datasets/{dataset_id}/export", tags=["datasets"])
async def export_dataset(dataset_id: str, request: DatasetExportRequest):
    """Export a dataset to various formats."""
    import csv
    import io

    # Find the dataset
    dataset_dir = DATA_DIR / "datasets" / dataset_id
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    # Collect all text files
    samples = []
    for file_path in dataset_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in {
            ".txt",
            ".md",
            ".py",
            ".js",
            ".ts",
            ".json",
        }:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                samples.append(
                    {
                        "path": str(file_path.relative_to(dataset_dir)),
                        "content": content[:10000],  # Limit content size
                        "size": file_path.stat().st_size,
                    }
                )
            except Exception:
                continue

    if not samples:
        raise HTTPException(status_code=404, detail="No samples found in dataset")

    # Format output based on requested format
    if request.format == DatasetExportFormat.JSON:
        output = json.dumps(
            {
                "dataset_id": dataset_id,
                "total_samples": len(samples),
                "samples": samples if request.include_metadata else [s["content"] for s in samples],
            },
            indent=2,
        )

    elif request.format == DatasetExportFormat.JSONL:
        lines = []
        for sample in samples:
            if request.include_metadata:
                lines.append(json.dumps(sample))
            else:
                lines.append(json.dumps({"content": sample["content"]}))
        output = "\n".join(lines)

    elif request.format == DatasetExportFormat.CSV:
        output_buffer = io.StringIO()
        if request.include_metadata:
            writer = csv.DictWriter(output_buffer, fieldnames=["path", "content", "size"])
            writer.writeheader()
            writer.writerows(samples)
        else:
            writer = csv.writer(output_buffer)
            writer.writerow(["content"])
            for sample in samples:
                writer.writerow([sample["content"]])
        output = output_buffer.getvalue()

    else:  # TXT
        output = "\n\n---\n\n".join(s["content"] for s in samples)

    return {
        "dataset_id": dataset_id,
        "format": request.format,
        "total_samples": len(samples),
        "content": output,
        "size_bytes": len(output.encode("utf-8")),
    }


@app.get("/datasets/{dataset_id}/download", tags=["datasets"])
async def download_dataset(dataset_id: str, format: str = "json"):
    """Download a dataset as a file."""
    export_req = DatasetExportRequest(format=format)
    return await export_dataset(dataset_id, export_req)


@app.get("/models")
async def list_models():
    """List available models (local + HuggingFace)."""
    from pathlib import Path

    models = []

    models_dir = Path("models")
    if models_dir.exists():
        extensions = (".pt", ".pth", ".sou", ".safetensors", ".onnx")
        for pattern in ("*.pt", "*.pth", "*.sou", "*.safetensors", "*.onnx"):
            for m in models_dir.glob(pattern):
                if m.suffix.lower() not in extensions:
                    continue
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


def _load_hf_model_core(request: LoadModelRequest) -> Dict[str, Any]:
    """Load HuggingFace weights into process globals; shared by ``POST /models/load`` and startup autoload."""
    global model, tokenizer, model_type

    try:
        from domains.training.model_registry import load_hf_model

        mode = request.mode or "local"
        load_kwargs: Dict[str, Any] = {}
        if mode == "local":
            load_kwargs["device"] = request.device if request.device is not None else "auto"

        hf_client = load_hf_model(request.model_id, mode=mode, **load_kwargs)

        if mode == "local":
            loader = getattr(hf_client, "_client", None)
            if (
                loader is not None
                and getattr(loader, "model", None) is not None
                and getattr(loader, "tokenizer", None) is not None
            ):
                model = loader.model
                tokenizer = loader.tokenizer
                model_type = "gpt2"

                # Apply dynamic quantization for ~2x speedup on MPS
                if request.device == "mps" or request.device == "auto":
                    try:
                        logger.info("Applying dynamic quantization for faster inference...")
                        import torch

                        if hasattr(torch, "quantization") and hasattr(
                            torch.quantization, "quantize_dynamic"
                        ):
                            model = torch.quantization.quantize_dynamic(
                                model, {torch.nn.Linear}, dtype=torch.qint8
                            )
                            logger.info("Dynamic quantization applied successfully")
                    except Exception as e:
                        logger.warning(f"Quantization failed: {e}")

                # torch.compile can cause issues on MPS, skip for now
                # TODO: Enable when CUDA is available or MPS issues are resolved
            else:
                model_type = f"hf/{request.model_id}"
                logger.warning(
                    "HF local load did not expose model/tokenizer on loader; /generate may stay in demo mode"
                )
        else:
            model_type = f"hf/{request.model_id}"

        effective = None
        if mode == "local" and model is not None:
            effective = _inference_engine_device_str(model)

        return {
            "status": "loaded",
            "model": request.model_id,
            "mode": mode,
            "device": request.device,
            "effective_device": effective,
            "model_type": model_type,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/models/load")
async def load_hf_model_endpoint(request: LoadModelRequest):
    """Load a HuggingFace model."""
    return _load_hf_model_core(request)


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


_experiment_tracker = None


def get_experiment_tracker():
    """Get or create the experiment tracker."""
    global _experiment_tracker
    if _experiment_tracker is None:
        from domains.ml_infrastructure.experiment_tracker import ExperimentTracker

        _experiment_tracker = ExperimentTracker(storage_path="data/experiments")
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
        return exp.to_api_dict()
    return {"experiment_id": experiment_id, "name": name}


@app.get("/experiments", tags=["experiments"])
async def list_experiments():
    """List all experiments."""
    tracker = get_experiment_tracker()
    experiments = tracker.list_experiments()
    return [exp.to_api_dict() for exp in experiments]


@app.get("/experiments/{experiment_id}", tags=["experiments"])
async def get_experiment(experiment_id: str):
    """Get a specific experiment."""
    tracker = get_experiment_tracker()
    exp = tracker.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp.to_api_dict()


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

    exp.metrics[metric_name].append(MetricPoint(timestamp=time.time(), step=step, value=value))

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


def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port (DEPRECATED - use domains.shared.find_available_port)."""
    from domains.shared import find_available_port as _find_available_port

    return _find_available_port(host="", start_port=start_port, max_attempts=max_attempts)


# Global inference engine (lazy loaded)
_inference_engine = None


def get_inference_engine():
    """Get or create the inference engine using existing model."""
    global _inference_engine, model, tokenizer

    if model is None or tokenizer is None:
        return None

    if _inference_engine is not None and getattr(_inference_engine, "model", None) is not model:
        _inference_engine = None

    if _inference_engine is None:
        from domains.inference.engine import InferenceEngine

        device = _inference_engine_device_str(model)
        _inference_engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
    return _inference_engine


@app.post("/inference/generate", tags=["inference"])
async def inference_generate(gr: GenerateRequest, req: Request):
    """Generate text using the production inference engine."""
    client_ip = req.client.host if req.client else "unknown"

    # Validate input
    prompt = require_non_empty_prompt(input_validator.validate_prompt(gr.prompt))
    max_tokens = input_validator.validate_max_tokens(gr.max_new_tokens or 100)
    temperature = input_validator.validate_temperature(
        gr.temperature if gr.temperature is not None else 0.8
    )
    top_p_val = max(0.0, min(1.0, float(gr.top_p if gr.top_p is not None else 0.9)))
    top_k = gr.top_k if gr.top_k is not None else 50
    top_k = max(1, min(500, int(top_k)))

    t0 = time.perf_counter()
    text = ""
    try:
        engine = get_inference_engine()

        if engine is None:
            audit_logger.log(
                "generate",
                client_ip,
                resource="/inference/generate",
                action="no_model",
                status="success",
            )
            return {"error": "Model not loaded", "text": ""}

        text = engine.generate_single(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p_val,
            top_k=top_k,
            repetition_penalty=1.0,
        )

        audit_logger.log(
            "generate",
            client_ip,
            resource="/inference/generate",
            action="inference",
            status="success",
        )

        return {
            "text": text,
            "model": "gpt2-engine",
            "tokens_generated": len(text.split()),
        }
    except SloughGPTDomainError:
        raise
    except Exception as e:
        return {"error": str(e), "text": ""}
    finally:
        record_inference_call(time.perf_counter() - t0, float(len(text.split()) if text else 0))


@app.post("/inference/generate/stream", tags=["inference"])
async def inference_generate_stream(request: GenerateRequest):
    """Streaming generation using llama.cpp or PyTorch."""
    require_non_empty_prompt(input_validator.validate_prompt(request.prompt))

    # Check for llama.cpp first
    llama_model_path = os.environ.get("SLOUGHGPT_MODEL_PATH", "").strip()
    if llama_model_path:
        try:
            from domains.inference.llama_engine import (
                detect_gpu,
                auto_select_backend,
                LlamaInferenceConfig,
                LlamaInferenceEngine,
            )

            gpu = detect_gpu()
            n_gpu_layers = auto_select_backend(1.5)
            config = LlamaInferenceConfig(model_path=llama_model_path, n_gpu_layers=n_gpu_layers)
            engine = LlamaInferenceEngine(config)

            max_tokens = input_validator.validate_max_tokens(request.max_new_tokens or 100)
            start_time = time.perf_counter()

            async def llama_stream():
                try:
                    for token in engine.generate_stream(request.prompt, max_tokens=max_tokens):
                        yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                    elapsed = time.perf_counter() - start_time
                    yield f"data: {json.dumps({'token': '', 'done': True, 'elapsed': elapsed})}\n\n"
                except Exception as e:
                    logger.error(f"llama.cpp stream failed: {e}")
                    yield f"data: {json.dumps({'error': str(e), 'token': '', 'done': True})}\n\n"

            return StreamingResponse(llama_stream(), media_type="text/event-stream")
        except Exception as e:
            logger.error(f"llama.cpp failed: {e}")

    engine = get_inference_engine()
    max_tokens = input_validator.validate_max_tokens(request.max_new_tokens or 100)
    temperature = input_validator.validate_temperature(
        request.temperature if request.temperature is not None else 0.8
    )
    top_p_val = max(0.0, min(1.0, float(request.top_p if request.top_p is not None else 0.9)))
    top_k = request.top_k if request.top_k is not None else 50
    top_k = max(1, min(500, int(top_k)))

    async def token_stream():
        if engine is None:
            yield f"data: {json.dumps({'error': 'Model not loaded', 'token': '', 'done': True})}\n\n"
            return
        async for token in engine.generate_stream(
            prompt=request.prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p_val,
            top_k=top_k,
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
async def batch_generate(batch: BatchGenerateRequest, http_request: Request):
    """
    Batch text generation for multiple prompts.
    Optionally uses caching for identical prompts.
    """
    client_ip = http_request.client.host if http_request.client else "unknown"
    results: List[BatchGenerateItem] = []

    for prompt in batch.prompts[:50]:
        validated_prompt = require_non_empty_prompt(input_validator.validate_prompt(prompt))
        max_tokens = input_validator.validate_max_tokens(batch.max_new_tokens or 100)
        temp = input_validator.validate_temperature(batch.temperature or 0.8)
        top_p_val = max(0.0, min(1.0, float(batch.top_p if batch.top_p is not None else 0.9)))
        top_k = batch.top_k if batch.top_k is not None else 50
        top_k = max(1, min(500, int(top_k)))

        cache_key_str = cache_key(
            validated_prompt, max_tokens=max_tokens, temp=temp, top_p=top_p_val, top_k=top_k
        )

        if batch.use_cache:
            cached_result = cache.get(cache_key_str)
            if cached_result:
                results.append(BatchGenerateItem(prompt=prompt, text=cached_result, cached=True))
                continue

        if model is None:
            results.append(
                BatchGenerateItem(
                    prompt=prompt, text=f"Demo: {validated_prompt[:30]}...", error=None
                )
            )
            continue

        try:
            if model_type == "gpt2" and tokenizer:
                inputs = tokenizer(validated_prompt, return_tensors="pt")
                inputs = _inputs_to_model_device(inputs, model)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temp,
                        top_k=top_k,
                        top_p=top_p_val,
                        do_sample=True,
                    )
                text = tokenizer.decode(
                    outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
                )
                cache.set(cache_key_str, text)
                results.append(BatchGenerateItem(prompt=prompt, text=text))
            else:
                results.append(
                    BatchGenerateItem(prompt=prompt, text=f"Model not ready", error="model_error")
                )
        except Exception as e:
            results.append(BatchGenerateItem(prompt=prompt, text="", error=str(e)))

    audit_logger.log(
        "batch_generate",
        client_ip,
        resource="/inference/batch",
        action="batch",
        status="success",
        details={"count": len(batch.prompts)},
    )

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
    """Calculate perplexity on inline text using the **loaded** API model and tokenizer.

    This is not the same path as char-LM ``cli.py eval`` / ``lm_eval_char`` on a native
    ``step_*.pt`` file (``stoi`` / ``itos`` / ``chars``). For that parity story, see
    ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
    """
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
                result = benchmarker.benchmark_inference(
                    "Hello world", max_new_tokens=20, num_runs=2
                )
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
    """Export current model to file.

    Output formats are for deployment; char-LM ``cli.py eval`` parity uses native
    ``step_*.pt`` (``stoi`` / ``itos`` / ``chars``) — ``docs/policies/CONTRIBUTING.md``
    (*Checkpoint vocabulary*).
    """
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
_registry_metrics: Dict[str, Dict[str, Any]] = defaultdict(
    lambda: {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "total_tokens": 0,
        "total_latency_ms": 0,
        "avg_latency_ms": 0,
        "min_latency_ms": float("inf"),
        "max_latency_ms": 0,
    }
)


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


# ============ Vector Store ============
_vector_store = None
_vector_store_type = "in_memory"


class VectorStoreConfig(BaseModel):
    provider: Optional[str] = "in_memory"
    dimension: Optional[int] = 768
    api_key: Optional[str] = None
    url: Optional[str] = None
    index: Optional[str] = "sloughgpt"
    environment: Optional[str] = "us-east-1"


class UpsertRequest(BaseModel):
    texts: List[str]
    embeddings: Optional[List[List[float]]] = None
    metadata: Optional[List[Dict[str, Any]]] = None


class QueryRequest(BaseModel):
    query: str
    embedding: Optional[List[float]] = None
    top_k: Optional[int] = 5
    filter_metadata: Optional[Dict[str, Any]] = None


async def get_vector_store():
    global _vector_store
    if _vector_store is None:
        from domains.inference.vector_store import create_vector_store, simple_embed

        _vector_store = await create_vector_store(provider=_vector_store_type)
    return _vector_store


@app.post("/vector/init", tags=["vector"])
async def init_vector_store(config: VectorStoreConfig):
    """Initialize vector store with specified provider."""
    global _vector_store, _vector_store_type
    _vector_store_type = config.provider or "in_memory"

    try:
        from domains.inference.vector_store import create_vector_store, VectorStoreType

        kwargs = {"dimension": config.dimension or 768}

        if config.provider == "pinecone":
            kwargs["api_key"] = config.api_key
            kwargs["index"] = config.index
            kwargs["environment"] = config.environment
        elif config.provider == "weaviate":
            kwargs["url"] = config.url or "http://localhost:8080"
            kwargs["api_key"] = config.api_key
        elif config.provider == "chromadb":
            kwargs["persist_directory"] = "data/vector_store"

        _vector_store = await create_vector_store(provider=_vector_store_type, **kwargs)

        return {
            "status": "connected",
            "provider": _vector_store_type,
            "type": str(VectorStoreType(_vector_store_type)),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/vector/stats", tags=["vector"])
async def vector_stats():
    """Get vector store statistics."""
    store = await get_vector_store()
    return {
        "provider": _vector_store_type,
        "count": await store.count(),
    }


@app.post("/vector/upsert", tags=["vector"])
async def upsert_vectors(request: UpsertRequest):
    """Upsert documents with embeddings to vector store."""
    from domains.inference.vector_store import VectorEntry, simple_embed

    store = await get_vector_store()
    from domains.inference.vector_store import VectorStoreType

    entries = []
    for i, text in enumerate(request.texts):
        embedding = (
            request.embeddings[i]
            if request.embeddings and i < len(request.embeddings)
            else simple_embed(text)
        )
        metadata = request.metadata[i] if request.metadata and i < len(request.metadata) else {}

        import hashlib

        entry_id = hashlib.md5(f"{text[:50]}{i}".encode()).hexdigest()[:12]

        entries.append(
            VectorEntry(
                id=entry_id,
                vector=embedding,
                text=text,
                metadata=metadata,
            )
        )

    count = await store.upsert(entries)
    return {"status": "success", "upserted": count}


@app.post("/vector/query", tags=["vector"])
async def query_vectors(request: QueryRequest):
    """Query vector store for similar documents."""
    from domains.inference.vector_store import simple_embed

    store = await get_vector_store()

    embedding = request.embedding or simple_embed(request.query)

    results = await store.query(
        vector=embedding,
        top_k=request.top_k or 5,
        filter_metadata=request.filter_metadata,
    )

    return {
        "query": request.query,
        "results": [
            {
                "id": r.id,
                "score": r.score,
                "text": r.text,
                "metadata": r.metadata,
            }
            for r in results
        ],
    }


@app.delete("/vector/delete", tags=["vector"])
async def delete_vectors(ids: List[str]):
    """Delete vectors by ID."""
    store = await get_vector_store()
    success = await store.delete(ids)
    return {"status": "success" if success else "error", "deleted": len(ids)}


@app.post("/vector/search", tags=["vector"])
async def search_vectors(
    query: str,
    top_k: int = 5,
    include_embeddings: bool = False,
):
    """Combined RAG-style search: embed query, retrieve docs, optionally generate."""
    from domains.inference.vector_store import simple_embed

    store = await get_vector_store()

    embedding = simple_embed(query)
    results = await store.query(vector=embedding, top_k=top_k)

    context = "\n\n".join([r.text for r in results])

    return {
        "query": query,
        "context": context,
        "sources": [
            {"id": r.id, "score": r.score, "text": r.text[:200], "metadata": r.metadata}
            for r in results
        ],
    }


def _autoload_hf_model_at_startup() -> None:
    """
    Load default inference weights without a manual ``POST /models/load``.

    - ``SLOUGHGPT_AUTOLOAD_MODEL``: HuggingFace model id (default: ``gpt2``). Set to empty to skip.
    - ``SLOUGHGPT_AUTOLOAD_DEVICE``: passed through to the loader (default: ``auto``).
    """
    global model

    raw = os.environ.get("SLOUGHGPT_AUTOLOAD_MODEL", "gpt2").strip()
    if not raw:
        logger.info("SLOUGHGPT_AUTOLOAD_MODEL is empty; skipping startup autoload")
        return
    if model is not None:
        return
    device = (os.environ.get("SLOUGHGPT_AUTOLOAD_DEVICE") or "auto").strip() or "auto"
    req = LoadModelRequest(model_id=raw, mode="local", device=device)
    result = _load_hf_model_core(req)
    if result.get("status") == "error":
        logger.warning("Startup autoload failed for %s: %s", raw, result.get("error"))
    else:
        logger.info(
            "Startup autoload ok: model_id=%s effective_device=%s",
            raw,
            result.get("effective_device"),
        )


if __name__ == "__main__":
    import uvicorn

    raw_port = os.environ.get("SLOUGHGPT_API_PORT", "").strip()
    if raw_port:
        port = int(raw_port)
    else:
        port = find_available_port(8000)
    print(f"Starting SloughGPT server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
