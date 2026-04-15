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
import threading
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

from domains.errors import require_non_empty_prompt, SloughGPTDomainError
from domains.ops.wandb_server import record_inference_call

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sloughgpt")

_PROCESS_START_MONOTONIC = time.monotonic()


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Task:
    id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class TaskQueueManager:
    """
    Centralized task manager for operations with retries and event-based notifications.
    Prevents duplicate logging from multiple retries.
    """

    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._retry_delays = [1, 2, 4, 8, 16]

    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler."""
        self._event_handlers[event].append(handler)

    def off(self, event: str, handler: Callable) -> None:
        """Remove an event handler."""
        if event in self._event_handlers:
            self._event_handlers[event] = [h for h in self._event_handlers[event] if h != handler]

    def _emit(self, event: str, *args, **kwargs) -> None:
        """Emit an event to all handlers."""
        for handler in self._event_handlers.get(event, []):
            try:
                handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    async def submit(
        self, name: str, coro: Callable, max_retries: int = 3, task_id: Optional[str] = None
    ) -> str:
        """Submit and run a task with retries."""
        if task_id is None:
            task_id = f"{name}_{int(time.time() * 1000)}"

        task = Task(id=task_id, name=name, max_retries=max_retries)
        self._tasks[task_id] = task
        self._emit("task:submitted", task)

        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self._emit("task:started", task)

        await self._run_task(task, coro)
        return task_id

    async def _run_task(self, task: Task, coro_fn: Callable) -> None:
        """Run a task with retry logic."""
        last_error = None

        for attempt in range(task.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(coro_fn):
                    result = await coro_fn()
                else:
                    maybe_coro = coro_fn()
                    if asyncio.iscoroutine(maybe_coro):
                        result = await maybe_coro
                    else:
                        result = maybe_coro

                task.status = TaskStatus.SUCCESS
                task.result = result
                task.completed_at = datetime.now()
                self._emit("task:success", task)
                return

            except Exception as e:
                last_error = e
                task.retry_count = attempt + 1

                if attempt < task.max_retries:
                    task.status = TaskStatus.RETRYING
                    delay = self._retry_delays[min(attempt, len(self._retry_delays) - 1)]
                    logger.info(
                        f"Task {task.name} failed (attempt {attempt + 1}/{task.max_retries + 1}): {e}. Retrying in {delay}s..."
                    )
                    self._emit("task:retry", task, attempt, delay)
                    await asyncio.sleep(delay)
                else:
                    task.status = TaskStatus.FAILED
                    task.error = str(last_error)
                    task.completed_at = datetime.now()
                    logger.error(
                        f"Task {task.name} failed permanently after {attempt + 1} attempts: {e}"
                    )
                    self._emit("task:failed", task, last_error)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """Get all tasks, optionally filtered by status."""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    async def wait_for_completion(self, task_id: str, timeout: float = 120.0) -> Optional[Task]:
        """Wait for a task to complete (success or failure)."""
        start = time.time()
        while time.time() - start < timeout:
            task = self._tasks.get(task_id)
            if task and task.status in (TaskStatus.SUCCESS, TaskStatus.FAILED):
                return task
            await asyncio.sleep(0.5)
        return self._tasks.get(task_id)


# Global task queue manager instance
task_queue = TaskQueueManager()


def _on_task_submitted(task):
    logger.info(f"Task queued: {task.name} (id={task.id})")


def _on_task_started(task):
    logger.info(f"Task started: {task.name}")


def _on_task_retry(task, attempt, delay):
    logger.warning(f"Task retry: {task.name} (attempt {attempt + 1}, waiting {delay}s)")


def _on_task_success(task):
    logger.info(f"Task completed: {task.name}")


def _on_task_failed(task, error):
    logger.error(f"Task failed: {task.name} - {error}")


task_queue.on("task:submitted", _on_task_submitted)
task_queue.on("task:started", _on_task_started)
task_queue.on("task:retry", _on_task_retry)
task_queue.on("task:success", _on_task_success)
task_queue.on("task:failed", _on_task_failed)


# ============ Rate Limiting ============
class RateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list[float]] = {}

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []

        self.requests[client_id] = [
            t for t in self.requests[client_id] if now - t < self.window_seconds
        ]

        if len(self.requests[client_id]) >= self.max_requests:
            return False

        self.requests[client_id].append(now)
        return True

    def get_retry_after(self, client_id: str) -> int:
        if client_id not in self.requests or not self.requests[client_id]:
            return 0
        now = time.time()
        oldest = min(self.requests[client_id])
        return max(0, int(self.window_seconds - (now - oldest)))


rate_limiter = RateLimiter(max_requests=30, window_seconds=60)


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
            formatted += f"Assistant: {content}\n\n"
    formatted += "Assistant:"
    return formatted


def _strip_assistant_prefix(text: str) -> str:
    """Strip 'Assistant:' prefix from generated text."""
    import re

    prefixes = [
        r"^Assistant:\s*",
        r"^\n?Assistant:\s*",
        r"^\s*Assistant:\s*",
        r"^\s*>\s*Assistant:\s*",
    ]
    for prefix in prefixes:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)
    return text.strip()


def _clean_generated_text(text: str) -> str:
    """Remove repetition and unwanted content from generated text."""
    if not text:
        return text

    # Remove common prefixes
    text = re.sub(r"User:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Assistant:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]+\]", "", text)

    # Remove > artifacts (common in chat responses)
    text = re.sub(r"^\s*>\s*", "", text, flags=re.MULTILINE)

    lines = text.split("\n")
    seen_words = set()
    final_lines = []
    seen_line_prefixes = set()

    for line in lines:
        line_clean = line.strip()
        line_lower = line_clean.lower()

        if not line_clean or len(line_clean) < 2:
            continue

        prefix = line_lower[:15]
        if prefix in seen_line_prefixes:
            continue
        seen_line_prefixes.add(prefix)

        words = line_lower.split()
        unique_words = [w for w in words if w not in seen_words][:15]
        seen_words.update(unique_words)

        if unique_words:
            final_lines.append(line_clean)

    result = " ".join(final_lines)

    word_chunks = result.split()
    unique_chunks = []
    seen_phrases = set()

    for i in range(len(word_chunks)):
        chunk = " ".join(word_chunks[i : min(i + 8, len(word_chunks))])
        if chunk not in seen_phrases and len(chunk) > 3:
            seen_phrases.add(chunk)
            unique_chunks.append(word_chunks[i])

    return " ".join(unique_chunks).strip()


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


# ============ Message Feedback System ============
class MessageFeedback:
    """Stores feedback for messages (thumbs up/down)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._feedback: Dict[str, Dict[str, Any]] = {}
        self._regenerations: Dict[str, Dict[str, Any]] = {}
        self._session_contexts: Dict[str, List[ChatMessage]] = {}

    def record_feedback(
        self,
        message_id: str,
        rating: str,  # "thumbs_up" or "thumbs_down"
        session_id: Optional[str] = None,
        message_content: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record feedback for a message."""
        with self._lock:
            timestamp = datetime.utcnow().isoformat()
            feedback_entry = {
                "message_id": message_id,
                "rating": rating,
                "timestamp": timestamp,
                "session_id": session_id,
            }
            self._feedback[message_id] = feedback_entry

            if context:
                self._feedback[message_id]["context"] = (
                    context[:1000] if len(context) > 1000 else context
                )

            logger.info(f"Feedback recorded: {rating} for message {message_id}")
            return feedback_entry

    def get_feedback(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get feedback for a message."""
        return self._feedback.get(message_id)

    def store_session_context(self, session_id: str, messages: List[ChatMessage]) -> None:
        """Store conversation context for regeneration."""
        with self._lock:
            self._session_contexts[session_id] = list(messages)

    def get_session_context(self, session_id: str) -> Optional[List[ChatMessage]]:
        """Get stored conversation context."""
        with self._lock:
            return self._session_contexts.get(session_id)

    def clear_session_context(self, session_id: str) -> None:
        """Clear stored context for a session."""
        with self._lock:
            if session_id in self._session_contexts:
                del self._session_contexts[session_id]

    def record_regeneration(
        self,
        original_message_id: str,
        new_message_id: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a regeneration event."""
        with self._lock:
            regen_entry = {
                "original_message_id": original_message_id,
                "new_message_id": new_message_id,
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id,
            }
            self._regenerations[original_message_id] = regen_entry
            return regen_entry

    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        with self._lock:
            thumbs_up = sum(1 for f in self._feedback.values() if f.get("rating") == "thumbs_up")
            thumbs_down = sum(
                1 for f in self._feedback.values() if f.get("rating") == "thumbs_down"
            )
            return {
                "total_feedback": len(self._feedback),
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "total_regenerations": len(self._regenerations),
                "active_sessions": len(self._session_contexts),
            }


message_feedback = MessageFeedback()


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
    # Shutdown: mark running jobs as interrupted for recovery
    try:
        from training.job_store import get_job_store

        store = get_job_store()
        running_jobs = store.list(status="running")
        for job in running_jobs:
            logger.info(f"Marking job {job['id']} as interrupted on shutdown")
            store.mark_crashed(job["id"])
    except Exception as e:
        logger.warning("Failed to mark running jobs as interrupted: %s", e)

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
    user_id: Optional[str] = "default"


# ============ Feedback Models ============
class FeedbackRequest(BaseModel):
    message_id: str
    rating: str  # "thumbs_up" or "thumbs_down"
    session_id: Optional[str] = None
    message_content: Optional[str] = None
    context: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    message_id: str
    rating: str
    timestamp: str


class RegenerateRequest(BaseModel):
    session_id: str
    last_message_index: Optional[int] = None


class RegenerateResponse(BaseModel):
    status: str
    original_message_id: str
    new_message_id: str
    text: str
    model: str


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


# ===== Soul Manager - Hot-Swappable Souls =====


@app.get("/souls", tags=["souls"])
async def list_souls():
    """List all available souls."""
    try:
        from domains.inference.soul_manager import get_soul_manager

        manager = get_soul_manager()
        souls = manager.list_souls()
        return {
            "souls": [
                {
                    "name": s.name,
                    "path": s.path,
                    "description": s.description,
                    "personality": s.personality,
                    "traits": s.traits,
                }
                for s in souls
            ],
            "current_soul": manager.get_current_soul().name if manager.get_current_soul() else None,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/souls/switch", tags=["souls"])
async def switch_soul(name: str):
    """Switch to a different soul/personality."""
    global current_soul, soul_engine

    try:
        from domains.inference.soul_manager import get_soul_manager

        manager = get_soul_manager()
        result = manager.switch_soul(name)

        if result.get("success"):
            # Load the soul into engine if available
            soul_info = manager.get_soul(name)
            if soul_info and soul_info.path:
                from domains.core import SoulEngine

                engine = SoulEngine(device="cpu")
                soul = engine.load_soul(soul_info.path)
                current_soul = soul
                soul_engine = engine

            return result
        else:
            return result

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/souls/current", tags=["souls"])
async def get_current_soul():
    """Get currently active soul."""
    try:
        from domains.inference.soul_manager import get_soul_manager

        manager = get_soul_manager()
        current = manager.get_current_soul()

        if current:
            return {
                "name": current.name,
                "path": current.path,
                "description": current.description,
                "personality": current.personality,
                "traits": current.traits,
            }
        return {"name": None}
    except Exception as e:
        return {"error": str(e)}


@app.get("/souls/stats", tags=["souls"])
async def get_soul_stats():
    """Get soul manager statistics."""
    try:
        from domains.inference.soul_manager import get_soul_manager

        return get_soul_manager().get_stats()
    except Exception as e:
        return {"error": str(e)}


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
async def chat_stream(request: ChatRequest, req: Request):
    """Streaming chat completion using Server-Sent Events with meta-weight adjustment."""
    client_ip = req.client.host if req.client else "unknown"

    if not rate_limiter.is_allowed(client_ip):
        retry_after = rate_limiter.get_retry_after(client_ip)
        return Response(
            content=json.dumps({"error": "Rate limit exceeded. Please wait."}),
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            media_type="application/json",
        )

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

    # Get last user message for meta-weight lookup
    last_user_msg = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            last_user_msg = msg.content
            break

    # Try to get meta-weight adjustments
    meta_temperate = temperature
    meta_rep_penalty = 1.1
    meta_k = top_k
    meta_p = top_p_val

    try:
        manager = get_meta_weight_manager()
        if manager is not None:
            meta_weights = manager.get_adjustment(
                last_user_msg, k=3, user_id=request.user_id or "default"
            )
            meta_temperate = meta_weights.temperature
            meta_rep_penalty = meta_weights.repetition_penalty
            meta_k = meta_weights.top_k
            meta_p = meta_weights.top_p
    except Exception:
        pass  # Use defaults if meta-weight system unavailable

    # Check if user has a LoRA adapter for extra personalization
    user_lora_adapter = None
    try:
        from domains.feedback import get_per_user_lora

        user_lora = get_per_user_lora()
        adapter = user_lora.get_adapter(request.user_id or "default")
        if adapter:
            user_lora_adapter = adapter
    except Exception:
        pass

    engine = get_inference_engine()

    # Apply per-user LoRA adapter if available
    if engine and user_lora_adapter:
        try:
            engine.set_lora_adapter(user_lora_adapter)
        except Exception:
            pass

    async def token_stream():
        if engine is None:
            yield f"data: {json.dumps({'error': 'Model not loaded', 'token': '', 'done': True})}\n\n"
            return

        try:
            loop = asyncio.get_event_loop()
            generated_text = await loop.run_in_executor(
                None,
                lambda: engine.generate_single(
                    prompt=prompt,
                    max_new_tokens=max_gen,
                    temperature=meta_temperate,
                    top_p=meta_p,
                    top_k=meta_k,
                    repetition_penalty=meta_rep_penalty,
                ),
            )
            generated_text = _clean_generated_text(generated_text)
            generated_text = _strip_assistant_prefix(generated_text)

            words = generated_text.split()
            for word in words:
                yield f"data: {json.dumps({'token': word + ' ', 'done': False})}\n\n"
                await asyncio.sleep(0.01)
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'token': '', 'done': True})}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


# ============ Message Feedback Endpoints ============
@app.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
async def submit_feedback(request: FeedbackRequest, req: Request):
    """
    Submit feedback for a message (thumbs up or thumbs down).

    This stores user feedback for quality tracking and model improvement.
    """
    client_ip = req.client.host if req.client else "unknown"

    if request.rating not in ("thumbs_up", "thumbs_down"):
        raise HTTPException(status_code=400, detail="rating must be 'thumbs_up' or 'thumbs_down'")

    feedback = message_feedback.record_feedback(
        message_id=request.message_id,
        rating=request.rating,
        session_id=request.session_id,
        message_content=request.message_content,
        context=request.context,
    )

    audit_logger.log(
        "feedback",
        client_ip,
        resource="/feedback",
        action=request.rating,
        status="success",
        details={"message_id": request.message_id},
    )

    return FeedbackResponse(
        status="recorded",
        message_id=feedback["message_id"],
        rating=feedback["rating"],
        timestamp=feedback["timestamp"],
    )


@app.get("/feedback/{message_id}", tags=["feedback"])
async def get_feedback(message_id: str, req: Request):
    """Get feedback for a specific message."""
    feedback = message_feedback.get_feedback(message_id)
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found for this message")
    return {"message_id": message_id, "feedback": feedback}


@app.get("/feedback/stats/summary", tags=["feedback"])
async def get_feedback_stats():
    """Get aggregated feedback statistics."""
    return message_feedback.get_stats()


# ============ Session Context Storage ============
@app.post("/session/{session_id}/context", tags=["session"])
async def store_session_context(session_id: str, req: Request):
    """
    Store conversation context for a session.
    Used for regeneration functionality.
    """
    try:
        body = await req.json()
        messages = body.get("messages", [])
        validated_messages = [ChatMessage(**msg) for msg in messages]
        message_feedback.store_session_context(session_id, validated_messages)
        return {"status": "stored", "session_id": session_id, "message_count": len(messages)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/session/{session_id}/regenerate", response_model=RegenerateResponse, tags=["session"])
async def regenerate_response(session_id: str, req: Request):
    """
    Regenerate the last assistant response using stored context.

    This endpoint retrieves the stored conversation context and generates
    a new response for the last user message.
    """
    client_ip = req.client.host if req.client else "unknown"

    context_messages = message_feedback.get_session_context(session_id)
    if not context_messages:
        raise HTTPException(
            status_code=404, detail="No session context found. Store context before regenerating."
        )

    last_user_idx = -1
    for i in range(len(context_messages) - 1, -1, -1):
        if context_messages[i].role == "user":
            last_user_idx = i
            break

    if last_user_idx == -1:
        raise HTTPException(status_code=400, detail="No user message found in context")

    prompt_messages = context_messages[: last_user_idx + 1]
    prompt = _format_chat_messages_for_standard(prompt_messages)

    engine = get_inference_engine()
    if engine is None:
        return RegenerateResponse(
            status="error",
            original_message_id=f"{session_id}-last",
            new_message_id=f"{session_id}-regen-{uuid.uuid4().hex[:8]}",
            text="Model not loaded",
            model="none",
        )

    try:
        loop = asyncio.get_event_loop()
        generated_text = await loop.run_in_executor(
            None,
            lambda: engine.generate_single(
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
            ),
        )
        generated_text = _clean_generated_text(generated_text)
        generated_text = _strip_assistant_prefix(generated_text)

        original_msg_id = f"{session_id}-msg-{last_user_idx}"
        new_msg_id = f"{session_id}-regen-{uuid.uuid4().hex[:8]}"

        message_feedback.record_regeneration(
            original_message_id=original_msg_id,
            new_message_id=new_msg_id,
            session_id=session_id,
        )

        audit_logger.log(
            "regenerate",
            client_ip,
            resource=f"/session/{session_id}/regenerate",
            action="success",
            status="success",
        )

        return RegenerateResponse(
            status="success",
            original_message_id=original_msg_id,
            new_message_id=new_msg_id,
            text=generated_text,
            model=model_type,
        )

    except Exception as e:
        audit_logger.log(
            "regenerate",
            client_ip,
            resource=f"/session/{session_id}/regenerate",
            action="error",
            status="failure",
            details={"error": str(e)},
        )
        raise HTTPException(status_code=500, detail=str(e))


# ============ Meta-Weight Learning Endpoints ============

_meta_weight_manager = None


def get_meta_weight_manager():
    """Lazy load meta-weight manager to avoid import issues."""
    global _meta_weight_manager
    if _meta_weight_manager is None:
        try:
            from domains.feedback import MetaWeightManager, get_meta_weight_manager as _get_manager

            _meta_weight_manager = _get_manager()
        except ImportError:
            return None
    return _meta_weight_manager


class RecordFeedbackRequest(BaseModel):
    user_message: str
    assistant_response: str
    rating: str  # "thumbs_up" or "thumbs_down"
    conversation_id: Optional[str] = None
    quality_score: Optional[float] = None
    user_id: Optional[str] = "default"


class GetMetaWeightsRequest(BaseModel):
    user_message: str
    k: Optional[int] = 5
    user_id: Optional[str] = "default"


class MetaWeightResponse(BaseModel):
    temperature: float
    repetition_penalty: float
    top_p: float
    top_k: int
    based_on_samples: int


@app.post("/meta-weights/get", response_model=MetaWeightResponse, tags=["meta-weights"])
async def get_meta_weights(request: GetMetaWeightsRequest, req: Request):
    """
    Get meta-weight adjustments based on similar past feedback.

    Retrieves k most similar messages with positive feedback and
    calculates generation parameter adjustments.
    """
    manager = get_meta_weight_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Meta-weight system not available")

    weights = manager.get_adjustment(
        user_message=request.user_message, k=request.k or 5, user_id=request.user_id or "default"
    )

    return MetaWeightResponse(
        temperature=weights.temperature,
        repetition_penalty=weights.repetition_penalty,
        top_p=weights.top_p,
        top_k=weights.top_k,
        based_on_samples=len(manager._weight_history),
    )


@app.post("/feedback/record", tags=["meta-weights"])
async def record_feedback_with_context(request: RecordFeedbackRequest, req: Request):
    """
    Record feedback with full conversation context for training.

    Stores in SQLite for vector search and training data export.
    """
    manager = get_meta_weight_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Meta-weight system not available")

    if request.rating not in ("thumbs_up", "thumbs_down"):
        raise HTTPException(status_code=400, detail="rating must be 'thumbs_up' or 'thumbs_down'")

    feedback_id = manager.record_feedback(
        user_message=request.user_message,
        assistant_response=request.assistant_response,
        rating=request.rating,
        conversation_id=request.conversation_id,
        quality_score=request.quality_score,
        user_id=request.user_id or "default",
    )

    # Trigger online LoRA update (runs in background)
    try:
        from domains.feedback import get_online_lora_updater

        lora_updater = get_online_lora_updater()
        lora_updater.add_feedback(
            prompt=request.user_message,
            response=request.assistant_response,
            rating=request.rating,
            quality_score=request.quality_score,
        )
    except Exception:
        pass  # Don't fail if online training unavailable

    # Update per-user LoRA adapter
    try:
        from domains.feedback import get_per_user_lora

        user_lora = get_per_user_lora()
        feedback_signal = 1.0 if request.rating == "thumbs_up" else -1.0
        user_lora.update_adapter(
            user_id=request.user_id or "default",
            feedback_signal=feedback_signal,
            learning_rate=0.01,
        )
    except Exception:
        pass  # Don't fail if per-user LoRA unavailable

    # Get updated stats
    stats = manager.get_stats()

    return {
        "status": "recorded",
        "feedback_id": feedback_id,
        "stats": stats,
    }


@app.get("/meta-weights/stats", tags=["meta-weights"])
async def get_meta_weight_stats(req: Request):
    """Get meta-weight system statistics."""
    manager = get_meta_weight_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Meta-weight system not available")

    return manager.get_stats()


class ExportFeedbackRequest(BaseModel):
    filepath: str = "data/training_feedback.jsonl"
    format: str = "jsonl"  # "jsonl" or "dpo"


@app.post("/feedback/export", tags=["meta-weights"])
async def export_feedback_data(request: ExportFeedbackRequest, req: Request):
    """Export feedback data for training."""
    manager = get_meta_weight_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Meta-weight system not available")

    try:
        manager.export_training_data(filepath=request.filepath, format=request.format)
        return {
            "status": "exported",
            "filepath": request.filepath,
            "format": request.format,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/conversations", tags=["meta-weights"])
async def list_conversations(req: Request):
    """List all conversations."""
    manager = get_meta_weight_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Meta-weight system not available")

    conversations = manager.db.list_conversations(limit=100)
    return {"conversations": conversations}


@app.get("/feedback/conversations/{conversation_id}/messages", tags=["meta-weights"])
async def get_conversation_messages(conversation_id: str, req: Request):
    """Get all messages in a conversation."""
    manager = get_meta_weight_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="Meta-weight system not available")

    messages = manager.db.get_messages(conversation_id)
    return {"conversation_id": conversation_id, "messages": messages}


@app.get("/feedback-stats/training", tags=["meta-weights"])
async def get_feedback_training_stats(req: Request):
    """Get training data statistics from feedback."""
    try:
        from domains.feedback.training import FeedbackTrainer

        trainer = FeedbackTrainer()
        stats = trainer.get_training_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ExportTrainingRequest(BaseModel):
    format: str = "dpo"
    filepath: Optional[str] = None


@app.post("/feedback/export-training", tags=["meta-weights"])
async def export_feedback_training(request: ExportTrainingRequest, req: Request):
    """Export training data in DPO, SFT, or reward format."""
    if request.format not in ("dpo", "sft", "reward"):
        raise HTTPException(status_code=400, detail="format must be 'dpo', 'sft', or 'reward'")

    try:
        from domains.feedback.training import FeedbackTrainer
        from pathlib import Path

        trainer = FeedbackTrainer()

        if request.filepath:
            filepath = request.filepath
        else:
            timestamp = int(time.time())
            filepath = f"data/training_exports/{request.format}_{timestamp}.jsonl"

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        if request.format == "dpo":
            count = trainer.export_dpo(filepath)
        elif request.format == "sft":
            count = trainer.export_sft(filepath)
        else:
            trainer.export_for_alignment(output_dir=str(Path(filepath).parent), formats=["reward"])
            count = trainer.get_training_stats()["total_responses"]

        return {
            "status": "exported",
            "format": request.format,
            "filepath": filepath,
            "count": count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/online-training/stats", tags=["online-training"])
async def get_online_training_stats(req: Request):
    """Get online LoRA training statistics."""
    try:
        from domains.feedback import get_online_lora_updater

        updater = get_online_lora_updater()
        return updater.get_stats()
    except ImportError:
        raise HTTPException(status_code=503, detail="Online training not available")


@app.post("/online-training/reset", tags=["online-training"])
async def reset_online_training(req: Request):
    """Reset online LoRA weights to initial state."""
    try:
        from domains.feedback import get_online_lora_updater

        updater = get_online_lora_updater()
        updater.reset()
        return {"status": "reset", "message": "Online training weights reset"}
    except ImportError:
        raise HTTPException(status_code=503, detail="Online training not available")


# ============ Per-User LoRA Endpoints ============


@app.get("/user-adapters", tags=["user-adapters"])
async def list_user_adapters(req: Request):
    """List all user adapters and their stats."""
    try:
        from domains.feedback import get_per_user_lora

        store = get_per_user_lora()
        adapters = store.get_all_adapters()
        stats = store.get_stats()
        return {
            "adapters": adapters,
            "stats": stats,
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Per-user LoRA not available")


@app.get("/user-adapters/{user_id}", tags=["user-adapters"])
async def get_user_adapter(user_id: str, req: Request):
    """Get a specific user's adapter info."""
    try:
        from domains.feedback import get_per_user_lora

        store = get_per_user_lora()
        adapter = store.get_adapter(user_id)
        if adapter is None:
            return {"user_id": user_id, "exists": False}
        return {
            "user_id": adapter.user_id,
            "exists": True,
            "feedback_count": adapter.feedback_count,
            "created_at": adapter.created_at,
            "updated_at": adapter.updated_at,
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Per-user LoRA not available")


@app.post("/user-adapters/{user_id}/update", tags=["user-adapters"])
async def update_user_adapter(user_id: str, req: Request):
    """Update a user's LoRA adapter based on feedback."""
    try:
        from domains.feedback import get_per_user_lora

        store = get_per_user_lora()

        # Get feedback signal from request
        body = await req.json()
        rating = body.get("rating", "thumbs_up")
        feedback_signal = 1.0 if rating == "thumbs_up" else -1.0
        learning_rate = body.get("learning_rate", 0.01)

        adapter = store.update_adapter(
            user_id=user_id,
            feedback_signal=feedback_signal,
            learning_rate=learning_rate,
        )

        return {
            "status": "updated",
            "user_id": user_id,
            "feedback_count": adapter.feedback_count,
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Per-user LoRA not available")


@app.delete("/user-adapters/{user_id}", tags=["user-adapters"])
async def delete_user_adapter(user_id: str, req: Request):
    """Delete a user's LoRA adapter."""
    try:
        from domains.feedback import get_per_user_lora

        store = get_per_user_lora()
        store.delete_adapter(user_id)
        return {"status": "deleted", "user_id": user_id}
    except ImportError:
        raise HTTPException(status_code=503, detail="Per-user LoRA not available")


@app.post("/user-adapters/merge", tags=["user-adapters"])
async def merge_user_adapters(req: Request):
    """Merge multiple user adapters into aggregated weights."""
    try:
        from domains.feedback import get_per_user_lora

        store = get_per_user_lora()

        body = await req.json()
        user_ids = body.get("user_ids", [])

        merged = store.merge_adapters(user_ids)

        return {
            "status": "merged",
            "user_count": merged["user_count"],
            "adapter_rank": store.adapter_rank,
            "model_dim": store.model_dim,
            "note": "Merged weights returned (not applied to model)",
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Per-user LoRA not available")


class AggregateBestRequest(BaseModel):
    top_k: int = 10
    min_feedback_count: int = 5
    output_name: str = "best_aggregated"


@app.post("/user-adapters/aggregate-best", tags=["user-adapters"])
async def aggregate_best_adapters(request: AggregateBestRequest, req: Request):
    """
    Aggregate top-k best-performing user adapters into a single merged adapter.

    Useful for:
    - Improving base model with learned patterns
    - Creating organization-wide adapters
    - Batch consolidation of good feedback
    """
    try:
        from domains.feedback import get_per_user_lora

        store = get_per_user_lora()
        result = store.aggregate_best_adapters(
            top_k=request.top_k,
            min_feedback_count=request.min_feedback_count,
            output_name=request.output_name,
        )

        if "error" in result:
            return {"status": "skipped", **result}

        return {
            "status": "aggregated",
            "output_path": result["output_path"],
            "user_count": result["user_count"],
            "total_feedback": result["total_feedback"],
            "source_users": result["source_users"][:5],
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Per-user LoRA not available")


class PruneAdaptersRequest(BaseModel):
    min_feedback_count: int = 1
    max_age_days: int = 30


@app.post("/user-adapters/prune", tags=["user-adapters"])
async def prune_low_quality_adapters(request: PruneAdaptersRequest, req: Request):
    """
    Remove adapters that haven't been updated or have too few feedback.

    This helps keep the adapter store clean and reduces storage.
    """
    try:
        from domains.feedback import get_per_user_lora

        store = get_per_user_lora()
        deleted = store.prune_low_quality(
            min_feedback_count=request.min_feedback_count,
            max_age_days=request.max_age_days,
        )

        return {
            "status": "pruned",
            "deleted_count": len(deleted),
            "deleted_users": deleted,
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Per-user LoRA not available")


@app.get("/user-adapters/quality", tags=["user-adapters"])
async def get_quality_adapters(req: Request):
    """
    Get adapters filtered by quality metrics.

    Shows only adapters with meaningful feedback history.
    """
    min_count = int(req.query_params.get("min_feedback_count", 3))
    max_age = int(req.query_params.get("max_age_days", 0)) or None

    try:
        from domains.feedback import get_per_user_lora

        store = get_per_user_lora()
        adapters = store.get_quality_adapters(
            min_feedback_count=min_count,
            max_age_days=max_age,
        )

        return {
            "count": len(adapters),
            "adapters": adapters,
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Per-user LoRA not available")


@app.post("/user-adapters/{user_id}/reset", tags=["user-adapters"])
async def reset_user_adapter(user_id: str, req: Request):
    """Reset a user's adapter to initial state."""
    try:
        from domains.feedback import get_per_user_lora

        store = get_per_user_lora()
        adapter = store.reset_user_adapter(user_id)

        return {
            "status": "reset",
            "user_id": user_id,
            "feedback_count": adapter.feedback_count,
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Per-user LoRA not available")


# Feedback Workflow Endpoints
@app.get("/workflow/status", tags=["workflow"])
async def get_workflow_status(req: Request):
    """Get the current status of the automated feedback workflow."""
    try:
        from domains.feedback import get_feedback_workflow

        workflow = get_feedback_workflow()
        return workflow.get_status()
    except ImportError:
        raise HTTPException(status_code=503, detail="Workflow not available")


class WorkflowStartRequest(BaseModel):
    aggregate_interval_minutes: int = 60
    prune_interval_minutes: int = 120
    export_interval_hours: int = 24
    health_check_interval_seconds: int = 30


@app.post("/workflow/start", tags=["workflow"])
async def start_workflow(request: WorkflowStartRequest, req: Request):
    """Start the automated feedback workflow."""
    try:
        from domains.feedback import get_feedback_workflow, WorkflowConfig

        config = WorkflowConfig(
            aggregate_interval_minutes=request.aggregate_interval_minutes,
            prune_interval_minutes=request.prune_interval_minutes,
            export_interval_hours=request.export_interval_hours,
            health_check_interval_seconds=request.health_check_interval_seconds,
        )
        workflow = get_feedback_workflow(config=config)
        workflow.start()

        return {"status": "started", "config": request.model_dump()}
    except ImportError:
        raise HTTPException(status_code=503, detail="Workflow not available")


@app.post("/workflow/stop", tags=["workflow"])
async def stop_workflow(req: Request):
    """Stop the automated feedback workflow."""
    try:
        from domains.feedback import get_feedback_workflow

        workflow = get_feedback_workflow()
        workflow.stop()

        return {"status": "stopped"}
    except ImportError:
        raise HTTPException(status_code=503, detail="Workflow not available")


@app.post("/workflow/trigger/{action}", tags=["workflow"])
async def trigger_workflow_action(action: str, req: Request):
    """Manually trigger a workflow action (aggregate, prune, export)."""
    try:
        from domains.feedback import get_feedback_workflow

        workflow = get_feedback_workflow()

        if action == "aggregate":
            return workflow.trigger_aggregate()
        elif action == "prune":
            return workflow.trigger_prune()
        elif action == "export":
            return workflow.trigger_export()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
    except ImportError:
        raise HTTPException(status_code=503, detail="Workflow not available")


@app.post("/feedback/workflow-record", tags=["meta-weights"])
async def record_feedback_via_workflow(request: RecordFeedbackRequest, req: Request):
    """
    Record feedback through the workflow manager for full automation.

    This records feedback and triggers all automated updates:
    - Meta-weights
    - Per-user LoRA
    - Online LoRA updater
    - Auto-aggregation/pruning
    """
    try:
        from domains.feedback import get_feedback_workflow

        workflow = get_feedback_workflow()

        feedback_id = workflow.record_feedback(
            user_message=request.user_message,
            assistant_response=request.assistant_response,
            rating=request.rating,
            conversation_id=request.conversation_id,
            quality_score=request.quality_score,
            user_id=request.user_id or "default",
        )

        return {
            "status": "recorded",
            "feedback_id": feedback_id,
            "workflow_active": workflow._running,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


class DatasetUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


@app.patch("/datasets/{dataset_id}", tags=["datasets"])
async def update_dataset(dataset_id: str, request: DatasetUpdateRequest):
    """Update dataset metadata (name, description, tags)."""
    from pathlib import Path
    import json

    dataset_path = Path(f"datasets/{dataset_id}")

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    meta_file = dataset_path / "metadata.json"
    metadata = {}
    if meta_file.exists():
        with open(meta_file, "r") as f:
            metadata = json.load(f)

    if request.name is not None:
        metadata["name"] = request.name
    if request.description is not None:
        metadata["description"] = request.description
    if request.tags is not None:
        metadata["tags"] = request.tags
    metadata["updated_at"] = datetime.now().isoformat()

    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return {"status": "updated", "dataset_id": dataset_id, "metadata": metadata}


@app.post("/datasets/{dataset_id}/versions", tags=["datasets"])
async def create_dataset_version(dataset_id: str, description: str = ""):
    """Create a new version snapshot of the dataset."""
    import shutil
    import uuid
    from pathlib import Path

    dataset_path = Path(f"datasets/{dataset_id}")
    versions_dir = dataset_path / ".versions"

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    versions_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_id = f"v{timestamp}_{uuid.uuid4().hex[:6]}"
    version_dir = versions_dir / version_id

    shutil.copytree(
        dataset_path, version_dir, ignore=shutil.ignore_patterns(".versions", "__pycache__")
    )

    version_meta = {
        "version_id": version_id,
        "created_at": datetime.now().isoformat(),
        "description": description or f"Auto-snapshot",
        "files": [
            f.name
            for f in dataset_path.iterdir()
            if f.is_file() and f.name not in (".versions", "__pycache__")
        ],
    }

    meta_file = version_dir / "version_meta.json"
    with open(meta_file, "w") as f:
        json.dump(version_meta, f, indent=2)

    return {"status": "created", "version_id": version_id, "version": version_meta}


@app.get("/datasets/{dataset_id}/versions", tags=["datasets"])
async def list_dataset_versions(dataset_id: str):
    """List all versions of a dataset."""
    from pathlib import Path

    dataset_path = Path(f"datasets/{dataset_id}")
    versions_dir = dataset_path / ".versions"

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not versions_dir.exists():
        return {"versions": [], "count": 0}

    versions = []
    for v_dir in sorted(versions_dir.iterdir(), reverse=True):
        if v_dir.is_dir():
            meta_file = v_dir / "version_meta.json"
            if meta_file.exists():
                with open(meta_file, "r") as f:
                    versions.append(json.load(f))
            else:
                versions.append(
                    {
                        "version_id": v_dir.name,
                        "created_at": datetime.fromtimestamp(v_dir.stat().st_mtime).isoformat(),
                        "description": "Manual snapshot",
                    }
                )

    return {"versions": versions, "count": len(versions)}


@app.post("/datasets/{dataset_id}/rollback/{version_id}", tags=["datasets"])
async def rollback_dataset_version(dataset_id: str, version_id: str):
    """Rollback dataset to a previous version."""
    import shutil
    from pathlib import Path

    dataset_path = Path(f"datasets/{dataset_id}")
    versions_dir = dataset_path / ".versions"
    version_dir = versions_dir / version_id

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not version_dir.exists():
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")

    for item in dataset_path.iterdir():
        if item.name in (".versions", "__pycache__"):
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    for item in version_dir.iterdir():
        if item.name == "version_meta.json":
            continue
        if item.is_dir():
            shutil.copytree(item, dataset_path / item.name)
        else:
            shutil.copy2(item, dataset_path / item.name)

    return {"status": "rolled_back", "version_id": version_id}


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


async def _do_huggingface_import(request: HuggingFaceImportRequest) -> dict:
    """Internal function for HuggingFace import, meant to be run via task queue."""
    from domains.training.data_import import HuggingFaceImporter

    hf = HuggingFaceImporter()
    name = request.name or request.dataset_id.split("/")[-1]

    result = hf.download_dataset(
        dataset_id=request.dataset_id,
        name=name,
    )

    if not result.success:
        raise Exception(result.error or "Download failed")

    return {
        "success": True,
        "dataset_id": name,
        "message": f"Downloaded {result.files_imported} splits ({result.total_chars} chars)",
        "output_path": result.output_path,
    }


@app.post("/datasets/import/huggingface", tags=["datasets"])
async def import_from_huggingface(request: HuggingFaceImportRequest):
    """Import dataset from HuggingFace Hub."""
    try:
        task_id = await task_queue.submit(
            name=f"hf_import:{request.dataset_id}",
            coro=lambda: _do_huggingface_import(request),
            max_retries=3,
            task_id=f"hf_import_{hash(request.dataset_id)}",
        )
        task = await task_queue.wait_for_completion(task_id)

        if task and task.status == TaskStatus.SUCCESS:
            return task.result
        elif task and task.status == TaskStatus.FAILED:
            raise HTTPException(status_code=400, detail=task.error or "Download failed")
        else:
            raise HTTPException(status_code=408, detail="Task timed out")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _do_url_import(request: URLImportRequest) -> dict:
    """Internal function for URL import, meant to be run via task queue."""
    from domains.training.data_import import URLImporter

    url_importer = URLImporter()
    result = url_importer.import_from_url(
        url=request.url,
        dataset_name=request.name,
    )

    if not result.success:
        raise Exception(result.error or "Download failed")

    return {
        "success": True,
        "dataset_id": request.name,
        "message": f"Downloaded {result.total_chars} chars",
        "output_path": result.output_path,
    }


@app.post("/datasets/import/url", tags=["datasets"])
async def import_from_url(request: URLImportRequest):
    """Import dataset from URL."""
    try:
        task_id = await task_queue.submit(
            name=f"url_import:{request.url[:50]}",
            coro=lambda: _do_url_import(request),
            max_retries=3,
            task_id=f"url_import_{hash(request.url)}",
        )
        task = await task_queue.wait_for_completion(task_id)

        if task and task.status == TaskStatus.SUCCESS:
            return task.result
        elif task and task.status == TaskStatus.FAILED:
            raise HTTPException(status_code=400, detail=task.error or "Download failed")
        else:
            raise HTTPException(status_code=408, detail="Task timed out")

    except HTTPException:
        raise
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


class BatchImportRequest(BaseModel):
    sources: List[Dict[str, Any]]  # List of {type: "url"|"local"|"github", ...params}


@app.post("/datasets/import/batch", tags=["datasets"])
async def batch_import(request: BatchImportRequest):
    """Import multiple datasets in one request."""
    from domains.training.data_import import DataImporter, URLImporter

    results = []
    errors = []

    for i, source in enumerate(request.sources[:20]):  # Max 20 imports
        source_type = source.get("type", "")
        name = source.get("name", f"batch_{i}")

        try:
            if source_type == "url":
                importer = URLImporter()
                result = importer.import_from_url(
                    url=source.get("url", ""),
                    dataset_name=name,
                )
            elif source_type == "local":
                importer = DataImporter()
                result = importer.import_from_local(
                    path=source.get("path", ""),
                    name=name,
                    extensions=source.get("extensions"),
                )
            else:
                errors.append({"index": i, "error": f"Unknown source type: {source_type}"})
                continue

            if result.success:
                results.append(
                    {
                        "index": i,
                        "dataset_id": name,
                        "success": True,
                        "message": f"Imported {result.files_imported or 1} items",
                    }
                )
            else:
                errors.append({"index": i, "error": result.error or "Import failed"})
        except Exception as e:
            errors.append({"index": i, "error": str(e)})

    return {
        "total": len(request.sources),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }


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

                # MPS doesn't support quantization well, use native inference
                if request.device == "mps" or request.device == "auto":
                    logger.info("Using native MPS inference")
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


def _start_feedback_workflow() -> None:
    """Start the automated feedback workflow at server startup."""
    try:
        from domains.feedback import get_feedback_workflow, WorkflowConfig

        auto_start = os.environ.get("SLOUGHGPT_AUTO_WORKFLOW", "true").lower() == "true"
        if not auto_start:
            logger.info("SLOUGHGPT_AUTO_WORKFLOW is false; skipping workflow startup")
            return

        workflow = get_feedback_workflow()
        if not workflow._running:
            workflow.start()
            logger.info("Feedback workflow started automatically")
    except Exception as e:
        logger.warning("Failed to start feedback workflow: %s", e)


if __name__ == "__main__":
    import uvicorn

    raw_port = os.environ.get("SLOUGHGPT_API_PORT", "").strip()
    if raw_port:
        port = int(raw_port)
    else:
        port = find_available_port(8000)

    _start_feedback_workflow()
    print(f"Starting SloughGPT server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
