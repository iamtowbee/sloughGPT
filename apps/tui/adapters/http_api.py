"""Read-only HTTP helpers for API-attached TUI rooms (health, metrics, detailed health)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


@dataclass(frozen=True)
class ApiJsonResult:
    """JSON body from a GET, or error metadata on transport failure."""

    status_code: int
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Backward-compatible alias (Phase 1 early tests).
HealthFetchResult = ApiJsonResult


def _get_json(base_url: str, path: str, *, timeout: float = 5.0) -> ApiJsonResult:
    url = base_url.rstrip("/") + path
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url)
            payload: Optional[Dict[str, Any]] = None
            try:
                payload = r.json()
            except ValueError:
                payload = None
            return ApiJsonResult(status_code=r.status_code, payload=payload, error=None)
    except httpx.HTTPError as e:
        return ApiJsonResult(status_code=0, payload=None, error=str(e))


def fetch_health(base_url: str, *, timeout: float = 5.0) -> ApiJsonResult:
    """GET ``{base_url}/health``."""
    return _get_json(base_url, "/health", timeout=timeout)


def fetch_metrics(base_url: str, *, timeout: float = 5.0) -> ApiJsonResult:
    """GET ``{base_url}/metrics`` (JSON monitoring blob)."""
    return _get_json(base_url, "/metrics", timeout=timeout)


def fetch_health_detailed(base_url: str, *, timeout: float = 5.0) -> ApiJsonResult:
    """GET ``{base_url}/health/detailed``."""
    return _get_json(base_url, "/health/detailed", timeout=timeout)
