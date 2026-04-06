"""Thin adapters from the TUI to HTTP APIs and (later) local trainers."""

from apps.tui.adapters.http_api import (
    ApiJsonResult,
    HealthFetchResult,
    fetch_health,
    fetch_health_detailed,
    fetch_metrics,
)
from apps.tui.adapters.local_status import LocalStatusSnapshot, scan_local_repo

__all__ = [
    "ApiJsonResult",
    "HealthFetchResult",
    "LocalStatusSnapshot",
    "fetch_health",
    "fetch_health_detailed",
    "fetch_metrics",
    "scan_local_repo",
]
