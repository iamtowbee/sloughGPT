"""
Centralized server configuration from environment.

Keeps secrets and tunables in one place for easier audit and AI-ops documentation.
Env var spellings match historical names (e.g. ``SLAUGHGPT_API_KEY``).
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from functools import lru_cache


def _parse_api_key_list(raw: str) -> frozenset[str]:
    return frozenset(x.strip() for x in raw.split(",") if x.strip())


@dataclass(frozen=True)
class SecuritySettings:
    """Auth material loaded once at process start."""

    primary_api_key: str
    jwt_secret: str
    jwt_algorithm: str
    jwt_expiration_hours: int
    valid_api_keys: frozenset[str]


@lru_cache(maxsize=1)
def get_security_settings() -> SecuritySettings:
    """Return cached security settings (singleton for the process)."""
    primary = os.getenv("SLAUGHGPT_API_KEY") or secrets.token_urlsafe(32)
    jwt_secret = os.getenv("SLAUGHGPT_JWT_SECRET") or secrets.token_urlsafe(64)
    keys = set(_parse_api_key_list(os.getenv("SLAUGHGPT_API_KEYS", "")))
    if primary:
        keys.add(primary)
    return SecuritySettings(
        primary_api_key=primary,
        jwt_secret=jwt_secret,
        jwt_algorithm="HS256",
        jwt_expiration_hours=24,
        valid_api_keys=frozenset(keys),
    )
