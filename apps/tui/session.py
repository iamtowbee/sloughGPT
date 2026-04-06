"""Persistent TUI session: repository root and API attachment (Phase 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _is_sloughgpt_repo_root(path: Path) -> bool:
    pyproject = path / "pyproject.toml"
    if pyproject.is_file():
        try:
            text = pyproject.read_text(encoding="utf-8")
        except OSError:
            return False
        if 'name = "sloughgpt"' in text or "name='sloughgpt'" in text:
            return True
    if (path / "config.yaml").is_file() and (path / "packages" / "core-py").is_dir():
        return True
    return False


def discover_repo_root(start: Optional[Path] = None, *, max_depth: int = 32) -> Optional[Path]:
    """Walk parents from ``start`` (default: cwd) until a SloughGPT repo root is found."""
    cur = (start or Path.cwd()).resolve()
    for _ in range(max_depth):
        if _is_sloughgpt_repo_root(cur):
            return cur
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return None


@dataclass
class TuiSession:
    """One interactive session: where the repo lives and which API to talk to."""

    repo_root: Path
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    device: Optional[str] = None
    last_checkpoint: Optional[Path] = None
    last_soul_path: Optional[Path] = None
    last_job_id: Optional[str] = None
    last_error: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def api_base_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"
