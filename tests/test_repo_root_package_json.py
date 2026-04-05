"""Integration-style checks for optional repo-root npm metadata (``dev:stack``)."""

from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_root_package_json_declares_dev_stack() -> None:
    pkg = _repo_root() / "package.json"
    assert pkg.is_file(), f"missing {pkg}"
    data = json.loads(pkg.read_text(encoding="utf-8"))
    scripts = data.get("scripts", {})
    assert "dev:stack" in scripts, "root package.json should expose npm run dev:stack"
    assert "concurrently" in scripts["dev:stack"]
    dev_deps = data.get("devDependencies", {})
    assert "concurrently" in dev_deps, "root package.json should list concurrently as a devDependency"


def test_verify_sh_requires_root_package_json() -> None:
    text = (_repo_root() / "verify.sh").read_text(encoding="utf-8")
    assert '"package.json"' in text, "verify.sh file list should include root package.json"
