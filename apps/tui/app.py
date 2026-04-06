"""CLI entry for the future Textual TUI; Phase 1 stub + read-only probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from apps.tui.adapters.http_api import ApiJsonResult
    from apps.tui.adapters.local_status import LocalStatusSnapshot


def _print_local_status(snap: "LocalStatusSnapshot") -> None:
    print(f"repo_root: {snap.repo_root}")
    print(f"models: dir_found={snap.models_dir_found} count={snap.model_file_count}")
    for p in snap.model_sample_paths:
        print(f"  - {p}")
    print(f"datasets: dir_found={snap.datasets_dir_found} count={snap.dataset_entry_count}")
    for name in snap.dataset_sample_names:
        print(f"  - {name}")


def _print_api_json(label: str, r: "ApiJsonResult", *, max_chars: int = 12_000) -> None:
    print(label)
    print(f"  status_code: {r.status_code}")
    if r.error:
        print(f"  error: {r.error}")
        return
    if r.payload is None:
        print("  (no JSON body)")
        return
    text = json.dumps(r.payload, indent=2)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n  ... (truncated)"
    print(text)


def main(argv: Optional[List[str]] = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="sloughgpt-tui",
        description="SloughGPT interactive terminal UI (Phase 1: session + read-only adapters).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="API host for --api-* probes")
    parser.add_argument("--port", type=int, default=8000, help="API port for --api-* probes")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        metavar="PATH",
        help="Repository root for --local-status (default: discover or cwd)",
    )
    parser.add_argument(
        "--local-status",
        action="store_true",
        help="Scan local models/ and datasets/ (same idea as cli.py status)",
    )
    parser.add_argument("--api-health", action="store_true", help="GET /health")
    parser.add_argument("--api-metrics", action="store_true", help="GET /metrics (JSON)")
    parser.add_argument("--api-health-detailed", action="store_true", help="GET /health/detailed")
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.0.1",
    )
    ns = parser.parse_args(args)

    base = f"http://{ns.host}:{ns.port}"
    any_action = (
        ns.local_status or ns.api_health or ns.api_metrics or ns.api_health_detailed
    )

    if ns.local_status:
        from apps.tui.adapters.local_status import scan_local_repo
        from apps.tui.session import discover_repo_root

        root = ns.repo_root.resolve() if ns.repo_root is not None else (discover_repo_root() or Path.cwd())
        snap = scan_local_repo(root)
        _print_local_status(snap)

    if ns.api_health:
        from apps.tui.adapters.http_api import fetch_health

        _print_api_json("GET /health", fetch_health(base))

    if ns.api_metrics:
        from apps.tui.adapters.http_api import fetch_metrics

        _print_api_json("GET /metrics", fetch_metrics(base))

    if ns.api_health_detailed:
        from apps.tui.adapters.http_api import fetch_health_detailed

        _print_api_json("GET /health/detailed", fetch_health_detailed(base))

    if not any_action:
        print(
            "TUI Phase 1: use --local-status, --api-health, --api-metrics, --api-health-detailed. "
            "Modules: apps.tui.session, apps.tui.adapters. Roadmap: docs/plans/tui-cli-port.md",
            file=sys.stdout,
        )


if __name__ == "__main__":
    main()
