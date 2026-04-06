"""Optional interactive TUI package (Phase 1: session + HTTP adapters)."""

from apps.tui.session import TuiSession, discover_repo_root

__all__ = ["TuiSession", "discover_repo_root"]
