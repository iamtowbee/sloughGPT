"""Soul Manager - Hot-swappable personality system.

Allows runtime switching between different AI personalities (souls)
without restarting the inference engine.
"""

import os
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("sloughgpt.soul_manager")


@dataclass
class SoulInfo:
    """Information about a registered soul."""

    name: str
    path: str
    description: str = ""
    personality: Dict[str, float] = field(default_factory=dict)
    traits: List[str] = field(default_factory=list)
    loaded_at: Optional[float] = None


class SoulManager:
    """
    Hot-swappable soul/personality manager.

    Features:
    - List available souls
    - Switch personalities at runtime
    - Persist soul preference
    - Profile-based switching

    Usage:
        manager = SoulManager()
        souls = manager.list_souls()
        manager.switch_soul("helpful_assistant")
        current = manager.get_current_soul()
    """

    def __init__(self, souls_dir: str = "models"):
        self.souls_dir = Path(souls_dir)
        self._current_soul: Optional[str] = None
        self._souls_cache: Dict[str, SoulInfo] = {}
        self._preference_file = Path("data/.soul_preference")

        # Load cached souls
        self._scan_souls()

        # Load saved preference
        self._load_preference()

    def _scan_souls(self) -> None:
        """Scan for available .sou files."""
        self._souls_cache.clear()

        if not self.souls_dir.exists():
            logger.warning(f"Souls directory not found: {self.souls_dir}")
            return

        # Find .sou files
        for sou_path in glob.glob(str(self.souls_dir / "*.sou")):
            try:
                soul_info = self._parse_soul_info(sou_path)
                if soul_info:
                    self._souls_cache[soul_info.name] = soul_info
            except Exception as e:
                logger.debug(f"Failed to parse soul {sou_path}: {e}")

        logger.info(f"Found {len(self._souls_cache)} souls")

    def _parse_soul_info(self, sou_path: str) -> Optional[SoulInfo]:
        """Parse soul file for metadata."""
        try:
            from domains.inference.sou_format import SouParser

            parser = SouParser(sou_path)
            soul = parser.parse()

            # Extract personality traits
            personality = {
                "warmth": getattr(soul.personality, "warmth", 0.5),
                "creativity": getattr(soul.personality, "creativity", 0.5),
                "curiosity": getattr(soul.personality, "curiosity", 0.5),
                "confidence": getattr(soul.personality, "confidence", 0.5),
            }

            # Extract behavioral traits
            traits = []
            behavior = getattr(soul, "behavior", None)
            if behavior:
                traits.append(getattr(behavior, "reasoning_approach", "balanced"))

            return SoulInfo(
                name=soul.name,
                path=sou_path,
                description=getattr(soul, "description", "") or soul.name,
                personality=personality,
                traits=traits,
            )
        except Exception as e:
            logger.debug(f"Parse error for {sou_path}: {e}")
            return None

    def _load_preference(self) -> None:
        """Load saved soul preference."""
        if self._preference_file.exists():
            try:
                name = self._preference_file.read_text().strip()
                if name in self._souls_cache:
                    self._current_soul = name
                    logger.info(f"Restored soul preference: {name}")
            except Exception:
                pass

    def _save_preference(self) -> None:
        """Save current soul preference."""
        if self._current_soul:
            try:
                self._preference_file.parent.mkdir(parents=True, exist_ok=True)
                self._preference_file.write_text(self._current_soul)
            except Exception:
                pass

    def list_souls(self) -> List[SoulInfo]:
        """List all available souls."""
        self._scan_souls()
        return list(self._souls_cache.values())

    def get_soul(self, name: str) -> Optional[SoulInfo]:
        """Get soul by name."""
        return self._souls_cache.get(name)

    def get_current_soul(self) -> Optional[SoulInfo]:
        """Get currently active soul."""
        if self._current_soul:
            return self._souls_cache.get(self._current_soul)
        return None

    def switch_soul(self, name: str) -> Dict[str, Any]:
        """
        Switch to a different soul/personality.

        Returns info about the new soul.
        """
        if name not in self._souls_cache:
            return {
                "success": False,
                "error": f"Soul '{name}' not found",
                "available": list(self._souls_cache.keys()),
            }

        self._current_soul = name
        self._save_preference()

        soul = self._souls_cache[name]
        soul.loaded_at = os.times().elapsed

        logger.info(f"Switched to soul: {name}")

        return {
            "success": True,
            "name": soul.name,
            "path": soul.path,
            "description": soul.description,
            "personality": soul.personality,
            "traits": soul.traits,
        }

    def register_soul(self, path: str, name: Optional[str] = None) -> SoulInfo:
        """
        Register a new soul file.

        Args:
            path: Path to .sou file
            name: Optional custom name

        Returns:
            SoulInfo for the registered soul
        """
        soul_info = self._parse_soul_info(path)

        if not soul_info:
            raise ValueError(f"Failed to parse soul file: {path}")

        if name:
            soul_info.name = name

        self._souls_cache[soul_info.name] = soul_info

        return soul_info

    def create_default_souls(self) -> None:
        """Create default souls if none exist."""
        default_souls = [
            {
                "name": "assistant",
                "description": "Helpful and informative assistant",
                "personality": {
                    "warmth": 0.7,
                    "creativity": 0.5,
                    "curiosity": 0.8,
                    "confidence": 0.6,
                },
            },
            {
                "name": "creative",
                "description": "Creative and imaginative AI",
                "personality": {
                    "warmth": 0.6,
                    "creativity": 0.9,
                    "curiosity": 0.9,
                    "confidence": 0.5,
                },
            },
            {
                "name": "analyst",
                "description": "Analytical and precise AI",
                "personality": {
                    "warmth": 0.4,
                    "creativity": 0.3,
                    "curiosity": 0.7,
                    "confidence": 0.8,
                },
            },
        ]

        for soul_def in default_souls:
            if soul_def["name"] not in self._souls_cache:
                soul = SoulInfo(**soul_def, path="")
                self._souls_cache[soul.name] = soul

    def get_stats(self) -> Dict[str, Any]:
        """Get soul manager statistics."""
        return {
            "total_souls": len(self._souls_cache),
            "current_soul": self._current_soul,
            "souls_dir": str(self.souls_dir),
            "available_souls": [s.name for s in self._souls_cache.values()],
        }


# Global manager instance
_soul_manager: Optional[SoulManager] = None


def get_soul_manager() -> SoulManager:
    """Get the global soul manager instance."""
    global _soul_manager
    if _soul_manager is None:
        _soul_manager = SoulManager()
    return _soul_manager


def switch_soul(name: str) -> Dict[str, Any]:
    """Quick function to switch soul."""
    return get_soul_manager().switch_soul(name)


def list_souls() -> List[SoulInfo]:
    """Quick function to list souls."""
    return get_soul_manager().list_souls()
