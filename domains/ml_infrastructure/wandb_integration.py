"""
Weights & Biases (W&B) Integration for SloughGPT

Provides seamless integration with W&B for experiment tracking.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger("sloughgpt.wandb")


@dataclass
class WandBConfig:
    project: str = "sloughgpt"
    entity: Optional[str] = None
    name: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[list] = None
    dir: Optional[str] = None
    mode: str = "online"
    resume: Optional[str] = None
    force: bool = False


class WandBIntegration:
    """Weights & Biases integration wrapper"""

    _instance = None

    def __init__(self, config: Optional[WandBConfig] = None):
        self.config = config or WandBConfig()
        self._run = None

    @classmethod
    def get_instance(cls, config: Optional[WandBConfig] = None):
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def is_available(self) -> bool:
        try:
            import wandb
            return True
        except ImportError:
            return False

    def setup(self, **kwargs) -> bool:
        """Initialize W&B"""
        if not self.is_available():
            logger.warning("wandb not installed. Install with: pip install wandb")
            return False

        import wandb

        config = {
            "project": self.config.project,
            "entity": self.config.entity,
            "name": self.config.name,
            "notes": self.config.notes,
            "tags": self.config.tags,
            "dir": self.config.dir,
            "mode": self.config.mode,
            "resume": self.config.resume,
            "force": self.config.force,
        }
        config.update(kwargs)
        config = {k: v for k, v in config.items() if v is not None}

        wandb.init(**config)
        logger.info(f"W&B initialized: {self.config.project}")
        return True

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics"""
        if not self.is_available():
            return
        import wandb
        wandb.log(metrics, step=step)

    def log_summary(self, metrics: Dict[str, Any]):
        """Log summary metrics"""
        if not self.is_available():
            return
        import wandb
        for key, value in metrics.items():
            wandb.run.summary[key] = value

    def watch(self, model, log: str = "gradients", log_freq: int = 100):
        """Watch model parameters"""
        if not self.is_available():
            return
        import wandb
        wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self):
        """Finish W&B run"""
        if not self.is_available():
            return
        import wandb
        wandb.finish()


def get_wandb_tracker(
    project: str = "sloughgpt",
    entity: Optional[str] = None,
    **kwargs,
) -> Optional[WandBIntegration]:
    """Get or create W&B tracker"""
    config = WandBConfig(project=project, entity=entity, **kwargs)
    tracker = WandBIntegration.get_instance(config)
    if tracker.setup():
        return tracker
    return None


__all__ = [
    "WandBIntegration",
    "WandBConfig",
    "get_wandb_tracker",
]
