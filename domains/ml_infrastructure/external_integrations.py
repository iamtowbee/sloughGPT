"""
External ML Platform Integrations - MLflow and Weights & Biases

Provides optional integration with external experiment tracking platforms.
These are optional dependencies - the system works without them.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging
import os

logger = logging.getLogger("sloughgpt.integrations")


class Platform(Enum):
    MLFLOW = "mlflow"
    WANDB = "wandb"
    LOCAL = "local"


@dataclass
class IntegrationConfig:
    """Configuration for external platform integration."""
    platform: Platform
    enabled: bool = False
    api_key: Optional[str] = None
    project: str = "sloughgpt"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


class MLflowIntegration:
    """MLflow integration wrapper."""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self._client = None
        self._run = None
    
    def _import_mlflow(self):
        try:
            import mlflow
            return mlflow
        except ImportError:
            raise ImportError(
                "MLflow not installed. Install with: pip install mlflow"
            )
    
    def enable(self):
        """Enable MLflow tracking."""
        mlflow = self._import_mlflow()
        
        if self.config.api_key:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.config.api_key
        
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.config.project)
        
        self._run = mlflow.start_run(
            run_name=self.config.run_name or "sloughgpt-run",
            tags=self.config.tags or {}
        )
        self._client = mlflow
        logger.info(f"MLflow tracking enabled: {tracking_uri}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric."""
        if self._client:
            self._client.log_metric(mlflow.entities.Metric(name, value, step or 0, 0))
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        if self._client:
            for name, value in metrics.items():
                self.log_metric(name, value, step)
    
    def log_param(self, name: str, value: Any):
        """Log a parameter."""
        if self._client:
            self._client.log_param(name, value)
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        if self._client:
            for name, value in params.items():
                self.log_param(name, value)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact."""
        if self._client:
            self._client.log_artifact(local_path, artifact_path)
    
    def end(self):
        """End the MLflow run."""
        if self._client:
            self._client.end_run()


class WandbIntegration:
    """Weights & Biases integration wrapper."""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self._run = None
    
    def _import_wandb(self):
        try:
            import wandb
            return wandb
        except ImportError:
            raise ImportError(
                "Weights & Biases not installed. Install with: pip install wandb"
            )
    
    def enable(self):
        """Enable W&B tracking."""
        wandb = self._import_wandb()
        
        wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=self.config.run_name,
            tags=list(self.config.tags.keys()) if self.config.tags else None,
            config=self.config.tags or {}
        )
        
        self._run = wandb
        logger.info(f"W&B tracking enabled: {self.config.project}")
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        if self._run:
            self._run.log(metrics, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        if self._run:
            self._run.log(metrics, step=step)
    
    def log_param(self, name: str, value: Any):
        """Log a parameter."""
        if self._run:
            self._run.config[name] = value
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        if self._run:
            self._run.config.update(params)
    
    def log_artifact(self, path: str, name: Optional[str] = None):
        """Log an artifact."""
        if self._run:
            artifact = self._run.Artifact(name or path, type="model")
            artifact.add_file(path)
            self._run.log_artifact(artifact)
    
    def end(self):
        """Finish the W&B run."""
        if self._run:
            self._run.finish()


class ExternalIntegrator:
    """
    Unified interface for external ML platform integrations.
    
    Supports MLflow and W&B with automatic fallback to local tracking.
    """
    
    def __init__(self):
        self._mlflow: Optional[MLflowIntegration] = None
        self._wandb: Optional[WandbIntegration] = None
        self._platform: Optional[Platform] = None
    
    def configure(
        self,
        platform: Platform,
        api_key: Optional[str] = None,
        project: str = "sloughgpt",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Configure the integration."""
        config = IntegrationConfig(
            platform=platform,
            enabled=True,
            api_key=api_key or os.environ.get("MLFLOW_TRACKING_TOKEN") or os.environ.get("WANDB_API_KEY"),
            project=project,
            entity=entity,
            run_name=run_name,
            tags=tags,
        )
        
        if platform == Platform.MLFLOW:
            self._mlflow = MLflowIntegration(config)
            self._platform = Platform.MLFLOW
        elif platform == Platform.WANDB:
            self._wandb = WandbIntegration(config)
            self._platform = Platform.WANDB
        else:
            self._platform = Platform.LOCAL
    
    def enable(self):
        """Enable the configured platform."""
        if self._mlflow and self._platform == Platform.MLFLOW:
            self._mlflow.enable()
        elif self._wandb and self._platform == Platform.WANDB:
            self._wandb.enable()
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric."""
        if self._mlflow and self._platform == Platform.MLFLOW:
            self._mlflow.log_metric(name, value, step)
        elif self._wandb and self._platform == Platform.WANDB:
            self._wandb.log({name: value}, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        if self._mlflow and self._platform == Platform.MLFLOW:
            self._mlflow.log_metrics(metrics, step)
        elif self._wandb and self._platform == Platform.WANDB:
            self._wandb.log_metrics(metrics, step)
    
    def log_param(self, name: str, value: Any):
        """Log a parameter."""
        if self._mlflow and self._platform == Platform.MLFLOW:
            self._mlflow.log_param(name, value)
        elif self._wandb and self._platform == Platform.WANDB:
            self._wandb.log_param(name, value)
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        if self._mlflow and self._platform == Platform.MLFLOW:
            self._mlflow.log_params(params)
        elif self._wandb and self._platform == Platform.WANDB:
            self._wandb.log_params(params)
    
    def log_artifact(self, path: str, name: Optional[str] = None):
        """Log an artifact."""
        if self._mlflow and self._platform == Platform.MLFLOW:
            self._mlflow.log_artifact(path, name)
        elif self._wandb and self._platform == Platform.WANDB:
            self._wandb.log_artifact(path, name)
    
    def end(self):
        """End the tracking session."""
        if self._mlflow:
            self._mlflow.end()
        if self._wandb:
            self._wandb.end()
    
    @property
    def is_enabled(self) -> bool:
        """Check if any platform is enabled."""
        return self._platform is not None and self._platform != Platform.LOCAL
    
    @property
    def platform(self) -> Optional[Platform]:
        """Get the current platform."""
        return self._platform


def get_integrator() -> ExternalIntegrator:
    """Get the global ExternalIntegrator instance."""
    global _integrator
    if _integrator is None:
        _integrator = ExternalIntegrator()
    return _integrator


_integrator: Optional[ExternalIntegrator] = None


__all__ = [
    "Platform",
    "IntegrationConfig", 
    "MLflowIntegration",
    "WandbIntegration",
    "ExternalIntegrator",
    "get_integrator",
]
