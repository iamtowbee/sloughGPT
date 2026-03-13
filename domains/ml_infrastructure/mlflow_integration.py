"""
MLflow Integration for SloughGPT

Provides seamless integration with MLflow for experiment tracking.
Supports both local MLflow server and MLflow cloud.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging
import os

logger = logging.getLogger("sloughgpt.mlflow")


class MLflowBackend(Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    DATABRICKS = "databricks"


@dataclass
class MLflowConfig:
    tracking_uri: Optional[str] = None
    backend: MLflowBackend = MLflowBackend.LOCAL
    experiment_name: str = "sloughgpt"
    run_name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    artifact_location: Optional[str] = None


class MLflowIntegration:
    """MLflow integration wrapper"""

    _instance = None
    _client = None

    def __init__(self, config: Optional[MLflowConfig] = None):
        self.config = config or MLflowConfig()
        self._run = None

    @classmethod
    def get_instance(cls, config: Optional[MLflowConfig] = None):
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def is_available(self) -> bool:
        try:
            import mlflow
            return True
        except ImportError:
            return False

    def setup(self) -> bool:
        """Initialize MLflow connection"""
        if not self.is_available():
            logger.warning("MLflow not installed. Install with: pip install mlflow")
            return False

        import mlflow

        if self.config.tracking_uri:
            mlflow.set_tracking_uri(self.config.tracking_uri)
        else:
            mlflow.set_tracking_uri("http://localhost:5000")

        mlflow.set_experiment(self.config.experiment_name)
        logger.info(f"MLflow initialized: {self.config.tracking_uri}")
        return True

    def start_run(self, run_name: Optional[str] = None) -> bool:
        """Start an MLflow run"""
        if not self.is_available():
            return False

        import mlflow

        try:
            self._run = mlflow.start_run(
                run_name=run_name or self.config.run_name,
                tags=self.config.tags,
                artifact_location=self.config.artifact_location,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            return False

    def log_param(self, key: str, value: Any):
        """Log a parameter"""
        if not self.is_available() or not self._run:
            return
        import mlflow
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters"""
        if not self.is_available() or not self._run:
            return
        import mlflow
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a metric"""
        if not self.is_available() or not self._run:
            return
        import mlflow
        mlflow.log_metric(key, value, step or 0)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics"""
        if not self.is_available() or not self._run:
            return
        import mlflow
        mlflow.log_metrics(metrics, step or 0)

    def log_artifact(self, local_path: str):
        """Log an artifact"""
        if not self.is_available() or not self._run:
            return
        import mlflow
        mlflow.log_artifact(local_path)

    def end_run(self, status: str = "FINISHED"):
        """End the current run"""
        if not self.is_available():
            return
        import mlflow
        mlflow.end_run(status)


def get_mlflow_tracker(
    tracking_uri: Optional[str] = None,
    experiment_name: str = "sloughgpt",
) -> Optional[MLflowIntegration]:
    """Get or create MLflow tracker"""
    config = MLflowConfig(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    tracker = MLflowIntegration.get_instance(config)
    if tracker.setup():
        return tracker
    return None


__all__ = [
    "MLflowIntegration",
    "MLflowConfig",
    "MLflowBackend",
    "get_mlflow_tracker",
]
