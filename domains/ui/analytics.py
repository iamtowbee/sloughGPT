"""
Analytics Dashboard - Ported from recovered analytics_dashboard.py
Real-time monitoring for training metrics and system health
"""

import time
import threading
from collections import deque
from typing import Dict, Any, Optional


class AnalyticsManager:
    """Manages training analytics and monitoring."""
    
    def __init__(self):
        self.start_time = time.time()
        self.epochs_completed = 0
        self.total_training_time = 0
        self.performance_data = deque(maxlen=100)
        self.training_logs = deque(maxlen=50)
        self._lock = threading.Lock()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_gb": psutil.virtual_memory().used / (1024**3),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_usage": psutil.disk_usage('/').percent,
                "timestamp": time.time()
            }
        except ImportError:
            return {"error": "psutil not available", "timestamp": time.time()}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get training performance metrics."""
        with self._lock:
            return {
                "epochs_completed": self.epochs_completed,
                "total_training_time": self.total_training_time,
                "uptime": time.time() - self.start_time,
                "timestamp": time.time()
            }
    
    def update_training_status(self, status: str, epoch: int = 0, loss: float = 0.0, lr: float = 0.0003) -> None:
        """Update training status."""
        with self._lock:
            self.epochs_completed = epoch
            self.performance_data.append({
                "epoch": epoch,
                "loss": loss,
                "learning_rate": lr,
                "timestamp": time.time()
            })
    
    def log_training_event(self, event: str, level: str = "info") -> None:
        """Log a training event."""
        self.training_logs.append({
            "event": event,
            "level": level,
            "timestamp": time.time()
        })
    
    def get_training_history(self, limit: int = 100) -> list:
        """Get training history."""
        return list(self.performance_data)[-limit:]
    
    def get_logs(self, limit: int = 50) -> list:
        """Get recent logs."""
        return list(self.training_logs)[-limit:]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all dashboard data."""
        return {
            "system": self.get_system_metrics(),
            "performance": self.get_performance_metrics(),
            "recent_logs": self.get_logs(10),
            "uptime_seconds": time.time() - self.start_time,
        }


class MetricsCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
    
    def record(self, metric_name: str, value: float) -> None:
        """Record a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = deque(maxlen=self.window_size)
        self.metrics[metric_name].append({"value": value, "timestamp": time.time()})
    
    def get_average(self, metric_name: str, window: Optional[int] = None) -> float:
        """Get average value for a metric."""
        if metric_name not in self.metrics:
            return 0.0
        
        data = list(self.metrics[metric_name])
        if not data:
            return 0.0
        
        window = window or self.window_size
        recent = data[-window:]
        return sum(d["value"] for d in recent) / len(recent)
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        return self.metrics[metric_name][-1]["value"]


__all__ = ["AnalyticsManager", "MetricsCollector"]
