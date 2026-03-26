"""
Monitoring Service Implementation

This module provides comprehensive monitoring capabilities including
metrics tracking, performance monitoring, and alerting.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List

from ...__init__ import BaseComponent, ComponentException, IMonitoringService


class MonitoringService(BaseComponent, IMonitoringService):
    """Advanced monitoring service"""

    def __init__(self) -> None:
        super().__init__("monitoring_service")
        self.logger = logging.getLogger(f"sloughgpt.{self.component_name}")

        # Monitoring storage
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.events: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []

        # Monitoring configuration
        self.metrics_retention = 86400 * 7  # 7 days
        self.events_retention = 86400 * 30  # 30 days

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize monitoring service"""
        try:
            self.logger.info("Initializing Monitoring Service...")

            # Start cleanup task
            asyncio.create_task(self._cleanup_loop())

            self.is_initialized = True
            self.logger.info("Monitoring Service initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Monitoring Service: {e}")
            raise ComponentException(f"Monitoring Service initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown monitoring service"""
        try:
            self.logger.info("Shutting down Monitoring Service...")
            self.is_initialized = False
            self.logger.info("Monitoring Service shutdown successfully")

        except Exception as e:
            self.logger.error(f"Failed to shutdown Monitoring Service: {e}")
            raise ComponentException(f"Monitoring Service shutdown failed: {e}")

    async def track_metric(self, metric_name: str, value: float, tags: Dict[str, str]) -> None:
        """Track a metric"""
        try:
            metric_data = {"value": value, "tags": tags, "timestamp": time.time()}

            if metric_name not in self.metrics:
                self.metrics[metric_name] = []

            self.metrics[metric_name].append(metric_data)

            # Keep only recent metrics
            if len(self.metrics[metric_name]) > 10000:
                self.metrics[metric_name] = self.metrics[metric_name][-5000:]

            self.logger.debug(f"Tracked metric {metric_name} = {value}")

        except Exception as e:
            self.logger.error(f"Failed to track metric {metric_name}: {e}")

    async def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event"""
        try:
            event = {"type": event_type, "data": data, "timestamp": time.time()}

            self.events.append(event)

            # Keep only recent events
            if len(self.events) > 50000:
                self.events = self.events[-25000:]

            self.logger.debug(f"Logged event {event_type}")

        except Exception as e:
            self.logger.error(f"Failed to log event {event_type}: {e}")

    async def record_metric(self, metric_name: str, value: float) -> None:
        """Record a metric"""
        await self.track_metric(metric_name, value, {})

    async def get_metrics(self, metric_name: str, time_range: str) -> List[Dict[str, Any]]:
        """Get metrics for a time range"""
        try:
            if metric_name not in self.metrics:
                return []

            # Parse time range
            time_seconds = self._parse_time_range(time_range)
            cutoff_time = time.time() - time_seconds

            # Filter metrics by time
            recent_metrics = [m for m in self.metrics[metric_name] if m["timestamp"] >= cutoff_time]

            return recent_metrics

        except Exception as e:
            self.logger.error(f"Failed to get metrics {metric_name}: {e}")
            return []

    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data"""
        try:
            dashboard: Dict[str, Any] = {
                "timestamp": time.time(),
                "metrics_summary": {},
                "recent_events": [],
                "active_alerts": [],
            }

            # Summarize metrics
            for metric_name, metric_list in self.metrics.items():
                if metric_list:
                    recent_values = [m["value"] for m in metric_list[-100:]]
                    dashboard["metrics_summary"][metric_name] = {
                        "current": recent_values[-1] if recent_values else 0,
                        "average": sum(recent_values) / len(recent_values),
                        "min": min(recent_values),
                        "max": max(recent_values),
                        "count": len(recent_values),
                    }

            # Recent events
            dashboard["recent_events"] = self.events[-20:]

            return dashboard

        except Exception as e:
            self.logger.error(f"Failed to get monitoring dashboard: {e}")
            return {}

    def _parse_time_range(self, time_range: str) -> int:
        """Parse time range string to seconds"""
        if time_range.endswith("h"):
            return int(time_range[:-1]) * 3600
        elif time_range.endswith("m"):
            return int(time_range[:-1]) * 60
        elif time_range.endswith("s"):
            return int(time_range[:-1])
        else:
            return 3600  # Default to 1 hour

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while self.is_initialized:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Clean up every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring cleanup error: {e}")
                await asyncio.sleep(300)

    async def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data"""
        current_time = time.time()

        # Clean up old metrics
        for metric_name in list(self.metrics.keys()):
            self.metrics[metric_name] = [
                m
                for m in self.metrics[metric_name]
                if current_time - m["timestamp"] < self.metrics_retention
            ]

            # Remove empty metric lists
            if not self.metrics[metric_name]:
                del self.metrics[metric_name]

        # Clean up old events
        self.events = [
            e for e in self.events if current_time - e["timestamp"] < self.events_retention
        ]
