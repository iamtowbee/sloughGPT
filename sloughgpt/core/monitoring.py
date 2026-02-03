"""
SloughGPT Monitoring and Alerting System
Comprehensive monitoring with Prometheus, Grafana, and alerting
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from collections import defaultdict, deque

from sloughgpt.core.logging_system import get_logger, timer
from sloughgpt.core.performance import get_performance_optimizer

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"

@dataclass
class Metric:
    """Base metric class"""
    name: str
    metric_type: MetricType
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    help_text: str = ""
    unit: str = ""
    
    def to_prometheus_format(self) -> str:
        """Convert to Prometheus metric format"""
        if self.metric_type == MetricType.COUNTER:
            return f"{self.name}_total {self._format_labels()} {self.value}"
        elif self.metric_type == MetricType.GAUGE:
            return f"{self.name} {self._format_labels()} {self.value}"
        elif self.metric_type == MetricType.HISTOGRAM:
            return f"{self.name}_bucket {self._format_labels()} {self.value}"
        elif self.metric_type == MetricType.SUMMARY:
            return f"{self.name}_sum {self._format_labels()} {self.value}"
        else:
            return f"{self.name} {self._format_labels()} {self.value}"
    
    def _format_labels(self) -> str:
        """Format labels for Prometheus"""
        if not self.labels:
            return ""
        
        label_pairs = [f'{k}="{v}"' for k, v in self.labels.items()]
        return "{" + ",".join(label_pairs) + "}"

@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    condition: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    fired_at: Optional[float] = None
    resolved_at: Optional[float] = None
    is_active: bool = False
    notification_sent: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "condition": self.condition,
            "labels": self.labels,
            "annotations": self.annotations,
            "fired_at": self.fired_at,
            "resolved_at": self.resolved_at,
            "is_active": self.is_active,
            "notification_sent": self.notification_sent
        }

@dataclass
class Dashboard:
    """Monitoring dashboard configuration"""
    dashboard_id: str
    name: str
    description: str
    panels: List[Dict[str, Any]] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    refresh_interval: str = "30s"

class MetricsCollector(ABC):
    """Abstract base class for metrics collectors"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"collector_{self.__class__.__name__.lower()}")
        self.metrics: List[Metric] = []
    
    @abstractmethod
    async def collect_metrics(self) -> List[Metric]:
        """Collect metrics"""
        pass
    
    def add_metric(self, metric: Metric) -> None:
        """Add metric to collector"""
        self.metrics.append(metric)
        self.logger.debug(f"Added metric: {metric.name} = {metric.value}")
    
    def get_metrics(self) -> List[Metric]:
        """Get all collected metrics"""
        return self.metrics.copy()

class SystemMetricsCollector(MetricsCollector):
    """Collects system-level metrics"""
    
    async def collect_metrics(self) -> List[Metric]:
        """Collect system metrics"""
        self.logger.debug("Collecting system metrics")
        
        try:
            # CPU usage
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            self.add_metric(Metric(
                name="system_cpu_usage_percent",
                metric_type=MetricType.GAUGE,
                value=cpu_percent,
                unit="percent",
                help_text="Current CPU usage percentage"
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.add_metric(Metric(
                name="system_memory_usage_bytes",
                metric_type=MetricType.GAUGE,
                value=memory.used,
                unit="bytes",
                help_text="Current memory usage in bytes"
            ))
            
            self.add_metric(Metric(
                name="system_memory_usage_percent",
                metric_type=MetricType.GAUGE,
                value=memory.percent,
                unit="percent",
                help_text="Current memory usage percentage"
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.add_metric(Metric(
                name="system_disk_usage_bytes",
                metric_type=MetricType.GAUGE,
                value=disk.used,
                unit="bytes",
                help_text="Current disk usage in bytes"
            ))
            
            # Network I/O
            network = psutil.net_io_counters()
            self.add_metric(Metric(
                name="system_network_bytes_sent_total",
                metric_type=MetricType.COUNTER,
                value=network.bytes_sent,
                unit="bytes",
                help_text="Total bytes sent over network"
            ))
            
            self.add_metric(Metric(
                name="system_network_bytes_recv_total",
                metric_type=MetricType.COUNTER,
                value=network.bytes_recv,
                unit="bytes",
                help_text="Total bytes received over network"
            ))
            
        except ImportError:
            self.logger.warning("psutil not available, system metrics collection disabled")
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
        
        return self.get_metrics()

class ApplicationMetricsCollector(MetricsCollector):
    """Collects application-specific metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.request_count = 0
        self.error_count = 0
        self.response_times = deque(maxlen=1000)
    
    async def collect_metrics(self) -> List[Metric]:
        """Collect application metrics"""
        self.logger.debug("Collecting application metrics")
        
        # Request count
        self.add_metric(Metric(
            name="app_requests_total",
            metric_type=MetricType.COUNTER,
            value=self.request_count,
            help_text="Total number of requests processed"
        ))
        
        # Error count
        self.add_metric(Metric(
            name="app_errors_total",
            metric_type=MetricType.COUNTER,
            value=self.error_count,
            help_text="Total number of errors encountered"
        ))
        
        # Response time metrics
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            self.add_metric(Metric(
                name="app_response_time_seconds",
                metric_type=MetricType.GAUGE,
                value=avg_response_time,
                unit="seconds",
                help_text="Average response time in seconds"
            ))
            
            # Response time histogram
            response_time_ms = avg_response_time * 1000
            self.add_metric(Metric(
                name="app_response_duration_ms",
                metric_type=MetricType.HISTOGRAM,
                value=response_time_ms,
                unit="ms",
                help_text="Response time histogram in milliseconds"
            ))
        
        # Active connections
        self.add_metric(Metric(
            name="app_active_connections",
            metric_type=MetricType.GAUGE,
            value=self._get_active_connections(),
            help_text="Number of active connections"
        ))
        
        return self.get_metrics()
    
    def _get_active_connections(self) -> int:
        """Get number of active connections (mock implementation)"""
        # In real implementation, this would track actual connections
        import random
        return random.randint(10, 100)
    
    def record_request(self, response_time: float) -> None:
        """Record a request"""
        self.request_count += 1
        self.response_times.append(response_time)
    
    def record_error(self) -> None:
        """Record an error"""
        self.error_count += 1

class AIMetricsCollector(MetricsCollector):
    """Collects AI model metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.inference_count = 0
        self.inference_times = deque(maxlen=500)
        self.token_counts = deque(maxlen=500)
        self.model_loads = {}
    
    async def collect_metrics(self) -> List[Metric]:
        """Collect AI metrics"""
        self.logger.debug("Collecting AI metrics")
        
        # Inference count
        self.add_metric(Metric(
            name="ai_inferences_total",
            metric_type=MetricType.COUNTER,
            value=self.inference_count,
            help_text="Total number of AI inferences"
        ))
        
        # Inference time metrics
        if self.inference_times:
            avg_inference_time = sum(self.inference_times) / len(self.inference_times)
            self.add_metric(Metric(
                name="ai_inference_time_seconds",
                metric_type=MetricType.GAUGE,
                value=avg_inference_time,
                unit="seconds",
                help_text="Average AI inference time in seconds"
            ))
            
            # Token generation rate
            if self.token_counts:
                avg_tokens = sum(self.token_counts) / len(self.token_counts)
                tokens_per_second = avg_tokens / avg_inference_time if avg_inference_time > 0 else 0
                
                self.add_metric(Metric(
                    name="ai_tokens_per_second",
                    metric_type=MetricType.GAUGE,
                    value=tokens_per_second,
                    unit="tokens/sec",
                    help_text="AI token generation rate"
                ))
        
        # Model utilization
        for model_name, load in self.model_loads.items():
            self.add_metric(Metric(
                name="ai_model_utilization",
                metric_type=MetricType.GAUGE,
                value=load,
                labels={"model": model_name},
                unit="percent",
                help_text=f"Model utilization for {model_name}"
            ))
        
        return self.get_metrics()
    
    def record_inference(self, inference_time: float, token_count: int, model_name: str) -> None:
        """Record an AI inference"""
        self.inference_count += 1
        self.inference_times.append(inference_time)
        self.token_counts.append(token_count)
        
        # Update model load
        if model_name not in self.model_loads:
            self.model_loads[model_name] = 0.0
        
        # Simple moving average for model load
        self.model_loads[model_name] = 0.9 * self.model_loads[model_name] + 0.1

class AlertManager:
    """Manages alerting and notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("alert_manager")
        
        # Alert rules
        self.alert_rules = []
        
        # Active alerts
        self.active_alerts: Dict[str, Alert] = {}
        
        # Notification channels
        self.notification_channels = []
        
        # Alert history
        self.alert_history: List[Alert] = []
        
        # Initialize default alert rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self) -> None:
        """Initialize default alert rules"""
        self.logger.info("Initializing default alert rules")
        
        # High error rate alert
        self.add_alert_rule({
            "name": "high_error_rate",
            "severity": AlertSeverity.ERROR,
            "condition": "app_errors_total > 10",
            "message": "High error rate detected",
            "duration": 300  # 5 minutes
        })
        
        # High response time alert
        self.add_alert_rule({
            "name": "slow_response_time",
            "severity": AlertSeverity.WARNING,
            "condition": "app_response_time_seconds > 5",
            "message": "Slow response time detected",
            "duration": 600  # 10 minutes
        })
        
        # High CPU usage alert
        self.add_alert_rule({
            "name": "high_cpu_usage",
            "severity": AlertSeverity.WARNING,
            "condition": "system_cpu_usage_percent > 80",
            "message": "High CPU usage detected",
            "duration": 300
        })
        
        # High memory usage alert
        self.add_alert_rule({
            "name": "high_memory_usage",
            "severity": AlertSeverity.CRITICAL,
            "condition": "system_memory_usage_percent > 90",
            "message": "High memory usage detected",
            "duration": 180  # 3 minutes
        })
        
        # AI model alert
        self.add_alert_rule({
            "name": "ai_model_overload",
            "severity": AlertSeverity.WARNING,
            "condition": "ai_model_utilization > 0.9",
            "message": "AI model utilization high",
            "duration": 300
        })
    
    def add_alert_rule(self, rule: Dict[str, Any]) -> None:
        """Add an alert rule"""
        alert_id = str(uuid.uuid4())
        
        alert_rule = {
            "id": alert_id,
            **rule,
            "created_at": time.time()
        }
        
        self.alert_rules.append(alert_rule)
        self.logger.debug(f"Added alert rule: {rule['name']}")
    
    async def check_alerts(self, metrics: List[Metric]) -> None:
        """Check all alert rules against current metrics"""
        self.logger.debug("Checking alert rules")
        
        for rule in self.alert_rules:
            try:
                await self._evaluate_alert_rule(rule, metrics)
            except Exception as e:
                self.logger.error(f"Error evaluating alert rule {rule['name']}: {e}")
    
    async def _evaluate_alert_rule(self, rule: Dict[str, Any], metrics: List[Metric]) -> None:
        """Evaluate a single alert rule"""
        condition = rule["condition"]
        
        # Parse condition (simplified)
        if self._evaluate_condition(condition, metrics):
            await self._trigger_alert(rule)
        else:
            await self._resolve_alert(rule["id"])
    
    def _evaluate_condition(self, condition: str, metrics: List[Metric]) -> bool:
        """Evaluate alert condition (simplified parser)"""
        # Mock condition evaluation - in real implementation, use a proper expression parser
        
        # Parse metric name and threshold
        if ">" in condition:
            parts = condition.split(">")
            if len(parts) == 2:
                metric_name = parts[0].strip()
                threshold = float(parts[1].strip())
                
                # Find metric value
                metric_value = self._get_metric_value(metric_name, metrics)
                if metric_value is not None:
                    return metric_value > threshold
        
        return False
    
    def _get_metric_value(self, metric_name: str, metrics: List[Metric]) -> Optional[float]:
        """Get metric value by name"""
        for metric in metrics:
            if metric.name == metric_name:
                return float(metric.value)
        return None
    
    async def _trigger_alert(self, rule: Dict[str, Any]) -> None:
        """Trigger an alert"""
        rule_id = rule["id"]
        
        if rule_id not in self.active_alerts:
            alert = Alert(
                alert_id=rule_id,
                name=rule["name"],
                severity=rule["severity"],
                message=rule["message"],
                condition=rule["condition"],
                fired_at=time.time(),
                is_active=True
            )
            
            self.active_alerts[rule_id] = alert
            self.alert_history.append(alert)
            
            self.logger.warning(f"Alert triggered: {rule['name']} - {rule['message']}")
            
            # Send notifications
            await self._send_notification(alert)
    
    async def _resolve_alert(self, rule_id: str) -> None:
        """Resolve an alert"""
        if rule_id in self.active_alerts:
            alert = self.active_alerts[rule_id]
            alert.resolved_at = time.time()
            alert.is_active = False
            
            # Move to history
            del self.active_alerts[rule_id]
            self.alert_history.append(alert)
            
            self.logger.info(f"Alert resolved: {alert.name}")
            
            # Send resolution notification
            await self._send_resolution_notification(alert)
    
    async def _send_notification(self, alert: Alert) -> None:
        """Send alert notification"""
        if alert.notification_sent:
            return
        
        # Mock notification sending
        await asyncio.sleep(0.1)
        alert.notification_sent = True
        
        self.logger.info(f"Notification sent for alert: {alert.name}")
    
    async def _send_resolution_notification(self, alert: Alert) -> None:
        """Send alert resolution notification"""
        # Mock resolution notification
        await asyncio.sleep(0.05)
        
        self.logger.info(f"Resolution notification sent for alert: {alert.name}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        return sorted(self.alert_history, key=lambda a: a.fired_at or 0, reverse=True)[:limit]

class MonitoringSystem:
    """Main monitoring system orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("monitoring_system")
        self.optimizer = get_performance_optimizer()
        
        # Metrics collectors
        self.collectors = {
            "system": SystemMetricsCollector(config.get("system", {})),
            "application": ApplicationMetricsCollector(config.get("application", {})),
            "ai": AIMetricsCollector(config.get("ai", {}))
        }
        
        # Alert manager
        self.alert_manager = AlertManager(config.get("alerting", {}))
        
        # Metrics storage
        self.metrics_storage: List[Metric] = []
        
        # Background tasks
        self._background_tasks = set()
        self._running = False
        
        # Performance tracking
        self._performance_stats = {
            "metrics_collected": 0,
            "alerts_triggered": 0,
            "collection_time": 0.0,
            "alert_processing_time": 0.0
        }
    
    async def start(self) -> None:
        """Start the monitoring system"""
        self.logger.info("Starting monitoring system")
        self._running = True
        
        # Start background tasks
        self._background_tasks.add(asyncio.create_task(self._collect_metrics_loop()))
        self._background_tasks.add(asyncio.create_task(self._process_alerts_loop()))
        self._background_tasks.add(asyncio.create_task(self._generate_dashboard_data()))
        
        self.logger.info("Monitoring system started successfully")
    
    async def stop(self) -> None:
        """Stop the monitoring system"""
        self.logger.info("Stopping monitoring system")
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self.logger.info("Monitoring system stopped")
    
    async def _collect_metrics_loop(self) -> None:
        """Background loop to collect metrics"""
        collection_interval = self.config.get("collection_interval", 30)  # seconds
        
        while self._running:
            try:
                start_time = time.time()
                
                # Collect metrics from all collectors
                all_metrics = []
                for collector in self.collectors.values():
                    metrics = await collector.collect_metrics()
                    all_metrics.extend(metrics)
                
                # Store metrics
                self.metrics_storage.extend(all_metrics)
                
                # Trim metrics storage (keep last 10000)
                if len(self.metrics_storage) > 10000:
                    self.metrics_storage = self.metrics_storage[-10000:]
                
                # Check alerts
                await self.alert_manager.check_alerts(all_metrics)
                
                # Update performance stats
                collection_time = (time.time() - start_time) * 1000
                self._update_performance_stats("collection_time", collection_time)
                self._performance_stats["metrics_collected"] += len(all_metrics)
                
                self.logger.debug(f"Collected {len(all_metrics)} metrics in {collection_time:.2f}ms")
                
                await asyncio.sleep(collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(collection_interval)
    
    async def _process_alerts_loop(self) -> None:
        """Background loop to process alerts"""
        # This loop would handle alert processing logic
        while self._running:
            await asyncio.sleep(60)  # Check every minute
            # Alert processing is mainly handled in alert_manager
    
    async def _generate_dashboard_data(self) -> None:
        """Generate dashboard data"""
        dashboard_interval = self.config.get("dashboard_interval", 60)  # seconds
        
        while self._running:
            try:
                await self._update_dashboard()
                await asyncio.sleep(dashboard_interval)
                
            except Exception as e:
                self.logger.error(f"Error in dashboard generation: {e}")
                await asyncio.sleep(dashboard_interval)
    
    async def _update_dashboard(self) -> None:
        """Update dashboard data"""
        # Mock dashboard update
        await asyncio.sleep(0.1)
        self.logger.debug("Dashboard data updated")
    
    def _update_performance_stats(self, metric: str, value: float) -> None:
        """Update performance statistics"""
        current_avg = self._performance_stats.get(metric, 0.0)
        count = self._performance_stats.get(f"{metric}_count", 1)
        
        self._performance_stats[metric] = (current_avg * (count - 1) + value) / count
        self._performance_stats[f"{metric}_count"] = count
    
    def get_metrics(self, collector_name: Optional[str] = None) -> List[Metric]:
        """Get collected metrics"""
        if collector_name:
            collector = self.collectors.get(collector_name)
            if collector:
                return collector.get_metrics()
            return []
        
        return self.metrics_storage.copy()
    
    def get_alerts(self) -> Dict[str, Any]:
        """Get alert information"""
        return {
            "active_alerts": self.alert_manager.get_active_alerts(),
            "alert_history": self.alert_manager.get_alert_history(),
            "alert_rules": len(self.alert_manager.alert_rules)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self._performance_stats.copy()
        
        # Calculate alert rate
        total_alerts = stats.get("alerts_triggered", 0)
        stats["alert_rate_per_hour"] = total_alerts / (stats.get("collection_time", 1) / 3600000) if stats.get("collection_time", 1) > 0 else 0
        
        return stats
    
    def create_dashboard(self, name: str, description: str) -> Dashboard:
        """Create a monitoring dashboard"""
        dashboard = Dashboard(
            dashboard_id=str(uuid.uuid4()),
            name=name,
            description=description
        )
        
        # Add default panels
        dashboard.panels = [
            {
                "title": "Request Rate",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(app_requests_total[5m])",
                        "legendFormat": "{{instance}} {{job}}"
                    }
                ]
            },
            {
                "title": "Response Time",
                "type": "graph",
                "targets": [
                    {
                        "expr": "avg(app_response_time_seconds[5m])",
                        "legendFormat": "{{instance}} {{job}}"
                    }
                ]
            },
            {
                "title": "Error Rate",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(app_errors_total[5m])",
                        "legendFormat": "{{instance}} {{job}}"
                    }
                ]
            },
            {
                "title": "System Resources",
                "type": "graph",
                "targets": [
                    {
                        "expr": "system_cpu_usage_percent",
                        "legendFormat": "CPU Usage"
                    },
                    {
                        "expr": "system_memory_usage_percent",
                        "legendFormat": "Memory Usage"
                    }
                ]
            }
        ]
        
        return dashboard

# Global monitoring system instance
_global_monitoring_system: Optional[MonitoringSystem] = None

def get_monitoring_system(config: Optional[Dict[str, Any]] = None) -> MonitoringSystem:
    """Get or create global monitoring system"""
    global _global_monitoring_system
    if _global_monitoring_system is None:
        _global_monitoring_system = MonitoringSystem(config or {})
    return _global_monitoring_system

# Decorators for easy use
def monitor_metric(metric_name: str, metric_type: MetricType = MetricType.COUNTER):
    """Decorator to monitor function calls as metrics"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record success metric
                monitoring_system = get_monitoring_system()
                collector = monitoring_system.collectors.get("application")
                if collector:
                    if metric_type == MetricType.COUNTER:
                        collector.record_request(time.time() - start_time)
                
                return result
                
            except Exception as e:
                # Record error metric
                monitoring_system = get_monitoring_system()
                collector = monitoring_system.collectors.get("application")
                if collector:
                    collector.record_error()
                
                raise
        
        return wrapper
    return decorator

def alert_on_error(severity: AlertSeverity = AlertSeverity.ERROR):
    """Decorator to trigger alerts on errors"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Trigger alert
                monitoring_system = get_monitoring_system()
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    name=f"function_error_{func.__name__}",
                    severity=severity,
                    message=f"Error in {func.__name__}: {str(e)}",
                    condition="exception",
                    fired_at=time.time(),
                    is_active=True
                )
                
                alert_manager = monitoring_system.alert_manager
                await alert_manager._trigger_alert({
                    "id": alert.alert_id,
                    "name": alert.name,
                    "severity": alert.severity,
                    "message": alert.message,
                    "condition": alert.condition
                })
                
                raise
        
        return wrapper
    return decorator