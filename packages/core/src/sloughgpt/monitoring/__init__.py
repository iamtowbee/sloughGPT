"""Monitoring and metrics collection service."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import time
import threading
from collections import defaultdict, deque


@dataclass
class MetricPoint:
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    metric_type: str = "counter"  # counter, gauge, histogram


@dataclass
class AlertRule:
    name: str
    metric_name: str
    condition: str  # gt, lt, eq, gte, lte
    threshold: float
    duration: int  # seconds
    severity: str  # info, warning, critical
    enabled: bool = True


@dataclass
class Alert:
    id: str
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    message: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None


class MonitoringService:
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.metric_history: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.aggregation_window = 300  # 5 minutes
        self.running = True
        self.alert_thread = threading.Thread(target=self._alert_monitor, daemon=True)
        self.alert_thread.start()
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None,
                     metric_type: str = "counter") -> None:
        """Record a metric point."""
        labels = labels or {}
        metric_key = self._create_metric_key(name, labels)
        
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels,
            metric_type=metric_type
        )
        
        self.metrics[metric_key].append(metric_point)
        self.metric_history[name].append(metric_point)
        
        # Keep history manageable
        if len(self.metric_history[name]) > 100000:
            self.metric_history[name] = self.metric_history[name][-50000:]
    
    def increment_counter(self, name: str, value: float = 1.0, 
                         labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        self.record_metric(name, value, labels, "counter")
    
    def set_gauge(self, name: str, value: float, 
                 labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        self.record_metric(name, value, labels, "gauge")
    
    def record_histogram(self, name: str, value: float, 
                        labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        self.record_metric(name, value, labels, "histogram")
    
    def get_metric(self, name: str, labels: Optional[Dict[str, str]] = None,
                  time_range: Optional[int] = None) -> List[MetricPoint]:
        """Get metric points for a given metric."""
        labels = labels or {}
        metric_key = self._create_metric_key(name, labels)
        
        if time_range:
            cutoff_time = datetime.now() - timedelta(seconds=time_range)
            return [point for point in self.metrics[metric_key] 
                   if point.timestamp >= cutoff_time]
        
        return list(self.metrics[metric_key])
    
    def get_aggregated_metric(self, name: str, aggregation: str = "avg",
                             time_range: int = 300) -> Optional[float]:
        """Get aggregated metric value."""
        points = [point for point in self.metric_history.get(name, [])
                 if (datetime.now() - point.timestamp).total_seconds() <= time_range]
        
        if not points:
            return None
        
        values = [point.value for point in points]
        
        if aggregation == "avg":
            return sum(values) / len(values)
        elif aggregation == "sum":
            return sum(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "count":
            return len(values)
        else:
            return sum(values) / len(values)
    
    def create_alert_rule(self, rule_name: str, metric_name: str, condition: str,
                         threshold: float, duration: int = 60, severity: str = "warning") -> None:
        """Create an alert rule."""
        alert_rule = AlertRule(
            name=rule_name,
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            duration=duration,
            severity=severity
        )
        
        self.alert_rules[rule_name] = alert_rule
    
    def _create_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"
    
    def _alert_monitor(self) -> None:
        """Monitor metrics and trigger alerts."""
        while self.running:
            try:
                self._check_alert_rules()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                print(f"Alert monitor error: {e}")
                time.sleep(30)
    
    def _check_alert_rules(self) -> None:
        """Check all alert rules against current metrics."""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            current_value = self.get_aggregated_metric(rule.metric_name, "avg", rule.duration)
            
            if current_value is None:
                continue
            
            alert_triggered = self._evaluate_condition(current_value, rule.condition, rule.threshold)
            alert_id = f"{rule_name}_{rule.metric_name}"
            
            if alert_triggered and alert_id not in self.active_alerts:
                # Trigger new alert
                alert = Alert(
                    id=alert_id,
                    rule_name=rule_name,
                    metric_name=rule.metric_name,
                    current_value=current_value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    message=f"Alert: {rule.metric_name} {rule.condition} {rule.threshold} (current: {current_value})",
                    triggered_at=datetime.now()
                )
                
                self.active_alerts[alert_id] = alert
                self._send_notification(alert)
                
            elif not alert_triggered and alert_id in self.active_alerts:
                # Resolve alert
                alert = self.active_alerts[alert_id]
                alert.resolved_at = datetime.now()
                del self.active_alerts[alert_id]
                self._send_resolution_notification(alert)
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if condition == "gt":
            return value > threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "lte":
            return value <= threshold
        elif condition == "eq":
            return value == threshold
        else:
            return False
    
    def _send_notification(self, alert: Alert) -> None:
        """Send alert notification."""
        print(f"ALERT TRIGGERED: {alert.message}")
        # In production, integrate with notification systems
        
    def _send_resolution_notification(self, alert: Alert) -> None:
        """Send alert resolution notification."""
        print(f"ALERT RESOLVED: {alert.rule_name} - {alert.metric_name}")
        # In production, integrate with notification systems
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        dashboard_data = {
            "metrics": {},
            "alerts": {
                "active": len(self.active_alerts),
                "critical": len([a for a in self.active_alerts.values() if a.severity == "critical"]),
                "warning": len([a for a in self.active_alerts.values() if a.severity == "warning"])
            },
            "system_health": self._get_system_health()
        }
        
        # Add key metrics
        key_metrics = [
            "api_requests_total", "api_response_time", "error_rate",
            "cpu_usage", "memory_usage", "active_users"
        ]
        
        for metric_name in key_metrics:
            current = self.get_aggregated_metric(metric_name, "avg", 60)
            if current is not None:
                dashboard_data["metrics"][metric_name] = {
                    "current": current,
                    "trend": self._calculate_trend(metric_name)
                }
        
        return dashboard_data
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health_score = 100
        
        # Check critical alerts
        critical_alerts = [a for a in self.active_alerts.values() if a.severity == "critical"]
        if critical_alerts:
            health_score -= len(critical_alerts) * 20
        
        # Check warning alerts
        warning_alerts = [a for a in self.active_alerts.values() if a.severity == "warning"]
        if warning_alerts:
            health_score -= len(warning_alerts) * 5
        
        health_score = max(0, health_score)
        
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "status": status,
            "score": health_score,
            "active_alerts": len(self.active_alerts),
            "last_check": datetime.now().isoformat()
        }
    
    def _calculate_trend(self, metric_name: str) -> str:
        """Calculate trend for a metric."""
        recent_points = [point for point in self.metric_history.get(metric_name, [])
                         if (datetime.now() - point.timestamp).total_seconds() <= 300]
        
        if len(recent_points) < 2:
            return "stable"
        
        values = [point.value for point in recent_points[-10:]]
        if len(values) < 2:
            return "stable"
        
        # Simple trend calculation
        avg_first_half = sum(values[:len(values)//2]) / (len(values)//2 or 1)
        avg_second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2 or 1)
        
        change_percent = ((avg_second_half - avg_first_half) / avg_first_half) * 100 if avg_first_half != 0 else 0
        
        if abs(change_percent) < 5:
            return "stable"
        elif change_percent > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in specified format."""
        if format == "prometheus":
            return self._export_prometheus_format()
        elif format == "json":
            return self._export_json_format()
        else:
            return self._export_prometheus_format()
    
    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        output = []
        
        for metric_name, points in self.metric_history.items():
            if not points:
                continue
            
            latest_point = points[-1]
            labels_str = ""
            
            if latest_point.labels:
                labels_str = "{" + ",".join([f'{k}="{v}"' for k, v in latest_point.labels.items()]) + "}"
            
            output.append(f"# TYPE {metric_name} {latest_point.metric_type}")
            output.append(f"{metric_name}{labels_str} {latest_point.value}")
        
        return "\n".join(output)
    
    def _export_json_format(self) -> str:
        """Export metrics in JSON format."""
        metrics_data = {}
        
        for metric_name, points in self.metric_history.items():
            if points:
                latest_point = points[-1]
                metrics_data[metric_name] = {
                    "value": latest_point.value,
                    "timestamp": latest_point.timestamp.isoformat(),
                    "labels": latest_point.labels,
                    "type": latest_point.metric_type
                }
        
        return json.dumps(metrics_data, indent=2)
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.running = False
        if self.alert_thread.is_alive():
            self.alert_thread.join(timeout=5)