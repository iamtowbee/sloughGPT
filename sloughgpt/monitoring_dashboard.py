#!/usr/bin/env python3
"""
SloughGPT Dashboard
Advanced monitoring and management dashboard for SloughGPT models and services
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    gpu_memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = None
    temperature: float = 0.0
    processes: int = 0

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_name: str
    timestamp: str
    inference_time_ms: float
    tokens_per_second: float
    memory_usage_mb: float
    parameters: int
    model_size_mb: float
    cache_hit_rate: float = 0.0
    queue_length: int = 0

@dataclass
class RequestMetrics:
    """API request metrics"""
    timestamp: str
    total_requests: int
    requests_per_minute: float
    avg_response_time_ms: float
    error_rate: float
    status_distribution: Dict[str, int] = None
    endpoint_usage: Dict[str, int] = None

class MonitoringDashboard:
    """Advanced monitoring dashboard for SloughGPT"""
    
    def __init__(self):
        self.system_metrics = []
        self.model_metrics = []
        self.request_metrics = []
        self.alerts = []
        self.active_models = set()
        
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        try:
            import psutil
            
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU metrics
            gpu_memory = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            except ImportError:
                pass
            
            # Network I/O
            network_io = {}
            try:
                net_io = psutil.net_io_counters()
                network_io = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
            except:
                network_io = {}
            
            # Temperature and processes
            try:
                temps = psutil.sensors_temperatures()
                temperature = temps[0].current if temps else 0.0
            except:
                temperature = 0.0
            
            processes = len(psutil.pids())
            
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_usage=cpu_percent,
                memory_usage=memory.used,
                gpu_memory_usage=gpu_memory,
                disk_usage=disk.used,
                network_io=network_io,
                temperature=temperature,
                processes=processes
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(timestamp=datetime.now().isoformat(), cpu_usage=0.0, memory_usage=0.0)
    
    def collect_model_metrics(self, model_name: str, metrics: Dict[str, Any]) -> ModelMetrics:
        """Collect model performance metrics"""
        try:
            return ModelMetrics(
                model_name=model_name,
                timestamp=datetime.now().isoformat(),
                inference_time_ms=metrics.get('inference_time_ms', 0.0),
                tokens_per_second=metrics.get('tokens_per_second', 0.0),
                memory_usage_mb=metrics.get('memory_usage_mb', 0.0),
                parameters=metrics.get('parameters', 0),
                model_size_mb=metrics.get('model_size_mb', 0.0),
                cache_hit_rate=metrics.get('cache_hit_rate', 0.0),
                queue_length=metrics.get('queue_length', 0)
            )
        except Exception as e:
            logger.error(f"Failed to collect model metrics: {e}")
            return ModelMetrics(model_name=model_name, timestamp=datetime.now().isoformat())
    
    def collect_request_metrics(self) -> RequestMetrics:
        """Collect API request metrics"""
        try:
            # This would normally be collected from API server logs
            # For demo, we'll generate sample metrics
            current_time = datetime.now()
            
            return RequestMetrics(
                timestamp=current_time.isoformat(),
                total_requests=1000 + int(time.time() % 1000),
                requests_per_minute=60.0 + (time.time() % 100),
                avg_response_time_ms=50.0 + (time.time() % 100),
                error_rate=0.01 + (time.time() % 10),
                status_distribution={
                    "200": 950,
                    "400": 30,
                    "500": 20
                },
                endpoint_usage={
                    "/generate": 600,
                    "/model/info": 200,
                    "/health": 200
                }
            )
        except Exception as e:
            logger.error(f"Failed to collect request metrics: {e}")
            return RequestMetrics(timestamp=datetime.now().isoformat())
    
    def add_alert(self, level: str, message: str, component: str, metadata: Dict[str, Any] = None):
        """Add an alert to the monitoring system"""
        alert = {
            'id': len(self.alerts),
            'level': level,
            'message': message,
            'component': component,
            'timestamp': datetime.now().isoformat(),
            'resolved': False,
            'metadata': metadata or {}
        }
        
        self.alerts.append(alert)
        
        # Log critical alerts
        if level in ['critical', 'error']:
            logger.error(f"ALERT [{level.upper()}] {component}: {message}")
        
        # Limit alert history
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]
    
    def resolve_alert(self, alert_id: int):
        """Mark an alert as resolved"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['resolved'] = True
                logger.info(f"Resolved alert {alert_id}: {alert['message']}")
                break
    
    def get_recent_alerts(self, hours: int = 24, level: str = None) -> List[Dict]:
        """Get recent alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) >= cutoff_time
            and (level is None or alert['level'] == level)
        ]
        
        return sorted(filtered_alerts, key=lambda x: x['timestamp'], reverse=True)
    
    def get_system_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get system performance summary"""
        if not self.system_metrics:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.system_metrics
            if datetime.fromisoformat(m.timestamp) >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_gpu = sum(m.gpu_memory_usage for m in recent_metrics) / len(recent_metrics)
        
        return {
            'avg_cpu_usage': avg_cpu,
            'peak_cpu_usage': max(m.cpu_usage for m in recent_metrics),
            'avg_memory_usage_mb': avg_memory / (1024**2),
            'peak_memory_usage_mb': max(m.memory_usage for m in recent_metrics) / (1024**2),
            'avg_gpu_memory_mb': avg_gpu / (1024**2),
            'peak_gpu_memory_mb': max(m.gpu_memory_usage for m in recent_metrics) / (1024**2),
            'sample_count': len(recent_metrics)
            'time_period_hours': hours
        }
    
    def get_model_summary(self, model_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get model performance summary"""
        if not self.model_metrics:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.model_metrics
            if m.model_name == model_name and datetime.fromisoformat(m.timestamp) >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        avg_inference_time = sum(m.inference_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_tokens_per_sec = sum(m.tokens_per_second for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        
        return {
            'model_name': model_name,
            'avg_inference_time_ms': avg_inference_time,
            'peak_inference_time_ms': max(m.inference_time_ms for m in recent_metrics),
            'avg_tokens_per_sec': avg_tokens_per_sec,
            'peak_tokens_per_sec': max(m.tokens_per_second for m in recent_metrics),
            'avg_memory_usage_mb': avg_memory,
            'peak_memory_usage_mb': max(m.memory_usage_mb for m in recent_metrics),
            'sample_count': len(recent_metrics),
            'time_period_hours': hours
        }
    
    def get_request_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get request metrics summary"""
        if not self.request_metrics:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.request_metrics
            if datetime.fromisoformat(m.timestamp) >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        total_requests = sum(m.total_requests for m in recent_metrics)
        avg_response_time = sum(m.avg_response_time_ms for m in recent_metrics) / len(recent_metrics)
        total_errors = sum(m.error_rate * m.total_requests for m in recent_metrics)
        
        return {
            'total_requests': total_requests,
            'requests_per_minute': total_requests / hours * 60,
            'avg_response_time_ms': avg_response_time,
            'error_rate_percent': (total_errors / total_requests) * 100 if total_requests > 0 else 0,
            'uptime_hours': hours,
            'sample_count': len(recent_metrics)
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        alerts = self.get_recent_alerts(hours=1)
        critical_alerts = [a for a in alerts if a['level'] == 'critical']
        error_alerts = [a for a in alerts if a['level'] == 'error']
        
        system_summary = self.get_system_summary()
        request_summary = self.get_request_summary()
        
        # Determine health status
        health_score = 100
        
        if critical_alerts:
            health_score = max(0, health_score - 50)
        elif error_alerts:
            health_score = max(0, health_score - 25)
        elif system_summary.get('avg_cpu_usage', 0) > 80:
            health_score = max(0, health_score - 15)
        elif system_summary.get('avg_memory_usage_mb', 0) > 4096:  # 4GB
            health_score = max(0, health_score - 10)
        
        return {
            'status': 'critical' if health_score < 50 else 'warning' if health_score < 80 else 'healthy',
            'health_score': health_score,
            'critical_alerts': len(critical_alerts),
            'error_alerts': len(error_alerts),
            'system_load': system_summary.get('avg_cpu_usage', 0),
            'memory_usage': system_summary.get('avg_memory_usage_mb', 0),
            'uptime_hours': 1,
            'last_updated': datetime.now().isoformat()
        }
    
    def start_monitoring(self):
        """Start background monitoring loop"""
        logger.info("🔍 Starting SloughGPT monitoring dashboard...")
        
        async def monitoring_loop():
            while True:
                # Collect system metrics
                system_metrics = self.collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Limit history
                if len(self.system_metrics) > 10000:
                    self.system_metrics = self.system_metrics[-5000:]
                
                # Collect request metrics (would normally come from API server)
                request_metrics = self.collect_request_metrics()
                self.request_metrics.append(request_metrics)
                
                # Check for alerts
                if system_metrics.cpu_usage > 90:
                    self.add_alert('warning', 'High CPU usage detected', 'system', 
                              {'cpu_percent': system_metrics.cpu_usage})
                
                if system_metrics.memory_usage > 0.9:  # 90% of 16GB
                    self.add_alert('warning', 'High memory usage detected', 'system',
                              {'memory_usage_mb': system_metrics.memory_usage / (1024**2)})
                
                # Clean old metrics
                if len(self.system_metrics) > 10000:
                    self.system_metrics = self.system_metrics[-5000:]
                if len(self.request_metrics) > 10000:
                    self.request_metrics = self.request_metrics[-5000:]
                
                await asyncio.sleep(10)  # Update every 10 seconds
        
        try:
            asyncio.run(monitoring_loop())
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
    
    def generate_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        system_summary = self.get_system_summary(hours)
        request_summary = self.get_request_summary(hours)
        recent_alerts = self.get_recent_alerts(hours)
        active_models = list(self.active_models)
        
        # Get model summaries for active models
        model_summaries = {}
        for model_name in active_models:
            model_summaries[model_name] = self.get_model_summary(model_name, hours)
        
        return {
            'report_generated': datetime.now().isoformat(),
            'time_period_hours': hours,
            'system_summary': system_summary,
            'request_summary': request_summary,
            'recent_alerts': recent_alerts,
            'active_models': active_models,
            'model_summaries': model_summaries,
            'total_alerts': len(self.alerts),
            'metrics_collected': len(self.system_metrics),
            'system_health': self.get_health_status()
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return {
            'active_models': list(self.active_models),
            'system_health': self.get_health_status(),
            'recent_alerts': self.get_recent_alerts(hours=1),
            'system_metrics': self.system_metrics[-10:] if self.system_metrics else [],
            'request_metrics': self.request_metrics[-10:] if self.request_metrics else [],
            'alerts_count': len(self.alerts),
            'last_updated': datetime.now().isoformat()
        }

# Global dashboard instance
_dashboard = MonitoringDashboard()

if __name__ == "__main__":
    """Demo dashboard functionality"""
    print("🔍 SloughGPT Monitoring Dashboard")
    print("=" * 50)
    
    # Start monitoring in background
    import threading
    monitor_thread = threading.Thread(target=_dashboard.start_monitoring)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        while True:
            # Generate sample data
            import random
            import time
            
            # Simulate different model metrics
            model_names = ["sloughgpt-small", "sloughgpt-medium", "sloughgpt-large"]
            
            for model_name in model_names:
                if model_name not in _dashboard.active_models:
                    _dashboard.active_models.add(model_name)
                
                # Simulate metrics collection
                metrics = {
                    'inference_time_ms': random.uniform(10, 100),
                    'tokens_per_second': random.uniform(50, 500),
                    'memory_usage_mb': random.uniform(100, 500),
                    'parameters': random.randint(1000000, 50000000),
                    'model_size_mb': random.uniform(50, 200),
                    'cache_hit_rate': random.uniform(0.7, 0.95)
                }
                
                model_metrics = _dashboard.collect_model_metrics(model_name, metrics)
                _dashboard.model_metrics.append(model_metrics)
                
                print(f"📊 Monitoring {model_name}: {metrics['tokens_per_second']:.1f} tokens/sec")
            
            # Simulate system metrics
            system_metrics = _dashboard.collect_system_metrics()
            _dashboard.system_metrics.append(system_metrics)
            
            print(f"💻 System: CPU: {system_metrics.cpu_usage:.1f}% | Memory: {system_metrics.memory_usage_mb:.1f}MB")
            
            # Generate alerts occasionally
            if random.random() < 0.1:
                if system_metrics.cpu_usage > 80:
                    _dashboard.add_alert('warning', 'High CPU usage detected', 'system')
                elif system_metrics.memory_usage > 0.8:
                    _dashboard.add_alert('warning', 'High memory usage detected', 'system')
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped")
        
        # Generate final report
        report = _dashboard.generate_report(hours=1)
        print("\n📊 Monitoring Summary:")
        print(f"  System Health: {report['system_health']['status']}")
        print(f"  CPU Usage: {report['system_summary']['avg_cpu_usage']:.1f}%")
        print(f"  Memory Usage: {report['system_summary']['avg_memory_usage_mb']:.1f}MB")
        print(f"  Total Requests: {report['request_summary']['total_requests']}")
        print(f"  Error Rate: {report['request_summary']['error_rate_percent']:.1f}%")
        print(f"  Active Models: {len(report['active_models'])}")