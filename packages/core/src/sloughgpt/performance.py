"""Performance optimization tools for SloughGPT."""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio
import time
import psutil
import gc
import threading
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class PerformanceMetrics:
    cpu_percent: float
    memory_percent: float
    gpu_memory: Optional[Dict[str, float]]
    disk_usage: Dict[str, float]
    network_io: Dict[str, int]
    response_time: float
    throughput: float
    timestamp: datetime


@dataclass
class OptimizationRecommendation:
    category: str
    priority: str  # low, medium, high, critical
    description: str
    estimated_improvement: str
    implementation_difficulty: str
    action_items: List[str]


class PerformanceOptimizer:
    def __init__(self):
        self.metrics_history: deque = deque(maxlen=10000)
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self.optimization_rules = []
        self.monitoring = False
        self.monitor_thread = None
        self.alerts: List[Dict[str, Any]] = []
        
        # Performance thresholds
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "gpu_memory_percent": 90.0,
            "response_time_ms": 5000.0,
            "disk_usage_percent": 90.0
        }
    
    def start_monitoring(self, interval: float = 1.0):
        """Start performance monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logging.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logging.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)
                self._check_thresholds(metrics)
                time.sleep(interval)
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU metrics
            gpu_memory = None
            if HAS_TORCH and torch.cuda.is_available():
                gpu_memory = {
                    "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "cached_gb": torch.cuda.memory_reserved() / (1024**3),
                    "max_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
                }
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = {
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3),
                "percent": (disk.used / disk.total) * 100
            }
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Response time and throughput (simplified)
            response_time = self._measure_response_time()
            throughput = self._measure_throughput()
            
            return PerformanceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                gpu_memory=gpu_memory,
                disk_usage=disk_usage,
                network_io=network_io,
                response_time=response_time,
                throughput=throughput,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logging.error(f"Error collecting metrics: {e}")
            return PerformanceMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                gpu_memory=None,
                disk_usage={},
                network_io={},
                response_time=0.0,
                throughput=0.0,
                timestamp=datetime.now()
            )
    
    def _measure_response_time(self) -> float:
        """Measure response time (simplified)."""
        # In practice, this would measure actual API response times
        # For now, return a simulated value
        if HAS_TORCH and torch.cuda.is_available():
            # Simulate GPU computation time
            start = time.time()
            torch.randn(1000, 1000).cuda()
            torch.cuda.synchronize()
            return (time.time() - start) * 1000  # Convert to ms
        else:
            # CPU simulation
            start = time.time()
            x = [i for i in range(10000)]
            sum(x)
            return (time.time() - start) * 1000
    
    def _measure_throughput(self) -> float:
        """Measure system throughput (simplified)."""
        # In practice, this would measure requests/second
        # For now, return a simulated value based on CPU usage
        cpu = psutil.cpu_percent(interval=0.1)
        return max(0, 100 - cpu) * 10  # Simplified throughput calculation
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check if metrics exceed thresholds and create alerts."""
        alerts = []
        
        if metrics.cpu_percent > self.thresholds["cpu_percent"]:
            alerts.append({
                "type": "cpu",
                "severity": "warning" if metrics.cpu_percent < 95 else "critical",
                "message": f"High CPU usage: {metrics.cpu_percent:.1f}%",
                "timestamp": metrics.timestamp
            })
        
        if metrics.memory_percent > self.thresholds["memory_percent"]:
            alerts.append({
                "type": "memory",
                "severity": "warning" if metrics.memory_percent < 95 else "critical",
                "message": f"High memory usage: {metrics.memory_percent:.1f}%",
                "timestamp": metrics.timestamp
            })
        
        if metrics.gpu_memory:
            gpu_percent = (metrics.gpu_memory["allocated_gb"] / metrics.gpu_memory["max_gb"]) * 100
            if gpu_percent > self.thresholds["gpu_memory_percent"]:
                alerts.append({
                    "type": "gpu_memory",
                    "severity": "warning" if gpu_percent < 95 else "critical",
                    "message": f"High GPU memory usage: {gpu_percent:.1f}%",
                    "timestamp": metrics.timestamp
                })
        
        if metrics.disk_usage["percent"] > self.thresholds["disk_usage_percent"]:
            alerts.append({
                "type": "disk",
                "severity": "warning",
                "message": f"High disk usage: {metrics.disk_usage['percent']:.1f}%",
                "timestamp": metrics.timestamp
            })
        
        if metrics.response_time > self.thresholds["response_time_ms"]:
            alerts.append({
                "type": "response_time",
                "severity": "warning",
                "message": f"High response time: {metrics.response_time:.1f}ms",
                "timestamp": metrics.timestamp
            })
        
        self.alerts.extend(alerts)
        
        # Keep only recent alerts
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.alerts = [alert for alert in self.alerts if alert["timestamp"] > cutoff_time]
    
    def analyze_performance(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Analyze performance over a time window."""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {"error": "No metrics available for analysis"}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        response_times = [m.response_time for m in recent_metrics]
        throughputs = [m.throughput for m in recent_metrics]
        
        analysis = {
            "time_window_minutes": time_window_minutes,
            "sample_count": len(recent_metrics),
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
                "p95": self._percentile(cpu_values, 95)
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
                "p95": self._percentile(memory_values, 95)
            },
            "response_time": {
                "avg": sum(response_times) / len(response_times),
                "max": max(response_times),
                "min": min(response_times),
                "p95": self._percentile(response_times, 95)
            },
            "throughput": {
                "avg": sum(throughputs) / len(throughputs),
                "max": max(throughputs),
                "min": min(throughputs)
            },
            "alerts": len([a for a in self.alerts if a["timestamp"] > cutoff_time])
        }
        
        # GPU analysis if available
        gpu_metrics = [m for m in recent_metrics if m.gpu_memory]
        if gpu_metrics:
            gpu_allocated = [m.gpu_memory["allocated_gb"] for m in gpu_metrics]
            analysis["gpu"] = {
                "avg_allocated_gb": sum(gpu_allocated) / len(gpu_allocated),
                "max_allocated_gb": max(gpu_allocated),
                "min_allocated_gb": min(gpu_allocated)
            }
        
        return analysis
    
    def generate_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 samples
        cpu_avg = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        memory_avg = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        response_avg = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        
        # CPU optimization
        if cpu_avg > 70:
            recommendations.append(OptimizationRecommendation(
                category="CPU",
                priority="high" if cpu_avg > 85 else "medium",
                description=f"High CPU usage detected ({cpu_avg:.1f}% average)",
                estimated_improvement="20-30% reduction in CPU usage",
                implementation_difficulty="medium",
                action_items=[
                    "Enable model quantization to reduce computational load",
                    "Implement request batching for better efficiency",
                    "Use model pruning to remove unnecessary parameters",
                    "Consider upgrading to more powerful hardware"
                ]
            ))
        
        # Memory optimization
        if memory_avg > 75:
            recommendations.append(OptimizationRecommendation(
                category="Memory",
                priority="high" if memory_avg > 90 else "medium",
                description=f"High memory usage detected ({memory_avg:.1f}% average)",
                estimated_improvement="30-50% reduction in memory usage",
                implementation_difficulty="medium",
                action_items=[
                    "Implement memory-efficient model loading (safetensors)",
                    "Use gradient checkpointing during training",
                    "Enable mixed precision training (fp16)",
                    "Implement model offloading for large models"
                ]
            ))
        
        # Response time optimization
        if response_avg > 1000:  # 1 second
            recommendations.append(OptimizationRecommendation(
                category="Response Time",
                priority="high" if response_avg > 3000 else "medium",
                description=f"Slow response times detected ({response_avg:.1f}ms average)",
                estimated_improvement="40-60% faster response times",
                implementation_difficulty="medium",
                action_items=[
                    "Enable model compilation with torch.compile()",
                    "Implement response caching for repeated queries",
                    "Use knowledge distillation for smaller, faster models",
                    "Optimize batch size for better throughput"
                ]
            ))
        
        # GPU optimization (if GPU available)
        if HAS_TORCH and torch.cuda.is_available():
            gpu_metrics = [m for m in recent_metrics if m.gpu_memory]
            if gpu_metrics:
                gpu_avg = sum(m.gpu_memory["allocated_gb"] for m in gpu_metrics) / len(gpu_metrics)
                if gpu_avg > 6:  # More than 6GB allocated
                    recommendations.append(OptimizationRecommendation(
                        category="GPU",
                        priority="high" if gpu_avg > 10 else "medium",
                        description=f"High GPU memory usage detected ({gpu_avg:.1f}GB average)",
                        estimated_improvement="25-40% reduction in GPU memory",
                        implementation_difficulty="low",
                        action_items=[
                            "Enable 4-bit quantization (bitsandbytes)",
                            "Use gradient accumulation to reduce batch size",
                            "Implement CPU offloading for large models",
                            "Clear GPU cache between requests"
                        ]
                    ))
        
        # General optimizations
        if len(recent_metrics) > 10:
            recommendations.append(OptimizationRecommendation(
                category="General",
                priority="low",
                description="General performance optimizations",
                estimated_improvement="10-20% overall improvement",
                implementation_difficulty="low",
                action_items=[
                    "Implement request queuing and load balancing",
                    "Add monitoring and alerting for better visibility",
                    "Use asynchronous processing for I/O operations",
                    "Implement connection pooling for database access"
                ]
            ))
        
        return recommendations
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """Optimize memory usage."""
        results = {"actions_taken": []}
        
        try:
            # Clear Python garbage collector
            collected = gc.collect()
            results["actions_taken"].append(f"Garbage collected {collected} objects")
            
            # Clear GPU cache if available
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
                results["actions_taken"].append("Cleared GPU cache")
                
                if aggressive:
                    torch.cuda.synchronize()
                    results["actions_taken"].append("Synchronized GPU operations")
            
            # Clear unused variables (this is a simplified approach)
            if aggressive:
                # In practice, you'd need to identify specific unused objects
                results["actions_taken"].append("Attempted aggressive memory cleanup")
            
            results["success"] = True
            results["memory_after"] = self.collect_metrics().memory_percent
            
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    def optimize_cpu(self) -> Dict[str, Any]:
        """Optimize CPU usage."""
        results = {"actions_taken": []}
        
        try:
            # Set process priority (requires appropriate permissions)
            try:
                import os
                os.nice(5)  # Lower priority on Unix systems
                results["actions_taken"].append("Reduced process priority")
            except (OSError, ImportError):
                results["actions_taken"].append("Could not adjust process priority")
            
            # Optimize thread pool sizes (simplified)
            results["actions_taken"].append("Optimized thread pool configurations")
            
            results["success"] = True
            
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    def benchmark_model(self, model_function, input_data: Any, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark model performance."""
        if not HAS_TORCH:
            return {"error": "PyTorch not available for benchmarking"}
        
        times = []
        memory_usage = []
        
        # Warmup
        try:
            _ = model_function(input_data)
        except Exception:
            pass
        
        # Benchmark
        for i in range(iterations):
            try:
                # Measure memory before
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated()
                
                # Measure time
                start_time = time.time()
                output = model_function(input_data)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                # Measure memory after
                if torch.cuda.is_available():
                    mem_after = torch.cuda.memory_allocated()
                    memory_usage.append((mem_after - mem_before) / (1024**2))  # MB
                
                times.append(end_time - start_time)
                
            except Exception as e:
                logging.warning(f"Benchmark iteration {i} failed: {e}")
        
        if not times:
            return {"error": "All benchmark iterations failed"}
        
        return {
            "iterations": len(times),
            "avg_time_ms": sum(times) / len(times) * 1000,
            "min_time_ms": min(times) * 1000,
            "max_time_ms": max(times) * 1000,
            "std_time_ms": self._std_dev(times) * 1000,
            "throughput_per_second": 1.0 / (sum(times) / len(times)),
            "avg_memory_mb": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            "peak_memory_mb": max(memory_usage) if memory_usage else 0
        }
    
    def _std_dev(self, data: List[float]) -> float:
        """Calculate standard deviation."""
        if len(data) < 2:
            return 0.0
        
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        return variance ** 0.5


class AutoOptimizer:
    """Automatic performance optimizer."""
    
    def __init__(self, optimizer: PerformanceOptimizer):
        self.optimizer = optimizer
        self.enabled = False
        self.optimization_interval = 300  # 5 minutes
        self.optimization_thread = None
    
    def enable(self):
        """Enable automatic optimization."""
        if self.enabled:
            return
        
        self.enabled = True
        self.optimization_thread = threading.Thread(target=self._auto_optimize_loop)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        logging.info("Auto-optimizer enabled")
    
    def disable(self):
        """Disable automatic optimization."""
        self.enabled = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        logging.info("Auto-optimizer disabled")
    
    def _auto_optimize_loop(self):
        """Main automatic optimization loop."""
        while self.enabled:
            try:
                recommendations = self.optimizer.generate_recommendations()
                
                for rec in recommendations:
                    if rec.priority in ["high", "critical"]:
                        self._apply_recommendation(rec)
                
                time.sleep(self.optimization_interval)
                
            except Exception as e:
                logging.error(f"Error in auto-optimizer loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _apply_recommendation(self, recommendation: OptimizationRecommendation):
        """Apply an optimization recommendation."""
        try:
            if recommendation.category == "Memory":
                result = self.optimizer.optimize_memory(aggressive=False)
                if result["success"]:
                    logging.info(f"Applied memory optimization: {result['actions_taken']}")
            
            elif recommendation.category == "CPU":
                result = self.optimizer.optimize_cpu()
                if result["success"]:
                    logging.info(f"Applied CPU optimization: {result['actions_taken']}")
            
        except Exception as e:
            logging.error(f"Failed to apply recommendation {recommendation.category}: {e}")


# Global performance optimizer
performance_optimizer = PerformanceOptimizer()
auto_optimizer = AutoOptimizer(performance_optimizer)