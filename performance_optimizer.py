#!/usr/bin/env python3
"""
Performance Monitor and Optimizer for SloGPT Training

Real-time monitoring, optimization suggestions, and performance tuning.
"""

import time
import psutil
import torch
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import subprocess


class PerformanceMonitor:
    """Monitor training performance and suggest optimizations."""
    
    def __init__(self):
        self.metrics = {
            "start_time": None,
            "epoch_times": [],
            "memory_usage": [],
            "gpu_utilization": [],
            "disk_io": [],
            "cpu_usage": []
        }
        self.device = self._detect_device()
        
    def _detect_device(self) -> str:
        """Detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_memory_info(self) -> Dict:
        """Get memory usage info."""
        memory = psutil.virtual_memory()
        
        if self.device == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            gpu_reserved = torch.cuda.memory_reserved(0)
            
            return {
                "system_total_gb": memory.total / (1024**3),
                "system_used_gb": memory.used / (1024**3),
                "gpu_total_gb": gpu_memory / (1024**3),
                "gpu_allocated_gb": gpu_allocated / (1024**3),
                "gpu_reserved_gb": gpu_reserved / (1024**3)
            }
        else:
            return {
                "system_total_gb": memory.total / (1024**3),
                "system_used_gb": memory.used / (1024**3),
                "system_available_gb": memory.available / (1024**3)
            }
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        if self.device == "cuda":
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return float(result.stdout.strip())
            except:
                pass
        return 0.0
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.metrics["start_time"] = time.time()
        print(f"🔍 Performance monitoring started on {self.device} device")
    
    def record_metrics(self, epoch: Optional[int] = None):
        """Record current performance metrics."""
        current_time = time.time()
        
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_info = self._get_memory_info()
        disk_io = psutil.disk_io_counters()
        
        # GPU metrics
        gpu_util = self._get_gpu_utilization()
        
        # Store metrics
        metric_point = {
            "timestamp": current_time,
            "epoch": epoch,
            "cpu_percent": cpu_percent,
            "memory_info": memory_info,
            "gpu_utilization": gpu_util
        }
        
        if disk_io:
            metric_point["disk_read_mb"] = disk_io.read_bytes / (1024**2)
            metric_point["disk_write_mb"] = disk_io.write_bytes / (1024**2)
        
        self.metrics["memory_usage"].append(memory_info)
        self.metrics["cpu_usage"].append(cpu_percent)
        self.metrics["gpu_utilization"].append(gpu_util)
        
        return metric_point
    
    def analyze_performance(self) -> Dict:
        """Analyze collected metrics and provide recommendations."""
        if not self.metrics["memory_usage"]:
            return {"status": "no_data", "message": "No metrics recorded"}
        
        # Calculate averages
        avg_cpu = sum(self.metrics["cpu_usage"]) / len(self.metrics["cpu_usage"])
        avg_gpu = sum(self.metrics["gpu_utilization"]) / len(self.metrics["gpu_utilization"])
        
        # Memory analysis
        latest_memory = self.metrics["memory_usage"][-1]
        memory_pressure = latest_memory["system_used_gb"] / latest_memory["system_total_gb"]
        
        # Generate recommendations
        recommendations = []
        
        # CPU recommendations
        if avg_cpu > 80:
            recommendations.append({
                "type": "cpu",
                "severity": "high",
                "message": "High CPU usage detected. Consider reducing batch size or using more efficient data loading."
            })
        elif avg_cpu > 60:
            recommendations.append({
                "type": "cpu",
                "severity": "medium", 
                "message": "Moderate CPU usage. Monitor for performance bottlenecks."
            })
        
        # GPU recommendations
        if self.device == "cuda":
            if avg_gpu < 50:
                recommendations.append({
                    "type": "gpu",
                    "severity": "medium",
                    "message": "Low GPU utilization. Consider increasing batch size or model size."
                })
            elif avg_gpu > 90:
                recommendations.append({
                    "type": "gpu",
                    "severity": "high", 
                    "message": "Very high GPU usage. Consider reducing batch size to avoid OOM errors."
                })
        
        # Memory recommendations
        if memory_pressure > 0.9:
            recommendations.append({
                "type": "memory",
                "severity": "high",
                "message": "High memory pressure. Close other applications or use streaming mode."
            })
        elif memory_pressure > 0.7:
            recommendations.append({
                "type": "memory",
                "severity": "medium",
                "message": "Moderate memory usage. Monitor for memory leaks."
            })
        
        # Device-specific recommendations
        if self.device == "mps":
            recommendations.append({
                "type": "device",
                "severity": "info",
                "message": "MPS detected. Consider using --compile=False for better compatibility."
            })
        
        return {
            "status": "analyzed",
            "device": self.device,
            "avg_cpu": avg_cpu,
            "avg_gpu": avg_gpu,
            "memory_pressure": memory_pressure,
            "recommendations": recommendations
        }
    
    def save_report(self, output_path: str = "performance_report.json"):
        """Save performance report to file."""
        analysis = self.analyze_performance()
        report = {
            "timestamp": time.time(),
            "device": self.device,
            "metrics": self.metrics,
            "analysis": analysis
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Performance report saved to: {output_path}")
        return output_path


class TrainingOptimizer:
    """Optimize training parameters based on system capabilities."""
    
    def __init__(self):
        self.device_info = self._get_device_info()
        self.system_info = self._get_system_info()
    
    def _get_device_info(self) -> Dict:
        """Get detailed device information."""
        info = {"type": "cpu"}
        
        if torch.cuda.is_available():
            info = {
                "type": "cuda",
                "count": torch.cuda.device_count(),
                "devices": []
            }
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "id": i,
                    "name": props.name,
                    "memory_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}"
                })
        
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info = {
                "type": "mps",
                "name": "Apple Silicon GPU",
                "memory_gb": psutil.virtual_memory().total / (1024**3)  # Approximate
            }
        
        return info
    
    def _get_system_info(self) -> Dict:
        """Get system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "disk_space_gb": psutil.disk_usage('/').free / (1024**3)
        }
    
    def suggest_training_config(self, dataset_size_mb: float = None) -> Dict:
        """Suggest optimal training configuration."""
        config = {
            "device": self.device_info["type"],
            "batch_size": 32,
            "num_workers": 4,
            "compile": True,
            "mixed_precision": False,
            "gradient_accumulation": 1
        }
        
        # Device-specific optimizations
        if self.device_info["type"] == "cuda":
            # Calculate optimal batch size based on GPU memory
            if self.device_info["devices"]:
                gpu_memory = self.device_info["devices"][0]["memory_gb"]
                
                if gpu_memory >= 24:  # RTX 3090/4090, A100
                    config["batch_size"] = 64
                    config["mixed_precision"] = True
                elif gpu_memory >= 16:  # RTX 3080, V100
                    config["batch_size"] = 48
                    config["mixed_precision"] = True
                elif gpu_memory >= 12:  # RTX 3060/4070
                    config["batch_size"] = 32
                    config["mixed_precision"] = True
                elif gpu_memory >= 8:  # RTX 3050/3060
                    config["batch_size"] = 24
                else:  # < 8GB
                    config["batch_size"] = 16
                    config["gradient_accumulation"] = 2
                
                config["compile"] = True  # CUDA supports compilation
        
        elif self.device_info["type"] == "mps":
            # MPS-specific optimizations
            config["batch_size"] = 16
            config["compile"] = False  # MPS doesn't support torch.compile
            config["mixed_precision"] = True  # MPS supports mixed precision
        
        else:  # CPU
            config["batch_size"] = 8
            config["num_workers"] = min(8, self.system_info["cpu_count"])
            config["compile"] = False
        
        # Adjust for dataset size
        if dataset_size_mb:
            if dataset_size_mb > 1000:  # > 1GB
                config["use_streaming"] = True
                config["num_workers"] = min(4, config["num_workers"])
        
        # Adjust for available system memory
        if self.system_info["memory_gb"] < 16:
            config["batch_size"] = min(config["batch_size"], 16)
            config["num_workers"] = min(config["num_workers"], 2)
        
        return config
    
    def generate_command(self, dataset_name: str, dataset_size_mb: float = None) -> str:
        """Generate optimized training command."""
        config = self.suggest_training_config(dataset_size_mb)
        
        cmd_parts = ["python3", "train_simple.py"]
        
        if dataset_name:
            cmd_parts.extend(["--dataset", dataset_name])
        
        # Add device-specific flags
        if config["device"] == "cuda":
            cmd_parts.extend(["--device", "cuda"])
            if config["mixed_precision"]:
                cmd_parts.append("--mixed_precision")
        elif config["device"] == "mps":
            cmd_parts.extend(["--device", "mps", "--compile=False"])
        
        # Add performance flags
        cmd_parts.extend([
            f"--batch_size={config['batch_size']}",
            f"--num_workers={config['num_workers']}"
        ])
        
        if config.get("gradient_accumulation", 1) > 1:
            cmd_parts.extend(["--gradient_accumulation", str(config["gradient_accumulation"])])
        
        return " ".join(cmd_parts)


def main():
    parser = argparse.ArgumentParser(description="Performance monitor and optimizer for SloGPT")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor training performance')
    monitor_parser.add_argument('--output', default='performance_report.json', help='Output report file')
    monitor_parser.add_argument('--interval', type=int, default=30, help='Monitoring interval (seconds)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze system and suggest optimizations')
    analyze_parser.add_argument('--dataset', help='Dataset name for optimization')
    analyze_parser.add_argument('--dataset_size_mb', type=float, help='Dataset size in MB')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Generate optimized training command')
    optimize_parser.add_argument('dataset', help='Dataset name')
    optimize_parser.add_argument('--dataset_size_mb', type=float, help='Dataset size in MB')
    
    args = parser.parse_args()
    
    if args.command == 'monitor':
        print("🔍 Starting performance monitoring...")
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        try:
            while True:
                metrics = monitor.record_metrics()
                print(f"📊 CPU: {metrics['cpu_percent']:.1f}% | GPU: {metrics['gpu_utilization']:.1f}% | Memory: {metrics['memory_info']['system_used_gb']:.1f}GB")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            monitor.save_report(args.output)
            print("\n✅ Monitoring completed")
    
    elif args.command == 'analyze':
        print("🔬 Analyzing system capabilities...")
        optimizer = TrainingOptimizer()
        config = optimizer.suggest_training_config(args.dataset_size_mb)
        
        print(f"\n📋 System Analysis:")
        print(f"  Device: {optimizer.device_info['type']}")
        print(f"  CPU Cores: {optimizer.system_info['cpu_count']}")
        print(f"  System Memory: {optimizer.system_info['memory_gb']:.1f}GB")
        
        print(f"\n⚡ Recommended Configuration:")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Num Workers: {config['num_workers']}")
        print(f"  Mixed Precision: {config['mixed_precision']}")
        print(f"  Compile: {config['compile']}")
        
        if args.dataset:
            cmd = optimizer.generate_command(args.dataset, args.dataset_size_mb)
            print(f"\n🚀 Optimized Command:")
            print(f"  {cmd}")
    
    elif args.command == 'optimize':
        optimizer = TrainingOptimizer()
        cmd = optimizer.generate_command(args.dataset, args.dataset_size_mb)
        print(f"🚀 Optimized training command:")
        print(f"{cmd}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()