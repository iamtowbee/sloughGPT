#!/usr/bin/env python3
"""
SloughGPT Benchmark Suite
Comprehensive benchmarking for model performance and comparison
"""

import time
import torch
import psutil
import json
import platform
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from sloughgpt.config import ModelConfig
from sloughgpt.neural_network import SloughGPT
from sloughgpt.optimizations import OptimizedSloughGPT, create_optimized_model
from sloughgpt.model_zoo import get_model_zoo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Result of a single benchmark test"""
    test_name: str
    model_type: str
    configuration: str
    results: Dict[str, float]
    system_info: Dict[str, Any]
    timestamp: str
    performance_metrics: Dict[str, Any]

@dataclass 
class SystemInfo:
    """System information for benchmarking"""
    cpu_count: int
    cpu_freq: float
    memory_total_gb: float
    gpu_available: bool
    gpu_name: str = "N/A"
    gpu_memory_gb: float = 0.0
    platform: str = ""
    python_version: str = ""
    pytorch_version: str = ""

class BenchmarkSuite:
    """Comprehensive benchmarking suite for SloughGPT models"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.results = []
        
    def _get_system_info(self) -> SystemInfo:
        """Get system information"""
        info = SystemInfo(
            cpu_count=psutil.cpu_count(),
            cpu_freq=psutil.cpu_freq().current if psutil.cpu_freq() else 0.0,
            memory_total_gb=psutil.virtual_memory().total / (1024**3),
            gpu_available=torch.cuda.is_available(),
            platform=platform.system(),
            python_version=platform.python_version(),
            pytorch_version=torch.__version__
        )
        
        if torch.cuda.is_available():
            info.gpu_name = torch.cuda.get_device_name(0)
            info.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info
    
    def benchmark_model_inference(self, 
                               model: Union[SloughGPT, OptimizedSloughGPT],
                               model_name: str,
                               sequence_lengths: List[int] = [10, 50, 100, 500],
                               batch_sizes: List[int] = [1, 4, 8],
                               num_runs: int = 10) -> BenchmarkResult:
        """Benchmark model inference performance"""
        print(f"🚀 Benchmarking {model_name}...")
        
        results = {}
        performance_metrics = {}
        
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                test_name = f"seq_{seq_len}_batch_{batch_size}"
                
                # Warmup
                for _ in range(3):
                    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
                    with torch.no_grad():
                        _ = model(input_ids)
                
                # Actual benchmarking
                times = []
                memory_usage = []
                
                for _ in range(num_runs):
                    # Reset memory tracking
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    
                    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
                    
                    start_time = time.perf_counter()
                    with torch.no_grad():
                        output = model(input_ids)
                    end_time = time.perf_counter()
                    
                    # Record metrics
                    times.append(end_time - start_time)
                    
                    if torch.cuda.is_available():
                        memory_usage.append(torch.cuda.max_memory_allocated() / (1024**2))  # MB
                    else:
                        memory_usage.append(psutil.virtual_memory().used / (1024**2))  # MB
                
                # Calculate statistics
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                tokens_per_second = (batch_size * seq_len) / avg_time
                
                avg_memory = sum(memory_usage) / len(memory_usage)
                max_memory = max(memory_usage)
                
                results[test_name] = {
                    "avg_time_seconds": avg_time,
                    "min_time_seconds": min_time,
                    "max_time_seconds": max_time,
                    "tokens_per_second": tokens_per_second,
                    "avg_memory_mb": avg_memory,
                    "max_memory_mb": max_memory
                }
                
                print(f"  ✅ {test_name}: {tokens_per_second:.1f} tokens/sec, {avg_time*1000:.1f}ms avg")
        
        # Calculate performance summary
        all_throughputs = [result["tokens_per_second"] for result in results.values()]
        performance_metrics = {
            "avg_throughput_tokens_per_sec": sum(all_throughputs) / len(all_throughputs),
            "max_throughput_tokens_per_sec": max(all_throughputs),
            "min_throughput_tokens_per_sec": min(all_throughputs),
            "total_parameters": model.count_parameters(),
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2),
            "memory_efficiency": "high" if max(m.values(), key=lambda x: x.get("max_memory_mb", 0)) < 2000 else "medium"
        }
        
        return BenchmarkResult(
            test_name="inference_benchmark",
            model_type=model_name,
            configuration=f"seq={max(sequence_lengths)}, batch={max(batch_sizes)}",
            results=results,
            system_info=asdict(self.system_info),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            performance_metrics=performance_metrics
        )
    
    def benchmark_model_generation(self,
                              model: Union[SloughGPT, OptimizedSloughGPT],
                              model_name: str,
                              generation_lengths: List[int] = [20, 50, 100],
                              temperatures: List[float] = [0.7, 1.0, 1.3],
                              num_runs: int = 5) -> BenchmarkResult:
        """Benchmark model text generation performance"""
        print(f"✍ Benchmarking {model_name} generation...")
        
        results = {}
        performance_metrics = {}
        
        for gen_len in generation_lengths:
            for temp in temperatures:
                test_name = f"gen_{gen_len}_temp_{temp}"
                
                times = []
                memory_usage = []
                
                for _ in range(num_runs):
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    
                    input_ids = torch.randint(0, model.config.vocab_size, (1, 10))
                    
                    start_time = time.perf_counter()
                    with torch.no_grad():
                        generated_ids = model.generate(
                            input_ids=input_ids,
                            max_length=gen_len,
                            temperature=temp,
                            do_sample=True
                        )
                    end_time = time.perf_counter()
                    
                    times.append(end_time - start_time)
                    
                    if torch.cuda.is_available():
                        memory_usage.append(torch.cuda.max_memory_allocated() / (1024**2))
                    else:
                        memory_usage.append(psutil.virtual_memory().used / (1024**2))
                
                # Calculate statistics
                avg_time = sum(times) / len(times)
                tokens_generated = gen_len - 10  # Generated tokens (excluding input)
                tokens_per_second = tokens_generated / avg_time if avg_time > 0 else 0
                
                results[test_name] = {
                    "avg_time_seconds": avg_time,
                    "tokens_per_second": tokens_per_second,
                    "avg_memory_mb": sum(memory_usage) / len(memory_usage),
                    "max_memory_mb": max(memory_usage)
                }
                
                print(f"  ✅ {test_name}: {tokens_per_second:.1f} gen tokens/sec")
        
        # Calculate performance summary
        all_throughputs = [result["tokens_per_second"] for result in results.values()]
        performance_metrics = {
            "avg_generation_tokens_per_sec": sum(all_throughputs) / len(all_throughputs),
            "max_generation_tokens_per_sec": max(all_throughputs),
            "min_generation_tokens_per_sec": min(all_throughputs),
            "temperature_sensitivity": max(all_throughputs) / min(all_throughputs) if min(all_throughputs) > 0 else 1.0
        }
        
        return BenchmarkResult(
            test_name="generation_benchmark",
            model_type=model_name,
            configuration=f"gen_len={max(generation_lengths)}, temp={temperatures}",
            results=results,
            system_info=asdict(self.system_info),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            performance_metrics=performance_metrics
        )
    
    def benchmark_optimizations(self, 
                            base_config: ModelConfig,
                            optimizations: List[str] = ["quantization", "compilation", "mixed_precision"]) -> BenchmarkResult:
        """Benchmark different optimization techniques"""
        print(f"⚡ Benchmarking optimizations...")
        
        # Create base model
        base_model = SloughGPT(base_config)
        base_benchmark = self.benchmark_model_inference(base_model, "baseline")
        base_throughput = base_benchmark.performance_metrics["avg_throughput_tokens_per_sec"]
        
        results = {}
        optimization_stats = {}
        
        for optimization in optimizations:
            print(f"  🧪 Testing {optimization}...")
            
            try:
                # Create optimized model
                if optimization == "quantization":
                    optimized_model = SloughGPT(base_config)
                    optimized_model.enable_quantization()
                    model_instance = optimized_model.quantized_model
                elif optimization == "compilation":
                    optimized_model = create_optimized_model(base_config, enable_compilation=True)
                    model_instance = optimized_model.get_model_for_inference()
                elif optimization == "mixed_precision":
                    optimized_model = create_optimized_model(base_config, enable_mixed_precision=True)
                    model_instance = optimized_model.get_model_for_inference()
                else:
                    continue
                
                if model_instance is None:
                    print(f"    ❌ {optimization} failed to initialize")
                    continue
                
                # Benchmark optimized model
                opt_benchmark = self.benchmark_model_inference(
                    model_instance, 
                    f"optimized_{optimization}",
                    sequence_lengths=[50, 100],
                    batch_sizes=[1, 4],
                    num_runs=5
                )
                
                opt_throughput = opt_benchmark.performance_metrics["avg_throughput_tokens_per_sec"]
                speedup = opt_throughput / base_throughput if base_throughput > 0 else 1.0
                
                # Calculate memory reduction
                base_memory = base_benchmark.results["seq_50_batch_1"]["avg_memory_mb"]
                opt_memory = opt_benchmark.results.get("seq_50_batch_1", {}).get("avg_memory_mb", base_memory)
                memory_reduction = (base_memory - opt_memory) / base_memory * 100 if base_memory > 0 else 0
                
                optimization_stats[optimization] = {
                    "speedup": speedup,
                    "memory_reduction_percent": memory_reduction,
                    "throughput_tokens_per_sec": opt_throughput
                }
                
                print(f"    ✅ {optimization}: {speedup:.2f}x speedup, {memory_reduction:.1f}% memory reduction")
                
            except Exception as e:
                print(f"    ❌ {optimization} failed: {e}")
                optimization_stats[optimization] = {"error": str(e)}
        
        return BenchmarkResult(
            test_name="optimization_benchmark",
            model_type="optimization_comparison",
            configuration=", ".join(optimizations),
            results=optimization_stats,
            system_info=asdict(self.system_info),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            performance_metrics=optimization_stats
        )
    
    def run_full_benchmark(self, 
                       configs: Optional[List[ModelConfig]] = None,
                       output_file: str = "benchmark_results.json") -> List[BenchmarkResult]:
        """Run comprehensive benchmark suite"""
        print("🏁 Starting Full SloughGPT Benchmark Suite")
        print("=" * 60)
        
        if configs is None:
            configs = [
                ModelConfig(d_model=256, n_heads=4, n_layers=4),   # Small
                ModelConfig(d_model=512, n_heads=8, n_layers=6),   # Medium
                ModelConfig(d_model=1024, n_heads=16, n_layers=12), # Large
            ]
        
        results = []
        
        # Benchmark each configuration
        for i, config in enumerate(configs):
            size_name = ["small", "medium", "large"][i]
            print(f"\n📊 Benchmarking {size_name} model...")
            
            try:
                # Create models
                base_model = SloughGPT(config)
                optimized_model = create_optimized_model(config, enable_quantization=True)
                
                # Inference benchmarks
                base_inf = self.benchmark_model_inference(base_model, f"{size_name}_baseline")
                opt_inf = self.benchmark_model_inference(optimized_model, f"{size_name}_optimized")
                
                # Generation benchmarks
                base_gen = self.benchmark_model_generation(base_model, f"{size_name}_baseline_gen")
                opt_gen = self.benchmark_model_generation(optimized_model, f"{size_name}_optimized_gen")
                
                # Optimization benchmarks
                opt_bench = self.benchmark_optimizations(config)
                
                results.extend([base_inf, opt_inf, base_gen, opt_gen, opt_bench])
                
            except Exception as e:
                logger.error(f"Failed to benchmark {size_name} model: {e}")
        
        # Save results
        self.save_results(results, output_file)
        
        # Generate summary
        self.generate_summary(results)
        
        return results
    
    def save_results(self, results: List[BenchmarkResult], output_file: str):
        """Save benchmark results to JSON file"""
        output_path = Path(output_file)
        
        try:
            with open(output_path, 'w') as f:
                json.dump([asdict(result) for result in results], f, indent=2, default=str)
            
            print(f"\n💾 Results saved to {output_path.absolute()}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def generate_summary(self, results: List[BenchmarkResult]):
        """Generate benchmark summary report"""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY REPORT")
        print("=" * 60)
        
        # System info
        sys_info = results[0].system_info if results else {}
        print("\n💻 SYSTEM INFORMATION:")
        print(f"  Platform: {sys_info.get('platform', 'N/A')}")
        print(f"  CPU: {sys_info.get('cpu_count', 'N/A')} cores @ {sys_info.get('cpu_freq', 0):.1f}GHz")
        print(f"  Memory: {sys_info.get('memory_total_gb', 0):.1f}GB")
        print(f"  GPU: {sys_info.get('gpu_name', 'N/A')}")
        if sys_info.get('gpu_available', False):
            print(f"  GPU Memory: {sys_info.get('gpu_memory_gb', 0):.1f}GB")
        print(f"  Python: {sys_info.get('python_version', 'N/A')}")
        print(f"  PyTorch: {sys_info.get('pytorch_version', 'N/A')}")
        
        # Performance summary
        print("\n🚀 PERFORMANCE SUMMARY:")
        
        # Group results by model type
        inference_results = {}
        generation_results = {}
        
        for result in results:
            if result.test_name == "inference_benchmark":
                inference_results[result.model_type] = result
            elif result.test_name == "generation_benchmark":
                generation_results[result.model_type] = result
        
        # Inference performance table
        print("\n📈 INFERENCE PERFORMANCE:")
        print(f"{'Model Type':<15} {'Throughput (tokens/sec)':<20} {'Memory Usage (MB)':<15}")
        print("-" * 60)
        
        for model_type, result in inference_results.items():
            throughput = result.performance_metrics.get("avg_throughput_tokens_per_sec", 0)
            memory = result.performance_metrics.get("model_size_mb", 0)
            print(f"{model_type:<15} {throughput:>15.1f} {memory:>15.1f}")
        
        # Optimization impact
        print("\n⚡ OPTIMIZATION IMPACT:")
        opt_results = [r for r in results if r.test_name == "optimization_benchmark"]
        if opt_results:
            opt_result = opt_results[0]
            print(f"{'Optimization':<15} {'Speedup':<10} {'Memory Reduction':<15}")
            print("-" * 50)
            
            for opt, stats in opt_result.results.items():
                if isinstance(stats, dict):
                    speedup = stats.get("speedup", 0)
                    memory_red = stats.get("memory_reduction_percent", 0)
                    print(f"{opt:<15} {speedup:>10.2f}x {memory_red:>15.1f}%")
        
        print(f"\n📅 Benchmark completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main benchmark execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SloughGPT Benchmark Suite")
    parser.add_argument("--output", "-o", default="benchmark_results.json",
                       help="Output JSON file for results")
    parser.add_argument("--models", "-m", nargs='+', 
                       choices=["small", "medium", "large", "all"],
                       default=["small", "medium"],
                       help="Models to benchmark")
    parser.add_argument("--optimizations", action="store_true",
                       help="Include optimization benchmarks")
    
    args = parser.parse_args()
    
    # Create benchmark suite
    suite = BenchmarkSuite()
    
    # Determine models to test
    configs = []
    if "small" in args.models or "all" in args.models:
        configs.append(ModelConfig(d_model=256, n_heads=4, n_layers=4))
    if "medium" in args.models or "all" in args.models:
        configs.append(ModelConfig(d_model=512, n_heads=8, n_layers=6))
    if "large" in args.models or "all" in args.models:
        configs.append(ModelConfig(d_model=1024, n_heads=16, n_layers=12))
    
    # Run benchmarks
    results = suite.run_full_benchmark(configs, args.output)
    
    print(f"\n🎉 Benchmark completed! {len(results)} tests executed.")

if __name__ == "__main__":
    main()