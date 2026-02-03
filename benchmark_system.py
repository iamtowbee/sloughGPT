#!/usr/bin/env python3
"""
Automated Benchmarking System for SloGPT
Comprehensive performance testing and analysis.
"""

import os
import json
import time
import psutil
import torch
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_mb: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    accuracy: Optional[float] = None
    parameters: Dict[str, Any] = None


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.system_info = self._get_system_info()
        self.baseline_metrics = {}
        
    def _get_system_info(self) -> Dict:
        """Collect system information."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu_info': []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info['gpu_info'].append({
                    'id': i,
                    'name': props.name,
                    'memory_gb': props.total_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}"
                })
        
        return info
    
    def benchmark_dataset_loading(self, dataset_name: str) -> BenchmarkResult:
        """Benchmark dataset loading performance."""
        print(f"📊 Benchmarking dataset loading: {dataset_name}")
        
        # Monitor system resources
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**2)
        start_cpu = psutil.cpu_percent()
        
        try:
            # Load dataset
            from simple_gpt_model import load_dataset
            train_data, val_data, meta = load_dataset(dataset_name)
            
            # Calculate metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024**2)
            end_cpu = psutil.cpu_percent()
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            cpu_usage = (end_cpu + start_cpu) / 2
            
            total_tokens = len(train_data) + len(val_data)
            throughput = total_tokens / execution_time if execution_time > 0 else 0
            
            result = BenchmarkResult(
                test_name=f"dataset_loading_{dataset_name}",
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                throughput_tokens_per_sec=throughput,
                parameters={
                    'train_tokens': len(train_data),
                    'val_tokens': len(val_data),
                    'vocab_size': meta['vocab_size']
                }
            )
            
            print(f"   ✅ Loaded {total_tokens:,} tokens in {execution_time:.2f}s")
            print(f"   📊 Throughput: {throughput:.0f} tokens/sec")
            print(f"   💾 Memory usage: {memory_usage:.1f} MB")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Dataset loading failed: {e}")
            return BenchmarkResult(
                test_name=f"dataset_loading_{dataset_name}_error",
                execution_time=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                parameters={'error': str(e)}
            )
    
    def benchmark_model_inference(self, model_path: str, dataset_name: str, num_runs: int = 10) -> List[BenchmarkResult]:
        """Benchmark model inference performance."""
        print(f"🧠 Benchmarking model inference: {model_path}")
        
        results = []
        
        try:
            # Load model and dataset
            from simple_gpt_model import GPT, load_dataset
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                config = checkpoint.get('config', {})
                vocab_size = config.get('vocab_size', 50000)
            else:
                state_dict = checkpoint
                vocab_size = 50000
            
            # Create model
            model = GPT(vocab_size)
            model.load_state_dict(state_dict)
            model.eval()
            
            # Load test data
            _, val_data, meta = load_dataset(dataset_name)
            
            # Prepare test sequences
            sequence_length = 128
            num_sequences = min(100, len(val_data) // sequence_length)
            
            test_sequences = []
            for i in range(num_sequences):
                start_idx = (i * sequence_length) % len(val_data)
                end_idx = start_idx + sequence_length
                if end_idx > len(val_data):
                    end_idx = len(val_data)
                    start_idx = max(0, end_idx - sequence_length)
                
                sequence = val_data[start_idx:end_idx]
                test_sequences.append(sequence)
            
            print(f"   Prepared {num_sequences} test sequences of length {sequence_length}")
            
            # Benchmark different configurations
            configs = [
                {'name': 'cpu_fp32', 'device': 'cpu', 'dtype': torch.float32},
                {'name': 'cpu_fp16', 'device': 'cpu', 'dtype': torch.float16},
            ]
            
            if torch.cuda.is_available():
                configs.extend([
                    {'name': 'gpu_fp32', 'device': 'cuda', 'dtype': torch.float32},
                    {'name': 'gpu_fp16', 'device': 'cuda', 'dtype': torch.float16},
                ])
            
            for config in configs:
                print(f"   Testing {config['name']}...")
                
                # Move model to device
                device = torch.device(config['device'])
                model = model.to(device)
                model = model.type(config['dtype'])
                
                # Warm up
                with torch.no_grad():
                    for _ in range(3):
                        test_seq = torch.from_numpy(test_sequences[0]).unsqueeze(0).to(device)
                        _ = model(test_seq)
                
                # Benchmark inference
                inference_times = []
                memory_usage = []
                
                for run in range(num_runs):
                    # Monitor resources
                    start_time = time.time()
                    start_memory = psutil.Process().memory_info().rss / (1024**2)
                    
                    # Run inference
                    with torch.no_grad():
                        for sequence in test_sequences[:10]:  # Test subset for speed
                            test_seq = torch.from_numpy(sequence).unsqueeze(0).to(device)
                            _ = model(test_seq)
                    
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / (1024**2)
                    
                    inference_times.append(end_time - start_time)
                    memory_usage.append(end_memory - start_memory)
                
                # Calculate metrics
                avg_time = np.mean(inference_times)
                std_time = np.std(inference_times)
                avg_memory = np.mean(memory_usage)
                
                total_tokens = sum(len(seq) for seq in test_sequences[:10])
                throughput = total_tokens / avg_time if avg_time > 0 else 0
                
                result = BenchmarkResult(
                    test_name=f"inference_{config['name']}",
                    execution_time=avg_time,
                    memory_usage_mb=avg_memory,
                    cpu_usage_percent=psutil.cpu_percent(),
                    throughput_tokens_per_sec=throughput,
                    parameters={
                        'std_time': std_time,
                        'num_runs': num_runs,
                        'batch_size': 1,
                        'sequence_length': sequence_length,
                        'device': config['device'],
                        'dtype': str(config['dtype'])
                    }
                )
                
                results.append(result)
                
                print(f"     ⚡ Time: {avg_time:.4f}s ± {std_time:.4f}s")
                print(f"     🚀 Throughput: {throughput:.1f} tokens/sec")
                print(f"     💾 Memory: {avg_memory:.1f} MB")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Inference benchmark failed: {e}")
            return [BenchmarkResult(
                test_name=f"inference_error",
                execution_time=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                parameters={'error': str(e)}
            )]
    
    def benchmark_training_speed(self, dataset_name: str, model_config: Dict) -> BenchmarkResult:
        """Benchmark training speed with different configurations."""
        print(f"🏋 Benchmarking training speed: {dataset_name}")
        
        try:
            from simple_gpt_model import GPT, load_dataset
            import torch.optim as optim
            
            # Load dataset
            train_data, _, meta = load_dataset(dataset_name)
            
            # Create small model for benchmarking
            model = GPT(
                vocab_size=meta['vocab_size'],
                n_embed=model_config.get('n_embed', 128),
                n_layer=model_config.get('n_layer', 2),
                n_head=model_config.get('n_head', 4)
            )
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3)
            
            # Training benchmark
            batch_size = model_config.get('batch_size', 8)
            sequence_length = model_config.get('sequence_length', 64)
            num_batches = 50  # Small number for benchmarking
            
            print(f"   Training {num_batches} batches of size {batch_size}")
            
            # Monitor resources
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024**2)
            
            model.train()
            total_loss = 0
            
            for batch_idx in range(num_batches):
                # Get batch
                start_idx = (batch_idx * batch_size * sequence_length) % len(train_data)
                end_idx = min(start_idx + batch_size * sequence_length, len(train_data))
                
                if end_idx - start_idx < batch_size * sequence_length:
                    continue
                
                batch_data = train_data[start_idx:end_idx]
                if len(batch_data) < 2:
                    continue
                
                batch_x = torch.from_numpy(batch_data[:-1].astype(np.int64)).reshape(batch_size, -1).to(device)
                batch_y = torch.from_numpy(batch_data[1:].astype(np.int64)).reshape(batch_size, -1).to(device)
                
                # Training step
                optimizer.zero_grad()
                logits, loss = model(batch_x, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"     Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024**2)
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            avg_loss = total_loss / num_batches
            tokens_per_sec = (num_batches * batch_size * sequence_length) / execution_time
            
            result = BenchmarkResult(
                test_name="training_speed",
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=psutil.cpu_percent(),
                throughput_tokens_per_sec=tokens_per_sec,
                accuracy=avg_loss,
                parameters={
                    'model_config': model_config,
                    'num_batches': num_batches,
                    'batch_size': batch_size,
                    'sequence_length': sequence_length,
                    'final_loss': avg_loss
                }
            )
            
            print(f"   ✅ Training completed in {execution_time:.2f}s")
            print(f"   🚀 Throughput: {tokens_per_sec:.1f} tokens/sec")
            print(f"   📊 Final loss: {avg_loss:.4f}")
            print(f"   💾 Memory usage: {memory_usage:.1f} MB")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Training benchmark failed: {e}")
            return BenchmarkResult(
                test_name="training_speed_error",
                execution_time=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                parameters={'error': str(e)}
            )
    
    def run_full_benchmark_suite(self, dataset_name: str = "distributed_test") -> Dict:
        """Run comprehensive benchmark suite."""
        print("🔬 Starting Full Benchmark Suite")
        print("=" * 50)
        
        # Create test dataset if needed
        try:
            from create_dataset_fixed import create_dataset
            
            test_text = """
            The quick brown fox jumps over the lazy dog.
            Pack my box with five dozen liquor jugs.
            How vexingly quick daft zebras jump!
            The five boxing wizards jump quickly.
            Sphinx of black quartz, judge my vow.
            """ * 20  # Repeat for more data
            
            result = create_dataset(dataset_name, test_text)
            if not result or not result.get('success'):
                print(f"❌ Failed to create benchmark dataset: {dataset_name}")
                return {}
                
        except Exception as e:
            print(f"❌ Error creating benchmark dataset: {e}")
            return {}
        
        all_results = []
        
        # Benchmark 1: Dataset Loading
        print("\n📊 Phase 1: Dataset Loading")
        print("-" * 30)
        dataset_result = self.benchmark_dataset_loading(dataset_name)
        all_results.append(dataset_result)
        
        # Create a simple model for inference benchmarking
        print("\n🏗️ Creating Benchmark Model")
        model_config = {
            'n_embed': 128,
            'n_layer': 2,
            'n_head': 4,
            'batch_size': 8,
            'sequence_length': 64
        }
        
        try:
            from simple_gpt_model import GPT, load_dataset
            _, _, meta = load_dataset(dataset_name)
            
            model = GPT(vocab_size=meta['vocab_size'], **{k: v for k, v in model_config.items() if k in ['n_embed', 'n_layer', 'n_head']})
            
            # Save temporary model
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / "benchmark_model.pt"
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {**model_config, 'vocab_size': meta['vocab_size']}
            }, model_path)
            
            # Benchmark 2: Model Inference
            print(f"\n🧠 Phase 2: Model Inference")
            print("-" * 30)
            inference_results = self.benchmark_model_inference(str(model_path), dataset_name, num_runs=5)
            all_results.extend(inference_results)
            
            # Benchmark 3: Training Speed
            print(f"\n🏋 Phase 3: Training Speed")
            print("-" * 30)
            training_result = self.benchmark_training_speed(dataset_name, model_config)
            all_results.append(training_result)
            
        except Exception as e:
            print(f"❌ Benchmarking failed: {e}")
        
        # Generate comprehensive report
        print(f"\n📊 Generating Comprehensive Report")
        print("-" * 30)
        report = self._generate_comprehensive_report(all_results)
        
        # Save report
        report_path = self.output_dir / f"benchmark_report_{dataset_name}_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate visualizations
        self._generate_benchmark_visualizations(all_results, dataset_name)
        
        print(f"✅ Full benchmark suite completed!")
        print(f"📁 Report saved: {report_path}")
        print(f"📈 Visualizations saved in: {self.output_dir}")
        
        return report
    
    def _generate_comprehensive_report(self, results: List[BenchmarkResult]) -> Dict:
        """Generate comprehensive benchmark report."""
        
        report = {
            'system_info': self.system_info,
            'timestamp': time.time(),
            'benchmark_results': [],
            'summary': {},
            'recommendations': []
        }
        
        # Process results
        for result in results:
            if result.parameters and 'error' not in result.parameters:
                report['benchmark_results'].append({
                    'test_name': result.test_name,
                    'execution_time': result.execution_time,
                    'memory_usage_mb': result.memory_usage_mb,
                    'cpu_usage_percent': result.cpu_usage_percent,
                    'gpu_usage_mb': result.gpu_usage_mb,
                    'throughput_tokens_per_sec': result.throughput_tokens_per_sec,
                    'accuracy': result.accuracy,
                    'parameters': result.parameters
                })
        
        # Generate summary statistics
        inference_results = [r for r in results if 'inference' in r.test_name and 'error' not in r.test_name]
        if inference_results:
            cpu_times = [r.execution_time for r in inference_results if 'cpu' in r.test_name]
            gpu_times = [r.execution_time for r in inference_results if 'gpu' in r.test_name]
            
            report['summary'] = {
                'fastest_cpu_inference': min(cpu_times) if cpu_times else 0,
                'fastest_gpu_inference': min(gpu_times) if gpu_times else 0,
                'average_throughput': np.mean([r.throughput_tokens_per_sec for r in results if r.throughput_tokens_per_sec > 0]),
                'total_tests': len([r for r in results if 'error' not in str(r.parameters)])
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(results)
        
        return report
    
    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Analyze inference performance
        inference_results = [r for r in results if 'inference' in r.test_name and 'error' not in r.test_name]
        if inference_results:
            avg_throughput = np.mean([r.throughput_tokens_per_sec for r in inference_results])
            
            if avg_throughput < 1000:
                recommendations.append("Consider model optimization for faster inference")
            if avg_throughput < 500:
                recommendations.append("Use quantization (INT8/FP16) to improve performance")
            if avg_throughput < 200:
                recommendations.append("Consider model pruning to reduce complexity")
        
        # Analyze memory usage
        memory_usage = [r.memory_usage_mb for r in results if r.memory_usage_mb > 0]
        if memory_usage:
            avg_memory = np.mean(memory_usage)
            if avg_memory > 1000:  # > 1GB
                recommendations.append("Consider reducing model size or batch size")
            if avg_memory > 2000:  # > 2GB
                recommendations.append("Use gradient accumulation with smaller batches")
        
        # GPU recommendations
        if self.system_info['cuda_available']:
            recommendations.append("GPU detected - use GPU for training and inference")
            recommendations.append("Consider multi-GPU training for large models")
        
        return recommendations
    
    def _generate_benchmark_visualizations(self, results: List[BenchmarkResult], dataset_name: str):
        """Generate benchmark visualization plots."""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Performance Benchmark Results - {dataset_name}', fontsize=16, fontweight='bold')
            
            # Filter valid results
            valid_results = [r for r in results if 'error' not in str(r.parameters)]
            
            if valid_results:
                # Plot 1: Execution Time Comparison
                test_names = [r.test_name for r in valid_results]
                exec_times = [r.execution_time for r in valid_results]
                
                axes[0, 0].bar(range(len(test_names)), exec_times)
                axes[0, 0].set_title('Execution Time Comparison')
                axes[0, 0].set_ylabel('Time (seconds)')
                axes[0, 0].set_xticks(range(len(test_names)), test_names, rotation=45, ha='right')
                
                # Plot 2: Memory Usage
                memory_usage = [r.memory_usage_mb for r in valid_results]
                axes[0, 1].bar(range(len(test_names)), memory_usage)
                axes[0, 1].set_title('Memory Usage')
                axes[0, 1].set_ylabel('Memory (MB)')
                axes[0, 1].set_xticks(range(len(test_names)), test_names, rotation=45, ha='right')
                
                # Plot 3: Throughput
                throughput = [r.throughput_tokens_per_sec for r in valid_results if r.throughput_tokens_per_sec > 0]
                if throughput:
                    throughput_names = [r.test_name for r in valid_results if r.throughput_tokens_per_sec > 0]
                    axes[1, 0].bar(range(len(throughput_names)), throughput)
                    axes[1, 0].set_title('Throughput Comparison')
                    axes[1, 0].set_ylabel('Tokens/second')
                    axes[1, 0].set_xticks(range(len(throughput_names)), throughput_names, rotation=45, ha='right')
                else:
                    axes[1, 0].text(0.5, 0.5, 'No throughput data', ha='center', va='center', transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title('Throughput Comparison')
                
                # Plot 4: Performance Summary
                categories = ['Dataset Loading', 'Training', 'CPU Inference', 'GPU Inference']
                performance_scores = []
                
                # Calculate normalized performance scores
                for category in categories:
                    category_results = [r for r in valid_results if category.lower() in r.test_name.lower()]
                    if category_results:
                        avg_score = np.mean([r.throughput_tokens_per_sec for r in category_results if r.throughput_tokens_per_sec > 0])
                        performance_scores.append(avg_score if not np.isnan(avg_score) else 0)
                    else:
                        performance_scores.append(0)
                
                axes[1, 1].bar(categories, performance_scores)
                axes[1, 1].set_title('Performance Summary')
                axes[1, 1].set_ylabel('Throughput (tokens/sec)')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.output_dir / f"benchmark_visualization_{dataset_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   📈 Visualization saved: {plot_path}")
            
        except Exception as e:
            print(f"   ⚠️ Visualization generation failed: {e}")


def main():
    """Command line interface for benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SloGPT Performance Benchmarking")
    parser.add_argument('--dataset', default='benchmark_test', help='Dataset name for benchmarking')
    parser.add_argument('--output', default='benchmark_results', help='Output directory for results')
    parser.add_argument('--dataset-only', action='store_true', help='Only benchmark dataset loading')
    parser.add_argument('--inference-only', action='store_true', help='Only benchmark model inference')
    parser.add_argument('--training-only', action='store_true', help='Only benchmark training speed')
    
    args = parser.parse_args()
    
    # Create benchmark suite
    benchmark = PerformanceBenchmark(args.output)
    
    if args.dataset_only:
        result = benchmark.benchmark_dataset_loading(args.dataset)
        benchmark.results = [result]
    elif args.inference_only:
        # Need a model for inference benchmarking
        print("❌ Inference benchmark requires a trained model")
        return 1
    elif args.training_only:
        model_config = {
            'n_embed': 128,
            'n_layer': 2,
            'n_head': 4,
            'batch_size': 8,
            'sequence_length': 64
        }
        result = benchmark.benchmark_training_speed(args.dataset, model_config)
        benchmark.results = [result]
    else:
        # Run full benchmark suite
        benchmark.run_full_benchmark_suite(args.dataset)
    
    print(f"\n🎉 Benchmarking completed!")
    print(f"📁 Results saved in: {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())