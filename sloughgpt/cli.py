#!/usr/bin/env python3
"""
SloughGPT CLI Tool
Command-line interface for SloughGPT management and operations
"""

import argparse
import json
import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sloughgpt.config import ModelConfig
from sloughgpt.neural_network import SloughGPT
from sloughgpt.optimizations import create_optimized_model
from sloughgpt.trainer import SloughGPTTrainer, TrainingConfig
from sloughgpt.model_zoo import get_model_zoo
from sloughgpt.benchmark_suite import BenchmarkSuite

def cmd_list_models(args):
    """List available models"""
    zoo = get_model_zoo()
    models = zoo.list_models()
    
    print("🦁 Available SloughGPT Models")
    print("=" * 50)
    
    for model in models:
        print(f"📊 {model.name}")
        print(f"   Description: {model.description}")
        print(f"   Parameters: {model.parameters:,}")
        print(f"   Tags: {', '.join(model.tags)}")
        print(f"   Config: {model.config.d_model}x{model.config.n_heads}x{model.config.n_layers}")
        print()

def cmd_info(args):
    """Show model information"""
    zoo = get_model_zoo()
    model = zoo.get_model(args.name)
    
    if not model:
        print(f"❌ Model not found: {args.name}")
        return 1
    
    print(f"📊 Model Information: {model.name}")
    print("=" * 50)
    print(f"Name: {model.name}")
    print(f"Description: {model.description}")
    print(f"Version: {model.version}")
    print(f"License: {model.license}")
    print(f"Created: {model.created_at}")
    print(f"Tags: {', '.join(model.tags)}")
    print(f"File Size: {model.file_size / (1024*1024):.1f} MB")
    print(f"Parameters: {model.parameters:,}")
    print()
    print("🏗️ Architecture:")
    print(f"  Hidden Size: {model.config.d_model}")
    print(f"  Attention Heads: {model.config.n_heads}")
    print(f"  Layers: {model.config.n_layers}")
    print(f"  Vocabulary Size: {model.config.vocab_size}")
    print(f"  Max Sequence Length: {model.config.max_seq_length}")

def cmd_generate(args):
    """Generate text using a model"""
    try:
        zoo = get_model_zoo()
        model = zoo.load_model(args.model or "sloughgpt-medium", optimize=True)
        
        print(f"🧠 Using model: {args.model}")
        print(f"📝 Input: {args.text}")
        print(f"⚙️  Temperature: {args.temperature}")
        print(f"🎯 Max Length: {args.max_length}")
        print()
        
        # Generate text
        import torch
        input_ids = torch.randint(0, model.config.vocab_size, (1, min(len(args.text), 10)))
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.sample
            )
        
        # Simple character-level detokenization
        generated_tokens = generated_ids[0][10:].tolist()
        generated_text = ''.join([chr(token % 256) for token in generated_tokens])
        
        print("✍ Generated Text:")
        print("=" * 30)
        print(generated_text)
        print("=" * 30)
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return 1

def cmd_benchmark(args):
    """Run benchmarks"""
    suite = BenchmarkSuite()
    
    if args.model:
        # Benchmark specific model
        configs = {
            "small": ModelConfig(d_model=256, n_heads=4, n_layers=4),
            "medium": ModelConfig(d_model=512, n_heads=8, n_layers=6),
            "large": ModelConfig(d_model=1024, n_heads=16, n_layers=12)
        }
        
        if args.model not in configs:
            print(f"❌ Unknown model: {args.model}")
            return 1
        
        config = configs[args.model]
        
        print(f"🏃 Benchmarking {args.model} model...")
        
        from sloughgpt.neural_network import SloughGPT
        model = SloughGPT(config)
        benchmark = suite.benchmark_model_inference(model, args.model)
        
        print("📊 Results:")
        for test_name, result in benchmark.results.items():
            metrics = result
            print(f"  {test_name}:")
            print(f"    Time: {metrics.get('avg_time_seconds', 0):.3f}s")
            print(f"    Throughput: {metrics.get('tokens_per_second', 0):.1f} tokens/sec")
            print(f"    Memory: {metrics.get('avg_memory_mb', 0):.1f} MB")
        
    else:
        # Run quick benchmark
        print("🚀 Running quick benchmark...")
        os.system("./run_benchmark.sh")

def cmd_train(args):
    """Train a model"""
    try:
        print(f"🏋️‍♂️ Starting training...")
        
        # Load configuration
        config = ModelConfig(
            vocab_size=args.vocab_size,
            d_model=args.hidden_size,
            n_heads=args.attention_heads,
            n_layers=args.layers
        )
        
        # Create training config
        training_config = TrainingConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            save_interval=args.save_interval
        )
        
        # Create trainer
        from sloughgpt.neural_network import SloughGPT
        model = SloughGPT(config)
        trainer = SloughGPTTrainer(model, training_config)
        
        print(f"📊 Model: {config.d_model}x{config.n_heads}x{config.n_layers}")
        print(f"🎯 Parameters: {model.count_parameters():,}")
        
        # Start training
        if args.data:
            stats = trainer.train(args.data)
        else:
            # Sample data
            sample_data = ["Hello world!"] * 1000
            stats = trainer.train(sample_data)
        
        print("✅ Training completed!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1

def cmd_serve(args):
    """Start the API server"""
    try:
        print(f"🌐 Starting SloughGPT server...")
        print(f"📍 Host: {args.host}")
        print(f"🔌 Port: {args.port}")
        
        if args.dev:
            print("🔧 Development mode with auto-reload")
            os.system(f"python -m sloughgpt.web_server")
        else:
            print("🚀 Production mode")
            os.system(f"uvicorn sloughgpt.web_server:create_app() --host {args.host} --port {args.port}")
        
    except Exception as e:
        print(f"❌ Server failed: {e}")
        return 1

def cmd_status(args):
    """Show system status"""
    import psutil
    import torch
    
    print("🖥️ SloughGPT System Status")
    print("=" * 40)
    
    # System info
    print("💻 System:")
    print(f"  Platform: {psutil.platform.system()}")
    print(f"  CPU: {psutil.cpu_count(logical=False)} cores")
    print(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        print("  GPU: Not available")
    
    # PyTorch info
    print(f"  PyTorch: {torch.__version__}")
    
    # Package info
    zoo = get_model_zoo()
    stats = zoo.get_statistics()
    print(f"\n📦 Model Zoo:")
    print(f"  Models: {stats['total_models']}")
    print(f"  Size: {stats['total_size_mb']:.1f} MB")
    print(f"  Parameters: {stats['total_parameters']:,}")

def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="SloughGPT CLI - Advanced AI Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List models
    list_parser = subparsers.add_parser('list', help='List available models')
    list_parser.set_defaults(func=cmd_list_models)
    
    # Model info
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('name', help='Model name')
    info_parser.set_defaults(func=cmd_info)
    
    # Generate
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument('--model', default='sloughgpt-medium', help='Model to use')
    gen_parser.add_argument('text', required=True, help='Input text')
    gen_parser.add_argument('--max-length', type=int, default=50, help='Max generation length')
    gen_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    gen_parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    gen_parser.add_argument('--top-p', type=float, default=0.9, help='Top-p sampling')
    gen_parser.add_argument('--no-sample', action='store_true', help='Disable sampling')
    gen_parser.set_defaults(func=cmd_generate)
    
    # Benchmark
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    bench_parser.add_argument('--model', help='Specific model to benchmark')
    bench_parser.set_defaults(func=cmd_benchmark)
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--vocab-size', type=int, default=10000, help='Vocabulary size')
    train_parser.add_argument('--hidden-size', type=int, default=512, help='Hidden layer size')
    train_parser.add_argument('--attention-heads', type=int, default=8, help='Attention heads')
    train_parser.add_argument('--layers', type=int, default=6, help='Number of layers')
    train_parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    train_parser.add_argument('--save-interval', type=int, default=1000, help='Save checkpoint interval')
    train_parser.add_argument('--data', help='Training data file')
    train_parser.set_defaults(func=cmd_train)
    
    # Serve
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    serve_parser.add_argument('--dev', action='store_true', help='Development mode with reload')
    serve_parser.set_defaults(func=cmd_serve)
    
    # Status
    status_parser = subparsers.add_parser('status', help='Show system status')
    status_parser.set_defaults(func=cmd_status)
    
    return parser

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())