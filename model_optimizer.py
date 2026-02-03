#!/usr/bin/env python3
"""
Model Optimization and Quantization for SloGPT
Advanced model optimization for production deployment.
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time


class ModelOptimizer:
    """Advanced model optimization toolkit."""
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.original_size = 0
        
    def load_model(self):
        """Load model for optimization."""
        try:
            if self.model_path.suffix == '.pt':
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Load from checkpoint format
                    state_dict = checkpoint['model_state_dict']
                    config = checkpoint.get('config', {})
                    
                    # Create model from config
                    from simple_gpt_model import GPT
                    vocab_size = config.get('vocab_size', 50000)
                    n_embed = config.get('n_embed', 384)
                    n_layer = config.get('n_layer', 6)
                    n_head = config.get('n_head', 6)
                    
                    self.model = GPT(vocab_size, n_embed, n_layer, n_head)
                    self.model.load_state_dict(state_dict)
                else:
                    # Direct model state dict
                    from simple_gpt_model import GPT
                    self.model = GPT(vocab_size=50000)  # Default
                    self.model.load_state_dict(checkpoint)
            else:
                print(f"❌ Unsupported model format: {self.model_path.suffix}")
                return False
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Calculate original size
            self.original_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            
            print(f"✅ Model loaded successfully")
            print(f"   Original size: {self.original_size / (1024**2):.2f} MB")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def quantize_model_int8(self):
        """Quantize model to INT8 for inference optimization."""
        print("🔢 Starting INT8 quantization...")
        
        try:
            # Prepare model for quantization
            self.model.eval()
            
            # Create sample input for tracing
            vocab_size = getattr(self.model, 'vocab_size', 50000)
            sample_input = torch.randint(0, vocab_size, (1, 32)).to(self.device)
            
            # Dynamically quantize the model
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv1d},
                dtype=torch.qint8
            )
            
            # Test quantized model
            with torch.no_grad():
                output = quantized_model(sample_input)
                print(f"   Quantized output shape: {output.shape}")
            
            # Calculate size reduction
            quantized_size = self._calculate_quantized_size(quantized_model)
            size_reduction = (1 - quantized_size / self.original_size) * 100
            
            print(f"   Quantized size: {quantized_size / (1024**2):.2f} MB")
            print(f"   Size reduction: {size_reduction:.1f}%")
            
            # Save quantized model
            quantized_path = self.output_dir / "model_int8.pt"
            torch.save(quantized_model.state_dict(), quantized_path)
            
            # Save metadata
            metadata = {
                'quantization_type': 'dynamic_int8',
                'original_size_mb': self.original_size / (1024**2),
                'quantized_size_mb': quantized_size / (1024**2),
                'size_reduction_percent': size_reduction,
                'timestamp': time.time()
            }
            
            with open(self.output_dir / "quantization_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ INT8 quantization completed")
            print(f"   Saved: {quantized_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ INT8 quantization failed: {e}")
            return False
    
    def quantize_model_fp16(self):
        """Convert model to FP16 for reduced memory usage."""
        print("🔢 Starting FP16 quantization...")
        
        try:
            # Create FP16 model
            fp16_model = self.model.half()
            
            # Test FP16 model
            vocab_size = getattr(self.model, 'vocab_size', 50000)
            sample_input = torch.randint(0, vocab_size, (1, 32)).half().to(self.device)
            
            with torch.no_grad():
                output = fp16_model(sample_input)
                print(f"   FP16 output shape: {output.shape}")
            
            # Calculate size reduction
            fp16_size = self.original_size / 2  # FP16 is half the size of FP32
            size_reduction = 50.0
            
            print(f"   FP16 size: {fp16_size / (1024**2):.2f} MB")
            print(f"   Size reduction: {size_reduction:.1f}%")
            
            # Save FP16 model
            fp16_path = self.output_dir / "model_fp16.pt"
            torch.save(fp16_model.state_dict(), fp16_path)
            
            # Save metadata
            metadata = {
                'quantization_type': 'fp16',
                'original_size_mb': self.original_size / (1024**2),
                'fp16_size_mb': fp16_size / (1024**2),
                'size_reduction_percent': size_reduction,
                'timestamp': time.time()
            }
            
            with open(self.output_dir / "fp16_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ FP16 quantization completed")
            print(f"   Saved: {fp16_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ FP16 quantization failed: {e}")
            return False
    
    def prune_model(self, sparsity: float = 0.2):
        """Apply structured pruning to reduce model size."""
        print(f"✂️ Starting model pruning (sparsity: {sparsity})...")
        
        try:
            # Calculate pruning parameters
            parameters_to_prune = []
            total_params = 0
            
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    # Determine number of parameters to prune
                    weight = module.weight.data
                    num_params = weight.numel()
                    params_to_prune = int(num_params * sparsity)
                    
                    parameters_to_prune.append((name, module, params_to_prune))
                    total_params += num_params
            
            print(f"   Modules to prune: {len(parameters_to_prune)}")
            print(f"   Total parameters: {total_params:,}")
            
            # Apply pruning
            import torch.nn.utils.prune as prune
            
            pruned_model = self.model
            total_pruned = 0
            
            for name, module, params_to_prune in parameters_to_prune:
                # Structured pruning
                prune.ln_structured(module, 'weight', amount=params_to_prune / module.weight.numel())
                total_pruned += params_to_prune
                print(f"   Pruned {name}: {params_to_prune:,} parameters")
            
            # Calculate size reduction
            size_reduction = (total_pruned / total_params) * 100
            pruned_size = self.original_size * (1 - sparsity)
            
            print(f"   Pruned parameters: {total_pruned:,}")
            print(f"   Size reduction: {size_reduction:.1f}%")
            print(f"   Pruned size: {pruned_size / (1024**2):.2f} MB")
            
            # Save pruned model
            pruned_path = self.output_dir / "model_pruned.pt"
            
            # Remove pruning masks for deployment
            for name, module, _ in parameters_to_prune:
                prune.remove(module, 'weight')
            
            torch.save(pruned_model.state_dict(), pruned_path)
            
            # Save pruning metadata
            metadata = {
                'pruning_type': 'structured',
                'sparsity': sparsity,
                'original_size_mb': self.original_size / (1024**2),
                'pruned_size_mb': pruned_size / (1024**2),
                'size_reduction_percent': size_reduction,
                'parameters_pruned': total_pruned,
                'timestamp': time.time()
            }
            
            with open(self.output_dir / "pruning_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Model pruning completed")
            print(f"   Saved: {pruned_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model pruning failed: {e}")
            return False
    
    def optimize_for_inference(self):
        """Optimize model for faster inference."""
        print("⚡ Optimizing model for inference...")
        
        try:
            # Create optimized script
            self.model.eval()
            
            # Sample input for tracing
            vocab_size = getattr(self.model, 'vocab_size', 50000)
            sample_input = torch.randint(0, vocab_size, (1, 32)).to(self.device)
            
            # Trace the model
            traced_model = torch.jit.trace(self.model, sample_input)
            
            # Optimize the traced model
            optimized_model = torch.jit.optimize_for_inference(traced_model)
            
            # Test optimized model
            with torch.no_grad():
                output = optimized_model(sample_input)
                print(f"   Optimized output shape: {output.shape}")
            
            # Save optimized model
            optimized_path = self.output_dir / "model_optimized.pt"
            optimized_model.save(optimized_path)
            
            # Save metadata
            metadata = {
                'optimization_type': 'torchscript_inference',
                'original_size_mb': self.original_size / (1024**2),
                'timestamp': time.time()
            }
            
            with open(self.output_dir / "optimization_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Model optimization completed")
            print(f"   Saved: {optimized_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model optimization failed: {e}")
            return False
    
    def _calculate_quantized_size(self, quantized_model):
        """Calculate size of quantized model."""
        total_size = 0
        
        for param in quantized_model.parameters():
            if hasattr(param, 'element_size'):
                total_size += param.numel() * param.element_size()
            else:
                # Estimate based on data type
                if param.dtype == torch.qint8:
                    total_size += param.numel() * 1  # 1 byte for int8
                elif param.dtype == torch.float16:
                    total_size += param.numel() * 2  # 2 bytes for fp16
                else:
                    total_size += param.numel() * 4  # 4 bytes for fp32
        
        return total_size
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report."""
        print("📊 Generating optimization report...")
        
        report = {
            'model_path': str(self.model_path),
            'original_size_mb': self.original_size / (1024**2),
            'original_parameters': sum(p.numel() for p in self.model.parameters()),
            'optimizations_available': [
                {
                    'name': 'INT8 Quantization',
                    'description': '8-bit integer quantization for maximum compression',
                    'expected_reduction': '70-75%'
                },
                {
                    'name': 'FP16 Conversion',
                    'description': '16-bit floating point for balanced compression',
                    'expected_reduction': '50%'
                },
                {
                    'name': 'Structured Pruning',
                    'description': 'Remove less important model weights',
                    'expected_reduction': 'Configurable (10-50%)'
                },
                {
                    'name': 'Inference Optimization',
                    'description': 'TorchScript optimization for faster inference',
                    'expected_reduction': 'Minimal optimization benefits'
                }
            ],
            'recommendations': self._get_optimization_recommendations()
        }
        
        report_path = self.output_dir / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Optimization report saved: {report_path}")
        return report
    
    def _get_optimization_recommendations(self):
        """Get optimization recommendations based on model size."""
        size_mb = self.original_size / (1024**2)
        
        if size_mb > 1000:
            return [
                "Use INT8 quantization for large models",
                "Consider structured pruning to reduce size",
                "Apply FP16 conversion for memory efficiency"
            ]
        elif size_mb > 500:
            return [
                "Apply FP16 conversion for memory efficiency",
                "Consider light pruning if needed",
                "Use inference optimization"
            ]
        elif size_mb > 100:
            return [
                "Use inference optimization for better performance",
                "Consider FP16 for memory savings"
            ]
        else:
            return [
                "Model is already optimized for size",
                "Consider inference optimization for speed"
            ]


def main():
    """Command line interface for model optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SloGPT Model Optimization")
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--int8', action='store_true', help='Apply INT8 quantization')
    parser.add_argument('--fp16', action='store_true', help='Apply FP16 conversion')
    parser.add_argument('--prune', type=float, help='Apply pruning with specified sparsity (0.0-1.0)')
    parser.add_argument('--optimize', action='store_true', help='Optimize for inference')
    parser.add_argument('--all', action='store_true', help='Apply all optimizations')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = ModelOptimizer(args.model, args.output)
    
    # Load model
    if not optimizer.load_model():
        return 1
    
    # Generate report
    optimizer.generate_optimization_report()
    
    # Apply optimizations
    success = True
    
    if args.all or args.int8:
        if not optimizer.quantize_model_int8():
            success = False
    
    if args.all or args.fp16:
        if not optimizer.quantize_model_fp16():
            success = False
    
    if args.all or args.prune:
        sparsity = args.prune if args.prune else 0.2
        if not optimizer.prune_model(sparsity):
            success = False
    
    if args.all or args.optimize:
        if not optimizer.optimize_for_inference():
            success = False
    
    if success:
        print(f"\n🎉 Model optimization completed successfully!")
        print(f"📁 Check {args.output} for optimized models")
    else:
        print(f"\n❌ Some optimizations failed")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())