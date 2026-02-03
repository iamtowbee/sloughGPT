#!/usr/bin/env python3
"""
SloughGPT Performance Optimizations
Quantization, compilation, and performance enhancements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, Any, Optional, Tuple
import logging
from contextlib import contextmanager

from .config import ModelConfig
from .neural_network import SloughGPT

class OptimizedSloughGPT(SloughGPT):
    """Enhanced SloughGPT with performance optimizations"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.quantized_model = None
        self.compiled_model = None
        self.optimization_config = {
            'enable_quantization': False,
            'enable_compilation': False,
            'enable_gradient_checkpointing': False,
            'mixed_precision': False
        }
        
    def enable_quantization(self, quantization_type: str = 'dynamic'):
        """Enable model quantization for reduced memory usage"""
        try:
            if quantization_type == 'dynamic':
                # Dynamic quantization for linear layers
                self.quantized_model = torch.quantization.quantize_dynamic(
                    self, {nn.Linear}, dtype=torch.qint8
                )
            elif quantization_type == 'static':
                # Static quantization (requires calibration)
                self.quantized_model = self._prepare_static_quantization()
            
            self.optimization_config['enable_quantization'] = True
            logging.info(f"Enabled {quantization_type} quantization")
            return True
            
        except Exception as e:
            logging.error(f"Quantization failed: {e}")
            return False
    
    def _prepare_static_quantization(self):
        """Prepare model for static quantization"""
        # Prepare model for quantization
        model = torch.quantization.prepare(self)
        
        # Calibration step (simplified - in practice use real data)
        dummy_input = torch.randint(0, self.config.vocab_size, (1, 10))
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Convert to quantized model
        return torch.quantization.convert(model)
    
    def enable_compilation(self):
        """Enable torch.compile for performance acceleration"""
        try:
            if hasattr(torch, 'compile'):
                self.compiled_model = torch.compile(self)
                self.optimization_config['enable_compilation'] = True
                logging.info("Enabled torch compilation")
                return True
            else:
                logging.warning("torch.compile not available (PyTorch < 2.0)")
                return False
        except Exception as e:
            logging.error(f"Compilation failed: {e}")
            return False
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory optimization"""
        try:
            if hasattr(self, 'gradient_checkpointing_enable'):
                self.gradient_checkpointing_enable()
                self.optimization_config['enable_gradient_checkpointing'] = True
                logging.info("Enabled gradient checkpointing")
                return True
            else:
                logging.warning("Gradient checkpointing not available")
                return False
        except Exception as e:
            logging.error(f"Gradient checkpointing failed: {e}")
            return False
    
    def enable_mixed_precision(self):
        """Enable mixed precision training/inference"""
        try:
            self.optimization_config['mixed_precision'] = True
            logging.info("Enabled mixed precision")
            return True
        except Exception as e:
            logging.error(f"Mixed precision setup failed: {e}")
            return False
    
    def get_model_for_inference(self):
        """Get the optimized model for inference"""
        if self.compiled_model is not None:
            return self.compiled_model
        elif self.quantized_model is not None:
            return self.quantized_model
        else:
            return self
    
    @contextmanager
    def inference_context(self):
        """Context manager for optimized inference"""
        if self.optimization_config['mixed_precision']:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    
    def benchmark_model(self, input_sizes: list[tuple] = None) -> Dict[str, Any]:
        """Benchmark model performance with different input sizes"""
        if input_sizes is None:
            input_sizes = [(1, 10), (1, 50), (1, 100), (2, 50)]
        
        results = {}
        
        for batch_size, seq_len in input_sizes:
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
            
            # Test original model
            original_time = self._time_forward_pass(input_ids, self)
            
            # Test quantized model (if available)
            quantized_time = None
            if self.quantized_model is not None:
                quantized_time = self._time_forward_pass(input_ids, self.quantized_model)
            
            # Test compiled model (if available)
            compiled_time = None
            if self.compiled_model is not None:
                compiled_time = self._time_forward_pass(input_ids, self.compiled_model)
            
            results[f"batch_{batch_size}_seq_{seq_len}"] = {
                'original_time': original_time,
                'quantized_time': quantized_time,
                'compiled_time': compiled_time,
                'memory_usage': self._measure_memory_usage(input_ids)
            }
        
        return results
    
    def _time_forward_pass(self, input_ids: torch.Tensor, model: nn.Module) -> float:
        """Time a single forward pass"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)
        
        # Timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        return (end_time - start_time) / 10  # Average time
    
    def _measure_memory_usage(self, input_ids: torch.Tensor) -> Dict[str, float]:
        """Measure memory usage during inference"""
        if not torch.cuda.is_available():
            return {'peak_memory_mb': 0, 'allocated_memory_mb': 0}
        
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = self(input_ids)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        allocated_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        return {
            'peak_memory_mb': peak_memory,
            'allocated_memory_mb': allocated_memory
        }
    
    def optimize_for_production(self):
        """Apply all available optimizations for production"""
        optimizations_applied = []
        
        # Enable quantization
        if self.enable_quantization():
            optimizations_applied.append('quantization')
        
        # Enable compilation
        if self.enable_compilation():
            optimizations_applied.append('compilation')
        
        # Enable mixed precision
        if self.enable_mixed_precision():
            optimizations_applied.append('mixed_precision')
        
        # Enable gradient checkpointing (for training)
        if self.enable_gradient_checkpointing():
            optimizations_applied.append('gradient_checkpointing')
        
        logging.info(f"Applied optimizations: {optimizations_applied}")
        return optimizations_applied
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of applied optimizations"""
        return {
            'optimizations': self.optimization_config,
            'quantization_enabled': self.quantized_model is not None,
            'compilation_enabled': self.compiled_model is not None,
            'model_size_mb': self._get_model_size(),
            'device': str(self.device)
        }
    
    def _get_model_size(self) -> float:
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / 1024**2  # MB

class ModelProfiler:
    """Profile model performance and bottlenecks"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.profiler = None
    
    def start_profiling(self):
        """Start profiling the model"""
        try:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profile'),
                record_shapes=True,
                with_stack=True
            )
            self.profiler.start()
            return True
        except Exception as e:
            logging.error(f"Profiling failed: {e}")
            return False
    
    def step(self):
        """Advance profiler step"""
        if self.profiler is not None:
            self.profiler.step()
    
    def stop_profiling(self):
        """Stop profiling and export results"""
        if self.profiler is not None:
            self.profiler.stop()
            return True
        return False
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze model bottlenecks"""
        # This would implement detailed bottleneck analysis
        # For now, return basic suggestions
        return {
            'recommendations': [
                'Consider gradient checkpointing for large models',
                'Enable mixed precision for faster training',
                'Use quantization for reduced memory usage',
                'Apply model compilation if available'
            ],
            'memory_efficiency': 'medium',
            'compute_efficiency': 'medium'
        }

def create_optimized_model(config: ModelConfig, 
                         enable_quantization: bool = False,
                         enable_compilation: bool = False,
                         enable_mixed_precision: bool = False) -> OptimizedSloughGPT:
    """Factory function to create optimized SloughGPT model"""
    
    model = OptimizedSloughGPT(config)
    
    if enable_quantization:
        model.enable_quantization()
    
    if enable_compilation:
        model.enable_compilation()
    
    if enable_mixed_precision:
        model.enable_mixed_precision()
    
    return model

def benchmark_comparison(base_config: ModelConfig) -> Dict[str, Any]:
    """Benchmark different model configurations"""
    results = {}
    
    # Create test input
    test_input = torch.randint(0, base_config.vocab_size, (1, 50))
    
    # Base model
    base_model = SloughGPT(base_config)
    
    # Time base model
    base_model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = base_model(test_input)
        
        # Timing
        start_time = time.time()
        for _ in range(10):
            _ = base_model(test_input)
        base_time = (time.time() - start_time) / 10
    
    # Optimized model
    optimized_model = create_optimized_model(
        base_config,
        enable_quantization=True,
        enable_compilation=False  # Skip compilation for testing
    )
    
    # Time optimized model
    inference_model = optimized_model.get_model_for_inference()
    inference_model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = inference_model(test_input)
        
        # Timing
        start_time = time.time()
        for _ in range(10):
            _ = inference_model(test_input)
        optimized_time = (time.time() - start_time) / 10
    
    # Calculate model sizes
    def get_model_size(model):
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1024**2  # MB
    
    base_size = get_model_size(base_model)
    optimized_size = get_model_size(optimized_model)
    
    results = {
        'base_model': {
            'time_ms': base_time * 1000,
            'model_size_mb': base_size
        },
        'optimized_model': {
            'time_ms': optimized_time * 1000,
            'model_size_mb': optimized_size
        },
        'speedup': base_time / optimized_time if optimized_time > 0 else 0,
        'memory_reduction': (base_size - optimized_size) / base_size * 100 if base_size > 0 else 0
    }
    
    return results