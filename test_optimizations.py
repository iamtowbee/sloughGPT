#!/usr/bin/env python3
"""
Test SloughGPT Performance Optimizations
Tests quantization, compilation, and other optimizations
"""

import sys
import os
import time
import torch
import unittest
from typing import Dict, Any

# Add current directory to Python path
current_dir = "/Users/mac/sloughGPT"
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from sloughgpt.config import ModelConfig
from sloughgpt.optimizations import OptimizedSloughGPT, create_optimized_model, benchmark_comparison

class TestPerformanceOptimizations(unittest.TestCase):
    """Test performance optimization features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ModelConfig(vocab_size=1000, d_model=256, n_heads=4, n_layers=2)  # Smaller for testing
        
    def test_optimized_model_creation(self):
        """Test creation of optimized model"""
        print("\n🧪 Testing Optimized Model Creation...")
        
        model = OptimizedSloughGPT(self.config)
        self.assertIsNotNone(model)
        
        # Test optimization config
        self.assertIsInstance(model.optimization_config, dict)
        self.assertFalse(model.optimization_config['enable_quantization'])
        self.assertFalse(model.optimization_config['enable_compilation'])
        
        print("✅ Optimized model creation tests passed")
    
    def test_quantization(self):
        """Test model quantization"""
        print("\n🧪 Testing Model Quantization...")
        
        model = OptimizedSloughGPT(self.config)
        
        # Test dynamic quantization
        success = model.enable_quantization('dynamic')
        self.assertTrue(success)
        self.assertIsNotNone(model.quantized_model)
        self.assertTrue(model.optimization_config['enable_quantization'])
        
        # Test quantized model inference
        input_ids = torch.randint(0, self.config.vocab_size, (1, 10))
        with torch.no_grad():
            output_original = model(input_ids)
            output_quantized = model.quantized_model(input_ids)
        
        # Outputs should have similar shapes
        self.assertEqual(output_original['logits'].shape, output_quantized['logits'].shape)
        
        print("✅ Model quantization tests passed")
    
    def test_mixed_precision(self):
        """Test mixed precision setup"""
        print("\n🧪 Testing Mixed Precision...")
        
        model = OptimizedSloughGPT(self.config)
        
        success = model.enable_mixed_precision()
        self.assertTrue(success)
        self.assertTrue(model.optimization_config['mixed_precision'])
        
        # Test inference context
        input_ids = torch.randint(0, self.config.vocab_size, (1, 10))
        
        with model.inference_context():
            with torch.no_grad():
                output = model(input_ids)
        
        self.assertIsInstance(output, dict)
        self.assertIn('logits', output)
        
        print("✅ Mixed precision tests passed")
    
    def test_model_benchmarking(self):
        """Test model benchmarking functionality"""
        print("\n🧪 Testing Model Benchmarking...")
        
        model = OptimizedSloughGPT(self.config)
        
        # Test benchmarking
        results = model.benchmark_model([(1, 10), (1, 20)])
        
        self.assertIsInstance(results, dict)
        self.assertIn('batch_1_seq_10', results)
        self.assertIn('batch_1_seq_20', results)
        
        # Check benchmark data structure
        for benchmark_name, benchmark_data in results.items():
            self.assertIn('original_time', benchmark_data)
            self.assertIn('memory_usage', benchmark_data)
            self.assertIsInstance(benchmark_data['original_time'], float)
            self.assertGreater(benchmark_data['original_time'], 0)
        
        print("✅ Model benchmarking tests passed")
    
    def test_optimization_summary(self):
        """Test optimization summary functionality"""
        print("\n🧪 Testing Optimization Summary...")
        
        model = OptimizedSloughGPT(self.config)
        model.enable_quantization()
        model.enable_mixed_precision()
        
        summary = model.get_optimization_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('optimizations', summary)
        self.assertIn('quantization_enabled', summary)
        self.assertIn('model_size_mb', summary)
        self.assertIn('device', summary)
        
        self.assertTrue(summary['quantization_enabled'])
        self.assertTrue(summary['optimizations']['mixed_precision'])
        self.assertGreater(summary['model_size_mb'], 0)
        
        print("✅ Optimization summary tests passed")
    
    def test_production_optimization(self):
        """Test production optimization workflow"""
        print("\n🧪 Testing Production Optimization...")
        
        model = OptimizedSloughGPT(self.config)
        
        optimizations = model.optimize_for_production()
        
        self.assertIsInstance(optimizations, list)
        # At least some optimizations should be applied
        self.assertGreater(len(optimizations), 0)
        
        # Verify optimizations were recorded
        summary = model.get_optimization_summary()
        applied_opts = summary['optimizations']
        
        # Check that at least quantization was attempted
        self.assertIn('enable_quantization', applied_opts)
        
        print("✅ Production optimization tests passed")
    
    def test_factory_function(self):
        """Test factory function for creating optimized models"""
        print("\n🧪 Testing Factory Function...")
        
        model = create_optimized_model(
            self.config,
            enable_quantization=True,
            enable_mixed_precision=True
        )
        
        self.assertIsInstance(model, OptimizedSloughGPT)
        self.assertTrue(model.optimization_config['enable_quantization'])
        self.assertTrue(model.optimization_config['mixed_precision'])
        
        print("✅ Factory function tests passed")
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison between models"""
        print("\n🧪 Testing Benchmark Comparison...")
        
        results = benchmark_comparison(self.config)
        
        self.assertIsInstance(results, dict)
        self.assertIn('base_model', results)
        self.assertIn('optimized_model', results)
        self.assertIn('speedup', results)
        self.assertIn('memory_reduction', results)
        
        # Verify data structure
        base_data = results['base_model']
        optimized_data = results['optimized_model']
        
        self.assertIn('time_ms', base_data)
        self.assertIn('model_size_mb', base_data)
        self.assertIn('time_ms', optimized_data)
        self.assertIn('model_size_mb', optimized_data)
        
        print("✅ Benchmark comparison tests passed")

def run_optimization_tests():
    """Run all optimization tests"""
    print("🚀 Running SloughGPT Performance Optimization Tests")
    print("=" * 65)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPerformanceOptimizations)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=open(os.devnull, 'w'))
    result = runner.run(suite)
    
    # Generate summary
    print("\n" + "=" * 65)
    print("📊 OPTIMIZATION TEST SUMMARY")
    print("=" * 65)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failures}")
    print(f"💥 Errors: {errors}")
    print(f"Success Rate: {(passed/total_tests)*100:.1f}%")
    
    if failures > 0:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if errors > 0:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if failures == 0 and errors == 0:
        print("\n🎉 All optimization tests passed!")
        return True
    else:
        print("\n⚠️  Some optimization tests failed.")
        return False

if __name__ == "__main__":
    success = run_optimization_tests()
    sys.exit(0 if success else 1)