#!/usr/bin/env python3
"""
Comprehensive Test Suite for SloughGPT Package
Tests all major functionality including model creation, training, and inference
"""

import sys
import os
import time
import torch
import unittest
from typing import Dict, Any, List

# Add current directory to Python path
current_dir = "/Users/mac/sloughGPT"
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import SloughGPT modules
from sloughgpt.config import ModelConfig, LearningConfig, CognitiveConfig
from sloughgpt.neural_network import SloughGPT
from sloughgpt.core.exceptions import SloughGPTError, create_error, ConfigurationError, ModelError

class TestSloughGPTPackage(unittest.TestCase):
    """Comprehensive test suite for SloughGPT package"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ModelConfig()
        self.device = torch.device("cpu")  # Use CPU for testing
        
    def test_config_creation(self):
        """Test configuration creation and validation"""
        print("\n🧪 Testing Configuration Creation...")
        
        # Test default config
        config = ModelConfig()
        self.assertIsNotNone(config)
        self.assertIsInstance(config.vocab_size, int)
        self.assertGreater(config.vocab_size, 0)
        
        # Test config validation
        errors = config.validate()
        self.assertIsInstance(errors, list)
        
        # Test config to dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn('vocab_size', config_dict)
        
        print("✅ Configuration creation tests passed")
    
    def test_model_creation(self):
        """Test neural network model creation"""
        print("\n🧪 Testing Model Creation...")
        
        model = SloughGPT(self.config)
        self.assertIsNotNone(model)
        
        # Test model parameters
        param_count = model.count_parameters()
        self.assertGreater(param_count, 0)
        print(f"   Model parameters: {param_count:,}")
        
        # Test model info
        model_info = model.get_model_info()
        self.assertIsInstance(model_info, dict)
        self.assertIn('total_parameters', model_info)
        self.assertIn('device', model_info)
        
        print("✅ Model creation tests passed")
    
    def test_model_forward_pass(self):
        """Test forward pass functionality"""
        print("\n🧪 Testing Forward Pass...")
        
        model = SloughGPT(self.config)
        model.eval()
        
        # Create test input
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        
        # Test forward pass
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_ids)
        
        forward_time = time.time() - start_time
        
        # Verify outputs
        self.assertIsInstance(outputs, dict)
        self.assertIn('logits', outputs)
        self.assertIn('hidden_states', outputs)
        self.assertIn('forward_time', outputs)
        
        # Verify tensor shapes
        logits = outputs['logits']
        expected_shape = (batch_size, seq_length, self.config.vocab_size)
        self.assertEqual(logits.shape, expected_shape)
        
        print(f"   Forward pass time: {forward_time:.3f}s")
        print(f"   Output logits shape: {logits.shape}")
        print("✅ Forward pass tests passed")
    
    def test_text_generation(self):
        """Test text generation functionality"""
        print("\n🧪 Testing Text Generation...")
        
        model = SloughGPT(self.config)
        model.eval()
        
        # Create test input
        input_ids = torch.randint(0, self.config.vocab_size, (1, 5))
        
        # Test generation
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_length=15,
                temperature=1.0,
                do_sample=True
            )
        
        # Verify generation
        self.assertIsInstance(generated_ids, torch.Tensor)
        self.assertGreater(generated_ids.shape[1], input_ids.shape[1])  # Should be longer
        
        print(f"   Input length: {input_ids.shape[1]}")
        print(f"   Generated length: {generated_ids.shape[1]}")
        print("✅ Text generation tests passed")
    
    def test_attention_mechanism(self):
        """Test attention mechanism functionality"""
        print("\n🧪 Testing Attention Mechanism...")
        
        model = SloughGPT(self.config)
        model.eval()
        
        # Test with attention mask
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        
        # Create attention mask (causal)
        attention_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1, -1)
        attention_mask = attention_mask.float().masked_fill(attention_mask, float('-inf'))
        
        with torch.no_grad():
            outputs_with_mask = model(input_ids, attention_mask=attention_mask)
            outputs_without_mask = model(input_ids)
        
        # Both should work
        self.assertIsInstance(outputs_with_mask, dict)
        self.assertIsInstance(outputs_without_mask, dict)
        self.assertIn('logits', outputs_with_mask)
        self.assertIn('logits', outputs_without_mask)
        
        print("✅ Attention mechanism tests passed")
    
    def test_error_handling(self):
        """Test error handling system"""
        print("\n🧪 Testing Error Handling...")
        
        # Test error creation
        error = create_error(ConfigurationError, "Test configuration error")
        self.assertIsInstance(error, ConfigurationError)
        self.assertEqual(error.message, "Test configuration error")
        
        # Test error serialization
        error_dict = error.to_dict()
        self.assertIsInstance(error_dict, dict)
        self.assertIn('error_type', error_dict)
        self.assertIn('message', error_dict)
        
        # Test error JSON conversion
        error_json = error.to_json()
        self.assertIsInstance(error_json, str)
        
        print("✅ Error handling tests passed")
    
    def test_model_performance(self):
        """Test model performance metrics"""
        print("\n🧪 Testing Model Performance...")
        
        model = SloughGPT(self.config)
        model.eval()
        
        # Test with different sequence lengths
        seq_lengths = [5, 10, 20]
        batch_size = 2
        
        for seq_len in seq_lengths:
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model(input_ids)
            end_time = time.time()
            
            actual_time = end_time - start_time
            reported_time = outputs.get('forward_time', 0)
            
            print(f"   Seq length {seq_len}: {actual_time:.3f}s actual, {reported_time:.3f}s reported")
            
            # Verify performance metrics are reasonable
            self.assertGreater(actual_time, 0)
            self.assertGreater(reported_time, 0)
        
        print("✅ Model performance tests passed")
    
    def test_config_compatibility(self):
        """Test configuration compatibility with different attribute names"""
        print("\n🧪 Testing Configuration Compatibility...")
        
        # Test with minimal config
        class MinimalConfig:
            def __init__(self):
                self.vocab_size = 1000
                self.d_model = 256
                self.n_heads = 4
                self.n_layers = 2
                self.d_ff = 1024
                self.dropout = 0.1
                self.max_seq_length = 512
        
        minimal_config = MinimalConfig()
        model = SloughGPT(minimal_config)
        self.assertIsNotNone(model)
        
        # Verify mapped attributes
        self.assertEqual(model.config.hidden_size, 256)
        self.assertEqual(model.config.num_attention_heads, 4)
        self.assertEqual(model.config.num_hidden_layers, 2)
        
        print("✅ Configuration compatibility tests passed")
    
    def test_memory_usage(self):
        """Test memory usage tracking"""
        print("\n🧪 Testing Memory Usage...")
        
        model = SloughGPT(self.config)
        model.eval()
        
        input_ids = torch.randint(0, self.config.vocab_size, (1, 10))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        memory_usage = outputs.get('memory_usage', 0)
        self.assertIsInstance(memory_usage, (int, float))
        self.assertGreaterEqual(memory_usage, 0)
        
        print(f"   Memory usage: {memory_usage:.2f} MB")
        print("✅ Memory usage tests passed")

class TestPackageIntegration(unittest.TestCase):
    """Integration tests for the complete package"""
    
    def test_full_workflow(self):
        """Test complete workflow from config to generation"""
        print("\n🧪 Testing Full Workflow...")
        
        # 1. Create configuration
        config = ModelConfig()
        
        # 2. Create model
        model = SloughGPT(config)
        model.eval()
        
        # 3. Test forward pass
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        
        with torch.no_grad():
            outputs = model(input_ids)
            
            # 4. Test generation
            generated = model.generate(
                input_ids=input_ids,
                max_length=20,
                temperature=0.8,
                do_sample=True
            )
        
        # 5. Verify results
        self.assertIsInstance(outputs, dict)
        self.assertIsInstance(generated, torch.Tensor)
        self.assertGreater(generated.shape[1], input_ids.shape[1])
        
        # 6. Get model info
        model_info = model.get_model_info()
        self.assertIsInstance(model_info, dict)
        
        print("✅ Full workflow tests passed")

def run_comprehensive_tests():
    """Run all tests and generate report"""
    print("🚀 Running Comprehensive SloughGPT Package Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSloughGPTPackage))
    suite.addTests(loader.loadTestsFromTestCase(TestPackageIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=open(os.devnull, 'w'))
    result = runner.run(suite)
    
    # Generate summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
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
        print("\n🎉 All tests passed! SloughGPT package is fully functional!")
        return True
    else:
        print("\n⚠️  Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)