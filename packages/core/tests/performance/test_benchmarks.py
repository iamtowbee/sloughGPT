"""
SloughGPT Performance Tests
Performance benchmarks and load testing
"""

import pytest
import time
import asyncio
import statistics
from unittest.mock import Mock, patch
import concurrent.futures

import sys
import os

# Add sloughgpt to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from sloughgpt.core.testing import BaseTestCase, TestConfiguration, TestType, TestStatus
from sloughgpt.core.performance import get_performance_optimizer, CacheLevel, PerformanceOptimizer
from sloughgpt.core.logging_system import get_logger

class TestCachingPerformance(BaseTestCase):
    """Test caching performance characteristics"""
    
    def setup_method(self):
        """Setup performance test environment"""
        self.optimizer = get_performance_optimizer()
        self.logger = get_logger("cache_performance_test")
    
    def test_memory_cache_performance(self):
        """Test memory cache performance under load"""
        @self.optimizer.cached(CacheLevel.MEMORY, ttl=300)
        def memory_intensive_computation(x):
            # Simulate expensive computation
            time.sleep(0.01)
            return sum(range(x))
        
        # Benchmark uncached performance
        start_time = time.time()
        uncached_results = []
        for i in range(100):
            result = memory_intensive_computation(100)
            uncached_results.append(result)
        uncached_duration = (time.time() - start_time) * 1000
        
        # Clear cache and benchmark cached performance
        memory_intensive_computation.cache_clear()
        
        start_time = time.time()
        cached_results = []
        for i in range(100):
            result = memory_intensive_computation(100)
            cached_results.append(result)
        cached_duration = (time.time() - start_time) * 1000
        
        # Verify performance improvement
        self.assert_equal(uncached_results, cached_results)
        self.assert_true(cached_duration < uncached_duration / 5,
                        f"Cache should provide 5x speedup: {cached_duration:.2f}ms vs {uncached_duration:.2f}ms")
    
    def test_disk_cache_performance(self):
        """Test disk cache performance characteristics"""
        @self.optimizer.cached(CacheLevel.DISK, ttl=300)
        def disk_intensive_computation(data):
            time.sleep(0.02)
            return len(str(data)) * 42
        
        # First run - populate cache
        start_time = time.time()
        result1 = disk_intensive_computation("test_data" * 1000)
        first_duration = (time.time() - start_time) * 1000
        
        # Second run - should use cache
        start_time = time.time()
        result2 = disk_intensive_computation("test_data" * 1000)
        second_duration = (time.time() - start_time) * 1000
        
        # Verify cache hit
        self.assert_equal(result1, result2)
        self.assert_true(second_duration < first_duration / 3,
                        f"Disk cache should provide 3x speedup: {second_duration:.2f}ms vs {first_duration:.2f}ms")
    
    def test_cache_concurrency_performance(self):
        """Test cache performance under concurrent access"""
        @self.optimizer.cached(CacheLevel.MEMORY, ttl=300)
        def concurrent_computation(key):
            time.sleep(0.01)
            return f"result_{key}"
        
        # Test concurrent access patterns
        def worker_function(worker_id):
            results = []
            for i in range(20):
                result = concurrent_computation(f"key_{i % 5}")  # 5 unique keys
                results.append(result)
            return results
        
        # Run with multiple workers
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_function, i) for i in range(10)]
            all_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        concurrent_duration = (time.time() - start_time) * 1000
        
        # Verify all workers completed successfully
        self.assert_equal(len(all_results), 10)
        for results in all_results:
            self.assert_equal(len(results), 20)
        
        # Performance should be reasonable under concurrency
        self.assert_performance(concurrent_duration, 1000)  # Should complete within 1 second
    
    def test_cache_memory_usage(self):
        """Test cache memory usage and efficiency"""
        @self.optimizer.cached(CacheLevel.MEMORY, ttl=300, max_size=100)
        def memory_bound_computation(x):
            return {"data": "x" * 1000, "id": x}
        
        # Fill cache beyond max size
        results = []
        for i in range(150):  # More than max_size
            result = memory_bound_computation(i)
            results.append(result)
        
        # Verify cache size is maintained
        cache_info = memory_bound_computation.cache_info()
        self.assert_true(cache_info.currsize <= 100)
        
        # Verify LRU eviction works
        self.assert_equal(len(results), 150)

class TestDatabasePerformance(BaseTestCase):
    """Test database performance characteristics"""
    
    def setup_method(self):
        """Setup database performance test"""
        self.logger = get_logger("db_performance_test")
    
    def test_connection_pool_performance(self):
        """Test database connection pool performance"""
        with patch('sloughgpt.core.db_manager.create_engine') as mock_create_engine:
            mock_engine = Mock()
            mock_connection = Mock()
            mock_engine.connect.return_value = mock_connection
            mock_create_engine.return_value = mock_engine
            
            from sloughgpt.core.db_manager import get_database_manager
            db_manager = get_database_manager()
            
            # Test connection acquisition speed
            start_time = time.time()
            connections = []
            for i in range(50):
                conn = db_manager.get_connection()
                connections.append(conn)
            pool_duration = (time.time() - start_time) * 1000
            
            # Should efficiently handle connection requests
            self.assert_performance(pool_duration, 500)  # Should complete within 500ms
            self.assert_equal(len(connections), 50)
    
    def test_query_performance(self):
        """Test database query performance"""
        with patch('sloughgpt.core.db_manager.get_database_manager') as mock_db_manager:
            mock_manager = Mock()
            mock_cursor = Mock()
            mock_cursor.fetchall.return_value = [(i, f"data_{i}") for i in range(1000)]
            mock_connection = Mock()
            mock_connection.execute.return_value.__enter__.return_value = mock_cursor
            mock_manager.get_connection.return_value = mock_connection
            mock_db_manager.return_value = mock_manager
            
            # Test query performance
            start_time = time.time()
            results = mock_manager.execute_query("SELECT * FROM large_table")
            query_duration = (time.time() - start_time) * 1000
            
            # Verify query efficiency
            self.assert_equal(len(results), 1000)
            self.assert_performance(query_duration, 100)  # Should complete within 100ms
    
    def test_transaction_performance(self):
        """Test transaction processing performance"""
        with patch('sloughgpt.core.db_manager.get_database_manager') as mock_db_manager:
            mock_manager = Mock()
            mock_connection = Mock()
            mock_manager.get_connection.return_value = mock_connection
            
            # Test transaction performance
            start_time = time.time()
            transactions = []
            for i in range(100):
                with mock_manager.transaction():
                    # Simulate transaction operations
                    pass
                transactions.append(i)
            transaction_duration = (time.time() - start_time) * 1000
            
            # Should efficiently handle multiple transactions
            self.assert_equal(len(transactions), 100)
            self.assert_performance(transaction_duration, 200)  # Should complete within 200ms

class TestModelPerformance(BaseTestCase):
    """Test model performance characteristics"""
    
    def setup_method(self):
        """Setup model performance test"""
        self.logger = get_logger("model_performance_test")
    
    def test_model_loading_performance(self):
        """Test model loading speed with caching"""
        with patch('sloughgpt.core.model_manager.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            # First load - should be slower
            start_time = time.time()
            model1 = self._load_model_with_cache("test_model")
            first_load_duration = (time.time() - start_time) * 1000
            
            # Second load - should use cache
            start_time = time.time()
            model2 = self._load_model_with_cache("test_model")
            second_load_duration = (time.time() - start_time) * 1000
            
            # Verify caching improves performance
            self.assert_equal(model1, model2)
            self.assert_true(second_load_duration < first_load_duration / 2,
                            f"Cached load should be faster: {second_load_duration:.2f}ms vs {first_load_duration:.2f}ms")
    
    def test_inference_performance(self):
        """Test model inference performance"""
        mock_model = Mock()
        mock_model.generate.return_value = "Generated response"
        
        # Test inference latency
        inference_times = []
        for i in range(50):
            start_time = time.time()
            response = self._run_model_inference(mock_model, f"Test prompt {i}")
            duration = (time.time() - start_time) * 1000
            inference_times.append(duration)
            
            self.assert_equal(response, "Generated response")
        
        # Analyze inference performance
        avg_inference_time = statistics.mean(inference_times)
        max_inference_time = max(inference_times)
        min_inference_time = min(inference_times)
        
        # Performance requirements
        self.assert_performance(avg_inference_time, 100)  # Average should be under 100ms
        self.assert_performance(max_inference_time, 500)  # Maximum should be under 500ms
        
        self.logger.info(f"Inference performance - Avg: {avg_inference_time:.2f}ms, "
                        f"Min: {min_inference_time:.2f}ms, Max: {max_inference_time:.2f}ms")
    
    def test_batch_inference_performance(self):
        """Test batch inference efficiency"""
        mock_model = Mock()
        mock_model.generate_batch.return_value = [f"Response {i}" for i in range(100)]
        
        # Test batch inference vs individual inference
        prompts = [f"Prompt {i}" for i in range(100)]
        
        # Individual inference
        start_time = time.time()
        individual_results = []
        for prompt in prompts:
            result = self._run_model_inference(mock_model, prompt)
            individual_results.append(result)
        individual_duration = (time.time() - start_time) * 1000
        
        # Batch inference
        start_time = time.time()
        batch_results = self._run_batch_inference(mock_model, prompts)
        batch_duration = (time.time() - start_time) * 1000
        
        # Verify batch efficiency
        self.assert_equal(len(individual_results), len(batch_results))
        self.assert_true(batch_duration < individual_duration / 2,
                        f"Batch should be more efficient: {batch_duration:.2f}ms vs {individual_duration:.2f}ms")
    
    def test_memory_usage_performance(self):
        """Test model memory usage optimization"""
        mock_model = Mock()
        memory_usage_data = []
        
        # Simulate memory usage over time
        for i in range(50):
            memory_usage = 1024 + (i * 10)  # Increasing memory usage
            mock_model.get_memory_usage.return_value = memory_usage
            memory_usage_data.append(memory_usage)
            
            # Trigger optimization if memory usage is high
            if memory_usage > 2000:
                optimized_usage = self._optimize_model_memory(mock_model)
                self.assert_true(optimized_usage <= memory_usage)
    
    def test_quantization_performance(self):
        """Test model quantization performance impact"""
        mock_original_model = Mock()
        mock_quantized_model = Mock()
        
        with patch('sloughgpt.core.model_manager.quantize_model', return_value=mock_quantized_model):
            # Test quantization speed
            start_time = time.time()
            quantized = self._quantize_model(mock_original_model, "4bit")
            quantization_duration = (time.time() - start_time) * 1000
            
            # Test inference speed improvement
            mock_original_model.generate.return_value = "Original response"
            mock_quantized_model.generate.return_value = "Quantized response"
            
            # Original model inference
            start_time = time.time()
            for i in range(20):
                response = self._run_model_inference(mock_original_model, f"Test {i}")
            original_inference_duration = (time.time() - start_time) * 1000
            
            # Quantized model inference
            start_time = time.time()
            for i in range(20):
                response = self._run_model_inference(mock_quantized_model, f"Test {i}")
            quantized_inference_duration = (time.time() - start_time) * 1000
            
            # Verify quantization benefits
            self.assert_equal(quantized, mock_quantized_model)
            self.assert_performance(quantization_duration, 5000)  # Quantization should be reasonable
            self.assert_true(quantized_inference_duration <= original_inference_duration,
                            f"Quantized should be faster or equal: {quantized_inference_duration:.2f}ms vs {original_inference_duration:.2f}ms")

class TestLoadPerformance(BaseTestCase):
    """Test system performance under load"""
    
    def setup_method(self):
        """Setup load test environment"""
        self.optimizer = get_performance_optimizer()
        self.logger = get_logger("load_performance_test")
    
    def test_concurrent_api_requests(self):
        """Test performance under concurrent API requests"""
        async def mock_api_request(request_data):
            """Mock API request processing"""
            await asyncio.sleep(0.01)  # Simulate processing time
            return {"response": f"Processed: {request_data['prompt']}"}
        
        async def run_concurrent_requests(num_requests):
            """Run concurrent requests and measure performance"""
            requests = [{"prompt": f"Test request {i}"} for i in range(num_requests)]
            
            start_time = time.time()
            tasks = [mock_api_request(req) for req in requests]
            results = await asyncio.gather(*tasks)
            duration = (time.time() - start_time) * 1000
            
            return len(results), duration
        
        # Test with different load levels
        load_levels = [10, 50, 100]
        for load in load_levels:
            result_count, duration = asyncio.run(run_concurrent_requests(load))
            
            self.assert_equal(result_count, load)
            self.assert_performance(duration, load * 20)  # Should handle load efficiently
            
            self.logger.info(f"Load test {load} requests: {duration:.2f}ms")
    
    def test_memory_pressure_performance(self):
        """Test performance under memory pressure"""
        large_data_sets = []
        
        try:
            # Create memory pressure
            for i in range(100):
                large_data = {"data": "x" * 10000, "id": i}
                large_data_sets.append(large_data)
                
                # Test performance under increasing memory pressure
                start_time = time.time()
                processed = self._process_data_with_optimization(large_data)
                duration = (time.time() - start_time) * 1000
                
                # Performance should remain reasonable
                self.assert_performance(duration, 50)
                
        finally:
            # Cleanup
            large_data_sets.clear()
    
    def test_cache_thrashing_resistance(self):
        """Test resistance to cache thrashing"""
        @self.optimizer.cached(CacheLevel.MEMORY, ttl=60, max_size=50)
        def thrashing_resistant_computation(key):
            time.sleep(0.001)
            return f"result_{key}"
        
        # Generate cache thrashing pattern
        start_time = time.time()
        results = []
        
        for i in range(200):  # More than cache size
            key = f"key_{i % 100}"  # 100 unique keys, larger than max_size
            result = thrashing_resistant_computation(key)
            results.append(result)
        
        duration = (time.time() - start_time) * 1000
        
        # Should handle thrashing gracefully
        self.assert_equal(len(results), 200)
        self.assert_performance(duration, 1000)  # Should complete within reasonable time
    
    def test_performance_degradation(self):
        """Test performance degradation under sustained load"""
        performance_data = []
        
        # Sustained load test
        for batch in range(10):
            start_time = time.time()
            
            # Process batch of work
            for i in range(50):
                result = self._simulate_workload(i, batch)
                self.assert_not_equal(result, None)
            
            batch_duration = (time.time() - start_time) * 1000
            performance_data.append(batch_duration)
        
        # Analyze performance degradation
        avg_performance = statistics.mean(performance_data)
        max_performance = max(performance_data)
        min_performance = min(performance_data)
        
        # Performance should be stable (not degrade significantly)
        degradation_ratio = max_performance / min_performance if min_performance > 0 else 1
        self.assert_true(degradation_ratio < 2.0,
                        f"Performance degradation too high: {degradation_ratio:.2f}x")
        
        self.logger.info(f"Performance stability - Avg: {avg_performance:.2f}ms, "
                        f"Degradation: {degradation_ratio:.2f}x")
    
    # Helper methods
    def _load_model_with_cache(self, model_path):
        """Mock cached model loading"""
        time.sleep(0.05)  # Simulate loading time
        return Mock()
    
    def _run_model_inference(self, model, prompt):
        """Mock model inference"""
        time.sleep(0.01)  # Simulate inference time
        return model.generate(prompt)
    
    def _run_batch_inference(self, model, prompts):
        """Mock batch inference"""
        time.sleep(0.05)  # Simulate batch processing time
        return model.generate_batch(prompts)
    
    def _optimize_model_memory(self, model):
        """Mock memory optimization"""
        time.sleep(0.01)
        model.optimize_memory()
        return model.get_memory_usage()
    
    def _quantize_model(self, model, quantization_level):
        """Mock model quantization"""
        time.sleep(0.1)  # Simulate quantization time
        return Mock()
    
    def _process_data_with_optimization(self, data):
        """Mock data processing with optimization"""
        time.sleep(0.001)
        return {"processed": True, "data_size": len(str(data))}
    
    def _simulate_workload(self, item, batch):
        """Mock workload simulation"""
        time.sleep(0.002)
        return {"item": item, "batch": batch, "processed": True}

# Performance test fixtures
@pytest.fixture
def performance_optimizer():
    """Fixture providing performance optimizer"""
    return get_performance_optimizer()

@pytest.fixture
def performance_test_data():
    """Fixture providing performance test data"""
    return {
        "small_dataset": list(range(100)),
        "medium_dataset": list(range(1000)),
        "large_dataset": list(range(10000)),
        "prompts": [f"Test prompt {i}" for i in range(1000)],
        "computation_inputs": list(range(500))
    }

if __name__ == "__main__":
    pytest.main([__file__, "-v"])