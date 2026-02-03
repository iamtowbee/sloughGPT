"""
Unit Tests for API Backend Components

Tests individual components in isolation:
- Model Manager
- Cache Manager  
- Batch Processor
- Dataset Manager
- Middleware Components
"""

import pytest
import asyncio
import json
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import torch

from packages.apps.apps.api_server.core.model_manager import ModelManager
from packages.apps.apps.api_server.core.cache_manager import CacheManager, CacheScope
from packages.apps.apps.api_server.core.batch_processor import BatchProcessor, BatchPriority
from packages.apps.apps.api_server.core.async_dataset_manager import AsyncDatasetManager, DatasetInfo
from packages.apps.apps.api_server.monitoring.metrics_collector import MetricsCollector, RequestMetrics
from packages.apps.apps.api_server.middleware import (
    RequestLoggingMiddleware, 
    RateLimitMiddleware, 
    AuthenticationMiddleware
)

class TestModelManager:
    """Test Model Manager functionality"""
    
    @pytest.fixture
    async def mock_cache_manager(self):
        """Create mock cache manager"""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True
        return mock_cache
    
    @pytest.fixture
    async def model_manager(self, mock_cache_manager):
        """Create model manager with mock cache"""
        with patch('packages.apps.apps.api_server.core.model_manager.GPT'):
            manager = ModelManager("test_model.pt", mock_cache_manager)
            manager.model = Mock()
            manager.model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
            manager.device = torch.device("cpu")
            manager.load_time = time.time()
            return manager
    
    async def test_model_initialization(self, model_manager):
        """Test model manager initialization"""
        assert model_manager.model_path == "test_model.pt"
        assert model_manager.cache_manager is not None
        assert model_manager.device.type == "cpu"
    
    async def test_text_generation_without_cache(self, model_manager, mock_cache_manager):
        """Test text generation when cache miss"""
        mock_cache_manager.get.return_value = None
        
        result = await model_manager.generate(
            prompt="Hello world",
            max_tokens=50,
            temperature=0.7,
            top_p=0.9
        )
        
        assert result is not None
        mock_cache_manager.get.assert_called_once()
        mock_cache_manager.set.assert_called_once()
    
    async def test_text_generation_with_cache_hit(self, model_manager, mock_cache_manager):
        """Test text generation when cache hit"""
        cached_response = "Cached response"
        mock_cache_manager.get.return_value = cached_response
        
        result = await model_manager.generate(
            prompt="Hello world",
            max_tokens=50
        )
        
        assert result == cached_response
        mock_cache_manager.get.assert_called_once()
        mock_cache_manager.set.assert_not_called()
    
    async def test_batch_generation_priority(self, model_manager, mock_cache_manager):
        """Test generation with different priorities"""
        mock_cache_manager.get.return_value = None
        
        # Test with high priority
        result = await model_manager.generate(
            prompt="Urgent request",
            priority=BatchPriority.HIGH
        )
        
        assert result is not None
        assert isinstance(result, str)

class TestCacheManager:
    """Test Cache Manager functionality"""
    
    @pytest.fixture
    async def cache_manager(self):
        """Create cache manager for testing"""
        # Use mock Redis for testing
        manager = CacheManager("redis://localhost:6379/2")
        # Don't connect to Redis for unit tests
        manager.redis = None
        return manager
    
    async def test_cache_set_and_get(self, cache_manager):
        """Test basic cache set/get operations"""
        key = "test_key"
        value = {"data": "test_value", "timestamp": time.time()}
        
        # Set value
        result = await cache_manager.set(key, value)
        assert result is False  # Redis not connected, should still work
        
        # Get value
        retrieved = await cache_manager.get(key)
        assert retrieved == value
    
    async def test_cache_with_ttl(self, cache_manager):
        """Test cache with TTL"""
        key = "ttl_test"
        value = "test_value"
        ttl = 60  # 1 minute
        
        await cache_manager.set(key, value, ttl=ttl)
        
        retrieved = await cache_manager.get(key)
        assert retrieved == value
    
    async def test_cache_scopes(self, cache_manager):
        """Test different cache scopes"""
        key = "scope_test"
        value = "scoped_value"
        
        # Test different scopes
        await cache_manager.set(key, value, scope=CacheScope.SHORT_TERM)
        await cache_manager.set(key + "_medium", value, scope=CacheScope.MEDIUM_TERM)
        await cache_manager.set(key + "_long", value, scope=CacheScope.LONG_TERM)
        
        # All should be retrievable
        assert await cache_manager.get(key) == value
        assert await cache_manager.get(key + "_medium") == value
        assert await cache_manager.get(key + "_long") == value
    
    async def test_cache_pattern_operations(self, cache_manager):
        """Test pattern-based cache operations"""
        # Set multiple keys
        await cache_manager.set("user:1:data", "user1_data")
        await cache_manager.set("user:2:data", "user2_data")
        await cache_manager.set("product:1:data", "product1_data")
        
        # Get all user data
        user_data = await cache_manager.get_pattern("user:*:data")
        assert len(user_data) == 2
        assert "user:1:data" in user_data
        assert "user:2:data" in user_data
    
    async def test_cache_statistics(self, cache_manager):
        """Test cache statistics"""
        # Perform some operations
        await cache_manager.set("stat_test1", "value1")
        await cache_manager.get("stat_test1")  # Hit
        await cache_manager.get("nonexistent")   # Miss
        
        stats = await cache_manager.get_stats()
        assert "local_cache" in stats
        assert "performance" in stats
        assert stats["performance"]["hits"] >= 1
        assert stats["performance"]["misses"] >= 1

class TestBatchProcessor:
    """Test Batch Processor functionality"""
    
    @pytest.fixture
    async def mock_model_manager(self):
        """Create mock model manager"""
        mock_manager = AsyncMock()
        mock_manager.device = torch.device("cpu")
        mock_manager.model = Mock()
        mock_manager._encode = Mock(return_value=[1, 2, 3])
        mock_manager._decode = Mock(return_value="generated_text")
        return mock_manager
    
    @pytest.fixture
    async def batch_processor(self, mock_model_manager):
        """Create batch processor"""
        processor = BatchProcessor(mock_model_manager, max_batch_size=4, max_wait_time=0.1)
        await processor.start()
        yield processor
        await processor.stop()
    
    async def test_batch_request_addition(self, batch_processor):
        """Test adding requests to batch processor"""
        request_id = await batch_processor.add_request(
            prompt="Test prompt",
            max_tokens=50,
            temperature=0.7,
            top_p=0.9
        )
        
        assert request_id is not None
        assert isinstance(request_id, str)
    
    async def test_batch_priority_handling(self, batch_processor):
        """Test priority-based request handling"""
        # Add requests with different priorities
        low_id = await batch_processor.add_request("Low priority", priority=BatchPriority.LOW)
        high_id = await batch_processor.add_request("High priority", priority=BatchPriority.HIGH)
        urgent_id = await batch_processor.add_request("Urgent priority", priority=BatchPriority.URGENT)
        
        # Check that all IDs are unique
        assert len({low_id, high_id, urgent_id}) == 3
    
    async def test_batch_statistics(self, batch_processor):
        """Test batch processing statistics"""
        await batch_processor.add_request("Stat test", priority=BatchPriority.NORMAL)
        
        stats = await batch_processor.get_stats()
        assert "is_running" in stats
        assert "pending_requests" in stats
        assert "configuration" in stats
        assert stats["configuration"]["max_batch_size"] == 4

class TestAsyncDatasetManager:
    """Test Async Dataset Manager functionality"""
    
    @pytest.fixture
    async def temp_dataset_dir(self):
        """Create temporary dataset directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    async def dataset_manager(self, temp_dataset_dir):
        """Create dataset manager with temp directory"""
        manager = AsyncDatasetManager(temp_dataset_dir)
        await manager.initialize()
        return manager
    
    async def test_dataset_initialization(self, dataset_manager):
        """Test dataset manager initialization"""
        assert dataset_manager.dataset_path.exists()
        assert dataset_manager.file_manager is not None
    
    async def test_dataset_write_and_read(self, dataset_manager):
        """Test writing and reading datasets"""
        dataset_name = "test_dataset"
        test_data = [
            {"text": "Sample 1", "metadata": {"id": 1}},
            {"text": "Sample 2", "metadata": {"id": 2}}
        ]
        
        # Write dataset
        count = await dataset_manager.write_dataset(dataset_name, test_data)
        assert count == len(test_data)
        
        # Read dataset
        read_data = await dataset_manager.read_dataset(dataset_name)
        assert len(read_data) == len(test_data)
        assert read_data[0]["text"] == test_data[0]["text"]
    
    async def test_dataset_pagination(self, dataset_manager):
        """Test dataset pagination"""
        dataset_name = "paginated_dataset"
        test_data = [{"text": f"Sample {i}"} for i in range(10)]
        
        # Write dataset
        await dataset_manager.write_dataset(dataset_name, test_data)
        
        # Test pagination
        page1 = await dataset_manager.read_dataset(dataset_name, limit=3, offset=0)
        page2 = await dataset_manager.read_dataset(dataset_name, limit=3, offset=3)
        
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0]["text"] == "Sample 0"
        assert page2[0]["text"] == "Sample 3"
    
    async def test_dataset_statistics(self, dataset_manager):
        """Test dataset statistics calculation"""
        dataset_name = "stats_dataset"
        test_data = [
            {"text": "Short"},
            {"text": "Medium length text"},
            {"text": "This is a much longer text with more characters"}
        ]
        
        await dataset_manager.write_dataset(dataset_name, test_data)
        stats = await dataset_manager.get_dataset_stats(dataset_name)
        
        assert "total_samples" in stats
        assert "avg_text_length" in stats
        assert "min_text_length" in stats
        assert "max_text_length" in stats
        assert stats["total_samples"] == len(test_data)
    
    async def test_dataset_deletion(self, dataset_manager):
        """Test dataset deletion"""
        dataset_name = "delete_dataset"
        test_data = [{"text": "To be deleted"}]
        
        # Create dataset
        await dataset_manager.write_dataset(dataset_name, test_data)
        
        # Verify it exists
        datasets = await dataset_manager.list_datasets()
        dataset_names = [d.name for d in datasets]
        assert dataset_name in dataset_names
        
        # Delete dataset
        result = await dataset_manager.delete_dataset(dataset_name)
        assert result is True
        
        # Verify it's gone
        datasets = await dataset_manager.list_datasets()
        dataset_names = [d.name for d in datasets]
        assert dataset_name not in dataset_names

class TestMetricsCollector:
    """Test Metrics Collector functionality"""
    
    @pytest.fixture
    async def metrics_collector(self):
        """Create metrics collector"""
        collector = MetricsCollector(max_history_size=100)
        await collector.start_collection(interval=1.0)
        yield collector
        await collector.stop_collection()
    
    async def test_request_recording(self, metrics_collector):
        """Test recording of request metrics"""
        request_id = "test_req_1"
        
        # Record request start
        metrics_collector.record_request_start(
            request_id=request_id,
            endpoint="/api/v1/chat/completions",
            method="POST"
        )
        
        # Record request completion
        metrics_collector.record_request_end(
            request_id=request_id,
            endpoint="/api/v1/chat/completions",
            method="POST",
            status_code=200,
            duration=0.5,
            cache_hit=True,
            tokens_generated=25
        )
        
        # Check recent metrics
        recent = metrics_collector.get_recent_metrics(minutes=5)
        assert recent["total_requests"] == 1
        assert recent["cache_hit_rate"] == 1.0
        assert recent["avg_duration"] == 0.5
    
    async def test_error_tracking(self, metrics_collector):
        """Test error tracking"""
        # Record error requests
        metrics_collector.record_request_end(
            request_id="error_req_1",
            endpoint="/api/v1/invalid",
            method="GET",
            status_code=404,
            duration=0.1
        )
        
        metrics_collector.record_request_end(
            request_id="error_req_2", 
            endpoint="/api/v1/chat/completions",
            method="POST",
            status_code=500,
            duration=2.0
        )
        
        error_summary = metrics_collector.get_error_summary(minutes=60)
        assert error_summary["total_errors"] == 2
        assert len(error_summary["errors_by_endpoint"]) == 2
    
    async def test_endpoint_statistics(self, metrics_collector):
        """Test endpoint-specific statistics"""
        # Record multiple requests for same endpoint
        for i in range(3):
            request_id = f"req_{i}"
            metrics_collector.record_request_start(request_id, "/api/v1/chat", "POST")
            metrics_collector.record_request_end(
                request_id, "/api/v1/chat", "POST", 200, 0.5 + i * 0.1
            )
        
        endpoint_summary = metrics_collector.get_endpoint_summary("/api/v1/chat")
        assert endpoint_summary["total_requests"] == 3
        assert endpoint_summary["avg_duration"] > 0.5
        assert endpoint_summary["status_codes"][200] == 3

class TestMiddlewareComponents:
    """Test individual middleware components"""
    
    @pytest.fixture
    def mock_request(self):
        """Create mock FastAPI request"""
        request = Mock()
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/api/v1/chat/completions"
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.headers = {"User-Agent": "test-agent", "X-Forwarded-For": "10.0.0.1"}
        return request
    
    @pytest.fixture
    async def mock_call_next(self):
        """Create mock call_next function"""
        response = Mock()
        response.status_code = 200
        response.headers = {}
        
        async def call_next(request):
            await asyncio.sleep(0.01)  # Simulate processing
            return response
        
        return call_next
    
    async def test_rate_limiting_middleware(self, mock_request):
        """Test rate limiting middleware"""
        rate_limit = RateLimitMiddleware(None, requests_per_minute=2)
        
        # First request should pass
        result1 = await rate_limit.dispatch(mock_request, lambda req: Mock(status_code=200))
        assert result1.status_code != 429
        
        # Second request should pass
        result2 = await rate_limit.dispatch(mock_request, lambda req: Mock(status_code=200))
        assert result2.status_code != 429
        
        # Third request should be rate limited
        result3 = await rate_limit.dispatch(mock_request, lambda req: Mock(status_code=200))
        assert result3.status_code == 429
    
    async def test_client_ip_extraction(self):
        """Test client IP extraction with various headers"""
        rate_limit = RateLimitMiddleware(None)
        
        # Test direct IP
        request1 = Mock()
        request1.client = Mock()
        request1.client.host = "192.168.1.1"
        request1.headers = {}
        assert rate_limit._get_client_ip(request1) == "192.168.1.1"
        
        # Test X-Forwarded-For
        request2 = Mock()
        request2.client = Mock()
        request2.client.host = "10.0.0.1"
        request2.headers = {"X-Forwarded-For": "203.0.113.1, 198.51.100.1"}
        assert rate_limit._get_client_ip(request2) == "203.0.113.1"
        
        # Test X-Real-IP
        request3 = Mock()
        request3.client = Mock()
        request3.client.host = "10.0.0.1"
        request3.headers = {"X-Real-IP": "198.51.100.100"}
        assert rate_limit._get_client_ip(request3) == "198.51.100.100"

# Test Utilities and Fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

class TestIntegrationFlows:
    """Test integrated flows between components"""
    
    async def test_cache_and_batch_integration(self):
        """Test cache and batch processor integration"""
        # Create mock components
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None  # Cache miss
        
        with patch('packages.apps.apps.api_server.core.model_manager.GPT'):
            manager = ModelManager("test.pt", mock_cache)
            manager.model = Mock()
            manager.model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
            manager.device = torch.device("cpu")
            
            # Enable batching
            manager.batch_processor = Mock()
            manager.batch_processor.add_request = AsyncMock(return_value="batch_result")
            
            # Test generation with batching
            result = await manager.generate("Test prompt", priority=BatchPriority.HIGH)
            
            # Should have checked cache first
            mock_cache.get.assert_called_once()
            
            # Should have tried to add to batch
            manager.batch_processor.add_request.assert_called_once()
    
    async def test_metrics_and_middleware_integration(self):
        """Test metrics collection with middleware"""
        collector = MetricsCollector()
        
        # Simulate request lifecycle
        request_id = "integration_test"
        collector.record_request_start(request_id, "/api/v1/test", "POST")
        await asyncio.sleep(0.01)
        collector.record_request_end(request_id, "/api/v1/test", "POST", 200, 0.015)
        
        # Check metrics were recorded correctly
        recent = collector.get_recent_metrics(minutes=5)
        assert recent["total_requests"] == 1
        assert recent["avg_duration"] >= 0.01

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])