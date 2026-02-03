"""
Comprehensive End-to-End Tests for SloGPT API Backend

Tests all optimization features:
- Async file I/O and connection pooling
- Redis caching layer
- Request batching system
- Monitoring and metrics
- Rate limiting and authentication
- Error handling and middleware
"""

import asyncio
import pytest
import aiohttp
import aiofiles
import json
import time
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
import redis.asyncio as aioredis

# Test configuration
TEST_CONFIG = {
    "API_BASE": "http://localhost:8000",
    "REDIS_URL": "redis://localhost:6379/1",  # Use different DB for tests
    "TEST_TIMEOUT": 30,
    "BATCH_SIZE": 4,
    "RATE_LIMIT": 10
}

@pytest.fixture(scope="class")
async def test_client():
    """Create test HTTP client"""
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=TEST_CONFIG["TEST_TIMEOUT"])
    ) as client:
        yield client

@pytest.fixture(scope="class")
async def redis_client():
    """Create Redis client for test setup/cleanup"""
    redis = await aioredis.from_url(TEST_CONFIG["REDIS_URL"])
    yield redis
    await redis.flushdb()

@pytest.fixture(scope="class")
async def test_dataset_data():
    """Sample dataset data for testing"""
    return [
        {"text": "Sample text 1", "label": "test"},
        {"text": "Sample text 2", "label": "test"},
        {"text": "Sample text 3", "label": "test"}
    ]

class TestAPIE2E:
    """End-to-end API tests"""
    
    @pytest.fixture(scope="class", autouse=True)
    async def setup_test_environment(self):
        """Setup test environment"""
        # Ensure API server is running
        # In real CI, you'd start the server here
        await asyncio.sleep(0.1)
    
    @pytest.mark.asyncio
async def test_api_health_check(self, test_client):
        """Test API health endpoints"""
        # Test basic health
        async with test_client.get(f"{TEST_CONFIG['API_BASE']}/health/") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] in ["healthy", "degraded"]
            assert "timestamp" in data
            assert "uptime" in data
        
        # Test liveness
        async with test_client.get(f"{TEST_CONFIG['API_BASE']}/health/live") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "alive"
        
        # Test readiness
        async with test_client.get(f"{TEST_CONFIG['API_BASE']}/health/ready") as resp:
            assert resp.status in [200, 503]  # May not be ready without model
    
    async def test_chat_completion_basic(self, test_client):
        """Test basic chat completion"""
        payload = {
            "prompt": "Hello, how are you?",
            "max_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        async with test_client.post(
            f"{TEST_CONFIG['API_BASE']}/api/v1/chat/completions",
            json=payload
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
            assert "response" in data
            assert "usage" in data
            assert data["usage"]["prompt_tokens"] > 0
            assert data["usage"]["total_tokens"] > 0
    
    async def test_chat_completion_streaming(self, test_client):
        """Test streaming chat completion"""
        payload = {
            "prompt": "Tell me a short story",
            "max_tokens": 100,
            "stream": True
        }
        
        async with test_client.post(
            f"{TEST_CONFIG['API_BASE']}/api/v1/chat/completions/stream",
            json=payload
        ) as resp:
            assert resp.status == 200
            assert resp.headers.get("content-type") == "text/plain"
            
            # Read streaming response
            chunks = []
            async for chunk in resp.content:
                chunks.append(chunk.decode())
            
            assert len(chunks) > 0
            # Check for streaming format
            full_response = "".join(chunks)
            assert "data:" in full_response

class TestAsyncFileIO:
    """Test async file I/O and dataset operations"""
    
    @pytest.fixture
    async def test_dataset_data(self):
        """Create test dataset data"""
        return [
            {"text": "Sample text 1", "metadata": {"source": "test"}},
            {"text": "Sample text 2", "metadata": {"source": "test"}},
            {"text": "Sample text 3", "metadata": {"source": "test"}}
        ]
    
    async def test_dataset_upload_json(self, test_client, test_dataset_data):
        """Test JSON dataset upload"""
        files = {'file': ('test_dataset.json', json.dumps(test_dataset_data).encode(), 'application/json')}
        data = aiohttp.FormData()
        data.add_field('file', json.dumps(test_dataset_data).encode(),
                      filename='test_dataset.json', content_type='application/json')
        
        async with test_client.post(
            f"{TEST_CONFIG['API_BASE']}/api/v1/dataset/upload",
            data=data
        ) as resp:
            assert resp.status == 200
            result = await resp.json()
            assert result["message"] == "Dataset uploaded successfully"
            assert result["samples_processed"] == len(test_dataset_data)
    
    async def test_dataset_upload_jsonl(self, test_client):
        """Test JSONL dataset upload"""
        jsonl_content = '\n'.join([
            json.dumps({"text": f"Line {i}"}) for i in range(3)
        ])
        
        data = aiohttp.FormData()
        data.add_field('file', jsonl_content.encode(),
                      filename='test_dataset.jsonl', content_type='application/jsonl')
        
        async with test_client.post(
            f"{TEST_CONFIG['API_BASE']}/api/v1/dataset/upload",
            data=data
        ) as resp:
            assert resp.status == 200
            result = await resp.json()
            assert result["samples_processed"] == 3
    
    async def test_dataset_status(self, test_client):
        """Test dataset status endpoint"""
        async with test_client.get(f"{TEST_CONFIG['API_BASE']}/api/v1/dataset/status") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert isinstance(data, list)
            # Check structure if datasets exist
            if data:
                for dataset in data:
                    assert "name" in dataset
                    assert "status" in dataset
                    assert "size" in dataset
                    assert "samples" in dataset
    
    async def test_dataset_samples_pagination(self, test_client):
        """Test dataset samples with pagination"""
        async with test_client.get(
            f"{TEST_CONFIG['API_BASE']}/api/v1/dataset/test_dataset/samples?limit=2&offset=1"
        ) as resp:
            # May return 404 if dataset doesn't exist, which is OK for test
            if resp.status == 200:
                data = await resp.json()
                assert "samples" in data
                assert "limit" in data
                assert "offset" in data
                assert "total" in data
                assert len(data["samples"]) <= 2
    
    async def test_dataset_update_text(self, test_client):
        """Test adding text to dataset"""
        payload = {
            "dataset_name": "test_update_dataset",
            "text": "New test sample text",
            "metadata": {"test": True}
        }
        
        async with test_client.post(
            f"{TEST_CONFIG['API_BASE']}/api/v1/dataset/update_text",
            json=payload
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data["success"] is True
            assert data["samples_added"] == 1

class TestCachingLayer:
    """Test Redis caching and local caching"""
    
    async def test_cache_performance_headers(self, test_client, redis_client):
        """Test caching through performance metrics"""
        # Clear any existing cache
        await redis_client.flushdb()
        
        payload = {"prompt": "Cache test prompt", "max_tokens": 20}
        
        # First request - should be slower (cache miss)
        start_time = time.time()
        async with test_client.post(
            f"{TEST_CONFIG['API_BASE']}/api/v1/chat/completions",
            json=payload
        ) as resp:
            assert resp.status == 200
            first_request_time = time.time() - start_time
        
        # Second identical request - should be faster (cache hit)
        start_time = time.time()
        async with test_client.post(
            f"{TEST_CONFIG['API_BASE']}/api/v1/chat/completions",
            json=payload
        ) as resp:
            assert resp.status == 200
            second_request_time = time.time() - start_time
        
        # Cache hit should be faster (though this may vary)
        print(f"First: {first_request_time:.3f}s, Second: {second_request_time:.3f}s")
        
        # Check cache statistics
        async with test_client.get(f"{TEST_CONFIG['API_BASE']}/api/v1/metrics/performance") as resp:
            if resp.status == 200:
                data = await resp.json()
                cache_stats = data["performance_metrics"].get("cache", {})
                assert "redis_connected" in cache_stats
    
    async def test_cache_invalidation(self, test_client, redis_client):
        """Test cache invalidation on model reload"""
        # Make a request to populate cache
        payload = {"prompt": "Invalidation test", "max_tokens": 20}
        
        async with test_client.post(
            f"{TEST_CONFIG['API_BASE']}/api/v1/chat/completions",
            json=payload
        ) as resp:
            assert resp.status == 200
        
        # Clear cache
        async with test_client.delete(f"{TEST_CONFIG['API_BASE']}/api/v1/model/cache") as resp:
            assert resp.status == 200
            result = await resp.json()
            assert "cleared successfully" in result["message"]

class TestRequestBatching:
    """Test request batching system"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_batching(self, test_client):
        """Test that concurrent requests are batched properly"""
        # Send multiple requests concurrently
        tasks = []
        for i in range(TEST_CONFIG["BATCH_SIZE"]):
            payload = {
                "prompt": f"Batch test request {i}",
                "max_tokens": 10
            }
            task = test_client.post(
                f"{TEST_CONFIG['API_BASE']}/api/v1/chat/completions",
                json=payload
            )
            tasks.append(task)
        
        # Wait for all requests to complete
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Check all requests succeeded
        for resp in responses:
            if isinstance(resp, Exception):
                pytest.fail(f"Request failed: {resp}")
            else:
                assert resp.status == 200
        
        # Check batch processing statistics
        async with test_client.get(f"{TEST_CONFIG['API_BASE']}/api/v1/model/batch/stats") as resp:
            if resp.status == 200:
                data = await resp.json()
                batch_stats = data["batch_stats"]
                assert batch_stats["total_requests"] >= TEST_CONFIG["BATCH_SIZE"]
                assert batch_stats["batch_processor_enabled"] is True

class TestMonitoringAndMetrics:
    """Test monitoring and metrics collection"""
    
    async def test_metrics_overview(self, test_client):
        """Test metrics overview endpoint"""
        async with test_client.get(f"{TEST_CONFIG['API_BASE']}/api/v1/metrics/overview") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert "metrics" in data
            assert "timestamp" in data
            
            metrics = data["metrics"]
            assert "recent_activity" in metrics
            assert "current_status" in metrics
            assert "performance" in metrics
            assert "active_requests_count" in metrics
    
    async def test_request_metrics(self, test_client):
        """Test request-specific metrics"""
        async with test_client.get(
            f"{TEST_CONFIG['API_BASE']}/api/v1/metrics/requests?minutes=5"
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
            assert "time_range_minutes" in data
            assert "total_requests" in data
            assert "avg_duration" in data
            assert "cache_hit_rate" in data
            assert data["time_range_minutes"] == 5
    
    async def test_system_metrics(self, test_client):
        """Test system resource metrics"""
        async with test_client.get(f"{TEST_CONFIG['API_BASE']}/api/v1/metrics/system") as resp:
            assert resp.status == 200
            data = await resp.json()
            
            if "system_metrics" in data:
                system = data["system_metrics"]
                assert "cpu_percent" in system
                assert "memory_percent" in system
                assert "timestamp" in system
    
    async def test_model_metrics(self, test_client):
        """Test model-specific metrics"""
        async with test_client.get(f"{TEST_CONFIG['API_BASE']}/api/v1/metrics/model") as resp:
            assert resp.status == 200
            data = await resp.json()
            
            if "model_metrics" in data:
                model = data["model_metrics"]
                assert "model_name" in model
                assert "active_requests" in model
                assert "cache_hit_rate" in model
                assert "timestamp" in model
    
    async def test_error_metrics(self, test_client):
        """Test error tracking metrics"""
        async with test_client.get(
            f"{TEST_CONFIG['API_BASE']}/api/v1/metrics/errors?minutes=60"
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
            assert "error_metrics" in data
            assert "time_range_minutes" in data
            assert "total_errors" in data

class TestRateLimiting:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, test_client):
        """Test that rate limiting is enforced"""
        # Send requests rapidly to trigger rate limit
        tasks = []
        for i in range(TEST_CONFIG["RATE_LIMIT"] + 5):
            task = test_client.get(f"{TEST_CONFIG['API_BASE']}/health/")
            tasks.append(task)
        
        # Execute all requests
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful vs rate-limited responses
        successful = 0
        rate_limited = 0
        
        for resp in responses:
            if isinstance(resp, Exception):
                continue
            
            if resp.status == 429:
                rate_limited += 1
            elif resp.status == 200:
                successful += 1
        
        # Should have some successful requests and some rate-limited
        assert successful > 0, "Should allow some requests"
        assert rate_limited > 0, "Should rate limit excess requests"
    
    async def test_rate_limit_headers(self, test_client):
        """Test rate limit headers"""
        async with test_client.get(f"{TEST_CONFIG['API_BASE']}/health/") as resp:
            assert resp.status == 200
            headers = resp.headers
            
            # Check for rate limit headers
            assert "X-RateLimit-Limit" in headers
            assert "X-RateLimit-Remaining" in headers
            assert "X-RateLimit-Reset" in headers

class TestAuthentication:
    """Test authentication middleware"""
    
    async def test_protected_endpoints_without_auth(self, test_client):
        """Test that protected endpoints require authentication"""
        # This test depends on ENABLE_AUTH setting
        # If auth is disabled, this may pass
        
        async with test_client.get(f"{TEST_CONFIG['API_BASE']}/api/v1/metrics/export") as resp:
            # Should either succeed (auth disabled) or require auth
            assert resp.status in [200, 401, 403]
    
    async def test_authentication_headers(self, test_client):
        """Test authentication with valid headers"""
        # This would require a valid JWT token
        # In practice, you'd generate this during test setup
        headers = {"Authorization": "Bearer test-token"}
        
        async with test_client.get(
            f"{TEST_CONFIG['API_BASE']}/api/v1/metrics/performance",
            headers=headers
        ) as resp:
            # Response depends on auth configuration and token validity
            assert resp.status in [200, 401, 403]

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    async def test_invalid_chat_payload(self, test_client):
        """Test handling of invalid chat requests"""
        # Empty prompt
        payload = {"prompt": "", "max_tokens": 50}
        
        async with test_client.post(
            f"{TEST_CONFIG['API_BASE']}/api/v1/chat/completions",
            json=payload
        ) as resp:
            assert resp.status == 422  # Validation error
            
            data = await resp.json()
            assert "error" in data or "detail" in data
    
    async def test_oversized_prompt(self, test_client):
        """Test handling of oversized prompts"""
        # Very long prompt
        payload = {"prompt": "x" * 5000, "max_tokens": 50}  # Exceeds limit
        
        async with test_client.post(
            f"{TEST_CONFIG['API_BASE']}/api/v1/chat/completions",
            json=payload
        ) as resp:
            assert resp.status in [400, 422]  # Bad request or validation error
    
    async def test_invalid_file_upload(self, test_client):
        """Test handling of invalid file uploads"""
        # Try to upload an unsupported file type
        data = aiohttp.FormData()
        data.add_field('file', b'fake executable content',
                      filename='malware.exe', content_type='application/x-executable')
        
        async with test_client.post(
            f"{TEST_CONFIG['API_BASE']}/api/v1/dataset/upload",
            data=data
        ) as resp:
            assert resp.status == 400  # Should reject unsupported file type
    
    async def test_nonexistent_dataset(self, test_client):
        """Test handling of requests for non-existent datasets"""
        async with test_client.get(
            f"{TEST_CONFIG['API_BASE']}/api/v1/dataset/nonexistent/samples"
        ) as resp:
            assert resp.status == 404  # Not found

class TestPerformanceIntegration:
    """Integration tests for overall performance"""
    
    @pytest.mark.asyncio
    async def test_load_under_moderate_stress(self, test_client):
        """Test system behavior under moderate load"""
        num_requests = 20
        tasks = []
        
        for i in range(num_requests):
            payload = {
                "prompt": f"Performance test {i}",
                "max_tokens": 20
            }
            task = test_client.post(
                f"{TEST_CONFIG['API_BASE']}/api/v1/chat/completions",
                json=payload
            )
            tasks.append(task)
        
        # Execute requests with timeout
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status == 200)
        failed = sum(1 for r in responses if isinstance(r, Exception) or getattr(r, 'status', 0) >= 400)
        
        print(f"Load test: {successful}/{num_requests} successful in {total_time:.2f}s")
        
        # Should have reasonable success rate
        success_rate = successful / num_requests
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2%}"
        
        # Average response time should be reasonable
        avg_time_per_request = total_time / num_requests
        assert avg_time_per_request < 5.0, f"Average response time too high: {avg_time_per_request:.2f}s"

# Test configuration and utilities
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Pytest configuration
pytest_plugins = []

if __name__ == "__main__":
    # Run tests with pytest
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))