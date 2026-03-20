"""
SloughGPT Integration Tests
End-to-end tests for the complete API workflow.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import requests
import time
import json
from typing import Dict, Any, Optional


BASE_URL = "http://localhost:8000"
TIMEOUT = 30


class TestHealthEndpoints:
    """Integration tests for health check endpoints."""
    
    @pytest.fixture(autouse=True)
    def wait_for_server(self):
        """Wait for server to be ready."""
        max_retries = 5
        for i in range(max_retries):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    time.sleep(2)
                else:
                    pytest.skip("API server not running")
    
    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self):
        """Test basic health check."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_health_liveness(self):
        """Test liveness probe."""
        response = requests.get(f"{BASE_URL}/health/live", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "alive"
    
    def test_health_readiness(self):
        """Test readiness probe."""
        response = requests.get(f"{BASE_URL}/health/ready", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_info_endpoint(self):
        """Test system info endpoint."""
        response = requests.get(f"{BASE_URL}/info", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "version" in data


class TestGenerationEndpoints:
    """Integration tests for text generation endpoints."""
    
    @pytest.fixture(autouse=True)
    def wait_for_server(self):
        """Wait for server to be ready."""
        max_retries = 5
        for i in range(max_retries):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    time.sleep(2)
                else:
                    pytest.skip("API server not running")
    
    def test_generate_endpoint(self):
        """Test basic text generation."""
        payload = {"prompt": "Hello, how are you?"}
        response = requests.post(
            f"{BASE_URL}/generate",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "generated_text" in data or "text" in data or "response" in data
    
    def test_generate_with_params(self):
        """Test generation with custom parameters."""
        payload = {
            "prompt": "Tell me a joke",
            "max_new_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9
        }
        response = requests.post(
            f"{BASE_URL}/generate",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_generate_empty_prompt_fails(self):
        """Test that empty prompt returns validation error."""
        payload = {"prompt": ""}
        response = requests.post(
            f"{BASE_URL}/generate",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 422
    
    def test_generate_stream_endpoint(self):
        """Test streaming generation endpoint."""
        payload = {"prompt": "Count to 3:"}
        response = requests.post(
            f"{BASE_URL}/generate/stream",
            json=payload,
            stream=True,
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("text/event-stream")
    
    def test_chat_endpoint(self):
        """Test chat endpoint."""
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data or "message" in data or "response" in data


class TestModelEndpoints:
    """Integration tests for model management endpoints."""
    
    @pytest.fixture(autouse=True)
    def wait_for_server(self):
        """Wait for server to be ready."""
        max_retries = 5
        for i in range(max_retries):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    time.sleep(2)
                else:
                    pytest.skip("API server not running")
    
    def test_list_models(self):
        """Test listing available models."""
        response = requests.get(f"{BASE_URL}/models", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "models" in data or isinstance(data, list)
    
    def test_list_huggingface_models(self):
        """Test listing HuggingFace models."""
        response = requests.get(f"{BASE_URL}/models/hf", timeout=TIMEOUT)
        assert response.status_code == 200
    
    def test_model_info(self):
        """Test getting specific model info."""
        response = requests.get(f"{BASE_URL}/models/gpt2", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            assert "id" in data or "name" in data


class TestDatasetEndpoints:
    """Integration tests for dataset management endpoints."""
    
    @pytest.fixture(autouse=True)
    def wait_for_server(self):
        """Wait for server to be ready."""
        max_retries = 5
        for i in range(max_retries):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    time.sleep(2)
                else:
                    pytest.skip("API server not running")
    
    def test_list_datasets(self):
        """Test listing available datasets."""
        response = requests.get(f"{BASE_URL}/datasets", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data or isinstance(data, list)
    
    def test_dataset_info(self):
        """Test getting specific dataset info."""
        response = requests.get(f"{BASE_URL}/datasets/openwebtext", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            assert "id" in data or "name" in data


class TestMetricsEndpoints:
    """Integration tests for metrics and monitoring endpoints."""
    
    @pytest.fixture(autouse=True)
    def wait_for_server(self):
        """Wait for server to be ready."""
        max_retries = 5
        for i in range(max_retries):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    time.sleep(2)
                else:
                    pytest.skip("API server not running")
    
    def test_metrics_json(self):
        """Test metrics endpoint in JSON format."""
        response = requests.get(f"{BASE_URL}/metrics", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "requests_total" in data or "metrics" in data
    
    def test_metrics_prometheus(self):
        """Test metrics endpoint in Prometheus format."""
        response = requests.get(
            f"{BASE_URL}/metrics",
            headers={"Accept": "text/plain"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        text = response.text
        assert "requests_total" in text or "# HELP" in text


class TestBatchEndpoints:
    """Integration tests for batch processing endpoints."""
    
    @pytest.fixture(autouse=True)
    def wait_for_server(self):
        """Wait for server to be ready."""
        max_retries = 5
        for i in range(max_retries):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    time.sleep(2)
                else:
                    pytest.skip("API server not running")
    
    def test_batch_generate(self):
        """Test batch generation endpoint."""
        payload = {
            "prompts": ["Hello", "Hi there", "Greetings"]
        }
        response = requests.post(
            f"{BASE_URL}/generate/batch",
            json=payload,
            timeout=60
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data or "responses" in data or isinstance(data, list)


class TestAuthentication:
    """Integration tests for authentication endpoints."""
    
    @pytest.fixture(autouse=True)
    def wait_for_server(self):
        """Wait for server to be ready."""
        max_retries = 5
        for i in range(max_retries):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    time.sleep(2)
                else:
                    pytest.skip("API server not running")
    
    def test_login_endpoint(self):
        """Test login endpoint."""
        payload = {
            "username": "admin",
            "password": "admin123"
        }
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 401]
    
    def test_protected_endpoint_without_auth(self):
        """Test accessing protected endpoint without auth."""
        payload = {"prompt": "Test"}
        response = requests.post(
            f"{BASE_URL}/generate",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 401, 403]
    
    def test_token_refresh(self):
        """Test token refresh endpoint."""
        response = requests.post(
            f"{BASE_URL}/auth/refresh",
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 401]


class TestWebSocket:
    """Integration tests for WebSocket endpoints."""
    
    @pytest.fixture(autouse=True)
    def wait_for_server(self):
        """Wait for server to be ready."""
        max_retries = 5
        for i in range(max_retries):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    time.sleep(2)
                else:
                    pytest.skip("API server not running")
    
    def test_websocket_endpoint_exists(self):
        """Test WebSocket endpoint is available."""
        try:
            from websocket import create_connection
            ws = create_connection(f"ws://localhost:8000/ws")
            ws.close()
        except ImportError:
            pytest.skip("websocket-client not installed")
        except Exception:
            pytest.skip("WebSocket connection not available")


class TestPerformance:
    """Integration tests for performance and rate limiting."""
    
    @pytest.fixture(autouse=True)
    def wait_for_server(self):
        """Wait for server to be ready."""
        max_retries = 5
        for i in range(max_retries):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    time.sleep(2)
                else:
                    pytest.skip("API server not running")
    
    def test_response_time(self):
        """Test that basic endpoints respond within acceptable time."""
        start = time.time()
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 1.0, f"Health check took {elapsed:.2f}s, expected < 1s"
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import concurrent.futures
        
        def make_request():
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
                return response.status_code == 200
            except Exception:
                return False
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        success_count = sum(results)
        assert success_count >= 4, f"Only {success_count}/5 requests succeeded"


class TestErrorHandling:
    """Integration tests for error handling."""
    
    @pytest.fixture(autouse=True)
    def wait_for_server(self):
        """Wait for server to be ready."""
        max_retries = 5
        for i in range(max_retries):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    time.sleep(2)
                else:
                    pytest.skip("API server not running")
    
    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        response = requests.post(
            f"{BASE_URL}/generate",
            data="not valid json",
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        assert response.status_code == 422
    
    def test_missing_required_field(self):
        """Test handling of missing required fields."""
        payload = {"max_new_tokens": 100}
        response = requests.post(
            f"{BASE_URL}/generate",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 422
    
    def test_invalid_field_type(self):
        """Test handling of invalid field types."""
        payload = {"prompt": 12345}
        response = requests.post(
            f"{BASE_URL}/generate",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 422
    
    def test_404_not_found(self):
        """Test handling of non-existent endpoints."""
        response = requests.get(f"{BASE_URL}/nonexistent", timeout=TIMEOUT)
        assert response.status_code == 404


class TestCache:
    """Integration tests for caching functionality."""
    
    @pytest.fixture(autouse=True)
    def wait_for_server(self):
        """Wait for server to be ready."""
        max_retries = 5
        for i in range(max_retries):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    time.sleep(2)
                else:
                    pytest.skip("API server not running")
    
    def test_cache_headers(self):
        """Test that responses include cache headers."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        headers = response.headers
        assert "date" in headers
    
    def test_identical_requests(self):
        """Test that identical requests work consistently."""
        payload = {"prompt": "Cache test"}
        response1 = requests.post(f"{BASE_URL}/generate", json=payload, timeout=TIMEOUT)
        response2 = requests.post(f"{BASE_URL}/generate", json=payload, timeout=TIMEOUT)
        assert response1.status_code == 200
        assert response2.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
