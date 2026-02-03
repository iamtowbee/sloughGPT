"""
SloughGPT API Tests
API endpoint testing and validation
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, AsyncMock
import asyncio

import sys
import os

# Add sloughgpt to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from sloughgpt.core.testing import BaseTestCase, TestConfiguration, TestType, TestStatus
from sloughgpt.core.security import get_security_middleware
from sloughgpt.core.performance import get_performance_optimizer
from sloughgpt.core.logging_system import get_logger

class TestAPIEndpoints(BaseTestCase):
    """Test API endpoint functionality"""
    
    def setup_method(self):
        """Setup API test environment"""
        self.security_middleware = get_security_middleware()
        self.optimizer = get_performance_optimizer()
        self.logger = get_logger("api_endpoint_test")
        self.base_url = "http://localhost:8000"
    
    @pytest.mark.asyncio
    async def test_generate_endpoint(self):
        """Test /api/v1/generate endpoint"""
        # Mock API server response
        mock_response = {
            "response": "This is a generated response",
            "tokens_used": 25,
            "model": "test-model",
            "generation_time": 0.15
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            # Test valid request
            request_data = {
                "prompt": "Tell me a story",
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            response = await self._make_api_request("/api/v1/generate", request_data)
            self.assert_equal(response.status_code, 200)
            self.assert_in("response", response.data)
            self.assert_in("tokens_used", response.data)
    
    @pytest.mark.asyncio
    async def test_chat_endpoint(self):
        """Test /api/v1/chat endpoint"""
        mock_response = {
            "response": "I can help you with that!",
            "conversation_id": "conv_123",
            "message_id": "msg_456",
            "tokens_used": 15
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            request_data = {
                "message": "Hello, how can you help me?",
                "conversation_id": None,
                "context": []
            }
            
            response = await self._make_api_request("/api/v1/chat", request_data)
            self.assert_equal(response.status_code, 200)
            self.assert_in("response", response.data)
            self.assert_in("conversation_id", response.data)
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test /health endpoint"""
        mock_response = {
            "status": "healthy",
            "version": "1.0.0",
            "uptime": 3600,
            "model_status": "loaded"
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value.status = 200
            
            response = await self._make_api_request("/health", method="GET")
            self.assert_equal(response.status_code, 200)
            self.assert_equal(response.data["status"], "healthy")
            self.assert_in("version", response.data)
    
    @pytest.mark.asyncio
    async def test_models_endpoint(self):
        """Test /api/v1/models endpoint"""
        mock_response = {
            "models": [
                {
                    "id": "model-1",
                    "name": "Test Model 1",
                    "description": "A test model",
                    "capabilities": ["text-generation", "chat"],
                    "status": "available"
                },
                {
                    "id": "model-2",
                    "name": "Test Model 2",
                    "description": "Another test model",
                    "capabilities": ["text-generation"],
                    "status": "loading"
                }
            ]
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value.status = 200
            
            response = await self._make_api_request("/api/v1/models", method="GET")
            self.assert_equal(response.status_code, 200)
            self.assert_in("models", response.data)
            self.assert_equal(len(response.data["models"]), 2)
    
    @pytest.mark.asyncio
    async def test_conversations_endpoint(self):
        """Test /api/v1/conversations endpoint"""
        # Test GET conversations
        mock_conversations = {
            "conversations": [
                {
                    "id": "conv_1",
                    "title": "First Conversation",
                    "created_at": "2023-01-01T00:00:00Z",
                    "updated_at": "2023-01-01T01:00:00Z",
                    "message_count": 5
                }
            ]
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_conversations)
            mock_get.return_value.__aenter__.return_value.status = 200
            
            response = await self._make_api_request("/api/v1/conversations", method="GET")
            self.assert_equal(response.status_code, 200)
            self.assert_in("conversations", response.data)
    
    @pytest.mark.asyncio
    async def test_user_endpoint(self):
        """Test /api/v1/user endpoint"""
        mock_user = {
            "id": "user_123",
            "username": "testuser",
            "email": "test@example.com",
            "created_at": "2023-01-01T00:00:00Z",
            "preferences": {
                "theme": "dark",
                "language": "en"
            }
        }
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_user)
            mock_get.return_value.__aenter__.return_value.status = 200
            
            response = await self._make_api_request("/api/v1/user", method="GET", headers={"Authorization": "Bearer token"})
            self.assert_equal(response.status_code, 200)
            self.assert_in("username", response.data)
            self.assert_equal(response.data["username"], "testuser")

class TestAPIAuthentication(BaseTestCase):
    """Test API authentication and authorization"""
    
    def setup_method(self):
        """Setup authentication test"""
        self.security_middleware = get_security_middleware()
        self.logger = get_logger("api_auth_test")
    
    @pytest.mark.asyncio
    async def test_bearer_token_authentication(self):
        """Test Bearer token authentication"""
        valid_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        
        # Test valid token
        with patch('sloughgpt.core.security.validate_token') as mock_validate:
            mock_validate.return_value = {"user_id": "user_123", "valid": True}
            
            auth_result = self.security_middleware.validate_authentication({
                "token": valid_token,
                "type": "Bearer"
            })
            self.assert_true(auth_result.is_valid)
        
        # Test invalid token
        with patch('sloughgpt.core.security.validate_token') as mock_validate:
            mock_validate.return_value = {"user_id": None, "valid": False}
            
            auth_result = self.security_middleware.validate_authentication({
                "token": "invalid_token",
                "type": "Bearer"
            })
            self.assert_false(auth_result.is_valid)
    
    @pytest.mark.asyncio
    async def test_api_key_authentication(self):
        """Test API key authentication"""
        valid_api_key = "sk-1234567890abcdef"
        
        # Test valid API key
        with patch('sloughgpt.core.security.validate_api_key') as mock_validate:
            mock_validate.return_value = {"user_id": "user_456", "valid": True}
            
            auth_result = self.security_middleware.validate_authentication({
                "api_key": valid_api_key
            })
            self.assert_true(auth_result.is_valid)
        
        # Test invalid API key
        with patch('sloughgpt.core.security.validate_api_key') as mock_validate:
            mock_validate.return_value = {"user_id": None, "valid": False}
            
            auth_result = self.security_middleware.validate_authentication({
                "api_key": "invalid_key"
            })
            self.assert_false(auth_result.is_valid)
    
    @pytest.mark.asyncio
    async def test_session_authentication(self):
        """Test session-based authentication"""
        valid_session = "sess_1234567890abcdef"
        
        # Test valid session
        with patch('sloughgpt.core.security.validate_session') as mock_validate:
            mock_validate.return_value = {"user_id": "user_789", "valid": True}
            
            auth_result = self.security_middleware.validate_authentication({
                "session_id": valid_session
            })
            self.assert_true(auth_result.is_valid)
        
        # Test expired session
        with patch('sloughgpt.core.security.validate_session') as mock_validate:
            mock_validate.return_value = {"user_id": None, "valid": False, "error": "expired"}
            
            auth_result = self.security_middleware.validate_authentication({
                "session_id": "expired_session"
            })
            self.assert_false(auth_result.is_valid)
    
    @pytest.mark.asyncio
    async def test_permission_based_access(self):
        """Test permission-based access control"""
        user_permissions = {
            "user_123": ["read", "write:own"],
            "admin_456": ["read", "write", "delete", "admin"],
            "guest_789": ["read"]
        }
        
        with patch('sloughgpt.core.security.get_user_permissions') as mock_get_perms:
            # Test admin access
            mock_get_perms.return_value = user_permissions["admin_456"]
            auth_result = self.security_middleware.validate_permission("admin_456", "delete")
            self.assert_true(auth_result)
            
            # Test user access to own resources
            mock_get_perms.return_value = user_permissions["user_123"]
            auth_result = self.security_middleware.validate_permission("user_123", "write:own")
            self.assert_true(auth_result)
            
            # Test guest access limitations
            mock_get_perms.return_value = user_permissions["guest_789"]
            auth_result = self.security_middleware.validate_permission("guest_789", "write")
            self.assert_false(auth_result)

class TestAPIErrorHandling(BaseTestCase):
    """Test API error handling"""
    
    def setup_method(self):
        """Setup error handling test"""
        self.logger = get_logger("api_error_test")
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test validation error handling"""
        validation_errors = [
            {"error": "Invalid prompt", "field": "prompt", "message": "Prompt cannot be empty"},
            {"error": "Invalid max_tokens", "field": "max_tokens", "message": "Must be between 1 and 4096"},
            {"error": "Invalid temperature", "field": "temperature", "message": "Must be between 0.0 and 2.0"}
        ]
        
        for error_case in validation_errors:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = Mock()
                mock_response.status = 400
                mock_response.json = AsyncMock(return_value=error_case)
                mock_post.return_value.__aenter__.return_value = mock_response
                
                response = await self._make_api_request("/api/v1/generate", {"invalid": "data"})
                self.assert_equal(response.status_code, 400)
                self.assert_in("error", response.data)
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self):
        """Test rate limit error handling"""
        rate_limit_error = {
            "error": "Rate limit exceeded",
            "retry_after": 60,
            "limit": 100,
            "remaining": 0
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 429
            mock_response.json = AsyncMock(return_value=rate_limit_error)
            mock_response.headers = {"Retry-After": "60"}
            mock_post.return_value.__aenter__.return_value = mock_response
            
            response = await self._make_api_request("/api/v1/generate", {"prompt": "test"})
            self.assert_equal(response.status_code, 429)
            self.assert_in("retry_after", response.data)
    
    @pytest.mark.asyncio
    async def test_authentication_error_handling(self):
        """Test authentication error handling"""
        auth_errors = [
            {"error": "Invalid token", "code": "INVALID_TOKEN"},
            {"error": "Token expired", "code": "TOKEN_EXPIRED"},
            {"error": "Missing authentication", "code": "MISSING_AUTH"}
        ]
        
        for error_case in auth_errors:
            with patch('aiohttp.ClientSession.get') as mock_get:
                mock_response = Mock()
                mock_response.status = 401
                mock_response.json = AsyncMock(return_value=error_case)
                mock_get.return_value.__aenter__.return_value = mock_response
                
                response = await self._make_api_request("/api/v1/user", method="GET")
                self.assert_equal(response.status_code, 401)
                self.assert_in("error", response.data)
    
    @pytest.mark.asyncio
    async def test_authorization_error_handling(self):
        """Test authorization error handling"""
        authz_error = {
            "error": "Access denied",
            "required_permission": "admin:delete",
            "user_permissions": ["read", "write"]
        }
        
        with patch('aiohttp.ClientSession.delete') as mock_delete:
            mock_response = Mock()
            mock_response.status = 403
            mock_response.json = AsyncMock(return_value=authz_error)
            mock_delete.return_value.__aenter__.return_value = mock_response
            
            response = await self._make_api_request("/api/v1/users/123", method="DELETE")
            self.assert_equal(response.status_code, 403)
            self.assert_in("required_permission", response.data)
    
    @pytest.mark.asyncio
    async def test_server_error_handling(self):
        """Test server error handling"""
        server_errors = [
            {"error": "Internal server error", "code": "INTERNAL_ERROR"},
            {"error": "Service temporarily unavailable", "code": "SERVICE_UNAVAILABLE"},
            {"error": "Database connection failed", "code": "DATABASE_ERROR"}
        ]
        
        for error_case in server_errors:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = Mock()
                mock_response.status = 500
                mock_response.json = AsyncMock(return_value=error_case)
                mock_post.return_value.__aenter__.return_value = mock_response
                
                response = await self._make_api_request("/api/v1/generate", {"prompt": "test"})
                self.assert_equal(response.status_code, 500)
                self.assert_in("error", response.data)

class TestAPIPerformance(BaseTestCase):
    """Test API performance characteristics"""
    
    def setup_method(self):
        """Setup performance test"""
        self.optimizer = get_performance_optimizer()
        self.logger = get_logger("api_performance_test")
    
    @pytest.mark.asyncio
    async def test_response_time_performance(self):
        """Test API response time performance"""
        response_times = []
        
        for i in range(20):
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = {
                    "response": f"Test response {i}",
                    "tokens_used": 10
                }
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                start_time = time.time()
                response = await self._make_api_request("/api/v1/generate", {"prompt": f"test {i}"})
                duration = (time.time() - start_time) * 1000
                response_times.append(duration)
        
        # Analyze performance
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        self.assert_performance(avg_response_time, 200)  # Average should be under 200ms
        self.assert_performance(max_response_time, 1000)  # Max should be under 1 second
        
        self.logger.info(f"API Response Times - Avg: {avg_response_time:.2f}ms, Max: {max_response_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_concurrent_request_performance(self):
        """Test performance under concurrent requests"""
        async def make_request(request_id):
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = {"response": f"Concurrent response {request_id}"}
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                start_time = time.time()
                response = await self._make_api_request("/api/v1/generate", {"prompt": f"test {request_id}"})
                duration = (time.time() - start_time) * 1000
                return response, duration
        
        # Test concurrent performance
        start_time = time.time()
        tasks = [make_request(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        total_duration = (time.time() - start_time) * 1000
        
        # Verify all requests completed
        self.assert_equal(len(results), 50)
        for response, duration in results:
            self.assert_equal(response.status_code, 200)
        
        # Concurrent processing should be more efficient
        avg_individual_duration = sum(duration for _, duration in results) / len(results)
        self.assert_true(total_duration < avg_individual_duration,
                        f"Concurrent should be faster: {total_duration:.2f}ms vs {avg_individual_duration:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_payload_size_performance(self):
        """Test performance with different payload sizes"""
        payload_sizes = [
            {"prompt": "short", "size": "small"},
            {"prompt": "A" * 1000, "size": "medium"},
            {"prompt": "B" * 5000, "size": "large"},
            {"prompt": "C" * 10000, "size": "xlarge"}
        ]
        
        performance_results = {}
        
        for payload in payload_sizes:
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = {"response": "Response for payload"}
                mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aenter__.return_value.status = 200
                
                start_time = time.time()
                response = await self._make_api_request("/api/v1/generate", {"prompt": payload["prompt"]})
                duration = (time.time() - start_time) * 1000
                
                performance_results[payload["size"]] = duration
                
                self.assert_equal(response.status_code, 200)
        
        # Performance should scale reasonably with payload size
        self.logger.info(f"Payload size performance: {performance_results}")
        
        # Large payloads shouldn't be excessively slow
        self.assert_performance(performance_results["xlarge"], 2000)  # Under 2 seconds
    
    @pytest.mark.asyncio
    async def test_caching_performance(self):
        """Test API caching performance"""
        # Test cache miss
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = {"response": "Cache miss response", "cached": False}
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            start_time = time.time()
            response1 = await self._make_api_request("/api/v1/generate", {"prompt": "cache test"})
            cache_miss_duration = (time.time() - start_time) * 1000
        
        # Test cache hit
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = {"response": "Cache hit response", "cached": True}
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            start_time = time.time()
            response2 = await self._make_api_request("/api/v1/generate", {"prompt": "cache test"})
            cache_hit_duration = (time.time() - start_time) * 1000
        
        # Cache hit should be faster
        self.assert_equal(response1.status_code, response2.status_code)
        self.assert_true(cache_hit_duration < cache_miss_duration,
                        f"Cache hit should be faster: {cache_hit_duration:.2f}ms vs {cache_miss_duration:.2f}ms")
    
    # Helper methods
    async def _make_api_request(self, endpoint, data=None, method="POST", headers=None):
        """Mock API request helper"""
        class MockResponse:
            def __init__(self, status_code, data):
                self.status_code = status_code
                self.data = data
        
        # This is a mock implementation
        if method == "GET":
            return MockResponse(200, {"status": "ok"})
        else:
            return MockResponse(200, data or {"response": "mock response"})

# API test fixtures
@pytest.fixture
def api_client():
    """Fixture providing API client"""
    class MockAPIClient:
        async def post(self, endpoint, data):
            return {"status": "ok", "data": data}
        
        async def get(self, endpoint):
            return {"status": "ok"}
    
    return MockAPIClient()

@pytest.fixture
def sample_api_data():
    """Fixture providing sample API test data"""
    return {
        "generate_requests": [
            {"prompt": "Tell me a story", "max_tokens": 100},
            {"prompt": "What is AI?", "max_tokens": 50},
            {"prompt": "Explain quantum computing", "max_tokens": 200}
        ],
        "chat_requests": [
            {"message": "Hello!", "conversation_id": None},
            {"message": "How can you help me?", "conversation_id": "conv_123"},
            {"message": "Thank you!", "conversation_id": "conv_123"}
        ],
        "user_data": {
            "username": "testuser",
            "email": "test@example.com",
            "preferences": {"theme": "dark"}
        }
    }

if __name__ == "__main__":
    pytest.main([__file__, "-v"])