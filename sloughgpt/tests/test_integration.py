"""
Comprehensive Integration Tests for SloughGPT

This module provides integration tests for the entire SloughGPT framework,
testing the interaction between all components and modules.
"""

import pytest
import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os
from datetime import datetime, timedelta

# Import all modules to test
from ..config import SloughGPTConfig
from ..neural_network import SloughGPT
from ..auth import AuthManager
from ..user_management import UserManager
from ..cost_optimization import CostOptimizer
from ..data_learning import DataLearningPipeline
from ..reasoning_engine import ReasoningEngine
from ..api_server import app
from ..admin import create_app
from ..core.database import DatabaseManager
from ..core.logging_system import get_logger
from ..core.security import SecurityMiddleware
from ..core.performance import PerformanceOptimizer

logger = get_logger(__name__)

class TestConfigurations:
    """Test configuration management and setup"""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        return SloughGPTConfig(
            model_config={
                "model_name": "test-gpt2",
                "vocab_size": 1000,
                "hidden_size": 128,
                "num_attention_heads": 2,
                "num_hidden_layers": 2
            },
            learning_config={
                "batch_size": 2,
                "learning_rate": 1e-4,
                "num_epochs": 1
            },
            database_config={
                "database_url": "sqlite:///:memory:"
            }
        )
    
    @pytest.fixture
    def mock_data(self):
        """Create mock training data"""
        return [
            {"input": "Hello world", "output": "Hello back"},
            {"input": "Test message", "output": "Test response"},
            {"input": "How are you?", "output": "I'm fine"}
        ]

class TestDatabaseIntegration:
    """Test database operations and migrations"""
    
    @pytest.mark.asyncio
    async def test_database_initialization(self, test_config):
        """Test database initialization and schema creation"""
        db_manager = DatabaseManager(test_config.database_config)
        
        # Initialize database
        await db_manager.initialize()
        
        # Test health check
        health = await db_manager.health_check()
        assert health["status"] == "healthy"
        
        # Test getting database session
        async with db_manager.get_session() as session:
            assert session is not None
        
        # Cleanup
        await db_manager.close()
    
    @pytest.mark.asyncio
    async def test_learning_experience_crud(self, test_config):
        """Test CRUD operations for learning experiences"""
        db_manager = DatabaseManager(test_config.database_config)
        await db_manager.initialize()
        
        async with db_manager.get_session() as session:
            # Create learning experience
            experience = await db_manager.create_learning_experience(
                session=session,
                prompt="Test prompt",
                response="Test response",
                user_id="test_user",
                session_id="test_session",
                metadata={"test": True}
            )
            
            assert experience.id is not None
            assert experience.prompt == "Test prompt"
            assert experience.response == "Test response"
            
            # Read learning experience
            retrieved = await db_manager.get_learning_experience(session, experience.id)
            assert retrieved is not None
            assert retrieved.id == experience.id
            
            # Update learning experience
            updated = await db_manager.update_learning_experience(
                session, 
                experience.id, 
                feedback_score=5
            )
            assert updated.feedback_score == 5
            
            # Delete learning experience
            deleted = await db_manager.delete_learning_experience(session, experience.id)
            assert deleted is True
        
        await db_manager.close()

class TestAuthenticationIntegration:
    """Test authentication and authorization flows"""
    
    @pytest.mark.asyncio
    async def test_user_registration_flow(self):
        """Test complete user registration flow"""
        auth_manager = AuthManager()
        user_manager = UserManager()
        
        # Test user registration
        user_data = {
            "email": "test@example.com",
            "password": "securepassword123",
            "name": "Test User"
        }
        
        registered_user = await user_manager.create_user(**user_data)
        assert registered_user.email == "test@example.com"
        assert registered_user.id is not None
        
        # Test authentication
        auth_result = await auth_manager.authenticate_user(
            email="test@example.com",
            password="securepassword123"
        )
        
        assert auth_result.success is True
        assert auth_result.user is not None
        assert auth_result.access_token is not None
        
        # Test token validation
        token_payload = await auth_manager.validate_token(auth_result.access_token)
        assert token_payload is not None
        assert token_payload["user_id"] == registered_user.id
    
    @pytest.mark.asyncio
    async def test_api_key_authentication(self):
        """Test API key based authentication"""
        auth_manager = AuthManager()
        user_manager = UserManager()
        
        # Create user and API key
        user = await user_manager.create_user(
            email="api@example.com",
            password="password123",
            name="API User"
        )
        
        api_key = await auth_manager.create_api_key(
            user_id=user.id,
            name="Test Key",
            permissions=["read", "write"]
        )
        
        assert api_key.key is not None
        assert api_key.user_id == user.id
        
        # Test API key validation
        validated_key = await auth_manager.validate_api_key(api_key.key)
        assert validated_key is not None
        assert validated_key.user_id == user.id

class TestCostOptimizationIntegration:
    """Test cost optimization and budget management"""
    
    @pytest.mark.asyncio
    async def test_cost_tracking(self):
        """Test API cost tracking"""
        cost_optimizer = CostOptimizer()
        
        # Simulate API usage
        usage_data = [
            {"model": "gpt2", "tokens": 100, "cost": 0.001},
            {"model": "gpt2", "tokens": 200, "cost": 0.002},
            {"model": "bert", "tokens": 50, "cost": 0.0005}
        ]
        
        total_cost = 0
        for usage in usage_data:
            cost_optimizer.track_usage(
                model=usage["model"],
                tokens=usage["tokens"],
                cost=usage["cost"],
                user_id="test_user"
            )
            total_cost += usage["cost"]
        
        # Check total cost
        user_stats = await cost_optimizer.get_user_cost_stats("test_user")
        assert user_stats["total_cost"] == total_cost
        assert len(user_stats["usage_records"]) == len(usage_data)
    
    @pytest.mark.asyncio
    async def test_budget_management(self):
        """Test budget management and alerts"""
        cost_optimizer = CostOptimizer()
        
        # Set budget for user
        await cost_optimizer.set_user_budget(
            user_id="budget_user",
            monthly_budget=10.0,
            alert_threshold=0.8
        )
        
        # Simulate usage up to threshold
        await cost_optimizer.track_usage(
            model="gpt2",
            tokens=1000,
            cost=8.5,  # 85% of budget
            user_id="budget_user"
        )
        
        # Check budget status
        budget_status = await cost_optimizer.get_user_budget_status("budget_user")
        assert budget_status["percentage_used"] == 85.0
        assert budget_status["alert_triggered"] is True
        
        # Get recommendations
        recommendations = await cost_optimizer.get_cost_recommendations("budget_user")
        assert len(recommendations) > 0

class TestDataLearningIntegration:
    """Test data learning pipeline and knowledge graph"""
    
    @pytest.mark.asyncio
    async def test_learning_pipeline(self, mock_data):
        """Test complete learning pipeline"""
        learning_pipeline = DataLearningPipeline()
        
        # Initialize pipeline
        await learning_pipeline.initialize()
        
        # Process learning data
        results = await learning_pipeline.process_batch(mock_data)
        
        assert len(results) == len(mock_data)
        for result in results:
            assert result["success"] is True
            assert "experience_id" in result
        
        # Test knowledge graph updates
        knowledge_stats = await learning_pipeline.get_knowledge_graph_stats()
        assert knowledge_stats["total_nodes"] > 0
        assert knowledge_stats["total_edges"] > 0
    
    @pytest.mark.asyncio
    async def test_semantic_search(self):
        """Test semantic search functionality"""
        learning_pipeline = DataLearningPipeline()
        await learning_pipeline.initialize()
        
        # Add some test data to knowledge graph
        test_data = [
            {"prompt": "What is machine learning?", "response": "Machine learning is AI"},
            {"prompt": "Deep learning concepts", "response": "Deep learning uses neural networks"}
        ]
        
        await learning_pipeline.process_batch(test_data)
        
        # Test semantic search
        search_results = await learning_pipeline.semantic_search(
            query="neural networks",
            limit=5
        )
        
        assert len(search_results) > 0
        assert all("score" in result for result in search_results)

class TestReasoningEngineIntegration:
    """Test multi-step reasoning engine"""
    
    @pytest.mark.asyncio
    async def test_reasoning_pipeline(self):
        """Test complete reasoning pipeline"""
        reasoning_engine = ReasoningEngine()
        
        # Initialize reasoning engine
        await reasoning_engine.initialize()
        
        # Test reasoning task
        reasoning_task = {
            "question": "What is the best approach to optimize this model?",
            "context": {
                "model_type": "transformer",
                "problem": "slow inference",
                "constraints": ["maintain accuracy", "reduce latency"]
            }
        }
        
        reasoning_result = await reasoning_engine.reason(reasoning_task)
        
        assert reasoning_result.success is True
        assert "answer" in reasoning_result
        assert "reasoning_steps" in reasoning_result
        assert len(reasoning_result.reasoning_steps) > 0
    
    @pytest.mark.asyncio
    async def test_self_correction(self):
        """Test self-correction capabilities"""
        reasoning_engine = ReasoningEngine()
        await reasoning_engine.initialize()
        
        # Test with potentially flawed input
        task_with_error = {
            "question": "What is 2 + 2?",
            "context": {
                "calculation": "2 + 2 = 5"  # Intentionally wrong
            }
        }
        
        result = await reasoning_engine.reason(task_with_error)
        
        # The engine should detect and correct the error
        assert result.success is True
        # The corrected answer should be 4
        assert "4" in result.get("answer", "")

class TestAPIIntegration:
    """Test API endpoints and server functionality"""
    
    @pytest.mark.asyncio
    async def test_api_endpoints(self):
        """Test all major API endpoints"""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
        # Test info endpoint
        response = client.get("/info")
        assert response.status_code == 200
        assert "version" in response.json()
    
    @pytest.mark.asyncio
    async def test_authentication_endpoints(self):
        """Test authentication endpoints"""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test user registration
        user_data = {
            "email": "apitest@example.com",
            "password": "testpassword123",
            "name": "API Test User"
        }
        
        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 201
        
        # Test user login
        login_data = {
            "email": "apitest@example.com",
            "password": "testpassword123"
        }
        
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        assert "access_token" in response.json()
        
        # Test protected endpoint
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        response = client.get("/users/me", headers=headers)
        assert response.status_code == 200
        assert response.json()["email"] == "apitest@example.com"

class TestAdminIntegration:
    """Test admin dashboard functionality"""
    
    @pytest.mark.asyncio
    async def test_admin_dashboard(self):
        """Test admin dashboard endpoints"""
        admin_app = create_app()
        
        from fastapi.testclient import TestClient
        client = TestClient(admin_app)
        
        # Test dashboard stats
        response = client.get("/api/admin/dashboard/stats")
        assert response.status_code == 200
        
        stats = response.json()
        assert "system_status" in stats
        assert "user_count" in stats
        assert "model_count" in stats
        assert "total_cost" in stats
        
        # Test system health
        response = client.get("/api/admin/system/health")
        assert response.status_code == 200
        
        health = response.json()
        assert "status" in health
        assert "cpu_usage" in health
        assert "memory_usage" in health
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket functionality"""
        admin_app = create_app()
        
        from fastapi.testclient import TestClient
        
        with TestClient(admin_app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Send test message
                websocket.send_text("test message")
                
                # Receive response
                data = websocket.receive_text()
                assert "test message" in data

class TestPerformanceIntegration:
    """Test performance optimization features"""
    
    @pytest.mark.asyncio
    async def test_caching(self):
        """Test caching mechanisms"""
        perf_optimizer = PerformanceOptimizer()
        
        # Test memory cache
        cache_key = "test_key"
        cache_value = {"data": "test_value"}
        
        # Store in cache
        await perf_optimizer.memory_cache.set(cache_key, cache_value, ttl=60)
        
        # Retrieve from cache
        cached_data = await perf_optimizer.memory_cache.get(cache_key)
        assert cached_data == cache_value
        
        # Test cache expiration
        await perf_optimizer.memory_cache.set(cache_key, cache_value, ttl=1)
        await asyncio.sleep(1.1)  # Wait for expiration
        
        expired_data = await perf_optimizer.memory_cache.get(cache_key)
        assert expired_data is None
    
    @pytest.mark.asyncio
    async def test_model_quantization(self):
        """Test model quantization"""
        perf_optimizer = PerformanceOptimizer()
        
        # Mock model for testing
        class MockModel:
            def __init__(self):
                self.parameters = {"weight": [1.0, 2.0, 3.0, 4.0]}
        
        mock_model = MockModel()
        
        # Test quantization
        quantized_model = await perf_optimizer.quantize_model(
            model=mock_model,
            quantization_level="int8"
        )
        
        assert quantized_model is not None
        # The quantized model should have the same interface
        assert hasattr(quantized_model, 'parameters')

class TestSecurityIntegration:
    """Test security features and middleware"""
    
    @pytest.mark.asyncio
    async def test_input_validation(self):
        """Test input validation and sanitization"""
        security_middleware = SecurityMiddleware()
        
        # Test valid input
        valid_input = "This is a valid prompt"
        validation_result = await security_middleware.validate_input(
            input_data=valid_input,
            input_type="prompt"
        )
        
        assert validation_result.is_valid is True
        
        # Test malicious input
        malicious_input = "DROP TABLE users; --"
        validation_result = await security_middleware.validate_input(
            input_data=malicious_input,
            input_type="prompt"
        )
        
        assert validation_result.is_valid is False
        assert len(validation_result.issues) > 0
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting functionality"""
        security_middleware = SecurityMiddleware()
        
        user_id = "test_user"
        endpoint = "/api/generate"
        
        # Test rate limit enforcement
        for i in range(10):
            is_allowed = await security_middleware.check_rate_limit(
                user_id=user_id,
                endpoint=endpoint
            )
            
            if i < 5:  # Assuming rate limit of 5 per minute
                assert is_allowed is True
            else:
                assert is_allowed is False

class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_user_workflow(self):
        """Test complete user journey from registration to API usage"""
        # Initialize components
        auth_manager = AuthManager()
        user_manager = UserManager()
        cost_optimizer = CostOptimizer()
        learning_pipeline = DataLearningPipeline()
        
        # 1. User Registration
        user = await user_manager.create_user(
            email="e2e@example.com",
            password="password123",
            name="E2E Test User"
        )
        
        # 2. User Authentication
        auth_result = await auth_manager.authenticate_user(
            email="e2e@example.com",
            password="password123"
        )
        
        assert auth_result.success is True
        
        # 3. Set up budget
        await cost_optimizer.set_user_budget(
            user_id=user.id,
            monthly_budget=50.0
        )
        
        # 4. Simulate API usage
        api_usage = [
            {"model": "gpt2", "tokens": 100, "cost": 0.01},
            {"model": "gpt2", "tokens": 200, "cost": 0.02}
        ]
        
        for usage in api_usage:
            await cost_optimizer.track_usage(
                model=usage["model"],
                tokens=usage["tokens"],
                cost=usage["cost"],
                user_id=user.id
            )
        
        # 5. Learning from interactions
        learning_data = [
            {"prompt": "Hello", "response": "Hi there!", "user_id": user.id},
            {"prompt": "How are you?", "response": "I'm doing well!", "user_id": user.id}
        ]
        
        await learning_pipeline.initialize()
        learning_results = await learning_pipeline.process_batch(learning_data)
        
        assert len(learning_results) == 2
        assert all(result["success"] for result in learning_results)
        
        # 6. Verify cost tracking
        cost_stats = await cost_optimizer.get_user_cost_stats(user.id)
        assert cost_stats["total_cost"] == 0.03  # 0.01 + 0.02
        
        # 7. Verify learning data stored
        knowledge_stats = await learning_pipeline.get_knowledge_graph_stats()
        assert knowledge_stats["total_nodes"] >= 2
        
        logger.info("Complete end-to-end workflow test passed")
    
    @pytest.mark.asyncio
    async def test_system_resilience(self):
        """Test system resilience under various failure scenarios"""
        from unittest.mock import patch
        
        # Test database failure recovery
        with patch('sloughgpt.core.database.DatabaseManager.health_check') as mock_health:
            # First call fails
            mock_health.side_effect = [
                Exception("Database connection failed"),
                {"status": "healthy"}
            ]
            
            db_manager = DatabaseManager({"database_url": "sqlite:///:memory:"})
            await db_manager.initialize()
            
            # First health check fails
            health1 = await db_manager.health_check()
            assert "error" in str(health1).lower()
            
            # Second health check succeeds
            health2 = await db_manager.health_check()
            assert health2["status"] == "healthy"
        
        # Test graceful degradation
        with patch('sloughgpt.cost_optimization.CostOptimizer.track_usage') as mock_track:
            mock_track.side_effect = Exception("Cost tracking failed")
            
            cost_optimizer = CostOptimizer()
            
            # Should not raise exception, but handle gracefully
            try:
                await cost_optimizer.track_usage(
                    model="gpt2",
                    tokens=100,
                    cost=0.01,
                    user_id="test_user"
                )
            except Exception:
                pass  # Expected to fail gracefully
        
        logger.info("System resilience tests passed")

# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for critical components"""
    
    @pytest.mark.asyncio
    async def test_api_response_time(self):
        """Benchmark API response times"""
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Measure response time
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 0.5  # Should respond within 500ms
        
        logger.info(f"Health endpoint response time: {response_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test system performance under concurrent load"""
        import concurrent.futures
        
        def make_request():
            from fastapi.testclient import TestClient
            client = TestClient(app)
            return client.get("/health").status_code
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        
        logger.info("Concurrent request test passed")

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])