"""
Test Configuration and Fixtures

This module provides shared test configuration and fixtures for all SloughGPT tests.
"""

import pytest
import asyncio
import tempfile
import os
from typing import Dict, Any, AsyncGenerator
from unittest.mock import Mock, AsyncMock
import logging

# Suppress logging during tests
logging.getLogger("sloughgpt").setLevel(logging.CRITICAL)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def temp_db_file() -> AsyncGenerator[str, None]:
    """Create a temporary database file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        yield f"sqlite:///{db_path}"
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)

@pytest.fixture
def test_config_dict() -> Dict[str, Any]:
    """Create test configuration dictionary"""
    return {
        "model_config": {
            "model_name": "test-gpt2",
            "vocab_size": 1000,
            "hidden_size": 128,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "max_position_embeddings": 512,
            "dropout": 0.1
        },
        "learning_config": {
            "batch_size": 2,
            "learning_rate": 1e-4,
            "num_epochs": 1,
            "warmup_steps": 10,
            "weight_decay": 0.01
        },
        "cognitive_config": {
            "enable_reasoning": True,
            "enable_learning": True,
            "max_reasoning_steps": 5,
            "context_window": 2048
        },
        "database_config": {
            "database_url": "sqlite:///:memory:",
            "pool_size": 1,
            "max_overflow": 0
        },
        "security_config": {
            "jwt_secret_key": "test-secret-key",
            "jwt_algorithm": "HS256",
            "bcrypt_rounds": 4
        },
        "performance_config": {
            "enable_caching": True,
            "cache_ttl": 60,
            "enable_quantization": False
        },
        "logging_config": {
            "level": "WARNING",
            "enable_file_logging": False
        }
    }

@pytest.fixture
def mock_model_data():
    """Create mock model training data"""
    return [
        {"input": "Hello world", "output": "Hello back", "metadata": {"source": "test"}},
        {"input": "Test message", "output": "Test response", "metadata": {"source": "test"}},
        {"input": "How are you?", "output": "I'm fine", "metadata": {"source": "test"}},
        {"input": "What is AI?", "output": "Artificial Intelligence", "metadata": {"source": "test"}}
    ]

@pytest.fixture
def mock_user_data():
    """Create mock user data"""
    return {
        "email": "test@example.com",
        "password": "testpassword123",
        "name": "Test User",
        "role": "user"
    }

@pytest.fixture
def mock_api_usage_data():
    """Create mock API usage data"""
    return [
        {"model": "gpt2", "tokens": 100, "cost": 0.001, "user_id": "test_user"},
        {"model": "gpt2", "tokens": 200, "cost": 0.002, "user_id": "test_user"},
        {"model": "bert", "tokens": 50, "cost": 0.0005, "user_id": "test_user"}
    ]

@pytest.fixture
def mock_learning_data():
    """Create mock learning experience data"""
    return [
        {
            "prompt": "What is machine learning?",
            "response": "Machine learning is a subset of AI",
            "user_id": "test_user",
            "session_id": "test_session",
            "metadata": {"source": "test"}
        },
        {
            "prompt": "Explain deep learning",
            "response": "Deep learning uses neural networks with multiple layers",
            "user_id": "test_user", 
            "session_id": "test_session",
            "metadata": {"source": "test"}
        }
    ]

class MockModel:
    """Mock model for testing"""
    
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 128):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.training = False
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def forward(self, x):
        # Simple mock forward pass
        return x
    
    def parameters(self):
        return {"weight": [1.0] * self.hidden_size}

class MockTokenizer:
    """Mock tokenizer for testing"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {str(i): i for i in range(vocab_size)}
    
    def encode(self, text: str):
        # Simple mock encoding
        return [hash(word) % self.vocab_size for word in text.split()]
    
    def decode(self, tokens):
        # Simple mock decoding
        return " ".join(str(t) for t in tokens)

@pytest.fixture
def mock_model():
    """Create a mock model"""
    return MockModel()

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer"""
    return MockTokenizer()

@pytest.fixture
def mock_database_session():
    """Create a mock database session"""
    session = AsyncMock()
    session.add = Mock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.close = AsyncMock()
    return session

@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection"""
    websocket = AsyncMock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.receive_text = AsyncMock(return_value="test message")
    return websocket

# Pytest markers for different test categories
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )

# Test utilities
async def create_test_user(user_manager, **kwargs):
    """Helper to create a test user"""
    default_data = {
        "email": "test@example.com",
        "password": "testpassword123",
        "name": "Test User"
    }
    default_data.update(kwargs)
    return await user_manager.create_user(**default_data)

async def create_test_api_key(auth_manager, user_id: str, **kwargs):
    """Helper to create a test API key"""
    default_data = {
        "name": "Test Key",
        "permissions": ["read", "write"]
    }
    default_data.update(kwargs)
    return await auth_manager.create_api_key(user_id=user_id, **default_data)

def assert_valid_response(response, expected_status: int = 200):
    """Assert that response is valid"""
    assert response.status_code == expected_status
    assert response.headers["content-type"] is not None

def assert_error_response(response, expected_status: int = 400):
    """Assert that response contains an error"""
    assert response.status_code >= 400
    data = response.json()
    assert "error" in data or "detail" in data

# Performance testing utilities
import time
from contextlib import asynccontextmanager

@asynccontextmanager
async def measure_time(operation_name: str):
    """Measure execution time of an operation"""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print(f"{operation_name}: {duration:.3f}s")

async def run_concurrent_tasks(coro_func, num_tasks: int, *args, **kwargs):
    """Run the same coroutine function concurrently"""
    tasks = [coro_func(*args, **kwargs) for _ in range(num_tasks)]
    return await asyncio.gather(*tasks, return_exceptions=True)