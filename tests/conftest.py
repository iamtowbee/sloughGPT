"""
Pytest configuration and shared fixtures
"""

import pytest
from typing import Generator
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def api_base_url() -> str:
    """Base URL for API tests"""
    return "http://localhost:8000"


@pytest.fixture
def test_user() -> dict:
    """Test user credentials"""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123"
    }


@pytest.fixture
def test_model() -> str:
    """Test model ID"""
    return "gpt2"


@pytest.fixture
def test_dataset() -> str:
    """Test dataset ID"""
    return "wikitext-103"


@pytest.fixture
def test_training_config() -> dict:
    """Test training configuration"""
    return {
        "name": "Test Training",
        "model": "gpt2",
        "dataset": "wikitext-103",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-5
    }


class TestHelpers:
    """Helper methods for tests"""
    
    @staticmethod
    def assert_response_success(response, status_code=200):
        """Assert response is successful"""
        assert response.status_code == status_code, f"Expected {status_code}, got {response.status_code}"
    
    @staticmethod
    def assert_has_keys(data: dict, keys: list):
        """Assert data has required keys"""
        for key in keys:
            assert key in data, f"Missing key: {key}"


@pytest.fixture
def helpers() -> TestHelpers:
    """Test helper methods"""
    return TestHelpers()
