"""
SloughGPT Python SDK
A Python client library for the SloughGPT API.

Usage:
    from sloughgpt_sdk import SloughGPTClient
    from sloughgpt_sdk.models import GenerationResult, ChatMessage
    
    # Or for WebSocket streaming
    from sloughgpt_sdk.websocket import WebSocketClient
    
    # Or for caching
    from sloughgpt_sdk.cache import InMemoryCache
    
    # CLI
    # sloughgpt-cli generate "Hello"
"""

__version__ = "1.0.0"
__author__ = "SloughGPT"
__email__ = "dev@sloughgpt.ai"
__url__ = "https://github.com/sloughgpt/sloughgpt"

import sys
import importlib.util
import os

_package_dir = os.path.dirname(__file__)
_models_path = os.path.join(_package_dir, "models.py")

_spec = importlib.util.spec_from_file_location("sloughgpt_sdk.models", _models_path)
_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models)

sys.modules["sloughgpt_sdk.models"] = _models

GenerateRequest = _models.GenerateRequest
ChatMessage = _models.ChatMessage
ChatRequest = _models.ChatRequest
BatchRequest = _models.BatchRequest
BatchResult = _models.BatchResult
GenerationResult = _models.GenerationResult
ChatResult = _models.ChatResult
ModelInfo = _models.ModelInfo
DatasetInfo = _models.DatasetInfo
HealthStatus = _models.HealthStatus
SystemInfo = _models.SystemInfo
MetricsData = _models.MetricsData

import sys
sys.modules["sloughgpt_sdk"].GenerateRequest = GenerateRequest
sys.modules["sloughgpt_sdk"].ChatMessage = ChatMessage
sys.modules["sloughgpt_sdk"].ChatRequest = ChatRequest
sys.modules["sloughgpt_sdk"].BatchRequest = BatchRequest
sys.modules["sloughgpt_sdk"].BatchResult = BatchResult
sys.modules["sloughgpt_sdk"].GenerationResult = GenerationResult
sys.modules["sloughgpt_sdk"].ChatResult = ChatResult
sys.modules["sloughgpt_sdk"].ModelInfo = ModelInfo
sys.modules["sloughgpt_sdk"].DatasetInfo = DatasetInfo
sys.modules["sloughgpt_sdk"].HealthStatus = HealthStatus
sys.modules["sloughgpt_sdk"].SystemInfo = SystemInfo
sys.modules["sloughgpt_sdk"].MetricsData = MetricsData

from sloughgpt_sdk.client import SloughGPTClient

__all__ = [
    "SloughGPTClient",
    "GenerateRequest",
    "ChatMessage",
    "ChatRequest",
    "BatchRequest",
    "BatchResult",
    "GenerationResult",
    "ChatResult",
    "ModelInfo",
    "DatasetInfo",
    "HealthStatus",
    "SystemInfo",
    "MetricsData",
]
