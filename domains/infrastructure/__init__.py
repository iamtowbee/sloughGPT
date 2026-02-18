"""
Data & Infrastructure Domain

This domain contains all components related to data management,
infrastructure services, deployment, and system configuration.
"""

from .base import InfrastructureDomain
from .cache import CacheManager
from .config import ConfigurationManager
from .database import DatabaseManager
from .deployment import DeploymentManager
from .rag import RAGSystem

try:
    from .vector_store import VectorStore, VectorIndex, Document
    _VECTOR_STORE_AVAILABLE = True
except ImportError:
    _VECTOR_STORE_AVAILABLE = False
    VectorStore = None
    VectorIndex = None
    Document = None

try:
    from .hauls_store import HaulsStore
    _HAULS_STORE_AVAILABLE = True
except ImportError:
    _HAULS_STORE_AVAILABLE = False
    HaulsStore = None

try:
    from .performance import PerformanceMonitor, PerformanceOptimizer
    _PERFORMANCE_AVAILABLE = True
except ImportError:
    _PERFORMANCE_AVAILABLE = False
    PerformanceMonitor = None
    PerformanceOptimizer = None

try:
    from .llm_integration import LLMIntegration, OpenAIProvider, AnthropicProvider, LocalProvider
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False
    LLMIntegration = None
    OpenAIProvider = None
    AnthropicProvider = None
    LocalProvider = None

__all__ = [
    "InfrastructureDomain",
    "DatabaseManager",
    "CacheManager",
    "DeploymentManager",
    "ConfigurationManager",
    "VectorStore",
    "VectorIndex",
    "Document",
    "PerformanceMonitor",
    "PerformanceOptimizer",
    "HaulsStore",
    "LLMIntegration",
    "OpenAIProvider",
    "AnthropicProvider",
    "LocalProvider",
]
