#!/usr/bin/env python3
"""
SloughGPT Package Structure
A modular transformer training framework with advanced cognitive capabilities
"""

__version__ = "1.0.0"
__author__ = "SloughGPT Team"
__description__ = "Modular transformer training framework with cognitive architecture"

# Core components
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from .config import (
        SloughGPTConfig, 
        ModelConfig, 
        LearningConfig, 
        CognitiveConfig,
        PerformanceConfig,
        SecurityConfig,
        DatabaseConfig,
        LoggingConfig,
        ServerConfig
    )
    __all_config__ = [
        'SloughGPTConfig', 'ModelConfig', 'LearningConfig', 'CognitiveConfig',
        'PerformanceConfig', 'SecurityConfig', 'DatabaseConfig', 
        'LoggingConfig', 'ServerConfig'
    ]
except ImportError as e:
    print(f"⚠️  Configuration module import failed: {e}")
    __all_config__ = []

# Neural network components
try:
    from .neural_network import SloughGPT
    __all_neural__ = ['SloughGPT']
except ImportError as e:
    print(f"⚠️  Neural network module import failed: {e}")
    __all_neural__ = []

# Exception handling
try:
    from .core.exceptions import (
        SloughGPTError,
        ConfigurationError,
        DatabaseError,
        ModelError,
        APIError,
        CognitiveError,
        LearningError,
        SecurityError,
        PerformanceError,
        create_error,
        handle_exception
    )
    __all_exceptions__ = [
        'SloughGPTError', 'ConfigurationError', 'DatabaseError', 'ModelError',
        'APIError', 'CognitiveError', 'LearningError', 'SecurityError',
        'PerformanceError', 'create_error', 'handle_exception'
    ]
except ImportError as e:
    print(f"⚠️  Exceptions module import failed: {e}")
    __all_exceptions__ = []

# Database components
try:
    from .core.database import (
        Base, LearningExperience, LearningSession, KnowledgeNode, 
        KnowledgeEdge, CognitiveState, ModelCheckpoint, ApiRequest
    )
    from .core.db_manager import (
        DatabaseManager, get_database_manager, initialize_database, 
        get_db_session, db_health_check, get_db_stats
    )
    from .core.migration import DatabaseMigrator, migrate_legacy_database
    __all_database__ = [
        'Base', 'LearningExperience', 'LearningSession', 'KnowledgeNode',
        'KnowledgeEdge', 'CognitiveState', 'ModelCheckpoint', 'ApiRequest',
        'DatabaseManager', 'get_database_manager', 'initialize_database',
        'get_db_session', 'db_health_check', 'get_db_stats',
        'DatabaseMigrator', 'migrate_legacy_database'
    ]
except ImportError as e:
    print(f"⚠️  Database module import failed: {e}")
    __all_database__ = []

# Logging and error handling components
try:
    from .core.logging_system import (
        StructuredLogger, LogLevel, LogFormat, PerformanceTimer,
        ErrorTracker, get_logger, setup_logging, 
        get_error_tracker, timer
    )
    from .core.error_handling import (
        ErrorHandler, CircuitBreaker, RetryHandler, ErrorMetrics,
        CircuitBreakerConfig, RetryConfig, CircuitState,
        circuit_breaker, retry, handle_error,
        get_error_handler, DEFAULT_RETRY, DATABASE_RETRY
    )
    __all_logging__ = [
        'StructuredLogger', 'LogLevel', 'LogFormat', 'PerformanceTimer',
        'ErrorTracker', 'get_logger', 'setup_logging', 
        'get_error_tracker', 'timer'
    ]
    __all_error_handling__ = [
        'ErrorHandler', 'CircuitBreaker', 'RetryHandler', 'ErrorMetrics',
        'CircuitBreakerConfig', 'RetryConfig', 'CircuitState',
        'circuit_breaker', 'retry', 'handle_error',
        'get_error_handler', 'DEFAULT_RETRY', 'DATABASE_RETRY'
    ]
except ImportError as e:
    print(f"⚠️  Logging/Error handling module import failed: {e}")
    __all_logging__ = []
    __all_error_handling__ = []

# Security components
try:
    from .core.security import (
        SecurityConfig, SecurityMiddleware, InputValidator, RateLimiter, 
        ContentFilter, ValidationLevel, InputType,
        validate_prompt, rate_limit, get_security_middleware
    )
    __all_security__ = [
        'SecurityConfig', 'SecurityMiddleware', 'InputValidator', 'RateLimiter', 
        'ContentFilter', 'ValidationLevel', 'InputType',
        'validate_prompt', 'rate_limit', 'get_security_middleware'
    ]
except ImportError as e:
    print(f"⚠️  Security module import failed: {e}")
    __all_security__ = []

# Performance optimization components
try:
    from .core.performance import (
        PerformanceOptimizer, MemoryCache, DiskCache, ModelQuantizer,
        CacheLevel, QuantizationLevel, CacheEntry, PerformanceMetrics,
        memory_cache, disk_cache, quantized, performance_monitor,
        get_performance_optimizer, initialize_performance
    )
    __all_performance__ = [
        'PerformanceOptimizer', 'MemoryCache', 'DiskCache', 'ModelQuantizer',
        'CacheLevel', 'QuantizationLevel', 'CacheEntry', 'PerformanceMetrics',
        'memory_cache', 'disk_cache', 'quantized', 'performance_monitor',
        'get_performance_optimizer', 'initialize_performance'
    ]
except ImportError as e:
    print(f"⚠️  Performance module import failed: {e}")
    __all_performance__ = []

# Testing infrastructure components
try:
    from .core.testing import (
        run_all_tests, TestConfiguration,
        unit_test, integration_test, 
        performance_test, security_test
    )
    __all_testing__ = [
        'run_all_tests', 'TestConfiguration',
        'unit_test', 'integration_test', 
        'performance_test', 'security_test'
    ]
except ImportError as e:
    print(f"⚠️  Testing module import failed: {e}")
    __all_testing__ = []

# API Server
try:
    from .api_server import app
    __all_api__ = ['app']
except ImportError as e:
    print(f"⚠️  API server module import failed: {e}")
    __all_api__ = []

# Trainer
try:
    from .trainer import SloughGPTTrainer
    __all_trainer__ = ['SloughGPTTrainer']
except ImportError as e:
    print(f"⚠️  Trainer module import failed: {e}")
    __all_trainer__ = []

# Combine all exports
__all__ = (
    __all_config__ + 
    __all_neural__ + 
    __all_exceptions__ + 
    __all_database__ +
    __all_logging__ +
    __all_error_handling__ +
    __all_security__ +
    __all_performance__ +
    __all_api__ + 
    __all_trainer__
)

# Package initialization
def get_version():
    """Get package version"""
    return __version__

def get_package_info():
    """Get package information"""
    return {
        "name": "sloughgpt",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "components": {
            "config": len(__all_config__),
            "neural": len(__all_neural__),
            "exceptions": len(__all_exceptions__),
            "api": len(__all_api__),
            "trainer": len(__all_trainer__)
        }
    }

# Health check function
def health_check():
    """Check package health and component availability"""
    status = {
        "package": "sloughgpt",
        "version": __version__,
        "status": "healthy",
        "components": {},
        "issues": []
    }
    
    # Check each component
    components = {
        "Configuration": __all_config__,
        "Neural Network": __all_neural__,
        "Exceptions": __all_exceptions__,
        "API Server": __all_api__,
        "Trainer": __all_trainer__
    }
    
    for name, exports in components.items():
        if exports:
            status["components"][name] = "available"
        else:
            status["components"][name] = "unavailable"
            status["issues"].append(f"{name} module failed to import")
    
    if status["issues"]:
        status["status"] = "degraded"
    
    return status

if __name__ == "__main__":
    # Test package initialization
    print("🚀 SloughGPT Package Initialization")
    print("=" * 50)
    
    info = get_package_info()
    print(f"📦 Package: {info['name']} v{info['version']}")
    print(f"👤 Author: {info['author']}")
    print(f"📝 Description: {info['description']}")
    print()
    
    print("🔧 Component Status:")
    for component, count in info["components"].items():
        status_icon = "✅" if count > 0 else "❌"
        print(f"   {status_icon} {component}: {count} exports")
    
    print()
    
    # Health check
    health = health_check()
    health_icon = "✅" if health["status"] == "healthy" else "⚠️"
    print(f"{health_icon} Overall Status: {health['status']}")
    
    if health["issues"]:
        print("🔍 Issues found:")
        for issue in health["issues"]:
            print(f"   • {issue}")
    
    print()
    if health["status"] == "healthy":
        print("🎉 SloughGPT package is ready for use!")
    else:
        print("⚠️  Some components are missing - check imports")