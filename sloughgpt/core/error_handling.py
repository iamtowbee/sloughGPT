"""
SloughGPT Error Handling Middleware
Advanced error handling, recovery, and circuit breaker functionality
"""

import time
import asyncio
from typing import Dict, Any, Optional, Callable, List, Union, Type
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum

from .exceptions import (
    SloughGPTError, SloughGPTErrorCode, create_error,
    DatabaseError, ModelError, APIError, PerformanceError
)
from .logging_system import get_logger, track_exception

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open" # Testing if failure is resolved

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5          # Open after N failures
    recovery_timeout: float = 60.0      # Try recovery after N seconds
    expected_exception: Type[Exception] = Exception
    success_threshold: int = 2          # Successes needed to close circuit
    
@dataclass 
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0             # Base delay in seconds
    max_delay: float = 30.0             # Maximum delay
    exponential_base: float = 2.0         # Exponential backoff base
    jitter: bool = True                   # Add random jitter to delays
    retry_on: List[Type[Exception]] = field(default_factory=lambda: [Exception])

@dataclass
class ErrorMetrics:
    """Error handling metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    average_response_time: float = 0.0
    last_error_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        return 1.0 - self.success_rate

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.logger = get_logger(f"circuit_breaker.{name}")
        self.metrics = ErrorMetrics()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker"""
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._call_sync(func, *args, **kwargs)
            return sync_wrapper
    
    async def _call_async(self, func: Callable, *args, **kwargs):
        """Handle async function call"""
        start_time = time.time()
        
        try:
            if self.state == CircuitState.OPEN:
                if not self._should_attempt_reset():
                    raise create_error(
                        PerformanceError,
                        f"Circuit breaker '{self.name}' is OPEN",
                        SloughGPTErrorCode.CONCURRENT_REQUESTS_EXCEEDED,
                        context={"circuit_state": self.state.value}
                    )
                
                self.state = CircuitState.HALF_OPEN
            
            result = await func(*args, **kwargs)
            
            # Success
            self._on_success()
            self.metrics.successful_requests += 1
            duration = (time.time() - start_time) * 1000
            self.logger.performance(
                f"circuit_breaker_success_{self.name}",
                duration,
                state=self.state.value,
                function=func.__name__
            )
            
            return result
            
        except Exception as e:
            # Failure
            self._on_failure()
            self.metrics.failed_requests += 1
            duration = (time.time() - start_time) * 1000
            
            self.logger.error(
                f"Circuit breaker failure: {self.name}",
                function=func.__name__,
                error_type=e.__class__.__name__,
                error_message=str(e),
                duration_ms=duration,
                circuit_state=self.state.value,
                failure_count=self.failure_count
            )
            
            # Track exception
            track_exception(e, {
                "circuit_breaker": self.name,
                "function": func.__name__,
                "circuit_state": self.state.value
            })
            
            raise
    
    def _call_sync(self, func: Callable, *args, **kwargs):
        """Handle sync function call"""
        start_time = time.time()
        
        try:
            if self.state == CircuitState.OPEN:
                if not self._should_attempt_reset():
                    raise create_error(
                        PerformanceError,
                        f"Circuit breaker '{self.name}' is OPEN",
                        SloughGPTErrorCode.CONCURRENT_REQUESTS_EXCEEDED,
                        context={"circuit_state": self.state.value}
                    )
                
                self.state = CircuitState.HALF_OPEN
            
            result = func(*args, **kwargs)
            
            # Success
            self._on_success()
            self.metrics.successful_requests += 1
            duration = (time.time() - start_time) * 1000
            self.logger.performance(
                f"circuit_breaker_success_{self.name}",
                duration,
                state=self.state.value,
                function=func.__name__
            )
            
            return result
            
        except Exception as e:
            # Failure
            self._on_failure()
            self.metrics.failed_requests += 1
            duration = (time.time() - start_time) * 1000
            
            self.logger.error(
                f"Circuit breaker failure: {self.name}",
                function=func.__name__,
                error_type=e.__class__.__name__,
                error_message=str(e),
                duration_ms=duration,
                circuit_state=self.state.value,
                failure_count=self.failure_count
            )
            
            # Track exception
            track_exception(e, {
                "circuit_breaker": self.name,
                "function": func.__name__,
                "circuit_state": self.state.value
            })
            
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        return (time.time() - (self.last_failure_time or 0)) >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                self.logger.info(f"Circuit breaker '{self.name}' closed after recovery")
        elif self.state == CircuitState.CLOSED:
            # Already in good state
            pass
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.success_count = 0
            self.logger.warning(f"Circuit breaker '{self.name}' re-opened after failed recovery")
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.logger.error(f"Circuit breaker '{self.name}' opened after {self.failure_count} failures")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold
            },
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "success_rate": self.metrics.success_rate,
                "error_rate": self.metrics.error_rate
            }
        }

class RetryHandler:
    """Advanced retry handler with exponential backoff"""
    
    def __init__(self, config: RetryConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.logger = get_logger(f"retry_handler.{name}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with retry logic"""
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._retry_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._retry_sync(func, *args, **kwargs)
            return sync_wrapper
    
    async def _retry_async(self, func: Callable, *args, **kwargs):
        """Retry async function"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                # Check if this exception type should be retried
                if not any(isinstance(e, exc_type) for exc_type in self.config.retry_on):
                    raise
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    
                    self.logger.warning(
                        f"Retry attempt {attempt + 1}/{self.config.max_attempts} failed",
                        function=func.__name__,
                        error_type=e.__class__.__name__,
                        error_message=str(e),
                        delay_seconds=delay,
                        attempt=attempt + 1
                    )
                    
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        f"All retry attempts exhausted for {func.__name__}",
                        function=func.__name__,
                        total_attempts=self.config.max_attempts,
                        final_error=str(last_exception)
                    )
        
        raise last_exception
    
    def _retry_sync(self, func: Callable, *args, **kwargs):
        """Retry sync function"""
        import time as time_module
        
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                # Check if this exception type should be retried
                if not any(isinstance(e, exc_type) for exc_type in self.config.retry_on):
                    raise
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    
                    self.logger.warning(
                        f"Retry attempt {attempt + 1}/{self.config.max_attempts} failed",
                        function=func.__name__,
                        error_type=e.__class__.__name__,
                        error_message=str(e),
                        delay_seconds=delay,
                        attempt=attempt + 1
                    )
                    
                    time_module.sleep(delay)
                else:
                    self.logger.error(
                        f"All retry attempts exhausted for {func.__name__}",
                        function=func.__name__,
                        total_attempts=self.config.max_attempts,
                        final_error=str(last_exception)
                    )
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter"""
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # 50-100% of base delay
        
        return delay

class ErrorHandler:
    """Centralized error handling and recovery"""
    
    def __init__(self, name: str = "global"):
        self.name = name
        self.logger = get_logger(f"error_handler.{name}")
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handlers: Dict[str, RetryHandler] = {}
        self.error_handlers: Dict[Type[Exception], Callable] = {}
        
        # Register default error handlers
        self._register_default_handlers()
    
    def circuit_breaker(self, config: CircuitBreakerConfig, name: Optional[str] = None):
        """Decorator for circuit breaker"""
        def decorator(func: Callable) -> Callable:
            cb_name = name or f"{func.__module__}.{func.__name__}"
            cb = CircuitBreaker(config, cb_name)
            self.circuit_breakers[cb_name] = cb
            return cb(func)
        return decorator
    
    def retry(self, config: RetryConfig, name: Optional[str] = None):
        """Decorator for retry handler"""
        def decorator(func: Callable) -> Callable:
            retry_name = name or f"{func.__module__}.{func.__name__}"
            rh = RetryHandler(config, retry_name)
            self.retry_handlers[retry_name] = rh
            return rh(func)
        return decorator
    
    def handle_error(self, exception_type: Type[Exception]):
        """Decorator to register custom error handler"""
        def decorator(handler: Callable):
            self.error_handlers[exception_type] = handler
            return handler
        return decorator
    
    def _register_default_handlers(self):
        """Register default error handlers"""
        
        @self.handle_error(DatabaseError)
        def handle_database_error(error: DatabaseError, context: Optional[Dict[str, Any]] = None):
            """Handle database errors with automatic recovery"""
            self.logger.error("Database error occurred", error=error.to_dict(), context=context)
            
            # Could implement automatic database reconnection here
            if "connection" in str(error.message).lower():
                self.logger.info("Attempting database reconnection...")
                # Database reconnection logic would go here
        
        @self.handle_error(ModelError)
        def handle_model_error(error: ModelError, context: Optional[Dict[str, Any]] = None):
            """Handle model inference errors"""
            self.logger.error("Model inference error occurred", error=error.to_dict(), context=context)
            
            # Could implement model reload or fallback logic here
            if "not_initialized" in str(error.message).lower():
                self.logger.info("Attempting model reinitialization...")
                # Model reinitialization logic would go here
        
        @self.handle_error(PerformanceError)
        def handle_performance_error(error: PerformanceError, context: Optional[Dict[str, Any]] = None):
            """Handle performance-related errors"""
            self.logger.warning("Performance issue detected", error=error.to_dict(), context=context)
            
            # Could implement performance optimization here
            if error.error_code == SloughGPTErrorCode.RESPONSE_TIME_EXCEEDED:
                self.logger.info("Response time exceeded, consider optimization")
    
    def process_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None):
        """Process exception through registered handlers"""
        # Track the exception
        track_exception(exception, context)
        
        # Find appropriate handler
        for exc_type, handler in self.error_handlers.items():
            if isinstance(exception, exc_type):
                try:
                    return handler(exception, context)
                except Exception as handler_error:
                    self.logger.error(
                        "Error handler itself failed",
                        handler_type=exc_type.__name__,
                        handler_error=str(handler_error),
                        original_exception=str(exception)
                    )
        
        # No specific handler found
        self.logger.error(
            "Unhandled exception occurred",
            exception_type=exception.__class__.__name__,
            exception_message=str(exception),
            context=context
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive error handling status"""
        return {
            "error_handler": {
                "name": self.name,
                "registered_handlers": len(self.error_handlers)
            },
            "circuit_breakers": {
                name: cb.get_status() 
                for name, cb in self.circuit_breakers.items()
            },
            "retry_handlers": {
                name: {
                    "max_attempts": rh.config.max_attempts,
                    "base_delay": rh.config.base_delay,
                    "retry_on": [exc.__name__ for exc in rh.config.retry_on]
                }
                for name, rh in self.retry_handlers.items()
            }
        }

# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None

def get_error_handler(name: str = "global") -> ErrorHandler:
    """Get or create global error handler"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler(name)
    return _global_error_handler

# Convenience decorators
def circuit_breaker(config: CircuitBreakerConfig, name: Optional[str] = None):
    """Circuit breaker decorator"""
    return get_error_handler().circuit_breaker(config, name)

def retry(config: RetryConfig, name: Optional[str] = None):
    """Retry decorator"""
    return get_error_handler().retry(config, name)

def handle_error(exception_type: Type[Exception]):
    """Error handler registration decorator"""
    return get_error_handler().handle_error(exception_type)

# Predefined configurations
DEFAULT_CIRCUIT_BREAKER = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception=Exception
)

DEFAULT_RETRY = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True
)

DATABASE_RETRY = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=10.0,
    exponential_base=1.5,
    jitter=True,
    retry_on=[DatabaseError]
)

MODEL_RETRY = RetryConfig(
    max_attempts=2,
    base_delay=0.1,
    max_delay=5.0,
    exponential_base=2.0,
    jitter=False,
    retry_on=[ModelError]
)