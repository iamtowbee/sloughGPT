"""
SloughGPT Exception Classes
Structured error handling for better error management and debugging
"""

import traceback
from typing import Optional, Dict, Any, List
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)

class SloughGPTErrorCode(Enum):
    """Standardized error codes for SloughGPT"""
    # Configuration errors (1000-1099)
    CONFIG_INVALID = "CONFIG_1000"
    CONFIG_VALIDATION_FAILED = "CONFIG_1001"
    
    # Database errors (2000-2099)
    DATABASE_CONNECTION_FAILED = "DATABASE_2000"
    DATABASE_QUERY_FAILED = "DATABASE_2001"
    DATABASE_INTEGRITY_ERROR = "DATABASE_2002"
    DATABASE_TIMEOUT = "DATABASE_2003"
    
    # Model errors (3000-3099)
    MODEL_NOT_INITIALIZED = "MODEL_3000"
    MODEL_LOAD_FAILED = "MODEL_3001"
    MODEL_INFERENCE_FAILED = "MODEL_3002"
    MODEL_QUANTIZATION_FAILED = "MODEL_3003"
    
    # API errors (4000-4099)
    INVALID_REQUEST_FORMAT = "API_4000"
    REQUEST_VALIDATION_FAILED = "API_4001"
    RATE_LIMIT_EXCEEDED = "API_4002"
    AUTHENTICATION_FAILED = "API_4003"
    
    # Cognitive system errors (5000-5099)
    COGNITIVE_PROCESSING_FAILED = "COGNITIVE_5000"
    REASONING_CHAIN_FAILED = "COGNITIVE_5001"
    MEMORY_CAPACITY_EXCEEDED = "COGNITIVE_5002"
    
    # Learning system errors (6000-6099)
    LEARNING_UPDATE_FAILED = "LEARNING_6000"
    EXPERIENCE_BUFFER_FULL = "LEARNING_6001"
    PARAMETER_UPDATE_FAILED = "LEARNING_6002"
    
    # Security errors (7000-7099)
    INPUT_SANITIZATION_FAILED = "SECURITY_7000"
    SECURITY_VALIDATION_FAILED = "SECURITY_7001"
    UNAUTHORIZED_ACCESS = "SECURITY_7002"
    
    # Performance errors (8000-8099)
    MEMORY_LEAK_DETECTED = "PERFORMANCE_8000"
    CACHE_MISS_BURST = "PERFORMANCE_8001"
    RESPONSE_TIME_EXCEEDED = "PERFORMANCE_8002"
    CONCURRENT_REQUESTS_EXCEEDED = "PERFORMANCE_8003"

class SloughGPTError(Exception):
    """Base exception for all SloughGPT errors"""
    
    def __init__(self, 
                 message: str, 
                 error_code: Optional[SloughGPTErrorCode] = None,
                 details: Optional[Dict[str, Any]] = None,
                 cause: Optional[Exception] = None,
                 context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        self.context = context or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON serialization"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code.value if self.error_code else None,
            "details": self.details,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None
        }
    
    def to_json(self) -> str:
        """Convert error to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def log_error(self, additional_info: Optional[Dict[str, Any]] = None):
        """Log the error with structured information"""
        error_data = self.to_dict()
        if additional_info:
            error_data.update(additional_info)
        
        # Log based on error severity
        if self.error_code:
            error_code_str = self.error_code.value
            if error_code_str.startswith('SECURITY'):
                logger.error(f"SECURITY ERROR: {self.error_code}: {self.message}")
            elif error_code_str.startswith('PERFORMANCE'):
                logger.warning(f"PERFORMANCE ISSUE: {self.error_code}: {self.message}")
            else:
                logger.error(f"SloughGPT Error {self.error_code}: {self.message}")
        else:
            logger.error(f"SloughGPT Error: {self.message}")
        
        logger.debug(f"Error details: {error_data}")

class ConfigurationError(SloughGPTError):
    """Configuration related errors"""
    pass

class DatabaseError(SloughGPTError):
    """Database operation errors"""
    pass

class ModelError(SloughGPTError):
    """Model inference errors"""
    pass

class APIError(SloughGPTError):
    """API request/response errors"""
    pass

class CognitiveError(SloughGPTError):
    """Cognitive processing errors"""
    pass

class LearningError(SloughGPTError):
    """Learning system errors"""
    pass

class SecurityError(SloughGPTError):
    """Security and validation errors"""
    pass

class PerformanceError(SloughGPTError):
    """Performance and resource errors"""
    pass

def create_error(error_type: type[SloughGPTError], 
                message: str, 
                error_code: Optional[SloughGPTErrorCode] = None,
                details: Optional[Dict[str, Any]] = None,
                cause: Optional[Exception] = None,
                context: Optional[Dict[str, Any]] = None) -> SloughGPTError:
    """Factory function to create specific error types"""
    return error_type(
        message=message,
        error_code=error_code,
        details=details,
        cause=cause,
        context=context
    )

def handle_exception(func):
    """Decorator for automatic exception handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SloughGPTError as e:
            e.log_error({
                "function": func.__name__,
                "module": func.__module__,
                "line_number": traceback.extract_tb(e.__traceback__)[-1].lineno if e.__traceback__ else None,
                "stack_trace": traceback.format_exc()
            })
            # Re-raise as SloughGPTError for proper handling
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            wrapped_error = create_error(
                SloughGPTError,
                f"Unexpected error in {func.__name__}: {str(e)}",
                None,  # No specific error code for unexpected errors
                cause=e,
                context={
                    "function": func.__name__,
                    "module": func.__module__,
                    "args": str(args)[:100],
                    "kwargs": str(kwargs)[:100]
                }
            )
            wrapped_error.log_error({
                "function": func.__name__,
                "module": func.__module__,
                "line_number": None,
                "stack_trace": traceback.format_exc()
            })
            raise wrapped_error
    return wrapper

# Error context manager for structured error tracking
class ErrorContext:
    """Context manager for adding error context"""
    
    def __init__(self, context: Dict[str, Any]):
        self.context = context
        self.original_exception = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # Add context to the exception
            if hasattr(exc_val, 'add_context'):
                exc_val.add_context(self.context)
            self.original_exception = exc_val
        return False

def safe_execute(func):
    """Execute function with comprehensive error handling"""
    @handle_exception
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper