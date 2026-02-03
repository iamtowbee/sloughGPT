"""
SloughGPT Structured Logging System
Comprehensive logging with structured output, multiple handlers, and performance monitoring
"""

import logging
import logging.handlers
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from enum import Enum

import structlog
from pythonjsonlogger import jsonlogger

from .exceptions import SloughGPTError, SloughGPTErrorCode

class LogLevel(Enum):
    """Standardized log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class LogFormat(Enum):
    """Available log formats"""
    JSON = "json"
    TEXT = "text"
    STRUCTURED = "structured"

class PerformanceTimer:
    """Context manager for performance timing"""
    
    def __init__(self, logger, operation: str, level: LogLevel = LogLevel.INFO):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}", operation=self.operation, phase="start")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        
        log_data = {
            "operation": self.operation,
            "phase": "complete",
            "duration_ms": round(self.duration * 1000, 2)
        }
        
        if exc_type:
            log_data.update({
                "status": "error",
                "error_type": exc_type.__name__,
                "error_message": str(exc_val)
            })
            self.logger.error(f"Failed {self.operation} ({self.duration:.2f}ms)", **log_data)
        else:
            log_data["status"] = "success"
            self.logger.info(f"Completed {self.operation} ({self.duration:.2f}ms)", **log_data)

class StructuredLogger:
    """Enhanced structured logger with additional capabilities"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = None
        self.performance_loggers = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup structured logging with multiple handlers"""
        # Get configuration
        log_level = self.config.get("level", LogLevel.INFO).value.upper()
        log_format = self.config.get("format", LogFormat.STRUCTURED).value
        log_file = self.config.get("file_path")
        max_file_size = self.config.get("max_file_size", "10MB")
        backup_count = self.config.get("backup_count", 5)
        enable_console = self.config.get("enable_console", True)
        enable_performance = self.config.get("enable_performance", True)
        
        # Configure structlog processors
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        if log_format == LogFormat.JSON.value:
            processors.append(structlog.processors.JSONRenderer())
        elif log_format == LogFormat.TEXT.value:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:  # STRUCTURED
            processors.append(structlog.processors.UnicodeDecoder())
            processors.append(CustomStructuredRenderer())
        
        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Setup standard library logging
        self.logger = structlog.get_logger(self.name)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if enable_console:
            if log_format == LogFormat.JSON.value:
                console_formatter = jsonlogger.JsonFormatter(
                    '%(asctime)s %(name)s %(levelname)s %(message)s'
                )
            else:
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if log_file:
            # Parse max file size
            size_map = {'KB': 1024, 'MB': 1024*1024, 'GB': 1024*1024*1024}
            size_str = max_file_size.upper()
            size_bytes = int(size_str[:-2]) * size_map.get(size_str[-2:], 1024*1024)
            
            file_formatter = jsonlogger.JsonFormatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s'
            )
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=size_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        # Performance logger
        if enable_performance:
            self._setup_performance_logging()
    
    def _setup_performance_logging(self):
        """Setup dedicated performance logging"""
        perf_logger = logging.getLogger(f"{self.name}.performance")
        perf_logger.setLevel(logging.INFO)
        
        # Performance log file
        perf_file = self.config.get("performance_file", f"logs/{self.name}_performance.log")
        os.makedirs(os.path.dirname(perf_file), exist_ok=True)
        
        perf_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
        
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_file,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=3
        )
        perf_handler.setFormatter(perf_formatter)
        perf_logger.addHandler(perf_handler)
        
        self.performance_loggers[self.name] = perf_logger
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log exception with full traceback"""
        if exception:
            kwargs.update({
                "exception_type": exception.__class__.__name__,
                "exception_message": str(exception),
                "traceback": traceback.format_exc()
            })
        
        self.logger.error(message, **kwargs)
    
    def performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metric"""
        perf_data = {
            "operation": operation,
            "duration_ms": duration_ms,
            "timestamp": datetime.utcnow().isoformat()
        }
        perf_data.update(kwargs)
        
        if self.name in self.performance_loggers:
            # Use structlog for performance logging
            perf_logger = structlog.get_logger(f"{self.name}.performance")
            perf_logger.info(f"Performance: {operation}", **perf_data)
    
    def timer(self, operation: str, level: LogLevel = LogLevel.INFO) -> PerformanceTimer:
        """Create performance timer context manager"""
        return PerformanceTimer(self, operation, level)
    
    def api_request(self, endpoint: str, method: str, status_code: int, 
                   duration_ms: float, **kwargs):
        """Log API request"""
        self.performance(
            "api_request",
            duration_ms,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            **kwargs
        )
    
    def model_inference(self, model_name: str, input_tokens: int, 
                      output_tokens: int, duration_ms: float, **kwargs):
        """Log model inference"""
        self.performance(
            "model_inference",
            duration_ms,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_second=output_tokens / (duration_ms / 1000) if duration_ms > 0 else 0,
            **kwargs
        )
    
    def database_operation(self, operation: str, table: str, 
                        affected_rows: int, duration_ms: float, **kwargs):
        """Log database operation"""
        self.performance(
            "database_operation",
            duration_ms,
            operation=operation,
            table=table,
            affected_rows=affected_rows,
            **kwargs
        )

class CustomStructuredRenderer:
    """Custom renderer for structured log format"""
    
    def __call__(self, logger, method_name: str, event_dict):
        """Render structured log entry"""
        # Core fields
        timestamp = event_dict.get('timestamp', datetime.utcnow().isoformat())
        level = event_dict.get('level', 'INFO').upper()
        logger_name = event_dict.get('logger', 'unknown')
        message = event_dict.get('event', '')
        
        # Build structured log entry
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'logger': logger_name,
            'message': message
        }
        
        # Add structured fields (excluding standard ones)
        exclude_fields = {'timestamp', 'level', 'logger', 'event', 'module', 'line_number'}
        
        for key, value in event_dict.items():
            if key not in exclude_fields and value is not None:
                # Handle special cases
                if key == 'exc_info' and value:
                    log_entry['exception'] = {
                        'type': value[0].__name__ if value[0] else None,
                        'message': str(value[1]) if value[1] else None,
                        'traceback': ''.join(traceback.format_exception(*value)) if len(value) > 2 else None
                    }
                else:
                    log_entry[key] = value
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)

class ErrorTracker:
    """Track and analyze error patterns"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.error_counts = {}
        self.error_history = []
        self.critical_errors = []
    
    def track_error(self, error: SloughGPTError, context: Optional[Dict[str, Any]] = None):
        """Track error occurrence"""
        error_type = error.__class__.__name__
        error_code = error.error_code.value if error.error_code else None
        
        # Update counts
        error_key = f"{error_type}:{error_code}" if error_code else error_type
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Add to history
        error_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": error_type,
            "code": error_code,
            "message": error.message,
            "context": context or {}
        }
        self.error_history.append(error_record)
        
        # Keep only last 1000 errors in memory
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # Log error with structured data
        log_data = {
            "error_type": error_type,
            "error_code": error_code,
            "error_message": error.message,
            "error_count": self.error_counts[error_key],
            "context": context or {}
        }
        
        if error.details:
            log_data["error_details"] = error.details
        
        # Determine log level based on error severity
        if error_code:
            # Handle both string and enum types
            if hasattr(error_code, 'value'):
                error_code_str = error_code.value
            else:
                error_code_str = str(error_code)
            
            if error_code_str.startswith(('CRITICAL', 'SECURITY')):
                self.logger.critical(f"Critical error: {error.message}", **log_data)
                self.critical_errors.append(error_record)
            elif error_code_str.startswith('PERFORMANCE'):
                self.logger.warning(f"Performance issue: {error.message}", **log_data)
            else:
                self.logger.error(f"Application error: {error.message}", **log_data)
        else:
            self.logger.error(f"Application error: {error.message}", **log_data)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error tracking statistics"""
        return {
            "total_errors": len(self.error_history),
            "error_types": dict(self.error_counts),
            "critical_errors": len(self.critical_errors),
            "recent_errors": [e for e in self.error_history 
                           if (datetime.utcnow() - datetime.fromisoformat(e["timestamp"])).total_seconds() < 3600],
            "most_common_errors": sorted(self.error_counts.items(), 
                                     key=lambda x: x[1], reverse=True)[:10]
        }

# Global logger instances
_loggers: Dict[str, StructuredLogger] = {}
_error_tracker: Optional[ErrorTracker] = None

def get_logger(name: str, config: Optional[Dict[str, Any]] = None) -> StructuredLogger:
    """Get or create a structured logger instance"""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, config)
    return _loggers[name]

def get_error_tracker() -> ErrorTracker:
    """Get global error tracker instance"""
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker(get_logger("error_tracker"))
    return _error_tracker

def setup_logging(config: Optional[Dict[str, Any]] = None) -> StructuredLogger:
    """
    Setup global logging configuration
    
    Args:
        config: Logging configuration dictionary
        
    Returns:
        Main application logger
    """
    # Create main logger
    main_logger = get_logger("sloughgpt", config)
    
    # Setup error tracking
    global _error_tracker
    _error_tracker = ErrorTracker(main_logger)
    
    return main_logger

# Convenience functions
def log_performance(operation: str, duration_ms: float, **kwargs):
    """Log performance metric using default logger"""
    logger = get_logger("performance")
    logger.performance(operation, duration_ms, **kwargs)

def timer(operation: str) -> PerformanceTimer:
    """Create performance timer using default logger"""
    logger = get_logger("performance")
    return logger.timer(operation)

def track_exception(exception: Exception, context: Optional[Dict[str, Any]] = None):
    """Track exception using global error tracker"""
    from .exceptions import SloughGPTError, create_error
    
    error_tracker = get_error_tracker()
    
    if isinstance(exception, SloughGPTError):
        error_tracker.track_error(exception, context)
    else:
        # Wrap unknown exceptions
        wrapped_error = create_error(
            SloughGPTError,
            f"Unexpected exception: {str(exception)}",
            None,
            cause=exception,
            context=context
        )
        error_tracker.track_error(wrapped_error, context)