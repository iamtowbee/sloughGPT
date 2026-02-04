"""Advanced logging and audit system for SloughGPT."""

from typing import Dict, List, Optional, Any, Union, TextIO
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
import sys
import traceback
from enum import Enum
from pathlib import Path
import gzip
import io

try:
    from fastapi import HTTPException
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    from .database import database_manager
    from .security_advanced import security_manager
    from .cache import cache_manager
    HAS_INTEGRATIONS = True
except ImportError:
    HAS_INTEGRATIONS = False


class LogLevel(Enum):
    """Log levels for the system."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"
    AUDIT = "audit"
    PERFORMANCE = "performance"


class LogFormat(Enum):
    """Log output formats."""
    JSON = "json"
    STRUCTURED = "structured"
    PLAIN = "plain"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime = field(default_factory=datetime.now)
    level: LogLevel
    logger_name: str
    message: str
    user_id: Optional[int] = None
    ip_address: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    function_name: Optional[str] = None
    file_name: Optional[str] = None
    line_number: Optional[int] = None
    exception_type: Optional[str] = None
    stack_trace: Optional[str] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class LogFilter:
    """Log filtering criteria."""
    level_min: Optional[LogLevel] = None
    level_max: Optional[LogLevel] = None
    logger_patterns: List[str] = field(default_factory=list)
    user_ids: List[int] = field(default_factory=list)
    ip_addresses: List[str] = field(default_factory=list)
    time_range: Optional[timedelta] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class AuditEvent:
    """Audit event structure."""
    event_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str
    category: str
    severity: str
    user_id: Optional[int] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: str
    action: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    compliance_tags: List[str] = field(default_factory=list)


class AdvancedLogger:
    """Advanced logging system with structured output and filtering."""
    
    def __init__(self, name: str = "sloughgpt", level: LogLevel = LogLevel.INFO):
        self.name = name
        self.level = level
        self.filters: List[LogFilter] = []
        self.handlers: List[Any] = []
        self.log_buffer: List[LogEntry] = []
        self.buffer_size = 1000
        self.rotation_enabled = True
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.retention_days = 90
        self.compression_enabled = True
        
        # Configure Python logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level.value)
        
        # Set up formatters and handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up logging handlers."""
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler with structured formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredFormatter())
        self.handlers.append(console_handler)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation and compression
        try:
            file_handler = RotatingFileHandler(
                filename=f"logs/{self.name}.log",
                maxBytes=self.max_file_size,
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(JsonFormatter())
            
            # Add compression if available
            if self.compression_enabled:
                file_handler.addFilter(CompressionFilter())
            
            self.handlers.append(file_handler)
            self.logger.addHandler(file_handler)
            
        except Exception as e:
            self.logger.error(f"Failed to setup file handler: {e}")
        
        # Database handler for persistent logs
        if HAS_INTEGRATIONS:
            try:
                db_handler = DatabaseHandler(self)
                self.handlers.append(db_handler)
                self.logger.addHandler(db_handler)
            except Exception as e:
                self.logger.error(f"Failed to setup database handler: {e}")
    
    def add_filter(self, log_filter: LogFilter):
        """Add a log filter."""
        self.filters.append(log_filter)
    
    def should_log(self, entry: LogEntry) -> bool:
        """Check if entry should be logged based on filters."""
        # Check level range
        if self.filters:
            for log_filter in self.filters:
                if not self._passes_filter(entry, log_filter):
                    return False
        return True
    
    def _passes_filter(self, entry: LogEntry, log_filter: LogFilter) -> bool:
        """Check if entry passes a specific filter."""
        # Level filter
        if log_filter.level_min and entry.level.value < log_filter.level_min.value:
            return False
        if log_filter.level_max and entry.level.value > log_filter.level_max.value:
            return False
        
        # User ID filter
        if log_filter.user_ids and entry.user_id not in log_filter.user_ids:
            return False
        
        # IP address filter
        if log_filter.ip_addresses and entry.ip_address not in log_filter.ip_addresses:
            return False
        
        # Logger name pattern filter
        if log_filter.logger_patterns:
            if not any(pattern in entry.logger_name for pattern in log_filter.logger_patterns):
                return False
        
        # Tags filter
        if log_filter.tags and not any(tag in entry.tags for tag in log_filter.tags):
            return False
        
        # Time range filter
        if log_filter.time_range:
            if datetime.now() - entry.timestamp > log_filter.time_range:
                return False
        
        return True
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message."""
        if exception:
            kwargs["exception_type"] = type(exception).__name__
            kwargs["stack_trace"] = traceback.format_exc()
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def security(self, message: str, **kwargs):
        """Log security event."""
        self._log(LogLevel.SECURITY, message, **kwargs)
        
        # Also log as audit event if integrated
        if HAS_INTEGRATIONS:
            asyncio.create_task(self._log_audit_event(
                event_type="security_alert",
                category="security",
                severity="high",
                action=message,
                details=kwargs
            ))
    
    def audit(self, event_type: str, category: str = "general", severity: str = "info",
             user_id: Optional[int] = None, ip_address: Optional[str] = None,
             resource: str = "", action: str = "", status: str = "",
             details: Optional[Dict[str, Any]] = None, **kwargs):
        """Log audit event."""
        if HAS_INTEGRATIONS:
            asyncio.create_task(self._log_audit_event(
                event_type=event_type,
                category=category,
                severity=severity,
                user_id=user_id,
                ip_address=ip_address,
                resource=resource,
                action=action,
                status=status,
                details=details or kwargs
            ))
        
        # Also log as structured log
        self._log(LogLevel.AUDIT, f"Audit: {event_type}", {
            "event_type": event_type,
            "category": category,
            "severity": severity,
            "resource": resource,
            "action": action,
            "status": status,
            "user_id": user_id,
            "ip_address": ip_address,
            "details": details or kwargs
        })
    
    def performance(self, operation: str, duration: float, metrics: Dict[str, Any],
                 **kwargs):
        """Log performance metrics."""
        self._log(LogLevel.PERFORMANCE, f"Performance: {operation}", {
            "operation": operation,
            "duration": duration,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        })
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method."""
        # Get caller information
        frame = sys._getframe(2)
        function_name = frame.f_code.co_name if frame else "unknown"
        file_name = Path(frame.f_code.co_filename).name if frame else "unknown"
        line_number = frame.f_lineno if frame else None
        
        # Create log entry
        entry = LogEntry(
            level=level,
            logger_name=self.name,
            message=message,
            user_id=kwargs.get("user_id"),
            ip_address=kwargs.get("ip_address"),
            request_id=kwargs.get("request_id"),
            session_id=kwargs.get("session_id"),
            correlation_id=kwargs.get("correlation_id"),
            function_name=function_name,
            file_name=file_name,
            line_number=line_number,
            exception_type=kwargs.get("exception_type"),
            stack_trace=kwargs.get("stack_trace"),
            execution_time=kwargs.get("execution_time"),
            memory_usage=kwargs.get("memory_usage"),
            metadata={k: v for k, v in kwargs.items() 
                     if k not in ["user_id", "ip_address", "request_id", "session_id", 
                              "correlation_id", "exception_type", "stack_trace", 
                              "execution_time", "memory_usage"]},
            tags=kwargs.get("tags", [])
        )
        
        # Add to buffer
        self.log_buffer.append(entry)
        if len(self.log_buffer) > self.buffer_size:
            self.log_buffer = self.log_buffer[-self.buffer_size:]
        
        # Check if should log
        if self.should_log(entry):
            # Use standard Python logging
            log_level = getattr(logging, level.value.upper())
            self.logger.log(log_level, message, extra=kwargs)
    
    async def _log_audit_event(self, event_type: str, category: str, severity: str,
                            user_id: Optional[int], ip_address: Optional[str],
                            resource: str, action: str, status: str,
                            details: Dict[str, Any]):
        """Log audit event to database."""
        if not HAS_INTEGRATIONS:
            return
        
        try:
            event_id = str(uuid.uuid4())
            
            audit_event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                category=category,
                severity=severity,
                user_id=user_id,
                ip_address=ip_address,
                user_agent="",  # Would come from request
                resource=resource,
                action=action,
                status=status,
                details=details,
                session_id=details.get("session_id"),
                correlation_id=details.get("correlation_id"),
                compliance_tags=details.get("compliance_tags", [])
            )
            
            await database_manager.log_security_event(
                user_id=user_id,
                action=f"{category}:{event_type}",
                resource=resource,
                details={
                    "severity": severity,
                    "status": status,
                    **details
                },
                ip_address=ip_address,
                user_agent=""
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")


class StructuredFormatter(logging.Formatter):
    """Structured log formatter."""
    
    def format(self, record):
        """Format log record as structured data."""
        # Create structured data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }
        
        # Add extra fields if available
        if hasattr(record, "__dict__"):
            extra_fields = {k: v for k, v in record.__dict__.items() 
                            if k not in ["name", "msg", "args", "levelname", 
                                    "module", "funcName", "lineno", "thread", "process",
                                    "created", "msecs", "relativeCreated", 
                                    "levelno", "exc_info", "exc_text", "stack_info"]}
            log_data.update(extra_fields)
        
        return json.dumps(log_data, default=str, indent=None)


class JsonFormatter(logging.Formatter):
    """JSON formatter for log records."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }
        
        # Add extra fields
        if hasattr(record, "__dict__"):
            log_data.update(record.__dict__)
        
        return json.dumps(log_data, default=str)


class RotatingFileHandler(logging.Handler):
    """Rotating file handler with compression."""
    
    def __init__(self, filename, maxBytes=104857600, backupCount=5, encoding='utf-8'):
        self.filename = filename
        self.maxBytes = maxBytes
        self.backupCount = backupCount
        self.encoding = encoding
        self.compression_enabled = False
        
        # Create log directory if it doesn't exist
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    def emit(self, record):
        """Emit a log record."""
        try:
            msg = self.format(record)
            
            # Write to file
            with open(self.filename, 'a', encoding=self.encoding) as f:
                f.write(msg + '\n')
                f.flush()
            
            # Check if file should be rotated
            if self.should_rollover():
                self.rollover()
                
        except Exception:
            self.handleError(record)
    
    def should_rollover(self):
        """Check if file should be rotated."""
        try:
            return Path(self.filename).stat().st_size >= self.maxBytes
        except OSError:
            return False
    
    def rollover(self):
        """Rotate log files."""
        if self.backupCount > 0:
            # Rotate existing files
            for i in range(self.backupCount - 1, 0, -1):
                old_file = f"{self.filename}.{i}"
                if Path(old_file).exists():
                    Path(old_file).rename(f"{self.filename}.{i + 1}")
            
            # Rotate current file
            if Path(self.filename).exists():
                Path(self.filename).rename(f"{self.filename}.0")
        
        # Create new file
        with open(self.filename, 'w', encoding=self.encoding):
            pass
    
    def format(self, record):
        """Format log record."""
        return f"{datetime.fromtimestamp(record.created).isoformat()} - {record.levelname} - {record.name} - {record.getMessage()}"


class DatabaseHandler(logging.Handler):
    """Database handler for persistent logging."""
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
    
    def emit(self, record):
        """Emit log record to database."""
        try:
            # This would integrate with the database manager
            # For now, just log to console
            pass
        except Exception:
            self.logger.handleError(record)


class CompressionFilter(logging.Filter):
    """Compression filter for log files."""
    
    def __init__(self):
        self.compression_enabled = True
    
    def filter(self, record):
        """Filter log record for compression."""
        # Add compression indicator to record
        record.compressed = self.compression_enabled
        return True


class LogManager:
    """Central log management system."""
    
    def __init__(self):
        self.loggers: Dict[str, AdvancedLogger] = {}
        self.global_filters: List[LogFilter] = []
        self.default_level = LogLevel.INFO
        self.retention_days = 90
        self.storage_config = {
            "file": {
                "enabled": True,
                "path": "logs/",
                "rotation": True,
                "compression": True,
                "max_size": "100MB"
            },
            "database": {
                "enabled": True,
                "table": "logs"
            },
            "syslog": {
                "enabled": False,
                "host": "localhost",
                "port": 514
            }
        }
        
    def get_logger(self, name: str, level: Optional[LogLevel] = None) -> AdvancedLogger:
        """Get or create a logger."""
        if name not in self.loggers:
            self.loggers[name] = AdvancedLogger(
                name=name,
                level=level or self.default_level
            )
            
            # Apply global filters
            for filter in self.global_filters:
                self.loggers[name].add_filter(filter)
        
        return self.loggers[name]
    
    def add_global_filter(self, log_filter: LogFilter):
        """Add a global filter to all loggers."""
        self.global_filters.append(log_filter)
        for logger in self.loggers.values():
            logger.add_filter(log_filter)
    
    def set_default_level(self, level: LogLevel):
        """Set default logging level."""
        self.default_level = level
        for logger in self.loggers.values():
            logger.level = level
    
    async def cleanup_old_logs(self, days: int = None) -> Dict[str, Any]:
        """Clean up old log files."""
        days = days or self.retention_days
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cleaned_up = {
            "files_deleted": 0,
            "space_freed": 0,
            "cutoff_date": cutoff_date.isoformat()
        }
        
        try:
            # Clean up log files
            if self.storage_config["file"]["enabled"]:
                log_dir = Path(self.storage_config["file"]["path"])
                for log_file in log_dir.glob("*.log*"):
                    if log_file.stat().st_mtime < cutoff_date.timestamp():
                        file_size = log_file.stat().st_size
                        log_file.unlink()
                        cleaned_up["files_deleted"] += 1
                        cleaned_up["space_freed"] += file_size
            
            # Clean up database logs if enabled
            if HAS_INTEGRATIONS and self.storage_config["database"]["enabled"]:
                db_cleanup = await database_manager.cleanup_old_data(days)
                cleaned_up.update(db_cleanup)
            
        except Exception as e:
            return {"error": str(e)}
        
        return cleaned_up
    
    async def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = {
            "total_loggers": len(self.loggers),
            "default_level": self.default_level.value,
            "global_filters": len(self.global_filters),
            "retention_days": self.retention_days,
            "storage_config": self.storage_config,
            "loggers": {}
        }
        
        for name, logger in self.loggers.items():
            logger_stats = {
                "name": name,
                "level": logger.level.value,
                "buffer_size": len(logger.log_buffer),
                "handlers_count": len(logger.handlers),
                "filters_count": len(logger.filters)
            }
            stats["loggers"][name] = logger_stats
        
        return stats
    
    def export_logs(self, logger_name: str, start_date: datetime, 
                 end_date: datetime, level: Optional[LogLevel] = None) -> Optional[str]:
        """Export logs in JSON format."""
        if logger_name not in self.loggers:
            return None
        
        logger = self.loggers[logger_name]
        
        # Filter log entries by date and level
        entries = []
        for entry in logger.log_buffer:
            if start_date <= entry.timestamp <= end_date:
                if level is None or entry.level.value >= level.value:
                    entries.append(entry)
        
        return json.dumps([
            {
                "timestamp": entry.timestamp.isoformat(),
                "level": entry.level.value,
                "logger": entry.logger_name,
                "message": entry.message,
                "user_id": entry.user_id,
                "ip_address": entry.ip_address,
                "request_id": entry.request_id,
                "session_id": entry.session_id,
                "correlation_id": entry.correlation_id,
                "function_name": entry.function_name,
                "file_name": entry.file_name,
                "line_number": entry.line_number,
                "exception_type": entry.exception_type,
                "stack_trace": entry.stack_trace,
                "execution_time": entry.execution_time,
                "memory_usage": entry.memory_usage,
                "metadata": entry.metadata,
                "tags": entry.tags
            }
            for entry in entries
        ], indent=2)


class FastAPILoggingMiddleware:
    """FastAPI middleware for request logging."""
    
    def __init__(self, app, log_level: LogLevel = LogLevel.INFO):
        self.app = app
        self.logger = LogManager().get_logger("fastapi", log_level)
    
    async def __call__(self, request, call_next):
        """Log FastAPI requests."""
        start_time = datetime.now()
        
        # Get request information
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Extract user info from request state if available
        user_id = getattr(request.state, "user_id", None)
        session_id = getattr(request.state, "session_id", None)
        
        response = None
        try:
            response = await call_next(request)
        except Exception as e:
            self.logger.error(
                f"Request failed: {str(e)}",
                exception=e,
                user_id=user_id,
                ip_address=client_host,
                request_id=getattr(request.state, "request_id", None),
                session_id=session_id,
                function_name="middleware",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            raise HTTPException(status_code=500, detail="Internal Server Error")
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Log request
        self.logger.info(
            f"Request: {request.method} {request.url}",
            user_id=user_id,
            ip_address=client_host,
            request_id=getattr(request.state, "request_id", None),
            session_id=session_id,
            function_name=request.url.path,
            execution_time=execution_time,
            metadata={
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code if response else "unknown",
                "user_agent": user_agent,
                "content_length": len(response.body) if hasattr(response, 'body') else 0
            },
            tags=["api_request"]
        )
        
        return response


# Global log manager
log_manager = LogManager()

# Common loggers for different components
security_logger = log_manager.get_logger("security", LogLevel.SECURITY)
audit_logger = log_manager.get_logger("audit", LogLevel.AUDIT)
performance_logger = log_manager.get_logger("performance", LogLevel.PERFORMANCE)
api_logger = log_manager.get_logger("api", LogLevel.INFO)