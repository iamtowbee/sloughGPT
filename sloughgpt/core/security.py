"""
SloughGPT Security Validation System
Comprehensive input validation, sanitization, and rate limiting
"""

import re
import hashlib
import hmac
import time
import json
from typing import Dict, Any, List, Optional, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import asyncio
from functools import wraps

from .exceptions import SecurityError, create_error, SloughGPTErrorCode
from .logging_system import get_logger, timer

class ValidationLevel(Enum):
    """Security validation levels"""
    PERMISSIVE = "permissive"
    STANDARD = "standard"
    STRICT = "strict"

class InputType(Enum):
    """Types of input to validate"""
    TEXT = "text"
    JSON = "json"
    URL = "url"
    EMAIL = "email"
    FILENAME = "filename"
    SQL = "sql"
    HTML = "html"
    PROMPT = "prompt"
    API_KEY = "api_key"

@dataclass
class ValidationRule:
    """Individual validation rule"""
    name: str
    validator: Callable[[str], bool]
    error_message: str
    level: ValidationLevel = ValidationLevel.STANDARD
    
    def validate(self, value: str) -> tuple[bool, str]:
        """Validate value against this rule"""
        try:
            is_valid = self.validator(value)
            if is_valid:
                return True, ""
            return False, self.error_message
        except Exception as e:
            return False, f"Validation error: {str(e)}"

@dataclass
class ValidationResult:
    """Result of validation"""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityConfig:
    """Security configuration"""
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    enable_rate_limiting: bool = True
    enable_input_sanitization: bool = True
    enable_content_filtering: bool = True
    max_prompt_length: int = 1000
    rate_limit_window: int = 60  # seconds
    rate_limit_max_requests: int = 100
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    blocked_ip_addresses: List[str] = field(default_factory=list)
    api_keys: Set[str] = field(default_factory=set)
    sensitive_data_patterns: List[str] = field(default_factory=lambda: [
        r'\b\d{1,3}-\d{1,3}-\d{1,3}-\b\d{1,3}-\b',  # Basic auth
        r'password\s*=\s*["\'][^"\']+["\']',           # Password in URL
        r'api[_-]?key\s*=\s*["\'][^"\']+["\']',        # API keys
        r'token\s*=\s*["\'][^"\']+["\']',              # Tokens
        r'secret\s*=\s*["\'][^"\']+["\']',              # Secrets
    ])

class InputValidator:
    """Input validation and sanitization"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.logger = get_logger("security_validator")
        self._setup_validation_rules()
    
    def _setup_validation_rules(self):
        """Setup validation rules based on security level"""
        self.rules = {
            InputType.TEXT: self._get_text_rules(),
            InputType.JSON: self._get_json_rules(),
            InputType.URL: self._get_url_rules(),
            InputType.EMAIL: self._get_email_rules(),
            InputType.FILENAME: self._get_filename_rules(),
            InputType.SQL: self._get_sql_rules(),
            InputType.HTML: self._get_html_rules(),
            InputType.PROMPT: self._get_prompt_rules(),
            InputType.API_KEY: self._get_api_key_rules(),
        }
    
    def _get_text_rules(self) -> List[ValidationRule]:
        """Get text validation rules"""
        rules = [
            ValidationRule(
                name="max_length",
                validator=lambda x: len(x) <= self.config.max_prompt_length,
                error_message=f"Text exceeds maximum length of {self.config.max_prompt_length} characters",
                level=ValidationLevel.STRICT
            ),
            ValidationRule(
                name="no_null_bytes",
                validator=lambda x: '\x00' not in x,
                error_message="Text contains null bytes",
                level=ValidationLevel.STRICT
            ),
        ]
        
        if self.config.validation_level in [ValidationLevel.STRICT, ValidationLevel.STANDARD]:
            rules.extend([
                ValidationRule(
                    name="no_control_chars",
                    validator=lambda x: not re.search(r'[\x00-\x1f\x7f-\x9f]', x),
                    error_message="Text contains control characters",
                    level=ValidationLevel.STANDARD
                ),
                ValidationRule(
                    name="reasonable_unicode",
                    validator=lambda x: len(x.encode('utf-8')) <= len(x) * 4,  # Prevent overlong Unicode
                    error_message="Text contains suspicious Unicode sequences",
                    level=ValidationLevel.STANDARD
                ),
            ])
        
        return rules
    
    def _get_json_rules(self) -> List[ValidationRule]:
        """Get JSON validation rules"""
        return [
            ValidationRule(
                name="valid_json",
                validator=lambda x: self._is_valid_json(x),
                error_message="Invalid JSON format",
                level=ValidationLevel.STRICT
            ),
            ValidationRule(
                name="json_size_limit",
                validator=lambda x: len(x.encode('utf-8')) <= 1024 * 1024,  # 1MB limit
                error_message="JSON exceeds maximum size",
                level=ValidationLevel.STRICT
            ),
        ]
    
    def _get_url_rules(self) -> List[ValidationRule]:
        """Get URL validation rules"""
        return [
            ValidationRule(
                name="safe_url",
                validator=lambda x: self._is_safe_url(x),
                error_message="URL contains potentially dangerous content",
                level=ValidationLevel.STRICT
            ),
            ValidationRule(
                name="no_local_files",
                validator=lambda x: not re.search(r'file://|localhost|127\.0\.0\.1|0x7f000001', x, re.IGNORECASE),
                error_message="URL references local files or localhost",
                level=ValidationLevel.STRICT
            ),
        ]
    
    def _get_email_rules(self) -> List[ValidationRule]:
        """Get email validation rules"""
        return [
            ValidationRule(
                name="valid_email",
                validator=lambda x: re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', x),
                error_message="Invalid email format",
                level=ValidationLevel.STANDARD
            ),
        ]
    
    def _get_filename_rules(self) -> List[ValidationRule]:
        """Get filename validation rules"""
        return [
            ValidationRule(
                name="safe_filename",
                validator=lambda x: self._is_safe_filename(x),
                error_message="Filename contains unsafe characters",
                level=ValidationLevel.STRICT
            ),
            ValidationRule(
                name="no_path_traversal",
                validator=lambda x: '..' not in x and not x.startswith('/'),
                error_message="Filename contains path traversal sequences",
                level=ValidationLevel.STRICT
            ),
        ]
    
    def _get_sql_rules(self) -> List[ValidationRule]:
        """Get SQL injection prevention rules"""
        dangerous_patterns = [
            r'(?i)\b(?:union|select|insert|update|delete|drop|create|alter|exec|execute)\b',
            r'(?i)(?:--|#|/\*|\*/)',
            r'(?i)(?:or|and)\s+\d+\s*=\s*\d+',
        ]
        
        return [
            ValidationRule(
                name="no_sql_injection",
                validator=lambda x: not any(re.search(pattern, x) for pattern in dangerous_patterns),
                error_message="Input contains potentially dangerous SQL patterns",
                level=ValidationLevel.STRICT
            ),
        ]
    
    def _get_html_rules(self) -> List[ValidationRule]:
        """Get HTML/XSS prevention rules"""
        dangerous_tags = [
            r'<script[^>]*>.*?</script>',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'javascript:',
            r'on\w+\s*=',  # Event handlers like onclick, onload
        ]
        
        return [
            ValidationRule(
                name="no_xss",
                validator=lambda x: not any(re.search(tag, x, re.IGNORECASE) for tag in dangerous_tags),
                error_message="Input contains potentially dangerous HTML/JavaScript",
                level=ValidationLevel.STRICT
            ),
        ]
    
    def _get_prompt_rules(self) -> List[ValidationRule]:
        """Get prompt injection prevention rules"""
        injection_patterns = [
            r'(?i)\b(?:ignore|forget|previous|system)\s+.*?(?:prompt|instruction|command)',
            r'(?i)\b(?:roleplay|act as|pretend)\s+.*?(?:system|admin|root)',
            r'(?i)\b(?:execute|run|eval)\s+.*?(?:code|script|command)',
            r'(?i)\]\s*\(?.*\)\s*\{',  # JSON injection in prompts
        ]
        
        rules = [
            ValidationRule(
                name="no_prompt_injection",
                validator=lambda x: not any(re.search(pattern, x) for pattern in injection_patterns),
                error_message="Input contains potentially dangerous prompt injection patterns",
                level=ValidationLevel.STRICT
            ),
        ]
        
        # Add length rules for prompts
        rules.extend(self._get_text_rules())
        return rules
    
    def _get_api_key_rules(self) -> List[ValidationRule]:
        """Get API key validation rules"""
        return [
            ValidationRule(
                name="api_key_format",
                validator=lambda x: re.match(r'^[a-zA-Z0-9_-]{20,64}$', x),
                error_message="Invalid API key format",
                level=ValidationLevel.STANDARD
            ),
            ValidationRule(
                name="api_key_length",
                validator=lambda x: 20 <= len(x) <= 64,
                error_message="API key must be between 20 and 64 characters",
                level=ValidationLevel.STANDARD
            ),
        ]
    
    def _is_valid_json(self, value: str) -> bool:
        """Check if string is valid JSON"""
        try:
            json.loads(value)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    
    def _is_safe_url(self, url: str) -> bool:
        """Check if URL is safe"""
        dangerous_schemes = ['javascript:', 'data:', 'vbscript:']
        return not any(url.lower().startswith(scheme) for scheme in dangerous_schemes)
    
    def _is_safe_filename(self, filename: str) -> bool:
        """Check if filename is safe"""
        dangerous_chars = '<>:"|?*\\'
        return not any(char in filename for char in dangerous_chars)
    
    def validate(self, input_type: InputType, value: str) -> ValidationResult:
        """Validate input based on type"""
        result = ValidationResult()
        
        # Get validation rules
        rules = self.rules.get(input_type, [])
        
        # Apply validation rules based on level
        for rule in rules:
            if self._should_apply_rule(rule):
                is_valid, error_msg = rule.validate(value)
                if not is_valid:
                    result.errors.append(error_msg)
                elif error_msg:  # Warning
                    result.warnings.append(error_msg)
        
        # Sanitize input if enabled
        if self.config.enable_input_sanitization:
            result.sanitized_value = self._sanitize_input(input_type, value)
        
        result.is_valid = len(result.errors) == 0
        
        # Log validation attempt
        self.logger.info("Input validation attempt",
                      input_type=input_type.value,
                      input_length=len(value),
                      is_valid=result.is_valid,
                      error_count=len(result.errors),
                      warning_count=len(result.warnings))
        
        return result
    
    def _should_apply_rule(self, rule: ValidationRule) -> bool:
        """Check if rule should be applied based on level"""
        if self.config.validation_level == ValidationLevel.PERMISSIVE:
            return rule.level == ValidationLevel.STRICT
        elif self.config.validation_level == ValidationLevel.STANDARD:
            return rule.level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]
        else:  # STRICT
            return True
    
    def _sanitize_input(self, input_type: InputType, value: str) -> str:
        """Sanitize input based on type"""
        if not value:
            return value
        
        # Basic sanitization for all types
        sanitized = value.strip()
        sanitized = sanitized.replace('\x00', '')  # Remove null bytes
        
        # Type-specific sanitization
        if input_type in [InputType.HTML, InputType.TEXT]:
            # HTML sanitization
            sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(r'<iframe[^>]*>.*?</iframe>', '', sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
        
        elif input_type == InputType.SQL:
            # SQL sanitization
            sanitized = re.sub(r"(['\"])", r"\\\1", sanitized)
        
        elif input_type == InputType.FILENAME:
            # Filename sanitization
            sanitized = re.sub(r'[<>:"|?*\\]', '_', sanitized)
        
        return sanitized

class RateLimiter:
    """Rate limiting with sliding window"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.logger = get_logger("rate_limiter")
        self.requests = defaultdict(list)  # IP -> [timestamps]
        
    def is_allowed(self, identifier: str) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed based on rate limit"""
        current_time = time.time()
        window_start = current_time - self.config.rate_limit_window
        
        # Clean old requests
        self._cleanup_old_requests(identifier, window_start)
        
        # Count recent requests
        recent_requests = [
            req_time for req_time in self.requests[identifier]
            if req_time >= window_start
        ]
        
        request_count = len(recent_requests)
        is_allowed = request_count < self.config.rate_limit_max_requests
        
        # Add current request
        self.requests[identifier].append(current_time)
        
        metadata = {
            "request_count": request_count,
            "limit": self.config.rate_limit_max_requests,
            "window": self.config.rate_limit_window,
            "retry_after": window_start + self.config.rate_limit_window if not is_allowed else None,
            "allowed_requests_remaining": max(0, self.config.rate_limit_max_requests - request_count)
        }
        
        if not is_allowed:
            self.logger.warning("Rate limit exceeded",
                              identifier=identifier,
                              request_count=request_count,
                              limit=self.config.rate_limit_max_requests)
        else:
            self.logger.debug("Rate limit check passed",
                             identifier=identifier,
                             request_count=request_count)
        
        return is_allowed, metadata
    
    def _cleanup_old_requests(self, identifier: str, cutoff_time: float):
        """Remove old request timestamps"""
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time >= cutoff_time
            ]

class ContentFilter:
    """Content filtering for sensitive/inappropriate content"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.logger = get_logger("content_filter")
        self._setup_filters()
    
    def _setup_filters(self):
        """Setup content filters"""
        self.blocked_patterns = [
            # Basic inappropriate content patterns (example)
            r'(?i)\b(?:password|secret|token|key)\b.*?=',
            # Personal information patterns
            r'(?i)\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',  # SSN pattern
            r'(?i)\b(?:\d{4}[-.\s]?){3}\d{4}\b',  # Credit card pattern
            # Contact information
            r'(?i)\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Phone pattern
            # Dangerous content
            r'(?i)\b(?:bomb|weapon|exploit|hack)\b',
        ]
        
        if self.config.validation_level == ValidationLevel.STRICT:
            self.blocked_patterns.extend([
                r'(?i)\b(?:adult|explicit|nsfw)\b',
            ])
    
    def filter_content(self, content: str) -> ValidationResult:
        """Filter content for inappropriate material"""
        result = ValidationResult()
        
        if not self.config.enable_content_filtering:
            result.is_valid = True
            result.sanitized_value = content
            return result
        
        # Check against blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                result.errors.append(f"Content contains blocked pattern: {pattern}")
                self.logger.warning("Content filter triggered",
                                 pattern=pattern,
                                 content_length=len(content))
        
        # Additional sanitization
        if result.is_valid:
            result.sanitized_value = self._sanitize_content(content)
        
        result.is_valid = len(result.errors) == 0
        return result
    
    def _sanitize_content(self, content: str) -> str:
        """Basic content sanitization"""
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', content)
        sanitized = sanitized.strip()
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
        
        return sanitized

class SecurityMiddleware:
    """Combined security middleware"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.validator = InputValidator(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.content_filter = ContentFilter(self.config)
        self.logger = get_logger("security_middleware")
    
    def validate_api_request(self, 
                         data: Dict[str, Any],
                         client_ip: str,
                         user_agent: str,
                         api_key: Optional[str] = None) -> ValidationResult:
        """Validate API request"""
        result = ValidationResult()
        
        with timer("security_validation") as perf_timer:
            # Check API key if required
            if self.config.api_keys and (not api_key or api_key not in self.config.api_keys):
                result.errors.append("Invalid or missing API key")
                self.logger.warning("Invalid API key attempted",
                                 client_ip=client_ip,
                                 user_agent=user_agent)
            
            # Check rate limiting
            if self.config.enable_rate_limiting:
                is_rate_allowed, rate_metadata = self.rate_limiter.is_allowed(client_ip)
                if not is_rate_allowed:
                    result.errors.append(f"Rate limit exceeded. Try again after {rate_metadata['retry_after']}")
                    result.metadata.update(rate_metadata)
            
            # Validate request data
            if isinstance(data.get('prompt'), str):
                prompt_validation = self.validator.validate(InputType.PROMPT, data['prompt'])
                result.errors.extend(prompt_validation.errors)
                result.warnings.extend(prompt_validation.warnings)
                if prompt_validation.sanitized_value:
                    data['prompt'] = prompt_validation.sanitized_value
            
            # Content filtering
            if isinstance(data.get('prompt'), str):
                content_validation = self.content_filter.filter_content(data['prompt'])
                result.errors.extend(content_validation.errors)
                result.warnings.extend(content_validation.warnings)
                if content_validation.sanitized_value:
                    data['prompt'] = content_validation.sanitized_value
            
            # Validate JSON fields
            for key, value in data.items():
                if isinstance(value, str) and key in ['config', 'metadata', 'options']:
                    json_validation = self.validator.validate(InputType.JSON, value)
                    if not json_validation.is_valid:
                        result.errors.extend([f"Invalid JSON in field '{key}': {err}" for err in json_validation.errors])
            
            # Check for sensitive data
            sensitive_content = self._check_sensitive_data(data)
            if sensitive_content:
                result.errors.append("Request contains potentially sensitive information")
                self.logger.warning("Sensitive data detected in request",
                                 client_ip=client_ip,
                                 sensitive_fields=list(sensitive_content.keys()))
            
            result.is_valid = len(result.errors) == 0
            
            # Log security validation result
            self.logger.info("API security validation completed",
                           client_ip=client_ip,
                           is_valid=result.is_valid,
                           error_count=len(result.errors),
                           validation_time_ms=perf_timer.duration * 1000 if perf_timer.duration else 0)
        
        return result
    
    def _check_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for sensitive data patterns"""
        sensitive_found = {}
        data_str = json.dumps(data, default=str)
        
        for pattern in self.config.sensitive_data_patterns:
            matches = re.findall(pattern, data_str, re.IGNORECASE)
            if matches:
                sensitive_found[pattern] = matches
        
        return sensitive_found
    
    def get_client_identifier(self, 
                           client_ip: str, 
                           user_agent: str = None,
                           api_key: str = None) -> str:
        """Generate unique client identifier"""
        # Create hash of identifying information
        identifier_data = f"{client_ip}:{user_agent or ''}:{api_key or ''}"
        return hashlib.sha256(identifier_data.encode()).hexdigest()[:16]

# Decorators for easy use
def validate_prompt(input_type: InputType = InputType.PROMPT, level: ValidationLevel = ValidationLevel.STANDARD):
    """Decorator to validate prompt input"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get prompt from args or kwargs
            prompt = kwargs.get('prompt') or (args[0] if args else None)
            
            if prompt and isinstance(prompt, str):
                validator = InputValidator(SecurityConfig(validation_level=level))
                result = validator.validate(input_type, prompt)
                
                if not result.is_valid:
                    raise create_error(
                        SecurityError,
                        f"Input validation failed: {'; '.join(result.errors)}",
                        SloughGPTErrorCode.SECURITY_VALIDATION_FAILED,
                        context={"input_type": input_type.value, "errors": result.errors}
                    )
                
                # Use sanitized value if available
                if result.sanitized_value:
                    if 'prompt' in kwargs:
                        kwargs['prompt'] = result.sanitized_value
                    elif args:
                        args = list(args)
                        args[0] = result.sanitized_value
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def rate_limit(max_requests: int = 100, window: int = 60):
    """Decorator for rate limiting"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Simple rate limiting implementation
            # In production, this would use Redis or similar
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Global security middleware instance
_global_security_middleware: Optional[SecurityMiddleware] = None

def get_security_middleware(config: Optional[SecurityConfig] = None) -> SecurityMiddleware:
    """Get or create global security middleware"""
    global _global_security_middleware
    if _global_security_middleware is None:
        _global_security_middleware = SecurityMiddleware(config)
    return _global_security_middleware