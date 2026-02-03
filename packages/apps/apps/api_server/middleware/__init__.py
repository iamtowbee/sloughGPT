import time
import logging
import asyncio
import hashlib
import secrets
from typing import Callable, Dict, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import jwt

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from ..core.config import settings
from ..monitoring.metrics_collector import metrics_collector

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware for logging HTTP requests with performance metrics"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = secrets.token_hex(8)
        
        # Record request start
        await metrics_collector.record_request_start(
            request_id=request_id,
            endpoint=str(request.url.path),
            method=request.method,
            user_agent=request.headers.get("user-agent", ""),
            ip_address=request.client.host if request.client else ""
        )
        
        # Log request
        logger.info(f"[{request_id}] {request.method} {request.url.path} - "
                   f"Client: {request.client.host if request.client else 'unknown'}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        process_time = time.time() - start_time
        
        # Record request completion
        await metrics_collector.record_request_end(
            request_id=request_id,
            endpoint=str(request.url.path),
            method=request.method,
            status_code=response.status_code,
            user_agent=request.headers.get("user-agent", ""),
            ip_address=request.client.host if request.client else ""
        )
        
        # Log response with metrics
        logger.info(f"[{request_id}] {request.method} {request.url.path} - "
                   f"Status: {response.status_code} - "
                   f"Time: {process_time:.3f}s")
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        return response

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware for consistent error handling and logging"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except HTTPException as e:
            # Log HTTP exceptions
            logger.warning(f"HTTP {e.status_code} in {request.method} {request.url.path}: {e.detail}")
            
            # Return consistent error response
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": {
                        "code": e.status_code,
                        "message": e.detail,
                        "timestamp": time.time()
                    }
                }
            )
        except Exception as e:
            # Log unhandled exceptions
            logger.error(f"Unhandled error in {request.method} {request.url.path}: {e}", 
                        exc_info=True)
            
            # Return consistent error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": 500,
                        "message": "Internal server error",
                        "timestamp": time.time()
                    }
                }
            )

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding comprehensive security headers"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response

class RateLimitEntry:
    """Rate limit entry with timestamp and request details"""
    
    def __init__(self, timestamp: float, endpoint: str = "", method: str = ""):
        self.timestamp = timestamp
        self.endpoint = endpoint
        self.method = method

class AdvancedRateLimitMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting with multiple strategies"""
    
    def __init__(self, app):
        super().__init__(app)
        self.global_requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.endpoint_requests: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=500)))
        self.blocked_ips: Set[str] = set()
        self.block_until: Dict[str, float] = {}
        
        # Rate limits from settings
        self.global_limit = settings.RATE_LIMIT_REQUESTS
        self.global_window = settings.RATE_LIMIT_WINDOW
        
        # Cleanup task
        self.cleanup_task = None
        
    async def start_cleanup_task(self):
        """Start background cleanup task"""
        if not self.cleanup_task:
            self.cleanup_task = asyncio.create_task(self._cleanup_expired_entries())
    
    async def _cleanup_expired_entries(self):
        """Cleanup expired rate limit entries"""
        while True:
            try:
                current_time = time.time()
                cutoff_time = current_time - self.global_window
                
                # Cleanup global requests
                for ip in list(self.global_requests.keys()):
                    requests = self.global_requests[ip]
                    self.global_requests[ip] = deque(
                        (req for req in requests if req.timestamp > cutoff_time),
                        maxlen=1000
                    )
                    
                    # Remove empty entries
                    if not self.global_requests[ip]:
                        del self.global_requests[ip]
                
                # Cleanup temporary blocks
                for ip in list(self.block_until.keys()):
                    if current_time > self.block_until[ip]:
                        del self.block_until[ip]
                        self.blocked_ips.discard(ip)
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                logger.error(f"Rate limit cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Check if IP is temporarily blocked
        if client_ip in self.blocked_ips and current_time < self.block_until.get(client_ip, 0):
            return self._rate_limit_response("IP temporarily blocked due to excessive requests")
        
        # Check global rate limit
        global_count = len([
            req for req in self.global_requests[client_ip]
            if current_time - req.timestamp < self.global_window
        ])
        
        if global_count >= self.global_limit:
            # Implement progressive blocking
            if global_count > self.global_limit * 2:
                # Block for longer period
                block_duration = 3600  # 1 hour
            else:
                # Short block
                block_duration = 300  # 5 minutes
            
            self.block_until[client_ip] = current_time + block_duration
            self.blocked_ips.add(client_ip)
            
            logger.warning(f"Rate limit exceeded for {client_ip}. Blocked for {block_duration}s")
            return self._rate_limit_response("Rate limit exceeded")
        
        # Record this request
        rate_entry = RateLimitEntry(
            timestamp=current_time,
            endpoint=str(request.url.path),
            method=request.method
        )
        self.global_requests[client_ip].append(rate_entry)
        
        # Add rate limit headers
        remaining = max(0, self.global_limit - global_count)
        response = await call_next(request)
        
        response.headers["X-RateLimit-Limit"] = str(self.global_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.global_window))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP from request, accounting for proxies"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        return request.client.host if request.client else "unknown"
    
    def _rate_limit_response(self, message: str) -> JSONResponse:
        """Create rate limit response"""
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "code": 429,
                    "message": message,
                    "retry_after": 60,
                    "timestamp": time.time()
                }
            },
            headers={"Retry-After": "60"}
        )

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """JWT-based authentication middleware"""
    
    def __init__(self, app, optional: bool = False):
        super().__init__(app)
        self.optional = optional
        self.jwt_secret = getattr(settings, 'JWT_SECRET', 'your-secret-key-change-this')
        self.jwt_algorithm = 'HS256'
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth for certain endpoints
        if self._should_skip_auth(request.url.path):
            return await call_next(request)
        
        # Get token from header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            if self.optional:
                return await call_next(request)
            return self._auth_error_response("Missing authorization header")
        
        try:
            # Extract token
            scheme, token = auth_header.split()
            if scheme.lower() != "bearer":
                raise ValueError("Invalid authorization scheme")
            
            # Verify token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Add user info to request state
            request.state.user = payload
            request.state.user_id = payload.get("sub")
            request.state.permissions = payload.get("permissions", [])
            
            return await call_next(request)
            
        except jwt.ExpiredSignatureError:
            return self._auth_error_response("Token has expired")
        except jwt.InvalidTokenError as e:
            return self._auth_error_response(f"Invalid token: {str(e)}")
        except Exception as e:
            return self._auth_error_response(f"Authentication error: {str(e)}")
    
    def _should_skip_auth(self, path: str) -> bool:
        """Check if path should skip authentication"""
        skip_patterns = [
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico"
        ]
        
        return any(path.startswith(pattern) for pattern in skip_patterns)
    
    def _auth_error_response(self, message: str) -> JSONResponse:
        """Create authentication error response"""
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "code": 401,
                    "message": message,
                    "timestamp": time.time()
                }
            }
        )

# Enhanced middleware classes
RateLimitMiddleware = AdvancedRateLimitMiddleware