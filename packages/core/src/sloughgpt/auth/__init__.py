"""Authentication and authorization services."""

from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
import jwt
import secrets
import hashlib
from ..user_management import UserManager, UserRole


class AuthService:
    def __init__(self, user_manager: UserManager):
        self.user_manager = user_manager
        self.secret_key = "your-jwt-secret-key-change-in-production"
        self.token_expiry = timedelta(hours=24)
        self.refresh_token_expiry = timedelta(days=7)
        self.active_tokens: Dict[str, Dict[str, Any]] = {}
    
    def authenticate(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user and return tokens."""
        auth_result = self.user_manager.authenticate_user(username, password)
        
        if "error" in auth_result:
            return {"error": "Invalid credentials"}
        
        user = auth_result["user"]
        
        # Generate access token
        access_token = self._generate_token(
            user_id=user["id"],
            username=user["username"],
            role=user["role"],
            expiry=self.token_expiry
        )
        
        # Generate refresh token
        refresh_token = self._generate_token(
            user_id=user["id"],
            username=user["username"],
            role=user["role"],
            expiry=self.refresh_token_expiry,
            token_type="refresh"
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": int(self.token_expiry.total_seconds()),
            "user": user
        }
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return payload."""
        try:
            # Check if token is blacklisted
            if token in self.active_tokens and self.active_tokens[token]["blacklisted"]:
                return None
            
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            # Check expiry
            if datetime.fromtimestamp(payload["exp"]) < datetime.now():
                return None
            
            return payload
        
        except jwt.InvalidTokenError:
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """Generate new access token from refresh token."""
        payload = self.validate_token(refresh_token)
        
        if not payload or payload.get("type") != "refresh":
            return None
        
        # Generate new access token
        access_token = self._generate_token(
            user_id=payload["user_id"],
            username=payload["username"],
            role=payload["role"],
            expiry=self.token_expiry
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": int(self.token_expiry.total_seconds())
        }
    
    def logout(self, token: str) -> bool:
        """Logout user by blacklisting token."""
        payload = self.validate_token(token)
        
        if not payload:
            return False
        
        # Add token to blacklist
        self.active_tokens[token] = {
            "blacklisted": True,
            "blacklisted_at": datetime.now()
        }
        
        return True
    
    def has_permission(self, payload: Dict[str, Any], required_permission: str) -> bool:
        """Check if user has required permission."""
        user_role = UserRole(payload.get("role", "guest"))
        
        # Define role permissions
        role_permissions = {
            UserRole.ADMIN: [
                "user:create", "user:read", "user:update", "user:delete",
                "model:inference", "model:training", "model:management",
                "data:read", "data:write", "data:delete",
                "system:config", "system:monitor", "system:deploy"
            ],
            UserRole.MODERATOR: [
                "user:read", "user:update",
                "model:inference", "model:training",
                "data:read", "data:write",
                "system:monitor"
            ],
            UserRole.USER: [
                "model:inference",
                "data:read", "data:write"
            ],
            UserRole.GUEST: [
                "model:inference",
                "data:read"
            ]
        }
        
        return required_permission in role_permissions.get(user_role, [])
    
    def validate_api_key(self, api_key: str, required_permission: str) -> Optional[Dict[str, Any]]:
        """Validate API key and check permissions."""
        # This would integrate with the user manager's API key validation
        # For now, return a simple implementation
        
        if not api_key.startswith("sk-"):
            return None
        
        # Mock API key validation (in production, use proper key lookup)
        mock_key_info = {
            "user_id": 1,
            "key_name": "API Key",
            "permissions": ["model:inference", "data:read"],
            "rate_limit": 1000
        }
        
        if required_permission not in mock_key_info["permissions"]:
            return None
        
        return mock_key_info
    
    def _generate_token(self, user_id: int, username: str, role: str, 
                       expiry: timedelta, token_type: str = "access") -> str:
        """Generate JWT token."""
        now = datetime.now()
        payload = {
            "user_id": user_id,
            "username": username,
            "role": role,
            "type": token_type,
            "iat": int(now.timestamp()),
            "exp": int((now + expiry).timestamp())
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        try:
            import bcrypt
            salt = bcrypt.gensalt()
            return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        except ImportError:
            # Fallback to simple hash
            import hashlib
            return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        try:
            import bcrypt
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except ImportError:
            # Fallback verification
            import hashlib
            return hashlib.sha256(password.encode()).hexdigest() == hashed
    
    def generate_session_id(self) -> str:
        """Generate secure session ID."""
        return secrets.token_urlsafe(32)
    
    def get_rate_limit_info(self, user_id: int) -> Dict[str, Any]:
        """Get rate limiting information for user."""
        # Mock implementation - in production, check against rate limiter
        return {
            "requests_per_minute": 100,
            "requests_per_hour": 1000,
            "requests_per_day": 10000,
            "current_usage": {
                "minute": 45,
                "hour": 234,
                "day": 1234
            }
        }
    
    def is_rate_limited(self, user_id: int) -> bool:
        """Check if user is rate limited."""
        rate_info = self.get_rate_limit_info(user_id)
        return (
            rate_info["current_usage"]["minute"] >= rate_info["requests_per_minute"] or
            rate_info["current_usage"]["hour"] >= rate_info["requests_per_hour"] or
            rate_info["current_usage"]["day"] >= rate_info["requests_per_day"]
        )