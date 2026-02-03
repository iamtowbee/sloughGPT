#!/usr/bin/env python3
"""
SloughGPT User Management System
Comprehensive user authentication, authorization, and management
"""

import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import jwt
from passlib.context import CryptContext
import json

from .core.database import get_db_session, Base
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, JSON, Float
from sqlalchemy.orm import relationship
from .core.exceptions import create_error, SecurityError

class UserRole(Enum):
    """User roles with hierarchical permissions"""
    ADMIN = "admin"
    MODERATOR = "moderator" 
    USER = "user"
    GUEST = "guest"

class Permission(Enum):
    """System permissions"""
    # Model permissions
    MODEL_TRAIN = "model:train"
    MODEL_INFERENCE = "model:inference"
    MODEL_DEPLOY = "model:deploy"
    MODEL_CONFIGURE = "model:configure"
    
    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_ADMIN = "data:admin"
    
    # User permissions
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_ADMIN = "user:admin"
    
    # System permissions
    SYSTEM_CONFIGURE = "system:configure"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_ADMIN = "system:admin"
    
    # Learning permissions
    LEARNING_START = "learning:start"
    LEARNING_STOP = "learning:stop"
    LEARNING_CONFIGURE = "learning:configure"

# Role-based permission mapping
ROLE_PERMISSIONS = {
    UserRole.GUEST: {
        Permission.MODEL_INFERENCE,
        Permission.DATA_READ
    },
    UserRole.USER: {
        Permission.MODEL_INFERENCE,
        Permission.DATA_READ,
        Permission.DATA_WRITE,
        Permission.LEARNING_START
    },
    UserRole.MODERATOR: {
        Permission.MODEL_INFERENCE,
        Permission.MODEL_TRAIN,
        Permission.DATA_READ,
        Permission.DATA_WRITE,
        Permission.DATA_DELETE,
        Permission.USER_READ,
        Permission.USER_UPDATE,
        Permission.LEARNING_START,
        Permission.LEARNING_CONFIGURE,
        Permission.SYSTEM_MONITOR
    },
    UserRole.ADMIN: {
        # Admins have all permissions
        permission for permission in Permission
    }
}

@dataclass
class UserConfig:
    """User configuration and preferences"""
    max_requests_per_hour: int = 100
    max_tokens_per_request: int = 2048
    allowed_models: List[str] = None
    default_model: str = "sloughgpt-base"
    inference_temperature: float = 0.7
    enable_learning: bool = True
    cost_limit_monthly: float = 100.0
    api_key_limit: int = 5
    
    def __post_init__(self):
        if self.allowed_models is None:
            self.allowed_models = ["sloughgpt-base", "sloughgpt-small"]

class User(Base):
    """User model for database"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default=UserRole.USER.value)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # API keys and authentication
    api_keys = Column(JSON, default=list)
    refresh_tokens = Column(JSON, default=list)
    
    # User preferences
    config = Column(JSON, default=lambda: asdict(UserConfig()))
    
    # Usage tracking
    requests_this_hour = Column(Integer, default=0)
    tokens_this_hour = Column(Integer, default=0)
    cost_this_month = Column(Float, default=0.0)
    last_hour_reset = Column(DateTime, default=datetime.utcnow)
    last_month_reset = Column(DateTime, default=datetime.utcnow)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary (excluding sensitive data)"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "config": self.config,
            "requests_this_hour": self.requests_this_hour,
            "tokens_this_hour": self.tokens_this_hour,
            "cost_this_month": self.cost_this_month,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class APIKey(Base):
    """API key management"""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    key_hash = Column(String(255), nullable=False, unique=True, index=True)
    key_prefix = Column(String(10), nullable=False)  # First 10 chars for identification
    name = Column(String(100), nullable=False)
    permissions = Column(JSON, default=list)  # List of Permission values
    rate_limit = Column(Integer, default=100)  # Requests per hour
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime, nullable=True)
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert API key to dictionary (excluding the actual key)"""
        return {
            "id": self.id,
            "key_prefix": self.key_prefix,
            "name": self.name,
            "permissions": self.permissions,
            "rate_limit": self.rate_limit,
            "is_active": self.is_active,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class UserSession(Base):
    """User session management"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    session_token = Column(String(255), nullable=False, unique=True, index=True)
    refresh_token = Column(String(255), nullable=False, unique=True)
    
    # Session metadata
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "user_agent": self.user_agent,
            "ip_address": self.ip_address,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }

class UserManager:
    """Comprehensive user management system"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.token_expiry = timedelta(hours=24)
        self.refresh_expiry = timedelta(days=30)
        
    def hash_password(self, password: str) -> str:
        """Hash a password securely"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(password, hashed_password)
    
    def generate_api_key(self) -> str:
        """Generate a secure API key"""
        return f"sk-{secrets.token_urlsafe(32)}"
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def generate_jwt_token(self, user_id: int, permissions: List[str], expires_delta: Optional[timedelta] = None) -> Dict[str, str]:
        """Generate JWT tokens for authentication"""
        if expires_delta is None:
            expires_delta = self.token_expiry
            
        now = datetime.utcnow()
        expires = now + expires_delta
        
        payload = {
            "sub": str(user_id),
            "permissions": permissions,
            "iat": now.timestamp(),
            "exp": expires.timestamp(),
            "type": "access"
        }
        
        access_token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Generate refresh token
        refresh_payload = {
            "sub": str(user_id),
            "iat": now.timestamp(),
            "exp": (now + self.refresh_expiry).timestamp(),
            "type": "refresh"
        }
        
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": int(expires_delta.total_seconds())
        }
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def get_user_permissions(self, user_role: UserRole) -> Set[Permission]:
        """Get permissions for a user role"""
        return ROLE_PERMISSIONS.get(user_role, set())
    
    def check_permission(self, user_permissions: Set[Permission], required_permission: Permission) -> bool:
        """Check if user has required permission"""
        return required_permission in user_permissions
    
    def create_user(self, username: str, email: str, password: str, role: UserRole = UserRole.USER, config: Optional[UserConfig] = None) -> Dict[str, Any]:
        """Create a new user"""
        # Validate input
        if not username or not email or not password:
            raise create_error(SecurityError, "Username, email, and password are required", None)
            
        if len(password) < 8:
            raise create_error(SecurityError, "Password must be at least 8 characters", None)
            
        # Check if user already exists
        with get_db_session() as session:
            existing_user = session.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                raise create_error(SecurityError, "Username or email already exists", None)
            
            # Create new user
            password_hash = self.hash_password(password)
            user_config = asdict(config or UserConfig())
            
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                role=role.value,
                config=user_config
            )
            
            session.add(user)
            session.commit()
            session.refresh(user)
            
            # Get user permissions
            permissions = self.get_user_permissions(role)
            permission_names = [p.value for p in permissions]
            
            return {
                "user": user.to_dict(),
                "permissions": permission_names
            }
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with username/password"""
        with get_db_session() as session:
            user = session.query(User).filter(User.username == username).first()
            
            if not user or not user.is_active:
                return None
                
            if not self.verify_password(password, user.password_hash):
                return None
                
            # Update last login
            user.last_login = datetime.utcnow()
            session.commit()
            
            # Get permissions
            user_role = UserRole(user.role)
            permissions = self.get_user_permissions(user_role)
            permission_names = [p.value for p in permissions]
            
            # Generate tokens
            tokens = self.generate_jwt_token(user.id, permission_names)
            
            # Store refresh token
            if user.refresh_tokens is None:
                user.refresh_tokens = []
            user.refresh_tokens.append(tokens["refresh_token"])
            session.commit()
            
            return {
                "user": user.to_dict(),
                "permissions": permission_names,
                **tokens
            }
    
    def authenticate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with API key"""
        key_hash = self.hash_api_key(api_key)
        
        with get_db_session() as session:
            # Find API key
            api_key_record = session.query(APIKey).filter(
                APIKey.key_hash == key_hash,
                APIKey.is_active == True
            ).first()
            
            if not api_key_record:
                return None
                
            # Check if key is expired
            if api_key_record.expires_at and api_key_record.expires_at < datetime.utcnow():
                return None
                
            # Get user
            user = session.query(User).filter(User.id == api_key_record.user_id).first()
            
            if not user or not user.is_active:
                return None
                
            # Update usage
            api_key_record.usage_count += 1
            api_key_record.last_used = datetime.utcnow()
            session.commit()
            
            # Get permissions from API key or user role
            if api_key_record.permissions:
                permissions = set(Permission(p) for p in api_key_record.permissions)
            else:
                user_role = UserRole(user.role)
                permissions = self.get_user_permissions(user_role)
                
            permission_names = [p.value for p in permissions]
            
            return {
                "user": user.to_dict(),
                "permissions": permission_names,
                "api_key_id": api_key_record.id,
                "rate_limit": api_key_record.rate_limit
            }
    
    def create_api_key(self, user_id: int, name: str, permissions: Optional[List[str]] = None, 
                      rate_limit: int = 100, expires_in_days: Optional[int] = None) -> Dict[str, Any]:
        """Create a new API key for a user"""
        with get_db_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            
            if not user:
                raise create_error(SecurityError, "User not found", None)
                
            # Check API key limit
            user_config = UserConfig(**user.config)
            existing_keys = session.query(APIKey).filter(APIKey.user_id == user_id).count()
            
            if existing_keys >= user_config.api_key_limit:
                raise create_error(SecurityError, "API key limit reached", None)
            
            # Generate API key
            api_key = self.generate_api_key()
            api_key_hash = self.hash_api_key(api_key)
            
            # Set expiration
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            # Create API key record
            api_key_record = APIKey(
                user_id=user_id,
                key_hash=api_key_hash,
                key_prefix=api_key[:10],
                name=name,
                permissions=permissions or [],
                rate_limit=rate_limit,
                expires_at=expires_at
            )
            
            session.add(api_key_record)
            session.commit()
            session.refresh(api_key_record)
            
            return {
                "api_key": api_key,  # Only return once during creation
                "key_record": api_key_record.to_dict()
            }
    
    def revoke_api_key(self, api_key_id: int, user_id: int) -> bool:
        """Revoke an API key"""
        with get_db_session() as session:
            api_key = session.query(APIKey).filter(
                APIKey.id == api_key_id,
                APIKey.user_id == user_id
            ).first()
            
            if not api_key:
                return False
                
            api_key.is_active = False
            session.commit()
            return True
    
    def update_user(self, user_id: int, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update user information"""
        with get_db_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            
            if not user:
                return None
                
            # Update allowed fields
            allowed_fields = {"email", "role", "is_active", "config"}
            
            for field, value in updates.items():
                if field in allowed_fields:
                    setattr(user, field, value)
            
            user.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(user)
            
            return user.to_dict()
    
    def change_password(self, user_id: int, current_password: str, new_password: str) -> bool:
        """Change user password"""
        with get_db_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            
            if not user:
                return False
                
            if not self.verify_password(current_password, user.password_hash):
                return False
                
            if len(new_password) < 8:
                return False
                
            user.password_hash = self.hash_password(new_password)
            session.commit()
            return True
    
    def reset_password(self, email: str) -> str:
        """Generate password reset token"""
        import secrets
        
        reset_token = secrets.token_urlsafe(32)
        expiry = datetime.utcnow() + timedelta(hours=1)
        
        # Store reset token (you might want to create a separate table for this)
        with get_db_session() as session:
            user = session.query(User).filter(User.email == email).first()
            
            if not user:
                # Don't reveal if email exists
                return reset_token
                
            # For simplicity, store token in user's config
            if isinstance(user.config, dict):
                user.config["password_reset_token"] = {
                    "token": reset_token,
                    "expires": expiry.isoformat()
                }
            
            session.commit()
            return reset_token
    
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get user statistics and usage"""
        with get_db_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            
            if not user:
                return {}
                
            # Get API keys
            api_keys = session.query(APIKey).filter(APIKey.user_id == user_id).all()
            
            # Get sessions
            sessions = session.query(UserSession).filter(
                UserSession.user_id == user_id,
                UserSession.is_active == True
            ).all()
            
            return {
                "user": user.to_dict(),
                "api_keys": [key.to_dict() for key in api_keys],
                "active_sessions": [session.to_dict() for session in sessions],
                "usage_summary": {
                    "requests_this_hour": user.requests_this_hour,
                    "tokens_this_hour": user.tokens_this_hour,
                    "cost_this_month": user.cost_this_month
                }
            }

# Global user manager instance
_user_manager = None

def get_user_manager() -> UserManager:
    """Get the global user manager instance"""
    global _user_manager
    if _user_manager is None:
        secret_key = secrets.token_urlsafe(32)  # In production, use environment variable
        _user_manager = UserManager(secret_key)
    return _user_manager

# Decorators for authentication and authorization
def require_auth(permission: Optional[Permission] = None):
    """Decorator to require authentication"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This is a simplified version - in a real FastAPI app, you'd use proper middleware
            token = kwargs.get('token') or args[0] if args else None
            
            if not token:
                raise create_error(SecurityError, "Authentication required", None)
                
            user_manager = get_user_manager()
            payload = user_manager.verify_jwt_token(token)
            
            if not payload:
                raise create_error(SecurityError, "Invalid or expired token", None)
                
            user_id = int(payload['sub'])
            
            if permission:
                # Check permission here
                pass
                
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_api_key():
    """Decorator to require API key authentication"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            api_key = kwargs.get('api_key')
            
            if not api_key:
                raise create_error(SecurityError, "API key required", None)
                
            user_manager = get_user_manager()
            auth_result = user_manager.authenticate_api_key(api_key)
            
            if not auth_result:
                raise create_error(SecurityError, "Invalid API key", None)
                
            return func(*args, **kwargs)
        return wrapper
    return decorator

# CLI interface
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="SloughGPT User Management")
    parser.add_argument("command", choices=["create-user", "authenticate", "create-api-key"], 
                       help="Command to execute")
    parser.add_argument("--username", help="Username")
    parser.add_argument("--email", help="Email")
    parser.add_argument("--password", help="Password")
    parser.add_argument("--role", default="user", choices=["admin", "moderator", "user", "guest"],
                       help="User role")
    parser.add_argument("--api-key-name", help="API key name")
    
    def main():
        args = parser.parse_args()
        user_manager = get_user_manager()
        
        if args.command == "create-user":
            if not all([args.username, args.email, args.password]):
                print("❌ Username, email, and password are required")
                sys.exit(1)
                
            role = UserRole(args.role)
            result = user_manager.create_user(args.username, args.email, args.password, role)
            print(f"✅ User created: {result['user']['username']} ({result['user']['role']})")
            
        elif args.command == "authenticate":
            if not all([args.username, args.password]):
                print("❌ Username and password are required")
                sys.exit(1)
                
            result = user_manager.authenticate_user(args.username, args.password)
            if result:
                print(f"✅ Authentication successful")
                print(f"🔑 Access token: {result['access_token'][:50]}...")
            else:
                print("❌ Authentication failed")
                
        elif args.command == "create-api-key":
            if not all([args.username, args.api_key_name]):
                print("❌ Username and API key name are required")
                sys.exit(1)
                
            # First authenticate to get user ID
            auth_result = user_manager.authenticate_user(args.username, args.password or "")
            if auth_result:
                user_id = auth_result['user']['id']
                result = user_manager.create_api_key(user_id, args.api_key_name)
                print(f"✅ API key created: {result['api_key'][:20]}...")
                print(f"📋 Key name: {result['key_record']['name']}")
            else:
                print("❌ Authentication failed")
    
    main()