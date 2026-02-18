"""
Authentication Service Implementation

This module provides comprehensive authentication capabilities including
JWT tokens, password management, and session handling.
"""


import asyncio
import hashlib
import logging
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt

from ...__init__ import (
    BaseComponent,
    ComponentException,
    IAuthenticationService,
    SecurityLevel,
    User,
    UserRole,
)


@dataclass
class AuthSession:
    """Authentication session information"""

    session_id: str
    user_id: str
    token: str
    created_at: float
    expires_at: float
    last_activity: float
    ip_address: Optional[str]
    user_agent: Optional[str]


@dataclass
class PasswordResetRequest:
    """Password reset request information"""

    request_id: str
    user_id: str
    token: str
    created_at: float
    expires_at: float
    is_used: bool


class AuthenticationService(BaseComponent, IAuthenticationService):
    """Advanced authentication service"""

    def __init__(self) -> None:
        super().__init__("authentication_service")
        self.logger = logging.getLogger(f"sloughgpt.{self.component_name}")

        # Authentication storage
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, AuthSession] = {}
        self.password_resets: Dict[str, PasswordResetRequest] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}

        # Security configuration
        self.jwt_secret = secrets.token_urlsafe(32)
        self.jwt_algorithm = "HS256"
        self.token_expiry = 3600  # 1 hour
        self.refresh_token_expiry = 86400  # 24 hours
        self.max_sessions_per_user = 5

        # Password policies
        self.password_policies = {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_digits": True,
            "require_special_chars": True,
            "max_age_days": 90,
        }

        # Security metrics
        self.security_metrics: Dict[str, Any] = {
            "total_logins": 0,
            "successful_logins": 0,
            "failed_logins": 0,
            "active_sessions": 0,
            "password_resets": 0,
        }

        # Background tasks
        self.cleanup_task: Optional[asyncio.Task[None]] = None

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize authentication service"""
        try:
            self.logger.info("Initializing Authentication Service...")

            # Create default admin user if not exists
            await self._create_default_admin()

            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

            self.is_initialized = True
            self.logger.info("Authentication Service initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Authentication Service: {e}")
            raise ComponentException(f"Authentication Service initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown authentication service"""
        try:
            self.logger.info("Shutting down Authentication Service...")

            # Cancel cleanup task
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass

            self.is_initialized = False
            self.logger.info("Authentication Service shutdown successfully")

        except Exception as e:
            self.logger.error(f"Failed to shutdown Authentication Service: {e}")
            raise ComponentException(f"Authentication Service shutdown failed: {e}")

    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[User]:
        """Authenticate user credentials"""
        try:
            username = credentials.get("username")
            password = credentials.get("password")

            if not username or not password:
                self.security_metrics["failed_logins"] += 1
                return None

            # Find user
            user = None
            for u in self.users.values():
                if u.username == username:
                    user = u
                    break

            if not user:
                self.security_metrics["failed_logins"] += 1
                self.logger.warning(f"Authentication failed: user {username} not found")
                return None

            # Verify password
            if not await self._verify_password(password, user.metadata.get("password_hash", "")):
                self.security_metrics["failed_logins"] += 1
                self.logger.warning(f"Authentication failed: invalid password for user {username}")
                return None

            # Check if user is active
            if not await self._is_user_active(user):
                self.security_metrics["failed_logins"] += 1
                self.logger.warning(f"Authentication failed: user {username} is not active")
                return None

            self.security_metrics["successful_logins"] += 1
            self.security_metrics["total_logins"] += 1

            self.logger.info(f"User {username} authenticated successfully")
            return user

        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            self.security_metrics["failed_logins"] += 1
            return None

    async def authorize(self, user: User, resource: str, action: str) -> bool:
        """Authorize user for resource action"""
        try:
            # Check user role permissions
            if not await self._check_role_permission(user.role, resource, action):
                return False

            # Check security level
            if not await self._check_security_level(user, resource):
                return False

            # Check custom permissions (if any)
            if "permissions" in user.metadata:
                user_permissions = user.metadata["permissions"]
                required_permission = f"{resource}:{action}"
                if required_permission not in user_permissions:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Authorization error: {e}")
            return False

    async def generate_token(self, user: User) -> str:
        """Generate authentication token"""
        try:
            # Create JWT payload
            payload = {
                "user_id": user.id,
                "username": user.username,
                "role": user.role.value,
                "security_level": user.security_level.value,
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(seconds=self.token_expiry),
                "type": "access",
            }

            # Generate token
            token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

            # Create session
            await self._create_session(user.id, token)

            self.logger.debug(f"Generated token for user {user.id}")
            return token

        except Exception as e:
            self.logger.error(f"Token generation error: {e}")
            raise ComponentException(f"Token generation failed: {e}")

    async def validate_token(self, token: str) -> Optional[User]:
        """Validate authentication token"""
        try:
            # Decode JWT
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])

            # Find user
            user_id = payload.get("user_id")
            if user_id not in self.users:
                return None

            user = self.users[user_id]

            # Check if session is still valid
            session_id = payload.get("session_id")
            if session_id and session_id not in self.sessions:
                return None

            return user

        except jwt.ExpiredSignatureError:
            self.logger.warning("Token validation failed: token expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Token validation failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Token validation error: {e}")
            return None

    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a new user"""
        try:
            # Generate user ID
            import uuid

            user_id = str(uuid.uuid4())

            # Hash password
            password = user_data.get("password", "")
            password_hash = await self._hash_password(password)

            # Create user object
            user = User(
                id=user_id,
                username=user_data["username"],
                email=user_data.get("email", ""),
                role=UserRole(user_data.get("role", "user")),
                security_level=SecurityLevel(user_data.get("security_level", "internal")),
                created_at=time.time(),
                last_active=time.time(),
                metadata={
                    "password_hash": password_hash,
                    "created_by": "system",
                    "last_password_change": time.time(),
                },
            )

            # Store user
            self.users[user_id] = user

            self.logger.info(f"Created user {user.username} with ID {user_id}")
            return user

        except Exception as e:
            self.logger.error(f"User creation error: {e}")
            raise ComponentException(f"User creation failed: {e}")

    async def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """Change user password"""
        try:
            if user_id not in self.users:
                return False

            user = self.users[user_id]
            current_hash = user.metadata.get("password_hash", "")

            # Verify old password
            if not await self._verify_password(old_password, current_hash):
                return False

            # Validate new password
            if not await self._validate_password(new_password):
                return False

            # Update password
            new_hash = await self._hash_password(new_password)
            user.metadata["password_hash"] = new_hash
            user.metadata["last_password_change"] = time.time()

            # Invalidate existing sessions
            await self._invalidate_user_sessions(user_id)

            self.logger.info(f"Password changed for user {user_id}")
            return True

        except Exception as e:
            self.logger.error(f"Password change error: {e}")
            return False

    async def request_password_reset(self, email: str) -> Optional[str]:
        """Request password reset"""
        try:
            # Find user by email
            user = None
            for u in self.users.values():
                if u.email == email:
                    user = u
                    break

            if not user:
                return None

            # Generate reset token
            reset_token = secrets.token_urlsafe(32)
            request_id = str(uuid.uuid4())

            # Create reset request
            reset_request = PasswordResetRequest(
                request_id=request_id,
                user_id=user.id,
                token=reset_token,
                created_at=time.time(),
                expires_at=time.time() + 3600,  # 1 hour
                is_used=False,
            )

            self.password_resets[request_id] = reset_request
            self.security_metrics["password_resets"] += 1

            self.logger.info(f"Password reset requested for user {user.id}")
            return reset_token

        except Exception as e:
            self.logger.error(f"Password reset request error: {e}")
            return None

    async def reset_password(self, reset_token: str, new_password: str) -> bool:
        """Reset password using reset token"""
        try:
            # Find reset request
            reset_request = None
            for request in self.password_resets.values():
                if request.token == reset_token and not request.is_used:
                    reset_request = request
                    break

            if not reset_request:
                return False

            # Check if expired
            if time.time() > reset_request.expires_at:
                return False

            # Validate new password
            if not await self._validate_password(new_password):
                return False

            # Update password
            user_id = reset_request.user_id
            if user_id in self.users:
                user = self.users[user_id]
                new_hash = await self._hash_password(new_password)
                user.metadata["password_hash"] = new_hash
                user.metadata["last_password_change"] = time.time()

                # Mark reset request as used
                reset_request.is_used = True

                # Invalidate existing sessions
                await self._invalidate_user_sessions(user_id)

                self.logger.info(f"Password reset completed for user {user_id}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Password reset error: {e}")
            return False

    async def logout(self, token: str) -> bool:
        """Logout user and invalidate token"""
        try:
            # Decode token to get session info
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            session_id = payload.get("session_id")

            if session_id and session_id in self.sessions:
                del self.sessions[session_id]
                self.security_metrics["active_sessions"] = max(
                    0, self.security_metrics["active_sessions"] - 1
                )

                self.logger.info(f"User logged out, session {session_id} invalidated")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Logout error: {e}")
            return False

    async def get_authentication_statistics(self) -> Dict[str, Any]:
        """Get authentication statistics"""
        stats = self.security_metrics.copy()

        # Add current state
        stats["current_users"] = len(self.users)
        stats["current_sessions"] = len(self.sessions)
        stats["active_password_resets"] = len(
            [r for r in self.password_resets.values() if not r.is_used]
        )

        # Calculate success rate
        if stats["total_logins"] > 0:
            stats["success_rate"] = stats["successful_logins"] / stats["total_logins"]
        else:
            stats["success_rate"] = 0.0

        return stats

    # Private helper methods

    async def _hash_password(self, password: str) -> str:
        """Hash password using secure algorithm"""
        # Use SHA-256 with salt (in production, use bcrypt or argon2)
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"

    async def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            if ":" not in password_hash:
                return False

            salt, hash_value = password_hash.split(":", 1)
            computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return computed_hash == hash_value

        except Exception:
            return False

    async def _validate_password(self, password: str) -> bool:
        """Validate password against policies"""
        if len(password) < self.password_policies["min_length"]:
            return False

        if self.password_policies["require_uppercase"] and not any(c.isupper() for c in password):
            return False

        if self.password_policies["require_lowercase"] and not any(c.islower() for c in password):
            return False

        if self.password_policies["require_digits"] and not any(c.isdigit() for c in password):
            return False

        if self.password_policies["require_special_chars"] and not any(
            c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password
        ):
            return False

        return True

    async def _is_user_active(self, user: User) -> bool:
        """Check if user is active"""
        # Check if user is not suspended/blocked
        if user.metadata.get("suspended", False):
            return False

        # Check if user account is not expired
        if "expires_at" in user.metadata and time.time() > user.metadata["expires_at"]:
            return False

        return True

    async def _check_role_permission(self, role: UserRole, resource: str, action: str) -> bool:
        """Check role-based permissions"""
        # Define role permissions
        role_permissions = {
            UserRole.ADMIN: ["*:*"],  # All permissions
            UserRole.USER: [
                "profile:read",
                "profile:update",
                "data:read",
                "data:create",
                "api:read",
            ],
            UserRole.USER: ["data:create", "api:read"],
            UserRole.GUEST: ["public:read"],
        }

        user_permissions = role_permissions.get(role, [])
        required_permission = f"{resource}:{action}"

        # Check for wildcard permission
        if "*:*" in user_permissions:
            return True

        # Check for specific permission
        return required_permission in user_permissions

    async def _check_security_level(self, user: User, resource: str) -> bool:
        """Check security level access"""
        # Define resource security levels
        resource_security = {
            "admin": SecurityLevel.RESTRICTED,
            "user_management": SecurityLevel.CONFIDENTIAL,
            "sensitive_data": SecurityLevel.CONFIDENTIAL,
            "public_data": SecurityLevel.PUBLIC,
            "internal_data": SecurityLevel.INTERNAL,
        }

        required_level = resource_security.get(resource, SecurityLevel.INTERNAL)

        # User can access resources at or below their security level
        level_order = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.RESTRICTED: 3,
        }

        return level_order.get(user.security_level, 0) >= level_order.get(required_level, 0)

    async def _create_session(self, user_id: str, token: str) -> str:
        """Create authentication session"""
        import uuid

        session_id = str(uuid.uuid4())

        session = AuthSession(
            session_id=session_id,
            user_id=user_id,
            token=token,
            created_at=time.time(),
            expires_at=time.time() + self.token_expiry,
            last_activity=time.time(),
            ip_address=None,  # Would be set from request
            user_agent=None,  # Would be set from request
        )

        self.sessions[session_id] = session
        self.security_metrics["active_sessions"] += 1

        return session_id

    async def _invalidate_user_sessions(self, user_id: str) -> None:
        """Invalidate all sessions for a user"""
        sessions_to_remove = []

        for session_id, session in self.sessions.items():
            if session.user_id == user_id:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.sessions[session_id]
            self.security_metrics["active_sessions"] = max(
                0, self.security_metrics["active_sessions"] - 1
            )

    async def _create_default_admin(self) -> None:
        """Create default admin user if not exists"""
        admin_exists = any(u.role == UserRole.ADMIN for u in self.users.values())

        if not admin_exists:
            await self.create_user(
                {
                    "username": "admin",
                    "email": "admin@sloughgpt.local",
                    "password": "admin123!",  # Should be changed on first login
                    "role": "admin",
                    "security_level": "confidential",
                }
            )

            self.logger.info("Created default admin user")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while self.is_initialized:
            try:
                await self._cleanup_expired_sessions()
                await self._cleanup_expired_resets()
                await asyncio.sleep(300)  # Clean up every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Authentication cleanup error: {e}")
                await asyncio.sleep(60)

    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if current_time > session.expires_at:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.sessions[session_id]
            self.security_metrics["active_sessions"] = max(
                0, self.security_metrics["active_sessions"] - 1
            )

        if expired_sessions:
            self.logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def _cleanup_expired_resets(self) -> None:
        """Clean up expired password reset requests"""
        current_time = time.time()
        expired_resets = []

        for request_id, request in self.password_resets.items():
            if current_time > request.expires_at:
                expired_resets.append(request_id)

        for request_id in expired_resets:
            del self.password_resets[request_id]

        if expired_resets:
            self.logger.debug(f"Cleaned up {len(expired_resets)} expired password reset requests")
