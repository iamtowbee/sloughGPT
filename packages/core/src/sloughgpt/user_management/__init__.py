"""User management module with authentication and authorization."""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import jwt
from datetime import datetime, timedelta


class UserRole(Enum):
    ADMIN = "admin"
    MODERATOR = "moderator" 
    USER = "user"
    GUEST = "guest"


@dataclass
class User:
    id: int
    username: str
    email: str
    password_hash: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime] = None


@dataclass
class APIKey:
    id: int
    user_id: int
    name: str
    key_hash: str
    permissions: List[str]
    rate_limit: int
    created_at: datetime


class UserManager:
    def __init__(self):
        self.users: Dict[int, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.next_user_id = 1
        self.next_key_id = 1
        self.secret_key = "your-secret-key-here"
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        try:
            import bcrypt
            salt = bcrypt.gensalt()
            return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        except ImportError:
            # Fallback to simple hash if bcrypt not available
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
    
    def create_user(self, username: str, email: str, password: str, role: UserRole = UserRole.USER) -> Dict[str, Any]:
        """Create a new user."""
        user_id = self.next_user_id
        self.next_user_id += 1
        
        password_hash = self.hash_password(password)
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
            created_at=datetime.now()
        )
        
        self.users[user_id] = user
        
        return {
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "created_at": user.created_at.isoformat()
            },
            "message": "User created successfully"
        }
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user and return JWT token."""
        for user in self.users.values():
            if user.username == username:
                # In production, verify against stored hash
                if self.verify_password(password, user.password_hash):
                    token = jwt.encode({
                        'user_id': user.id,
                        'username': user.username,
                        'role': user.role.value,
                        'exp': datetime.utcnow() + timedelta(hours=24)
                    }, self.secret_key, algorithm='HS256')
                    
                    user.last_login = datetime.now()
                    
                    return {
                        "access_token": token,
                        "token_type": "bearer",
                        "user": {
                            "id": user.id,
                            "username": user.username,
                            "email": user.email,
                            "role": user.role.value
                        }
                    }
        
        return {"error": "Invalid credentials"}
    
    def create_api_key(self, user_id: int, name: str, permissions: List[str], rate_limit: int = 1000) -> str:
        """Create API key for user."""
        import secrets
        
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")
        
        api_key = secrets.token_urlsafe(32)
        key_hash = self.hash_password(api_key)
        
        key_obj = APIKey(
            id=self.next_key_id,
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            permissions=permissions,
            rate_limit=rate_limit,
            created_at=datetime.now()
        )
        
        self.api_keys[api_key] = key_obj
        self.next_key_id += 1
        
        return api_key


def get_user_manager() -> UserManager:
    """Get singleton user manager instance."""
    return UserManager()