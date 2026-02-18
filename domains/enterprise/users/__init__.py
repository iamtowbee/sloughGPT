"""
User Manager Implementation

This module provides user management capabilities including
user CRUD operations, role management, and profile management.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from ...__init__ import (
    BaseComponent,
    ComponentException,
    IUserManager,
    SecurityLevel,
    User,
    UserRole,
)


class UserManager(BaseComponent, IUserManager):
    """Advanced user management system"""

    def __init__(self) -> None:
        super().__init__("user_manager")
        self.logger = logging.getLogger(f"sloughgpt.{self.component_name}")

        # User storage
        self.users: Dict[str, User] = {}
        self.user_profiles: Dict[str, Dict[str, Any]] = {}

        # User statistics
        self.stats: Dict[str, Any] = {
            "total_users": 0,
            "active_users": 0,
            "users_by_role": {},
            "users_created_today": 0,
        }

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize user manager"""
        try:
            self.logger.info("Initializing User Manager...")
            self.is_initialized = True
            self.logger.info("User Manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize User Manager: {e}")
            raise ComponentException(f"User Manager initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown user manager"""
        try:
            self.logger.info("Shutting down User Manager...")
            self.is_initialized = False
            self.logger.info("User Manager shutdown successfully")

        except Exception as e:
            self.logger.error(f"Failed to shutdown User Manager: {e}")
            raise ComponentException(f"User Manager shutdown failed: {e}")

    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a new user"""
        try:
            user_id = str(uuid.uuid4())

            user = User(
                id=user_id,
                username=user_data["username"],
                email=user_data.get("email", ""),
                role=UserRole(user_data.get("role", "user")),
                security_level=SecurityLevel(user_data.get("security_level", "internal")),
                created_at=time.time(),
                last_active=time.time(),
                metadata=user_data.get("metadata", {}),
            )

            self.users[user_id] = user
            self.stats["total_users"] = self.stats["total_users"] + 1

            self.logger.info(f"Created user {user.username}")
            return user

        except Exception as e:
            self.logger.error(f"Failed to create user: {e}")
            raise ComponentException(f"User creation failed: {e}")

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)

    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user"""
        try:
            if user_id not in self.users:
                return False

            user = self.users[user_id]

            # Update allowed fields
            if "email" in updates:
                user.email = updates["email"]
            if "role" in updates:
                user.role = UserRole(updates["role"])
            if "security_level" in updates:
                user.security_level = SecurityLevel(updates["security_level"])
            if "metadata" in updates:
                user.metadata.update(updates["metadata"])

            user.last_active = time.time()

            self.logger.info(f"Updated user {user_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update user {user_id}: {e}")
            return False

    async def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        try:
            if user_id in self.users:
                username = self.users[user_id].username
                del self.users[user_id]
                current_count = self.stats.get("total_users", 0)
                self.stats["total_users"] = max(0, int(current_count) - 1)

                self.logger.info(f"Deleted user {username}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to delete user {user_id}: {e}")
            return False

    async def list_users(
        self, filters: Optional[Dict[str, Any]] = None, limit: int = 100
    ) -> List[User]:
        """List users with optional filters"""
        users = list(self.users.values())

        if filters:
            # Apply filters
            if "role" in filters:
                users = [u for u in users if u.role.value == filters["role"]]
            if "security_level" in filters:
                users = [u for u in users if u.security_level.value == filters["security_level"]]

        # Sort by creation time (newest first)
        users.sort(key=lambda u: u.created_at, reverse=True)

        return users[:limit]

    async def get_user_statistics(self) -> Dict[str, Any]:
        """Get user statistics"""
        stats = self.stats.copy()

        # Calculate role distribution
        role_counts: Dict[str, int] = {}
        for user in self.users.values():
            role = user.role.value
            current_count = role_counts.get(role, 0)
            role_counts[role] = current_count + 1

        stats["users_by_role"] = role_counts
        stats["active_users"] = len(
            [u for u in self.users.values() if time.time() - u.last_active < 86400]
        )  # Active in last 24h

        return stats
