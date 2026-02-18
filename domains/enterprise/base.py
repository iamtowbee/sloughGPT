"""
Enterprise Domain Base Class

This module contains the base enterprise domain implementation.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from ..__init__ import (
    BaseDomain,
    DomainException,
    IAuthenticationService,
    ICostOptimizer,
    IMonitoringService,
    IUserManager,
    User,
)


class EnterpriseDomain(BaseDomain):
    """Main enterprise core domain"""

    def __init__(self) -> None:
        super().__init__("enterprise")
        self.logger = logging.getLogger(f"sloughgpt.{self.domain_name}")

        # Core components
        self.authentication_service: Optional[IAuthenticationService] = None
        self.user_manager: Optional[IUserManager] = None
        self.monitoring_service: Optional[IMonitoringService] = None
        self.cost_optimizer: Optional[ICostOptimizer] = None

        # Enterprise state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.security_events: List[Dict[str, Any]] = []
        self.cost_metrics: Dict[str, Any] = {}

        # Security settings
        self.security_policies = {
            "max_failed_attempts": 5,
            "session_timeout": 3600,  # 1 hour
            "password_min_length": 8,
            "require_mfa": False,
        }

    async def _on_initialize(self) -> None:
        """Initialize enterprise domain components"""
        try:
            self.logger.info("Initializing Enterprise Domain...")

            # Initialize core components
            await self._initialize_authentication_service()
            await self._initialize_user_manager()
            await self._initialize_monitoring_service()
            await self._initialize_cost_optimizer()

            # Start security monitoring
            await self._start_security_monitoring()

            self.logger.info("Enterprise Domain initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Enterprise Domain: {e}")
            raise EnterpriseException(f"Initialization failed: {e}")

    async def _on_shutdown(self) -> None:
        """Shutdown enterprise domain components"""
        try:
            self.logger.info("Shutting down Enterprise Domain...")

            # Stop security monitoring
            await self._stop_security_monitoring()

            # Shutdown components
            if self.cost_optimizer:
                await self._shutdown_component("cost_optimizer")
            if self.monitoring_service:
                await self._shutdown_component("monitoring_service")
            if self.user_manager:
                await self._shutdown_component("user_manager")
            if self.authentication_service:
                await self._shutdown_component("authentication_service")

            self.logger.info("Enterprise Domain shutdown successfully")

        except Exception as e:
            self.logger.error(f"Failed to shutdown Enterprise Domain: {e}")
            raise EnterpriseException(f"Shutdown failed: {e}")

    async def _initialize_authentication_service(self) -> None:
        """Initialize authentication service"""
        from .auth import AuthenticationService

        self.authentication_service = AuthenticationService()
        await self.authentication_service.initialize()
        self.components["authentication_service"] = self.authentication_service

    async def _initialize_user_manager(self) -> None:
        """Initialize user manager"""
        from .users import UserManager

        self.user_manager = UserManager()
        await self.user_manager.initialize()
        self.components["user_manager"] = self.user_manager

    async def _initialize_monitoring_service(self) -> None:
        """Initialize monitoring service"""
        from .monitoring import MonitoringService

        self.monitoring_service = MonitoringService()
        await self.monitoring_service.initialize()
        self.components["monitoring_service"] = self.monitoring_service

    async def _initialize_cost_optimizer(self) -> None:
        """Initialize cost optimizer"""
        from .cost import CostOptimizer

        self.cost_optimizer = CostOptimizer()
        await self.cost_optimizer.initialize()
        self.components["cost_optimizer"] = self.cost_optimizer

    async def _start_security_monitoring(self) -> None:
        """Start security monitoring"""
        # Start security monitoring loops
        asyncio.create_task(self._security_monitoring_loop())
        asyncio.create_task(self._session_cleanup_loop())

    async def _stop_security_monitoring(self) -> None:
        """Stop security monitoring"""
        # Implementation for stopping security monitoring
        pass

    async def _security_monitoring_loop(self) -> None:
        """Background security monitoring loop"""
        while self.is_initialized:
            try:
                await self._monitor_security_events()
                await asyncio.sleep(60)  # Monitor every minute
            except Exception as e:
                self.logger.error(f"Security monitoring error: {e}")
                await asyncio.sleep(10)

    async def _session_cleanup_loop(self) -> None:
        """Background session cleanup loop"""
        while self.is_initialized:
            try:
                await self._cleanup_expired_sessions()
                await asyncio.sleep(300)  # Clean up every 5 minutes
            except Exception as e:
                self.logger.error(f"Session cleanup error: {e}")
                await asyncio.sleep(60)

    async def _monitor_security_events(self) -> None:
        """Monitor security events"""
        current_time = time.time()

        # Check for suspicious activities
        for session_id, session in list(self.active_sessions.items()):
            # Check session timeout
            if current_time - session["created_at"] > self.security_policies["session_timeout"]:
                await self._terminate_session(session_id)

        # Monitor failed login attempts
        failed_attempts = [
            event for event in self.security_events[-100:] if event.get("type") == "failed_login"
        ]

        # Detect potential brute force attacks
        if len(failed_attempts) > self.security_policies["max_failed_attempts"]:
            await self._handle_security_threat("potential_brute_force", failed_attempts)

    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = []

        for session_id, session in self.active_sessions.items():
            if current_time - session["last_activity"] > self.security_policies["session_timeout"]:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            await self._terminate_session(session_id)

    async def _terminate_session(self, session_id: str) -> None:
        """Terminate a session"""
        if session_id in self.active_sessions:
            user_id = self.active_sessions[session_id]["user_id"]
            del self.active_sessions[session_id]

            # Log session termination
            await self._log_security_event(
                "session_terminated",
                {"session_id": session_id, "user_id": user_id, "reason": "timeout"},
            )

            self.logger.info(f"Terminated session {session_id} for user {user_id}")

    async def _handle_security_threat(self, threat_type: str, events: List[Dict[str, Any]]) -> None:
        """Handle security threats"""
        threat_data = {"threat_type": threat_type, "events": events, "detected_at": time.time()}

        # Log security threat
        await self._log_security_event("security_threat", threat_data)

        # Implement threat response
        if threat_type == "potential_brute_force":
            # Block IP temporarily (implementation needed)
            self.logger.warning(
                f"Potential brute force attack detected: {len(events)} failed attempts"
            )

    async def _log_security_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a security event"""
        event = {"type": event_type, "data": data, "timestamp": time.time()}

        self.security_events.append(event)

        # Keep only recent events
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-5000:]

        # Send to monitoring service if available
        if self.monitoring_service:
            await self.monitoring_service.log_event(event_type, data)

    async def _shutdown_component(self, component_name: str) -> None:
        """Safely shutdown a component"""
        try:
            component = getattr(self, component_name)
            if hasattr(component, "shutdown"):
                await component.shutdown()
        except Exception as e:
            self.logger.error(f"Error shutting down {component_name}: {e}")

    # Public API methods

    async def authenticate_user(self, credentials: Dict[str, Any]) -> Optional[User]:
        """Authenticate a user"""
        if not self.authentication_service:
            raise EnterpriseException("Authentication service not initialized")

        try:
            user = await self.authentication_service.authenticate(credentials)

            if user:
                # Create session
                session_id = await self._create_session(user)

                # Log successful authentication
                await self._log_security_event(
                    "successful_login", {"user_id": user.id, "session_id": session_id}
                )

                self.logger.info(f"User {user.id} authenticated successfully")
            else:
                # Log failed authentication
                await self._log_security_event(
                    "failed_login",
                    {
                        "credentials": credentials.get("username", "unknown"),
                        "reason": "invalid_credentials",
                    },
                )

            return user

        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            await self._log_security_event("authentication_error", {"error": str(e)})
            return None

    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a new user"""
        if not self.user_manager:
            raise EnterpriseException("User manager not initialized")

        try:
            user = await self.user_manager.create_user(user_data)

            # Log user creation
            await self._log_security_event(
                "user_created", {"user_id": user.id, "username": user.username}
            )

            self.logger.info(f"User {user.id} created successfully")
            return user

        except Exception as e:
            self.logger.error(f"User creation failed: {e}")
            raise EnterpriseException(f"User creation failed: {e}")

    async def track_enterprise_metric(
        self, metric_name: str, value: float, tags: Dict[str, str]
    ) -> None:
        """Track enterprise-level metrics"""
        if self.monitoring_service:
            await self.monitoring_service.track_metric(metric_name, value, tags)

        # Track cost-related metrics
        if metric_name.startswith("cost_"):
            self.cost_metrics[metric_name] = value

    async def get_enterprise_status(self) -> Dict[str, Any]:
        """Get current enterprise status"""
        return {
            "active_sessions": len(self.active_sessions),
            "security_events_count": len(self.security_events),
            "components_status": {
                name: "initialized" if component else "not_initialized"
                for name, component in [
                    ("authentication_service", self.authentication_service),
                    ("user_manager", self.user_manager),
                    ("monitoring_service", self.monitoring_service),
                    ("cost_optimizer", self.cost_optimizer),
                ]
            },
            "cost_metrics": self.cost_metrics,
            "security_policies": self.security_policies,
        }

    async def _create_session(self, user: User) -> str:
        """Create a new session for user"""
        import uuid

        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            "user_id": user.id,
            "username": user.username,
            "role": user.role,
            "created_at": time.time(),
            "last_activity": time.time(),
        }

        return session_id


class EnterpriseException(DomainException):
    """Enterprise domain specific exceptions"""

    pass
