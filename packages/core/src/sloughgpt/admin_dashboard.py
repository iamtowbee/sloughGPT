"""Admin dashboard interface for SloughGPT."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import json
import logging

try:
    from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from pydantic import BaseModel
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    from .api import app
    from .user_management import get_user_manager, UserRole
    from .cost_optimization import get_cost_optimizer, CostMetricType
    from .monitoring import MonitoringService
    from .database import database_manager
    from .security_advanced import security_manager
    from .logging_system import log_manager
    from .ml_pipeline import ml_pipeline
    from .websocket import websocket_manager
    HAS_INTEGRATIONS = True
except ImportError:
    HAS_INTEGRATIONS = False


# Pydantic models for dashboard
if HAS_FASTAPI:
    class DashboardStats(BaseModel):
        total_users: int = 0
        active_users: int = 0
        total_api_keys: int = 0
        active_api_keys: int = 0
        total_cost_today: float = 0.0
        total_requests_today: int = 0
        total_models_trained: int = 0
        active_training_jobs: int = 0
        system_health: str = "healthy"
        storage_usage: Dict[str, Any] = {}
    
    class UserInfo(BaseModel):
        id: int
        username: str
        email: str
        role: str
        is_active: bool
        created_at: str
        last_login: Optional[str] = None
        total_cost: float = 0.0
        api_keys_count: int = 0
    
    class SystemMetrics(BaseModel):
        timestamp: str
        cpu_percent: float
        memory_percent: float
        disk_usage: Dict[str, Any]
        active_connections: int
        requests_per_minute: float
        response_time_avg: float
        cache_hit_rate: float
        error_rate: float
    
    class SecurityEvent(BaseModel):
        event_id: str
        timestamp: str
        event_type: str
        severity: str
        user_id: Optional[int]
        ip_address: str
        details: Dict[str, Any]
        resolved: bool
    
    class MLModelInfo(BaseModel):
        model_id: str
        name: str
        type: str
        status: str
        accuracy: Optional[float]
        created_at: str
        training_progress: Optional[float] = None
        metrics: Dict[str, Any]


class AdminDashboard:
    """Admin dashboard for SloughGPT management."""
    
    def __init__(self):
        if not HAS_FASTAPI:
            raise ImportError("FastAPI is required for admin dashboard")
        
        if not HAS_INTEGRATIONS:
            raise ImportError("Core integrations are required for admin dashboard")
        
        self.app = FastAPI(
            title="SloughGPT Admin Dashboard",
            description="Enterprise AI Management Interface",
            version="1.0.0"
        )
        
        # Setup dependencies
        self.user_manager = get_user_manager()
        self.cost_optimizer = get_cost_optimizer()
        self.monitoring_service = MonitoringService()
        self.security_manager = security_manager
        self.ml_pipeline = ml_pipeline
        
        # Setup middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"]
        )
        
        # Setup authentication
        self.security = HTTPBearer()
        
        # Setup templates
        self.templates = Jinja2Templates(directory="packages/apps/templates")
        self.static_files = StaticFiles(directory="packages/apps/static")
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup dashboard routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Main dashboard page."""
            return self.templates.TemplateResponse("dashboard.html", {
                "title": "SloughGPT Admin Dashboard"
                "stats": await self._get_dashboard_stats()
            })
        
        @self.app.get("/api/stats", response_model=DashboardStats)
        async def get_stats():
            """Get dashboard statistics."""
            return await self._get_dashboard_stats()
        
        @self.app.get("/api/users")
        async def get_users():
            """Get all users."""
            if not HAS_INTEGRATIONS:
                return []
            
            users = await self._get_all_users()
            return users
        
        @self.app.get("/api/users/{user_id}")
        async def get_user(user_id: int):
            """Get specific user."""
            if not HAS_INTEGRATIONS:
                return None
            
            user = await self._get_user(user_id)
            return user
        
        @self.app.get("/api/security/events")
        async def get_security_events():
            """Get recent security events."""
            if not HAS_INTEGRATIONS:
                return []
            
            events = await self._get_security_events()
            return events
        
        @self.app.get("/api/ml/models")
        async def get_ml_models():
            """Get ML models."""
            if not HAS_INTEGRATIONS:
                return []
            
            models = await self._get_ml_models()
            return models
        
        @self.app.get("/api/system/metrics")
        async def get_system_metrics():
            """Get system metrics."""
            if not HAS_INTEGRATIONS:
                return SystemMetrics(
                    timestamp=datetime.now().isoformat(),
                    cpu_percent=0.0,
                    memory_percent=0.0,
                    disk_usage={},
                    active_connections=0,
                    requests_per_minute=0.0,
                    response_time_avg=0.0,
                    cache_hit_rate=0.0,
                    error_rate=0.0
                )
            
            metrics = await self.monitoring_service.get_dashboard_data()
            
            return SystemMetrics(
                timestamp=metrics["timestamp"],
                cpu_percent=metrics.get("metrics", {}).get("cpu", {}).get("avg", 0.0),
                memory_percent=metrics.get("metrics", {}).get("memory", {}).get("avg", 0.0),
                disk_usage=metrics.get("metrics", {}).get("disk_usage", {}),
                active_connections=metrics.get("metrics", {}).get("active_connections", 0),
                requests_per_minute=metrics.get("metrics", {}).get("throughput", 0.0),
                response_time_avg=metrics.get("metrics", {}).get("response_time", 0.0),
                cache_hit_rate=metrics.get("cache_stats", {}).get("hit_rate", 0.0),
                error_rate=0.0
            )
        
        @self.app.post("/api/users/{user_id}/suspend")
        async def suspend_user(user_id: int):
            """Suspend a user."""
            if not HAS_INTEGRATIONS:
                return {"error": "Integrations not available"}
            
            # In a real implementation, this would update user status
            return {"success": True, "message": f"User {user_id} suspended"}
        
        @self.app.post("/api/users/{user_id}/activate")
        async def activate_user(user_id: int):
            """Activate a user."""
            if not HAS_INTEGRATIONS:
                return {"error": "Integrations not available"}
            
            return {"success": True, "message": f"User {user_id} activated"}
        
        @self.app.post("/api/ml/models/{model_id}/deploy")
        async def deploy_model(model_id: str):
            """Deploy an ML model."""
            if not HAS_INTEGRATIONS:
                return {"error": "Integrations not available"}
            
            success = await self.ml_pipeline.deploy_model(model_id, {
                "environment": "production",
                "api_endpoint": "/api/v1/models",
                "auto_scaling": True
            })
            
            return {"success": success, "message": f"Model {model_id} deployed"}
        
        @self.app.post("/api/system/cleanup")
        async def cleanup_system(days: int = 30):
            """Clean up old data."""
            if not HAS_INTEGRATIONS:
                return {"error": "Integrations not available"}
            
            # Cleanup logs
            log_cleanup = await log_manager.cleanup_old_logs(days)
            
            # Cleanup database
            db_cleanup = await database_manager.cleanup_old_data(days)
            
            return {
                "logs": log_cleanup,
                "database": db_cleanup,
                "total_files_deleted": log_cleanup.get("files_deleted", 0) + db_cleanup.get("files_deleted", 0),
                "space_freed": log_cleanup.get("space_freed", 0) + db_cleanup.get("space_freed", 0)
            }
        
        @self.app.get("/api/system/health")
        async def health_check():
            """System health check."""
            if not HAS_INTEGRATIONS:
                return {"status": "integrations_unavailable"}
            
            # Check all services
            db_health = await database_manager.health_check()
            
            return {
                "status": "healthy",
                "database": db_health.get("status", "unhealthy"),
                "cache": "healthy",  # Would check cache manager
                "monitoring": "healthy",
                "websocket": "healthy",  # Would check websocket manager
                "security": "healthy",  # Would check security manager
                "ml_pipeline": "healthy",
                "timestamp": datetime.now().isoformat()
            }
        
        # WebSocket endpoint for real-time updates
        @self.app.websocket("/ws/admin")
        async def admin_websocket(websocket: WebSocket):
            """WebSocket for real-time admin updates."""
            if not HAS_INTEGRATIONS:
                return
            
            connection_id = await websocket_manager.connect(websocket)
            await websocket_manager.authenticate(
                connection_id, 
                1,  # Admin user ID
                "admin",
                permissions=["system:admin", "user:read", "security:read", "ml:read"]
            )
            
            # Send initial data
            stats = await self._get_dashboard_stats()
            await websocket_manager.send_message(
                connection_id,
                websocket_manager.MessageType.SYSTEM_NOTIFICATION,
                {
                    "type": "admin_connected",
                    "stats": stats
                }
            )
            
            # Handle real-time updates
            try:
                while True:
                    message = await websocket.receive_text()
                    data = json.loads(message)
                    
                    # Handle different message types
                    if data.get("type") == "get_stats":
                        stats = await self._get_dashboard_stats()
                        await websocket_manager.send_message(
                            connection_id,
                            websocket_manager.MessageType.SYSTEM_NOTIFICATION,
                            {"type": "stats_update", "stats": stats}
                        )
                    
                    elif data.get("type") == "get_logs":
                        logs = await self._get_recent_logs(data.get("level", "info"))
                        await websocket_manager.send_message(
                            connection_id,
                            websocket_manager.MessageType.SYSTEM_NOTIFICATION,
                            {"type": "logs_update", "logs": logs}
                        )
                    
            except WebSocketDisconnect:
                await websocket_manager.disconnect(connection_id, "Client disconnected")
            except Exception as e:
                log_manager.logger.error(f"WebSocket error: {e}")
                await websocket_manager.disconnect(connection_id, f"Error: {e}")
    
    async def _get_dashboard_stats(self) -> DashboardStats:
        """Get comprehensive dashboard statistics."""
        if not HAS_INTEGRATIONS:
            return DashboardStats()
        
        # Get user statistics
        users = await self._get_all_users()
        total_users = len(users)
        active_users = sum(1 for user in users if user.get("is_active", False))
        
        # Get cost statistics
        if HAS_INTEGRATIONS:
            today = datetime.now().date()
            cost_metrics = []
            for user in users:
                user_costs = await database_manager.get_user_cost_metrics(user["id"], 1)
                cost_metrics.extend(user_costs)
            
            total_cost_today = sum(cost.get("cost", 0) for cost in cost_metrics)
        
        else:
            total_cost_today = 0.0
        
        # Get API key statistics
        if HAS_INTEGRATIONS:
            # This would query the database for API keys
            total_api_keys = total_users * 2  # Estimate
            active_api_keys = total_users * 2 - 5  # Estimate
        else:
            total_api_keys = 0
            active_api_keys = 0
        
        # Get ML model statistics
        if HAS_INTEGRATIONS:
            models = await self._get_ml_models()
            total_models_trained = len(models)
            active_training_jobs = len([m for m in models if m.get("status") == "running"])
        else:
            total_models_trained = 0
            active_training_jobs = 0
        
        # Get system health
        system_health = "healthy"
        
        # Get storage usage
        storage_usage = {"total": "1GB", "used": "500MB"}
        
        return DashboardStats(
            total_users=total_users,
            active_users=active_users,
            total_api_keys=total_api_keys,
            active_api_keys=active_api_keys,
            total_cost_today=total_cost_today,
            total_requests_today=1500,  # Estimate
            total_models_trained=total_models_trained,
            active_training_jobs=active_training_jobs,
            system_health=system_health,
            storage_usage=storage_usage
        )
    
    async def _get_all_users(self) -> List[UserInfo]:
        """Get all users with statistics."""
        if not HAS_INTEGRATIONS:
            return []
        
        users = []
        
        # Mock user data - in production, query database
        user_data = [
            {"id": 1, "username": "admin", "email": "admin@example.com", "role": "admin", "is_active": True, "created_at": "2024-01-01", "total_cost": 150.75},
            {"id": 2, "username": "user1", "email": "user1@example.com", "role": "user", "is_active": True, "created_at": "2024-01-02", "total_cost": 25.50},
            {"id": 3, "username": "user2", "email": "user2@example.com", "role": "user", "is_active": True, "created_at": "2024-01-03", "total_cost": 12.25},
            {"id": 4, "username": "user3", "email": "user3@example.com", "role": "user", "is_active": False, "created_at": "2024-01-04", "total_cost": 8.75},
            {"id": 5, "username": "moderator", "email": "mod@example.com", "role": "moderator", "is_active": True, "created_at": "2024-01-05", "total_cost": 200.00},
        ]
        
        for user_info in user_data:
            users.append(UserInfo(**user_info))
        
        return users
    
    async def _get_user(self, user_id: int) -> Optional[UserInfo]:
        """Get specific user information."""
        users = await self._get_all_users()
        for user in users:
            if user.id == user_id:
                return user
        return None
    
    async def _get_security_events(self) -> List[SecurityEvent]:
        """Get recent security events."""
        if not HAS_INTEGRATIONS:
            return []
        
        # Mock security event data
        events = [
            {
                "event_id": "sec_001",
                "timestamp": datetime.now().isoformat(),
                "event_type": "failed_login",
                "severity": "medium",
                "user_id": None,
                "ip_address": "192.168.1.100",
                "details": {"attempts": 3, "reason": "invalid_credentials"},
                "resolved": False
            },
            {
                "event_id": "sec_002",
                "timestamp": datetime.now().isoformat(),
                "event_type": "rate_limit_exceeded",
                "severity": "high",
                "user_id": 2,
                "ip_address": "192.168.1.101",
                "details": {"requests": 101, "limit": 100, "window": "1min"},
                "resolved": False
            },
            {
                "event_id": "sec_003",
                "timestamp": datetime.now().isoformat(),
                "event_type": "suspicious_activity",
                "severity": "critical",
                "user_id": None,
                "ip_address": "10.0.0.1",
                "details": {"pattern": "sql_injection_attempt", "payload": "SELECT * FROM users"},
                "resolved": True,
                "resolution": "blocked_ip"
            }
        ]
        
        return [SecurityEvent(**event) for event in events]
    
    async def _get_ml_models(self) -> List[MLModelInfo]:
        """Get ML models."""
        if not HAS_INTEGRATIONS:
            return []
        
        # Mock ML model data
        models = [
            {
                "model_id": "ml_001",
                "name": "Customer Support Classifier",
                "type": "classification",
                "status": "completed",
                "accuracy": 0.95,
                "created_at": "2024-01-10",
                "training_progress": None,
                "metrics": {"val_accuracy": 0.93, "val_precision": 0.91}
            },
            {
                "model_id": "ml_002",
                "name": "Demand Forecast",
                "type": "regression",
                "status": "running",
                "accuracy": None,
                "created_at": "2024-01-12",
                "training_progress": 0.75,
                "metrics": {}
            },
            {
                "model_id": "ml_003",
                "name": "User Clustering",
                "type": "clustering",
                "status": "completed",
                "accuracy": None,
                "created_at": "2024-01-08",
                "training_progress": None,
                "metrics": {"n_clusters": 5}
            }
        ]
        
        return [MLModelInfo(**model) for model in models]
    
    async def _get_recent_logs(self, level: str = "info") -> List[Dict[str, Any]]:
        """Get recent log entries."""
        # This would integrate with the logging system
        # For now, return mock data
        logs = [
            {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "logger": "fastapi",
                "message": f"Sample {level} log message {datetime.now()}",
                "details": {"request_id": "req_123"}
            },
            {
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "level": level,
                "logger": "security",
                "message": "Authentication successful",
                "details": {"user_id": 1, "ip": "192.168.1.1"}
            },
            {
                "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat(),
                "level": level,
                "logger": "performance",
                "message": "Request processed in 250ms",
                "details": {"endpoint": "/api/generate", "duration": 0.25}
            }
        ]
        
        return logs
    
    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the admin dashboard server."""
        if not HAS_FASTAPI:
            print("FastAPI not available. Cannot run admin dashboard.")
            return
        
        log_manager.get_logger("admin").info(f"Starting admin dashboard on {host}:{port}")
        
        import uvicorn
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )


# Global admin dashboard instance
admin_dashboard = AdminDashboard() if HAS_FASTAPI and HAS_INTEGRATIONS else None