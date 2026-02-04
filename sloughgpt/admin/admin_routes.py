"""
Admin Dashboard Routes
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio

from .admin_utils import (
    WebSocketManager, get_system_metrics, get_user_stats, 
    get_model_stats, export_data, cleanup_data
)
from ..core.logging_system import get_logger

logger = get_logger(__name__)

# Create router
admin_router = APIRouter(prefix="/admin", tags=["admin"])

# WebSocket manager for real-time updates
ws_manager = WebSocketManager()

@admin_router.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get comprehensive dashboard statistics"""
    try:
        # Get system metrics
        system_metrics = await get_system_metrics()
        user_stats = await get_user_stats()
        model_stats = await get_model_stats()
        
        # Calculate recent activity (mock data for now)
        recent_activity = [
            {
                "id": 1,
                "type": "user",
                "title": "New user registered",
                "description": "User john.doe@example.com registered",
                "time": "2 minutes ago"
            },
            {
                "id": 2,
                "type": "model",
                "title": "Model training completed",
                "description": "GPT-2-small training completed successfully",
                "time": "1 hour ago"
            },
            {
                "id": 3,
                "type": "system",
                "title": "System backup completed",
                "description": "Automatic system backup completed successfully",
                "time": "3 hours ago"
            }
        ]
        
        # Chart data
        chart_data = {
            "usage": {
                "labels": ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"],
                "data": [45, 52, 38, 65, 72, 58]
            },
            "cost": {
                "data": [1200, 800, 400, 200]
            }
        }
        
        return {
            "system_status": system_metrics.get("status", "Unknown"),
            "user_count": user_stats.get("total_users", 0),
            "model_count": model_stats.get("total_models", 0),
            "total_cost": system_metrics.get("total_cost", 0.0),
            "recent_activity": recent_activity,
            "chart_data": chart_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dashboard statistics")

@admin_router.get("/system/health")
async def get_system_health():
    """Get detailed system health information"""
    try:
        system_metrics = await get_system_metrics()
        
        return {
            "status": system_metrics.get("status", "unknown"),
            "uptime": system_metrics.get("uptime_seconds", 0),
            "cpu_usage": system_metrics.get("cpu_percent", 0),
            "memory_usage": system_metrics.get("memory_percent", 0),
            "disk_usage": system_metrics.get("disk_percent", 0),
            "database_status": system_metrics.get("database_status", "unknown"),
            "cache_status": system_metrics.get("cache_status", "unknown"),
            "services": system_metrics.get("services", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system health")

@admin_router.get("/users")
async def get_users(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    status: Optional[str] = None
):
    """Get list of users with filtering and pagination"""
    try:
        user_stats = await get_user_stats()
        
        # Mock user data - in real implementation, this would query the database
        users = [
            {
                "id": 1,
                "email": "john.doe@example.com",
                "name": "John Doe",
                "role": "user",
                "status": "active",
                "created_at": "2024-01-15T10:30:00Z",
                "last_login": "2024-02-04T09:15:00Z",
                "api_usage": 1247,
                "total_cost": 23.45
            },
            {
                "id": 2,
                "email": "jane.smith@example.com",
                "name": "Jane Smith", 
                "role": "admin",
                "status": "active",
                "created_at": "2024-01-20T14:22:00Z",
                "last_login": "2024-02-03T16:45:00Z",
                "api_usage": 892,
                "total_cost": 15.67
            }
        ]
        
        # Apply filters
        if search:
            users = [u for u in users if search.lower() in u["email"].lower() or search.lower() in u["name"].lower()]
        
        if status:
            users = [u for u in users if u["status"] == status]
        
        # Apply pagination
        total = len(users)
        users = users[skip:skip + limit]
        
        return {
            "users": users,
            "total": total,
            "skip": skip,
            "limit": limit,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        raise HTTPException(status_code=500, detail="Failed to get users")

@admin_router.get("/models")
async def get_models(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None
):
    """Get list of models with filtering and pagination"""
    try:
        model_stats = await get_model_stats()
        
        # Mock model data
        models = [
            {
                "id": 1,
                "name": "gpt2-small",
                "type": "language_model",
                "status": "active",
                "created_at": "2024-01-10T08:00:00Z",
                "size": "117M",
                "accuracy": 0.89,
                "usage_count": 3421,
                "total_cost": 67.89
            },
            {
                "id": 2,
                "name": "bert-base",
                "type": "embedding_model",
                "status": "training",
                "created_at": "2024-01-25T12:30:00Z",
                "size": "110M",
                "accuracy": 0.0,
                "usage_count": 0,
                "total_cost": 12.34
            }
        ]
        
        # Apply filters
        if status:
            models = [m for m in models if m["status"] == status]
        
        # Apply pagination
        total = len(models)
        models = models[skip:skip + limit]
        
        return {
            "models": models,
            "total": total,
            "skip": skip,
            "limit": limit,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail="Failed to get models")

@admin_router.get("/analytics")
async def get_analytics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    metric: str = "usage"
):
    """Get analytics data for specified time period"""
    try:
        # Parse dates
        if start_date:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        else:
            start_dt = datetime.utcnow() - timedelta(days=30)
            
        if end_date:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        else:
            end_dt = datetime.utcnow()
        
        # Mock analytics data based on metric type
        analytics_data = {
            "usage": {
                "total_requests": 15420,
                "unique_users": 342,
                "avg_response_time": 1.23,
                "success_rate": 0.987,
                "daily_breakdown": [
                    {"date": "2024-02-01", "requests": 1245},
                    {"date": "2024-02-02", "requests": 1567},
                    {"date": "2024-02-03", "requests": 1432},
                ]
            },
            "cost": {
                "total_cost": 234.56,
                "daily_breakdown": [
                    {"date": "2024-02-01", "cost": 45.67},
                    {"date": "2024-02-02", "cost": 52.34},
                    {"date": "2024-02-03", "cost": 48.90},
                ],
                "cost_by_service": {
                    "inference": 156.78,
                    "training": 67.89,
                    "storage": 9.89
                }
            },
            "performance": {
                "avg_latency": 1.23,
                "p95_latency": 2.45,
                "p99_latency": 3.67,
                "throughput": 1245.6,
                "error_rate": 0.013
            }
        }
        
        data = analytics_data.get(metric, {})
        
        return {
            "metric": metric,
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

@admin_router.get("/export/{data_type}")
async def export_data_endpoint(data_type: str):
    """Export data in various formats"""
    try:
        supported_formats = ["json", "csv", "excel"]
        format_type = "json"  # Default format
        
        # Export data using utility function
        export_result = await export_data(data_type, format_type)
        
        if not export_result["success"]:
            raise HTTPException(
                status_code=400, 
                detail=export_result.get("error", "Export failed")
            )
        
        return JSONResponse(
            content=export_result,
            headers={
                "Content-Disposition": f"attachment; filename={data_type}_export.{format_type}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail="Failed to export data")

@admin_router.post("/cleanup")
async def cleanup_data_endpoint(
    data_type: Optional[str] = None,
    older_than_days: int = 30
):
    """Clean up old data"""
    try:
        cleanup_result = await cleanup_data(data_type, older_than_days)
        
        return {
            "success": True,
            "message": "Cleanup completed successfully",
            "details": cleanup_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup data")

@admin_router.get("/logs")
async def get_logs(
    level: Optional[str] = None,
    limit: int = 100,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get system logs with filtering"""
    try:
        # Mock log data
        logs = [
            {
                "id": 1,
                "level": "INFO",
                "message": "API server started successfully",
                "timestamp": "2024-02-04T10:30:00Z",
                "module": "api_server"
            },
            {
                "id": 2,
                "level": "WARNING",
                "message": "High memory usage detected",
                "timestamp": "2024-02-04T10:25:00Z",
                "module": "performance"
            },
            {
                "id": 3,
                "level": "ERROR",
                "message": "Database connection failed",
                "timestamp": "2024-02-04T10:20:00Z",
                "module": "database"
            }
        ]
        
        # Apply filters
        if level:
            logs = [log for log in logs if log["level"] == level.upper()]
        
        # Limit results
        logs = logs[:limit]
        
        return {
            "logs": logs,
            "total": len(logs),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get logs")

@admin_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            # Process message (echo for now)
            await websocket.send_text(f"Received: {data}")
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)

# Broadcast function to send updates to all connected clients
async def broadcast_update(message: Dict[str, Any]):
    """Broadcast message to all connected WebSocket clients"""
    await ws_manager.broadcast(message)