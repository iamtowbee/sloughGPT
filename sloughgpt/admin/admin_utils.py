"""
Admin Dashboard Utilities and Helper Functions
"""

from fastapi import WebSocket
from typing import Dict, List, Any, Optional
import psutil
import asyncio
import json
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from ..core.logging_system import get_logger

logger = get_logger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time dashboard updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        """Accept and add a WebSocket connection"""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected WebSockets"""
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        async with self._lock:
            for conn in disconnected:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)
    
    async def get_connection_count(self) -> int:
        """Get the number of active connections"""
        async with self._lock:
            return len(self.active_connections)

async def get_system_metrics() -> Dict[str, Any]:
    """Get comprehensive system metrics"""
    try:
        metrics = {}
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        metrics["cpu_percent"] = cpu_percent
        metrics["cpu_count"] = cpu_count
        metrics["cpu_freq_current"] = cpu_freq.current if cpu_freq else None
        metrics["cpu_freq_max"] = cpu_freq.max if cpu_freq else None
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics["memory_percent"] = memory.percent
        metrics["memory_available_gb"] = memory.available / (1024**3)
        metrics["memory_used_gb"] = memory.used / (1024**3)
        metrics["memory_total_gb"] = memory.total / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics["disk_percent"] = (disk.used / disk.total) * 100
        metrics["disk_free_gb"] = disk.free / (1024**3)
        metrics["disk_used_gb"] = disk.used / (1024**3)
        metrics["disk_total_gb"] = disk.total / (1024**3)
        
        # Network metrics
        net_io = psutil.net_io_counters()
        metrics["network_bytes_sent"] = net_io.bytes_sent
        metrics["network_bytes_recv"] = net_io.bytes_recv
        metrics["network_packets_sent"] = net_io.packets_sent
        metrics["network_packets_recv"] = net_io.packets_recv
        
        # System info
        boot_time = psutil.boot_time()
        metrics["uptime_seconds"] = datetime.now().timestamp() - boot_time
        metrics["process_count"] = len(psutil.pids())
        
        # Overall status
        if cpu_percent > 90 or memory.percent > 90 or (disk.used / disk.total) * 100 > 90:
            metrics["status"] = "critical"
        elif cpu_percent > 70 or memory.percent > 70 or (disk.used / disk.total) * 100 > 80:
            metrics["status"] = "warning"
        else:
            metrics["status"] = "healthy"
        
        # Cost estimation (mock for now)
        metrics["total_cost"] = 156.78
        
        # Service status (mock for now)
        metrics["services"] = {
            "api_server": "healthy",
            "database": "healthy", 
            "cache": "healthy",
            "model_serving": "healthy"
        }
        
        metrics["database_status"] = "connected"
        metrics["cache_status"] = "connected"
        
        metrics["timestamp"] = datetime.utcnow().isoformat()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

async def get_user_stats() -> Dict[str, Any]:
    """Get user statistics"""
    try:
        # Mock user statistics - in real implementation, this would query the database
        stats = {
            "total_users": 1247,
            "active_users": 892,
            "new_users_today": 15,
            "new_users_this_week": 67,
            "new_users_this_month": 234,
            "user_roles": {
                "admin": 12,
                "user": 1235
            },
            "top_users": [
                {"email": "john.doe@example.com", "usage_count": 3421, "total_cost": 67.89},
                {"email": "jane.smith@example.com", "usage_count": 2156, "total_cost": 45.67}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

async def get_model_stats() -> Dict[str, Any]:
    """Get model statistics"""
    try:
        # Mock model statistics
        stats = {
            "total_models": 23,
            "active_models": 18,
            "training_models": 3,
            "failed_models": 2,
            "model_types": {
                "language_model": 12,
                "embedding_model": 8,
                "classification_model": 3
            },
            "model_sizes": {
                "small": 15,
                "medium": 6,
                "large": 2
            },
            "total_training_time_hours": 1567,
            "avg_accuracy": 0.87,
            "top_models": [
                {"name": "gpt2-small", "usage_count": 12456, "accuracy": 0.89},
                {"name": "bert-base", "usage_count": 8934, "accuracy": 0.91}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting model stats: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

async def export_data(data_type: str, format_type: str = "json") -> Dict[str, Any]:
    """Export data in specified format"""
    try:
        supported_types = ["users", "models", "logs", "analytics", "system"]
        supported_formats = ["json", "csv", "excel"]
        
        if data_type not in supported_types:
            return {
                "success": False,
                "error": f"Unsupported data type: {data_type}. Supported types: {supported_types}"
            }
        
        if format_type not in supported_formats:
            return {
                "success": False,
                "error": f"Unsupported format: {format_type}. Supported formats: {supported_formats}"
            }
        
        # Mock export data based on type
        export_data_map = {
            "users": await get_user_stats(),
            "models": await get_model_stats(),
            "system": await get_system_metrics(),
            "logs": {"logs": []},  # Would fetch from log system
            "analytics": {"analytics": {}}  # Would fetch from analytics system
        }
        
        data = export_data_map.get(data_type, {})
        
        # Format the data
        if format_type == "json":
            formatted_data = json.dumps(data, indent=2, default=str)
            content_type = "application/json"
        elif format_type == "csv":
            # Simple CSV conversion for basic data structures
            formatted_data = convert_to_csv(data)
            content_type = "text/csv"
        elif format_type == "excel":
            # Would use pandas/openpyxl in real implementation
            formatted_data = convert_to_csv(data)  # Fallback to CSV
            content_type = "application/vnd.ms-excel"
        
        return {
            "success": True,
            "data": formatted_data,
            "content_type": content_type,
            "filename": f"{data_type}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type}",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def cleanup_data(data_type: Optional[str] = None, older_than_days: int = 30) -> Dict[str, Any]:
    """Clean up old data"""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        
        # Mock cleanup results
        cleanup_results = {
            "logs": {
                "records_deleted": 1567,
                "space_freed_mb": 12.3
            },
            "analytics": {
                "records_deleted": 892,
                "space_freed_mb": 45.6
            },
            "temp_files": {
                "files_deleted": 234,
                "space_freed_mb": 78.9
            }
        }
        
        if data_type and data_type in cleanup_results:
            results = {data_type: cleanup_results[data_type]}
        else:
            results = cleanup_results
        
        total_deleted = sum(r.get("records_deleted", 0) for r in results.values())
        total_space_freed = sum(r.get("space_freed_mb", 0) for r in results.values())
        
        return {
            "data_type": data_type or "all",
            "cutoff_date": cutoff_date.isoformat(),
            "results": results,
            "total_records_deleted": total_deleted,
            "total_space_freed_mb": total_space_freed,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

def convert_to_csv(data: Dict[str, Any]) -> str:
    """Convert data dictionary to CSV format"""
    try:
        import io
        import csv
        
        output = io.StringIO()
        
        if isinstance(data, dict):
            # Flatten the dictionary for CSV export
            flattened = {}
            
            def flatten_dict(d, parent_key=''):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key).items())
                    elif isinstance(v, list):
                        # Convert lists to comma-separated strings
                        flattened[new_key] = ', '.join(str(x) for x in v)
                    else:
                        flattened[new_key] = str(v)
                return dict(items)
            
            flattened = flatten_dict(data)
            
            writer = csv.writer(output)
            writer.writerow(['key', 'value'])
            for key, value in flattened.items():
                writer.writerow([key, value])
        
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error converting to CSV: {e}")
        return f"Error converting to CSV: {e}"

@asynccontextmanager
async def performance_timer(operation_name: str):
    """Context manager for timing operations"""
    start_time = datetime.utcnow()
    logger.info(f"Starting operation: {operation_name}")
    
    try:
        yield
    finally:
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Completed operation: {operation_name} in {duration:.2f} seconds")

class MetricsCollector:
    """Collects and aggregates metrics for the admin dashboard"""
    
    def __init__(self):
        self.metrics_cache = {}
        self.cache_ttl = 60  # seconds
    
    async def get_cached_metrics(self, metric_type: str) -> Dict[str, Any]:
        """Get cached metrics or fetch new ones if cache is expired"""
        now = datetime.utcnow()
        
        # Check cache
        if metric_type in self.metrics_cache:
            cached_data, timestamp = self.metrics_cache[metric_type]
            if (now - timestamp).total_seconds() < self.cache_ttl:
                return cached_data
        
        # Fetch fresh data
        if metric_type == "system":
            fresh_data = await get_system_metrics()
        elif metric_type == "users":
            fresh_data = await get_user_stats()
        elif metric_type == "models":
            fresh_data = await get_model_stats()
        else:
            fresh_data = {"error": f"Unknown metric type: {metric_type}"}
        
        # Update cache
        self.metrics_cache[metric_type] = (fresh_data, now)
        
        return fresh_data
    
    def clear_cache(self):
        """Clear the metrics cache"""
        self.metrics_cache.clear()
        logger.info("Metrics cache cleared")

# Global metrics collector instance
metrics_collector = MetricsCollector()

# Background task for periodic metrics collection
async def start_metrics_collection_task():
    """Start background task to collect metrics periodically"""
    while True:
        try:
            # Collect and cache all metrics
            await metrics_collector.get_cached_metrics("system")
            await metrics_collector.get_cached_metrics("users") 
            await metrics_collector.get_cached_metrics("models")
            
            logger.debug("Metrics collection completed")
            
        except Exception as e:
            logger.error(f"Error in metrics collection: {e}")
        
        # Wait for next collection
        await asyncio.sleep(60)  # Collect every minute