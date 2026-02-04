"""WebSocket support for real-time features in SloughGPT."""

from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import json
import uuid
import logging
from enum import Enum

try:
    from fastapi import WebSocket, WebSocketDisconnect, WebSocketException
    from fastapi.websockets import WebSocketState
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class MessageType(Enum):
    """WebSocket message types."""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    AUTHENTICATE = "authenticate"
    TEXT_GENERATION = "text_generation"
    TRAINING_UPDATE = "training_update"
    COST_UPDATE = "cost_update"
    SYSTEM_NOTIFICATION = "system_notification"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[int] = None
    room_id: Optional[str] = None


@dataclass
class WebSocketConnection:
    """WebSocket connection information."""
    websocket: WebSocket
    user_id: Optional[int] = None
    username: Optional[str] = None
    is_authenticated: bool = False
    permissions: List[str] = field(default_factory=list)
    room_ids: Set[str] = field(default_factory=set)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    connection_time: datetime = field(default_factory=datetime.now)
    ip_address: Optional[str] = None


@dataclass
class Room:
    """WebSocket room for grouping connections."""
    room_id: str
    name: str
    description: Optional[str] = None
    is_private: bool = False
    allowed_users: Set[int] = field(default_factory=set)
    connections: Set[str] = field(default_factory=set)  # connection_ids
    created_at: datetime = field(default_factory=datetime.now)
    max_connections: int = 100


class WebSocketManager:
    """WebSocket connection manager for real-time features."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[int, Set[str]] = {}  # user_id -> connection_ids
        self.rooms: Dict[str, Room] = {}
        self.room_connections: Dict[str, Set[str]] = {}  # room_id -> connection_ids
        self.redis_publisher = None
        self.redis_subscriber = None
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.initialized = False
        
    async def initialize(self, redis_url: Optional[str] = None) -> bool:
        """Initialize WebSocket manager."""
        try:
            # Initialize Redis for pub/sub if available
            if HAS_REDIS and redis_url:
                self.redis_publisher = await aioredis.from_url(redis_url)
                self.redis_subscriber = await aioredis.from_url(redis_url)
                
                # Subscribe to system channels
                await self.redis_subscriber.subscribe("sloughgpt:system")
                await self.redis_subscriber.subscribe("sloughgpt:notifications")
            
            self.initialized = True
            logging.info("WebSocket manager initialized")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize WebSocket manager: {e}")
            return False
    
    async def close(self):
        """Close WebSocket manager and cleanup."""
        try:
            # Close all active connections
            for connection_id in list(self.active_connections.keys()):
                await self.disconnect(connection_id, "Server shutdown")
            
            # Close Redis connections
            if self.redis_publisher:
                await self.redis_publisher.close()
            if self.redis_subscriber:
                await self.redis_subscriber.close()
            
            self.initialized = False
            logging.info("WebSocket manager closed")
            
        except Exception as e:
            logging.error(f"Error closing WebSocket manager: {e}")
    
    async def connect(self, websocket: WebSocket, connection_id: str = None) -> str:
        """Accept new WebSocket connection."""
        if not self.initialized:
            raise RuntimeError("WebSocket manager not initialized")
        
        if connection_id is None:
            connection_id = str(uuid.uuid4())
        
        # Get client IP
        client_host = websocket.client.host if websocket.client else "unknown"
        
        # Create connection object
        connection = WebSocketConnection(
            websocket=websocket,
            ip_address=client_host
        )
        
        self.active_connections[connection_id] = connection
        logging.info(f"WebSocket connected: {connection_id} from {client_host}")
        
        return connection_id
    
    async def disconnect(self, connection_id: str, reason: str = "Disconnected"):
        """Disconnect WebSocket connection."""
        if connection_id not in self.active_connections:
            return
        
        connection = self.active_connections[connection_id]
        
        # Remove from user connections
        if connection.user_id:
            if connection.user_id in self.user_connections:
                self.user_connections[connection.user_id].discard(connection_id)
                if not self.user_connections[connection.user_id]:
                    del self.user_connections[connection.user_id]
        
        # Remove from room connections
        for room_id in connection.room_ids:
            if room_id in self.room_connections:
                self.room_connections[room_id].discard(connection_id)
                if connection.room_id in self.rooms:
                    self.rooms[room_id].connections.discard(connection_id)
        
        # Close WebSocket
        try:
            if connection.websocket.client_state != WebSocketState.DISCONNECTED:
                await connection.websocket.close(code=1000, reason=reason)
        except Exception as e:
            logging.warning(f"Error closing WebSocket {connection_id}: {e}")
        
        # Remove from active connections
        del self.active_connections[connection_id]
        logging.info(f"WebSocket disconnected: {connection_id} - {reason}")
    
    async def authenticate(self, connection_id: str, user_id: int, username: str,
                         permissions: List[str] = None) -> bool:
        """Authenticate WebSocket connection."""
        if connection_id not in self.active_connections:
            return False
        
        connection = self.active_connections[connection_id]
        connection.user_id = user_id
        connection.username = username
        connection.is_authenticated = True
        connection.permissions = permissions or []
        
        # Add to user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        # Send authentication success message
        await self.send_message(connection_id, MessageType.AUTHENTICATE, {
            "success": True,
            "user_id": user_id,
            "username": username,
            "permissions": permissions
        })
        
        logging.info(f"WebSocket authenticated: {connection_id} - user {user_id}")
        return True
    
    async def join_room(self, connection_id: str, room_id: str) -> bool:
        """Join a room."""
        if connection_id not in self.active_connections:
            return False
        
        connection = self.active_connections[connection_id]
        
        # Check if room exists and user has permission
        if room_id not in self.rooms:
            return False
        
        room = self.rooms[room_id]
        
        # Check private room permissions
        if room.is_private and connection.user_id not in room.allowed_users:
            return False
        
        # Check room capacity
        if len(self.room_connections.get(room_id, set())) >= room.max_connections:
            return False
        
        # Add to room
        connection.room_ids.add(room_id)
        if room_id not in self.room_connections:
            self.room_connections[room_id] = set()
        self.room_connections[room_id].add(connection_id)
        room.connections.add(connection_id)
        
        # Notify room members
        await self.broadcast_to_room(room_id, MessageType.SYSTEM_NOTIFICATION, {
            "type": "user_joined",
            "user_id": connection.user_id,
            "username": connection.username,
            "total_users": len(self.room_connections[room_id])
        }, exclude_connection_id=connection_id)
        
        # Send room info to user
        await self.send_message(connection_id, MessageType.SYSTEM_NOTIFICATION, {
            "type": "room_joined",
            "room_id": room_id,
            "room_name": room.name,
            "total_users": len(self.room_connections[room_id])
        })
        
        logging.info(f"WebSocket {connection_id} joined room {room_id}")
        return True
    
    async def leave_room(self, connection_id: str, room_id: str) -> bool:
        """Leave a room."""
        if connection_id not in self.active_connections:
            return False
        
        connection = self.active_connections[connection_id]
        
        if room_id not in connection.room_ids:
            return False
        
        # Remove from room
        connection.room_ids.discard(room_id)
        if room_id in self.room_connections:
            self.room_connections[room_id].discard(connection_id)
            if room_id in self.rooms:
                self.rooms[room_id].connections.discard(connection_id)
        
        # Notify room members
        await self.broadcast_to_room(room_id, MessageType.SYSTEM_NOTIFICATION, {
            "type": "user_left",
            "user_id": connection.user_id,
            "username": connection.username,
            "total_users": len(self.room_connections.get(room_id, set()))
        }, exclude_connection_id=connection_id)
        
        logging.info(f"WebSocket {connection_id} left room {room_id}")
        return True
    
    async def create_room(self, room_id: str, name: str, description: str = None,
                        is_private: bool = False, max_connections: int = 100,
                        created_by: Optional[int] = None) -> bool:
        """Create a new room."""
        if room_id in self.rooms:
            return False
        
        room = Room(
            room_id=room_id,
            name=name,
            description=description,
            is_private=is_private,
            max_connections=max_connections
        )
        
        # Add creator to allowed users for private rooms
        if is_private and created_by:
            room.allowed_users.add(created_by)
        
        self.rooms[room_id] = room
        self.room_connections[room_id] = set()
        
        logging.info(f"Created room: {room_id} - {name}")
        return True
    
    async def send_message(self, connection_id: str, message_type: MessageType,
                        data: Dict[str, Any], room_id: Optional[str] = None) -> bool:
        """Send message to specific connection."""
        if connection_id not in self.active_connections:
            return False
        
        connection = self.active_connections[connection_id]
        
        # Check permissions if required
        if self._requires_permission(message_type):
            if not self._has_permission(connection, message_type):
                return False
        
        message = WebSocketMessage(
            type=message_type.value,
            data=data,
            user_id=connection.user_id,
            room_id=room_id
        )
        
        try:
            await connection.websocket.send_text(json.dumps({
                "type": message.type,
                "message_id": message.message_id,
                "timestamp": message.timestamp.isoformat(),
                "user_id": message.user_id,
                "room_id": message.room_id,
                "data": message.data
            }))
            return True
            
        except Exception as e:
            logging.error(f"Error sending message to {connection_id}: {e}")
            return False
    
    async def broadcast_to_user(self, user_id: int, message_type: MessageType,
                           data: Dict[str, Any]) -> int:
        """Broadcast message to all connections for a user."""
        if user_id not in self.user_connections:
            return 0
        
        connection_ids = self.user_connections[user_id]
        sent_count = 0
        
        for connection_id in connection_ids:
            if await self.send_message(connection_id, message_type, data):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_to_room(self, room_id: str, message_type: MessageType,
                           data: Dict[str, Any], exclude_connection_id: Optional[str] = None) -> int:
        """Broadcast message to all connections in a room."""
        if room_id not in self.room_connections:
            return 0
        
        connection_ids = self.room_connections[room_id]
        sent_count = 0
        
        for connection_id in connection_ids:
            if connection_id != exclude_connection_id:
                if await self.send_message(connection_id, message_type, data, room_id):
                    sent_count += 1
        
        return sent_count
    
    async def broadcast_to_all(self, message_type: MessageType, data: Dict[str, Any],
                            require_permission: Optional[str] = None) -> int:
        """Broadcast message to all authenticated connections."""
        sent_count = 0
        
        for connection_id, connection in self.active_connections.items():
            if not connection.is_authenticated:
                continue
            
            # Check permission if required
            if require_permission and require_permission not in connection.permissions:
                continue
            
            if await self.send_message(connection_id, message_type, data):
                sent_count += 1
        
        return sent_count
    
    async def send_heartbeat(self, connection_id: str) -> bool:
        """Send heartbeat to connection."""
        return await self.send_message(connection_id, MessageType.HEARTBEAT, {
            "timestamp": datetime.now().isoformat()
        })
    
    async def handle_training_update(self, job_id: str, status: str, progress: float = None,
                               metrics: Dict[str, Any] = None) -> int:
        """Handle training job updates."""
        data = {
            "job_id": job_id,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        if progress is not None:
            data["progress"] = progress
        if metrics:
            data["metrics"] = metrics
        
        # Broadcast to all connections of the user who created the job
        sent_count = 0
        
        for connection_id, connection in self.active_connections.items():
            if not connection.is_authenticated:
                continue
            if "training:read" in connection.permissions:
                sent_count += await self.send_message(connection_id, MessageType.TRAINING_UPDATE, data)
        
        return sent_count
    
    async def handle_cost_update(self, user_id: int, metric_type: str, cost: float,
                              budget_remaining: Optional[float] = None) -> int:
        """Handle cost updates."""
        data = {
            "user_id": user_id,
            "metric_type": metric_type,
            "cost": cost,
            "timestamp": datetime.now().isoformat()
        }
        
        if budget_remaining is not None:
            data["budget_remaining"] = budget_remaining
        
        return await self.broadcast_to_user(user_id, MessageType.COST_UPDATE, data)
    
    def _requires_permission(self, message_type: MessageType) -> bool:
        """Check if message type requires special permission."""
        permission_required = {
            MessageType.TEXT_GENERATION: "text:generate",
            MessageType.TRAINING_UPDATE: "training:read",
            MessageType.SYSTEM_NOTIFICATION: "system:admin",
        }
        return message_type in permission_required
    
    def _has_permission(self, connection: WebSocketConnection, message_type: MessageType) -> bool:
        """Check if connection has required permission."""
        if not self._requires_permission(message_type):
            return True
        
        permission_required = {
            MessageType.TEXT_GENERATION: "text:generate",
            MessageType.TRAINING_UPDATE: "training:read",
            MessageType.SYSTEM_NOTIFICATION: "system:admin",
        }
        
        required_permission = permission_required.get(message_type)
        return required_permission in connection.permissions
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register custom message handler."""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    async def handle_message(self, connection_id: str, message_data: Dict[str, Any]) -> bool:
        """Handle incoming message from connection."""
        if connection_id not in self.active_connections:
            return False
        
        try:
            message_type_str = message_data.get("type", "")
            message_type = MessageType(message_type_str)
            
            # Call registered handlers
            if message_type in self.message_handlers:
                connection = self.active_connections[connection_id]
                for handler in self.message_handlers[message_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(connection_id, connection, message_data)
                        else:
                            handler(connection_id, connection, message_data)
                    except Exception as e:
                        logging.error(f"Message handler error: {e}")
            
            # Handle built-in message types
            if message_type == MessageType.HEARTBEAT:
                connection = self.active_connections[connection_id]
                connection.last_heartbeat = datetime.now()
                await self.send_heartbeat(connection_id)
            
            return True
            
        except Exception as e:
            logging.error(f"Error handling message from {connection_id}: {e}")
            return False
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        stats = {
            "total_connections": len(self.active_connections),
            "authenticated_connections": sum(1 for c in self.active_connections.values() if c.is_authenticated),
            "total_rooms": len(self.rooms),
            "users_online": len(self.user_connections),
            "connections_by_room": {room_id: len(connections) for room_id, connections in self.room_connections.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        # Add user details
        stats["users"] = [
            {
                "user_id": user_id,
                "connection_count": len(connections),
                "connections": list(connections)
            }
            for user_id, connections in self.user_connections.items()
        ]
        
        return stats
    
    async def cleanup_inactive_connections(self, inactive_minutes: int = 30) -> int:
        """Clean up inactive connections."""
        cutoff_time = datetime.now() - timedelta(minutes=inactive_minutes)
        cleaned_up = 0
        
        for connection_id, connection in list(self.active_connections.items()):
            if connection.last_heartbeat < cutoff_time:
                await self.disconnect(connection_id, "Inactive connection")
                cleaned_up += 1
        
        return cleaned_up


# Global WebSocket manager
websocket_manager = WebSocketManager()