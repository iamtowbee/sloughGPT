"""
SloughGPT Domain Architecture

This module provides the domain-based architecture for SloughGPT,
organizing functionality into logical domains:
- UI (User Interfaces)
- Cognitive (Processing, Memory, Reasoning)
- Infrastructure (Database, Cache, Config)
- Enterprise (Auth, Users, Monitoring, Cost)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

# ============================================================================
# BASE CLASSES AND INTERFACES
# ============================================================================


class BaseDomain:
    """Base class for all domains"""

    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.is_initialized = False
        self.components: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize the domain"""
        self.is_initialized = True

    async def shutdown(self) -> None:
        """Shutdown the domain"""
        self.is_initialized = False


class DomainException(Exception):
    """Exception raised by domains"""

    pass


class BaseComponent:
    """Base class for all components"""

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize the component"""
        self.is_initialized = True

    async def shutdown(self) -> None:
        """Shutdown the component"""
        self.is_initialized = False


class ComponentException(Exception):
    """Exception raised by components"""

    pass


# ============================================================================
# UI INTERFACES AND TYPES
# ============================================================================


class UIType(Enum):
    """UI types"""

    WEB = "web"
    CHAT = "chat"
    API = "api"
    CLI = "cli"


class ResponseFormat(Enum):
    """Response formats"""

    TEXT = "text"
    HTML = "html"
    JSON = "json"
    STREAM = "stream"


class UserRole(Enum):
    """User roles"""

    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class SecurityLevel(Enum):
    """Security levels"""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class User:
    """User information"""

    id: str
    username: str
    email: str
    role: UserRole
    security_level: SecurityLevel
    created_at: float
    last_active: float
    metadata: Dict[str, Any]


@dataclass
class UIRequest:
    """UI request"""

    request_id: str
    ui_type: UIType
    endpoint: str
    parameters: Dict[str, Any]
    user: Optional[User]
    timestamp: float


@dataclass
class UIResponse:
    """UI response"""

    request_id: str
    status: str
    data: Any
    timestamp: float
    format: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class IUIController(ABC):
    """Interface for UI controllers"""

    @abstractmethod
    async def handle_request(self, request: UIRequest) -> UIResponse:
        """Handle a UI request"""
        pass


class IWebInterface(ABC):
    """Interface for web interfaces"""

    @abstractmethod
    async def start_server(self) -> None:
        """Start web server"""
        pass


class IChatInterface(ABC):
    """Interface for chat interfaces"""

    @abstractmethod
    async def send_message(self, message: str, user: User) -> str:
        """Send a message"""
        pass


# ============================================================================
# COGNITIVE INTERFACES AND TYPES
# ============================================================================


class ThoughtType(Enum):
    """Types of thoughts"""

    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    METACOGNITIVE = "metacognitive"
    INTUITIVE = "intuitive"


class ReasoningStrategy(Enum):
    """Reasoning strategies"""

    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    PROBABILISTIC = "probabilistic"
    CREATIVE = "creative"
    METACOGNITIVE = "metacognitive"


@dataclass
class Thought:
    """Represents a thought"""

    thought_id: str
    content: Any
    thought_type: ThoughtType
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class Memory:
    """Represents a memory unit"""

    content: Any
    memory_type: str
    importance: float
    associations: List[str]
    retrieval_count: int
    last_accessed: float
    metadata: Dict[str, Any]


class ICognitiveProcessor(ABC):
    """Interface for cognitive processors"""

    @abstractmethod
    async def process_thought(self, thought: Thought) -> Thought:
        """Process a thought"""
        pass


class IMemoryManager(ABC):
    """Interface for memory management"""

    @abstractmethod
    async def store_memory(self, memory: Any) -> str:
        """Store a memory"""
        pass

    @abstractmethod
    async def retrieve_memory(self, memory_id: str) -> Optional[Any]:
        """Retrieve a memory"""
        pass

    @abstractmethod
    async def consolidate_memories(self) -> Dict[str, Any]:
        """Consolidate memories"""
        pass


class IReasoningEngine(ABC):
    """Interface for reasoning engines"""

    @abstractmethod
    async def reason(self, premise: str, context: Dict[str, Any]) -> str:
        """Perform reasoning"""
        pass

    @abstractmethod
    async def get_reasoning_path(self) -> List[str]:
        """Get reasoning path"""
        pass

    @abstractmethod
    async def optimize_reasoning_strategies(self) -> Dict[str, Any]:
        """Optimize reasoning strategies"""
        pass


class IMetacognitiveMonitor(ABC):
    """Interface for metacognitive monitoring"""

    @abstractmethod
    async def assess_confidence(self, thought: Thought) -> float:
        """Assess confidence"""
        pass

    @abstractmethod
    async def monitor_thought_process(self, thoughts: List[Thought]) -> Dict[str, Any]:
        """Monitor thought process"""
        pass


# ============================================================================
# INFRASTRUCTURE INTERFACES
# ============================================================================


class IDatabaseManager(ABC):
    """Interface for database management"""

    @abstractmethod
    async def connect(self, config: "DatabaseConfig") -> bool:
        """Connect to database"""
        pass

    @abstractmethod
    async def get_repository(self, collection_name: str) -> Any:
        """Get a repository"""
        pass


class ICacheManager(ABC):
    """Interface for cache management"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = 3600) -> bool:
        """Set cached value"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cached value"""
        pass


@dataclass
class DatabaseConfig:
    """Database configuration"""

    db_type: "DatabaseType"
    host: str
    port: int
    database: str
    username: Optional[str]
    password: Optional[str]
    ssl_enabled: bool = False


class DatabaseType(Enum):
    """Database types"""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    REDIS = "redis"


class IDataRepository(ABC):
    """Interface for data repositories"""

    @abstractmethod
    async def create(self, data: Dict[str, Any]) -> str:
        """Create a record"""
        pass

    @abstractmethod
    async def read(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Read a record"""
        pass

    @abstractmethod
    async def update(self, record_id: str, data: Dict[str, Any]) -> bool:
        """Update a record"""
        pass

    @abstractmethod
    async def delete(self, record_id: str) -> bool:
        """Delete a record"""
        pass


@dataclass
class CacheEntry:
    """Cache entry information"""

    key: str
    value: Any
    created_at: float
    expires_at: float
    hits: int = 0


# ============================================================================
# ENTERPRISE INTERFACES
# ============================================================================


class IAuthenticationService(ABC):
    """Interface for authentication services"""

    @abstractmethod
    async def authenticate(self, credentials: Dict[str, str]) -> Optional[Any]:
        """Authenticate a user"""
        pass


class IUserManager(ABC):
    """Interface for user management"""

    @abstractmethod
    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a user"""
        pass


class IMonitoringService(ABC):
    """Interface for monitoring services"""

    @abstractmethod
    async def record_metric(self, metric_name: str, value: float) -> None:
        """Record a metric"""
        pass

    @abstractmethod
    async def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event"""
        pass

    @abstractmethod
    async def track_metric(self, metric_name: str, value: float, tags: Dict[str, str]) -> None:
        """Track a metric"""
        pass


class ICostOptimizer(ABC):
    """Interface for cost optimization"""

    @abstractmethod
    async def calculate_cost(self, operation: str, parameters: Dict[str, Any]) -> float:
        """Calculate operation cost"""
        pass


class IDeploymentManager(ABC):
    """Interface for deployment management"""

    @abstractmethod
    async def deploy(self, config: Dict[str, Any], environment: str) -> str:
        """Deploy a service"""
        pass

    @abstractmethod
    async def scale(self, service_id: str, replicas: int) -> bool:
        """Scale a service"""
        pass


# ============================================================================
# DOMAIN EXPORTS
# ============================================================================

# Import domain modules (avoid circular imports by importing at the end)
# These will be available when imported directly:
#   from domains.ui import WebInterface
#   from domains.cognitive import MemoryManager
