"""Database integration and ORM layer for SloughGPT."""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager
import logging

try:
    from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, JSON, ForeignKey, Index
    from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
    from sqlalchemy.orm import relationship, sessionmaker, Session
    from sqlalchemy.pool import StaticPool
    from sqlalchemy.sql import func
    HAS_SQLALCHEMY = True
    Column = Column
    Integer = Integer
    String = String
    DateTime = DateTime
    Boolean = Boolean
    Float = Float
    Text = Text
    JSON = JSON
    ForeignKey = ForeignKey
    Index = Index
    declarative_base = declarative_base
    func = func
    relationship = relationship
    async_sessionmaker = async_sessionmaker
    AsyncSession = AsyncSession
    create_async_engine = create_async_engine
    StaticPool = StaticPool
except ImportError:
    HAS_SQLALCHEMY = False
    Column = None
    Integer = None
    String = None
    DateTime = None
    Boolean = None
    Float = None
    Text = None
    JSON = None
    ForeignKey = None
    Index = None
    declarative_base = None
    func = None
    relationship = None
    async_sessionmaker = None
    AsyncSession = None
    create_async_engine = None
    StaticPool = None

try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False

if HAS_SQLALCHEMY:
    Base = declarative_base()


# Database models
if HAS_SQLALCHEMY:
    @dataclass
    class User(Base):
        __tablename__ = "users"
        
        id = Column(Integer, primary_key=True, index=True)
        username = Column(String(50), unique=True, nullable=False, index=True)
        email = Column(String(255), unique=True, nullable=False, index=True)
        password_hash = Column(String(255), nullable=False)
        role = Column(String(20), nullable=False, default="user")
        is_active = Column(Boolean, default=True, nullable=False)
        created_at = Column(DateTime, default=func.now(), nullable=False)
        updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
        last_login = Column(DateTime, nullable=True)
        
        # Relationships
        api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
        cost_metrics = relationship("CostMetric", back_populates="user")
        training_jobs = relationship("TrainingJob", back_populates="user")
        
        __table_args__ = (
            Index('idx_user_username', 'username'),
            Index('idx_user_email', 'email'),
            Index('idx_user_role', 'role'),
        )
    
    @dataclass
    class APIKey(Base):
        __tablename__ = "api_keys"
        
        id = Column(Integer, primary_key=True, index=True)
        user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
        name = Column(String(100), nullable=False)
        key_hash = Column(String(255), nullable=False, unique=True)
        permissions = Column(JSON, nullable=False)
        rate_limit = Column(Integer, default=1000, nullable=False)
        is_active = Column(Boolean, default=True, nullable=False)
        last_used = Column(DateTime, nullable=True)
        created_at = Column(DateTime, default=func.now(), nullable=False)
        expires_at = Column(DateTime, nullable=True)
        
        # Relationships
        user = relationship("User", back_populates="api_keys")
        
        __table_args__ = (
            Index('idx_api_key_user', 'user_id'),
            Index('idx_api_key_active', 'is_active'),
        )
    
    @dataclass
    class CostMetric(Base):
        __tablename__ = "cost_metrics"
        
        id = Column(Integer, primary_key=True, index=True)
        user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
        metric_type = Column(String(50), nullable=False)
        amount = Column(Float, nullable=False)
        unit = Column(String(20), nullable=False)
        cost = Column(Float, nullable=False)
        model_name = Column(String(100), nullable=True)
        metadata_data = Column(JSON, nullable=True)
        timestamp = Column(DateTime, default=func.now(), nullable=False)
        
        # Relationships
        user = relationship("User", back_populates="cost_metrics")
        
        __table_args__ = (
            Index('idx_cost_user', 'user_id'),
            Index('idx_cost_type', 'metric_type'),
            Index('idx_cost_timestamp', 'timestamp'),
        )
    
    @dataclass
    class TrainingJob(Base):
        __tablename__ = "training_jobs"
        
        id = Column(Integer, primary_key=True, index=True)
        job_id = Column(String(100), unique=True, nullable=False, index=True)
        user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
        model_name = Column(String(100), nullable=False)
        dataset_path = Column(String(500), nullable=False)
        output_dir = Column(String(500), nullable=False)
        config = Column(JSON, nullable=False)
        status = Column(String(20), nullable=False, default="pending")
        created_at = Column(DateTime, default=func.now(), nullable=False)
        started_at = Column(DateTime, nullable=True)
        completed_at = Column(DateTime, nullable=True)
        metrics = Column(JSON, nullable=True)
        checkpoints = Column(JSON, nullable=True)
        final_model_path = Column(String(500), nullable=True)
        
        # Relationships
        user = relationship("User", back_populates="training_jobs")
        
        __table_args__ = (
            Index('idx_training_user', 'user_id'),
            Index('idx_training_status', 'status'),
            Index('idx_training_created', 'created_at'),
        )
    
    @dataclass
    class DatasetSource(Base):
        __tablename__ = "dataset_sources"
        
        id = Column(Integer, primary_key=True, index=True)
        source_id = Column(String(100), unique=True, nullable=False, index=True)
        name = Column(String(100), nullable=False)
        type = Column(String(20), nullable=False)
        path = Column(String(500), nullable=False)
        format = Column(String(20), nullable=False)
        config = Column(JSON, nullable=True)
        status = Column(String(20), nullable=False, default="pending")
        last_processed = Column(DateTime, nullable=True)
        created_at = Column(DateTime, default=func.now(), nullable=False)
        created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
        
        __table_args__ = (
            Index('idx_dataset_source', 'source_id'),
            Index('idx_dataset_status', 'status'),
        )
    
    @dataclass
    class KnowledgeItem(Base):
        __tablename__ = "knowledge_items"
        
        id = Column(Integer, primary_key=True, index=True)
        item_id = Column(String(100), unique=True, nullable=False, index=True)
        source_id = Column(String(100), nullable=False)
        text = Column(Text, nullable=False)
        embedding = Column(JSON, nullable=True)
        metadata = Column(JSON, nullable=False)
        quality_score = Column(Float, nullable=False)
        created_at = Column(DateTime, default=func.now(), nullable=False)
        
        __table_args__ = (
            Index('idx_knowledge_item', 'item_id'),
            Index('idx_knowledge_source', 'source_id'),
            Index('idx_knowledge_quality', 'quality_score'),
        )
    
    @dataclass
    class AuditLog(Base):
        __tablename__ = "audit_logs"
        
        id = Column(Integer, primary_key=True, index=True)
        user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
        action = Column(String(100), nullable=False)
        resource = Column(String(100), nullable=False)
        details = Column(JSON, nullable=True)
        ip_address = Column(String(45), nullable=True)
        user_agent = Column(String(500), nullable=True)
        timestamp = Column(DateTime, default=func.now(), nullable=False)
        
        __table_args__ = (
            Index('idx_audit_user', 'user_id'),
            Index('idx_audit_action', 'action'),
            Index('idx_audit_timestamp', 'timestamp'),
        )


class DatabaseManager:
    """Database manager with async support."""
    
    def __init__(self, database_url: str = "sqlite+aiosqlite:///./sloughgpt.db"):
        if not HAS_SQLALCHEMY:
            raise ImportError("SQLAlchemy is required for database operations")
        
        self.database_url = database_url
        self.engine = None
        self.session_factory = None
        
    async def initialize(self):
        """Initialize database connection and create tables."""
        try:
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 20
                } if "sqlite" in self.database_url else {}
            )
            
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create all tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logging.info(f"Database initialized: {self.database_url}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize database: {e}")
            return False
    
    async def close(self):
        """Close database connection."""
        if self.engine:
            await self.engine.dispose()
            logging.info("Database connection closed")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic cleanup."""
        if not self.session_factory:
            raise RuntimeError("Database not initialized")
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logging.error(f"Database session error: {e}")
                raise
    
    # User operations
    async def create_user(self, username: str, email: str, password_hash: str, 
                        role: str = "user") -> Optional[int]:
        """Create a new user."""
        async with self.get_session() as session:
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                role=role
            )
            session.add(user)
            await session.flush()
            return user.id
    
    async def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        async with self.get_session() as session:
            result = await session.execute(
                f"SELECT * FROM users WHERE id = {user_id}"
            )
            row = result.fetchone()
            return dict(row) if row else None
    
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        async with self.get_session() as session:
            result = await session.execute(
                f"SELECT * FROM users WHERE username = '{username}'"
            )
            row = result.fetchone()
            return dict(row) if row else None
    
    async def update_user_last_login(self, user_id: int) -> bool:
        """Update user's last login time."""
        async with self.get_session() as session:
            result = await session.execute(
                f"UPDATE users SET last_login = '{datetime.now()}' WHERE id = {user_id}"
            )
            return result.rowcount > 0
    
    # API Key operations
    async def create_api_key(self, user_id: int, name: str, key_hash: str,
                           permissions: List[str], rate_limit: int = 1000,
                           expires_days: int = 365) -> Optional[int]:
        """Create an API key."""
        async with self.get_session() as session:
            api_key = APIKey(
                user_id=user_id,
                name=name,
                key_hash=key_hash,
                permissions=permissions,
                rate_limit=rate_limit,
                expires_at=datetime.now() + timedelta(days=expires_days)
            )
            session.add(api_key)
            await session.flush()
            return api_key.id
    
    async def get_api_key(self, key_hash: str) -> Optional[Dict[str, Any]]:
        """Get API key by hash."""
        async with self.get_session() as session:
            result = await session.execute(
                f"SELECT * FROM api_keys WHERE key_hash = '{key_hash}' AND is_active = 1"
            )
            row = result.fetchone()
            return dict(row) if row else None
    
    # Cost metrics operations
    async def save_cost_metric(self, user_id: int, metric_type: str, amount: float,
                           unit: str, cost: float, model_name: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save a cost metric."""
        async with self.get_session() as session:
            metric = CostMetric(
                user_id=user_id,
                metric_type=metric_type,
                amount=amount,
                unit=unit,
                cost=cost,
                model_name=model_name,
                metadata=metadata
            )
            session.add(metric)
            await session.flush()
            return True
    
    async def get_user_cost_metrics(self, user_id: int, days: int = 30) -> List[Dict[str, Any]]:
        """Get cost metrics for a user."""
        cutoff_date = datetime.now() - timedelta(days=days)
        async with self.get_session() as session:
            result = await session.execute(
                f"SELECT * FROM cost_metrics WHERE user_id = {user_id} "
                f"AND timestamp >= '{cutoff_date}' ORDER BY timestamp DESC"
            )
            return [dict(row) for row in result.fetchall()]
    
    # Training job operations
    async def create_training_job(self, job_id: str, user_id: int, model_name: str,
                              dataset_path: str, output_dir: str, config: Dict[str, Any]) -> bool:
        """Create a training job."""
        async with self.get_session() as session:
            job = TrainingJob(
                job_id=job_id,
                user_id=user_id,
                model_name=model_name,
                dataset_path=dataset_path,
                output_dir=output_dir,
                config=config
            )
            session.add(job)
            await session.flush()
            return True
    
    async def update_training_job_status(self, job_id: str, status: str,
                                   metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Update training job status."""
        async with self.get_session() as session:
            update_fields = f"status = '{status}'"
            if status in ["running", "completed", "failed"]:
                update_fields += f", started_at = '{datetime.now()}'"
            if status in ["completed", "failed"]:
                update_fields += f", completed_at = '{datetime.now()}'"
            if metrics:
                update_fields += f", metrics = '{json.dumps(metrics)}'"
            
            result = await session.execute(
                f"UPDATE training_jobs SET {update_fields} WHERE job_id = '{job_id}'"
            )
            return result.rowcount > 0
    
    async def get_training_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get training job by ID."""
        async with self.get_session() as session:
            result = await session.execute(
                f"SELECT * FROM training_jobs WHERE job_id = '{job_id}'"
            )
            row = result.fetchone()
            return dict(row) if row else None
    
    # Dataset operations
    async def create_dataset_source(self, source_id: str, name: str, type: str,
                                path: str, format: str, config: Dict[str, Any],
                                created_by: Optional[int] = None) -> bool:
        """Create a dataset source."""
        async with self.get_session() as session:
            source = DatasetSource(
                source_id=source_id,
                name=name,
                type=type,
                path=path,
                format=format,
                config=config,
                created_by=created_by
            )
            session.add(source)
            await session.flush()
            return True
    
    async def get_dataset_sources(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get dataset sources."""
        async with self.get_session() as session:
            query = "SELECT * FROM dataset_sources"
            if status:
                query += f" WHERE status = '{status}'"
            query += " ORDER BY created_at DESC"
            
            result = await session.execute(query)
            return [dict(row) for row in result.fetchall()]
    
    # Knowledge operations
    async def save_knowledge_item(self, item_id: str, source_id: str, text: str,
                               embedding: Optional[List[float]], metadata: Dict[str, Any],
                               quality_score: float) -> bool:
        """Save a knowledge item."""
        async with self.get_session() as session:
            item = KnowledgeItem(
                item_id=item_id,
                source_id=source_id,
                text=text,
                embedding=embedding,
                metadata=metadata,
                quality_score=quality_score
            )
            session.add(item)
            await session.flush()
            return True
    
    async def search_knowledge_items(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search knowledge items (simple text search)."""
        async with self.get_session() as session:
            result = await session.execute(
                f"SELECT * FROM knowledge_items WHERE text LIKE '%{query}%' "
                f"ORDER BY quality_score DESC LIMIT {limit}"
            )
            return [dict(row) for row in result.fetchall()]
    
    # Audit operations
    async def log_audit_event(self, user_id: Optional[int], action: str, resource: str,
                           details: Optional[Dict[str, Any]] = None,
                           ip_address: Optional[str] = None,
                           user_agent: Optional[str] = None) -> bool:
        """Log an audit event."""
        async with self.get_session() as session:
            audit = AuditLog(
                user_id=user_id,
                action=action,
                resource=resource,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent
            )
            session.add(audit)
            await session.flush()
            return True
    
    async def get_audit_logs(self, user_id: Optional[int] = None, days: int = 30,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit logs."""
        cutoff_date = datetime.now() - timedelta(days=days)
        async with self.get_session() as session:
            query = f"SELECT * FROM audit_logs WHERE timestamp >= '{cutoff_date}'"
            if user_id:
                query += f" AND user_id = {user_id}"
            query += f" ORDER BY timestamp DESC LIMIT {limit}"
            
            result = await session.execute(query)
            return [dict(row) for row in result.fetchall()]
    
    # Health and maintenance
    async def health_check(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            async with self.get_session() as session:
                result = await session.execute("SELECT 1")
                await result.fetchone()
            
            return {
                "status": "healthy",
                "database_url": self.database_url,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def cleanup_old_data(self, days: int = 90) -> Dict[str, int]:
        """Clean up old data."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        async with self.get_session() as session:
            # Clean up old audit logs
            audit_result = await session.execute(
                f"DELETE FROM audit_logs WHERE timestamp < '{cutoff_date}'"
            )
            
            # Clean up old cost metrics (keep longer for analysis)
            cost_cutoff = datetime.now() - timedelta(days=365)
            cost_result = await session.execute(
                f"DELETE FROM cost_metrics WHERE timestamp < '{cost_cutoff}'"
            )
            
            await session.commit()
            
            return {
                "audit_logs_deleted": audit_result.rowcount,
                "cost_metrics_deleted": cost_result.rowcount,
                "cutoff_date": cutoff_date.isoformat()
            }


# Global database manager instance
database_manager = DatabaseManager()