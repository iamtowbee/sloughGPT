"""
SloughGPT Database Manager
Provides centralized database management with connection pooling and proper error handling
"""

import os
import logging
from typing import Optional, Dict, Any, List, Union
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
from datetime import datetime

from .database import Base, LearningExperience, LearningSession, KnowledgeNode, KnowledgeEdge, CognitiveState, ModelCheckpoint, ApiRequest
from .exceptions import DatabaseError, create_error

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Centralized database manager with connection pooling and error handling"""
    
    def __init__(self, database_url: Optional[str] = None, pool_size: int = 5, echo: bool = False):
        """
        Initialize database manager
        
        Args:
            database_url: Database connection URL. If None, uses SQLite in-memory
            pool_size: Connection pool size for PostgreSQL/MySQL
            echo: Enable SQLAlchemy query logging
        """
        self.database_url = database_url or "sqlite:///:memory:"
        self.pool_size = pool_size
        self.echo = echo
        self.engine = None
        self.SessionLocal = None
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize database connection and create tables"""
        try:
            # Create engine with appropriate configuration
            if self.database_url.startswith("sqlite"):
                # SQLite configuration
                self.engine = create_engine(
                    self.database_url,
                    poolclass=StaticPool,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": 20,
                        "isolation_level": None
                    },
                    echo=self.echo,
                    pool_pre_ping=True
                )
            else:
                # PostgreSQL/MySQL configuration
                self.engine = create_engine(
                    self.database_url,
                    poolclass=QueuePool,
                    pool_size=self.pool_size,
                    max_overflow=10,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=self.echo
                )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            
            self._initialized = True
            logger.info(f"Database initialized: {self.database_url}")
            
        except SQLAlchemyError as e:
            error = create_error(
                DatabaseError,
                f"Failed to initialize database: {str(e)}",
                None,  # Will be set by exception handler
                cause=e,
                context={"database_url": self.database_url}
            )
            raise error
            
    def health_check(self) -> Dict[str, Any]:
        """Check database health and connectivity"""
        try:
            with self.get_session() as session:
                # Test basic connectivity
                result = session.execute(text("SELECT 1"))
                result.fetchone()
                
                # Count records in each table
                stats = {
                    "status": "healthy",
                    "database_url": self.database_url,
                    "connected_at": datetime.utcnow().isoformat(),
                    "tables": {}
                }
                
                tables = [
                    "learning_experiences", "learning_sessions", "knowledge_nodes",
                    "knowledge_edges", "cognitive_states", "model_checkpoints", "api_requests"
                ]
                
                for table in tables:
                    try:
                        count = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        stats["tables"][table] = count.scalar()
                    except Exception as e:
                        stats["tables"][table] = f"error: {str(e)}"
                
                return stats
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "database_url": self.database_url,
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }
    
    def get_session(self):
        """
        Get a database session
        
        Usage:
            session = db.get_session()
            # Use session here
            user = session.query(User).first()
            session.close()
        """
        if not self._initialized:
            raise create_error(
                DatabaseError,
                "Database not initialized. Call initialize() first.",
                None
            )
        
        return self.SessionLocal()
    
    def execute_raw_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute raw SQL query safely
        
        Args:
            sql: SQL query with named parameters
            params: Dictionary of parameters
            
        Returns:
            List of dictionaries representing rows
        """
        try:
            with self.get_session() as session:
                result = session.execute(text(sql), params or {})
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result.fetchall()]
        except SQLAlchemyError as e:
            raise create_error(
                DatabaseError,
                f"Raw SQL execution failed: {str(e)}",
                None,
                cause=e,
                context={"sql": sql[:200]}  # Limit SQL length in logs
            )
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Create database backup (SQLite only)
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if backup successful
        """
        if not self.database_url.startswith("sqlite"):
            logger.warning("Backup only supported for SQLite databases")
            return False
            
        try:
            with self.get_session() as session:
                # For SQLite, we can use the backup API
                conn = session.connection()
                backup_conn = create_engine(f"sqlite:///{backup_path}").connect()
                
                # Backup the database
                conn.execute(f"ATTACH DATABASE '{backup_path}' AS backup")
                conn.execute("DELETE FROM backup.sqlite_master")  # Clear backup
                conn.execute("INSERT INTO backup.sqlite_master SELECT * FROM main.sqlite_master")
                
                # Copy all tables
                result = conn.execute("SELECT name FROM main.sqlite_master WHERE type='table'")
                for (table_name,) in result:
                    conn.execute(f"INSERT INTO backup.{table_name} SELECT * FROM main.{table_name}")
                
                conn.execute("DETACH DATABASE backup")
                backup_conn.close()
                
                logger.info(f"Database backed up to: {backup_path}")
                return True
                
        except Exception as e:
            logger.error(f"Database backup failed: {str(e)}")
            return False
    
    def cleanup_expired_data(self, days_old: int = 30) -> Dict[str, int]:
        """
        Clean up old data to prevent database bloat
        
        Args:
            days_old: Delete data older than this many days
            
        Returns:
            Dictionary with counts of deleted records
        """
        from datetime import timedelta
        
        cutoff_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date - timedelta(days=days_old)
        
        deleted_counts = {}
        
        try:
            with self.get_session() as session:
                # Clean old API requests
                result = session.execute(
                    text("DELETE FROM api_requests WHERE timestamp < :cutoff"),
                    {"cutoff": cutoff_date}
                )
                deleted_counts["api_requests"] = result.rowcount
                
                # Clean expired cognitive states
                result = session.execute(
                    text("DELETE FROM cognitive_states WHERE expiration_time < :now"),
                    {"now": datetime.utcnow()}
                )
                deleted_counts["cognitive_states"] = result.rowcount
                
                # Clean old learning experiences (keep only recent ones)
                result = session.execute(
                    text("DELETE FROM learning_experiences WHERE timestamp < :cutoff AND rating < 0.5"),
                    {"cutoff": cutoff_date}
                )
                deleted_counts["old_learning_experiences"] = result.rowcount
                
                session.commit()
                logger.info(f"Cleaned up old data: {deleted_counts}")
                
        except SQLAlchemyError as e:
            raise create_error(
                DatabaseError,
                f"Database cleanup failed: {str(e)}",
                None,
                cause=e,
                context={"days_old": days_old}
            )
        
        return deleted_counts
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics and usage information"""
        try:
            with self.get_session() as session:
                stats = {
                    "database_url": self.database_url,
                    "pool_size": self.pool_size,
                    "checked_at": datetime.utcnow().isoformat(),
                    "tables": {}
                }
                
                # Get counts and latest timestamps for each table
                tables_info = {
                    "learning_experiences": LearningExperience,
                    "learning_sessions": LearningSession,
                    "knowledge_nodes": KnowledgeNode,
                    "knowledge_edges": KnowledgeEdge,
                    "cognitive_states": CognitiveState,
                    "model_checkpoints": ModelCheckpoint,
                    "api_requests": ApiRequest
                }
                
                for table_name, model_class in tables_info.items():
                    try:
                        count = session.query(model_class).count()
                        stats["tables"][table_name] = {"count": count}
                        
                        # Get latest timestamp if available
                        if hasattr(model_class, 'timestamp'):
                            latest = session.query(model_class).order_by(
                                model_class.timestamp.desc()
                            ).first()
                            if latest:
                                stats["tables"][table_name]["latest"] = latest.timestamp.isoformat()
                        elif hasattr(model_class, 'creation_time'):
                            latest = session.query(model_class).order_by(
                                model_class.creation_time.desc()
                            ).first()
                            if latest:
                                stats["tables"][table_name]["latest"] = latest.creation_time.isoformat()
                                
                    except Exception as e:
                        stats["tables"][table_name] = {"error": str(e)}
                
                return stats
                
        except SQLAlchemyError as e:
            raise create_error(
                DatabaseError,
                f"Failed to get database statistics: {str(e)}",
                None,
                cause=e
            )
    
    def close(self) -> None:
        """Close database connections and cleanup"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")
            self._initialized = False

# Global database manager instance
_db_manager: Optional[DatabaseManager] = None

def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        database_url = os.getenv("SLOGHPT_DATABASE_URL")
        pool_size = int(os.getenv("SLOGHPT_POOL_SIZE", "5"))
        echo = os.getenv("SLOGHPT_DB_ECHO", "false").lower() == "true"
        
        _db_manager = DatabaseManager(database_url, pool_size, echo)
        _db_manager.initialize()
    
    return _db_manager

def initialize_database(database_url: Optional[str] = None, pool_size: int = 5, echo: bool = False) -> DatabaseManager:
    """
    Initialize the global database manager
    
    Args:
        database_url: Database connection URL
        pool_size: Connection pool size
        echo: Enable query logging
        
    Returns:
        Initialized database manager
    """
    global _db_manager
    _db_manager = DatabaseManager(database_url, pool_size, echo)
    _db_manager.initialize()
    return _db_manager

# Convenience functions
def get_db_session():
    """Get a database session using the global manager"""
    return get_database_manager().get_session()

def db_health_check() -> Dict[str, Any]:
    """Perform database health check"""
    return get_database_manager().health_check()

def get_db_stats() -> Dict[str, Any]:
    """Get database statistics"""
    return get_database_manager().get_statistics()