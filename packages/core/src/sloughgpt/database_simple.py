"""Simple database implementation for SloughGPT without external dependencies."""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import json
import sqlite3
import logging
from contextlib import asynccontextmanager
import uuid


@dataclass
class User:
    """Simple user model."""
    id: int
    username: str
    email: str
    password_hash: str
    role: str
    is_active: bool = True
    created_at: datetime = None
    updated_at: datetime = None
    last_login: Optional[datetime] = None


@dataclass
class APIKey:
    """Simple API key model."""
    id: int
    user_id: int
    name: str
    key_hash: str
    permissions: List[str]
    rate_limit: int = 1000
    is_active: bool = True
    last_used: Optional[datetime] = None
    created_at: datetime = None
    expires_at: Optional[datetime] = None


@dataclass
class CostMetric:
    """Simple cost metric model."""
    id: int
    user_id: int
    metric_type: str
    amount: float
    unit: str
    cost: float
    model_name: Optional[str]
    metadata_data: Optional[Dict[str, Any]]
    timestamp: datetime


@dataclass
class TrainingJob:
    """Simple training job model."""
    id: int
    job_id: str
    user_id: int
    model_name: str
    dataset_path: str
    output_dir: str
    config: Dict[str, Any]
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    metrics: Optional[Dict[str, Any]]
    checkpoints: Optional[List[str]]
    final_model_path: Optional[str]


@dataclass
class SecurityEvent:
    """Simple security event model."""
    id: int
    event_id: str
    event_type: str
    threat_level: str
    user_id: Optional[int]
    ip_address: str
    user_agent: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolution: Optional[str]


class SimpleDatabaseManager:
    """Simple database manager using SQLite."""
    
    def __init__(self, database_path: str = "./sloughgpt.db"):
        self.database_path = database_path
        self.connection = None
        self.initialized = False
        
    async def initialize(self) -> bool:
        """Initialize database connection."""
        try:
            self.connection = sqlite3.connect(self.database_path)
            self.connection.row_factory = sqlite3.Row
            
            # Create tables
            await self._create_tables()
            
            self.initialized = True
            logging.info(f"Database initialized: {self.database_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize database: {e}")
            return False
    
    async def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.initialized = False
            logging.info("Database connection closed")
    
    async def _create_tables(self):
        """Create database tables."""
        cursor = self.connection.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)
        
        # API keys table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                permissions TEXT NOT NULL,
                rate_limit INTEGER DEFAULT 1000,
                is_active BOOLEAN DEFAULT TRUE,
                last_used TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Cost metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cost_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                metric_type TEXT NOT NULL,
                amount REAL NOT NULL,
                unit TEXT NOT NULL,
                cost REAL NOT NULL,
                model_name TEXT,
                metadata_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Training jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                dataset_path TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                config TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                metrics TEXT,
                checkpoints TEXT,
                final_model_path TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Security events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                event_type TEXT NOT NULL,
                threat_level TEXT NOT NULL,
                user_id INTEGER,
                ip_address TEXT NOT NULL,
                user_agent TEXT,
                details TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved BOOLEAN DEFAULT FALSE,
                resolution TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users (username)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys (key_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cost_metrics_user ON cost_metrics (user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cost_metrics_timestamp ON cost_metrics (timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_user ON training_jobs (user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_security_events_user ON security_events (user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON security_events (timestamp)")
        
        self.connection.commit()
    
    # User operations
    async def create_user(self, username: str, email: str, password_hash: str, 
                        role: str = "user") -> Optional[int]:
        """Create a new user."""
        if not self.initialized:
            return None
        
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, role, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (username, email, password_hash, role, datetime.now(), datetime.now()))
            
            self.connection.commit()
            return cursor.lastrowid
            
        except Exception as e:
            logging.error(f"Failed to create user: {e}")
            return None
    
    async def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        if not self.initialized:
            return None
        
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        if not self.initialized:
            return None
        
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    # API Key operations
    async def create_api_key(self, user_id: int, name: str, key_hash: str,
                           permissions: List[str], rate_limit: int = 1000,
                           expires_days: int = 365) -> Optional[int]:
        """Create an API key."""
        if not self.initialized:
            return None
        
        cursor = self.connection.cursor()
        try:
            expires_at = datetime.now() + timedelta(days=expires_days)
            
            cursor.execute("""
                INSERT INTO api_keys (user_id, name, key_hash, permissions, rate_limit, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, name, key_hash, json.dumps(permissions), rate_limit, datetime.now(), expires_at))
            
            self.connection.commit()
            return cursor.lastrowid
            
        except Exception as e:
            logging.error(f"Failed to create API key: {e}")
            return None
    
    # Cost metrics operations
    async def save_cost_metric(self, user_id: int, metric_type: str, amount: float,
                           unit: str, cost: float, model_name: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save a cost metric."""
        if not self.initialized:
            return False
        
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                INSERT INTO cost_metrics (user_id, metric_type, amount, unit, cost, model_name, metadata_data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, metric_type, amount, unit, cost, model_name, json.dumps(metadata) if metadata else None, datetime.now()))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logging.error(f"Failed to save cost metric: {e}")
            return False
    
    async def get_user_cost_metrics(self, user_id: int, days: int = 30) -> List[Dict[str, Any]]:
        """Get cost metrics for a user."""
        if not self.initialized:
            return []
        
        cursor = self.connection.cursor()
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            SELECT * FROM cost_metrics 
            WHERE user_id = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        """, (user_id, cutoff_date.isoformat()))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    # Training job operations
    async def create_training_job(self, job_id: str, user_id: int, model_name: str,
                              dataset_path: str, output_dir: str, config: Dict[str, Any]) -> bool:
        """Create a training job."""
        if not self.initialized:
            return False
        
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                INSERT INTO training_jobs (job_id, user_id, model_name, dataset_path, output_dir, config, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (job_id, user_id, model_name, dataset_path, output_dir, json.dumps(config), datetime.now()))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logging.error(f"Failed to create training job: {e}")
            return False
    
    async def update_training_job_status(self, job_id: str, status: str,
                                   metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Update training job status."""
        if not self.initialized:
            return False
        
        cursor = self.connection.cursor()
        try:
            updates = ["status = ?", "started_at = ?" if status in ["running", "completed", "failed"] else None]
            updates.append("completed_at = ?" if status in ["completed", "failed"] else None)
            updates.append("metrics = ?" if metrics else None)
            
            values = [status, datetime.now() if status in ["running", "completed", "failed"] else None]
            values.append(datetime.now() if status in ["completed", "failed"] else None)
            values.append(json.dumps(metrics) if metrics else None)
            
            # Filter out None values
            update_parts = []
            update_values = []
            for i, update in enumerate(updates):
                if update and i < len(values) and values[i] is not None:
                    update_parts.append(update)
                    update_values.append(values[i])
            
            if update_parts:
                query = f"UPDATE training_jobs SET {', '.join(update_parts)} WHERE job_id = ?"
                update_values.append(job_id)
                
                cursor.execute(query, update_values)
                self.connection.commit()
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Failed to update training job: {e}")
            return False
    
    # Security operations
    async def log_security_event(self, user_id: Optional[int], action: str, resource: str,
                           details: Optional[Dict[str, Any]] = None,
                           ip_address: Optional[str] = None,
                           user_agent: Optional[str] = None) -> bool:
        """Log an audit event."""
        if not self.initialized:
            return False
        
        cursor = self.connection.cursor()
        try:
            event_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO security_events (event_id, event_type, threat_level, user_id, ip_address, user_agent, details, timestamp)
                VALUES (?, ?, 'medium', ?, ?, ?, ?, ?, ?)
            """, (event_id, action, user_id, ip_address or "unknown", user_agent or "unknown", json.dumps(details) or "{}", datetime.now()))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logging.error(f"Failed to log security event: {e}")
            return False
    
    # Health and maintenance
    async def health_check(self) -> Dict[str, Any]:
        """Check database health."""
        if not self.initialized:
            return {
                "status": "not_initialized",
                "error": "Database not initialized"
            }
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            
            return {
                "status": "healthy",
                "database_path": self.database_path,
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
        if not self.initialized:
            return {}
        
        cursor = self.connection.cursor()
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            # Clean up old security events
            cursor.execute("DELETE FROM security_events WHERE timestamp < ?", (cutoff_date.isoformat(),))
            security_deleted = cursor.rowcount
            
            # Clean up old cost metrics (keep longer for analysis)
            cost_cutoff = datetime.now() - timedelta(days=365)
            cursor.execute("DELETE FROM cost_metrics WHERE timestamp < ?", (cost_cutoff.isoformat(),))
            cost_deleted = cursor.rowcount
            
            self.connection.commit()
            
            return {
                "security_events_deleted": security_deleted,
                "cost_metrics_deleted": cost_deleted,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            logging.error(f"Cleanup failed: {e}")
            return {}


# Global database manager instance
database_manager = SimpleDatabaseManager()