"""
SloughGPT Database Module
Supports SQLite (default) and PostgreSQL
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import uuid

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./sloughgpt.db")

# Try to import database libraries
try:
    import psycopg2

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import sqlalchemy
    from sqlalchemy import (
        create_engine,
        Column,
        Integer,
        String,
        Float,
        DateTime,
        Text,
        JSON,
        Boolean,
    )
    from sqlalchemy.orm import sessionmaker, Session, declarative_base

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


Base = declarative_base()


# ============================================================================
# Database Models
# ============================================================================


class ConversationModel(Base):
    """Conversation database model."""

    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    meta = Column(JSON, default={})

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "meta": self.meta or {},
        }


class MessageModel(Base):
    """Message database model."""

    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    meta = Column(JSON, default={})

    def to_dict(self):
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "meta": self.meta or {},
        }


class ModelModel(Base):
    """Model database model."""

    __tablename__ = "models"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    provider = Column(String, default="local")
    status = Column(String, default="available")
    config = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "status": self.status,
            "config": self.config or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TrainingJobModel(Base):
    """Training job database model."""

    __tablename__ = "training_jobs"

    id = Column(String, primary_key=True)
    status = Column(String, default="pending")
    progress = Column(Integer, default=0)
    loss = Column(Float, nullable=True)
    config = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "loss": self.loss,
            "config": self.config or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class DatasetModel(Base):
    """Dataset database model."""

    __tablename__ = "datasets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False)
    path = Column(String, nullable=False)
    size = Column(Integer, default=0)
    config = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "size": self.size,
            "config": self.config or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================================
# Database Manager
# ============================================================================


class DatabaseManager:
    """Manages database connections and operations."""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or DATABASE_URL
        self.engine = None
        self.SessionLocal = None
        self._connected = False

    def connect(self):
        """Connect to database."""
        if not SQLALCHEMY_AVAILABLE:
            # Fall back to JSON file storage
            self._connected = True
            return False

        try:
            self.engine = create_engine(self.database_url)
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(bind=self.engine)
            self._connected = True
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            self._connected = False
            return False

    def get_session(self) -> Optional[Session]:
        """Get a database session."""
        if self.SessionLocal:
            return self.SessionLocal()
        return None

    # Conversation operations
    def create_conversation(self, name: str = None, metadata: dict = None) -> Dict:
        """Create a new conversation."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            conv = ConversationModel(name=name, metadata=metadata or {})
            session.add(conv)
            session.commit()
            result = conv.to_dict()
            session.close()
            return result
        else:
            # Fallback to file storage
            return self._create_conversation_file(name, metadata)

    def get_conversation(self, conv_id: str) -> Optional[Dict]:
        """Get a conversation by ID."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            conv = session.query(ConversationModel).filter_by(id=conv_id).first()
            result = conv.to_dict() if conv else None
            session.close()
            return result
        else:
            return self._get_conversation_file(conv_id)

    def list_conversations(self) -> List[Dict]:
        """List all conversations."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            convs = session.query(ConversationModel).all()
            result = [c.to_dict() for c in convs]
            session.close()
            return result
        else:
            return self._list_conversations_file()

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            conv = session.query(ConversationModel).filter_by(id=conv_id).first()
            if conv:
                session.delete(conv)
                session.commit()
                session.close()
                return True
            session.close()
            return False
        else:
            return self._delete_conversation_file(conv_id)

    # Message operations
    def add_message(
        self, conversation_id: str, role: str, content: str, metadata: dict = None
    ) -> Dict:
        """Add a message to a conversation."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            msg = MessageModel(
                conversation_id=conversation_id, role=role, content=content, metadata=metadata or {}
            )
            session.add(msg)
            session.commit()
            result = msg.to_dict()
            session.close()
            return result
        else:
            return self._add_message_file(conversation_id, role, content, metadata)

    def get_messages(self, conversation_id: str) -> List[Dict]:
        """Get all messages in a conversation."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            msgs = session.query(MessageModel).filter_by(conversation_id=conversation_id).all()
            result = [m.to_dict() for m in msgs]
            session.close()
            return result
        else:
            return self._get_messages_file(conversation_id)

    # Training job operations
    def create_training_job(self, job_id: str, config: dict) -> Dict:
        """Create a training job record."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            job = TrainingJobModel(id=job_id, config=config)
            session.add(job)
            session.commit()
            result = job.to_dict()
            session.close()
            return result
        return {"id": job_id, "config": config}

    def update_training_job(self, job_id: str, **kwargs) -> bool:
        """Update a training job."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            job = session.query(TrainingJobModel).filter_by(id=job_id).first()
            if job:
                for key, value in kwargs.items():
                    setattr(job, key, value)
                session.commit()
                session.close()
                return True
            session.close()
            return False
        return False

    def get_training_job(self, job_id: str) -> Optional[Dict]:
        """Get a training job."""
        if self._connected and self.SessionLocal:
            session = self.get_session()
            job = session.query(TrainingJobModel).filter_by(id=job_id).first()
            result = job.to_dict() if job else None
            session.close()
            return result
        return None

    # File-based fallback methods
    def _create_conversation_file(self, name, metadata):
        conv_id = str(uuid.uuid4())
        conv = {"id": conv_id, "name": name, "metadata": metadata, "messages": []}

        Path("data/conversations").mkdir(parents=True, exist_ok=True)
        with open(f"data/conversations/{conv_id}.json", "w") as f:
            json.dump(conv, f)
        return conv

    def _get_conversation_file(self, conv_id):
        try:
            with open(f"data/conversations/{conv_id}.json", "r") as f:
                return json.load(f)
        except:
            return None

    def _list_conversations_file(self):
        Path("data/conversations").mkdir(parents=True, exist_ok=True)
        convs = []
        for f in Path("data/conversations").glob("*.json"):
            with open(f) as fp:
                convs.append(json.load(fp))
        return convs

    def _delete_conversation_file(self, conv_id):
        try:
            Path(f"data/conversations/{conv_id}.json").unlink()
            return True
        except:
            return False

    def _add_message_file(self, conv_id, role, content, metadata):
        conv = self._get_conversation_file(conv_id)
        if conv:
            msg = {"id": str(uuid.uuid4()), "role": role, "content": content, "metadata": metadata}
            conv["messages"].append(msg)
            with open(f"data/conversations/{conv_id}.json", "w") as f:
                json.dump(conv, f)
            return msg
        return None

    def _get_messages_file(self, conv_id):
        conv = self._get_conversation_file(conv_id)
        if conv:
            return conv.get("messages", [])
        return []


# Default database manager instance
db = DatabaseManager()


def init_db(database_url: str = None):
    """Initialize the database."""
    global db
    db = DatabaseManager(database_url)
    return db.connect()


__all__ = [
    "DatabaseManager",
    "db",
    "init_db",
    "ConversationModel",
    "MessageModel",
    "ModelModel",
    "TrainingJobModel",
    "DatasetModel",
]
