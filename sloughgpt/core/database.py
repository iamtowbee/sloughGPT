"""
SloughGPT Database Models
SQLAlchemy-based database models for proper ORM support
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, Boolean, 
    ForeignKey, Index, JSON, create_engine, event
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from sqlalchemy.pool import StaticPool
import json

Base = declarative_base()

class LearningExperience(Base):
    """Learning experiences and feedback"""
    __tablename__ = "learning_experiences"
    
    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    rating = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON, default=dict)
    model_version = Column(String(50), default="1.0")
    session_id = Column(String(100), ForeignKey("learning_sessions.session_id"), index=True)
    
    # Relationships
    session = relationship("LearningSession", back_populates="experiences")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "response": self.response,
            "rating": self.rating,
            "feedback": self.feedback,
            "learned_from": self.learned_from,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp is not None else None,
            "metadata": self.metadata_json or {},
            "model_version": self.model_version,
            "session_id": self.session_id
        }

class LearningSession(Base):
    """Learning session tracking"""
    __tablename__ = "learning_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_experiences = Column(Integer, default=0)
    average_rating = Column(Float, default=0.0)
    model_config = Column(JSON, default=dict)
    
    # Relationships
    experiences = relationship("LearningExperience", back_populates="session")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat() if self.start_time is not None else None,
            "end_time": self.end_time.isoformat() if self.end_time is not None else None,
            "total_experiences": self.total_experiences,
            "average_rating": self.average_rating,
            "model_config": self.model_config or {}
        }

class KnowledgeNode(Base):
    """Knowledge graph nodes"""
    __tablename__ = "knowledge_nodes"
    
    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String(100), unique=True, nullable=False, index=True)
    content = Column(Text, nullable=False)
    node_type = Column(String(50), default="concept")
    importance = Column(Float, default=1.0)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    embedding = Column(JSON, nullable=True)  # Store as JSON for flexibility
    metadata_json = Column(JSON, default=dict)
    
    # Relationships
    outgoing_edges = relationship("KnowledgeEdge", foreign_keys="KnowledgeEdge.source_node_id", back_populates="source_node")
    incoming_edges = relationship("KnowledgeEdge", foreign_keys="KnowledgeEdge.target_node_id", back_populates="target_node")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "content": self.content,
            "node_type": self.node_type,
            "importance": self.importance,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed is not None else None,
            "access_count": self.access_count,
            "embedding": self.embedding,
            "metadata": self.metadata_json or {}
        }

class KnowledgeEdge(Base):
    """Knowledge graph edges/relationships"""
    __tablename__ = "knowledge_edges"
    
    id = Column(Integer, primary_key=True, index=True)
    edge_id = Column(String(100), unique=True, nullable=False, index=True)
    source_node_id = Column(String(100), ForeignKey("knowledge_nodes.node_id"), nullable=False, index=True)
    target_node_id = Column(String(100), ForeignKey("knowledge_nodes.node_id"), nullable=False, index=True)
    relationship_type = Column(String(50), default="related_to")
    strength = Column(Float, default=1.0)
    last_activation = Column(DateTime, default=datetime.utcnow)
    activation_count = Column(Integer, default=0)
    metadata_json = Column(JSON, default=dict)
    
    # Relationships
    source_node = relationship("KnowledgeNode", foreign_keys=[source_node_id], back_populates="outgoing_edges")
    target_node = relationship("KnowledgeNode", foreign_keys=[target_node_id], back_populates="incoming_edges")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "relationship_type": self.relationship_type,
            "strength": self.strength,
            "last_activation": self.last_activation.isoformat() if self.last_activation is not None else None,
            "activation_count": self.activation_count,
            "metadata": self.metadata_json or {}
        }

class CognitiveState(Base):
    """Cognitive system state tracking"""
    __tablename__ = "cognitive_states"
    
    id = Column(Integer, primary_key=True, index=True)
    state_id = Column(String(100), unique=True, nullable=False, index=True)
    state_type = Column(String(50), default="working_memory")
    content = Column(Text, nullable=False)
    priority = Column(Float, default=1.0)
    creation_time = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    expiration_time = Column(DateTime, nullable=True)
    metadata_json = Column(JSON, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "state_id": self.state_id,
            "state_type": self.state_type,
            "content": self.content,
            "priority": self.priority,
            "creation_time": self.creation_time.isoformat() if self.creation_time is not None else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed is not None else None,
            "access_count": self.access_count,
            "expiration_time": self.expiration_time.isoformat() if self.expiration_time is not None else None,
            "metadata": self.metadata_json or {}
        }

class ModelCheckpoint(Base):
    """Model checkpoint tracking"""
    __tablename__ = "model_checkpoints"
    
    id = Column(Integer, primary_key=True, index=True)
    checkpoint_id = Column(String(100), unique=True, nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    checkpoint_path = Column(String(500), nullable=False)
    training_loss = Column(Float, nullable=True)
    validation_loss = Column(Float, nullable=True)
    training_step = Column(Integer, default=0)
    epoch = Column(Integer, default=0)
    creation_time = Column(DateTime, default=datetime.utcnow)
    file_size = Column(Integer, default=0)
    metadata_json = Column(JSON, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "checkpoint_id": self.checkpoint_id,
            "model_name": self.model_name,
            "checkpoint_path": self.checkpoint_path,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "training_step": self.training_step,
            "epoch": self.epoch,
            "creation_time": self.creation_time.isoformat() if self.creation_time is not None else None,
            "file_size": self.file_size,
            "metadata": self.metadata_json or {}
        }

class ApiRequest(Base):
    """API request logging and analytics"""
    __tablename__ = "api_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(100), unique=True, nullable=False, index=True)
    endpoint = Column(String(200), nullable=False)
    method = Column(String(10), nullable=False)
    request_data = Column(JSON, nullable=True)
    response_status = Column(Integer, default=200)
    response_time_ms = Column(Integer, default=0)
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    timestamp = Column(DateTime, default=datetime.utcnow)
    error_message = Column(Text, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "method": self.method,
            "request_data": self.request_data,
            "response_status": self.response_status,
            "response_time_ms": self.response_time_ms,
            "user_agent": self.user_agent,
            "ip_address": self.ip_address,
            "timestamp": self.timestamp.isoformat() if self.timestamp is not None else None,
            "error_message": self.error_message
        }

# Create indexes for better performance
Index('idx_learning_experiences_timestamp', LearningExperience.timestamp)
Index('idx_learning_experiences_rating', LearningExperience.rating)
Index('idx_knowledge_nodes_importance', KnowledgeNode.importance)
Index('idx_knowledge_edges_strength', KnowledgeEdge.strength)
Index('idx_cognitive_states_priority', CognitiveState.priority)
Index('idx_api_requests_timestamp', ApiRequest.timestamp)
Index('idx_api_requests_endpoint', ApiRequest.endpoint)

# Event listeners for automatic timestamp updates
@event.listens_for(LearningExperience, 'before_update')
@event.listens_for(KnowledgeNode, 'before_update')
@event.listens_for(KnowledgeEdge, 'before_update')
@event.listens_for(CognitiveState, 'before_update')
def update_last_accessed(mapper, connection, target):
    """Automatically update last_accessed timestamp"""
    if hasattr(target, 'last_accessed'):
        target.last_accessed = datetime.utcnow()
    if hasattr(target, 'access_count'):
        target.access_count = (target.access_count or 0) + 1

@event.listens_for(KnowledgeEdge, 'before_update')
def update_edge_activation(mapper, connection, target):
    """Automatically update edge activation"""
    target.last_activation = datetime.utcnow()
    target.activation_count = (target.activation_count or 0) + 1