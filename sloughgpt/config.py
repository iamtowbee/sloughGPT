"""
SloughGPT Configuration Module
Centralized configuration management
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class DatabaseType(Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"

@dataclass
class ModelConfig:
    """Model configuration with default values"""
    vocab_size: int = 50257
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 1024
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE") else "cpu"
    quantized: bool = False
    
    # Additional attributes for compatibility
    hidden_size: int = 512
    num_attention_heads: int = 8
    num_hidden_layers: int = 6
    intermediate_size: int = 2048
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 1024
    layer_norm_eps: float = 1e-6
    activation: str = "gelu"
    
    def __post_init__(self):
        """Initialize compatibility attributes"""
        self.hidden_size = self.d_model
        self.num_attention_heads = self.n_heads
        self.num_hidden_layers = self.n_layers
        self.intermediate_size = self.d_ff
        self.hidden_dropout = self.dropout
        self.max_position_embeddings = self.max_seq_length
    
    def validate(self) -> list[str]:
        """Validate configuration and return errors"""
        errors = []
        if self.vocab_size <= 0:
            errors.append("vocab_size must be positive")
        if self.d_model <= 0:
            errors.append("d_model must be positive")
        if self.n_heads <= 0:
            errors.append("n_heads must be positive")
        if self.n_layers <= 0:
            errors.append("n_layers must be positive")
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "max_seq_length": self.max_seq_length,
            "device": self.device,
            "quantized": self.quantized,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "hidden_dropout": self.hidden_dropout,
            "max_position_embeddings": self.max_position_embeddings,
            "layer_norm_eps": self.layer_norm_eps,
            "activation": self.activation
        }

@dataclass
class LearningConfig:
    """Learning system configuration"""
    experience_buffer_size: int = 10000
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_gradient_norm: float = 1.0
    save_frequency: int = 100
    adaptive_lr: bool = True

@dataclass
class CognitiveConfig:
    """Cognitive system configuration"""
    confidence_threshold: float = 0.7
    creativity_threshold: float = 0.6
    reasoning_depth: int = 3
    max_thoughts_per_request: int = 5
    exploration_rate: float = 0.1

@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    enable_caching: bool = True
    cache_size: int = 1000
    enable_quantization: bool = True
    enable_async: bool = True
    max_concurrent_requests: int = 100

@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 60
    enable_authentication: bool = True
    api_key_required: bool = False
    max_prompt_length: int = 1000
    allowed_origins: Optional[list] = None
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = []

@dataclass
class DatabaseConfig:
    """Database configuration"""
    database_type: DatabaseType = DatabaseType.SQLITE
    database_url: Optional[str] = None
    connection_pool_size: int = 5
    echo: bool = False

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: LogLevel = LogLevel.INFO
    format: str = "json"
    file_path: Optional[str] = None
    max_file_size: str = "10MB"
    backup_count: int = 5

@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    access_log: bool = True

@dataclass
class SloughGPTConfig:
    """Complete SloughGPT configuration"""
    model: Optional[ModelConfig] = None
    learning: Optional[LearningConfig] = None
    cognitive: Optional[CognitiveConfig] = None
    performance: Optional[PerformanceConfig] = None
    security: Optional[SecurityConfig] = None
    database: Optional[DatabaseConfig] = None
    logging: Optional[LoggingConfig] = None
    server: Optional[ServerConfig] = None
    
    def __post_init__(self):
        # Initialize default configs if not provided
        if self.model is None:
            self.model = ModelConfig()
        if self.learning is None:
            self.learning = LearningConfig()
        if self.cognitive is None:
            self.cognitive = CognitiveConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.database is None:
            self.database = DatabaseConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.server is None:
            self.server = ServerConfig()
    
    @classmethod
    def from_env(cls) -> "SloughGPTConfig":
        """Create configuration from environment variables"""
        config = cls()
        
        # Ensure all configs are initialized
        if config.model is None:
            config.model = ModelConfig()
        if config.learning is None:
            config.learning = LearningConfig()
        if config.cognitive is None:
            config.cognitive = CognitiveConfig()
        if config.performance is None:
            config.performance = PerformanceConfig()
        if config.security is None:
            config.security = SecurityConfig()
        if config.database is None:
            config.database = DatabaseConfig()
        if config.logging is None:
            config.logging = LoggingConfig()
        if config.server is None:
            config.server = ServerConfig()
        
        # Model configuration
        if config.model:
            config.model.vocab_size = int(os.environ.get("SLOGHPT_VOCAB_SIZE", "10000"))
            config.model.d_model = int(os.environ.get("SLOGHPT_D_MODEL", "512"))
            config.model.n_heads = int(os.environ.get("SLOGHPT_N_HEADS", "8"))
            config.model.n_layers = int(os.environ.get("SLOGHPT_N_LAYERS", "6"))
            config.model.device = os.environ.get("SLOGHPT_DEVICE", config.model.device)
            config.model.quantized = os.environ.get("SLOGHPT_QUANTIZED", "false").lower() == "true"
        
        # Learning configuration
        if config.learning:
            config.learning.learning_rate = float(os.environ.get("SLOGHPT_LEARNING_RATE", "1e-4"))
            config.learning.experience_buffer_size = int(os.environ.get("SLOGHPT_BUFFER_SIZE", "10000"))
            config.learning.adaptive_lr = os.environ.get("SLOGHPT_ADAPTIVE_LR", "true").lower() == "true"
        
        # Security configuration
        if config.security:
            config.security.enable_rate_limiting = os.environ.get("SLOGHPT_RATE_LIMIT", "true").lower() == "true"
            config.security.rate_limit_per_minute = int(os.environ.get("SLOGHPT_RATE_PER_MINUTE", "60"))
            config.security.enable_authentication = os.environ.get("SLOGHPT_AUTH", "true").lower() == "true"
            config.security.api_key_required = os.environ.get("SLOGHPT_API_KEY_REQUIRED", "false").lower() == "true"
            config.security.max_prompt_length = int(os.environ.get("SLOGHPT_MAX_PROMPT_LENGTH", "1000"))
        
        # Server configuration
        if config.server:
            config.server.host = os.environ.get("SLOGHPT_HOST", "0.0.0.0")
            config.server.port = int(os.environ.get("SLOGHPT_PORT", "8000"))
            config.server.workers = int(os.environ.get("SLOGHPT_WORKERS", "1"))
        
        # Database configuration
        if config.database:
            db_type = os.environ.get("SLOGHPT_DB_TYPE", "sqlite")
            config.database.database_type = DatabaseType(db_type)
            config.database.database_url = os.environ.get("SLOGHPT_DATABASE_URL")
            config.database.connection_pool_size = int(os.environ.get("SLOGHPT_POOL_SIZE", "5"))
        
        # Logging configuration
        if config.logging:
            log_level = os.environ.get("SLOGHPT_LOG_LEVEL", "info")
            config.logging.level = LogLevel(log_level)
            config.logging.format = os.environ.get("SLOGHPT_LOG_FORMAT", "json")
            config.logging.file_path = os.environ.get("SLOGHPT_LOG_FILE")
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model": self.model.__dict__ if self.model else {},
            "learning": self.learning.__dict__ if self.learning else {},
            "cognitive": self.cognitive.__dict__ if self.cognitive else {},
            "performance": self.performance.__dict__ if self.performance else {},
            "security": self.security.__dict__ if self.security else {},
            "database": self.database.__dict__ if self.database else {},
            "logging": self.logging.__dict__ if self.logging else {},
            "server": self.server.__dict__ if self.server else {}
        }
    
    def validate(self) -> list[str]:
        """Validate configuration and return errors"""
        errors = []
        
        # Validate model configuration
        if self.model:
            if self.model.vocab_size <= 0:
                errors.append("Model vocab_size must be positive")
            if self.model.d_model <= 0:
                errors.append("Model d_model must be positive")
            if self.model.n_heads <= 0:
                errors.append("Model n_heads must be positive")
            if self.model.n_layers <= 0:
                errors.append("Model n_layers must be positive")
        
        # Validate security configuration
        if self.security:
            if self.security.rate_limit_per_minute <= 0:
                errors.append("Rate limit per minute must be positive")
            if self.security.max_prompt_length <= 0:
                errors.append("Max prompt length must be positive")
            if self.security.max_prompt_length > 10000:
                errors.append("Max prompt length too large (max: 10000)")
        
        # Validate server configuration
        if self.server:
            if self.server.port < 1 or self.server.port > 65535:
                errors.append("Server port must be between 1 and 65535")
            if self.server.workers < 1:
                errors.append("Server workers must be positive")
        
        return errors