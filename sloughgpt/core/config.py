"""
SloughGPT Core Configuration Module
Centralized configuration management
"""

import os
from typing import Optional, Dict, Any
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
    """Model configuration"""
    vocab_size: int = 10000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 1024
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE") else "cpu"
    quantized: bool = False
    compile: bool = False
    attention_type: str = "scaled_dot_product_attention"  # "flash_attention", "efficient_attention"

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
    allowed_origins: list = None

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
    model: ModelConfig = None
    learning: LearningConfig = None
    cognitive: CognitiveConfig = None
    performance: PerformanceConfig = None
    security: SecurityConfig = None
    database: DatabaseConfig = None
    logging: LoggingConfig = None
    server: ServerConfig = None
    
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
        
        # Model configuration
        config.model.vocab_size = int(os.environ.get("SLOGHPT_VOCAB_SIZE", "10000"))
        config.model.d_model = int(os.environ.get("SLOGHPT_D_MODEL", "512"))
        config.model.n_heads = int(os.environ.get("SLOGHPT_N_HEADS", "8"))
        config.model.n_layers = int(os.environ.get("SLOGHPT_N_LAYERS", "6"))
        config.model.d_ff = int(os.environ.get("SLOGHPT_D_FF", "2048"))
        config.model.dropout = float(os.environ.get("SLOGHPT_DROPOUT", "0.1"))
        config.model.max_seq_length = int(os.environ.get("SLOGHPT_MAX_SEQ_LENGTH", "1024"))
        config.model.device = os.environ.get("SLOGHPT_DEVICE", "cuda" if os.environ.get("CUDA_VISIBLE") else "cpu")
        config.model.quantized = os.environ.get("SLOGHPT_QUANTIZED", "false").lower() == "true"
        config.model.compile = os.environ.get("SLOGHPT_COMPILE", "false").lower() == "true"
        
        # Learning configuration
        config.learning.learning_rate = float(os.environ.get("SLOGHPT_LEARNING_RATE", "1e-4"))
        config.learning.experience_buffer_size = int(os.environ.get("SLOGHPT_BUFFER_SIZE", "10000"))
        config.learning.batch_size = int(os.environ.get("SLOGHPT_BATCH_SIZE", "32"))
        config.learning.max_gradient_norm = float(os.environ.get("SLOGHPT_MAX_GRAD_NORM", "1.0"))
        config.learning.save_frequency = int(os.environ.get("SLOGHPT_SAVE_FREQUENCY", "100"))
        config.learning.adaptive_lr = os.environ.get("SLOGHPT_ADAPTIVE_LR", "true").lower() == "true"
        
        # Cognitive configuration
        config.cognitive.confidence_threshold = float(os.environ.get("SLOGHPT_CONFIDENCE_THRESHOLD", "0.7"))
        config.cognitive.creativity_threshold = float(os.environ.get("SLOGHPT_CREATIVITY_THRESHOLD", "0.6"))
        config.cognitive.reasoning_depth = int(os.environ.get("SLOGHPT_REASONING_DEPTH", "3"))
        config.cognitive.max_thoughts_per_request = int(os.environ.get("SLOGHPT_MAX_THOGHTS_PER_REQUEST", "5"))
        config.cognitive.exploration_rate = float(os.environ.get("SLOGHPT_EXPLORATION_RATE", "0.1"))
        
        # Performance configuration
        config.performance.enable_caching = os.environ.get("SLOGHPT_ENABLE_CACHING", "true").lower() == "true"
        config.performance.cache_size = int(os.environ.get("SLOGHPT_CACHE_SIZE", "1000"))
        config.performance.enable_quantization = os.environ.get("SLOGHPT_ENABLE_QUANTIZATION", "true").lower() == "true")
        config.performance.enable_async = os.environ.get("SLOGHPT_ENABLE_ASYNC", "true").lower() == "true"
        config.performance.max_concurrent_requests = int(os.environ.get("SLOGHPT_MAX_CONCURRENT_REQUESTS", "100"))
        
        # Security configuration
        config.security.enable_rate_limiting = os.environ.get("SLOGHPT_RATE_LIMIT", "true").lower() == "true")
        config.security.rate_limit_per_minute = int(os.environ.get("SLOGHPT_RATE_PER_MINUTE", "60"))
        config.security.enable_authentication = os.environ.get("SLOGHPT_AUTH", "true").lower() == "true")
        config.security.api_key_required = os.environ.get("SLOGHPT_API_KEY_REQUIRED", "false").lower() == "true"
        config.security.max_prompt_length = int(os.environ.get("SLOGHPT_MAX_PROMPT_LENGTH", "1000"))
        config.security.allowed_origins = None
        
        # Database configuration
        config.database.database_type = DatabaseType(os.environ.get("SLOGHPT_DB_TYPE", "sqlite"))
        config.database.database_url = os.environ.get("SLOGHPT_DATABASE_URL")
        config.database.connection_pool_size = int(os.environ.get("SLOGHPT_POOL_SIZE", "5"))
        
        # Logging configuration
        log_level = os.environ.get("SLOGHPT_LOG_LEVEL", "info")
        config.logging.format = os.environ.get("SLOGHPT_LOG_FORMAT", "json")
        config.logging.file_path = os.environ.get("SLOGHPT_LOG_FILE")
        config.logging.max_file_size = str(os.environ.get("SLOGHPT_MAX_FILE_SIZE", "10MB"))
        
        # Server configuration
        config.server.host = os.environ.get("SLOGHPT_HOST", "0.0.0.0")
        config.server.port = int(os.environ.get("SLOGHPT_PORT", "8000"))
        config.server.workers = int(os.environ.get("SLOGHPT_WORKERS", "1"))
        config.server.reload = os.environ.get("SLOGHPT_RELOAD", "false").lower() == "true"
        config.server.access_log = os.environ.get("SLOGHPT_ACCESS_LOG", "true").lower() == "true"
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for JSON serialization"""
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
    
    def validate(self) -> List[str]:
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
            if self.model.max_seq_length <= 0:
                errors.append("Model max_seq_length must be positive")
        
        # Validate security configuration
        if self.security:
            if self.security.rate_limit_per_minute <= 0:
                errors.append("Rate limit per minute must be positive")
            if self.security.max_prompt_length <= 0:
                errors.append("Max prompt length must be positive")
        
        # Validate server configuration
        if self.server:
            if self.server.port < 1 or self.server.port > 65535:
                errors.append("Server port must be between 1 and 65535")
            if self.server.workers < 1:
                errors.append("Server workers must be positive")
        
        return errors