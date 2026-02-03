from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    WORKERS: int = 4
    LOG_LEVEL: str = "INFO"
    ENABLE_DOCS: bool = True
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Model settings
    MODEL_PATH: str = "models/slogpt.pt"
    MODEL_CONTEXT_SIZE: int = 2048
    MODEL_BATCH_SIZE: int = 8
    MODEL_MAX_TOKENS: int = 1024
    
    # Cache settings
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL: int = 3600  # 1 hour
    ENABLE_CACHE: bool = True
    
    # Dataset settings
    DATASET_PATH: str = "data/datasets"
    DATASET_CACHE_SIZE: int = 1000
    
    # Performance settings
    REQUEST_TIMEOUT: int = 30
    MAX_CONCURRENT_REQUESTS: int = 100
    ENABLE_REQUEST_BATCHING: bool = True
    BATCH_SIZE: int = 4
    BATCH_TIMEOUT: float = 0.1
    
    # Monitoring settings
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Security settings
    API_KEY_HEADER: str = "X-API-Key"
    ENABLE_AUTH: bool = False
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Global settings instance
settings = Settings()