"""
Admin Dashboard Configuration
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class AdminTheme(str, Enum):
    """Available dashboard themes"""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"

class LogLevel(str, Enum):
    """Log levels for admin dashboard"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class AdminConfig(BaseModel):
    """Configuration for the admin dashboard"""
    
    # Server settings
    host: str = Field(default="127.0.0.1", description="Host to bind the admin server")
    port: int = Field(default=8080, description="Port for the admin dashboard")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # UI settings
    theme: AdminTheme = Field(default=AdminTheme.LIGHT, description="Dashboard theme")
    title: str = Field(default="SloughGPT Admin", description="Dashboard title")
    logo_url: Optional[str] = Field(default=None, description="URL to custom logo")
    
    # Features
    enable_docs: bool = Field(default=True, description="Enable API documentation")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_websockets: bool = Field(default=True, description="Enable WebSocket updates")
    enable_alerts: bool = Field(default=True, description="Enable alerting system")
    
    # Security settings
    allowed_hosts: List[str] = Field(default=["*"], description="CORS allowed hosts")
    enable_auth: bool = Field(default=False, description="Require authentication for dashboard")
    session_timeout: int = Field(default=3600, description="Session timeout in seconds")
    
    # Metrics settings
    metrics_retention_days: int = Field(default=30, description="Days to retain metrics")
    refresh_interval: int = Field(default=30, description="Auto-refresh interval in seconds")
    
    # Alert settings
    alert_webhooks: List[str] = Field(default=[], description="Webhook URLs for alerts")
    alert_email: Optional[str] = Field(default=None, description="Email for alerts")
    
    # Cache settings
    enable_cache: bool = Field(default=True, description="Enable dashboard caching")
    cache_ttl: int = Field(default=300, description="Cache TTL in seconds")
    
    class Config:
        """Pydantic configuration"""
        env_prefix = "SLOUGHGPT_ADMIN_"
        case_sensitive = False
        
    def get_theme_mode(self) -> str:
        """Get the theme mode for CSS"""
        if self.theme == AdminTheme.AUTO:
            # Auto theme - could implement system preference detection
            return "light"
        return self.theme.value.lower()
        
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.debug or self.host in ["localhost", "127.0.0.1", "0.0.0.0"]
        
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        return {
            "allow_origins": self.allowed_hosts if self.allowed_hosts != ["*"] else ["*"],
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["*"],
        }

# Default configuration instance
DEFAULT_ADMIN_CONFIG = AdminConfig()

# Environment-based configuration helpers
def create_admin_config(**kwargs) -> AdminConfig:
    """Create admin configuration with environment variables support"""
    # Load from environment variables first
    config_dict = {}
    
    # Map common environment variables
    env_mappings = {
        "ADMIN_HOST": "host",
        "ADMIN_PORT": "port", 
        "ADMIN_DEBUG": "debug",
        "ADMIN_THEME": "theme",
        "ADMIN_TITLE": "title",
        "ADMIN_ENABLE_AUTH": "enable_auth",
        "ADMIN_ENABLE_DOCS": "enable_docs",
        "ADMIN_ENABLE_METRICS": "enable_metrics",
        "ADMIN_REFRESH_INTERVAL": "refresh_interval",
    }
    
    import os
    
    for env_var, config_key in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Type conversion
            if config_key in ["debug", "enable_auth", "enable_docs", "enable_metrics"]:
                config_dict[config_key] = value.lower() in ["true", "1", "yes"]
            elif config_key in ["port", "refresh_interval"]:
                config_dict[config_key] = int(value)
            elif config_key == "theme":
                config_dict[config_key] = AdminTheme(value.upper())
            else:
                config_dict[config_key] = value
    
    # Override with provided kwargs
    config_dict.update(kwargs)
    
    return AdminConfig(**config_dict)

# Configuration validation
def validate_admin_config(config: AdminConfig) -> List[str]:
    """Validate admin configuration and return list of issues"""
    issues = []
    
    # Port validation
    if config.port < 1 or config.port > 65535:
        issues.append("Port must be between 1 and 65535")
    
    # Host validation
    if not config.host:
        issues.append("Host cannot be empty")
    
    # Session timeout validation
    if config.session_timeout < 60:
        issues.append("Session timeout must be at least 60 seconds")
    
    # Cache TTL validation
    if config.cache_ttl < 1:
        issues.append("Cache TTL must be at least 1 second")
    
    # Metrics retention validation
    if config.metrics_retention_days < 1:
        issues.append("Metrics retention must be at least 1 day")
    
    # Refresh interval validation
    if config.refresh_interval < 5:
        issues.append("Refresh interval must be at least 5 seconds")
    
    return issues

# Configuration for production environments
def get_production_config() -> AdminConfig:
    """Get production-ready admin configuration"""
    return AdminConfig(
        host="0.0.0.0",
        port=8080,
        debug=False,
        theme=AdminTheme.DARK,
        enable_auth=True,
        enable_docs=False,  # Disable in production
        enable_metrics=True,
        enable_websockets=True,
        enable_alerts=True,
        allowed_hosts=[],  # Restrict to specific domains
        session_timeout=1800,  # 30 minutes
        metrics_retention_days=90,
        refresh_interval=60,
        enable_cache=True,
        cache_ttl=600,  # 10 minutes
    )

# Configuration for development environments  
def get_development_config() -> AdminConfig:
    """Get development admin configuration"""
    return AdminConfig(
        host="127.0.0.1",
        port=8080,
        debug=True,
        theme=AdminTheme.LIGHT,
        enable_auth=False,
        enable_docs=True,
        enable_metrics=True,
        enable_websockets=True,
        enable_alerts=True,
        allowed_hosts=["*"],
        session_timeout=3600,
        metrics_retention_days=7,
        refresh_interval=10,
        enable_cache=True,
        cache_ttl=60,
    )