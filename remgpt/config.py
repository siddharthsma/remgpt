"""
Configuration management for RemGPT.
Provides centralized configuration with environment variable support.
"""

import os
import logging
from typing import List, Optional
from pydantic import Field

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings


class RemGPTConfig(BaseSettings):
    """RemGPT configuration with environment variable support."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host address")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")
    
    # Authentication
    auth_mode: str = Field(default="mock", description="Authentication mode: mock, jwt, oauth")
    jwt_secret: Optional[str] = Field(default=None, description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiry_hours: int = Field(default=24, description="JWT token expiry in hours")
    
    # LLM Configuration
    default_llm_provider: str = Field(default="openai", description="Default LLM provider")
    default_model: str = Field(default="gpt-4", description="Default LLM model")
    llm_timeout: int = Field(default=60, description="LLM request timeout in seconds")
    
    # Context Management
    default_max_tokens: int = Field(default=4000, description="Default maximum tokens")
    token_limit_threshold: float = Field(default=0.7, description="Token limit warning threshold")
    default_system_instructions: str = Field(
        default="You are a helpful AI assistant with context management abilities.",
        description="Default system instructions"
    )
    
    # Topic Drift Detection
    drift_similarity_threshold: float = Field(default=0.7, description="Topic drift similarity threshold")
    drift_detection_threshold: float = Field(default=0.5, description="Topic drift detection threshold")
    drift_alpha: float = Field(default=0.05, description="Page-Hinkley test alpha value")
    drift_window_size: int = Field(default=10, description="Drift detection window size")
    
    # Remote Tools
    enable_mcp: bool = Field(default=True, description="Enable MCP (Model Context Protocol) support")
    enable_a2a: bool = Field(default=True, description="Enable A2A (Agent-to-Agent) support")
    default_mcp_servers: List[str] = Field(default_factory=list, description="Default MCP servers")
    default_a2a_agents: List[str] = Field(default_factory=list, description="Default A2A agents")
    
    # Performance and Limits
    max_concurrent_requests: int = Field(default=100, description="Maximum concurrent requests")
    request_timeout: int = Field(default=300, description="Request timeout in seconds")
    max_request_size: int = Field(default=10 * 1024 * 1024, description="Maximum request size in bytes")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    # Development
    debug: bool = Field(default=False, description="Enable debug mode")
    auto_reload: bool = Field(default=False, description="Enable auto-reload in development")
    
    class Config:
        env_file = ".env"
        env_prefix = "REMGPT_"
        case_sensitive = False


# Global configuration instance
_config: Optional[RemGPTConfig] = None


def get_config() -> RemGPTConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = RemGPTConfig()
        setup_logging(_config)
    return _config


def setup_logging(config: RemGPTConfig) -> None:
    """Set up logging based on configuration."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format=config.log_format
    )
    
    # Adjust logging for specific modules if needed
    if config.debug:
        logging.getLogger("remgpt").setLevel(logging.DEBUG)
    else:
        # Reduce noise from external libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


def reload_config() -> RemGPTConfig:
    """Reload configuration from environment."""
    global _config
    _config = None
    return get_config()


# Environment variable helpers
def is_production() -> bool:
    """Check if running in production environment."""
    return os.getenv("REMGPT_ENVIRONMENT", "development").lower() == "production"


def is_development() -> bool:
    """Check if running in development environment."""
    return not is_production()


def get_database_url() -> Optional[str]:
    """Get database URL from environment."""
    return os.getenv("REMGPT_DATABASE_URL")


def get_redis_url() -> Optional[str]:
    """Get Redis URL from environment."""
    return os.getenv("REMGPT_REDIS_URL") 