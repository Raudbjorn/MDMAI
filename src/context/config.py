"""Configuration for the Context Management System."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from config.settings import settings


class ContextConfig(BaseModel):
    """Configuration for context management system."""
    
    # Database configuration
    database_url: str = Field(
        default_factory=lambda: os.getenv(
            "CONTEXT_DATABASE_URL", 
            "postgresql://postgres:postgres@localhost:5432/ttrpg_context"
        ),
        description="PostgreSQL database URL for context storage"
    )
    
    # Redis configuration for real-time sync
    redis_url: str = Field(
        default_factory=lambda: os.getenv(
            "CONTEXT_REDIS_URL",
            "redis://localhost:6379/0"
        ),
        description="Redis URL for event bus and real-time synchronization"
    )
    
    # Feature flags
    enable_real_time_sync: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_ENABLE_REAL_TIME_SYNC", "true").lower() == "true",
        description="Enable real-time synchronization"
    )
    
    enable_compression: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_ENABLE_COMPRESSION", "true").lower() == "true",
        description="Enable context data compression"
    )
    
    enable_validation: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_ENABLE_VALIDATION", "true").lower() == "true",
        description="Enable context validation"
    )
    
    enable_versioning: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_ENABLE_VERSIONING", "true").lower() == "true",
        description="Enable context versioning"
    )
    
    enable_translation: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_ENABLE_TRANSLATION", "true").lower() == "true",
        description="Enable context translation between providers"
    )
    
    # Performance settings
    max_context_size_mb: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_MAX_SIZE_MB", "10")),
        description="Maximum context size in megabytes"
    )
    
    max_versions_per_context: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_MAX_VERSIONS", "100")),
        description="Maximum versions to keep per context"
    )
    
    context_cache_ttl: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_CACHE_TTL", "3600")),
        description="Context cache TTL in seconds"
    )
    
    sync_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_SYNC_TIMEOUT", "30")),
        description="Synchronization timeout in seconds"
    )
    
    # Storage settings
    compression_algorithm: str = Field(
        default_factory=lambda: os.getenv("CONTEXT_COMPRESSION_ALGORITHM", "zstd"),
        description="Default compression algorithm (zstd, lz4, gzip, brotli)"
    )
    
    serialization_format: str = Field(
        default_factory=lambda: os.getenv("CONTEXT_SERIALIZATION_FORMAT", "msgpack"),
        description="Default serialization format (msgpack, json, pickle)"
    )
    
    # Collaborative features
    max_participants_per_room: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_MAX_PARTICIPANTS", "50")),
        description="Maximum participants in collaborative context"
    )
    
    lock_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_LOCK_TIMEOUT", "300")),
        description="Lock timeout for collaborative contexts"
    )
    
    # Event bus settings
    event_history_size: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_EVENT_HISTORY_SIZE", "1000")),
        description="Maximum number of events to keep in history"
    )
    
    event_ttl_seconds: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_EVENT_TTL", "3600")),
        description="Event TTL in seconds"
    )
    
    # Connection pooling
    db_pool_size: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_DB_POOL_SIZE", "20")),
        description="Database connection pool size"
    )
    
    db_max_overflow: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_DB_MAX_OVERFLOW", "30")),
        description="Database connection pool max overflow"
    )
    
    # Security settings
    enable_access_control: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_ENABLE_ACCESS_CONTROL", "true").lower() == "true",
        description="Enable access control for contexts"
    )
    
    enable_audit_logging: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_ENABLE_AUDIT_LOGGING", "true").lower() == "true",
        description="Enable audit logging for context operations"
    )
    
    # Cleanup settings
    auto_cleanup_enabled: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_AUTO_CLEANUP", "true").lower() == "true",
        description="Enable automatic cleanup of old contexts"
    )
    
    auto_archive_days: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_AUTO_ARCHIVE_DAYS", "90")),
        description="Days after which to auto-archive inactive contexts"
    )
    
    cleanup_interval_hours: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_CLEANUP_INTERVAL", "24")),
        description="Cleanup task interval in hours"
    )
    
    # Integration settings
    integrate_with_bridge: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_INTEGRATE_BRIDGE", "true").lower() == "true",
        description="Integrate with MCP Bridge system"
    )
    
    integrate_with_security: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_INTEGRATE_SECURITY", "true").lower() == "true",
        description="Integrate with security system"
    )
    
    # AI Provider settings
    default_provider: str = Field(
        default_factory=lambda: os.getenv("CONTEXT_DEFAULT_PROVIDER", "anthropic"),
        description="Default AI provider for new contexts"
    )
    
    enable_cross_provider_sync: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_CROSS_PROVIDER_SYNC", "true").lower() == "true",
        description="Enable synchronization across different AI providers"
    )
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "CONTEXT_"
        case_sensitive = False


# Global context configuration instance
context_config = ContextConfig()


def get_context_config() -> ContextConfig:
    """Get the global context configuration."""
    return context_config


def validate_context_config() -> bool:
    """Validate context configuration settings."""
    config = get_context_config()
    
    # Validate database URL
    if not config.database_url or not config.database_url.startswith("postgresql"):
        raise ValueError("Invalid database URL. Must be a PostgreSQL connection string.")
    
    # Validate Redis URL
    if config.enable_real_time_sync:
        if not config.redis_url or not config.redis_url.startswith("redis"):
            raise ValueError("Invalid Redis URL. Required for real-time sync.")
    
    # Validate compression algorithm
    valid_compression = ["none", "gzip", "lz4", "zstd", "brotli"]
    if config.compression_algorithm not in valid_compression:
        raise ValueError(f"Invalid compression algorithm. Must be one of: {valid_compression}")
    
    # Validate serialization format
    valid_formats = ["json", "msgpack", "pickle"]
    if config.serialization_format not in valid_formats:
        raise ValueError(f"Invalid serialization format. Must be one of: {valid_formats}")
    
    # Validate numeric settings
    if config.max_context_size_mb <= 0:
        raise ValueError("max_context_size_mb must be positive")
    
    if config.max_versions_per_context <= 0:
        raise ValueError("max_versions_per_context must be positive")
    
    if config.db_pool_size <= 0:
        raise ValueError("db_pool_size must be positive")
    
    return True


def create_context_directories() -> None:
    """Create necessary directories for context management."""
    # Ensure cache directories exist
    cache_dir = Path(settings.cache_dir) / "context"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (cache_dir / "versions").mkdir(exist_ok=True)
    (cache_dir / "translations").mkdir(exist_ok=True)
    (cache_dir / "temp").mkdir(exist_ok=True)
    
    # Ensure log directory exists
    log_dir = Path("logs") / "context"
    log_dir.mkdir(parents=True, exist_ok=True)