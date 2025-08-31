"""Configuration module for the MCP Bridge Service."""

from functools import cached_property
from typing import Annotated, Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import BridgeConfig


class BridgeSettings(BaseSettings):
    """Bridge service configuration settings with improved Pydantic v2 patterns."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="BRIDGE_",
        extra="ignore",
    )

    # Server configuration
    host: Annotated[str, Field(default="127.0.0.1", alias="BRIDGE_HOST")]
    port: Annotated[int, Field(default=8080, ge=1, le=65535, alias="BRIDGE_PORT")]
    workers: Annotated[int, Field(default=1, ge=1, le=100, alias="BRIDGE_WORKERS")]

    # MCP server configuration
    mcp_server_path: Annotated[str, Field(default="src.main", alias="MCP_SERVER_PATH")]
    mcp_server_args: Annotated[List[str], Field(default_factory=list, alias="MCP_SERVER_ARGS")]
    mcp_server_env: Annotated[Dict[str, str], Field(default_factory=dict, alias="MCP_SERVER_ENV")]

    # Process management
    max_processes: Annotated[int, Field(default=10, ge=1, le=100)]
    process_timeout: Annotated[int, Field(default=300, ge=1)]
    process_idle_timeout: Annotated[int, Field(default=600, ge=1)]
    process_health_check_interval: Annotated[int, Field(default=30, ge=1)]
    process_restart_on_failure: Annotated[bool, Field(default=True)]
    max_restart_attempts: Annotated[int, Field(default=3, ge=0, le=10)]

    # Session management
    max_sessions_per_client: Annotated[int, Field(default=3, ge=1, le=50)]
    session_timeout: Annotated[int, Field(default=3600, ge=60)]
    session_cleanup_interval: Annotated[int, Field(default=60, ge=10)]

    # Transport configuration
    enable_websocket: Annotated[bool, Field(default=True)]
    enable_sse: Annotated[bool, Field(default=True)]
    enable_http: Annotated[bool, Field(default=True)]
    websocket_ping_interval: Annotated[int, Field(default=30, ge=5, alias="WS_PING_INTERVAL")]
    websocket_ping_timeout: Annotated[int, Field(default=10, ge=1, alias="WS_PING_TIMEOUT")]

    # Security
    require_auth: Annotated[bool, Field(default=False)]
    auth_header: Annotated[str, Field(default="Authorization")]
    api_keys: Annotated[List[str], Field(default_factory=list)]
    # CORS configuration - secure defaults
    cors_origins: Annotated[
        List[str],
        Field(default=["http://localhost:3000", "http://127.0.0.1:3000"]),
    ]
    cors_credentials: Annotated[bool, Field(default=False)]

    # Performance
    request_timeout: Annotated[float, Field(default=30.0, gt=0)]
    max_request_size: Annotated[int, Field(default=10 * 1024 * 1024, gt=0)]
    enable_request_batching: Annotated[bool, Field(default=True, alias="ENABLE_BATCHING")]
    batch_timeout: Annotated[float, Field(default=0.1, gt=0)]
    max_batch_size: Annotated[int, Field(default=10, ge=1, le=100)]

    # Rate limiting
    enable_rate_limiting: Annotated[bool, Field(default=True)]
    rate_limit_requests: Annotated[int, Field(default=100, ge=1)]
    rate_limit_period: Annotated[int, Field(default=60, ge=1)]

    # Monitoring
    enable_metrics: Annotated[bool, Field(default=True)]
    metrics_port: Annotated[int, Field(default=9090, ge=1, le=65535)]
    enable_tracing: Annotated[bool, Field(default=False)]
    tracing_endpoint: Annotated[Optional[str], Field(default=None)]

    # Logging
    log_level: Annotated[
        str,
        Field(
            default="INFO",
            pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        ),
    ]
    log_file: Annotated[Optional[str], Field(default=None)]
    log_requests: Annotated[bool, Field(default=False)]
    log_responses: Annotated[bool, Field(default=False)]

    # Static files
    static_dir: Annotated[str, Field(default="src/bridge/static")]
    enable_static_files: Annotated[bool, Field(default=True, alias="ENABLE_STATIC")]

    # Validators using modern Pydantic v2 patterns
    @field_validator("mcp_server_args", mode="before")
    @classmethod
    def parse_server_args(cls, v: str | List[str]) -> List[str]:
        """Parse server args from string if needed using pattern matching."""
        match v:
            case str() as s:
                return s.split() if s else []
            case list() as lst:
                return lst
            case _:
                return []

    @field_validator("mcp_server_env", mode="before")
    @classmethod
    def parse_server_env(cls, v: str | Dict[str, str]) -> Dict[str, str]:
        """Parse server env from string if needed using pattern matching."""
        match v:
            case str() as s:
                return dict(
                    item.split("=", 1)
                    for item in s.split(",")
                    if "=" in item and item.strip()
                )
            case dict() as d:
                return d
            case _:
                return {}

    @field_validator("api_keys", "cors_origins", mode="before")
    @classmethod
    def parse_comma_separated(cls, v: str | List[str]) -> List[str]:
        """Parse comma-separated string to list using pattern matching."""
        match v:
            case str() as s:
                return [item.strip() for item in s.split(",") if item.strip()]
            case list() as lst:
                return lst
            case _:
                return []

    @cached_property
    def is_secure_mode(self) -> bool:
        """Check if running in secure mode."""
        return self.require_auth and bool(self.api_keys)

    @cached_property
    def transport_modes(self) -> List[str]:
        """Get enabled transport modes."""
        modes = []
        if self.enable_http:
            modes.append("http")
        if self.enable_websocket:
            modes.append("websocket")
        if self.enable_sse:
            modes.append("sse")
        return modes

    def to_bridge_config(self) -> BridgeConfig:
        """Convert settings to BridgeConfig with selective field mapping."""
        # Map only the fields that exist in BridgeConfig
        return BridgeConfig(
            # MCP server settings
            mcp_server_path=self.mcp_server_path,
            mcp_server_args=self.mcp_server_args,
            mcp_server_env=self.mcp_server_env,
            # Process management
            max_processes=self.max_processes,
            process_timeout=self.process_timeout,
            process_idle_timeout=self.process_idle_timeout,
            process_health_check_interval=self.process_health_check_interval,
            # Session management
            max_sessions_per_client=self.max_sessions_per_client,
            session_timeout=self.session_timeout,
            session_cleanup_interval=self.session_cleanup_interval,
            # Transport
            enable_websocket=self.enable_websocket,
            enable_sse=self.enable_sse,
            enable_http=self.enable_http,
            # Security
            require_auth=self.require_auth,
            auth_header=self.auth_header,
            api_keys=self.api_keys,
            # Performance
            request_timeout=self.request_timeout,
            max_request_size=self.max_request_size,
            enable_request_batching=self.enable_request_batching,
            batch_timeout=self.batch_timeout,
            # Monitoring
            enable_metrics=self.enable_metrics,
            metrics_port=self.metrics_port,
            enable_tracing=self.enable_tracing,
            # Logging
            log_level=self.log_level,
            log_requests=self.log_requests,
            log_responses=self.log_responses,
        )

    def get_runtime_info(self) -> Dict[str, Any]:
        """Get runtime configuration information."""
        return {
            "host": f"{self.host}:{self.port}",
            "workers": self.workers,
            "secure_mode": self.is_secure_mode,
            "transport_modes": self.transport_modes,
            "max_processes": self.max_processes,
            "rate_limiting": {
                "enabled": self.enable_rate_limiting,
                "limit": f"{self.rate_limit_requests}/{self.rate_limit_period}s",
            },
            "monitoring": {
                "metrics": self.enable_metrics,
                "tracing": self.enable_tracing,
                "metrics_port": self.metrics_port if self.enable_metrics else None,
            },
        }


# Global settings instance with lazy initialization
_settings: Optional[BridgeSettings] = None


def get_settings() -> BridgeSettings:
    """Get or create settings instance (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = BridgeSettings()
    return _settings


def get_bridge_config() -> BridgeConfig:
    """Get the bridge configuration."""
    return get_settings().to_bridge_config()


def reload_settings() -> BridgeSettings:
    """Reload settings from environment (useful for testing)."""
    global _settings
    _settings = BridgeSettings()
    return _settings