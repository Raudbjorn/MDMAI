"""Configuration module for the MCP Bridge Service."""

from typing import Dict, List, Optional

from pydantic import BaseSettings, Field, validator

from .models import BridgeConfig


class BridgeSettings(BaseSettings):
    """Bridge service configuration settings."""
    
    # Server configuration
    host: str = Field(default="127.0.0.1", env="BRIDGE_HOST")
    port: int = Field(default=8080, env="BRIDGE_PORT")
    workers: int = Field(default=1, env="BRIDGE_WORKERS")
    
    # MCP server configuration
    mcp_server_path: str = Field(default="src.main", env="MCP_SERVER_PATH")
    mcp_server_args: List[str] = Field(default_factory=list, env="MCP_SERVER_ARGS")
    mcp_server_env: Dict[str, str] = Field(default_factory=dict, env="MCP_SERVER_ENV")
    
    # Process management
    max_processes: int = Field(default=10, env="BRIDGE_MAX_PROCESSES")
    process_timeout: int = Field(default=300, env="BRIDGE_PROCESS_TIMEOUT")
    process_idle_timeout: int = Field(default=600, env="BRIDGE_PROCESS_IDLE_TIMEOUT")
    process_health_check_interval: int = Field(default=30, env="BRIDGE_HEALTH_CHECK_INTERVAL")
    process_restart_on_failure: bool = Field(default=True, env="BRIDGE_RESTART_ON_FAILURE")
    max_restart_attempts: int = Field(default=3, env="BRIDGE_MAX_RESTART_ATTEMPTS")
    
    # Session management
    max_sessions_per_client: int = Field(default=3, env="BRIDGE_MAX_SESSIONS_PER_CLIENT")
    session_timeout: int = Field(default=3600, env="BRIDGE_SESSION_TIMEOUT")
    session_cleanup_interval: int = Field(default=60, env="BRIDGE_SESSION_CLEANUP_INTERVAL")
    
    # Transport configuration
    enable_websocket: bool = Field(default=True, env="BRIDGE_ENABLE_WEBSOCKET")
    enable_sse: bool = Field(default=True, env="BRIDGE_ENABLE_SSE")
    enable_http: bool = Field(default=True, env="BRIDGE_ENABLE_HTTP")
    websocket_ping_interval: int = Field(default=30, env="BRIDGE_WS_PING_INTERVAL")
    websocket_ping_timeout: int = Field(default=10, env="BRIDGE_WS_PING_TIMEOUT")
    
    # Security
    require_auth: bool = Field(default=False, env="BRIDGE_REQUIRE_AUTH")
    auth_header: str = Field(default="Authorization", env="BRIDGE_AUTH_HEADER")
    api_keys: List[str] = Field(default_factory=list, env="BRIDGE_API_KEYS")
    # CORS configuration - secure defaults to prevent security vulnerabilities
    # Using wildcard origins with credentials is a security risk
    cors_origins: List[str] = Field(default=["http://localhost:3000", "http://127.0.0.1:3000"], env="BRIDGE_CORS_ORIGINS")
    cors_credentials: bool = Field(default=False, env="BRIDGE_CORS_CREDENTIALS")
    
    # Performance
    request_timeout: float = Field(default=30.0, env="BRIDGE_REQUEST_TIMEOUT")
    max_request_size: int = Field(default=10 * 1024 * 1024, env="BRIDGE_MAX_REQUEST_SIZE")
    enable_request_batching: bool = Field(default=True, env="BRIDGE_ENABLE_BATCHING")
    batch_timeout: float = Field(default=0.1, env="BRIDGE_BATCH_TIMEOUT")
    max_batch_size: int = Field(default=10, env="BRIDGE_MAX_BATCH_SIZE")
    
    # Rate limiting
    enable_rate_limiting: bool = Field(default=True, env="BRIDGE_ENABLE_RATE_LIMITING")
    rate_limit_requests: int = Field(default=100, env="BRIDGE_RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, env="BRIDGE_RATE_LIMIT_PERIOD")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="BRIDGE_ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="BRIDGE_METRICS_PORT")
    enable_tracing: bool = Field(default=False, env="BRIDGE_ENABLE_TRACING")
    tracing_endpoint: Optional[str] = Field(default=None, env="BRIDGE_TRACING_ENDPOINT")
    
    # Logging
    log_level: str = Field(default="INFO", env="BRIDGE_LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="BRIDGE_LOG_FILE")
    log_requests: bool = Field(default=False, env="BRIDGE_LOG_REQUESTS")
    log_responses: bool = Field(default=False, env="BRIDGE_LOG_RESPONSES")
    
    # Static files
    static_dir: str = Field(default="src/bridge/static", env="BRIDGE_STATIC_DIR")
    enable_static_files: bool = Field(default=True, env="BRIDGE_ENABLE_STATIC")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("mcp_server_args", pre=True)
    def parse_server_args(cls, v):
        """Parse server args from string if needed."""
        if isinstance(v, str):
            return v.split() if v else []
        return v
    
    @validator("mcp_server_env", pre=True)
    def parse_server_env(cls, v):
        """Parse server env from string if needed."""
        if isinstance(v, str):
            env_vars = {}
            for item in v.split(","):
                if "=" in item:
                    key, value = item.split("=", 1)
                    env_vars[key.strip()] = value.strip()
            return env_vars
        return v
    
    @validator("api_keys", pre=True)
    def parse_api_keys(cls, v):
        """Parse API keys from comma-separated string."""
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v
    
    def to_bridge_config(self) -> BridgeConfig:
        """Convert settings to BridgeConfig."""
        return BridgeConfig(
            mcp_server_path=self.mcp_server_path,
            mcp_server_args=self.mcp_server_args,
            mcp_server_env=self.mcp_server_env,
            max_processes=self.max_processes,
            process_timeout=self.process_timeout,
            process_idle_timeout=self.process_idle_timeout,
            process_health_check_interval=self.process_health_check_interval,
            max_sessions_per_client=self.max_sessions_per_client,
            session_timeout=self.session_timeout,
            session_cleanup_interval=self.session_cleanup_interval,
            enable_websocket=self.enable_websocket,
            enable_sse=self.enable_sse,
            enable_http=self.enable_http,
            require_auth=self.require_auth,
            auth_header=self.auth_header,
            api_keys=self.api_keys,
            request_timeout=self.request_timeout,
            max_request_size=self.max_request_size,
            enable_request_batching=self.enable_request_batching,
            batch_timeout=self.batch_timeout,
            enable_metrics=self.enable_metrics,
            metrics_port=self.metrics_port,
            enable_tracing=self.enable_tracing,
            log_level=self.log_level,
            log_requests=self.log_requests,
            log_responses=self.log_responses,
        )


# Global settings instance
settings = BridgeSettings()


def get_bridge_config() -> BridgeConfig:
    """Get the bridge configuration."""
    return settings.to_bridge_config()