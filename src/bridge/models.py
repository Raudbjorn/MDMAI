"""Data models for the MCP Bridge Service."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SessionState(Enum):
    """Session lifecycle states."""
    
    INITIALIZING = "initializing"
    CONNECTED = "connected"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    DISCONNECTED = "disconnected"
    TERMINATED = "terminated"


class TransportType(Enum):
    """Supported transport types for client connections."""
    
    WEBSOCKET = "websocket"
    SSE = "sse"
    HTTP = "http"


class MessageType(Enum):
    """MCP message types."""
    
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class MCPSession:
    """Represents an MCP session with a stdio subprocess."""
    
    session_id: str = field(default_factory=lambda: str(uuid4()))
    process_id: Optional[int] = None
    client_id: Optional[str] = None
    state: SessionState = SessionState.INITIALIZING
    transport: Optional[TransportType] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    pending_requests: Dict[str, "PendingRequest"] = field(default_factory=dict)
    
    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.now()
    
    def is_active(self, timeout_seconds: int = 300) -> bool:
        """Check if session is still active within timeout."""
        if self.state in (SessionState.DISCONNECTED, SessionState.TERMINATED):
            return False
        elapsed = (datetime.now() - self.last_activity).total_seconds()
        return elapsed < timeout_seconds


@dataclass
class PendingRequest:
    """Tracks a pending MCP request awaiting response."""
    
    request_id: str
    method: str
    params: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    timeout: Optional[float] = None
    callback: Optional[Any] = None  # AsyncIO future or callback
    
    def is_expired(self) -> bool:
        """Check if request has exceeded timeout."""
        if self.timeout is None:
            return False
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.timeout


class MCPRequest(BaseModel):
    """MCP JSON-RPC 2.0 request."""
    
    jsonrpc: str = Field(default="2.0", const=True)
    id: Union[str, int, None] = Field(default_factory=lambda: str(uuid4()))
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    """MCP JSON-RPC 2.0 response."""
    
    jsonrpc: str = Field(default="2.0", const=True)
    id: Union[str, int, None]
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class MCPNotification(BaseModel):
    """MCP JSON-RPC 2.0 notification."""
    
    jsonrpc: str = Field(default="2.0", const=True)
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPError(BaseModel):
    """MCP JSON-RPC 2.0 error object."""
    
    code: int
    message: str
    data: Optional[Any] = None


# Standard MCP error codes
class MCPErrorCode:
    """Standard JSON-RPC 2.0 and MCP-specific error codes."""
    
    # JSON-RPC 2.0 standard errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # MCP-specific errors
    PROCESS_ERROR = -32001
    SESSION_ERROR = -32002
    TIMEOUT_ERROR = -32003
    TRANSPORT_ERROR = -32004
    INITIALIZATION_ERROR = -32005
    CAPABILITY_ERROR = -32006


@dataclass
class BridgeConfig:
    """Configuration for the MCP Bridge Service."""
    
    # MCP server configuration
    mcp_server_path: str = "src.main:main"
    mcp_server_args: List[str] = field(default_factory=list)
    mcp_server_env: Dict[str, str] = field(default_factory=dict)
    
    # Process management
    max_processes: int = 10
    process_timeout: int = 300  # seconds
    process_idle_timeout: int = 600  # seconds
    process_health_check_interval: int = 30  # seconds
    
    # Session management
    max_sessions_per_client: int = 3
    session_timeout: int = 3600  # seconds
    session_cleanup_interval: int = 60  # seconds
    
    # Transport configuration
    enable_websocket: bool = True
    enable_sse: bool = True
    enable_http: bool = True
    
    # Security
    require_auth: bool = False
    auth_header: str = "Authorization"
    api_keys: List[str] = field(default_factory=list)
    
    # Performance
    request_timeout: float = 30.0  # seconds
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    enable_request_batching: bool = True
    batch_timeout: float = 0.1  # seconds
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_requests: bool = False
    log_responses: bool = False


class ClientMessage(BaseModel):
    """Message from a web client to the bridge."""
    
    type: str  # "request", "notification", "batch"
    data: Union[MCPRequest, MCPNotification, List[Union[MCPRequest, MCPNotification]]]
    session_id: Optional[str] = None
    client_id: Optional[str] = None


class BridgeMessage(BaseModel):
    """Message from the bridge to a web client."""
    
    type: str  # "response", "notification", "error", "event"
    data: Union[MCPResponse, MCPNotification, MCPError, Dict[str, Any]]
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


@dataclass
class ProcessStats:
    """Statistics for an MCP process."""
    
    process_id: int
    session_id: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    num_requests: int = 0
    num_errors: int = 0
    uptime_seconds: float = 0.0
    last_request: Optional[datetime] = None


@dataclass
class BridgeStats:
    """Overall statistics for the bridge service."""
    
    active_sessions: int = 0
    active_processes: int = 0
    total_requests: int = 0
    total_errors: int = 0
    average_response_time_ms: float = 0.0
    uptime_seconds: float = 0.0
    process_stats: List[ProcessStats] = field(default_factory=list)