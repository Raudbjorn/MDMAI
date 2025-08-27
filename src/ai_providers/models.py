"""Data models for AI Provider Integration."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ProviderType(Enum):
    """Supported AI provider types."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai" 
    GOOGLE = "google"


class ProviderCapability(Enum):
    """AI provider capabilities."""
    
    TEXT_GENERATION = "text_generation"
    TOOL_CALLING = "tool_calling"
    VISION = "vision"
    STREAMING = "streaming"
    BATCH_PROCESSING = "batch_processing"
    FINE_TUNING = "fine_tuning"


class ProviderStatus(Enum):
    """AI provider status."""
    
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class CostTier(Enum):
    """Cost tiers for provider models."""
    
    FREE = "free"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PREMIUM = "premium"


@dataclass
class ProviderConfig:
    """Configuration for an AI provider."""
    
    provider_type: ProviderType
    api_key: str
    base_url: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_rpm: int = 60  # Requests per minute
    rate_limit_tpm: int = 40000  # Tokens per minute
    budget_limit: Optional[float] = None  # USD per day
    enabled: bool = True
    priority: int = 1  # Higher = preferred
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    """Specification for an AI model."""
    
    model_id: str
    provider_type: ProviderType
    display_name: str
    capabilities: List[ProviderCapability] = field(default_factory=list)
    context_length: int = 4096
    max_output_tokens: int = 2048
    cost_per_input_token: float = 0.0  # USD per 1K tokens
    cost_per_output_token: float = 0.0  # USD per 1K tokens
    cost_tier: CostTier = CostTier.MEDIUM
    supports_streaming: bool = False
    supports_tools: bool = False
    supports_vision: bool = False
    is_available: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCPTool(BaseModel):
    """MCP tool definition."""
    
    name: str
    description: str
    inputSchema: Dict[str, Any]


class ProviderTool(BaseModel):
    """Provider-specific tool definition."""
    
    name: str
    description: str 
    parameters: Dict[str, Any]
    provider_type: ProviderType


class AIRequest(BaseModel):
    """Request to an AI provider."""
    
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: Optional[str] = None
    model: str
    messages: List[Dict[str, Any]]
    tools: Optional[List[MCPTool]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    metadata: Optional[Dict[str, Any]] = None
    budget_limit: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)


class AIResponse(BaseModel):
    """Response from an AI provider."""
    
    request_id: str
    provider_type: ProviderType
    model: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    cost: Optional[float] = None
    latency_ms: Optional[float] = None
    cached: bool = False
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)


class StreamingChunk(BaseModel):
    """Streaming response chunk."""
    
    request_id: str
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    is_complete: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass 
class UsageRecord:
    """Usage tracking record."""
    
    request_id: str
    session_id: Optional[str] = None
    provider_type: ProviderType
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostBudget:
    """Cost budget configuration."""
    
    budget_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = "Default Budget"
    daily_limit: Optional[float] = None  # USD
    monthly_limit: Optional[float] = None  # USD
    provider_limits: Dict[ProviderType, float] = field(default_factory=dict)
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.95])
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProviderHealth:
    """Health status of an AI provider."""
    
    provider_type: ProviderType
    status: ProviderStatus
    last_success: Optional[datetime] = None
    last_error: Optional[datetime] = None
    error_count: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    quota_remaining: Optional[float] = None
    uptime_percentage: float = 100.0
    updated_at: datetime = field(default_factory=datetime.now)


class ProviderSelection(BaseModel):
    """Provider selection criteria."""
    
    required_capabilities: List[ProviderCapability] = Field(default_factory=list)
    preferred_providers: List[ProviderType] = Field(default_factory=list)
    exclude_providers: List[ProviderType] = Field(default_factory=list) 
    max_cost_per_request: Optional[float] = None
    max_latency_ms: Optional[float] = None
    require_streaming: bool = False
    require_tools: bool = False
    cost_optimization: bool = True


class AIProviderStats(BaseModel):
    """Statistics for AI provider usage."""
    
    provider_type: ProviderType
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    uptime_percentage: float = 100.0
    last_request: Optional[datetime] = None
    daily_usage: Dict[str, float] = Field(default_factory=dict)  # Date -> cost
    monthly_usage: Dict[str, float] = Field(default_factory=dict)  # Month -> cost