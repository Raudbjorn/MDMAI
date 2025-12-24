"""
MCP Protocol Schemas and JSON-RPC 2.0 Message Patterns for Provider Router.

This module defines the complete JSON-RPC 2.0 message patterns, error codes,
and protocol schemas for the Provider Router with Fallback system.

Key Components:
- JSON-RPC 2.0 compliant message structures
- Provider routing event schemas
- Error code definitions with detailed error handling
- Protocol versioning and compatibility
- Message validation and serialization
"""

import json
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator


# Protocol Version and Constants
PROTOCOL_VERSION = "1.0.0"
MCP_PROVIDER_ROUTER_NAMESPACE = "provider_router"


# JSON-RPC 2.0 Base Classes
class JSONRPCVersion(str, Enum):
    """JSON-RPC protocol version."""
    V2_0 = "2.0"


class JSONRPCRequest(BaseModel):
    """Base JSON-RPC 2.0 request structure."""
    jsonrpc: JSONRPCVersion = JSONRPCVersion.V2_0
    method: str = Field(..., description="Method name to invoke")
    params: Optional[Dict[str, Any]] = Field(None, description="Method parameters")
    id: Optional[Union[str, int]] = Field(default_factory=lambda: str(uuid4()), description="Request identifier")


class JSONRPCResponse(BaseModel):
    """Base JSON-RPC 2.0 response structure."""
    jsonrpc: JSONRPCVersion = JSONRPCVersion.V2_0
    id: Optional[Union[str, int]] = Field(..., description="Request identifier")


class JSONRPCSuccessResponse(JSONRPCResponse):
    """JSON-RPC 2.0 success response."""
    result: Any = Field(..., description="Method result")


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error object."""
    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional error data")


class JSONRPCErrorResponse(JSONRPCResponse):
    """JSON-RPC 2.0 error response."""
    error: JSONRPCError = Field(..., description="Error details")


class JSONRPCNotification(BaseModel):
    """JSON-RPC 2.0 notification (no response expected)."""
    jsonrpc: JSONRPCVersion = JSONRPCVersion.V2_0
    method: str = Field(..., description="Notification method")
    params: Optional[Dict[str, Any]] = Field(None, description="Notification parameters")


# Error Codes (JSON-RPC 2.0 Standard + Custom)
class JSONRPCErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


class MCPProviderErrorCode(IntEnum):
    """MCP Provider Router specific error codes."""
    # Provider Management Errors (32000-32099)
    INVALID_PROVIDER = -32001
    NO_PROVIDER_AVAILABLE = -32002
    PROVIDER_UNAVAILABLE = -32003
    PROVIDER_TIMEOUT = -32004
    PROVIDER_RATE_LIMITED = -32005
    
    # Routing Errors (32100-32199)
    ROUTING_FAILED = -32100
    INVALID_STRATEGY = -32101
    FAILOVER_EXHAUSTED = -32102
    ROUTING_TIMEOUT = -32103
    ROUTING_LOOP_DETECTED = -32104
    
    # Cost and Budget Errors (32200-32299)
    BUDGET_EXCEEDED = -32200
    COST_ESTIMATION_FAILED = -32201
    INSUFFICIENT_CREDITS = -32202
    COST_LIMIT_VIOLATED = -32203
    
    # Health and Monitoring Errors (32300-32399)
    HEALTH_CHECK_FAILED = -32300
    PROVIDER_UNHEALTHY = -32301
    MONITORING_UNAVAILABLE = -32302
    HEALTH_THRESHOLD_EXCEEDED = -32303
    
    # Configuration Errors (32400-32499)
    CONFIGURATION_ERROR = -32400
    INVALID_FALLBACK_CHAIN = -32401
    PRIORITY_CONFLICT = -32402
    PARAMETER_VALIDATION_FAILED = -32403


# Event Types for Notifications
class ProviderEventType(str, Enum):
    """Provider router event types."""
    # Health Events
    PROVIDER_HEALTH_CHANGED = "provider_health_changed"
    HEALTH_CHECK_COMPLETED = "health_check_completed"
    PROVIDER_RECOVERED = "provider_recovered"
    PROVIDER_DEGRADED = "provider_degraded"
    
    # Routing Events
    ROUTING_DECISION_MADE = "routing_decision_made"
    FAILOVER_TRIGGERED = "failover_triggered"
    FAILOVER_COMPLETED = "failover_completed"
    PROVIDER_BYPASSED = "provider_bypassed"
    
    # Cost Events
    COST_THRESHOLD_WARNING = "cost_threshold_warning"
    COST_THRESHOLD_EXCEEDED = "cost_threshold_exceeded"
    BUDGET_ALERT = "budget_alert"
    COST_OPTIMIZATION_APPLIED = "cost_optimization_applied"
    
    # Configuration Events
    ROUTING_CONFIG_CHANGED = "routing_config_changed"
    FALLBACK_CHAIN_UPDATED = "fallback_chain_updated"
    PROVIDER_PRIORITY_CHANGED = "provider_priority_changed"
    
    # Request Events
    REQUEST_ROUTED = "request_routed"
    REQUEST_FAILED = "request_failed"
    REQUEST_RETRIED = "request_retried"
    REQUEST_COMPLETED = "request_completed"


# Method Names
class MCPProviderMethod(str, Enum):
    """MCP Provider Router method names."""
    # Core Routing Methods
    ROUTE_REQUEST = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/route_request"
    STREAM_REQUEST = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/stream_request"
    
    # Configuration Methods
    CONFIGURE_ROUTING = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/configure_routing"
    UPDATE_FALLBACK_CHAIN = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/update_fallback_chain"
    SET_PROVIDER_PRIORITY = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/set_provider_priority"
    
    # Monitoring Methods
    GET_PROVIDER_STATUS = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/get_provider_status"
    GET_ROUTING_STATS = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/get_routing_stats"
    GET_HEALTH_METRICS = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/get_health_metrics"
    
    # Testing Methods
    TEST_PROVIDER_CHAIN = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/test_provider_chain"
    TEST_PROVIDER_HEALTH = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/test_provider_health"
    
    # Control Methods
    FORCE_FAILOVER = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/force_failover"
    RESET_PROVIDER = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/reset_provider"
    PAUSE_PROVIDER = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/pause_provider"
    RESUME_PROVIDER = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/resume_provider"
    
    # Cost Management Methods
    SET_BUDGET_LIMITS = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/set_budget_limits"
    GET_COST_ANALYSIS = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/get_cost_analysis"
    OPTIMIZE_COSTS = f"{MCP_PROVIDER_ROUTER_NAMESPACE}/optimize_costs"


# Request Parameter Schemas
class RouteRequestParams(BaseModel):
    """Parameters for route_request method."""
    request_payload: Dict[str, Any] = Field(..., description="AI request to route")
    routing_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Routing configuration options"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "request_payload": {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello, world!"}],
                    "temperature": 0.7,
                    "max_tokens": 150
                },
                "routing_options": {
                    "strategy": "cost",
                    "fallback_enabled": True,
                    "max_retries": 3,
                    "timeout": 30.0,
                    "preferred_providers": ["openai", "anthropic"],
                    "cost_limit": 0.01
                }
            }
        }


class ConfigureRoutingParams(BaseModel):
    """Parameters for configure_routing method."""
    routing_config: Dict[str, Any] = Field(..., description="New routing configuration")
    
    class Config:
        schema_extra = {
            "example": {
                "routing_config": {
                    "default_strategy": "balanced",
                    "fallback_chain": ["anthropic", "openai", "google"],
                    "health_check_interval": 300,
                    "retry_config": {
                        "max_attempts": 3,
                        "backoff_multiplier": 2,
                        "max_backoff": 60
                    },
                    "cost_thresholds": {
                        "daily_warning": 50.0,
                        "daily_limit": 100.0,
                        "monthly_limit": 1000.0
                    },
                    "provider_priorities": {
                        "anthropic": 100,
                        "openai": 90,
                        "google": 80
                    }
                }
            }
        }


class TestProviderChainParams(BaseModel):
    """Parameters for test_provider_chain method."""
    test_payload: Dict[str, Any] = Field(..., description="Test request payload")
    test_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Test configuration options"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "test_payload": {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Test message"}],
                    "max_tokens": 50
                },
                "test_options": {
                    "include_costs": True,
                    "timeout_per_provider": 10.0,
                    "test_streaming": False
                }
            }
        }


class ForceFailoverParams(BaseModel):
    """Parameters for force_failover method."""
    failover_config: Dict[str, Any] = Field(..., description="Failover configuration")
    
    class Config:
        schema_extra = {
            "example": {
                "failover_config": {
                    "from_provider": "openai",
                    "to_provider": "anthropic",
                    "reason": "High error rate detected",
                    "duration": 3600,
                    "automatic_restore": True
                }
            }
        }


# Response Schemas
class ProviderInfo(BaseModel):
    """Provider information schema."""
    name: str = Field(..., description="Provider name")
    type: str = Field(..., description="Provider type")
    status: str = Field(..., description="Current status")
    health_score: float = Field(..., ge=0.0, le=1.0, description="Health score (0-1)")
    response_time_ms: float = Field(..., ge=0, description="Average response time")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate")
    current_load: int = Field(..., ge=0, description="Current request load")
    cost_per_1k_tokens: float = Field(..., ge=0, description="Cost per 1K tokens")
    models_available: List[str] = Field(default_factory=list, description="Available models")
    capabilities: List[str] = Field(default_factory=list, description="Provider capabilities")
    last_health_check: str = Field(..., description="Last health check timestamp")


class RoutingDecision(BaseModel):
    """Routing decision information."""
    selected_provider: str = Field(..., description="Selected provider")
    selected_model: str = Field(..., description="Selected model")
    strategy_used: str = Field(..., description="Routing strategy applied")
    decision_factors: Dict[str, Any] = Field(..., description="Factors influencing decision")
    alternatives_considered: List[str] = Field(..., description="Alternative providers considered")
    estimated_cost: float = Field(..., ge=0, description="Estimated request cost")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Decision confidence")
    fallback_available: bool = Field(..., description="Whether fallback options exist")


class RouteRequestResult(BaseModel):
    """Result for route_request method."""
    success: bool = Field(..., description="Whether routing was successful")
    routing_decision: RoutingDecision = Field(..., description="Routing decision details")
    response_data: Optional[Dict[str, Any]] = Field(None, description="AI provider response")
    execution_metrics: Dict[str, Any] = Field(..., description="Execution metrics")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error information if failed")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "routing_decision": {
                    "selected_provider": "anthropic",
                    "selected_model": "claude-3-sonnet",
                    "strategy_used": "cost",
                    "decision_factors": {
                        "cost_per_token": 0.0003,
                        "response_time_ms": 1200,
                        "success_rate": 0.99,
                        "current_load": 5
                    },
                    "alternatives_considered": ["openai", "google"],
                    "estimated_cost": 0.0045,
                    "confidence_score": 0.95,
                    "fallback_available": True
                },
                "response_data": {
                    "content": "Response from AI provider",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "cost": 0.0045
                },
                "execution_metrics": {
                    "total_time_ms": 1350,
                    "routing_time_ms": 15,
                    "provider_time_ms": 1200,
                    "retries": 0
                }
            }
        }


class ProviderStatusResult(BaseModel):
    """Result for get_provider_status method."""
    providers: List[ProviderInfo] = Field(..., description="Provider status information")
    overall_system_health: Dict[str, Any] = Field(..., description="Overall system health metrics")
    active_configuration: Dict[str, Any] = Field(..., description="Current routing configuration")
    last_updated: str = Field(..., description="Last update timestamp")


class RoutingStatsResult(BaseModel):
    """Result for get_routing_stats method."""
    request_metrics: Dict[str, Any] = Field(..., description="Request routing metrics")
    provider_performance: Dict[str, Any] = Field(..., description="Provider performance stats")
    cost_analytics: Dict[str, Any] = Field(..., description="Cost analysis data")
    health_trends: Dict[str, Any] = Field(..., description="Health trend analysis")
    optimization_suggestions: List[str] = Field(..., description="Optimization recommendations")


# Event Notification Schemas
class ProviderHealthChangedEvent(BaseModel):
    """Provider health status changed event."""
    event_type: str = ProviderEventType.PROVIDER_HEALTH_CHANGED
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    provider_name: str = Field(..., description="Provider that changed")
    previous_status: str = Field(..., description="Previous health status")
    current_status: str = Field(..., description="Current health status")
    health_metrics: Dict[str, Any] = Field(..., description="Current health metrics")
    impact_assessment: Dict[str, Any] = Field(..., description="Impact on routing")


class FailoverTriggeredEvent(BaseModel):
    """Failover triggered event."""
    event_type: str = ProviderEventType.FAILOVER_TRIGGERED
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    trigger_reason: str = Field(..., description="Reason for failover")
    from_provider: str = Field(..., description="Provider being failed over from")
    to_provider: str = Field(..., description="Provider being failed over to")
    request_id: Optional[str] = Field(None, description="Request that triggered failover")
    automatic: bool = Field(..., description="Whether failover was automatic")
    estimated_downtime: Optional[float] = Field(None, description="Estimated downtime in seconds")


class RoutingDecisionMadeEvent(BaseModel):
    """Routing decision made event."""
    event_type: str = ProviderEventType.ROUTING_DECISION_MADE
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: str = Field(..., description="Request identifier")
    routing_decision: RoutingDecision = Field(..., description="Decision details")
    context: Dict[str, Any] = Field(..., description="Decision context")


class CostThresholdExceededEvent(BaseModel):
    """Cost threshold exceeded event."""
    event_type: str = ProviderEventType.COST_THRESHOLD_EXCEEDED
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    threshold_type: str = Field(..., description="Type of threshold exceeded")
    current_usage: float = Field(..., description="Current usage amount")
    threshold_limit: float = Field(..., description="Threshold limit")
    provider: Optional[str] = Field(None, description="Specific provider if applicable")
    recommended_actions: List[str] = Field(..., description="Recommended mitigation actions")


# Error Response Helpers
class MCPProviderErrorDetails(BaseModel):
    """Detailed error information for MCP provider errors."""
    error_type: str = Field(..., description="Type of error")
    provider: Optional[str] = Field(None, description="Provider that caused the error")
    request_id: Optional[str] = Field(None, description="Request identifier")
    retry_after: Optional[int] = Field(None, description="Retry after seconds")
    suggestions: List[str] = Field(default_factory=list, description="Error resolution suggestions")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional error context")


def create_error_response(
    error_code: Union[JSONRPCErrorCode, MCPProviderErrorCode],
    message: str,
    request_id: Optional[Union[str, int]] = None,
    error_data: Optional[Dict[str, Any]] = None
) -> JSONRPCErrorResponse:
    """
    Create a standardized JSON-RPC 2.0 error response.
    
    Args:
        error_code: Error code from standard or custom enums
        message: Human-readable error message
        request_id: Request identifier (if available)
        error_data: Additional error context data
    
    Returns:
        JSON-RPC 2.0 error response
    """
    error = JSONRPCError(
        code=error_code.value,
        message=message,
        data=error_data
    )
    
    return JSONRPCErrorResponse(
        id=request_id,
        error=error
    )


def create_success_response(
    result: Any,
    request_id: Union[str, int]
) -> JSONRPCSuccessResponse:
    """
    Create a standardized JSON-RPC 2.0 success response.
    
    Args:
        result: Method execution result
        request_id: Request identifier
    
    Returns:
        JSON-RPC 2.0 success response
    """
    return JSONRPCSuccessResponse(
        id=request_id,
        result=result
    )


def create_notification(
    method: str,
    params: Optional[Dict[str, Any]] = None
) -> JSONRPCNotification:
    """
    Create a JSON-RPC 2.0 notification.
    
    Args:
        method: Notification method name
        params: Notification parameters
    
    Returns:
        JSON-RPC 2.0 notification
    """
    return JSONRPCNotification(
        method=method,
        params=params
    )


# Message Validation Functions
def validate_jsonrpc_message(message: Dict[str, Any]) -> bool:
    """
    Validate a JSON-RPC 2.0 message format.
    
    Args:
        message: Message to validate
    
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required fields
        if "jsonrpc" not in message:
            return False
        
        if message["jsonrpc"] != "2.0":
            return False
        
        # Check message type
        if "method" in message:
            # Request or notification
            if not isinstance(message["method"], str):
                return False
        elif "result" in message or "error" in message:
            # Response
            if "id" not in message:
                return False
        else:
            return False
        
        return True
        
    except Exception:
        return False


def serialize_message(message: BaseModel) -> str:
    """
    Serialize a message to JSON string.
    
    Args:
        message: Pydantic message model
    
    Returns:
        JSON string representation
    """
    return message.json(exclude_none=True, ensure_ascii=False)


def deserialize_message(message_str: str) -> Dict[str, Any]:
    """
    Deserialize a JSON message string.
    
    Args:
        message_str: JSON string to deserialize
    
    Returns:
        Deserialized message dictionary
    """
    try:
        return json.loads(message_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON message: {e}")


# Protocol Documentation
PROTOCOL_DOCUMENTATION = {
    "version": PROTOCOL_VERSION,
    "namespace": MCP_PROVIDER_ROUTER_NAMESPACE,
    "description": "MCP Protocol for AI Provider Router with Fallback",
    
    "methods": {
        method.value: {
            "description": f"Method: {method.name.lower().replace('_', ' ')}",
            "namespace": MCP_PROVIDER_ROUTER_NAMESPACE
        }
        for method in MCPProviderMethod
    },
    
    "events": {
        event.value: {
            "description": f"Event: {event.name.lower().replace('_', ' ')}",
            "notification": True
        }
        for event in ProviderEventType
    },
    
    "error_codes": {
        **{
            code.value: {
                "name": code.name,
                "description": f"Standard JSON-RPC error: {code.name.lower().replace('_', ' ')}"
            }
            for code in JSONRPCErrorCode
        },
        **{
            code.value: {
                "name": code.name,
                "description": f"Provider router error: {code.name.lower().replace('_', ' ')}"
            }
            for code in MCPProviderErrorCode
        }
    },
    
    "examples": {
        "route_request": {
            "request": {
                "jsonrpc": "2.0",
                "method": MCPProviderMethod.ROUTE_REQUEST.value,
                "params": {
                    "request_payload": {
                        "model": "gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": "Hello!"}]
                    },
                    "routing_options": {
                        "strategy": "cost",
                        "fallback_enabled": True
                    }
                },
                "id": "req-123"
            },
            "success_response": {
                "jsonrpc": "2.0",
                "result": {
                    "success": True,
                    "routing_decision": {
                        "selected_provider": "anthropic",
                        "selected_model": "claude-3-sonnet",
                        "strategy_used": "cost"
                    }
                },
                "id": "req-123"
            },
            "error_response": {
                "jsonrpc": "2.0",
                "error": {
                    "code": MCPProviderErrorCode.NO_PROVIDER_AVAILABLE.value,
                    "message": "No suitable providers available",
                    "data": {
                        "error_type": "routing_failure",
                        "suggestions": ["Check provider configurations", "Verify API keys"]
                    }
                },
                "id": "req-123"
            }
        },
        
        "health_change_notification": {
            "jsonrpc": "2.0",
            "method": "notification",
            "params": {
                "event_type": ProviderEventType.PROVIDER_HEALTH_CHANGED.value,
                "timestamp": "2024-01-01T12:00:00Z",
                "provider_name": "openai",
                "previous_status": "healthy",
                "current_status": "degraded"
            }
        }
    }
}


# Export key classes and functions
__all__ = [
    # Base JSON-RPC classes
    "JSONRPCRequest",
    "JSONRPCSuccessResponse", 
    "JSONRPCErrorResponse",
    "JSONRPCNotification",
    
    # Error codes
    "JSONRPCErrorCode",
    "MCPProviderErrorCode", 
    
    # Method and event enums
    "MCPProviderMethod",
    "ProviderEventType",
    
    # Request/Response schemas
    "RouteRequestParams",
    "ConfigureRoutingParams", 
    "TestProviderChainParams",
    "ForceFailoverParams",
    "RouteRequestResult",
    "ProviderStatusResult",
    "RoutingStatsResult",
    
    # Event schemas
    "ProviderHealthChangedEvent",
    "FailoverTriggeredEvent", 
    "RoutingDecisionMadeEvent",
    "CostThresholdExceededEvent",
    
    # Helper functions
    "create_error_response",
    "create_success_response",
    "create_notification",
    "validate_jsonrpc_message",
    "serialize_message",
    "deserialize_message",
    
    # Constants
    "PROTOCOL_VERSION",
    "MCP_PROVIDER_ROUTER_NAMESPACE",
    "PROTOCOL_DOCUMENTATION"
]