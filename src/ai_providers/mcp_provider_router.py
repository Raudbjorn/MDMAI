"""MCP Protocol Integration for Provider Router with Fallback System.

This module implements the complete MCP protocol integration for intelligent AI provider
routing, automatic failover, health monitoring, and cost optimization management.

Key Features:
- Intelligent request routing with fallback chains
- Real-time provider health monitoring
- Dynamic routing configuration
- Comprehensive error handling
- Cost optimization integration
- JSON-RPC 2.0 compliant messaging
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator
from structlog import get_logger

from .models import (
    AIRequest, 
    AIResponse,
    ProviderType,
    ProviderStatus,
    ProviderSelection,
    ProviderHealth,
    CostBudget,
    StreamingChunk
)
from .provider_manager import AIProviderManager
from .error_handler import AIProviderError, NoProviderAvailableError

logger = get_logger(__name__)


# MCP Protocol Models
class MCPRequestType(str, Enum):
    """MCP request types for provider routing."""
    ROUTE_REQUEST = "route_request"
    CONFIGURE_ROUTING = "configure_routing"
    GET_PROVIDER_STATUS = "get_provider_status"
    TEST_PROVIDER_CHAIN = "test_provider_chain"
    FORCE_FAILOVER = "force_failover"
    GET_ROUTING_STATS = "get_routing_stats"
    UPDATE_FALLBACK_CHAIN = "update_fallback_chain"
    SET_PROVIDER_PRIORITY = "set_provider_priority"
    MONITOR_HEALTH = "monitor_health"
    OPTIMIZE_COSTS = "optimize_costs"


class MCPMessageType(str, Enum):
    """MCP message types for provider events."""
    PROVIDER_HEALTH_CHANGE = "provider_health_change"
    FAILOVER_TRIGGERED = "failover_triggered"
    ROUTING_DECISION = "routing_decision"
    COST_THRESHOLD_EXCEEDED = "cost_threshold_exceeded"
    PROVIDER_ERROR = "provider_error"
    HEALTH_CHECK_COMPLETED = "health_check_completed"


class MCPErrorCode(int, Enum):
    """MCP error codes for provider routing."""
    INVALID_PROVIDER = -32001
    NO_PROVIDER_AVAILABLE = -32002
    ROUTING_FAILED = -32003
    FAILOVER_EXHAUSTED = -32004
    BUDGET_EXCEEDED = -32005
    PROVIDER_UNHEALTHY = -32006
    CONFIGURATION_ERROR = -32007
    HEALTH_CHECK_FAILED = -32008


# Request/Response Schemas
class RouteRequestParams(BaseModel):
    """Parameters for route_request tool."""
    request: Dict[str, Any] = Field(..., description="AI request to route")
    strategy: str = Field(default="cost", description="Routing strategy")
    fallback_enabled: bool = Field(default=True, description="Enable fallback on failure")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout: Optional[float] = Field(default=30.0, description="Request timeout in seconds")
    cost_limit: Optional[float] = Field(None, description="Maximum cost limit")
    preferred_providers: Optional[List[str]] = Field(None, description="Preferred provider order")
    exclude_providers: Optional[List[str]] = Field(None, description="Providers to exclude")

    @validator('strategy')
    def validate_strategy(cls, v):
        allowed = ["cost", "speed", "capability", "priority", "load_balanced", "failover", "random"]
        if v not in allowed:
            raise ValueError(f"Strategy must be one of: {allowed}")
        return v


class RouteRequestResponse(BaseModel):
    """Response from route_request tool."""
    success: bool
    provider: Optional[str]
    model: Optional[str]
    estimated_cost: Optional[float]
    routing_decision: Dict[str, Any]
    fallback_chain: List[str]
    response_data: Optional[Dict[str, Any]]
    execution_time_ms: float
    error: Optional[str] = None


class ConfigureRoutingParams(BaseModel):
    """Parameters for configure_routing tool."""
    default_strategy: str = Field(..., description="Default routing strategy")
    fallback_chain: List[str] = Field(..., description="Provider fallback order")
    health_check_interval: int = Field(default=300, description="Health check interval in seconds")
    retry_config: Dict[str, Any] = Field(default_factory=dict, description="Retry configuration")
    cost_thresholds: Dict[str, float] = Field(default_factory=dict, description="Cost alert thresholds")
    provider_priorities: Dict[str, int] = Field(default_factory=dict, description="Provider priorities")


class ProviderStatusResponse(BaseModel):
    """Response from get_provider_status tool."""
    provider_statuses: Dict[str, Dict[str, Any]]
    overall_health: Dict[str, Any]
    active_requests: int
    total_requests_today: int
    cost_usage_today: float
    last_updated: str


class TestProviderChainParams(BaseModel):
    """Parameters for test_provider_chain tool."""
    test_request: Dict[str, Any] = Field(..., description="Test request to send")
    include_costs: bool = Field(default=True, description="Include cost estimates")
    timeout_per_provider: float = Field(default=10.0, description="Timeout per provider test")


class TestProviderChainResponse(BaseModel):
    """Response from test_provider_chain tool."""
    chain_test_results: List[Dict[str, Any]]
    successful_providers: List[str]
    failed_providers: List[str]
    total_test_time_ms: float
    recommendations: List[str]


class ForceFailoverParams(BaseModel):
    """Parameters for force_failover tool."""
    from_provider: str = Field(..., description="Provider to fail over from")
    to_provider: Optional[str] = Field(None, description="Specific provider to fail over to")
    reason: str = Field(..., description="Reason for manual failover")
    duration: Optional[int] = Field(None, description="Failover duration in seconds")


# Event Schemas
class ProviderHealthChangeEvent(BaseModel):
    """Provider health change event."""
    event_type: str = MCPMessageType.PROVIDER_HEALTH_CHANGE
    timestamp: str
    provider: str
    old_status: str
    new_status: str
    health_metrics: Dict[str, Any]
    uptime_percentage: float


class FailoverTriggeredEvent(BaseModel):
    """Failover triggered event."""
    event_type: str = MCPMessageType.FAILOVER_TRIGGERED
    timestamp: str
    from_provider: str
    to_provider: str
    reason: str
    request_id: str
    automatic: bool


class RoutingDecisionEvent(BaseModel):
    """Routing decision event."""
    event_type: str = MCPMessageType.ROUTING_DECISION
    timestamp: str
    request_id: str
    selected_provider: str
    strategy_used: str
    decision_factors: Dict[str, Any]
    alternatives_considered: List[str]
    execution_time_ms: float


class CostThresholdExceededEvent(BaseModel):
    """Cost threshold exceeded event."""
    event_type: str = MCPMessageType.COST_THRESHOLD_EXCEEDED
    timestamp: str
    threshold_type: str
    current_usage: float
    threshold_limit: float
    provider: Optional[str]
    recommendations: List[str]


# Main MCP Provider Router Class
class MCPProviderRouter:
    """MCP Protocol integration for Provider Router with comprehensive failover support."""
    
    def __init__(self, provider_manager: AIProviderManager):
        self.provider_manager = provider_manager
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.event_subscribers: List[callable] = []
        self.routing_config: Dict[str, Any] = {
            "default_strategy": "cost",
            "fallback_chain": [],
            "health_check_interval": 300,
            "retry_config": {"max_attempts": 3, "backoff_multiplier": 2},
            "cost_thresholds": {"daily": 100.0, "monthly": 1000.0},
            "provider_priorities": {}
        }
        self.health_monitor_task: Optional[asyncio.Task] = None
        self._request_metrics: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize the MCP provider router."""
        logger.info("Initializing MCP Provider Router")
        
        # Start health monitoring
        await self._start_health_monitoring()
        
        # Initialize metrics
        self._request_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "provider_usage": {},
            "cost_savings": 0.0
        }
        
        logger.info("MCP Provider Router initialized successfully")

    async def shutdown(self) -> None:
        """Shutdown the MCP provider router."""
        logger.info("Shutting down MCP Provider Router")
        
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("MCP Provider Router shut down")

    # MCP Tool Implementations
    async def route_request(self, params: RouteRequestParams) -> RouteRequestResponse:
        """
        Intelligent request routing with fallback support.
        
        This tool routes AI requests to the optimal provider based on the specified
        strategy, with automatic failover if the primary provider fails.
        """
        start_time = datetime.now()
        request_id = str(uuid.uuid4())
        
        try:
            # Convert params to AI request
            ai_request = AIRequest(
                request_id=request_id,
                model=params.request.get("model", "gpt-3.5-turbo"),
                messages=params.request.get("messages", []),
                temperature=params.request.get("temperature"),
                max_tokens=params.request.get("max_tokens"),
                stream=params.request.get("stream", False),
                tools=params.request.get("tools"),
                budget_limit=params.cost_limit
            )
            
            # Create provider selection criteria
            selection = ProviderSelection(
                preferred_providers=[ProviderType(p) for p in params.preferred_providers] if params.preferred_providers else None,
                exclude_providers=[ProviderType(p) for p in params.exclude_providers] if params.exclude_providers else None,
                require_streaming=ai_request.stream,
                require_tools=bool(ai_request.tools)
            )
            
            # Track active request
            self.active_requests[request_id] = {
                "start_time": start_time,
                "strategy": params.strategy,
                "fallback_enabled": params.fallback_enabled,
                "max_retries": params.max_retries
            }
            
            # Attempt request with fallback
            response, routing_info = await self._execute_with_fallback(
                ai_request, selection, params.strategy, 
                params.fallback_enabled, params.max_retries, params.timeout
            )
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Emit routing decision event
            await self._emit_event(RoutingDecisionEvent(
                timestamp=datetime.now().isoformat(),
                request_id=request_id,
                selected_provider=routing_info["provider"],
                strategy_used=params.strategy,
                decision_factors=routing_info["decision_factors"],
                alternatives_considered=routing_info["alternatives"],
                execution_time_ms=execution_time
            ))
            
            # Update metrics
            self._update_request_metrics(True, execution_time, routing_info["provider"])
            
            return RouteRequestResponse(
                success=True,
                provider=routing_info["provider"],
                model=routing_info["model"],
                estimated_cost=routing_info["cost"],
                routing_decision=routing_info["decision_factors"],
                fallback_chain=routing_info["fallback_chain"],
                response_data=response.model_dump() if response else None,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_request_metrics(False, execution_time)
            
            logger.error("Request routing failed", error=str(e), request_id=request_id)
            
            return RouteRequestResponse(
                success=False,
                provider=None,
                model=None,
                estimated_cost=None,
                routing_decision={},
                fallback_chain=[],
                response_data=None,
                execution_time_ms=execution_time,
                error=str(e)
            )
        finally:
            # Clean up active request tracking
            self.active_requests.pop(request_id, None)

    async def configure_routing(self, params: ConfigureRoutingParams) -> Dict[str, Any]:
        """
        Configure provider routing parameters and fallback chains.
        
        This tool allows dynamic configuration of routing strategies, failover chains,
        health monitoring, and cost thresholds.
        """
        try:
            # Validate fallback chain providers
            valid_providers = [p.value for p in ProviderType]
            for provider in params.fallback_chain:
                if provider not in valid_providers:
                    raise ValueError(f"Invalid provider in fallback chain: {provider}")
            
            # Update routing configuration
            self.routing_config.update({
                "default_strategy": params.default_strategy,
                "fallback_chain": params.fallback_chain,
                "health_check_interval": params.health_check_interval,
                "retry_config": params.retry_config,
                "cost_thresholds": params.cost_thresholds,
                "provider_priorities": params.provider_priorities
            })
            
            # Apply provider priorities
            for provider_name, priority in params.provider_priorities.items():
                try:
                    provider_type = ProviderType(provider_name)
                    self.provider_manager.configure_provider_priority(provider_type, priority)
                except ValueError:
                    logger.warning(f"Invalid provider type: {provider_name}")
            
            # Restart health monitoring with new interval
            if self.health_monitor_task:
                self.health_monitor_task.cancel()
            await self._start_health_monitoring()
            
            logger.info("Routing configuration updated", config=self.routing_config)
            
            return {
                "success": True,
                "message": "Routing configuration updated successfully",
                "active_config": self.routing_config
            }
            
        except Exception as e:
            logger.error("Failed to configure routing", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "active_config": self.routing_config
            }

    async def get_provider_status(self) -> ProviderStatusResponse:
        """
        Get real-time status of all AI providers.
        
        This tool provides comprehensive health information, performance metrics,
        and availability status for all registered providers.
        """
        try:
            # Get provider health information
            health_results = await self.provider_manager.registry.perform_health_check()
            
            # Build provider status dictionary
            provider_statuses = {}
            total_uptime = 0
            healthy_count = 0
            
            for provider_type, health in health_results.items():
                status_info = {
                    "name": provider_type.value,
                    "status": health.status.value,
                    "available": health.status == ProviderStatus.HEALTHY,
                    "uptime_percentage": health.uptime_percentage,
                    "last_success": health.last_success.isoformat() if health.last_success else None,
                    "last_error": health.last_error.isoformat() if health.last_error else None,
                    "error_count": health.error_count,
                    "response_time_ms": health.response_time_ms,
                    "models_available": len(self.provider_manager.registry.get_provider(provider_type).models) if self.provider_manager.registry.get_provider(provider_type) else 0,
                    "current_load": self.provider_manager.registry._load_tracker.get(provider_type, 0)
                }
                
                provider_statuses[provider_type.value] = status_info
                total_uptime += health.uptime_percentage
                if health.status == ProviderStatus.HEALTHY:
                    healthy_count += 1
            
            # Calculate overall health
            average_uptime = total_uptime / len(health_results) if health_results else 0
            overall_health = {
                "healthy_providers": healthy_count,
                "total_providers": len(health_results),
                "average_uptime": average_uptime,
                "system_status": "healthy" if healthy_count > 0 else "degraded"
            }
            
            # Get usage statistics
            usage_stats = self.provider_manager.usage_tracker.get_usage_stats()
            
            return ProviderStatusResponse(
                provider_statuses=provider_statuses,
                overall_health=overall_health,
                active_requests=len(self.active_requests),
                total_requests_today=usage_stats.get("total_requests", 0),
                cost_usage_today=usage_stats.get("total_cost", 0.0),
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error("Failed to get provider status", error=str(e))
            raise

    async def test_provider_chain(self, params: TestProviderChainParams) -> TestProviderChainResponse:
        """
        Test the entire provider fallback chain with a sample request.
        
        This tool validates the health and response capabilities of all providers
        in the fallback chain using a test request.
        """
        start_time = datetime.now()
        test_results = []
        successful_providers = []
        failed_providers = []
        
        try:
            # Create test AI request
            test_request = AIRequest(
                request_id=f"test-{uuid.uuid4()}",
                model=params.test_request.get("model", "gpt-3.5-turbo"),
                messages=params.test_request.get("messages", [{"role": "user", "content": "Hello, this is a test."}]),
                max_tokens=50  # Keep test small
            )
            
            # Get fallback chain or all available providers
            providers_to_test = (
                [ProviderType(p) for p in self.routing_config["fallback_chain"]] 
                if self.routing_config["fallback_chain"]
                else list(self.provider_manager.registry._providers.keys())
            )
            
            # Test each provider
            for provider_type in providers_to_test:
                provider_test_start = datetime.now()
                test_result = {
                    "provider": provider_type.value,
                    "success": False,
                    "response_time_ms": 0,
                    "error": None,
                    "cost_estimate": 0.0,
                    "model_available": False
                }
                
                try:
                    provider = self.provider_manager.registry.get_provider(provider_type)
                    if not provider:
                        test_result["error"] = "Provider not registered"
                        failed_providers.append(provider_type.value)
                        test_results.append(test_result)
                        continue
                    
                    # Check if model is available
                    test_result["model_available"] = test_request.model in provider.models
                    
                    # Estimate cost if requested
                    if params.include_costs:
                        try:
                            test_result["cost_estimate"] = self.provider_manager.cost_optimizer.estimate_request_cost(
                                test_request, provider_type
                            )
                        except Exception as e:
                            logger.warning(f"Cost estimation failed for {provider_type.value}: {e}")
                    
                    # Test provider with timeout
                    try:
                        response = await asyncio.wait_for(
                            provider.generate_response(test_request),
                            timeout=params.timeout_per_provider
                        )
                        
                        response_time = (datetime.now() - provider_test_start).total_seconds() * 1000
                        test_result.update({
                            "success": True,
                            "response_time_ms": response_time,
                            "response_length": len(response.content) if response.content else 0
                        })
                        successful_providers.append(provider_type.value)
                        
                    except asyncio.TimeoutError:
                        test_result["error"] = f"Timeout after {params.timeout_per_provider}s"
                        failed_providers.append(provider_type.value)
                    except Exception as e:
                        test_result["error"] = str(e)
                        failed_providers.append(provider_type.value)
                
                except Exception as e:
                    test_result["error"] = f"Provider test setup failed: {str(e)}"
                    failed_providers.append(provider_type.value)
                
                test_results.append(test_result)
            
            # Generate recommendations
            recommendations = self._generate_chain_recommendations(test_results)
            
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return TestProviderChainResponse(
                chain_test_results=test_results,
                successful_providers=successful_providers,
                failed_providers=failed_providers,
                total_test_time_ms=total_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error("Provider chain test failed", error=str(e))
            raise

    async def force_failover(self, params: ForceFailoverParams) -> Dict[str, Any]:
        """
        Force manual failover from one provider to another.
        
        This tool allows manual intervention to redirect traffic from a problematic
        provider to an alternative, with optional time-limited failover.
        """
        try:
            from_provider_type = ProviderType(params.from_provider)
            
            # Validate source provider exists
            if from_provider_type not in self.provider_manager.registry._providers:
                return {
                    "success": False,
                    "error": f"Source provider {params.from_provider} not found"
                }
            
            # Determine target provider
            if params.to_provider:
                to_provider_type = ProviderType(params.to_provider)
                if to_provider_type not in self.provider_manager.registry._providers:
                    return {
                        "success": False,
                        "error": f"Target provider {params.to_provider} not found"
                    }
            else:
                # Find next available provider in fallback chain
                available_providers = self.provider_manager.registry.get_available_providers()
                available_types = [p.provider_type for p in available_providers if p.provider_type != from_provider_type]
                
                if not available_types:
                    return {
                        "success": False,
                        "error": "No alternative providers available for failover"
                    }
                
                to_provider_type = available_types[0]  # Use first available
            
            # Temporarily adjust provider priority to force failover
            original_priority = self.provider_manager.registry._provider_priorities.get(from_provider_type, 0)
            target_priority = self.provider_manager.registry._provider_priorities.get(to_provider_type, 0)
            
            # Set source provider to lowest priority and target to highest
            self.provider_manager.configure_provider_priority(from_provider_type, -100)
            self.provider_manager.configure_provider_priority(to_provider_type, 100)
            
            # Emit failover event
            await self._emit_event(FailoverTriggeredEvent(
                timestamp=datetime.now().isoformat(),
                from_provider=params.from_provider,
                to_provider=to_provider_type.value,
                reason=params.reason,
                request_id="manual-failover",
                automatic=False
            ))
            
            # Schedule priority restoration if duration specified
            if params.duration:
                async def restore_priorities():
                    await asyncio.sleep(params.duration)
                    self.provider_manager.configure_provider_priority(from_provider_type, original_priority)
                    self.provider_manager.configure_provider_priority(to_provider_type, target_priority)
                    logger.info(f"Restored original provider priorities after {params.duration}s")
                
                asyncio.create_task(restore_priorities())
            
            logger.info(
                "Manual failover executed",
                from_provider=params.from_provider,
                to_provider=to_provider_type.value,
                reason=params.reason,
                duration=params.duration
            )
            
            return {
                "success": True,
                "message": f"Failover from {params.from_provider} to {to_provider_type.value} completed",
                "from_provider": params.from_provider,
                "to_provider": to_provider_type.value,
                "duration": params.duration,
                "restore_scheduled": params.duration is not None
            }
            
        except ValueError as e:
            return {
                "success": False,
                "error": f"Invalid provider name: {str(e)}"
            }
        except Exception as e:
            logger.error("Manual failover failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }

    # Internal Implementation Methods
    async def _execute_with_fallback(
        self,
        request: AIRequest,
        selection: Optional[ProviderSelection],
        strategy: str,
        fallback_enabled: bool,
        max_retries: int,
        timeout: Optional[float]
    ) -> Tuple[Optional[AIResponse], Dict[str, Any]]:
        """Execute request with fallback support."""
        attempts = 0
        last_error = None
        fallback_chain = []
        decision_factors = {}
        alternatives = []
        
        while attempts < max_retries:
            try:
                # Select provider
                provider = await self.provider_manager.registry.select_provider(
                    request, selection, strategy
                )
                
                if not provider:
                    if not fallback_enabled or attempts >= max_retries - 1:
                        raise NoProviderAvailableError("No suitable providers available")
                    
                    # Try with relaxed selection criteria
                    selection = ProviderSelection() if selection else None
                    attempts += 1
                    continue
                
                fallback_chain.append(provider.provider_type.value)
                alternatives.append(provider.provider_type.value)
                
                # Record decision factors
                decision_factors = {
                    "strategy": strategy,
                    "attempt": attempts + 1,
                    "provider_health": provider.health.status.value,
                    "provider_load": self.provider_manager.registry._load_tracker.get(provider.provider_type, 0),
                    "estimated_cost": self.provider_manager.cost_optimizer.estimate_request_cost(
                        request, provider.provider_type
                    )
                }
                
                # Execute request with timeout
                if timeout:
                    response = await asyncio.wait_for(
                        self.provider_manager.process_request(
                            request, selection=selection, strategy=strategy
                        ),
                        timeout=timeout
                    )
                else:
                    response = await self.provider_manager.process_request(
                        request, selection=selection, strategy=strategy
                    )
                
                routing_info = {
                    "provider": provider.provider_type.value,
                    "model": request.model,
                    "cost": response.cost,
                    "fallback_chain": fallback_chain,
                    "decision_factors": decision_factors,
                    "alternatives": alternatives
                }
                
                return response, routing_info
                
            except Exception as e:
                last_error = e
                attempts += 1
                
                logger.warning(
                    "Request attempt failed",
                    attempt=attempts,
                    error=str(e),
                    fallback_enabled=fallback_enabled,
                    max_retries=max_retries
                )
                
                if not fallback_enabled or attempts >= max_retries:
                    break
                
                # If specific provider failed, exclude it from next attempt
                if selection and hasattr(e, 'provider_type'):
                    if not selection.exclude_providers:
                        selection.exclude_providers = []
                    selection.exclude_providers.append(e.provider_type)
                
                # Wait before retry
                await asyncio.sleep(min(2 ** attempts, 10))
        
        # All attempts failed
        routing_info = {
            "provider": None,
            "model": None,
            "cost": 0.0,
            "fallback_chain": fallback_chain,
            "decision_factors": decision_factors,
            "alternatives": alternatives
        }
        
        raise last_error or Exception("All failover attempts exhausted")

    async def _start_health_monitoring(self) -> None:
        """Start health monitoring task."""
        interval = self.routing_config.get("health_check_interval", 300)
        
        async def health_monitor():
            while True:
                try:
                    await asyncio.sleep(interval)
                    health_results = await self.provider_manager.registry.perform_health_check()
                    
                    # Check for status changes and emit events
                    for provider_type, health in health_results.items():
                        # This is a simplified example - in practice, you'd want to track
                        # previous states to detect changes
                        await self._emit_event(ProviderHealthChangeEvent(
                            timestamp=datetime.now().isoformat(),
                            provider=provider_type.value,
                            old_status="unknown",  # Would be tracked from previous check
                            new_status=health.status.value,
                            health_metrics={
                                "uptime_percentage": health.uptime_percentage,
                                "response_time_ms": health.response_time_ms,
                                "error_count": health.error_count
                            },
                            uptime_percentage=health.uptime_percentage
                        ))
                    
                    # Emit health check completed event
                    await self._emit_event({
                        "event_type": MCPMessageType.HEALTH_CHECK_COMPLETED,
                        "timestamp": datetime.now().isoformat(),
                        "providers_checked": len(health_results),
                        "healthy_providers": sum(1 for h in health_results.values() if h.status == ProviderStatus.HEALTHY)
                    })
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Health monitoring error", error=str(e))
        
        self.health_monitor_task = asyncio.create_task(health_monitor())

    async def _emit_event(self, event: Union[BaseModel, Dict[str, Any]]) -> None:
        """Emit an event to all subscribers."""
        try:
            if isinstance(event, BaseModel):
                event_data = event.model_dump()
            else:
                event_data = event
            
            for subscriber in self.event_subscribers:
                try:
                    if asyncio.iscoroutinefunction(subscriber):
                        await subscriber(event_data)
                    else:
                        subscriber(event_data)
                except Exception as e:
                    logger.error("Event subscriber error", error=str(e))
        
        except Exception as e:
            logger.error("Event emission failed", error=str(e))

    def _update_request_metrics(self, success: bool, execution_time: float, provider: Optional[str] = None) -> None:
        """Update request metrics."""
        self._request_metrics["total_requests"] += 1
        
        if success:
            self._request_metrics["successful_requests"] += 1
        else:
            self._request_metrics["failed_requests"] += 1
        
        # Update average response time
        total = self._request_metrics["total_requests"]
        current_avg = self._request_metrics["average_response_time"]
        self._request_metrics["average_response_time"] = (
            (current_avg * (total - 1) + execution_time) / total
        )
        
        # Update provider usage
        if provider:
            if provider not in self._request_metrics["provider_usage"]:
                self._request_metrics["provider_usage"][provider] = 0
            self._request_metrics["provider_usage"][provider] += 1

    def _generate_chain_recommendations(self, test_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on chain test results."""
        recommendations = []
        
        successful_results = [r for r in test_results if r["success"]]
        failed_results = [r for r in test_results if not r["success"]]
        
        if not successful_results:
            recommendations.append("⚠️  No providers passed the test - check provider configurations")
            return recommendations
        
        # Sort by response time for performance recommendations
        successful_results.sort(key=lambda x: x["response_time_ms"])
        
        fastest = successful_results[0]
        recommendations.append(f"🚀 Fastest provider: {fastest['provider']} ({fastest['response_time_ms']:.1f}ms)")
        
        if len(successful_results) > 1:
            slowest = successful_results[-1]
            if slowest["response_time_ms"] > fastest["response_time_ms"] * 2:
                recommendations.append(
                    f"⚠️  Consider deprioritizing {slowest['provider']} due to slow response "
                    f"({slowest['response_time_ms']:.1f}ms vs {fastest['response_time_ms']:.1f}ms)"
                )
        
        # Cost recommendations
        cost_results = [r for r in successful_results if r.get("cost_estimate", 0) > 0]
        if cost_results:
            cheapest = min(cost_results, key=lambda x: x["cost_estimate"])
            recommendations.append(f"💰 Most cost-effective: {cheapest['provider']} (${cheapest['cost_estimate']:.4f})")
        
        # Failure recommendations
        if failed_results:
            for result in failed_results:
                recommendations.append(f"❌ {result['provider']}: {result['error']}")
        
        return recommendations

    def subscribe_to_events(self, callback: callable) -> None:
        """Subscribe to routing events."""
        self.event_subscribers.append(callback)

    def unsubscribe_from_events(self, callback: callable) -> None:
        """Unsubscribe from routing events."""
        if callback in self.event_subscribers:
            self.event_subscribers.remove(callback)

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        return {
            "metrics": self._request_metrics,
            "active_requests": len(self.active_requests),
            "routing_config": self.routing_config,
            "provider_registry_stats": self.provider_manager.registry.get_registry_stats(),
            "usage_stats": self.provider_manager.usage_tracker.get_usage_stats(),
            "cost_optimizer_stats": self.provider_manager.cost_optimizer.get_budget_alerts()
        }


# JSON-RPC 2.0 Error Response Helpers
def create_mcp_error(code: MCPErrorCode, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a JSON-RPC 2.0 error response."""
    error_response = {
        "jsonrpc": "2.0",
        "error": {
            "code": code.value,
            "message": message
        },
        "id": None
    }
    
    if data:
        error_response["error"]["data"] = data
    
    return error_response


def create_mcp_success(result: Any, request_id: str) -> Dict[str, Any]:
    """Create a JSON-RPC 2.0 success response."""
    return {
        "jsonrpc": "2.0",
        "result": result,
        "id": request_id
    }


# FastMCP Integration Helper
def register_provider_router_tools(mcp_server, provider_manager: AIProviderManager) -> MCPProviderRouter:
    """
    Register all provider router tools with a FastMCP server instance.
    
    Args:
        mcp_server: FastMCP server instance
        provider_manager: AI Provider Manager instance
    
    Returns:
        MCPProviderRouter instance for event subscription
    """
    router = MCPProviderRouter(provider_manager)
    
    @mcp_server.tool()
    async def route_request(
        request: Dict[str, Any],
        strategy: str = "cost",
        fallback_enabled: bool = True,
        max_retries: int = 3,
        timeout: float = 30.0,
        cost_limit: Optional[float] = None,
        preferred_providers: Optional[List[str]] = None,
        exclude_providers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Route AI requests to optimal providers with intelligent fallback.
        
        This tool automatically selects the best AI provider for each request based on
        cost, performance, capabilities, and availability. Supports automatic failover
        when providers fail or become unavailable.
        """
        params = RouteRequestParams(
            request=request,
            strategy=strategy,
            fallback_enabled=fallback_enabled,
            max_retries=max_retries,
            timeout=timeout,
            cost_limit=cost_limit,
            preferred_providers=preferred_providers,
            exclude_providers=exclude_providers
        )
        
        response = await router.route_request(params)
        return response.model_dump()
    
    @mcp_server.tool()
    async def configure_routing(
        default_strategy: str,
        fallback_chain: List[str],
        health_check_interval: int = 300,
        retry_config: Dict[str, Any] = None,
        cost_thresholds: Dict[str, float] = None,
        provider_priorities: Dict[str, int] = None
    ) -> Dict[str, Any]:
        """
        Configure provider routing strategies and fallback behavior.
        
        Allows dynamic configuration of how requests are routed between providers,
        including fallback chains, health monitoring intervals, and cost thresholds.
        """
        params = ConfigureRoutingParams(
            default_strategy=default_strategy,
            fallback_chain=fallback_chain,
            health_check_interval=health_check_interval,
            retry_config=retry_config or {},
            cost_thresholds=cost_thresholds or {},
            provider_priorities=provider_priorities or {}
        )
        
        return await router.configure_routing(params)
    
    @mcp_server.tool()
    async def get_provider_status() -> Dict[str, Any]:
        """
        Get real-time status and health information for all AI providers.
        
        Returns comprehensive health metrics, availability status, performance data,
        and current load for all registered providers.
        """
        response = await router.get_provider_status()
        return response.model_dump()
    
    @mcp_server.tool()
    async def test_provider_chain(
        test_request: Dict[str, Any],
        include_costs: bool = True,
        timeout_per_provider: float = 10.0
    ) -> Dict[str, Any]:
        """
        Test the complete provider fallback chain with a sample request.
        
        Validates that all providers in the fallback chain are working correctly
        and provides performance and cost metrics for optimization.
        """
        params = TestProviderChainParams(
            test_request=test_request,
            include_costs=include_costs,
            timeout_per_provider=timeout_per_provider
        )
        
        response = await router.test_provider_chain(params)
        return response.model_dump()
    
    @mcp_server.tool()
    async def force_failover(
        from_provider: str,
        to_provider: Optional[str] = None,
        reason: str = "Manual intervention",
        duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Force manual failover between providers.
        
        Allows manual intervention to redirect traffic from problematic providers
        to alternatives. Can be time-limited for temporary failover scenarios.
        """
        params = ForceFailoverParams(
            from_provider=from_provider,
            to_provider=to_provider,
            reason=reason,
            duration=duration
        )
        
        return await router.force_failover(params)
    
    @mcp_server.tool()
    async def get_routing_stats() -> Dict[str, Any]:
        """
        Get comprehensive routing and provider usage statistics.
        
        Returns detailed metrics about request routing patterns, provider performance,
        cost optimization, and system health for monitoring and optimization.
        """
        return router.get_routing_stats()
    
    return router