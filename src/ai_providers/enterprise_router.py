"""
Enterprise-Grade AI Provider Router with Advanced Fallback
Task 25.3: Complete Integration Architecture
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from structlog import get_logger

from .models import (
    AIRequest,
    AIResponse,
    ProviderType,
    ProviderSelection,
    StreamingChunk,
)
from .abstract_provider import AbstractProvider
from .intelligent_router import IntelligentRouter, SelectionStrategy, SelectionCriteria
from .fallback_manager import FallbackManager, FallbackTier
from .load_balancer import LoadBalancer, LoadBalancingAlgorithm
from .model_router import ModelRouter
from .advanced_cost_optimizer import AdvancedCostOptimizer, OptimizationStrategy
from .health_monitor import HealthMonitor
from .sla_monitor import SLAMonitor

logger = get_logger(__name__)


@dataclass
class RoutingDecision:
    """Complete routing decision with full context."""
    
    selected_provider: AbstractProvider
    selected_model: str
    selection_strategy: SelectionStrategy
    confidence_score: float
    
    # Decision context
    routing_reason: str
    alternatives_considered: int
    fallback_tier: FallbackTier
    cost_estimate: float
    latency_estimate: float
    
    # Optimization details
    cost_optimization: Dict[str, Any] = field(default_factory=dict)
    load_balancing: Dict[str, Any] = field(default_factory=dict)
    sla_compliance: Dict[str, Any] = field(default_factory=dict)
    
    timestamp: datetime = field(default_factory=datetime.now)


class EnterpriseRouter:
    """
    Enterprise-grade AI provider router with comprehensive routing capabilities.
    
    Architecture Integration:
    - Intelligent Provider Selection Algorithm
    - Multi-tier Fallback Strategy with Circuit Breakers  
    - Advanced Load Balancing and Performance Optimization
    - Model-specific Routing with Capability Matching
    - Real-time Cost Optimization and Budget Enforcement
    - Comprehensive SLA Monitoring and Alerting
    
    Performance Targets:
    - P95 Latency: < 200ms routing decision
    - Throughput: > 1000 requests/second
    - Availability: 99.99% uptime
    - Cost Optimization: 15-30% cost reduction
    """
    
    def __init__(
        self,
        providers: List[AbstractProvider],
        enable_all_features: bool = True,
    ):
        """Initialize the enterprise router with all components."""
        self.providers = {provider.provider_type: provider for provider in providers}
        self.enable_all_features = enable_all_features
        
        # Core routing components
        self.health_monitor = HealthMonitor()
        self.intelligent_router = IntelligentRouter(self.health_monitor)
        self.model_router = ModelRouter()
        self.load_balancer = LoadBalancer(
            config=self._create_load_balancer_config(),
            health_monitor=self.health_monitor,
        )
        
        # Advanced features
        if enable_all_features:
            self.cost_optimizer = AdvancedCostOptimizer(
                optimization_strategy=OptimizationStrategy.COST_QUALITY_BALANCE
            )
            self.fallback_manager = FallbackManager(
                self.intelligent_router,
                self.health_monitor,
            )
            self.sla_monitor = SLAMonitor(self.health_monitor)
        else:
            self.cost_optimizer = None
            self.fallback_manager = None
            self.sla_monitor = None
        
        # Routing statistics
        self.routing_stats = {
            "total_requests": 0,
            "successful_routes": 0,
            "fallback_used": 0,
            "cost_optimizations": 0,
            "sla_violations": 0,
        }
        
        # Performance metrics
        self.performance_targets = {
            "max_routing_latency_ms": 200,
            "min_success_rate": 0.999,
            "max_cost_per_request": 1.0,
            "min_throughput_rps": 1000,
        }
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all router components."""
        if self._initialized:
            return
        
        logger.info("Initializing Enterprise Router")
        start_time = datetime.now()
        
        try:
            # Initialize core components
            await self.health_monitor.start()
            await self.load_balancer.start()
            
            # Register providers with all components
            for provider in self.providers.values():
                self.health_monitor.register_provider(
                    provider.provider_type,
                    provider.health_check
                )
                self.load_balancer.register_provider(provider)
                
                # Register models with model router
                for model_id, model_spec in provider.models.items():
                    self.model_router.register_model(
                        provider.provider_type,
                        model_spec
                    )
            
            # Initialize advanced features
            if self.enable_all_features:
                await self.cost_optimizer.start()
                await self.sla_monitor.start()
                
                # Configure SLA monitoring callbacks
                self.sla_monitor.add_alert_callback(self._handle_sla_alert)
            
            self._initialized = True
            
            initialization_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(
                "Enterprise Router initialized successfully",
                providers=len(self.providers),
                initialization_time_ms=initialization_time,
                features_enabled=self.enable_all_features,
            )
            
        except Exception as e:
            logger.error("Failed to initialize Enterprise Router", error=str(e))
            raise
    
    async def shutdown(self) -> None:
        """Shutdown all router components."""
        if not self._initialized:
            return
        
        logger.info("Shutting down Enterprise Router")
        
        try:
            # Shutdown in reverse order
            if self.enable_all_features:
                await self.sla_monitor.stop()
                await self.cost_optimizer.stop()
            
            await self.load_balancer.stop()
            await self.health_monitor.stop()
            
            self._initialized = False
            logger.info("Enterprise Router shut down successfully")
            
        except Exception as e:
            logger.error("Error during Enterprise Router shutdown", error=str(e))
    
    async def route_request(
        self,
        request: AIRequest,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> Tuple[RoutingDecision, AIResponse]:
        """
        Route a request through the complete enterprise routing pipeline.
        
        Args:
            request: The AI request to route
            preferences: Optional routing preferences
            
        Returns:
            Tuple of (routing_decision, response)
        """
        if not self._initialized:
            raise RuntimeError("Enterprise Router not initialized")
        
        routing_start = datetime.now()
        self.routing_stats["total_requests"] += 1
        
        logger.info(
            "Starting enterprise routing",
            request_id=request.request_id,
            model=request.model,
        )
        
        try:
            # Stage 1: Cost optimization analysis
            cost_analysis = None
            if self.cost_optimizer:
                available_provider_models = {
                    ptype: list(provider.models.keys())
                    for ptype, provider in self.providers.items()
                    if provider.is_available
                }
                
                performance_data = await self._gather_performance_data()
                
                cost_analysis = await self.cost_optimizer.optimize_request_routing(
                    request, available_provider_models, performance_data
                )
                
                if cost_analysis:
                    self.routing_stats["cost_optimizations"] += 1
            
            # Stage 2: Model-specific routing with capability matching
            available_providers = list(self.providers.values())
            model_selection = await self.model_router.select_optimal_model(
                request, available_providers, preferences
            )
            
            if model_selection:
                selected_provider, selected_model = model_selection
                request.model = selected_model  # Update request with optimal model
            
            # Stage 3: Intelligent provider selection with load balancing
            load_balanced_provider = await self.load_balancer.select_provider(
                request,
                available_providers,
                LoadBalancingAlgorithm.ADAPTIVE_WEIGHTED,
            )
            
            if load_balanced_provider:
                selected_provider = load_balanced_provider
            
            # Stage 4: Final provider selection with intelligent routing
            selection_criteria = self._build_selection_criteria(cost_analysis, preferences)
            
            final_score = await self.intelligent_router.select_optimal_provider(
                request,
                available_providers,
                SelectionStrategy.WEIGHTED_COMPOSITE,
                selection_criteria,
            )
            
            if final_score:
                selected_provider = next(
                    p for p in available_providers 
                    if p.provider_type == final_score.provider_type
                )
                selected_model = final_score.model_id
            else:
                # Fallback to first available provider
                selected_provider = available_providers[0]
                selected_model = request.model or list(selected_provider.models.keys())[0]
            
            # Stage 5: Execute request with fallback protection
            routing_decision = RoutingDecision(
                selected_provider=selected_provider,
                selected_model=selected_model,
                selection_strategy=SelectionStrategy.WEIGHTED_COMPOSITE,
                confidence_score=final_score.total_score if final_score else 0.5,
                routing_reason=final_score.selection_reason if final_score else "Default selection",
                alternatives_considered=len(available_providers),
                fallback_tier=FallbackTier.PRIMARY,
                cost_estimate=final_score.estimated_cost if final_score else 0.0,
                latency_estimate=final_score.estimated_latency_ms if final_score else 3000.0,
                cost_optimization=cost_analysis or {},
            )
            
            # Execute request
            if self.fallback_manager:
                # Use fallback manager for resilient execution
                response, fallback_attempts = await self.fallback_manager.execute_with_fallback(
                    request, available_providers, SelectionStrategy.WEIGHTED_COMPOSITE
                )
                
                if fallback_attempts:
                    routing_decision.fallback_tier = fallback_attempts[-1].fallback_tier
                    self.routing_stats["fallback_used"] += 1
            else:
                # Direct execution
                response = await selected_provider.generate_response(request)
            
            # Stage 6: Post-request tracking and monitoring
            routing_time_ms = (datetime.now() - routing_start).total_seconds() * 1000
            
            await self._track_request_completion(
                request, response, routing_decision, routing_time_ms
            )
            
            self.routing_stats["successful_routes"] += 1
            
            logger.info(
                "Enterprise routing completed successfully",
                request_id=request.request_id,
                provider=selected_provider.provider_type.value,
                model=selected_model,
                routing_time_ms=routing_time_ms,
                cost=response.cost,
            )
            
            return routing_decision, response
            
        except Exception as e:
            logger.error(
                "Enterprise routing failed",
                request_id=request.request_id,
                error=str(e),
            )
            raise
    
    async def stream_request(
        self,
        request: AIRequest,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> Tuple[RoutingDecision, AsyncGenerator[StreamingChunk, None]]:
        """Stream a request with enterprise routing."""
        # Ensure streaming is enabled
        request.stream = True
        
        # Get routing decision (similar to route_request but for streaming)
        routing_decision = await self._make_routing_decision(request, preferences)
        
        # Create streaming response
        async def enterprise_stream():
            async for chunk in routing_decision.selected_provider.stream_response(request):
                yield chunk
        
        return routing_decision, enterprise_stream()
    
    async def _make_routing_decision(
        self,
        request: AIRequest,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Make routing decision without executing the request."""
        # Simplified version of routing logic for streaming
        available_providers = list(self.providers.values())
        
        # Use intelligent router for selection
        final_score = await self.intelligent_router.select_optimal_provider(
            request,
            available_providers,
            SelectionStrategy.SPEED_OPTIMIZED,  # Prefer speed for streaming
        )
        
        if final_score:
            selected_provider = next(
                p for p in available_providers 
                if p.provider_type == final_score.provider_type
            )
            selected_model = final_score.model_id
        else:
            selected_provider = available_providers[0]
            selected_model = request.model or list(selected_provider.models.keys())[0]
        
        return RoutingDecision(
            selected_provider=selected_provider,
            selected_model=selected_model,
            selection_strategy=SelectionStrategy.SPEED_OPTIMIZED,
            confidence_score=final_score.total_score if final_score else 0.5,
            routing_reason="Streaming optimized selection",
            alternatives_considered=len(available_providers),
            fallback_tier=FallbackTier.PRIMARY,
            cost_estimate=final_score.estimated_cost if final_score else 0.0,
            latency_estimate=final_score.estimated_latency_ms if final_score else 2000.0,
        )
    
    async def _gather_performance_data(self) -> Dict[str, Dict[str, Any]]:
        """Gather performance data from all monitoring components."""
        performance_data = {}
        
        for provider_type, provider in self.providers.items():
            for model_id in provider.models.keys():
                key = f"{provider_type.value}:{model_id}"
                
                # Get health metrics
                health_metrics = self.health_monitor.get_metrics(provider_type)
                
                # Get SLA metrics if available
                sla_data = {}
                if self.sla_monitor:
                    sla_key = (provider_type, model_id)
                    if sla_key in self.sla_monitor.performance_metrics:
                        sla_metrics = self.sla_monitor.performance_metrics[sla_key]
                        sla_data = {
                            "avg_latency_ms": sla_metrics.avg_latency_ms,
                            "success_rate": sla_metrics.successful_requests / max(1, sla_metrics.total_requests),
                            "quality_score": sla_metrics.avg_quality_score,
                        }
                
                performance_data[key] = {
                    "request_count": health_metrics.total_requests if health_metrics else 0,
                    "success_rate": health_metrics.uptime_percentage / 100 if health_metrics else 0.9,
                    "avg_latency_ms": health_metrics.avg_latency_ms if health_metrics else 3000,
                    **sla_data,
                }
        
        return performance_data
    
    def _build_selection_criteria(
        self,
        cost_analysis: Optional[Dict[str, Any]],
        preferences: Optional[Dict[str, Any]],
    ) -> SelectionCriteria:
        """Build selection criteria from cost analysis and preferences."""
        criteria = SelectionCriteria()
        
        # Apply cost optimization insights
        if cost_analysis and cost_analysis.get("optimization_reason"):
            if "cost" in cost_analysis["optimization_reason"].lower():
                criteria.cost_weight = 0.5
                criteria.speed_weight = 0.2
                criteria.quality_weight = 0.2
                criteria.reliability_weight = 0.1
        
        # Apply user preferences
        if preferences:
            if preferences.get("prefer_speed"):
                criteria.speed_weight = 0.6
                criteria.cost_weight = 0.2
            elif preferences.get("prefer_quality"):
                criteria.quality_weight = 0.6
                criteria.cost_weight = 0.2
            elif preferences.get("prefer_cost"):
                criteria.cost_weight = 0.6
                criteria.quality_weight = 0.2
            
            if preferences.get("max_latency_ms"):
                criteria.max_latency_ms = preferences["max_latency_ms"]
            if preferences.get("max_cost"):
                criteria.max_cost_per_request = preferences["max_cost"]
        
        return criteria
    
    async def _track_request_completion(
        self,
        request: AIRequest,
        response: AIResponse,
        routing_decision: RoutingDecision,
        routing_time_ms: float,
    ) -> None:
        """Track request completion across all monitoring systems."""
        provider_type = routing_decision.selected_provider.provider_type
        model = routing_decision.selected_model
        
        # Track with health monitor
        await self.health_monitor.record_request(
            provider_type,
            success=True,
            latency_ms=response.latency_ms,
            model=model,
        )
        
        # Track with load balancer
        await self.load_balancer.track_request_completion(
            provider_type,
            success=True,
            response_time_ms=response.latency_ms,
            cost=response.cost,
        )
        
        # Track with cost optimizer
        if self.cost_optimizer:
            await self.cost_optimizer.track_request_cost(
                request, response, provider_type, model
            )
        
        # Track with SLA monitor
        if self.sla_monitor:
            await self.sla_monitor.record_request(
                request, response, provider_type, model,
                success=True, latency_ms=response.latency_ms
            )
    
    def _create_load_balancer_config(self):
        """Create load balancer configuration."""
        from .load_balancer import LoadBalancingConfig
        
        return LoadBalancingConfig(
            algorithm=LoadBalancingAlgorithm.ADAPTIVE_WEIGHTED,
            enable_adaptive_weights=True,
            max_connections_per_provider=100,
            max_response_time_ms=30000,
            min_success_rate=0.85,
        )
    
    def _handle_sla_alert(self, alert) -> None:
        """Handle SLA alerts from the monitoring system."""
        self.routing_stats["sla_violations"] += 1
        
        logger.warning(
            "SLA alert received",
            alert_type=alert.alert_type.value,
            severity=alert.severity.value,
            provider=alert.provider_type.value,
            message=alert.message,
        )
        
        # Could implement auto-remediation actions here
        # For example, temporarily reducing load on a failing provider
    
    def get_enterprise_statistics(self) -> Dict[str, Any]:
        """Get comprehensive enterprise router statistics."""
        stats = {
            "routing_statistics": self.routing_stats.copy(),
            "performance_targets": self.performance_targets.copy(),
            "component_status": {
                "health_monitor": bool(self.health_monitor),
                "load_balancer": bool(self.load_balancer),
                "intelligent_router": bool(self.intelligent_router),
                "model_router": bool(self.model_router),
                "cost_optimizer": bool(self.cost_optimizer),
                "fallback_manager": bool(self.fallback_manager),
                "sla_monitor": bool(self.sla_monitor),
            },
        }
        
        # Add component-specific stats
        if self.intelligent_router:
            stats["intelligent_router"] = self.intelligent_router.get_router_statistics()
        
        if self.load_balancer:
            stats["load_balancer"] = self.load_balancer.get_load_balancer_statistics()
        
        if self.model_router:
            stats["model_router"] = self.model_router.get_router_statistics()
        
        if self.cost_optimizer:
            stats["cost_optimizer"] = self.cost_optimizer.get_cost_analysis()
        
        if self.fallback_manager:
            stats["fallback_manager"] = self.fallback_manager.get_fallback_statistics()
        
        if self.sla_monitor:
            stats["sla_monitor"] = self.sla_monitor.get_performance_summary()
        
        return stats
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        if not self.sla_monitor:
            return {"status": "monitoring_disabled"}
        
        return self.sla_monitor.get_sla_compliance_report()
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts across monitoring systems."""
        alerts = []
        
        if self.sla_monitor:
            alerts.extend(self.sla_monitor.get_active_alerts())
        
        if self.cost_optimizer:
            cost_alerts = self.cost_optimizer.get_budget_alerts()
            alerts.extend([
                {
                    "alert_id": alert["alert_id"],
                    "type": "budget_alert",
                    "severity": alert["severity"],
                    "message": alert["message"],
                    "triggered_at": alert["triggered_at"],
                }
                for alert in cost_alerts
            ])
        
        return alerts