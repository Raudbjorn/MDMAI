"""
Advanced Load Balancer with Performance Optimization
Task 25.3: Develop Provider Router with Fallback
"""

import asyncio
import hashlib
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from structlog import get_logger

from .models import ProviderType, AIRequest
from .abstract_provider import AbstractProvider
from .health_monitor import HealthMonitor

logger = get_logger(__name__)


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms."""
    
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    CONSISTENT_HASH = "consistent_hash"
    ADAPTIVE_WEIGHTED = "adaptive_weighted"
    PERFORMANCE_BASED = "performance_based"
    TOKEN_AWARE = "token_aware"


@dataclass
class ProviderMetrics:
    """Real-time provider performance metrics."""
    
    provider_type: ProviderType
    
    # Connection tracking
    active_connections: int = 0
    total_connections: int = 0
    
    # Performance metrics
    response_times: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Throughput tracking
    requests_per_minute: float = 0.0
    tokens_per_minute: float = 0.0
    
    # Success tracking
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 1.0
    
    # Load tracking
    current_load: float = 0.0  # 0.0 to 1.0
    capacity_utilization: float = 0.0
    
    # Cost tracking
    total_cost: float = 0.0
    cost_per_request: float = 0.0
    
    # Recent performance window
    request_timestamps: Deque[datetime] = field(default_factory=lambda: deque(maxlen=1000))
    
    # Weight for weighted algorithms
    weight: float = 1.0
    dynamic_weight: float = 1.0
    
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class LoadBalancingConfig:
    """Configuration for load balancing behavior."""
    
    algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ADAPTIVE_WEIGHTED
    
    # Weights for different factors
    performance_weight: float = 0.4
    reliability_weight: float = 0.3
    cost_weight: float = 0.2
    capacity_weight: float = 0.1
    
    # Thresholds
    max_connections_per_provider: int = 100
    max_response_time_ms: float = 30000
    min_success_rate: float = 0.85
    
    # Adaptive behavior
    enable_adaptive_weights: bool = True
    learning_rate: float = 0.1
    performance_window_minutes: int = 5
    
    # Circuit breaker integration
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60


class LoadBalancer:
    """
    Advanced load balancer with multiple algorithms and performance optimization.
    
    Features:
    - Multiple load balancing algorithms
    - Real-time performance tracking
    - Adaptive weight adjustment
    - Capacity-aware routing
    - Token-aware load distribution
    - Performance-based selection
    - Integration with health monitoring
    """
    
    def __init__(
        self,
        config: LoadBalancingConfig,
        health_monitor: HealthMonitor,
    ):
        self.config = config
        self.health_monitor = health_monitor
        
        # Provider metrics
        self.provider_metrics: Dict[ProviderType, ProviderMetrics] = {}
        
        # Algorithm state
        self._round_robin_counter = 0
        self._consistent_hash_ring: Dict[str, ProviderType] = {}
        self._hash_ring_size = 1000
        
        # Performance tracking
        self._performance_history: Dict[ProviderType, List[Tuple[datetime, float]]] = defaultdict(list)
        self._capacity_estimates: Dict[ProviderType, float] = {}
        
        # Async tasks
        self._metrics_update_task: Optional[asyncio.Task] = None
        self._weight_adjustment_task: Optional[asyncio.Task] = None
        
        # Algorithm implementations
        self._algorithm_functions = {
            LoadBalancingAlgorithm.ROUND_ROBIN: self._round_robin_select,
            LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN: self._weighted_round_robin_select,
            LoadBalancingAlgorithm.LEAST_CONNECTIONS: self._least_connections_select,
            LoadBalancingAlgorithm.LEAST_RESPONSE_TIME: self._least_response_time_select,
            LoadBalancingAlgorithm.CONSISTENT_HASH: self._consistent_hash_select,
            LoadBalancingAlgorithm.ADAPTIVE_WEIGHTED: self._adaptive_weighted_select,
            LoadBalancingAlgorithm.PERFORMANCE_BASED: self._performance_based_select,
            LoadBalancingAlgorithm.TOKEN_AWARE: self._token_aware_select,
        }
    
    async def start(self) -> None:
        """Start the load balancer and background tasks."""
        logger.info("Starting load balancer")
        
        # Start metrics update task
        self._metrics_update_task = asyncio.create_task(self._metrics_update_loop())
        
        # Start adaptive weight adjustment task if enabled
        if self.config.enable_adaptive_weights:
            self._weight_adjustment_task = asyncio.create_task(self._weight_adjustment_loop())
        
        logger.info("Load balancer started")
    
    async def stop(self) -> None:
        """Stop the load balancer and background tasks."""
        logger.info("Stopping load balancer")
        
        if self._metrics_update_task:
            self._metrics_update_task.cancel()
            try:
                await self._metrics_update_task
            except asyncio.CancelledError:
                pass
        
        if self._weight_adjustment_task:
            self._weight_adjustment_task.cancel()
            try:
                await self._weight_adjustment_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Load balancer stopped")
    
    def register_provider(
        self,
        provider: AbstractProvider,
        initial_weight: float = 1.0,
        max_connections: Optional[int] = None,
    ) -> None:
        """Register a provider with the load balancer."""
        provider_type = provider.provider_type
        
        if provider_type not in self.provider_metrics:
            self.provider_metrics[provider_type] = ProviderMetrics(
                provider_type=provider_type,
                weight=initial_weight,
                dynamic_weight=initial_weight,
            )
            
            # Add to consistent hash ring
            self._add_to_hash_ring(provider_type, initial_weight)
            
            logger.info(
                "Registered provider with load balancer",
                provider=provider_type.value,
                weight=initial_weight,
            )
    
    def unregister_provider(self, provider_type: ProviderType) -> None:
        """Unregister a provider from the load balancer."""
        if provider_type in self.provider_metrics:
            del self.provider_metrics[provider_type]
            self._remove_from_hash_ring(provider_type)
            
            logger.info("Unregistered provider from load balancer", provider=provider_type.value)
    
    async def select_provider(
        self,
        request: AIRequest,
        available_providers: List[AbstractProvider],
        algorithm: Optional[LoadBalancingAlgorithm] = None,
    ) -> Optional[AbstractProvider]:
        """
        Select a provider using the specified load balancing algorithm.
        
        Args:
            request: The AI request
            available_providers: List of available providers
            algorithm: Override default algorithm
            
        Returns:
            Selected provider or None
        """
        if not available_providers:
            return None
        
        # Filter providers by capacity and health
        viable_providers = self._filter_viable_providers(available_providers)
        if not viable_providers:
            logger.warning("No viable providers after filtering")
            return available_providers[0]  # Fallback to first available
        
        # Select algorithm
        selected_algorithm = algorithm or self.config.algorithm
        algorithm_func = self._algorithm_functions.get(
            selected_algorithm, self._adaptive_weighted_select
        )
        
        # Execute selection algorithm
        selected_provider = await algorithm_func(request, viable_providers)
        
        if selected_provider:
            # Track selection
            await self._track_selection(selected_provider, request)
            
            logger.debug(
                "Selected provider via load balancer",
                provider=selected_provider.provider_type.value,
                algorithm=selected_algorithm.value,
                request_id=request.request_id,
            )
        
        return selected_provider
    
    def _filter_viable_providers(
        self, providers: List[AbstractProvider]
    ) -> List[AbstractProvider]:
        """Filter providers based on capacity and health constraints."""
        viable_providers = []
        
        for provider in providers:
            provider_type = provider.provider_type
            
            # Ensure metrics exist
            if provider_type not in self.provider_metrics:
                self.register_provider(provider)
            
            metrics = self.provider_metrics[provider_type]
            
            # Check connection limits
            if (metrics.active_connections >= self.config.max_connections_per_provider):
                logger.debug(
                    "Provider excluded due to connection limit",
                    provider=provider_type.value,
                    connections=metrics.active_connections,
                )
                continue
            
            # Check response time limits
            if (metrics.avg_response_time_ms > self.config.max_response_time_ms):
                logger.debug(
                    "Provider excluded due to response time",
                    provider=provider_type.value,
                    response_time=metrics.avg_response_time_ms,
                )
                continue
            
            # Check success rate
            if metrics.success_rate < self.config.min_success_rate:
                logger.debug(
                    "Provider excluded due to low success rate",
                    provider=provider_type.value,
                    success_rate=metrics.success_rate,
                )
                continue
            
            # Check health status if circuit breaker enabled
            if self.config.circuit_breaker_enabled:
                health_metrics = self.health_monitor.get_metrics(provider_type)
                if health_metrics and health_metrics.consecutive_failures >= self.config.failure_threshold:
                    logger.debug(
                        "Provider excluded due to circuit breaker",
                        provider=provider_type.value,
                    )
                    continue
            
            viable_providers.append(provider)
        
        return viable_providers
    
    # Load balancing algorithm implementations
    
    async def _round_robin_select(
        self, request: AIRequest, providers: List[AbstractProvider]
    ) -> Optional[AbstractProvider]:
        """Simple round-robin selection."""
        if not providers:
            return None
        
        selected = providers[self._round_robin_counter % len(providers)]
        self._round_robin_counter += 1
        
        return selected
    
    async def _weighted_round_robin_select(
        self, request: AIRequest, providers: List[AbstractProvider]
    ) -> Optional[AbstractProvider]:
        """Weighted round-robin selection."""
        if not providers:
            return None
        
        # Calculate total weight
        total_weight = sum(
            self.provider_metrics[p.provider_type].dynamic_weight
            for p in providers
            if p.provider_type in self.provider_metrics
        )
        
        if total_weight == 0:
            return providers[0]
        
        # Generate weighted selection
        target = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for provider in providers:
            if provider.provider_type in self.provider_metrics:
                cumulative_weight += self.provider_metrics[provider.provider_type].dynamic_weight
                if target <= cumulative_weight:
                    return provider
        
        return providers[-1]  # Fallback
    
    async def _least_connections_select(
        self, request: AIRequest, providers: List[AbstractProvider]
    ) -> Optional[AbstractProvider]:
        """Select provider with least active connections."""
        if not providers:
            return None
        
        best_provider = None
        min_connections = float('inf')
        
        for provider in providers:
            if provider.provider_type in self.provider_metrics:
                connections = self.provider_metrics[provider.provider_type].active_connections
                if connections < min_connections:
                    min_connections = connections
                    best_provider = provider
        
        return best_provider or providers[0]
    
    async def _least_response_time_select(
        self, request: AIRequest, providers: List[AbstractProvider]
    ) -> Optional[AbstractProvider]:
        """Select provider with lowest response time."""
        if not providers:
            return None
        
        best_provider = None
        min_response_time = float('inf')
        
        for provider in providers:
            if provider.provider_type in self.provider_metrics:
                response_time = self.provider_metrics[provider.provider_type].avg_response_time_ms
                if response_time < min_response_time:
                    min_response_time = response_time
                    best_provider = provider
        
        return best_provider or providers[0]
    
    async def _consistent_hash_select(
        self, request: AIRequest, providers: List[AbstractProvider]
    ) -> Optional[AbstractProvider]:
        """Consistent hash selection based on request content."""
        if not providers:
            return None
        
        # Generate hash key from request
        hash_key = self._generate_request_hash(request)
        
        # Find closest provider in hash ring
        closest_provider_type = self._find_in_hash_ring(hash_key)
        
        # Find provider instance
        for provider in providers:
            if provider.provider_type == closest_provider_type:
                return provider
        
        return providers[0]  # Fallback if not found
    
    async def _adaptive_weighted_select(
        self, request: AIRequest, providers: List[AbstractProvider]
    ) -> Optional[AbstractProvider]:
        """Adaptive weighted selection based on multiple factors."""
        if not providers:
            return None
        
        scored_providers = []
        
        for provider in providers:
            if provider.provider_type not in self.provider_metrics:
                continue
            
            metrics = self.provider_metrics[provider.provider_type]
            
            # Calculate composite score
            performance_score = self._calculate_performance_score(metrics)
            reliability_score = self._calculate_reliability_score(metrics)
            cost_score = self._calculate_cost_score(metrics)
            capacity_score = self._calculate_capacity_score(metrics)
            
            total_score = (
                performance_score * self.config.performance_weight +
                reliability_score * self.config.reliability_weight +
                cost_score * self.config.cost_weight +
                capacity_score * self.config.capacity_weight
            )
            
            scored_providers.append((provider, total_score))
        
        if not scored_providers:
            return providers[0]
        
        # Select best provider or use probabilistic selection
        scored_providers.sort(key=lambda x: x[1], reverse=True)
        return scored_providers[0][0]
    
    async def _performance_based_select(
        self, request: AIRequest, providers: List[AbstractProvider]
    ) -> Optional[AbstractProvider]:
        """Performance-based selection optimizing for speed and reliability."""
        if not providers:
            return None
        
        best_provider = None
        best_score = float('-inf')
        
        for provider in providers:
            if provider.provider_type not in self.provider_metrics:
                continue
            
            metrics = self.provider_metrics[provider.provider_type]
            
            # Calculate performance score
            speed_score = 1.0 / (1.0 + metrics.avg_response_time_ms / 1000.0)  # Favor faster responses
            reliability_score = metrics.success_rate
            load_score = 1.0 - metrics.current_load  # Favor less loaded providers
            
            total_score = (speed_score * 0.4 + reliability_score * 0.4 + load_score * 0.2)
            
            if total_score > best_score:
                best_score = total_score
                best_provider = provider
        
        return best_provider or providers[0]
    
    async def _token_aware_select(
        self, request: AIRequest, providers: List[AbstractProvider]
    ) -> Optional[AbstractProvider]:
        """Token-aware selection considering token throughput capacity."""
        if not providers:
            return None
        
        # Estimate tokens for request
        estimated_tokens = self._estimate_request_tokens(request)
        
        best_provider = None
        best_capacity_ratio = float('inf')
        
        for provider in providers:
            if provider.provider_type not in self.provider_metrics:
                continue
            
            metrics = self.provider_metrics[provider.provider_type]
            
            # Calculate capacity ratio (current load vs estimated capacity)
            if metrics.tokens_per_minute > 0:
                projected_load = (metrics.tokens_per_minute + estimated_tokens) / 60.0  # per second
                capacity_ratio = projected_load / self._get_provider_token_capacity(provider.provider_type)
                
                if capacity_ratio < best_capacity_ratio:
                    best_capacity_ratio = capacity_ratio
                    best_provider = provider
        
        return best_provider or providers[0]
    
    # Helper methods for scoring
    
    def _calculate_performance_score(self, metrics: ProviderMetrics) -> float:
        """Calculate performance score (0.0 to 1.0)."""
        # Response time score (lower is better)
        max_acceptable_time = 10000  # 10 seconds
        time_score = max(0.0, 1.0 - (metrics.avg_response_time_ms / max_acceptable_time))
        
        # Throughput score
        throughput_score = min(1.0, metrics.requests_per_minute / 100.0)  # Normalize to 100 RPM
        
        return (time_score * 0.7 + throughput_score * 0.3)
    
    def _calculate_reliability_score(self, metrics: ProviderMetrics) -> float:
        """Calculate reliability score (0.0 to 1.0)."""
        return metrics.success_rate
    
    def _calculate_cost_score(self, metrics: ProviderMetrics) -> float:
        """Calculate cost score (0.0 to 1.0, higher is better/cheaper)."""
        if metrics.cost_per_request <= 0:
            return 1.0
        
        # Inverse relationship - lower cost = higher score
        max_acceptable_cost = 1.0  # $1 per request
        return max(0.0, 1.0 - (metrics.cost_per_request / max_acceptable_cost))
    
    def _calculate_capacity_score(self, metrics: ProviderMetrics) -> float:
        """Calculate capacity score (0.0 to 1.0)."""
        return 1.0 - metrics.capacity_utilization
    
    # Consistent hashing methods
    
    def _add_to_hash_ring(self, provider_type: ProviderType, weight: float) -> None:
        """Add provider to consistent hash ring."""
        virtual_nodes = int(weight * 100)  # Scale weight to virtual nodes
        
        for i in range(virtual_nodes):
            hash_key = hashlib.md5(f"{provider_type.value}:{i}".encode()).hexdigest()
            self._consistent_hash_ring[hash_key] = provider_type
    
    def _remove_from_hash_ring(self, provider_type: ProviderType) -> None:
        """Remove provider from consistent hash ring."""
        keys_to_remove = [
            key for key, provider in self._consistent_hash_ring.items()
            if provider == provider_type
        ]
        
        for key in keys_to_remove:
            del self._consistent_hash_ring[key]
    
    def _generate_request_hash(self, request: AIRequest) -> str:
        """Generate hash key for request."""
        # Use session_id for session affinity, or request content for consistency
        hash_content = request.session_id or str(request.messages)
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def _find_in_hash_ring(self, hash_key: str) -> ProviderType:
        """Find closest provider in hash ring."""
        if not self._consistent_hash_ring:
            return ProviderType.ANTHROPIC  # Default fallback
        
        ring_keys = sorted(self._consistent_hash_ring.keys())
        
        # Find first key >= hash_key
        for ring_key in ring_keys:
            if ring_key >= hash_key:
                return self._consistent_hash_ring[ring_key]
        
        # Wrap around to first key
        return self._consistent_hash_ring[ring_keys[0]]
    
    # Tracking and metrics methods
    
    async def _track_selection(self, provider: AbstractProvider, request: AIRequest) -> None:
        """Track provider selection for metrics."""
        provider_type = provider.provider_type
        metrics = self.provider_metrics.get(provider_type)
        
        if metrics:
            metrics.active_connections += 1
            metrics.total_connections += 1
            metrics.request_timestamps.append(datetime.now())
    
    async def track_request_completion(
        self,
        provider_type: ProviderType,
        success: bool,
        response_time_ms: float,
        cost: float = 0.0,
        tokens_used: int = 0,
    ) -> None:
        """Track completion of a request for metrics update."""
        if provider_type not in self.provider_metrics:
            return
        
        metrics = self.provider_metrics[provider_type]
        
        # Update connection count
        metrics.active_connections = max(0, metrics.active_connections - 1)
        
        # Update response times
        metrics.response_times.append(response_time_ms)
        
        # Update success tracking
        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
        
        # Update cost tracking
        metrics.total_cost += cost
        if metrics.total_connections > 0:
            metrics.cost_per_request = metrics.total_cost / metrics.total_connections
        
        # Update metrics calculations
        await self._update_provider_metrics(provider_type)
    
    async def _update_provider_metrics(self, provider_type: ProviderType) -> None:
        """Update calculated metrics for a provider."""
        if provider_type not in self.provider_metrics:
            return
        
        metrics = self.provider_metrics[provider_type]
        
        # Update response time statistics
        if metrics.response_times:
            sorted_times = sorted(metrics.response_times)
            metrics.avg_response_time_ms = sum(sorted_times) / len(sorted_times)
            
            if len(sorted_times) >= 20:  # Need sufficient data for percentiles
                metrics.p95_response_time_ms = sorted_times[int(len(sorted_times) * 0.95)]
                metrics.p99_response_time_ms = sorted_times[int(len(sorted_times) * 0.99)]
        
        # Update success rate
        total_attempts = metrics.success_count + metrics.failure_count
        if total_attempts > 0:
            metrics.success_rate = metrics.success_count / total_attempts
        
        # Update requests per minute
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        recent_requests = [
            ts for ts in metrics.request_timestamps
            if ts >= one_minute_ago
        ]
        metrics.requests_per_minute = len(recent_requests)
        
        # Update load calculation
        max_connections = self.config.max_connections_per_provider
        metrics.current_load = metrics.active_connections / max_connections
        
        # Update capacity utilization
        estimated_capacity = self._get_provider_token_capacity(provider_type)
        if estimated_capacity > 0:
            metrics.capacity_utilization = metrics.tokens_per_minute / estimated_capacity
        
        metrics.last_updated = now
    
    def _estimate_request_tokens(self, request: AIRequest) -> int:
        """Estimate tokens required for request."""
        # Simple estimation - could be enhanced with actual tokenizer
        total_chars = sum(len(str(msg.get("content", ""))) for msg in request.messages)
        return total_chars // 4  # Rough approximation
    
    def _get_provider_token_capacity(self, provider_type: ProviderType) -> float:
        """Get estimated token capacity for provider."""
        # Default capacities (tokens per minute) - could be configurable
        capacities = {
            ProviderType.ANTHROPIC: 40000,
            ProviderType.OPENAI: 60000,
            ProviderType.GOOGLE: 50000,
        }
        return capacities.get(provider_type, 40000)
    
    # Background tasks
    
    async def _metrics_update_loop(self) -> None:
        """Background task to update metrics periodically."""
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                for provider_type in list(self.provider_metrics.keys()):
                    await self._update_provider_metrics(provider_type)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in metrics update loop", error=str(e))
    
    async def _weight_adjustment_loop(self) -> None:
        """Background task to adjust weights adaptively."""
        while True:
            try:
                await asyncio.sleep(60)  # Adjust every minute
                
                if self.config.enable_adaptive_weights:
                    await self._adjust_dynamic_weights()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in weight adjustment loop", error=str(e))
    
    async def _adjust_dynamic_weights(self) -> None:
        """Adjust dynamic weights based on performance."""
        for provider_type, metrics in self.provider_metrics.items():
            # Calculate performance-based weight adjustment
            performance_factor = (
                metrics.success_rate * 
                (1.0 / max(1.0, metrics.avg_response_time_ms / 1000.0)) *
                (1.0 - metrics.current_load)
            )
            
            # Apply learning rate
            weight_adjustment = (performance_factor - 1.0) * self.config.learning_rate
            new_weight = metrics.dynamic_weight * (1.0 + weight_adjustment)
            
            # Clamp weight to reasonable bounds
            metrics.dynamic_weight = max(0.1, min(5.0, new_weight))
            
            logger.debug(
                "Adjusted dynamic weight",
                provider=provider_type.value,
                old_weight=metrics.weight,
                new_weight=metrics.dynamic_weight,
                performance_factor=performance_factor,
            )
    
    def get_load_balancer_statistics(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics."""
        provider_stats = {}
        
        for provider_type, metrics in self.provider_metrics.items():
            provider_stats[provider_type.value] = {
                "active_connections": metrics.active_connections,
                "total_connections": metrics.total_connections,
                "avg_response_time_ms": metrics.avg_response_time_ms,
                "p95_response_time_ms": metrics.p95_response_time_ms,
                "success_rate": metrics.success_rate,
                "requests_per_minute": metrics.requests_per_minute,
                "current_load": metrics.current_load,
                "capacity_utilization": metrics.capacity_utilization,
                "cost_per_request": metrics.cost_per_request,
                "weight": metrics.weight,
                "dynamic_weight": metrics.dynamic_weight,
            }
        
        return {
            "algorithm": self.config.algorithm.value,
            "total_providers": len(self.provider_metrics),
            "provider_statistics": provider_stats,
            "hash_ring_size": len(self._consistent_hash_ring),
            "adaptive_weights_enabled": self.config.enable_adaptive_weights,
        }