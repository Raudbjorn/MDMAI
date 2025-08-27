"""Provider registry for managing and selecting AI providers."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from structlog import get_logger

from .abstract_provider import AbstractProvider
from .latency_tracker import LatencyTracker
from .models import (
    AIRequest,
    ProviderCapability,
    ProviderHealth,
    ProviderSelection,
    ProviderStatus,
    ProviderType,
)

logger = get_logger(__name__)


class ProviderRegistry:
    """Registry for managing AI providers and implementing selection strategies."""
    
    def __init__(self):
        self._providers: Dict[ProviderType, AbstractProvider] = {}
        self._provider_priorities: Dict[ProviderType, int] = {}
        self._health_check_interval = 300  # 5 minutes
        self._health_check_task: Optional[asyncio.Task] = None
        self._latency_tracker = LatencyTracker()
        self._selection_strategies: Dict[str, callable] = {
            "round_robin": self._round_robin_strategy,
            "priority": self._priority_strategy,
            "cost": self._cost_optimized_strategy,
            "capability": self._capability_based_strategy,
            "load_balanced": self._load_balanced_strategy,
            "failover": self._failover_strategy,
            "random": self._random_strategy,
            "speed": self._speed_optimized_strategy,
        }
        self._round_robin_counter = 0
        self._load_tracker: Dict[ProviderType, int] = {}
        self._failover_chain: List[ProviderType] = []
    
    async def register_provider(
        self,
        provider: AbstractProvider,
        priority: int = 1,
        auto_initialize: bool = True,
    ) -> None:
        """Register a new AI provider.
        
        Args:
            provider: Provider instance to register
            priority: Provider priority (higher = preferred)
            auto_initialize: Whether to initialize the provider automatically
        """
        provider_type = provider.provider_type
        
        if provider_type in self._providers:
            logger.warning(
                "Provider already registered, replacing",
                provider=provider_type.value,
            )
        
        self._providers[provider_type] = provider
        self._provider_priorities[provider_type] = priority
        self._load_tracker[provider_type] = 0
        
        if auto_initialize:
            try:
                await provider.initialize()
            except Exception as e:
                logger.error(
                    "Failed to initialize provider during registration",
                    provider=provider_type.value,
                    error=str(e),
                )
        
        # Update failover chain based on priority
        self._update_failover_chain()
        
        logger.info(
            "Registered AI provider",
            provider=provider_type.value,
            priority=priority,
            initialized=provider._initialized,
        )
    
    async def unregister_provider(self, provider_type: ProviderType) -> None:
        """Unregister an AI provider.
        
        Args:
            provider_type: Type of provider to unregister
        """
        if provider_type not in self._providers:
            logger.warning("Provider not registered", provider=provider_type.value)
            return
        
        provider = self._providers[provider_type]
        
        try:
            await provider.shutdown()
        except Exception as e:
            logger.error(
                "Error shutting down provider during unregistration",
                provider=provider_type.value,
                error=str(e),
            )
        
        del self._providers[provider_type]
        del self._provider_priorities[provider_type]
        if provider_type in self._load_tracker:
            del self._load_tracker[provider_type]
        
        # Update failover chain
        self._update_failover_chain()
        
        logger.info("Unregistered AI provider", provider=provider_type.value)
    
    def get_provider(self, provider_type: ProviderType) -> Optional[AbstractProvider]:
        """Get a specific provider.
        
        Args:
            provider_type: Type of provider to retrieve
            
        Returns:
            Provider instance or None if not registered
        """
        return self._providers.get(provider_type)
    
    def get_available_providers(self) -> List[AbstractProvider]:
        """Get all available providers.
        
        Returns:
            List of available provider instances
        """
        return [
            provider for provider in self._providers.values()
            if provider.is_available
        ]
    
    def get_providers_by_capability(
        self, capability: ProviderCapability
    ) -> List[AbstractProvider]:
        """Get providers that support a specific capability.
        
        Args:
            capability: Required capability
            
        Returns:
            List of providers supporting the capability
        """
        return [
            provider for provider in self._providers.values()
            if provider.supports_capability(capability) and provider.is_available
        ]
    
    async def select_provider(
        self,
        request: AIRequest,
        selection: Optional[ProviderSelection] = None,
        strategy: str = "priority",
    ) -> Optional[AbstractProvider]:
        """Select a provider for a request based on strategy.
        
        Args:
            request: The AI request
            selection: Provider selection criteria
            strategy: Selection strategy name
            
        Returns:
            Selected provider or None if no suitable provider found
        """
        # Get available providers
        available_providers = self.get_available_providers()
        
        if not available_providers:
            logger.warning("No available providers")
            return None
        
        # Apply selection criteria if provided
        if selection:
            available_providers = self._filter_by_selection(
                available_providers, selection, request
            )
        
        if not available_providers:
            logger.warning("No providers match selection criteria")
            return None
        
        # Apply selection strategy
        if strategy not in self._selection_strategies:
            logger.warning(f"Unknown strategy {strategy}, using priority")
            strategy = "priority"
        
        selected = self._selection_strategies[strategy](
            available_providers, request
        )
        
        if selected:
            # Track load
            self._load_tracker[selected.provider_type] = \
                self._load_tracker.get(selected.provider_type, 0) + 1
            
            # Record latency for this request (will be updated when response completes)
            # This helps track provider selection patterns
            
            logger.debug(
                "Selected provider",
                provider=selected.provider_type.value,
                strategy=strategy,
                request_id=request.request_id,
            )
        
        return selected
    
    async def start_health_monitoring(self, interval: Optional[int] = None) -> None:
        """Start periodic health monitoring for all providers.
        
        Args:
            interval: Health check interval in seconds
        """
        if self._health_check_task and not self._health_check_task.done():
            logger.warning("Health monitoring already running")
            return
        
        if interval:
            self._health_check_interval = interval
        
        # Start latency tracker
        await self._latency_tracker.start()
        
        self._health_check_task = asyncio.create_task(self._health_monitoring_loop())
        logger.info(
            "Started health monitoring",
            interval=self._health_check_interval,
        )
    
    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped health monitoring")
        
        # Stop latency tracker
        await self._latency_tracker.stop()
    
    async def perform_health_check(self) -> Dict[ProviderType, ProviderHealth]:
        """Perform health check on all providers.
        
        Returns:
            Dictionary of provider health statuses
        """
        health_results = {}
        
        for provider_type, provider in self._providers.items():
            try:
                health = await provider.health_check()
                health_results[provider_type] = health
                
                logger.debug(
                    "Provider health check",
                    provider=provider_type.value,
                    status=health.status.value,
                    uptime=health.uptime_percentage,
                )
            except Exception as e:
                logger.error(
                    "Health check failed",
                    provider=provider_type.value,
                    error=str(e),
                )
                
                # Create error health status
                health_results[provider_type] = ProviderHealth(
                    provider_type=provider_type,
                    status=ProviderStatus.ERROR,
                    last_error=datetime.now(),
                    error_count=1,
                )
        
        return health_results
    
    async def shutdown_all_providers(self) -> None:
        """Shutdown all registered providers."""
        logger.info("Shutting down all providers")
        
        # Stop health monitoring first
        await self.stop_health_monitoring()
        
        # Shutdown each provider
        shutdown_tasks = []
        for provider_type, provider in self._providers.items():
            logger.debug("Shutting down provider", provider=provider_type.value)
            shutdown_tasks.append(provider.shutdown())
        
        # Wait for all shutdowns to complete
        if shutdown_tasks:
            results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            for provider_type, result in zip(self._providers.keys(), results):
                if isinstance(result, Exception):
                    logger.error(
                        "Error shutting down provider",
                        provider=provider_type.value,
                        error=str(result),
                    )
        
        # Clear registry
        self._providers.clear()
        self._provider_priorities.clear()
        self._load_tracker.clear()
        self._failover_chain.clear()
        
        logger.info("All providers shut down")
    
    # Selection strategy implementations
    
    def _round_robin_strategy(
        self, providers: List[AbstractProvider], request: AIRequest
    ) -> Optional[AbstractProvider]:
        """Round-robin selection strategy."""
        if not providers:
            return None
        
        # Get next provider in rotation
        selected = providers[self._round_robin_counter % len(providers)]
        self._round_robin_counter += 1
        
        return selected
    
    def _priority_strategy(
        self, providers: List[AbstractProvider], request: AIRequest
    ) -> Optional[AbstractProvider]:
        """Priority-based selection strategy."""
        if not providers:
            return None
        
        # Sort by priority (descending)
        sorted_providers = sorted(
            providers,
            key=lambda p: self._provider_priorities.get(p.provider_type, 0),
            reverse=True,
        )
        
        return sorted_providers[0]
    
    def _cost_optimized_strategy(
        self, providers: List[AbstractProvider], request: AIRequest
    ) -> Optional[AbstractProvider]:
        """Cost-optimized selection strategy."""
        if not providers:
            return None
        
        # Find provider with lowest cost for the requested model
        best_provider = None
        min_cost = float('inf')
        
        for provider in providers:
            # Check if provider has the requested model
            if request.model not in provider.models:
                continue
            
            # Estimate cost
            model_spec = provider.models[request.model]
            estimated_tokens = len(str(request.messages)) // 4  # Rough estimate
            cost = provider.get_model_cost(
                request.model,
                estimated_tokens,
                request.max_tokens or 1000,
            )
            
            if cost < min_cost:
                min_cost = cost
                best_provider = provider
        
        return best_provider or providers[0]  # Fallback to first provider
    
    def _capability_based_strategy(
        self, providers: List[AbstractProvider], request: AIRequest
    ) -> Optional[AbstractProvider]:
        """Capability-based selection strategy."""
        if not providers:
            return None
        
        # Score providers based on capability match
        scored_providers = []
        
        for provider in providers:
            score = 0
            
            # Check tool support if needed
            if request.tools and provider.supports_tools(request.model):
                score += 10
            
            # Check streaming support if needed
            if request.stream and provider.supports_streaming(request.model):
                score += 5
            
            # Add priority as secondary factor
            score += self._provider_priorities.get(provider.provider_type, 0)
            
            scored_providers.append((provider, score))
        
        # Sort by score (descending)
        scored_providers.sort(key=lambda x: x[1], reverse=True)
        
        return scored_providers[0][0] if scored_providers else None
    
    def _load_balanced_strategy(
        self, providers: List[AbstractProvider], request: AIRequest
    ) -> Optional[AbstractProvider]:
        """Load-balanced selection strategy."""
        if not providers:
            return None
        
        # Find provider with lowest load
        min_load = float('inf')
        selected = None
        
        for provider in providers:
            load = self._load_tracker.get(provider.provider_type, 0)
            if load < min_load:
                min_load = load
                selected = provider
        
        return selected
    
    def _failover_strategy(
        self, providers: List[AbstractProvider], request: AIRequest
    ) -> Optional[AbstractProvider]:
        """Failover selection strategy."""
        if not providers:
            return None
        
        # Use failover chain order
        for provider_type in self._failover_chain:
            provider = self._providers.get(provider_type)
            if provider and provider in providers:
                return provider
        
        # Fallback to first available
        return providers[0]
    
    def _random_strategy(
        self, providers: List[AbstractProvider], request: AIRequest
    ) -> Optional[AbstractProvider]:
        """Random selection strategy."""
        if not providers:
            return None
        
        import random
        return random.choice(providers)
    
    def _speed_optimized_strategy(
        self, providers: List[AbstractProvider], request: AIRequest
    ) -> Optional[AbstractProvider]:
        """Speed-optimized selection using real latency metrics."""
        if not providers:
            return None
        
        # Get provider types from available providers
        provider_types = [p.provider_type for p in providers]
        
        # Get fastest provider based on real latency
        fastest = self._latency_tracker.get_fastest_provider(
            provider_types, 
            model=request.model,
            min_success_rate=0.9
        )
        
        if fastest:
            for provider in providers:
                if provider.provider_type == fastest:
                    logger.debug(
                        "Selected provider based on latency",
                        provider=fastest.value,
                        model=request.model,
                    )
                    return provider
        
        # Fallback to priority if no latency data
        logger.debug("No latency data available, falling back to priority")
        return self._priority_strategy(providers, request)
    
    # Helper methods
    
    def _filter_by_selection(
        self,
        providers: List[AbstractProvider],
        selection: ProviderSelection,
        request: AIRequest,
    ) -> List[AbstractProvider]:
        """Filter providers based on selection criteria.
        
        Args:
            providers: List of providers to filter
            selection: Selection criteria
            request: The AI request
            
        Returns:
            Filtered list of providers
        """
        filtered = []
        
        for provider in providers:
            # Check required capabilities
            if selection.required_capabilities:
                if not all(
                    provider.supports_capability(cap)
                    for cap in selection.required_capabilities
                ):
                    continue
            
            # Check preferred providers
            if selection.preferred_providers:
                if provider.provider_type not in selection.preferred_providers:
                    continue
            
            # Check excluded providers
            if selection.exclude_providers:
                if provider.provider_type in selection.exclude_providers:
                    continue
            
            # Check streaming requirement
            if selection.require_streaming:
                if not provider.supports_streaming(request.model):
                    continue
            
            # Check tools requirement
            if selection.require_tools:
                if not provider.supports_tools(request.model):
                    continue
            
            filtered.append(provider)
        
        return filtered
    
    def _update_failover_chain(self) -> None:
        """Update failover chain based on provider priorities."""
        self._failover_chain = sorted(
            self._provider_priorities.keys(),
            key=lambda p: self._provider_priorities[p],
            reverse=True,
        )
    
    async def _health_monitoring_loop(self) -> None:
        """Health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self.perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health monitoring loop", error=str(e))
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        available_count = len(self.get_available_providers())
        total_count = len(self._providers)
        
        provider_stats = {}
        for provider_type, provider in self._providers.items():
            provider_stats[provider_type.value] = {
                "available": provider.is_available,
                "priority": self._provider_priorities.get(provider_type, 0),
                "load": self._load_tracker.get(provider_type, 0),
                "models": len(provider.models),
                "health_status": provider.health.status.value,
            }
        
        return {
            "total_providers": total_count,
            "available_providers": available_count,
            "health_monitoring": self._health_check_task is not None and not self._health_check_task.done(),
            "health_check_interval": self._health_check_interval,
            "failover_chain": [p.value for p in self._failover_chain],
            "providers": provider_stats,
            "total_requests": sum(self._load_tracker.values()),
        }