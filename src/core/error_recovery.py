"""
Error recovery strategies and graceful degradation for MDMAI TTRPG Assistant.

This module provides recovery mechanisms, fallback strategies, and service
degradation patterns for maintaining system availability during failures.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

from .error_handling import (
    BaseError,
    CircuitBreaker,
    error_aggregator,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RecoveryStrategy(Enum):
    """Available recovery strategies."""

    RETRY = auto()  # Simple retry
    FALLBACK = auto()  # Use fallback service/value
    CACHE = auto()  # Use cached data
    DEGRADE = auto()  # Provide degraded functionality
    QUEUE = auto()  # Queue for later processing
    SKIP = auto()  # Skip operation
    COMPENSATE = auto()  # Compensating transaction
    MANUAL = auto()  # Require manual intervention


class ServiceStatus(Enum):
    """Service health status."""

    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    OFFLINE = auto()


@dataclass
class ServiceHealth:
    """Service health information."""

    name: str
    status: ServiceStatus
    last_check: datetime
    error_count: int = 0
    success_count: int = 0
    response_time_ms: Optional[float] = None
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def availability(self) -> float:
        """Calculate service availability percentage."""
        total = self.success_count + self.error_count
        if total == 0:
            return 100.0
        return (self.success_count / total) * 100

    @property
    def is_available(self) -> bool:
        """Check if service is available."""
        return self.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]


class RecoveryHandler(ABC):
    """Abstract base class for recovery handlers."""

    @abstractmethod
    async def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        Check if recovery is possible for the given error.

        Args:
            error: The exception that occurred
            context: Error context information

        Returns:
            True if recovery is possible
        """
        pass

    @abstractmethod
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """
        Attempt to recover from the error.

        Args:
            error: The exception that occurred
            context: Error context information

        Returns:
            Recovery result or raises exception if recovery fails
        """
        pass


class FallbackHandler(RecoveryHandler):
    """Handler for fallback recovery strategy."""

    def __init__(
        self,
        fallback_value: Any = None,
        fallback_function: Optional[Callable] = None,
    ) -> None:
        """
        Initialize fallback handler.

        Args:
            fallback_value: Static fallback value
            fallback_function: Dynamic fallback function
        """
        self.fallback_value = fallback_value
        self.fallback_function = fallback_function

    async def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if fallback is available."""
        return self.fallback_value is not None or self.fallback_function is not None

    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Return fallback value or execute fallback function."""
        logger.warning(
            f"Using fallback recovery for error: {error}",
            extra={"error": str(error), "context": context},
        )
        
        if self.fallback_function:
            if asyncio.iscoroutinefunction(self.fallback_function):
                return await self.fallback_function(error, context)
            return self.fallback_function(error, context)
        
        return self.fallback_value


class CacheHandler(RecoveryHandler):
    """Handler for cache-based recovery."""

    def __init__(self, cache_ttl: float = 3600.0) -> None:
        """
        Initialize cache handler.

        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.cache_ttl = cache_ttl
        self._lock = asyncio.Lock()

    async def set_cache(self, key: str, value: Any) -> None:
        """Store value in cache."""
        async with self._lock:
            self.cache[key] = (value, datetime.utcnow())

    async def get_cache(self, key: str) -> Optional[Any]:
        """Retrieve value from cache if not expired."""
        async with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                age = (datetime.utcnow() - timestamp).total_seconds()
                
                if age <= self.cache_ttl:
                    return value
                else:
                    del self.cache[key]
        
        return None

    async def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if cached data is available."""
        cache_key = context.get("cache_key")
        if not cache_key:
            return False
        
        cached_value = await self.get_cache(cache_key)
        return cached_value is not None

    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Return cached value."""
        cache_key = context.get("cache_key")
        if not cache_key:
            raise ValueError("No cache key provided in context")
        
        cached_value = await self.get_cache(cache_key)
        if cached_value is None:
            raise ValueError(f"No cached value found for key: {cache_key}")
        
        logger.info(
            f"Using cached data for recovery: {cache_key}",
            extra={"cache_key": cache_key, "error": str(error)},
        )
        
        return cached_value


class QueueHandler(RecoveryHandler):
    """Handler for queue-based recovery."""

    def __init__(self, max_queue_size: int = 1000) -> None:
        """
        Initialize queue handler.

        Args:
            max_queue_size: Maximum queue size
        """
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing = False
        self._task: Optional[asyncio.Task] = None

    async def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if queueing is possible."""
        return not self.queue.full()

    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Queue operation for later processing."""
        if self.queue.full():
            raise RuntimeError("Recovery queue is full")
        
        await self.queue.put({
            "timestamp": datetime.utcnow(),
            "error": str(error),
            "context": context,
        })
        
        logger.info(
            f"Queued operation for later processing. Queue size: {self.queue.qsize()}",
            extra={"error": str(error), "queue_size": self.queue.qsize()},
        )
        
        # Start processing if not already running
        if not self.processing:
            self.processing = True
            self._task = asyncio.create_task(self._process_queue())
        
        return {"status": "queued", "queue_size": self.queue.qsize()}

    async def _process_queue(self) -> None:
        """Process queued operations."""
        while self.processing:
            try:
                if self.queue.empty():
                    await asyncio.sleep(5)
                    continue
                
                item = await self.queue.get()
                # Process item (implement actual processing logic)
                logger.info(f"Processing queued item: {item}")
                
            except Exception as e:
                logger.error(f"Error processing queue: {e}")
                await asyncio.sleep(1)

    async def stop_processing(self) -> None:
        """Stop queue processing."""
        self.processing = False
        if self._task:
            await self._task


class DegradationHandler(RecoveryHandler):
    """Handler for service degradation."""

    def __init__(self) -> None:
        """Initialize degradation handler."""
        self.degraded_features: Set[str] = set()
        self.feature_dependencies: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()

    async def register_feature(
        self,
        feature: str,
        dependencies: Optional[Set[str]] = None,
    ) -> None:
        """
        Register feature with its dependencies.

        Args:
            feature: Feature name
            dependencies: Set of dependent service names
        """
        async with self._lock:
            self.feature_dependencies[feature] = dependencies or set()

    async def degrade_feature(self, feature: str) -> None:
        """Mark feature as degraded."""
        async with self._lock:
            self.degraded_features.add(feature)
            logger.warning(f"Feature degraded: {feature}")

    async def restore_feature(self, feature: str) -> None:
        """Restore degraded feature."""
        async with self._lock:
            self.degraded_features.discard(feature)
            logger.info(f"Feature restored: {feature}")

    async def is_degraded(self, feature: str) -> bool:
        """Check if feature is degraded."""
        async with self._lock:
            return feature in self.degraded_features

    async def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if degradation is possible."""
        feature = context.get("feature")
        return feature is not None and feature in self.feature_dependencies

    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Provide degraded functionality."""
        feature = context.get("feature")
        if not feature:
            raise ValueError("No feature specified for degradation")
        
        await self.degrade_feature(feature)
        
        # Return degraded response
        return {
            "status": "degraded",
            "feature": feature,
            "message": f"Feature '{feature}' is running in degraded mode",
            "available_functions": await self._get_available_functions(feature),
        }

    async def _get_available_functions(self, feature: str) -> List[str]:
        """Get list of available functions in degraded mode."""
        # Implement logic to determine available functions
        return ["basic_operations"]


class ServiceRecoveryManager:
    """Manages service recovery and health monitoring."""

    def __init__(
        self,
        health_check_interval: float = 30.0,
        recovery_threshold: int = 3,
    ) -> None:
        """
        Initialize service recovery manager.

        Args:
            health_check_interval: Interval between health checks in seconds
            recovery_threshold: Number of successful checks before recovery
        """
        self.services: Dict[str, ServiceHealth] = {}
        self.recovery_handlers: Dict[RecoveryStrategy, RecoveryHandler] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.health_check_interval = health_check_interval
        self.recovery_threshold = recovery_threshold
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Initialize default handlers
        self._init_default_handlers()

    def _init_default_handlers(self) -> None:
        """Initialize default recovery handlers."""
        self.recovery_handlers[RecoveryStrategy.FALLBACK] = FallbackHandler()
        self.recovery_handlers[RecoveryStrategy.CACHE] = CacheHandler()
        self.recovery_handlers[RecoveryStrategy.QUEUE] = QueueHandler()
        self.recovery_handlers[RecoveryStrategy.DEGRADE] = DegradationHandler()

    async def register_service(
        self,
        name: str,
        health_check: Callable[[], bool],
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        """
        Register service for health monitoring.

        Args:
            name: Service name
            health_check: Health check function
            circuit_breaker: Optional circuit breaker
        """
        self.services[name] = ServiceHealth(
            name=name,
            status=ServiceStatus.HEALTHY,
            last_check=datetime.utcnow(),
        )
        
        if circuit_breaker:
            self.circuit_breakers[name] = circuit_breaker
        
        # Store health check function
        setattr(self.services[name], "_health_check", health_check)

    async def start_monitoring(self) -> None:
        """Start service health monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("Started service health monitoring")

    async def stop_monitoring(self) -> None:
        """Stop service health monitoring."""
        self._monitoring = False
        if self._monitor_task:
            await self._monitor_task
        logger.info("Stopped service health monitoring")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                await self._check_all_services()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _check_all_services(self) -> None:
        """Check health of all registered services."""
        for service_name, service_health in self.services.items():
            try:
                health_check = getattr(service_health, "_health_check", None)
                if not health_check:
                    continue
                
                start_time = asyncio.get_event_loop().time()
                
                if asyncio.iscoroutinefunction(health_check):
                    is_healthy = await health_check()
                else:
                    is_healthy = health_check()
                
                response_time = (asyncio.get_event_loop().time() - start_time) * 1000
                
                await self._update_service_health(
                    service_name,
                    is_healthy,
                    response_time,
                )
                
            except Exception as e:
                await self._update_service_health(
                    service_name,
                    False,
                    None,
                    str(e),
                )

    async def _update_service_health(
        self,
        service_name: str,
        is_healthy: bool,
        response_time: Optional[float],
        error: Optional[str] = None,
    ) -> None:
        """Update service health status."""
        service = self.services[service_name]
        service.last_check = datetime.utcnow()
        service.response_time_ms = response_time
        
        if is_healthy:
            service.success_count += 1
            service.last_error = None
            
            # Check for recovery
            if service.status != ServiceStatus.HEALTHY:
                recent_successes = service.success_count % (self.recovery_threshold + 1)
                if recent_successes >= self.recovery_threshold:
                    service.status = ServiceStatus.HEALTHY
                    logger.info(f"Service recovered: {service_name}")
                    
                    # Reset circuit breaker if exists
                    if service_name in self.circuit_breakers:
                        self.circuit_breakers[service_name].reset()
        else:
            service.error_count += 1
            service.last_error = error
            
            # Update status based on error rate
            if service.availability < 50:
                service.status = ServiceStatus.OFFLINE
            elif service.availability < 80:
                service.status = ServiceStatus.UNHEALTHY
            elif service.availability < 95:
                service.status = ServiceStatus.DEGRADED
            
            logger.warning(
                f"Service health check failed: {service_name}",
                extra={
                    "service": service_name,
                    "status": service.status.name,
                    "error": error,
                },
            )

    async def get_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """Get health status of specific service."""
        return self.services.get(service_name)

    async def get_all_service_health(self) -> Dict[str, ServiceHealth]:
        """Get health status of all services."""
        return dict(self.services)

    async def recover_with_strategy(
        self,
        error: Exception,
        strategy: RecoveryStrategy,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Attempt recovery using specified strategy.

        Args:
            error: The exception to recover from
            strategy: Recovery strategy to use
            context: Error context

        Returns:
            Recovery result

        Raises:
            Exception: If recovery fails
        """
        context = context or {}
        
        # Add error to aggregator
        if isinstance(error, BaseError):
            await error_aggregator.add_error(error)
        
        handler = self.recovery_handlers.get(strategy)
        if not handler:
            raise ValueError(f"No handler registered for strategy: {strategy}")
        
        if not await handler.can_recover(error, context):
            raise RuntimeError(f"Cannot recover using strategy: {strategy}")
        
        result = await handler.recover(error, context)
        
        logger.info(
            f"Successfully recovered using {strategy.name} strategy",
            extra={
                "strategy": strategy.name,
                "error": str(error),
                "context": context,
            },
        )
        
        return result


class GracefulDegradation:
    """Implements graceful degradation patterns."""

    def __init__(self) -> None:
        """Initialize graceful degradation."""
        self.feature_flags: Dict[str, bool] = {}
        self.degradation_levels: Dict[str, int] = {}
        self.max_degradation_level = 3
        self._lock = asyncio.Lock()

    async def set_feature_flag(self, feature: str, enabled: bool) -> None:
        """Set feature flag status."""
        async with self._lock:
            self.feature_flags[feature] = enabled
            logger.info(f"Feature flag '{feature}' set to {enabled}")

    async def is_feature_enabled(self, feature: str) -> bool:
        """Check if feature is enabled."""
        async with self._lock:
            return self.feature_flags.get(feature, True)

    async def increase_degradation(self, component: str) -> int:
        """
        Increase degradation level for component.

        Args:
            component: Component name

        Returns:
            New degradation level
        """
        async with self._lock:
            current_level = self.degradation_levels.get(component, 0)
            new_level = min(current_level + 1, self.max_degradation_level)
            self.degradation_levels[component] = new_level
            
            logger.warning(
                f"Increased degradation level for {component}: {new_level}",
                extra={"component": component, "level": new_level},
            )
            
            return new_level

    async def decrease_degradation(self, component: str) -> int:
        """
        Decrease degradation level for component.

        Args:
            component: Component name

        Returns:
            New degradation level
        """
        async with self._lock:
            current_level = self.degradation_levels.get(component, 0)
            new_level = max(current_level - 1, 0)
            
            if new_level == 0:
                self.degradation_levels.pop(component, None)
            else:
                self.degradation_levels[component] = new_level
            
            logger.info(
                f"Decreased degradation level for {component}: {new_level}",
                extra={"component": component, "level": new_level},
            )
            
            return new_level

    async def get_degradation_level(self, component: str) -> int:
        """Get current degradation level for component."""
        async with self._lock:
            return self.degradation_levels.get(component, 0)

    async def get_available_features(self, degradation_level: int) -> List[str]:
        """
        Get list of available features for degradation level.

        Args:
            degradation_level: Current degradation level

        Returns:
            List of available feature names
        """
        # Define features available at each level
        features_by_level = {
            0: ["all_features"],
            1: ["core_features", "basic_search", "simple_rules"],
            2: ["minimal_features", "basic_operations"],
            3: ["emergency_mode", "read_only"],
        }
        
        return features_by_level.get(degradation_level, ["emergency_mode"])


# Global instances
service_recovery_manager = ServiceRecoveryManager()
graceful_degradation = GracefulDegradation()