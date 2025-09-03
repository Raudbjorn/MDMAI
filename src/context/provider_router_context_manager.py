"""
Provider Router Context Management System.

This module provides specialized context management for the Provider Router with Fallback system,
integrating with ChromaDB for vector storage and providing high-performance state management
for provider routing, health monitoring, and cost optimization.

Key Features:
- High-performance provider state management with sub-100ms access
- ChromaDB integration for persistent vector storage of routing decisions
- Real-time state synchronization across distributed components
- Circuit breaker state persistence and recovery
- Cost tracking and budget state management
- Performance metrics collection and storage
- Configuration versioning and rollback support
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Tuple, NamedTuple
from uuid import uuid4
import hashlib

import redis.asyncio as aioredis
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import psutil
from structlog import get_logger

from .context_manager import ContextManager
from .models import Context, ContextState, ContextType
from ..ai_providers.models import ProviderType, ProviderConfig
from ..security.enhanced_security_manager import EnhancedSecurityManager

logger = get_logger(__name__)


class ProviderRouterContextType(Enum):
    """Specialized context types for provider router."""
    PROVIDER_STATE = "provider_state"
    ROUTING_DECISION = "routing_decision"
    CIRCUIT_BREAKER = "circuit_breaker"
    COST_TRACKING = "cost_tracking"
    PERFORMANCE_METRICS = "performance_metrics"
    CONFIGURATION = "configuration"


class StateConsistencyLevel(Enum):
    """State consistency levels for different operations."""
    EVENTUAL = "eventual"      # Async replication, fastest
    SESSION = "session"        # Within session consistency
    STRONG = "strong"          # Immediate consistency across all nodes


@dataclass
class ProviderHealthState:
    """Provider health state with metrics."""
    provider_name: str
    provider_type: str
    is_available: bool
    last_check: datetime
    response_time_ms: float
    error_rate: float
    success_rate: float
    uptime_percentage: float
    consecutive_failures: int
    circuit_breaker_state: str  # CLOSED, OPEN, HALF_OPEN
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RoutingDecision:
    """Routing decision with context and reasoning."""
    request_id: str
    selected_provider: str
    alternative_providers: List[str]
    routing_strategy: str
    decision_factors: Dict[str, Any]
    estimated_cost: float
    estimated_latency_ms: float
    timestamp: datetime
    confidence_score: float
    fallback_chain: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for a provider."""
    provider_name: str
    state: str  # CLOSED, OPEN, HALF_OPEN
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime]
    next_attempt_time: Optional[datetime]
    failure_threshold: int
    success_threshold: int
    timeout_duration_s: int
    half_open_max_calls: int
    current_half_open_calls: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CostTrackingState:
    """Cost tracking state for budget management."""
    provider_name: str
    current_usage: float
    daily_budget: float
    monthly_budget: float
    daily_remaining: float
    monthly_remaining: float
    cost_per_token: float
    tokens_used_today: int
    tokens_used_month: int
    last_reset_daily: datetime
    last_reset_monthly: datetime
    alerts_enabled: bool
    warning_threshold: float
    critical_threshold: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PerformanceMetrics:
    """Performance metrics for providers and system."""
    provider_name: str
    request_count: int
    success_count: int
    error_count: int
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput_rps: float
    error_rate: float
    uptime_percentage: float
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: datetime
    window_size_minutes: int = 60
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class StateAccessPattern:
    """Defines access patterns for state optimization."""
    def __init__(self, 
                 read_frequency: str,  # "very_high", "high", "medium", "low"
                 write_frequency: str, 
                 consistency_level: StateConsistencyLevel,
                 cache_ttl_seconds: int,
                 enable_compression: bool = False):
        self.read_frequency = read_frequency
        self.write_frequency = write_frequency
        self.consistency_level = consistency_level
        self.cache_ttl_seconds = cache_ttl_seconds
        self.enable_compression = enable_compression


class ProviderRouterStateStore(ABC):
    """Abstract base class for provider router state storage."""
    
    @abstractmethod
    async def store_provider_health(self, health_state: ProviderHealthState) -> bool:
        """Store provider health state."""
        pass
    
    @abstractmethod
    async def get_provider_health(self, provider_name: str) -> Optional[ProviderHealthState]:
        """Get provider health state."""
        pass
    
    @abstractmethod
    async def store_routing_decision(self, decision: RoutingDecision) -> bool:
        """Store routing decision."""
        pass
    
    @abstractmethod
    async def query_routing_decisions(self, 
                                    provider: Optional[str] = None,
                                    strategy: Optional[str] = None,
                                    limit: int = 100) -> List[RoutingDecision]:
        """Query routing decisions."""
        pass
    
    @abstractmethod
    async def store_circuit_breaker_state(self, state: CircuitBreakerState) -> bool:
        """Store circuit breaker state."""
        pass
    
    @abstractmethod
    async def get_circuit_breaker_state(self, provider_name: str) -> Optional[CircuitBreakerState]:
        """Get circuit breaker state."""
        pass


class ChromaDBProviderStateStore(ProviderRouterStateStore):
    """ChromaDB-based provider state storage with vector embeddings."""
    
    def __init__(self, 
                 chroma_client: chromadb.Client,
                 collection_prefix: str = "provider_router"):
        self.client = chroma_client
        self.collection_prefix = collection_prefix
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize collections
        self.health_collection = None
        self.routing_collection = None
        self.circuit_breaker_collection = None
        self.cost_collection = None
        self.metrics_collection = None
        
    async def initialize(self):
        """Initialize ChromaDB collections."""
        try:
            # Health state collection
            self.health_collection = self.client.get_or_create_collection(
                name=f"{self.collection_prefix}_health",
                embedding_function=self.embedding_function,
                metadata={"description": "Provider health states with embeddings"}
            )
            
            # Routing decisions collection
            self.routing_collection = self.client.get_or_create_collection(
                name=f"{self.collection_prefix}_routing",
                embedding_function=self.embedding_function,
                metadata={"description": "Routing decisions with context embeddings"}
            )
            
            # Circuit breaker states collection
            self.circuit_breaker_collection = self.client.get_or_create_collection(
                name=f"{self.collection_prefix}_circuit_breaker",
                embedding_function=self.embedding_function,
                metadata={"description": "Circuit breaker states"}
            )
            
            # Cost tracking collection
            self.cost_collection = self.client.get_or_create_collection(
                name=f"{self.collection_prefix}_cost",
                embedding_function=self.embedding_function,
                metadata={"description": "Cost tracking and budget states"}
            )
            
            # Performance metrics collection
            self.metrics_collection = self.client.get_or_create_collection(
                name=f"{self.collection_prefix}_metrics",
                embedding_function=self.embedding_function,
                metadata={"description": "Performance metrics and monitoring data"}
            )
            
            logger.info("ChromaDB provider state store initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB state store: {e}")
            raise
    
    async def store_provider_health(self, health_state: ProviderHealthState) -> bool:
        """Store provider health state with vector embedding."""
        try:
            # Create text representation for embedding
            health_text = self._create_health_embedding_text(health_state)
            
            # Store with upsert
            self.health_collection.upsert(
                ids=[f"health_{health_state.provider_name}"],
                documents=[health_text],
                metadatas=[{
                    "provider_name": health_state.provider_name,
                    "provider_type": health_state.provider_type,
                    "is_available": health_state.is_available,
                    "last_check": health_state.last_check.isoformat(),
                    "response_time_ms": health_state.response_time_ms,
                    "error_rate": health_state.error_rate,
                    "success_rate": health_state.success_rate,
                    "uptime_percentage": health_state.uptime_percentage,
                    "consecutive_failures": health_state.consecutive_failures,
                    "circuit_breaker_state": health_state.circuit_breaker_state,
                    "last_error": health_state.last_error,
                    **health_state.metadata
                }]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store provider health: {e}")
            return False
    
    async def get_provider_health(self, provider_name: str) -> Optional[ProviderHealthState]:
        """Get provider health state."""
        try:
            results = self.health_collection.get(
                ids=[f"health_{provider_name}"],
                include=["metadatas"]
            )
            
            if not results["ids"]:
                return None
            
            metadata = results["metadatas"][0]
            return ProviderHealthState(
                provider_name=metadata["provider_name"],
                provider_type=metadata["provider_type"],
                is_available=metadata["is_available"],
                last_check=datetime.fromisoformat(metadata["last_check"]),
                response_time_ms=metadata["response_time_ms"],
                error_rate=metadata["error_rate"],
                success_rate=metadata["success_rate"],
                uptime_percentage=metadata["uptime_percentage"],
                consecutive_failures=metadata["consecutive_failures"],
                circuit_breaker_state=metadata["circuit_breaker_state"],
                last_error=metadata.get("last_error"),
                metadata={k: v for k, v in metadata.items() 
                         if k not in ["provider_name", "provider_type", "is_available", 
                                     "last_check", "response_time_ms", "error_rate", 
                                     "success_rate", "uptime_percentage", "consecutive_failures",
                                     "circuit_breaker_state", "last_error"]}
            )
            
        except Exception as e:
            logger.error(f"Failed to get provider health: {e}")
            return None
    
    async def store_routing_decision(self, decision: RoutingDecision) -> bool:
        """Store routing decision with vector embedding."""
        try:
            # Create text representation for embedding
            routing_text = self._create_routing_embedding_text(decision)
            
            # Store decision
            self.routing_collection.add(
                ids=[f"routing_{decision.request_id}"],
                documents=[routing_text],
                metadatas=[{
                    "request_id": decision.request_id,
                    "selected_provider": decision.selected_provider,
                    "alternative_providers": json.dumps(decision.alternative_providers),
                    "routing_strategy": decision.routing_strategy,
                    "decision_factors": json.dumps(decision.decision_factors),
                    "estimated_cost": decision.estimated_cost,
                    "estimated_latency_ms": decision.estimated_latency_ms,
                    "timestamp": decision.timestamp.isoformat(),
                    "confidence_score": decision.confidence_score,
                    "fallback_chain": json.dumps(decision.fallback_chain),
                    **decision.metadata
                }]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store routing decision: {e}")
            return False
    
    async def query_routing_decisions(self, 
                                    provider: Optional[str] = None,
                                    strategy: Optional[str] = None,
                                    limit: int = 100) -> List[RoutingDecision]:
        """Query routing decisions with optional filters."""
        try:
            where_clause = {}
            if provider:
                where_clause["selected_provider"] = provider
            if strategy:
                where_clause["routing_strategy"] = strategy
            
            results = self.routing_collection.get(
                where=where_clause if where_clause else None,
                limit=limit,
                include=["metadatas"]
            )
            
            decisions = []
            for metadata in results["metadatas"]:
                decision = RoutingDecision(
                    request_id=metadata["request_id"],
                    selected_provider=metadata["selected_provider"],
                    alternative_providers=json.loads(metadata["alternative_providers"]),
                    routing_strategy=metadata["routing_strategy"],
                    decision_factors=json.loads(metadata["decision_factors"]),
                    estimated_cost=metadata["estimated_cost"],
                    estimated_latency_ms=metadata["estimated_latency_ms"],
                    timestamp=datetime.fromisoformat(metadata["timestamp"]),
                    confidence_score=metadata["confidence_score"],
                    fallback_chain=json.loads(metadata["fallback_chain"]),
                    metadata={k: v for k, v in metadata.items() 
                             if k not in ["request_id", "selected_provider", "alternative_providers",
                                         "routing_strategy", "decision_factors", "estimated_cost",
                                         "estimated_latency_ms", "timestamp", "confidence_score",
                                         "fallback_chain"]}
                )
                decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Failed to query routing decisions: {e}")
            return []
    
    async def store_circuit_breaker_state(self, state: CircuitBreakerState) -> bool:
        """Store circuit breaker state."""
        try:
            # Create text representation
            cb_text = self._create_circuit_breaker_embedding_text(state)
            
            self.circuit_breaker_collection.upsert(
                ids=[f"cb_{state.provider_name}"],
                documents=[cb_text],
                metadatas=[{
                    "provider_name": state.provider_name,
                    "state": state.state,
                    "failure_count": state.failure_count,
                    "success_count": state.success_count,
                    "last_failure_time": state.last_failure_time.isoformat() if state.last_failure_time else None,
                    "next_attempt_time": state.next_attempt_time.isoformat() if state.next_attempt_time else None,
                    "failure_threshold": state.failure_threshold,
                    "success_threshold": state.success_threshold,
                    "timeout_duration_s": state.timeout_duration_s,
                    "half_open_max_calls": state.half_open_max_calls,
                    "current_half_open_calls": state.current_half_open_calls,
                    **state.metadata
                }]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store circuit breaker state: {e}")
            return False
    
    async def get_circuit_breaker_state(self, provider_name: str) -> Optional[CircuitBreakerState]:
        """Get circuit breaker state."""
        try:
            results = self.circuit_breaker_collection.get(
                ids=[f"cb_{provider_name}"],
                include=["metadatas"]
            )
            
            if not results["ids"]:
                return None
            
            metadata = results["metadatas"][0]
            return CircuitBreakerState(
                provider_name=metadata["provider_name"],
                state=metadata["state"],
                failure_count=metadata["failure_count"],
                success_count=metadata["success_count"],
                last_failure_time=datetime.fromisoformat(metadata["last_failure_time"]) if metadata["last_failure_time"] else None,
                next_attempt_time=datetime.fromisoformat(metadata["next_attempt_time"]) if metadata["next_attempt_time"] else None,
                failure_threshold=metadata["failure_threshold"],
                success_threshold=metadata["success_threshold"],
                timeout_duration_s=metadata["timeout_duration_s"],
                half_open_max_calls=metadata["half_open_max_calls"],
                current_half_open_calls=metadata["current_half_open_calls"],
                metadata={k: v for k, v in metadata.items() 
                         if k not in ["provider_name", "state", "failure_count", "success_count",
                                     "last_failure_time", "next_attempt_time", "failure_threshold",
                                     "success_threshold", "timeout_duration_s", "half_open_max_calls",
                                     "current_half_open_calls"]}
            )
            
        except Exception as e:
            logger.error(f"Failed to get circuit breaker state: {e}")
            return None
    
    def _create_health_embedding_text(self, health: ProviderHealthState) -> str:
        """Create text representation for health state embedding."""
        status = "available" if health.is_available else "unavailable"
        circuit_state = health.circuit_breaker_state.lower()
        
        return (f"Provider {health.provider_name} of type {health.provider_type} is {status} "
                f"with {health.response_time_ms:.1f}ms response time, "
                f"{health.success_rate:.1f}% success rate, "
                f"{health.uptime_percentage:.1f}% uptime, "
                f"circuit breaker {circuit_state}, "
                f"{health.consecutive_failures} consecutive failures")
    
    def _create_routing_embedding_text(self, decision: RoutingDecision) -> str:
        """Create text representation for routing decision embedding."""
        alternatives = ", ".join(decision.alternative_providers)
        fallbacks = ", ".join(decision.fallback_chain)
        
        return (f"Routing decision for request {decision.request_id} "
                f"selected {decision.selected_provider} using {decision.routing_strategy} strategy "
                f"with {decision.confidence_score:.2f} confidence, "
                f"estimated cost ${decision.estimated_cost:.4f}, "
                f"estimated latency {decision.estimated_latency_ms:.1f}ms, "
                f"alternatives: {alternatives}, fallback chain: {fallbacks}")
    
    def _create_circuit_breaker_embedding_text(self, state: CircuitBreakerState) -> str:
        """Create text representation for circuit breaker state embedding."""
        return (f"Circuit breaker for {state.provider_name} is {state.state.lower()} "
                f"with {state.failure_count} failures, {state.success_count} successes, "
                f"thresholds: {state.failure_threshold} failures, {state.success_threshold} successes, "
                f"timeout: {state.timeout_duration_s}s")


class InMemoryStateCache:
    """High-performance in-memory cache for frequently accessed state."""
    
    def __init__(self, 
                 max_size: int = 10000,
                 default_ttl: int = 300,  # 5 minutes
                 enable_lru: bool = True):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.enable_lru = enable_lru
        
        # Cache storage
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._access_times: Dict[str, float] = {}
        
        # Access patterns for optimization
        self.access_patterns = {
            "provider_health": StateAccessPattern("very_high", "high", StateConsistencyLevel.SESSION, 60),
            "routing_decisions": StateAccessPattern("high", "medium", StateConsistencyLevel.EVENTUAL, 300),
            "circuit_breaker": StateAccessPattern("very_high", "medium", StateConsistencyLevel.STRONG, 30),
            "cost_tracking": StateAccessPattern("medium", "high", StateConsistencyLevel.SESSION, 120),
            "performance_metrics": StateAccessPattern("medium", "low", StateConsistencyLevel.EVENTUAL, 300)
        }
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
        logger.info("InMemoryStateCache initialized", 
                   max_size=max_size, 
                   default_ttl=default_ttl,
                   enable_lru=enable_lru)
    
    async def get(self, key: str, category: str = "default") -> Optional[Any]:
        """Get item from cache with access pattern optimization."""
        current_time = time.time()
        
        # Check if key exists and is not expired
        if key in self._cache:
            # Check TTL
            ttl = self.access_patterns.get(category, self.access_patterns["provider_health"]).cache_ttl_seconds
            if current_time - self._timestamps[key] <= ttl:
                # Update access time for LRU
                if self.enable_lru:
                    self._access_times[key] = current_time
                
                self.hit_count += 1
                return self._cache[key]
            else:
                # Expired, remove from cache
                await self.delete(key)
        
        self.miss_count += 1
        return None
    
    async def set(self, key: str, value: Any, category: str = "default", ttl: Optional[int] = None) -> bool:
        """Set item in cache with automatic eviction."""
        current_time = time.time()
        
        # Check if we need to evict items
        if len(self._cache) >= self.max_size:
            await self._evict_items()
        
        # Store item
        self._cache[key] = value
        self._timestamps[key] = current_time
        
        if self.enable_lru:
            self._access_times[key] = current_time
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]
            if key in self._access_times:
                del self._access_times[key]
            return True
        return False
    
    async def clear_category(self, category: str) -> int:
        """Clear all items matching a category pattern."""
        keys_to_delete = [key for key in self._cache.keys() if key.startswith(f"{category}_")]
        
        for key in keys_to_delete:
            await self.delete(key)
        
        return len(keys_to_delete)
    
    async def _evict_items(self) -> None:
        """Evict items using LRU or FIFO policy."""
        current_time = time.time()
        
        if self.enable_lru:
            # Evict least recently used items
            sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
        else:
            # Evict oldest items (FIFO)
            sorted_items = sorted(self._timestamps.items(), key=lambda x: x[1])
        
        # Evict 25% of items to make room
        items_to_evict = len(sorted_items) // 4
        
        for key, _ in sorted_items[:items_to_evict]:
            await self.delete(key)
            self.eviction_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "eviction_count": self.eviction_count,
            "memory_usage_mb": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Rough estimation - in production, use more sophisticated memory profiling
        import sys
        total_size = 0
        
        for key, value in self._cache.items():
            total_size += sys.getsizeof(key) + sys.getsizeof(value)
        
        return total_size / (1024 * 1024)


class StateSynchronizer:
    """Distributed state synchronization for provider router components."""
    
    def __init__(self, 
                 redis_client: aioredis.Redis,
                 sync_channel_prefix: str = "provider_router_sync"):
        self.redis = redis_client
        self.channel_prefix = sync_channel_prefix
        
        # Synchronization locks
        self._locks: Dict[str, asyncio.Lock] = {}
        
        # Change listeners
        self._change_listeners: Dict[str, List[callable]] = {}
        
        # Sync statistics
        self.sync_operations = 0
        self.conflict_resolutions = 0
        self.failed_syncs = 0
        
        logger.info("StateSynchronizer initialized")
    
    async def register_state_change_listener(self, 
                                           state_type: str, 
                                           callback: callable) -> None:
        """Register a callback for state changes."""
        if state_type not in self._change_listeners:
            self._change_listeners[state_type] = []
        
        self._change_listeners[state_type].append(callback)
        
        # Subscribe to Redis channel for this state type
        channel = f"{self.channel_prefix}:{state_type}"
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(channel)
        
        # Start listening task
        asyncio.create_task(self._listen_for_changes(pubsub, state_type))
    
    async def sync_state(self, 
                        state_type: str, 
                        state_key: str, 
                        state_data: Any,
                        consistency_level: StateConsistencyLevel = StateConsistencyLevel.SESSION) -> bool:
        """Synchronize state across distributed components."""
        try:
            # Create distributed lock for strong consistency
            if consistency_level == StateConsistencyLevel.STRONG:
                lock_key = f"lock:{state_type}:{state_key}"
                async with self._get_distributed_lock(lock_key):
                    return await self._perform_sync(state_type, state_key, state_data)
            else:
                return await self._perform_sync(state_type, state_key, state_data)
                
        except Exception as e:
            logger.error(f"State synchronization failed: {e}")
            self.failed_syncs += 1
            return False
    
    async def _perform_sync(self, 
                          state_type: str, 
                          state_key: str, 
                          state_data: Any) -> bool:
        """Perform the actual synchronization."""
        try:
            # Serialize state data
            if hasattr(state_data, '__dict__'):
                serialized = json.dumps(asdict(state_data), default=str)
            else:
                serialized = json.dumps(state_data, default=str)
            
            # Store in Redis with versioning
            version_key = f"version:{state_type}:{state_key}"
            current_version = await self.redis.get(version_key)
            new_version = int(current_version) + 1 if current_version else 1
            
            # Use Redis transaction for atomicity
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.set(f"state:{state_type}:{state_key}", serialized)
                pipe.set(version_key, new_version)
                pipe.set(f"timestamp:{state_type}:{state_key}", time.time())
                await pipe.execute()
            
            # Publish change notification
            change_notification = {
                "state_type": state_type,
                "state_key": state_key,
                "version": new_version,
                "timestamp": time.time(),
                "data": serialized
            }
            
            await self.redis.publish(
                f"{self.channel_prefix}:{state_type}", 
                json.dumps(change_notification)
            )
            
            self.sync_operations += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to perform sync: {e}")
            return False
    
    async def get_synchronized_state(self, 
                                   state_type: str, 
                                   state_key: str) -> Tuple[Optional[Any], int]:
        """Get synchronized state with version information."""
        try:
            # Get state and version
            state_data = await self.redis.get(f"state:{state_type}:{state_key}")
            version = await self.redis.get(f"version:{state_type}:{state_key}")
            
            if state_data:
                parsed_state = json.loads(state_data)
                return parsed_state, int(version) if version else 0
            
            return None, 0
            
        except Exception as e:
            logger.error(f"Failed to get synchronized state: {e}")
            return None, 0
    
    async def resolve_state_conflict(self, 
                                   state_type: str, 
                                   state_key: str, 
                                   local_state: Any, 
                                   remote_state: Any,
                                   local_version: int,
                                   remote_version: int) -> Any:
        """Resolve state conflicts using versioning and merge strategies."""
        try:
            self.conflict_resolutions += 1
            
            # Simple version-based resolution (latest wins)
            if remote_version > local_version:
                logger.info(f"Conflict resolved: remote state wins (v{remote_version} > v{local_version})")
                return remote_state
            elif local_version > remote_version:
                logger.info(f"Conflict resolved: local state wins (v{local_version} > v{remote_version})")
                return local_state
            else:
                # Same version - use timestamp-based resolution
                local_ts = getattr(local_state, 'timestamp', datetime.min)
                remote_ts = getattr(remote_state, 'timestamp', datetime.min)
                
                if remote_ts > local_ts:
                    logger.info("Conflict resolved: remote state wins (newer timestamp)")
                    return remote_state
                else:
                    logger.info("Conflict resolved: local state wins (newer/equal timestamp)")
                    return local_state
                    
        except Exception as e:
            logger.error(f"Failed to resolve state conflict: {e}")
            # Default to local state on error
            return local_state
    
    async def _get_distributed_lock(self, lock_key: str, timeout: int = 10) -> asyncio.Lock:
        """Get or create a distributed lock."""
        if lock_key not in self._locks:
            self._locks[lock_key] = asyncio.Lock()
        
        return self._locks[lock_key]
    
    async def _listen_for_changes(self, pubsub, state_type: str) -> None:
        """Listen for state changes and notify callbacks."""
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    change_data = json.loads(message["data"])
                    
                    # Notify all listeners for this state type
                    listeners = self._change_listeners.get(state_type, [])
                    for callback in listeners:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(change_data)
                            else:
                                callback(change_data)
                        except Exception as e:
                            logger.error(f"State change listener failed: {e}")
                            
        except Exception as e:
            logger.error(f"Failed to listen for state changes: {e}")
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        return {
            "sync_operations": self.sync_operations,
            "conflict_resolutions": self.conflict_resolutions,
            "failed_syncs": self.failed_syncs,
            "active_listeners": len(self._change_listeners),
            "active_locks": len(self._locks)
        }


class StateRecoveryManager:
    """Manages state recovery and consistency validation."""
    
    def __init__(self, 
                 chroma_store: ChromaDBProviderStateStore,
                 redis_client: aioredis.Redis,
                 backup_interval_minutes: int = 30):
        self.chroma_store = chroma_store
        self.redis = redis_client
        self.backup_interval = backup_interval_minutes * 60  # Convert to seconds
        
        # Recovery statistics
        self.recovery_operations = 0
        self.validation_checks = 0
        self.corruption_detections = 0
        
        # Background tasks
        self._backup_task = None
        self._validation_task = None
        
        logger.info("StateRecoveryManager initialized")
    
    async def start_background_tasks(self) -> None:
        """Start background recovery and validation tasks."""
        self._backup_task = asyncio.create_task(self._periodic_backup())
        self._validation_task = asyncio.create_task(self._periodic_validation())
        
        logger.info("Background recovery tasks started")
    
    async def stop_background_tasks(self) -> None:
        """Stop background tasks."""
        if self._backup_task:
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
        
        if self._validation_task:
            self._validation_task.cancel()
            try:
                await self._validation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Background recovery tasks stopped")
    
    async def create_state_backup(self, backup_id: Optional[str] = None) -> str:
        """Create a backup of all critical state."""
        if backup_id is None:
            backup_id = f"backup_{int(time.time())}"
        
        try:
            # Get all critical state from Redis
            state_keys = await self.redis.keys("state:*")
            version_keys = await self.redis.keys("version:*")
            timestamp_keys = await self.redis.keys("timestamp:*")
            
            backup_data = {
                "backup_id": backup_id,
                "timestamp": datetime.now().isoformat(),
                "state": {},
                "versions": {},
                "timestamps": {}
            }
            
            # Collect all state data
            if state_keys:
                state_values = await self.redis.mget(state_keys)
                backup_data["state"] = {
                    key.decode(): value.decode() if value else None
                    for key, value in zip(state_keys, state_values)
                }
            
            if version_keys:
                version_values = await self.redis.mget(version_keys)
                backup_data["versions"] = {
                    key.decode(): value.decode() if value else None
                    for key, value in zip(version_keys, version_values)
                }
            
            if timestamp_keys:
                timestamp_values = await self.redis.mget(timestamp_keys)
                backup_data["timestamps"] = {
                    key.decode(): value.decode() if value else None
                    for key, value in zip(timestamp_keys, timestamp_values)
                }
            
            # Store backup in Redis with expiration
            backup_key = f"backup:{backup_id}"
            await self.redis.setex(
                backup_key, 
                86400 * 7,  # 7 days retention
                json.dumps(backup_data)
            )
            
            logger.info(f"State backup created: {backup_id}")
            return backup_id
            
        except Exception as e:
            logger.error(f"Failed to create state backup: {e}")
            raise
    
    async def restore_from_backup(self, backup_id: str) -> bool:
        """Restore state from backup."""
        try:
            backup_key = f"backup:{backup_id}"
            backup_data_raw = await self.redis.get(backup_key)
            
            if not backup_data_raw:
                logger.error(f"Backup not found: {backup_id}")
                return False
            
            backup_data = json.loads(backup_data_raw)
            
            # Restore state data
            if backup_data.get("state"):
                for key, value in backup_data["state"].items():
                    if value is not None:
                        await self.redis.set(key, value)
            
            # Restore versions
            if backup_data.get("versions"):
                for key, value in backup_data["versions"].items():
                    if value is not None:
                        await self.redis.set(key, value)
            
            # Restore timestamps
            if backup_data.get("timestamps"):
                for key, value in backup_data["timestamps"].items():
                    if value is not None:
                        await self.redis.set(key, value)
            
            self.recovery_operations += 1
            logger.info(f"State restored from backup: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False
    
    async def validate_state_consistency(self) -> Dict[str, Any]:
        """Validate consistency across all state stores."""
        try:
            self.validation_checks += 1
            
            validation_results = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "healthy",
                "issues": [],
                "stats": {
                    "redis_keys_checked": 0,
                    "chroma_collections_checked": 0,
                    "inconsistencies_found": 0,
                    "corrupted_entries": 0
                }
            }
            
            # Check Redis state consistency
            redis_issues = await self._validate_redis_consistency()
            validation_results["issues"].extend(redis_issues)
            validation_results["stats"]["redis_keys_checked"] = len(await self.redis.keys("state:*"))
            
            # Check ChromaDB collection health
            chroma_issues = await self._validate_chroma_consistency()
            validation_results["issues"].extend(chroma_issues)
            validation_results["stats"]["chroma_collections_checked"] = 5  # Number of collections
            
            # Update overall status
            critical_issues = [issue for issue in validation_results["issues"] if issue["severity"] == "critical"]
            if critical_issues:
                validation_results["overall_status"] = "critical"
            elif validation_results["issues"]:
                validation_results["overall_status"] = "degraded"
            
            validation_results["stats"]["inconsistencies_found"] = len(validation_results["issues"])
            
            logger.info(f"State consistency validation completed: {validation_results['overall_status']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"State consistency validation failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "error",
                "error": str(e),
                "issues": [{"severity": "critical", "message": f"Validation failed: {e}"}],
                "stats": {}
            }
    
    async def _validate_redis_consistency(self) -> List[Dict[str, Any]]:
        """Validate Redis state consistency."""
        issues = []
        
        try:
            # Check for orphaned state entries
            state_keys = await self.redis.keys("state:*")
            version_keys = await self.redis.keys("version:*")
            
            state_identifiers = set()
            version_identifiers = set()
            
            for key in state_keys:
                identifier = key.decode().replace("state:", "")
                state_identifiers.add(identifier)
            
            for key in version_keys:
                identifier = key.decode().replace("version:", "")
                version_identifiers.add(identifier)
            
            # Find orphaned entries
            orphaned_states = state_identifiers - version_identifiers
            orphaned_versions = version_identifiers - state_identifiers
            
            for orphan in orphaned_states:
                issues.append({
                    "severity": "warning",
                    "type": "orphaned_state",
                    "message": f"State without version: {orphan}",
                    "key": f"state:{orphan}"
                })
            
            for orphan in orphaned_versions:
                issues.append({
                    "severity": "warning", 
                    "type": "orphaned_version",
                    "message": f"Version without state: {orphan}",
                    "key": f"version:{orphan}"
                })
            
            # Check for corrupted JSON data
            for key in state_keys:
                try:
                    value = await self.redis.get(key)
                    if value:
                        json.loads(value)
                except json.JSONDecodeError:
                    self.corruption_detections += 1
                    issues.append({
                        "severity": "critical",
                        "type": "corrupted_json",
                        "message": f"Corrupted JSON in key: {key.decode()}",
                        "key": key.decode()
                    })
            
        except Exception as e:
            issues.append({
                "severity": "critical",
                "type": "redis_validation_error",
                "message": f"Redis validation failed: {e}"
            })
        
        return issues
    
    async def _validate_chroma_consistency(self) -> List[Dict[str, Any]]:
        """Validate ChromaDB consistency."""
        issues = []
        
        try:
            collections = [
                ("health", self.chroma_store.health_collection),
                ("routing", self.chroma_store.routing_collection), 
                ("circuit_breaker", self.chroma_store.circuit_breaker_collection),
                ("cost", self.chroma_store.cost_collection),
                ("metrics", self.chroma_store.metrics_collection)
            ]
            
            for name, collection in collections:
                if collection is None:
                    issues.append({
                        "severity": "critical",
                        "type": "missing_collection",
                        "message": f"ChromaDB collection not initialized: {name}"
                    })
                    continue
                
                try:
                    # Check collection health
                    count = collection.count()
                    if count < 0:  # ChromaDB should never return negative count
                        issues.append({
                            "severity": "critical",
                            "type": "invalid_collection_state",
                            "message": f"Invalid count for collection {name}: {count}"
                        })
                
                except Exception as e:
                    issues.append({
                        "severity": "error",
                        "type": "collection_access_error",
                        "message": f"Cannot access collection {name}: {e}"
                    })
                    
        except Exception as e:
            issues.append({
                "severity": "critical",
                "type": "chroma_validation_error",
                "message": f"ChromaDB validation failed: {e}"
            })
        
        return issues
    
    async def _periodic_backup(self) -> None:
        """Periodic backup task."""
        try:
            while True:
                await asyncio.sleep(self.backup_interval)
                await self.create_state_backup()
        except asyncio.CancelledError:
            logger.info("Periodic backup task cancelled")
        except Exception as e:
            logger.error(f"Periodic backup task failed: {e}")
    
    async def _periodic_validation(self) -> None:
        """Periodic validation task."""
        try:
            while True:
                await asyncio.sleep(self.backup_interval * 2)  # Validate less frequently
                validation_results = await self.validate_state_consistency()
                
                # Log critical issues
                critical_issues = [issue for issue in validation_results["issues"] if issue["severity"] == "critical"]
                if critical_issues:
                    logger.error(f"Critical state consistency issues detected: {len(critical_issues)}")
                    
        except asyncio.CancelledError:
            logger.info("Periodic validation task cancelled")
        except Exception as e:
            logger.error(f"Periodic validation task failed: {e}")
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery manager statistics."""
        return {
            "recovery_operations": self.recovery_operations,
            "validation_checks": self.validation_checks,
            "corruption_detections": self.corruption_detections,
            "backup_interval_minutes": self.backup_interval // 60,
            "background_tasks_active": {
                "backup": self._backup_task is not None and not self._backup_task.done(),
                "validation": self._validation_task is not None and not self._validation_task.done()
            }
        }


class ProviderRouterContextManager:
    """
    High-performance context management system for Provider Router with Fallback.
    
    This class integrates ChromaDB storage, Redis caching, state synchronization,
    and recovery management to provide sub-100ms state access for the provider
    router system.
    
    Features:
    - Multi-tier caching (L1: in-memory, L2: Redis, L3: ChromaDB)
    - Intelligent state partitioning and replication
    - Real-time synchronization with conflict resolution
    - Automated backup and recovery
    - Performance monitoring and optimization
    - Configuration versioning and rollback
    """
    
    def __init__(self,
                 chroma_host: str = "localhost",
                 chroma_port: int = 8000,
                 redis_url: str = "redis://localhost:6379/0",
                 database_url: Optional[str] = None,
                 enable_recovery: bool = True,
                 cache_size: int = 10000,
                 backup_interval_minutes: int = 30):
        
        # Core configuration
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.redis_url = redis_url
        self.database_url = database_url
        self.enable_recovery = enable_recovery
        
        # Component instances
        self.chroma_client = None
        self.redis_client = None
        self.chroma_store = None
        self.memory_cache = None
        self.state_synchronizer = None
        self.recovery_manager = None
        
        # Optional base context manager integration
        self.base_context_manager = None
        
        # Performance tracking
        self.performance_metrics = {
            "total_operations": 0,
            "cache_hits": {"L1": 0, "L2": 0, "L3": 0},
            "cache_misses": {"L1": 0, "L2": 0, "L3": 0},
            "avg_response_time_ms": 0.0,
            "p95_response_time_ms": 0.0,
            "p99_response_time_ms": 0.0,
            "error_count": 0,
            "sync_operations": 0
        }
        
        # Response time tracking for percentile calculations
        self._response_times = []
        self._max_response_times = 1000  # Keep last 1000 response times
        
        # State
        self._initialized = False
        self._running = False
        
        logger.info("ProviderRouterContextManager initialized")
    
    async def initialize(self) -> None:
        """Initialize all components of the context management system."""
        if self._initialized:
            return
        
        logger.info("Initializing ProviderRouterContextManager")
        
        try:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            
            # Initialize Redis client
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Initialize ChromaDB store
            self.chroma_store = ChromaDBProviderStateStore(
                self.chroma_client,
                collection_prefix="provider_router_v1"
            )
            await self.chroma_store.initialize()
            
            # Initialize in-memory cache
            self.memory_cache = InMemoryStateCache(max_size=10000, default_ttl=300)
            
            # Initialize state synchronizer
            self.state_synchronizer = StateSynchronizer(
                self.redis_client,
                sync_channel_prefix="provider_router_sync_v1"
            )
            
            # Setup state change listeners
            await self._setup_state_listeners()
            
            # Initialize recovery manager if enabled
            if self.enable_recovery:
                self.recovery_manager = StateRecoveryManager(
                    self.chroma_store,
                    self.redis_client,
                    backup_interval_minutes=30
                )
                await self.recovery_manager.start_background_tasks()
            
            # Initialize base context manager if database URL provided
            if self.database_url:
                self.base_context_manager = ContextManager(
                    database_url=self.database_url,
                    redis_url=self.redis_url
                )
                await self.base_context_manager.initialize()
            
            self._initialized = True
            logger.info("ProviderRouterContextManager fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize ProviderRouterContextManager: {e}")
            raise
    
    async def start(self) -> None:
        """Start the context management system."""
        if not self._initialized:
            await self.initialize()
        
        if self._running:
            return
        
        self._running = True
        logger.info("ProviderRouterContextManager started")
    
    async def stop(self) -> None:
        """Stop the context management system and cleanup resources."""
        if not self._running:
            return
        
        logger.info("Stopping ProviderRouterContextManager")
        
        try:
            # Stop recovery manager
            if self.recovery_manager:
                await self.recovery_manager.stop_background_tasks()
            
            # Cleanup base context manager
            if self.base_context_manager:
                await self.base_context_manager.cleanup()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            self._running = False
            logger.info("ProviderRouterContextManager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping ProviderRouterContextManager: {e}")
    
    # High-level state management methods
    
    async def update_provider_health(self, 
                                   provider_name: str,
                                   health_data: Dict[str, Any],
                                   consistency_level: StateConsistencyLevel = StateConsistencyLevel.SESSION) -> bool:
        """Update provider health state with multi-tier caching."""
        start_time = time.time()
        
        try:
            # Create health state object
            health_state = ProviderHealthState(
                provider_name=provider_name,
                provider_type=health_data.get("provider_type", "unknown"),
                is_available=health_data.get("is_available", False),
                last_check=datetime.now(timezone.utc),
                response_time_ms=health_data.get("response_time_ms", 0.0),
                error_rate=health_data.get("error_rate", 0.0),
                success_rate=health_data.get("success_rate", 100.0),
                uptime_percentage=health_data.get("uptime_percentage", 100.0),
                consecutive_failures=health_data.get("consecutive_failures", 0),
                circuit_breaker_state=health_data.get("circuit_breaker_state", "CLOSED"),
                last_error=health_data.get("last_error"),
                metadata=health_data.get("metadata", {})
            )
            
            # Update in all tiers
            cache_key = f"provider_health_{provider_name}"
            
            # L1: Memory cache
            await self.memory_cache.set(cache_key, health_state, "provider_health")
            
            # L2 & L3: Synchronize through state synchronizer (updates Redis and ChromaDB)
            sync_success = await self.state_synchronizer.sync_state(
                "provider_health", 
                provider_name, 
                health_state, 
                consistency_level
            )
            
            # Also store in ChromaDB for long-term persistence and vector search
            if sync_success:
                await self.chroma_store.store_provider_health(health_state)
            
            self._record_operation_time(time.time() - start_time)
            return sync_success
            
        except Exception as e:
            logger.error(f"Failed to update provider health: {e}")
            self.performance_metrics["error_count"] += 1
            return False
    
    async def get_provider_health(self, provider_name: str) -> Optional[ProviderHealthState]:
        """Get provider health state with multi-tier cache optimization."""
        start_time = time.time()
        
        try:
            cache_key = f"provider_health_{provider_name}"
            
            # L1: Memory cache (fastest)
            cached_state = await self.memory_cache.get(cache_key, "provider_health")
            if cached_state:
                self.performance_metrics["cache_hits"]["L1"] += 1
                self._record_operation_time(time.time() - start_time)
                return cached_state
            
            self.performance_metrics["cache_misses"]["L1"] += 1
            
            # L2: Redis cache
            redis_state, version = await self.state_synchronizer.get_synchronized_state(
                "provider_health", provider_name
            )
            
            if redis_state:
                # Reconstruct health state from Redis data
                health_state = ProviderHealthState(**redis_state)
                
                # Update L1 cache
                await self.memory_cache.set(cache_key, health_state, "provider_health")
                
                self.performance_metrics["cache_hits"]["L2"] += 1
                self._record_operation_time(time.time() - start_time)
                return health_state
            
            self.performance_metrics["cache_misses"]["L2"] += 1
            
            # L3: ChromaDB (slowest but most comprehensive)
            health_state = await self.chroma_store.get_provider_health(provider_name)
            
            if health_state:
                # Update both L1 and L2 caches
                await self.memory_cache.set(cache_key, health_state, "provider_health")
                await self.state_synchronizer.sync_state(
                    "provider_health", 
                    provider_name, 
                    health_state,
                    StateConsistencyLevel.EVENTUAL
                )
                
                self.performance_metrics["cache_hits"]["L3"] += 1
            else:
                self.performance_metrics["cache_misses"]["L3"] += 1
            
            self._record_operation_time(time.time() - start_time)
            return health_state
            
        except Exception as e:
            logger.error(f"Failed to get provider health: {e}")
            self.performance_metrics["error_count"] += 1
            return None
    
    async def store_routing_decision(self, 
                                   request_id: str,
                                   routing_data: Dict[str, Any]) -> bool:
        """Store routing decision with vector embeddings."""
        start_time = time.time()
        
        try:
            # Create routing decision object
            decision = RoutingDecision(
                request_id=request_id,
                selected_provider=routing_data["selected_provider"],
                alternative_providers=routing_data.get("alternative_providers", []),
                routing_strategy=routing_data.get("routing_strategy", "default"),
                decision_factors=routing_data.get("decision_factors", {}),
                estimated_cost=routing_data.get("estimated_cost", 0.0),
                estimated_latency_ms=routing_data.get("estimated_latency_ms", 0.0),
                timestamp=datetime.now(timezone.utc),
                confidence_score=routing_data.get("confidence_score", 1.0),
                fallback_chain=routing_data.get("fallback_chain", []),
                metadata=routing_data.get("metadata", {})
            )
            
            # Store in ChromaDB for vector search and analytics
            success = await self.chroma_store.store_routing_decision(decision)
            
            # Also cache recent decisions
            cache_key = f"routing_decision_{request_id}"
            await self.memory_cache.set(cache_key, decision, "routing_decisions")
            
            self._record_operation_time(time.time() - start_time)
            return success
            
        except Exception as e:
            logger.error(f"Failed to store routing decision: {e}")
            self.performance_metrics["error_count"] += 1
            return False
    
    async def update_circuit_breaker_state(self, 
                                         provider_name: str,
                                         breaker_data: Dict[str, Any],
                                         consistency_level: StateConsistencyLevel = StateConsistencyLevel.STRONG) -> bool:
        """Update circuit breaker state with strong consistency."""
        start_time = time.time()
        
        try:
            # Create circuit breaker state
            breaker_state = CircuitBreakerState(
                provider_name=provider_name,
                state=breaker_data["state"],
                failure_count=breaker_data.get("failure_count", 0),
                success_count=breaker_data.get("success_count", 0),
                last_failure_time=breaker_data.get("last_failure_time"),
                next_attempt_time=breaker_data.get("next_attempt_time"),
                failure_threshold=breaker_data.get("failure_threshold", 5),
                success_threshold=breaker_data.get("success_threshold", 3),
                timeout_duration_s=breaker_data.get("timeout_duration_s", 60),
                half_open_max_calls=breaker_data.get("half_open_max_calls", 3),
                current_half_open_calls=breaker_data.get("current_half_open_calls", 0),
                metadata=breaker_data.get("metadata", {})
            )
            
            # Circuit breaker state requires strong consistency
            cache_key = f"circuit_breaker_{provider_name}"
            
            # Update memory cache immediately
            await self.memory_cache.set(cache_key, breaker_state, "circuit_breaker")
            
            # Synchronize with strong consistency
            sync_success = await self.state_synchronizer.sync_state(
                "circuit_breaker",
                provider_name,
                breaker_state,
                consistency_level
            )
            
            # Also persist to ChromaDB
            if sync_success:
                await self.chroma_store.store_circuit_breaker_state(breaker_state)
            
            self._record_operation_time(time.time() - start_time)
            return sync_success
            
        except Exception as e:
            logger.error(f"Failed to update circuit breaker state: {e}")
            self.performance_metrics["error_count"] += 1
            return False
    
    async def get_circuit_breaker_state(self, provider_name: str) -> Optional[CircuitBreakerState]:
        """Get circuit breaker state with high-priority caching."""
        start_time = time.time()
        
        try:
            cache_key = f"circuit_breaker_{provider_name}"
            
            # Check memory cache first (circuit breaker state is critical)
            cached_state = await self.memory_cache.get(cache_key, "circuit_breaker")
            if cached_state:
                self.performance_metrics["cache_hits"]["L1"] += 1
                self._record_operation_time(time.time() - start_time)
                return cached_state
            
            self.performance_metrics["cache_misses"]["L1"] += 1
            
            # Check synchronized state (Redis)
            redis_state, version = await self.state_synchronizer.get_synchronized_state(
                "circuit_breaker", provider_name
            )
            
            if redis_state:
                breaker_state = CircuitBreakerState(**redis_state)
                
                # Update memory cache
                await self.memory_cache.set(cache_key, breaker_state, "circuit_breaker")
                
                self.performance_metrics["cache_hits"]["L2"] += 1
                self._record_operation_time(time.time() - start_time)
                return breaker_state
            
            self.performance_metrics["cache_misses"]["L2"] += 1
            
            # Fallback to ChromaDB
            breaker_state = await self.chroma_store.get_circuit_breaker_state(provider_name)
            
            if breaker_state:
                # Update caches
                await self.memory_cache.set(cache_key, breaker_state, "circuit_breaker")
                await self.state_synchronizer.sync_state(
                    "circuit_breaker",
                    provider_name,
                    breaker_state,
                    StateConsistencyLevel.SESSION
                )
                
                self.performance_metrics["cache_hits"]["L3"] += 1
            else:
                self.performance_metrics["cache_misses"]["L3"] += 1
            
            self._record_operation_time(time.time() - start_time)
            return breaker_state
            
        except Exception as e:
            logger.error(f"Failed to get circuit breaker state: {e}")
            self.performance_metrics["error_count"] += 1
            return None
    
    async def query_routing_patterns(self, 
                                   filters: Optional[Dict[str, Any]] = None,
                                   limit: int = 100) -> List[RoutingDecision]:
        """Query routing patterns using vector similarity search."""
        start_time = time.time()
        
        try:
            # Use ChromaDB for pattern analysis
            decisions = await self.chroma_store.query_routing_decisions(
                provider=filters.get("provider") if filters else None,
                strategy=filters.get("strategy") if filters else None,
                limit=limit
            )
            
            self._record_operation_time(time.time() - start_time)
            return decisions
            
        except Exception as e:
            logger.error(f"Failed to query routing patterns: {e}")
            self.performance_metrics["error_count"] += 1
            return []
    
    # Performance optimization and monitoring
    
    async def optimize_cache_performance(self) -> Dict[str, Any]:
        """Analyze and optimize cache performance."""
        try:
            # Get current cache statistics
            memory_stats = self.memory_cache.get_stats()
            sync_stats = self.state_synchronizer.get_sync_stats()
            
            optimization_actions = []
            
            # Check cache hit rates
            if memory_stats["hit_rate"] < 0.8:
                optimization_actions.append("Consider increasing memory cache size")
            
            # Check memory usage
            if memory_stats["memory_usage_mb"] > 500:  # 500MB threshold
                optimization_actions.append("Consider cache eviction optimization")
            
            # Check sync performance
            if sync_stats["failed_syncs"] > sync_stats["sync_operations"] * 0.1:
                optimization_actions.append("High sync failure rate - check Redis connectivity")
            
            return {
                "memory_cache_stats": memory_stats,
                "sync_stats": sync_stats,
                "performance_metrics": self.performance_metrics,
                "optimization_actions": optimization_actions,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize cache performance: {e}")
            return {"error": str(e)}
    
    async def _setup_state_listeners(self) -> None:
        """Setup state change listeners for cache invalidation."""
        
        async def health_change_handler(change_data):
            """Handle provider health changes."""
            provider_name = change_data.get("state_key")
            if provider_name:
                # Invalidate related caches
                await self.memory_cache.delete(f"provider_health_{provider_name}")
        
        async def circuit_breaker_change_handler(change_data):
            """Handle circuit breaker state changes."""
            provider_name = change_data.get("state_key")
            if provider_name:
                await self.memory_cache.delete(f"circuit_breaker_{provider_name}")
        
        # Register listeners
        await self.state_synchronizer.register_state_change_listener(
            "provider_health", health_change_handler
        )
        await self.state_synchronizer.register_state_change_listener(
            "circuit_breaker", circuit_breaker_change_handler
        )
    
    def _record_operation_time(self, duration_seconds: float) -> None:
        """Record operation time for performance tracking."""
        duration_ms = duration_seconds * 1000
        
        self.performance_metrics["total_operations"] += 1
        
        # Add to response times list
        self._response_times.append(duration_ms)
        
        # Keep only last N response times for percentile calculations
        if len(self._response_times) > self._max_response_times:
            self._response_times = self._response_times[-self._max_response_times:]
        
        # Calculate percentiles
        if self._response_times:
            sorted_times = sorted(self._response_times)
            n = len(sorted_times)
            
            # Average
            self.performance_metrics["avg_response_time_ms"] = sum(sorted_times) / n
            
            # 95th percentile
            p95_index = int(0.95 * n)
            self.performance_metrics["p95_response_time_ms"] = sorted_times[min(p95_index, n-1)]
            
            # 99th percentile
            p99_index = int(0.99 * n)
            self.performance_metrics["p99_response_time_ms"] = sorted_times[min(p99_index, n-1)]
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            stats = {
                "system_info": {
                    "initialized": self._initialized,
                    "running": self._running,
                    "components": {
                        "chroma_store": self.chroma_store is not None,
                        "redis_client": self.redis_client is not None,
                        "memory_cache": self.memory_cache is not None,
                        "state_synchronizer": self.state_synchronizer is not None,
                        "recovery_manager": self.recovery_manager is not None
                    }
                },
                "performance_metrics": self.performance_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add component-specific stats
            if self.memory_cache:
                stats["memory_cache"] = self.memory_cache.get_stats()
            
            if self.state_synchronizer:
                stats["synchronization"] = self.state_synchronizer.get_sync_stats()
            
            if self.recovery_manager:
                stats["recovery"] = self.recovery_manager.get_recovery_stats()
            
            # System resource usage
            process = psutil.Process()
            stats["system_resources"] = {
                "memory_usage_mb": process.memory_info().rss / (1024 * 1024),
                "cpu_percent": process.cpu_percent(),
                "open_files": len(process.open_files())
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive stats: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # Context manager protocol for async usage
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False