"""
Robust Mocking and Test Doubles for Provider Router External Dependencies.

This module provides comprehensive mocking infrastructure for testing the provider
router system with realistic behavior patterns for AI provider APIs, network failures,
timeouts, database operations, and time-dependent scenarios.

Mock Coverage:
- AI provider APIs with realistic response patterns
- Network failures and timeout simulation  
- Database operations (Redis, ChromaDB) and failures
- Time-dependent scenarios (backoff, retry timing)
- Resource constraints and system limits
- Authentication and authorization failures
"""

import asyncio
import json
import random
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch, PropertyMock
from dataclasses import dataclass, field

import pytest
import pytest_asyncio

from src.ai_providers.models import (
    AIRequest, AIResponse, ProviderType, ProviderCapability,
    ModelSpec, CostTier, StreamingChunk
)
from src.ai_providers.abstract_provider import AbstractProvider
from src.ai_providers.error_handler import AIProviderError, RetryStrategy
from src.context.provider_router_context_manager import (
    ProviderRouterContextManager,
    ProviderHealthState,
    CircuitBreakerState,
    RoutingDecision
)


@dataclass 
class MockBehaviorConfig:
    """Configuration for mock behavior patterns."""
    failure_rate: float = 0.0
    latency_ms: float = 100.0
    timeout_probability: float = 0.0
    rate_limit_probability: float = 0.0
    authentication_failure_rate: float = 0.0
    network_error_rate: float = 0.0
    memory_pressure: bool = False
    disk_pressure: bool = False
    cpu_pressure: bool = False
    
    # Response patterns
    response_variability: float = 0.1  # Coefficient of variation for response times
    burst_failure_probability: float = 0.0  # Probability of burst failures
    degraded_performance_probability: float = 0.0
    
    # Time-dependent behavior
    enable_backoff: bool = False
    retry_delays: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0])
    circuit_breaker_behavior: bool = True


class MockAIProvider:
    """Comprehensive mock AI provider with realistic behavior patterns."""
    
    def __init__(self, provider_type: ProviderType, config: MockBehaviorConfig):
        self.provider_type = provider_type
        self.config = config
        
        # State tracking
        self.request_count = 0
        self.failure_count = 0
        self.consecutive_failures = 0
        self.last_request_time = None
        self.rate_limit_reset_time = None
        self.is_authenticated = True
        
        # Burst failure tracking
        self.in_burst_failure = False
        self.burst_failure_end_time = None
        
        # Performance degradation
        self.is_degraded = False
        self.degradation_end_time = None
        
        # Response patterns
        self.response_history = []
        
        self.models = {
            f"{provider_type.value}-mock": ModelSpec(
                model_id=f"{provider_type.value}-mock",
                provider_type=provider_type,
                display_name=f"Mock {provider_type.value.title()} Model",
                capabilities=[
                    ProviderCapability.TEXT_GENERATION,
                    ProviderCapability.STREAMING,
                    ProviderCapability.TOOL_CALLING
                ],
                cost_per_input_token=0.001,
                cost_per_output_token=0.002,
                supports_streaming=True,
                supports_tools=True,
                context_length=8192,
                max_output_tokens=4096,
                is_available=True
            )
        }
    
    async def generate_response(self, request: AIRequest) -> AIResponse:
        """Generate a mock response with realistic behavior."""
        self.request_count += 1
        self.last_request_time = time.time()
        
        # Check authentication
        if not self.is_authenticated:
            raise AIProviderError("Authentication failed", error_code="AUTH_ERROR")
        
        # Check for burst failures
        await self._check_burst_failures()
        
        # Check for performance degradation
        await self._check_performance_degradation()
        
        # Check rate limits
        await self._check_rate_limits()
        
        # Simulate network errors
        if random.random() < self.config.network_error_rate:
            raise AIProviderError("Network connection failed", error_code="NETWORK_ERROR")
        
        # Simulate authentication failures
        if random.random() < self.config.authentication_failure_rate:
            self.is_authenticated = False
            raise AIProviderError("Authentication token expired", error_code="AUTH_EXPIRED")
        
        # Simulate timeouts
        if random.random() < self.config.timeout_probability:
            # Long delay to simulate timeout
            await asyncio.sleep(30.0)
        
        # Calculate response latency
        base_latency = self.config.latency_ms
        if self.is_degraded:
            base_latency *= 3.0  # 3x slower when degraded
        
        # Add variability
        variance = base_latency * self.config.response_variability
        actual_latency = max(1.0, random.gauss(base_latency, variance))
        
        # Simulate processing time
        await asyncio.sleep(actual_latency / 1000.0)
        
        # Simulate general failures
        if random.random() < self.config.failure_rate:
            self.failure_count += 1
            self.consecutive_failures += 1
            
            # Variety of error types
            error_types = [
                ("API_ERROR", "Internal server error"),
                ("QUOTA_EXCEEDED", "API quota exceeded"),
                ("MODEL_OVERLOADED", "Model is currently overloaded"),
                ("INVALID_REQUEST", "Request format is invalid"),
                ("CONTENT_FILTER", "Content filtered by safety systems")
            ]
            error_code, error_message = random.choice(error_types)
            raise AIProviderError(error_message, error_code=error_code)
        
        # Successful response
        self.consecutive_failures = 0
        
        response = AIResponse(
            request_id=request.request_id,
            provider_type=self.provider_type,
            model=request.model or f"{self.provider_type.value}-mock",
            content=f"Mock response from {self.provider_type.value}",
            usage={
                "input_tokens": len(str(request.messages)) // 4,
                "output_tokens": 50,
                "total_tokens": len(str(request.messages)) // 4 + 50
            },
            cost=0.075,
            latency_ms=actual_latency,
            metadata={
                "mock": True,
                "request_count": self.request_count,
                "provider": self.provider_type.value
            }
        )
        
        # Track response
        self.response_history.append({
            "timestamp": time.time(),
            "latency_ms": actual_latency,
            "successful": True
        })
        
        return response
    
    async def stream_response(self, request: AIRequest):
        """Generate streaming mock response."""
        chunks = [
            "Mock", " streaming", " response", " from", f" {self.provider_type.value}"
        ]
        
        chunk_delay = self.config.latency_ms / len(chunks) / 1000.0
        
        for i, chunk_text in enumerate(chunks):
            # Check for mid-stream failures
            if i > 0 and random.random() < self.config.failure_rate / 2:
                raise AIProviderError("Streaming interrupted", error_code="STREAM_ERROR")
            
            await asyncio.sleep(chunk_delay)
            
            yield StreamingChunk(
                request_id=request.request_id,
                content=chunk_text,
                is_complete=(i == len(chunks) - 1),
                finish_reason="stop" if i == len(chunks) - 1 else None
            )
    
    async def _check_burst_failures(self):
        """Check and manage burst failure periods."""
        current_time = time.time()
        
        # End burst failure period
        if self.in_burst_failure and self.burst_failure_end_time and current_time > self.burst_failure_end_time:
            self.in_burst_failure = False
            self.burst_failure_end_time = None
        
        # Start new burst failure period
        if not self.in_burst_failure and random.random() < self.config.burst_failure_probability:
            self.in_burst_failure = True
            self.burst_failure_end_time = current_time + random.uniform(10, 60)  # 10-60 second bursts
        
        # Fail if in burst period
        if self.in_burst_failure:
            raise AIProviderError("Service experiencing burst failures", error_code="SERVICE_DEGRADED")
    
    async def _check_performance_degradation(self):
        """Check and manage performance degradation periods."""
        current_time = time.time()
        
        # End degradation period
        if self.is_degraded and self.degradation_end_time and current_time > self.degradation_end_time:
            self.is_degraded = False
            self.degradation_end_time = None
        
        # Start new degradation period
        if not self.is_degraded and random.random() < self.config.degraded_performance_probability:
            self.is_degraded = True
            self.degradation_end_time = current_time + random.uniform(30, 300)  # 30s-5min degradation
    
    async def _check_rate_limits(self):
        """Check and enforce rate limits."""
        if random.random() < self.config.rate_limit_probability:
            if not self.rate_limit_reset_time:
                self.rate_limit_reset_time = time.time() + 60  # 1 minute reset
            
            if time.time() < self.rate_limit_reset_time:
                raise AIProviderError(
                    "Rate limit exceeded", 
                    error_code="RATE_LIMIT_EXCEEDED",
                    retry_after=int(self.rate_limit_reset_time - time.time())
                )
            else:
                self.rate_limit_reset_time = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        # Health check can also fail
        if random.random() < self.config.failure_rate:
            raise Exception("Health check failed")
        
        recent_responses = [r for r in self.response_history if time.time() - r["timestamp"] < 300]
        
        return {
            "status": "healthy" if not self.is_degraded else "degraded",
            "request_count": self.request_count,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "recent_avg_latency": sum(r["latency_ms"] for r in recent_responses) / len(recent_responses) if recent_responses else 0,
            "is_degraded": self.is_degraded,
            "in_burst_failure": self.in_burst_failure,
            "rate_limited": bool(self.rate_limit_reset_time and time.time() < self.rate_limit_reset_time)
        }
    
    def reset_state(self):
        """Reset provider state for testing."""
        self.request_count = 0
        self.failure_count = 0
        self.consecutive_failures = 0
        self.is_authenticated = True
        self.in_burst_failure = False
        self.is_degraded = False
        self.rate_limit_reset_time = None
        self.response_history.clear()


class MockRedisClient:
    """Comprehensive mock Redis client with failure simulation."""
    
    def __init__(self, config: MockBehaviorConfig):
        self.config = config
        self.data = {}
        self.pubsub_channels = {}
        self.connection_failures = 0
        self.is_connected = True
        self.last_operation_time = time.time()
        
        # Performance simulation
        self.operation_latencies = []
        
    async def get(self, key: str) -> Optional[str]:
        """Mock Redis GET with failure simulation."""
        await self._simulate_operation_delay()
        await self._check_connection_health()
        
        if random.random() < self.config.failure_rate:
            self.connection_failures += 1
            raise Exception("Redis connection failed")
        
        return self.data.get(key)
    
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Mock Redis SET with failure simulation."""
        await self._simulate_operation_delay()
        await self._check_connection_health()
        
        if random.random() < self.config.failure_rate:
            self.connection_failures += 1
            raise Exception("Redis write operation failed")
        
        self.data[key] = {
            "value": value,
            "expires": time.time() + ex if ex else None
        }
        return True
    
    async def setex(self, key: str, time_seconds: int, value: str) -> bool:
        """Mock Redis SETEX."""
        return await self.set(key, value, ex=time_seconds)
    
    async def mget(self, keys: List[str]) -> List[Optional[str]]:
        """Mock Redis MGET."""
        await self._simulate_operation_delay(operation_multiplier=len(keys) * 0.1)
        await self._check_connection_health()
        
        if random.random() < self.config.failure_rate:
            raise Exception("Redis batch operation failed")
        
        results = []
        for key in keys:
            item = self.data.get(key)
            if item and (not item["expires"] or time.time() < item["expires"]):
                results.append(item["value"])
            else:
                results.append(None)
        
        return results
    
    async def keys(self, pattern: str) -> List[bytes]:
        """Mock Redis KEYS."""
        await self._simulate_operation_delay(operation_multiplier=2.0)  # Keys is slow
        await self._check_connection_health()
        
        if random.random() < self.config.failure_rate:
            raise Exception("Redis keys operation failed")
        
        # Simple pattern matching
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            matching_keys = [k for k in self.data.keys() if k.startswith(prefix)]
        else:
            matching_keys = [k for k in self.data.keys() if k == pattern]
        
        return [k.encode() for k in matching_keys]
    
    async def delete(self, key: str) -> int:
        """Mock Redis DELETE."""
        await self._simulate_operation_delay()
        await self._check_connection_health()
        
        if key in self.data:
            del self.data[key]
            return 1
        return 0
    
    async def ping(self) -> bool:
        """Mock Redis PING."""
        await self._simulate_operation_delay(operation_multiplier=0.1)  # Ping is fast
        
        if random.random() < self.config.network_error_rate:
            raise Exception("Redis ping failed - connection timeout")
        
        return True
    
    async def publish(self, channel: str, message: str) -> int:
        """Mock Redis PUBLISH."""
        await self._simulate_operation_delay()
        await self._check_connection_health()
        
        if random.random() < self.config.failure_rate:
            raise Exception("Redis publish failed")
        
        # Simulate subscribers
        subscribers = self.pubsub_channels.get(channel, [])
        
        # Simulate message delivery with some failures
        delivered = 0
        for subscriber in subscribers:
            if random.random() > 0.1:  # 90% delivery rate
                delivered += 1
        
        return delivered
    
    def pubsub(self):
        """Create mock pubsub client."""
        return MockRedisPubSub(self)
    
    async def pipeline(self, transaction: bool = False):
        """Create mock pipeline."""
        return MockRedisPipeline(self, transaction)
    
    async def close(self):
        """Mock close connection."""
        self.is_connected = False
    
    async def _simulate_operation_delay(self, operation_multiplier: float = 1.0):
        """Simulate Redis operation latency."""
        base_latency = self.config.latency_ms * operation_multiplier / 1000.0
        
        # Add variability
        variance = base_latency * 0.2
        actual_latency = max(0.001, random.gauss(base_latency, variance))
        
        self.operation_latencies.append(actual_latency * 1000)  # Store in ms
        
        # Simulate memory pressure affecting performance
        if self.config.memory_pressure:
            actual_latency *= 2.0
        
        await asyncio.sleep(actual_latency)
    
    async def _check_connection_health(self):
        """Check connection health and simulate failures."""
        if not self.is_connected:
            raise Exception("Redis connection closed")
        
        # Simulate connection drops
        if random.random() < self.config.network_error_rate:
            self.is_connected = False
            raise Exception("Redis connection dropped")
        
        self.last_operation_time = time.time()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.operation_latencies:
            return {"avg_latency_ms": 0, "operations": 0}
        
        import statistics
        return {
            "avg_latency_ms": statistics.mean(self.operation_latencies),
            "operations": len(self.operation_latencies),
            "connection_failures": self.connection_failures,
            "data_size": len(self.data)
        }


class MockRedisPubSub:
    """Mock Redis PubSub client."""
    
    def __init__(self, redis_client: MockRedisClient):
        self.redis_client = redis_client
        self.subscribed_channels = set()
    
    async def subscribe(self, channel: str):
        """Subscribe to channel."""
        self.subscribed_channels.add(channel)
        if channel not in self.redis_client.pubsub_channels:
            self.redis_client.pubsub_channels[channel] = []
    
    async def listen(self):
        """Listen for messages."""
        while True:
            await asyncio.sleep(0.1)
            
            # Simulate occasional messages
            if random.random() < 0.1:
                channel = random.choice(list(self.subscribed_channels)) if self.subscribed_channels else "test"
                yield {
                    "type": "message",
                    "channel": channel,
                    "data": json.dumps({"timestamp": time.time(), "mock": True})
                }


class MockRedisPipeline:
    """Mock Redis pipeline."""
    
    def __init__(self, redis_client: MockRedisClient, transaction: bool = False):
        self.redis_client = redis_client
        self.transaction = transaction
        self.commands = []
    
    def set(self, key: str, value: str):
        """Add SET command to pipeline."""
        self.commands.append(("set", key, value))
        return self
    
    def get(self, key: str):
        """Add GET command to pipeline."""
        self.commands.append(("get", key))
        return self
    
    async def execute(self) -> List[Any]:
        """Execute pipeline commands."""
        if random.random() < self.redis_client.config.failure_rate:
            raise Exception("Redis pipeline execution failed")
        
        results = []
        for command, *args in self.commands:
            if command == "set":
                await self.redis_client.set(*args)
                results.append(True)
            elif command == "get":
                result = await self.redis_client.get(*args)
                results.append(result)
        
        return results
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not exc_type and self.commands:
            await self.execute()


class MockChromaDBClient:
    """Mock ChromaDB client with failure simulation."""
    
    def __init__(self, config: MockBehaviorConfig):
        self.config = config
        self.collections = {}
        self.operation_count = 0
        self.failure_count = 0
    
    def get_or_create_collection(self, name: str, embedding_function=None, metadata=None):
        """Mock get or create collection."""
        if random.random() < self.config.failure_rate:
            self.failure_count += 1
            raise Exception("ChromaDB collection creation failed")
        
        if name not in self.collections:
            self.collections[name] = MockChromaCollection(name, self.config)
        
        return self.collections[name]
    
    def get_collection(self, name: str):
        """Mock get collection."""
        if random.random() < self.config.failure_rate:
            raise Exception("ChromaDB collection access failed")
        
        return self.collections.get(name)
    
    def list_collections(self):
        """Mock list collections."""
        if random.random() < self.config.failure_rate:
            raise Exception("ChromaDB list collections failed")
        
        return list(self.collections.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mock statistics."""
        total_documents = sum(len(c.data) for c in self.collections.values())
        return {
            "collections": len(self.collections),
            "total_documents": total_documents,
            "operations": self.operation_count,
            "failures": self.failure_count
        }


class MockChromaCollection:
    """Mock ChromaDB collection."""
    
    def __init__(self, name: str, config: MockBehaviorConfig):
        self.name = name
        self.config = config
        self.data = {}
        self.metadata_store = {}
    
    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]):
        """Mock add documents."""
        if random.random() < self.config.failure_rate:
            raise Exception("ChromaDB add operation failed")
        
        for i, doc_id in enumerate(ids):
            self.data[doc_id] = documents[i] if i < len(documents) else ""
            self.metadata_store[doc_id] = metadatas[i] if i < len(metadatas) else {}
    
    def upsert(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]):
        """Mock upsert documents."""
        self.add(ids, documents, metadatas)  # Same as add for mock
    
    def get(self, ids: Optional[List[str]] = None, where: Optional[Dict] = None,
            limit: Optional[int] = None, include: Optional[List[str]] = None):
        """Mock get documents."""
        if random.random() < self.config.failure_rate:
            raise Exception("ChromaDB get operation failed")
        
        # Simulate operation delay
        time.sleep(self.config.latency_ms / 1000.0)
        
        results = {"ids": [], "metadatas": []}
        
        if include and "documents" in include:
            results["documents"] = []
        
        if ids:
            # Get specific IDs
            for doc_id in ids:
                if doc_id in self.data:
                    results["ids"].append(doc_id)
                    results["metadatas"].append(self.metadata_store.get(doc_id, {}))
                    if "documents" in results:
                        results["documents"].append(self.data[doc_id])
        else:
            # Get all or filtered
            filtered_ids = self.data.keys()
            
            if where:
                # Simple where clause filtering
                filtered_ids = [
                    doc_id for doc_id in filtered_ids
                    if all(
                        self.metadata_store.get(doc_id, {}).get(k) == v
                        for k, v in where.items()
                    )
                ]
            
            # Apply limit
            if limit:
                filtered_ids = list(filtered_ids)[:limit]
            
            for doc_id in filtered_ids:
                results["ids"].append(doc_id)
                results["metadatas"].append(self.metadata_store.get(doc_id, {}))
                if "documents" in results:
                    results["documents"].append(self.data[doc_id])
        
        return results
    
    def count(self) -> int:
        """Mock count documents."""
        if random.random() < self.config.failure_rate:
            raise Exception("ChromaDB count operation failed")
        
        return len(self.data)


class MockTimeProvider:
    """Mock time provider for testing time-dependent scenarios."""
    
    def __init__(self, start_time: Optional[float] = None):
        self.current_time = start_time or time.time()
        self.time_multiplier = 1.0
        self.frozen = False
    
    def time(self) -> float:
        """Get current mock time."""
        if self.frozen:
            return self.current_time
        return self.current_time
    
    def sleep(self, duration: float) -> float:
        """Mock sleep that advances time."""
        if not self.frozen:
            self.current_time += duration * self.time_multiplier
        return self.current_time
    
    def advance_time(self, seconds: float):
        """Manually advance time."""
        self.current_time += seconds
    
    def freeze_time(self):
        """Freeze time at current value."""
        self.frozen = True
    
    def unfreeze_time(self):
        """Unfreeze time."""
        self.frozen = False
    
    def set_multiplier(self, multiplier: float):
        """Set time progression multiplier."""
        self.time_multiplier = multiplier
    
    def datetime_now(self, tz=None) -> datetime:
        """Get current datetime."""
        return datetime.fromtimestamp(self.current_time, tz=tz or timezone.utc)


class MockResourceMonitor:
    """Mock resource monitor for testing resource constraints."""
    
    def __init__(self, config: MockBehaviorConfig):
        self.config = config
        self.memory_usage_mb = 100.0
        self.cpu_usage_percent = 5.0
        self.disk_usage_percent = 50.0
        
        # Simulate resource pressure
        if config.memory_pressure:
            self.memory_usage_mb = 800.0
        if config.cpu_pressure:
            self.cpu_usage_percent = 95.0
        if config.disk_pressure:
            self.disk_usage_percent = 95.0
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB."""
        # Add some variability
        variance = self.memory_usage_mb * 0.1
        return max(50.0, random.gauss(self.memory_usage_mb, variance))
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        variance = self.cpu_usage_percent * 0.2
        return max(0.0, min(100.0, random.gauss(self.cpu_usage_percent, variance)))
    
    def get_disk_usage(self) -> float:
        """Get disk usage percentage."""
        return self.disk_usage_percent
    
    def simulate_pressure_spike(self, duration_s: float = 10.0):
        """Simulate a resource pressure spike."""
        async def spike():
            original_memory = self.memory_usage_mb
            original_cpu = self.cpu_usage_percent
            
            self.memory_usage_mb = min(1000.0, self.memory_usage_mb * 2)
            self.cpu_usage_percent = min(100.0, self.cpu_usage_percent + 40)
            
            await asyncio.sleep(duration_s)
            
            self.memory_usage_mb = original_memory
            self.cpu_usage_percent = original_cpu
        
        return asyncio.create_task(spike())


# Mock factories and fixtures

@pytest.fixture
def mock_behavior_config():
    """Default mock behavior configuration."""
    return MockBehaviorConfig()

@pytest.fixture
def stable_mock_config():
    """Stable mock configuration for baseline tests."""
    return MockBehaviorConfig(
        failure_rate=0.0,
        latency_ms=50.0,
        timeout_probability=0.0,
        rate_limit_probability=0.0
    )

@pytest.fixture
def unreliable_mock_config():
    """Unreliable mock configuration for failure testing."""
    return MockBehaviorConfig(
        failure_rate=0.2,
        latency_ms=150.0,
        timeout_probability=0.1,
        rate_limit_probability=0.05,
        burst_failure_probability=0.1,
        degraded_performance_probability=0.1
    )

@pytest.fixture
def resource_constrained_mock_config():
    """Resource constrained mock configuration."""
    return MockBehaviorConfig(
        latency_ms=200.0,
        memory_pressure=True,
        cpu_pressure=True,
        disk_pressure=True
    )

@pytest.fixture
def mock_ai_providers(mock_behavior_config):
    """Collection of mock AI providers."""
    return {
        "anthropic": MockAIProvider(ProviderType.ANTHROPIC, mock_behavior_config),
        "openai": MockAIProvider(ProviderType.OPENAI, mock_behavior_config),
        "google": MockAIProvider(ProviderType.GOOGLE, mock_behavior_config)
    }

@pytest.fixture
def mock_redis(mock_behavior_config):
    """Mock Redis client."""
    return MockRedisClient(mock_behavior_config)

@pytest.fixture
def mock_chroma(mock_behavior_config):
    """Mock ChromaDB client."""
    return MockChromaDBClient(mock_behavior_config)

@pytest.fixture
def mock_time_provider():
    """Mock time provider."""
    return MockTimeProvider()

@pytest.fixture
def mock_resource_monitor(mock_behavior_config):
    """Mock resource monitor."""
    return MockResourceMonitor(mock_behavior_config)


# Context managers for comprehensive mocking

@asynccontextmanager
async def mock_provider_router_environment(
    providers_config: Optional[Dict[str, MockBehaviorConfig]] = None,
    redis_config: Optional[MockBehaviorConfig] = None,
    chroma_config: Optional[MockBehaviorConfig] = None,
    time_frozen: bool = False
):
    """Comprehensive mock environment for provider router testing."""
    
    # Default configurations
    if providers_config is None:
        default_config = MockBehaviorConfig()
        providers_config = {
            "anthropic": default_config,
            "openai": default_config,
            "google": default_config
        }
    
    redis_config = redis_config or MockBehaviorConfig()
    chroma_config = chroma_config or MockBehaviorConfig()
    
    # Create mocks
    mock_providers = {
        name: MockAIProvider(getattr(ProviderType, name.upper()), config)
        for name, config in providers_config.items()
    }
    
    mock_redis = MockRedisClient(redis_config)
    mock_chroma = MockChromaDBClient(chroma_config)
    mock_time = MockTimeProvider()
    
    if time_frozen:
        mock_time.freeze_time()
    
    # Patch the external dependencies
    with patch('redis.asyncio.from_url', return_value=mock_redis), \
         patch('chromadb.HttpClient', return_value=mock_chroma), \
         patch('time.time', side_effect=mock_time.time), \
         patch('asyncio.sleep', side_effect=lambda d: mock_time.sleep(d)):
        
        yield {
            "providers": mock_providers,
            "redis": mock_redis,
            "chroma": mock_chroma,
            "time": mock_time
        }


# Test examples demonstrating mock usage

class TestMockingFramework:
    """Tests demonstrating the mocking framework capabilities."""
    
    @pytest.mark.asyncio
    async def test_mock_ai_provider_realistic_behavior(self):
        """Test mock AI provider with realistic behavior patterns."""
        config = MockBehaviorConfig(
            failure_rate=0.1,
            latency_ms=100.0,
            burst_failure_probability=0.05,
            degraded_performance_probability=0.05
        )
        
        provider = MockAIProvider(ProviderType.ANTHROPIC, config)
        
        # Test multiple requests to see behavior patterns
        successful_requests = 0
        total_latencies = []
        
        for i in range(50):
            request = AIRequest(
                model="anthropic-mock",
                messages=[{"role": "user", "content": f"Test request {i}"}],
                max_tokens=100
            )
            
            try:
                start_time = time.time()
                response = await provider.generate_response(request)
                end_time = time.time()
                
                successful_requests += 1
                actual_latency = (end_time - start_time) * 1000
                total_latencies.append(actual_latency)
                
                # Verify response structure
                assert response.provider_type == ProviderType.ANTHROPIC
                assert response.content.startswith("Mock response")
                assert response.request_id == request.request_id
                assert response.latency_ms > 0
                
            except AIProviderError as e:
                # Expected failures based on configuration
                assert e.error_code in [
                    "API_ERROR", "QUOTA_EXCEEDED", "MODEL_OVERLOADED",
                    "INVALID_REQUEST", "CONTENT_FILTER", "SERVICE_DEGRADED"
                ]
        
        # Verify behavior patterns
        success_rate = successful_requests / 50
        assert 0.7 <= success_rate <= 1.0, f"Success rate {success_rate:.2%} outside expected range"
        
        if total_latencies:
            import statistics
            avg_latency = statistics.mean(total_latencies)
            assert 80 <= avg_latency <= 400, f"Average latency {avg_latency:.1f}ms outside expected range"
        
        # Check health status
        health = await provider.health_check()
        assert "status" in health
        assert health["request_count"] == 50
        assert health["failure_count"] >= 0
    
    @pytest.mark.asyncio
    async def test_mock_redis_failure_patterns(self, unreliable_mock_config):
        """Test mock Redis with failure patterns."""
        redis_client = MockRedisClient(unreliable_mock_config)
        
        successful_ops = 0
        failed_ops = 0
        
        # Test various operations
        for i in range(100):
            try:
                # Mix of operations
                if i % 3 == 0:
                    await redis_client.set(f"key_{i}", f"value_{i}")
                elif i % 3 == 1:
                    await redis_client.get(f"key_{i-1}")
                else:
                    await redis_client.mget([f"key_{j}" for j in range(max(0, i-5), i)])
                
                successful_ops += 1
                
            except Exception as e:
                failed_ops += 1
                assert "Redis" in str(e) or "connection" in str(e)
        
        # Verify failure rate matches configuration
        failure_rate = failed_ops / (successful_ops + failed_ops)
        expected_rate = unreliable_mock_config.failure_rate
        
        # Allow some variance in actual failure rate
        assert abs(failure_rate - expected_rate) <= 0.15, \
            f"Failure rate {failure_rate:.2%} too far from expected {expected_rate:.2%}"
        
        # Check performance stats
        stats = redis_client.get_performance_stats()
        assert stats["operations"] > 0
        assert stats["avg_latency_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_mock_time_dependent_scenarios(self):
        """Test time-dependent mock scenarios."""
        time_provider = MockTimeProvider(start_time=1000000.0)
        
        # Test time advancement
        initial_time = time_provider.time()
        time_provider.advance_time(60.0)  # Advance 1 minute
        
        assert time_provider.time() == initial_time + 60.0
        
        # Test frozen time
        time_provider.freeze_time()
        frozen_time = time_provider.time()
        
        time_provider.sleep(30.0)  # Should not advance when frozen
        assert time_provider.time() == frozen_time
        
        # Test unfrozen time with multiplier
        time_provider.unfreeze_time()
        time_provider.set_multiplier(2.0)  # 2x speed
        
        start_time = time_provider.time()
        time_provider.sleep(10.0)  # Should advance 20 seconds (10 * 2)
        
        assert time_provider.time() == start_time + 20.0
    
    @pytest.mark.asyncio
    async def test_comprehensive_mock_environment(self):
        """Test comprehensive mock environment."""
        async with mock_provider_router_environment(
            providers_config={
                "anthropic": MockBehaviorConfig(latency_ms=50.0),
                "openai": MockBehaviorConfig(latency_ms=100.0, failure_rate=0.1)
            },
            redis_config=MockBehaviorConfig(latency_ms=10.0),
            chroma_config=MockBehaviorConfig(latency_ms=200.0)
        ) as env:
            
            # Test provider interactions
            anthropic = env["providers"]["anthropic"]
            openai = env["providers"]["openai"]
            
            request = AIRequest(
                model="test",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=100
            )
            
            # Test both providers
            anthropic_response = await anthropic.generate_response(request)
            assert anthropic_response.provider_type == ProviderType.ANTHROPIC
            
            try:
                openai_response = await openai.generate_response(request)
                assert openai_response.provider_type == ProviderType.OPENAI
            except AIProviderError:
                # OpenAI configured with 10% failure rate
                pass
            
            # Test Redis operations
            redis = env["redis"]
            await redis.set("test_key", "test_value")
            value = await redis.get("test_key")
            assert value == "test_value"
            
            # Test ChromaDB operations
            chroma = env["chroma"]
            collection = chroma.get_or_create_collection("test_collection")
            collection.add(["doc1"], ["Test document"], [{"type": "test"}])
            
            results = collection.get(ids=["doc1"], include=["documents"])
            assert len(results["ids"]) == 1
            assert results["documents"][0] == "Test document"


# Test markers for mock tests
pytestmark = [
    pytest.mark.mock,
    pytest.mark.unit
]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])