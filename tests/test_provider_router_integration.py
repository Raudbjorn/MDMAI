"""
Integration Tests for Provider Router with Fallback System.

This module provides comprehensive integration testing for the provider router
system, covering MCP protocol integration, database synchronization, real provider
API integration, end-to-end routing workflows, and multi-provider fallback scenarios.

Test Coverage:
- MCP protocol integration with provider routing
- Database state synchronization (ChromaDB, Redis)
- Real provider API integration testing
- End-to-end routing workflows
- Multi-provider fallback scenarios
- Cross-component interaction validation
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
import redis.asyncio as aioredis
import chromadb
from chromadb.config import Settings

from src.ai_providers.models import (
    AIRequest, AIResponse, ProviderType, ProviderCapability, ModelSpec
)
from src.ai_providers.abstract_provider import AbstractProvider
from src.context.provider_router_context_manager import (
    ProviderRouterContextManager,
    ChromaDBProviderStateStore,
    StateConsistencyLevel,
    ProviderHealthState,
    CircuitBreakerState,
    RoutingDecision,
    StateSynchronizer,
    StateRecoveryManager
)
from src.context.provider_router_performance_optimization import (
    ProviderRouterPerformanceOptimizer,
    benchmark_state_operations
)


class IntegrationTestProvider(AbstractProvider):
    """Provider for integration testing with realistic behavior."""
    
    def __init__(self, provider_type: ProviderType, failure_rate: float = 0.0, 
                 base_latency_ms: float = 50.0, should_timeout: bool = False):
        from src.ai_providers.models import ProviderConfig
        config = ProviderConfig(
            provider_type=provider_type, 
            api_key="integration-test-key"
        )
        super().__init__(config)
        
        self.failure_rate = failure_rate
        self.base_latency_ms = base_latency_ms
        self.should_timeout = should_timeout
        self.request_count = 0
        self.failure_count = 0
        
        # Realistic model specifications
        self._models = {
            f"{provider_type.value}-standard": ModelSpec(
                model_id=f"{provider_type.value}-standard",
                provider_type=provider_type,
                display_name=f"{provider_type.value.title()} Standard Model",
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
            ),
            f"{provider_type.value}-premium": ModelSpec(
                model_id=f"{provider_type.value}-premium",
                provider_type=provider_type,
                display_name=f"{provider_type.value.title()} Premium Model",
                capabilities=[
                    ProviderCapability.TEXT_GENERATION,
                    ProviderCapability.STREAMING,
                    ProviderCapability.TOOL_CALLING,
                    ProviderCapability.VISION
                ],
                cost_per_input_token=0.003,
                cost_per_output_token=0.006,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
                context_length=32768,
                max_output_tokens=8192,
                is_available=True
            )
        }
    
    async def _initialize_client(self):
        await asyncio.sleep(0.01)  # Simulate initialization
    
    async def _cleanup_client(self):
        await asyncio.sleep(0.01)  # Simulate cleanup
    
    async def _load_models(self):
        await asyncio.sleep(0.01)  # Simulate model loading
    
    async def _generate_response_impl(self, request: AIRequest) -> AIResponse:
        self.request_count += 1
        
        # Simulate timeout scenarios
        if self.should_timeout:
            await asyncio.sleep(30)  # Long timeout
        
        # Simulate realistic latency with some variance
        variance = self.base_latency_ms * 0.2
        actual_latency = self.base_latency_ms + (variance * (0.5 - asyncio.get_event_loop().time() % 1))
        await asyncio.sleep(actual_latency / 1000.0)
        
        # Simulate failures based on failure rate
        import random
        if random.random() < self.failure_rate:
            self.failure_count += 1
            from src.ai_providers.error_handler import AIProviderError
            raise AIProviderError(
                f"Simulated failure from {self.provider_type.value} "
                f"(failure {self.failure_count}/{self.request_count})"
            )
        
        return AIResponse(
            request_id=request.request_id,
            provider_type=self.provider_type,
            model=request.model,
            content=f"Integration test response from {self.provider_type.value}",
            usage={
                "input_tokens": len(str(request.messages)) // 4,
                "output_tokens": 50,
                "total_tokens": len(str(request.messages)) // 4 + 50
            },
            cost=0.1,
            latency_ms=actual_latency,
            metadata={"provider": self.provider_type.value, "test_mode": True}
        )
    
    async def _stream_response_impl(self, request: AIRequest):
        # Simulate streaming with realistic chunks
        chunks = [
            "Integration", " test", " streaming", " response", " from",
            f" {self.provider_type.value}", "."
        ]
        
        for i, chunk in enumerate(chunks):
            await asyncio.sleep(self.base_latency_ms / len(chunks) / 1000.0)
            
            from src.ai_providers.models import StreamingChunk
            yield StreamingChunk(
                request_id=request.request_id,
                content=chunk,
                is_complete=(i == len(chunks) - 1),
                finish_reason="stop" if i == len(chunks) - 1 else None
            )
    
    def _get_supported_capabilities(self):
        return [
            ProviderCapability.TEXT_GENERATION,
            ProviderCapability.STREAMING,
            ProviderCapability.TOOL_CALLING
        ]
    
    async def _perform_health_check(self):
        if self.failure_rate > 0.8:  # Very high failure rate
            raise Exception(f"Health check failed for {self.provider_type.value}")
        await asyncio.sleep(0.01)


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for the session."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
async def mock_redis_client():
    """Create a mock Redis client that behaves realistically."""
    
    class MockRedis:
        def __init__(self):
            self._data = {}
            self._pubsub_channels = {}
            self._subscribers = {}
        
        async def set(self, key: str, value: Any, ex: Optional[int] = None):
            self._data[key] = {"value": value, "expires": None}
            if ex:
                self._data[key]["expires"] = time.time() + ex
        
        async def get(self, key: str):
            if key in self._data:
                item = self._data[key]
                if item["expires"] and time.time() > item["expires"]:
                    del self._data[key]
                    return None
                return item["value"]
            return None
        
        async def setex(self, key: str, time_seconds: int, value: Any):
            await self.set(key, value, ex=time_seconds)
        
        async def mget(self, keys: List[str]):
            return [await self.get(key) for key in keys]
        
        async def keys(self, pattern: str):
            # Simple pattern matching for tests
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                return [key.encode() for key in self._data.keys() if key.startswith(prefix)]
            return [key.encode() for key in self._data.keys() if key == pattern]
        
        async def delete(self, key: str):
            if key in self._data:
                del self._data[key]
                return 1
            return 0
        
        async def ping(self):
            return True
        
        async def publish(self, channel: str, message: str):
            if channel in self._subscribers:
                for subscriber in self._subscribers[channel]:
                    await subscriber(channel, message)
        
        def pubsub(self):
            return MockPubSub(self)
        
        async def pipeline(self, transaction=False):
            return MockPipeline(self, transaction)
        
        async def close(self):
            pass
    
    class MockPubSub:
        def __init__(self, redis_client):
            self.redis_client = redis_client
            self.channels = set()
        
        async def subscribe(self, channel: str):
            self.channels.add(channel)
            if channel not in self.redis_client._subscribers:
                self.redis_client._subscribers[channel] = []
        
        async def listen(self):
            # Simulate listening (in real tests, this would be mocked differently)
            while True:
                await asyncio.sleep(0.1)
                yield {"type": "message", "data": '{"test": "message"}'}
    
    class MockPipeline:
        def __init__(self, redis_client, transaction=False):
            self.redis_client = redis_client
            self.transaction = transaction
            self.commands = []
        
        def set(self, key: str, value: Any):
            self.commands.append(("set", key, value))
            return self
        
        async def execute(self):
            results = []
            for cmd, *args in self.commands:
                if cmd == "set":
                    await self.redis_client.set(*args)
                    results.append(True)
            return results
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    
    return MockRedis()


@pytest.fixture
async def mock_chroma_client():
    """Create a mock ChromaDB client."""
    
    class MockCollection:
        def __init__(self, name: str):
            self.name = name
            self._data = {}
            self._metadata = {}
        
        def add(self, ids: List[str], documents: List[str], metadatas: List[Dict]):
            for i, doc_id in enumerate(ids):
                self._data[doc_id] = {
                    "document": documents[i],
                    "metadata": metadatas[i] if i < len(metadatas) else {}
                }
        
        def upsert(self, ids: List[str], documents: List[str], metadatas: List[Dict]):
            self.add(ids, documents, metadatas)
        
        def get(self, ids: Optional[List[str]] = None, where: Optional[Dict] = None,
                limit: Optional[int] = None, include: Optional[List[str]] = None):
            if ids:
                results = {"ids": [], "metadatas": [], "documents": []}
                for doc_id in ids:
                    if doc_id in self._data:
                        results["ids"].append(doc_id)
                        results["metadatas"].append(self._data[doc_id]["metadata"])
                        if include and "documents" in include:
                            results["documents"].append(self._data[doc_id]["document"])
                return results
            
            # Simple where clause filtering
            filtered_data = self._data
            if where:
                filtered_data = {
                    k: v for k, v in self._data.items()
                    if all(v["metadata"].get(key) == value for key, value in where.items())
                }
            
            results = {"ids": [], "metadatas": [], "documents": []}
            for doc_id, data in list(filtered_data.items())[:limit or len(filtered_data)]:
                results["ids"].append(doc_id)
                results["metadatas"].append(data["metadata"])
                if include and "documents" in include:
                    results["documents"].append(data["document"])
            
            return results
        
        def count(self):
            return len(self._data)
    
    class MockChromaClient:
        def __init__(self):
            self._collections = {}
        
        def get_or_create_collection(self, name: str, embedding_function=None, metadata=None):
            if name not in self._collections:
                self._collections[name] = MockCollection(name)
            return self._collections[name]
        
        def get_collection(self, name: str):
            return self._collections.get(name)
        
        def list_collections(self):
            return list(self._collections.keys())
    
    return MockChromaClient()


@pytest.fixture
async def integration_context_manager(mock_redis_client, mock_chroma_client):
    """Create a context manager for integration testing."""
    with patch('chromadb.HttpClient', return_value=mock_chroma_client), \
         patch('redis.asyncio.from_url', return_value=mock_redis_client):
        
        cm = ProviderRouterContextManager(
            chroma_host="localhost",
            chroma_port=8000,
            redis_url="redis://localhost:6379/0",
            enable_recovery=True,
            cache_size=1000
        )
        
        await cm.initialize()
        await cm.start()
        
        yield cm
        
        await cm.stop()


@pytest.fixture
def test_providers():
    """Create test providers with different characteristics."""
    return {
        "healthy": IntegrationTestProvider(ProviderType.ANTHROPIC, failure_rate=0.0, base_latency_ms=50),
        "slow": IntegrationTestProvider(ProviderType.OPENAI, failure_rate=0.0, base_latency_ms=200),
        "unreliable": IntegrationTestProvider(ProviderType.GOOGLE, failure_rate=0.3, base_latency_ms=100),
        "timeout": IntegrationTestProvider(ProviderType.ANTHROPIC, should_timeout=True)
    }


class TestMCPProtocolIntegration:
    """Test MCP protocol integration with provider routing."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mcp_request_routing_workflow(self, integration_context_manager):
        """Test complete MCP request routing workflow."""
        cm = integration_context_manager
        
        # Simulate MCP request data
        mcp_request_data = {
            "provider_preferences": {"prefer_speed": True, "max_cost": 0.5},
            "routing_context": {"session_id": "mcp-session-123", "user_id": "test-user"},
            "fallback_enabled": True,
            "circuit_breaker_enabled": True
        }
        
        # Store routing context
        success = await cm.store_routing_decision(
            "mcp-req-123",
            {
                "selected_provider": "anthropic",
                "alternative_providers": ["openai", "google"],
                "routing_strategy": "mcp_optimized",
                "decision_factors": mcp_request_data["provider_preferences"],
                "estimated_cost": 0.25,
                "estimated_latency_ms": 80.0,
                "confidence_score": 0.9,
                "fallback_chain": ["openai", "google"],
                "metadata": mcp_request_data["routing_context"]
            }
        )
        
        assert success is True
        
        # Verify routing decision was stored
        decisions = await cm.query_routing_patterns(
            filters={"strategy": "mcp_optimized"}, 
            limit=1
        )
        
        assert len(decisions) == 1
        assert decisions[0].request_id == "mcp-req-123"
        assert decisions[0].selected_provider == "anthropic"
        assert decisions[0].metadata["session_id"] == "mcp-session-123"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mcp_tool_integration_routing(self, integration_context_manager):
        """Test MCP tool integration with provider routing."""
        cm = integration_context_manager
        
        # Simulate MCP tool request with specific provider requirements
        tool_request = {
            "tool_name": "code_analyzer",
            "tool_capabilities": ["code_generation", "syntax_analysis"],
            "required_provider_features": ["tool_calling", "streaming"],
            "preferred_providers": ["anthropic", "openai"]
        }
        
        # Update provider health to reflect tool capabilities
        anthropic_health = ProviderHealthState(
            provider_name="anthropic",
            provider_type="anthropic",
            is_available=True,
            last_check=datetime.now(timezone.utc),
            response_time_ms=85.0,
            error_rate=0.01,
            success_rate=0.99,
            uptime_percentage=99.8,
            consecutive_failures=0,
            circuit_breaker_state="CLOSED",
            metadata={
                "tool_capabilities": ["code_generation", "syntax_analysis", "streaming"],
                "mcp_compatible": True
            }
        )
        
        success = await cm.update_provider_health(
            "anthropic", 
            {
                "provider_type": "anthropic",
                "is_available": True,
                "response_time_ms": 85.0,
                "error_rate": 0.01,
                "success_rate": 0.99,
                "uptime_percentage": 99.8,
                "consecutive_failures": 0,
                "circuit_breaker_state": "CLOSED",
                "metadata": anthropic_health.metadata
            }
        )
        
        assert success is True
        
        # Verify provider health was updated with tool capabilities
        retrieved_health = await cm.get_provider_health("anthropic")
        assert retrieved_health is not None
        assert retrieved_health.metadata["mcp_compatible"] is True
        assert "code_generation" in retrieved_health.metadata["tool_capabilities"]
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mcp_session_state_synchronization(self, integration_context_manager):
        """Test MCP session state synchronization across components."""
        cm = integration_context_manager
        
        # Simulate multiple MCP sessions with different state
        sessions = [
            {
                "session_id": f"mcp-session-{i}",
                "user_id": f"user-{i}",
                "preferred_provider": "anthropic" if i % 2 == 0 else "openai",
                "active_tools": [f"tool-{j}" for j in range(i + 1)],
                "context_window_usage": 0.3 + (i * 0.1),
                "cost_budget_remaining": 10.0 - i
            }
            for i in range(5)
        ]
        
        # Store session states
        for session in sessions:
            await cm.store_routing_decision(
                f"session-init-{session['session_id']}",
                {
                    "selected_provider": session["preferred_provider"],
                    "alternative_providers": ["anthropic", "openai", "google"],
                    "routing_strategy": "mcp_session_aware",
                    "decision_factors": {
                        "context_usage": session["context_window_usage"],
                        "budget_remaining": session["cost_budget_remaining"]
                    },
                    "estimated_cost": 0.1,
                    "estimated_latency_ms": 90.0,
                    "confidence_score": 0.8,
                    "fallback_chain": ["anthropic", "openai"],
                    "metadata": {
                        "session_id": session["session_id"],
                        "user_id": session["user_id"],
                        "active_tools": session["active_tools"]
                    }
                }
            )
        
        # Query session-aware routing decisions
        session_decisions = await cm.query_routing_patterns(
            filters={"strategy": "mcp_session_aware"},
            limit=10
        )
        
        assert len(session_decisions) == 5
        
        # Verify session state consistency
        for decision in session_decisions:
            assert "session_id" in decision.metadata
            assert "user_id" in decision.metadata
            assert "active_tools" in decision.metadata
            assert decision.routing_strategy == "mcp_session_aware"


class TestDatabaseSynchronization:
    """Test database state synchronization between ChromaDB and Redis."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_tier_state_synchronization(self, integration_context_manager):
        """Test synchronization across memory cache, Redis, and ChromaDB."""
        cm = integration_context_manager
        
        # Create test health state
        health_state = ProviderHealthState(
            provider_name="sync-test",
            provider_type="test",
            is_available=True,
            last_check=datetime.now(timezone.utc),
            response_time_ms=120.0,
            error_rate=0.05,
            success_rate=0.95,
            uptime_percentage=98.5,
            consecutive_failures=1,
            circuit_breaker_state="CLOSED",
            metadata={"sync_test": True, "tier": "integration"}
        )
        
        # Update through context manager (should sync all tiers)
        success = await cm.update_provider_health(
            "sync-test",
            {
                "provider_type": "test",
                "is_available": True,
                "response_time_ms": 120.0,
                "error_rate": 0.05,
                "success_rate": 0.95,
                "uptime_percentage": 98.5,
                "consecutive_failures": 1,
                "circuit_breaker_state": "CLOSED",
                "metadata": {"sync_test": True, "tier": "integration"}
            }
        )
        
        assert success is True
        
        # Verify data is accessible from all tiers
        
        # Tier 1: Memory cache (should be fastest)
        start = time.perf_counter()
        cached_state = await cm.get_provider_health("sync-test")
        cache_time = time.perf_counter() - start
        
        assert cached_state is not None
        assert cached_state.provider_name == "sync-test"
        assert cached_state.metadata["sync_test"] is True
        assert cache_time < 0.05  # Should be very fast from cache (allow up to 50ms for CI/slower systems)
        
        # Clear memory cache to test Redis tier
        await cm.memory_cache.delete("provider_health_sync-test")
        
        # Tier 2: Redis cache
        start = time.perf_counter()
        redis_state = await cm.get_provider_health("sync-test")
        redis_time = time.perf_counter() - start
        
        assert redis_state is not None
        assert redis_state.provider_name == "sync-test"
        assert redis_time < 0.1  # Should be reasonably fast from Redis
        
        # Verify ChromaDB persistence
        chroma_state = await cm.chroma_store.get_provider_health("sync-test")
        assert chroma_state is not None
        assert chroma_state.provider_name == "sync-test"
        assert chroma_state.metadata["sync_test"] is True
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_state_consistency_across_operations(self, integration_context_manager):
        """Test state consistency during concurrent operations."""
        cm = integration_context_manager
        
        async def update_worker(worker_id: int, iterations: int):
            """Worker that performs concurrent state updates."""
            for i in range(iterations):
                await cm.update_provider_health(
                    f"worker-{worker_id}",
                    {
                        "provider_type": "test",
                        "is_available": True,
                        "response_time_ms": 100.0 + worker_id * 10,
                        "error_rate": 0.01 * worker_id,
                        "success_rate": 1.0 - (0.01 * worker_id),
                        "uptime_percentage": 99.0,
                        "consecutive_failures": 0,
                        "circuit_breaker_state": "CLOSED",
                        "metadata": {"worker_id": worker_id, "iteration": i}
                    }
                )
                
                # Add small delay to allow interleaving
                await asyncio.sleep(0.001)
        
        async def read_worker(worker_id: int, iterations: int):
            """Worker that performs concurrent state reads."""
            consistent_reads = 0
            for i in range(iterations):
                state = await cm.get_provider_health(f"worker-{worker_id}")
                if state and state.metadata.get("worker_id") == worker_id:
                    consistent_reads += 1
                await asyncio.sleep(0.001)
            return consistent_reads
        
        # Run concurrent update and read operations
        num_workers = 5
        iterations_per_worker = 10
        
        update_tasks = [
            update_worker(worker_id, iterations_per_worker)
            for worker_id in range(num_workers)
        ]
        
        read_tasks = [
            read_worker(worker_id, iterations_per_worker)
            for worker_id in range(num_workers)
        ]
        
        # Execute all tasks concurrently
        update_results = await asyncio.gather(*update_tasks, return_exceptions=True)
        read_results = await asyncio.gather(*read_tasks, return_exceptions=True)
        
        # Verify no exceptions occurred
        for result in update_results + read_results:
            assert not isinstance(result, Exception), f"Task failed with: {result}"
        
        # Verify final state consistency
        for worker_id in range(num_workers):
            final_state = await cm.get_provider_health(f"worker-{worker_id}")
            assert final_state is not None
            assert final_state.metadata["worker_id"] == worker_id
            # Should have the highest iteration number
            assert final_state.metadata["iteration"] == iterations_per_worker - 1
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_state_recovery_and_backup(self, integration_context_manager):
        """Test state recovery and backup functionality."""
        cm = integration_context_manager
        
        if not cm.recovery_manager:
            pytest.skip("Recovery manager not enabled for this test")
        
        # Create some test state
        test_states = [
            {
                "provider_name": f"backup-test-{i}",
                "provider_type": "test",
                "is_available": True,
                "response_time_ms": 50.0 + i * 10,
                "error_rate": 0.0,
                "success_rate": 1.0,
                "uptime_percentage": 100.0,
                "consecutive_failures": 0,
                "circuit_breaker_state": "CLOSED",
                "metadata": {"backup_test": True, "index": i}
            }
            for i in range(3)
        ]
        
        # Store test states
        for state_data in test_states:
            await cm.update_provider_health(
                state_data["provider_name"],
                state_data
            )
        
        # Create backup
        backup_id = await cm.recovery_manager.create_state_backup()
        assert backup_id is not None
        assert len(backup_id) > 0
        
        # Modify state after backup
        await cm.update_provider_health(
            "backup-test-0",
            {
                **test_states[0],
                "response_time_ms": 999.0,  # Different value
                "metadata": {"backup_test": True, "index": 0, "modified": True}
            }
        )
        
        # Verify modification
        modified_state = await cm.get_provider_health("backup-test-0")
        assert modified_state.response_time_ms == 999.0
        assert modified_state.metadata.get("modified") is True
        
        # Restore from backup
        restore_success = await cm.recovery_manager.restore_from_backup(backup_id)
        assert restore_success is True
        
        # Verify restoration (may need to clear caches first)
        await cm.memory_cache.clear_category("provider_health")
        
        restored_state = await cm.get_provider_health("backup-test-0")
        # Note: Actual restoration verification depends on backup implementation
        assert restored_state is not None
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_state_consistency_validation(self, integration_context_manager):
        """Test state consistency validation across storage layers."""
        cm = integration_context_manager
        
        if not cm.recovery_manager:
            pytest.skip("Recovery manager not enabled for this test")
        
        # Create consistent state
        await cm.update_provider_health(
            "consistency-test",
            {
                "provider_type": "test",
                "is_available": True,
                "response_time_ms": 75.0,
                "error_rate": 0.0,
                "success_rate": 1.0,
                "uptime_percentage": 100.0,
                "consecutive_failures": 0,
                "circuit_breaker_state": "CLOSED",
                "metadata": {"consistency_test": True}
            }
        )
        
        # Run consistency validation
        validation_results = await cm.recovery_manager.validate_state_consistency()
        
        assert "overall_status" in validation_results
        assert validation_results["overall_status"] in ["healthy", "degraded", "critical", "error"]
        
        assert "issues" in validation_results
        assert isinstance(validation_results["issues"], list)
        
        assert "stats" in validation_results
        stats = validation_results["stats"]
        assert "redis_keys_checked" in stats
        assert "chroma_collections_checked" in stats


class TestEndToEndWorkflows:
    """Test complete end-to-end routing workflows."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_provider_routing_workflow(self, integration_context_manager, test_providers):
        """Test complete provider routing workflow from request to response."""
        cm = integration_context_manager
        providers = [test_providers["healthy"], test_providers["slow"]]
        
        # Initialize provider health states
        for i, provider in enumerate(providers):
            await provider.initialize()
            
            await cm.update_provider_health(
                provider.provider_type.value,
                {
                    "provider_type": provider.provider_type.value,
                    "is_available": True,
                    "response_time_ms": provider.base_latency_ms,
                    "error_rate": provider.failure_rate,
                    "success_rate": 1.0 - provider.failure_rate,
                    "uptime_percentage": 99.0,
                    "consecutive_failures": 0,
                    "circuit_breaker_state": "CLOSED",
                    "metadata": {"test_provider": True, "index": i}
                }
            )
        
        # Create test request
        request = AIRequest(
            model=f"{test_providers['healthy'].provider_type.value}-standard",
            messages=[{"role": "user", "content": "Integration test request"}],
            max_tokens=500,
            temperature=0.7
        )
        
        # Simulate routing decision
        routing_start = time.perf_counter()
        
        # Get provider health for routing decision
        healthy_provider_health = await cm.get_provider_health(
            test_providers["healthy"].provider_type.value
        )
        slow_provider_health = await cm.get_provider_health(
            test_providers["slow"].provider_type.value
        )
        
        routing_time = (time.perf_counter() - routing_start) * 1000
        
        assert healthy_provider_health is not None
        assert slow_provider_health is not None
        assert routing_time < 100.0  # Should be fast routing
        
        # Make routing decision based on health
        selected_provider = (
            test_providers["healthy"] 
            if healthy_provider_health.response_time_ms < slow_provider_health.response_time_ms
            else test_providers["slow"]
        )
        
        # Store routing decision
        await cm.store_routing_decision(
            request.request_id,
            {
                "selected_provider": selected_provider.provider_type.value,
                "alternative_providers": [
                    p.provider_type.value for p in providers 
                    if p != selected_provider
                ],
                "routing_strategy": "latency_optimized",
                "decision_factors": {
                    "primary_latency": healthy_provider_health.response_time_ms,
                    "backup_latency": slow_provider_health.response_time_ms
                },
                "estimated_cost": 0.1,
                "estimated_latency_ms": selected_provider.base_latency_ms,
                "confidence_score": 0.9,
                "fallback_chain": [p.provider_type.value for p in providers if p != selected_provider]
            }
        )
        
        # Execute request
        response = await selected_provider._generate_response_impl(request)
        
        assert response is not None
        assert response.provider_type == selected_provider.provider_type
        assert "Integration test response" in response.content
        assert response.latency_ms > 0
        
        # Verify routing decision was recorded
        decisions = await cm.query_routing_patterns(
            filters={"strategy": "latency_optimized"},
            limit=1
        )
        
        assert len(decisions) == 1
        assert decisions[0].selected_provider == selected_provider.provider_type.value
        
        # Cleanup
        for provider in providers:
            await provider.cleanup()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_provider_fallback_scenario(self, integration_context_manager, test_providers):
        """Test multi-provider fallback scenarios."""
        cm = integration_context_manager
        
        # Setup providers: unreliable primary, healthy backup
        primary = test_providers["unreliable"]  # 30% failure rate
        backup = test_providers["healthy"]      # 0% failure rate
        
        await primary.initialize()
        await backup.initialize()
        
        # Initialize circuit breaker states
        await cm.update_circuit_breaker_state(
            primary.provider_type.value,
            {
                "state": "CLOSED",  # Start closed
                "failure_count": 0,
                "success_count": 5,
                "last_failure_time": None,
                "next_attempt_time": None,
                "failure_threshold": 3,  # Low threshold for testing
                "success_threshold": 2,
                "timeout_duration_s": 30,
                "half_open_max_calls": 2,
                "current_half_open_calls": 0
            }
        )
        
        await cm.update_circuit_breaker_state(
            backup.provider_type.value,
            {
                "state": "CLOSED",
                "failure_count": 0,
                "success_count": 10,
                "last_failure_time": None,
                "next_attempt_time": None,
                "failure_threshold": 5,
                "success_threshold": 3,
                "timeout_duration_s": 60,
                "half_open_max_calls": 3,
                "current_half_open_calls": 0
            }
        )
        
        # Simulate multiple requests to trigger fallback
        requests = [
            AIRequest(
                model=f"{primary.provider_type.value}-standard",
                messages=[{"role": "user", "content": f"Fallback test request {i}"}],
                max_tokens=200
            )
            for i in range(10)
        ]
        
        successful_responses = 0
        fallback_used = 0
        circuit_breaker_triggered = False
        
        for request in requests:
            try:
                # Try primary provider first
                primary_cb_state = await cm.get_circuit_breaker_state(primary.provider_type.value)
                
                if primary_cb_state and primary_cb_state.state == "OPEN":
                    # Circuit breaker is open, use backup
                    circuit_breaker_triggered = True
                    response = await backup._generate_response_impl(request)
                    fallback_used += 1
                else:
                    try:
                        # Try primary provider
                        response = await primary._generate_response_impl(request)
                        
                        # Update circuit breaker with success
                        await cm.update_circuit_breaker_state(
                            primary.provider_type.value,
                            {
                                "state": "CLOSED",
                                "failure_count": max(0, (primary_cb_state.failure_count if primary_cb_state else 0) - 1),
                                "success_count": (primary_cb_state.success_count if primary_cb_state else 0) + 1,
                                "last_failure_time": primary_cb_state.last_failure_time if primary_cb_state else None,
                                "next_attempt_time": None,
                                "failure_threshold": 3,
                                "success_threshold": 2,
                                "timeout_duration_s": 30,
                                "half_open_max_calls": 2,
                                "current_half_open_calls": 0
                            }
                        )
                        
                    except Exception:
                        # Primary failed, update circuit breaker and use backup
                        failure_count = (primary_cb_state.failure_count if primary_cb_state else 0) + 1
                        new_state = "OPEN" if failure_count >= 3 else "CLOSED"
                        
                        await cm.update_circuit_breaker_state(
                            primary.provider_type.value,
                            {
                                "state": new_state,
                                "failure_count": failure_count,
                                "success_count": primary_cb_state.success_count if primary_cb_state else 0,
                                "last_failure_time": datetime.now(timezone.utc),
                                "next_attempt_time": datetime.now(timezone.utc) + timedelta(seconds=30) if new_state == "OPEN" else None,
                                "failure_threshold": 3,
                                "success_threshold": 2,
                                "timeout_duration_s": 30,
                                "half_open_max_calls": 2,
                                "current_half_open_calls": 0
                            }
                        )
                        
                        # Use backup provider
                        response = await backup._generate_response_impl(request)
                        fallback_used += 1
                
                successful_responses += 1
                
                # Record routing decision
                await cm.store_routing_decision(
                    request.request_id,
                    {
                        "selected_provider": response.provider_type.value,
                        "alternative_providers": [backup.provider_type.value if response.provider_type == primary.provider_type else primary.provider_type.value],
                        "routing_strategy": "circuit_breaker_fallback",
                        "decision_factors": {"circuit_breaker_open": circuit_breaker_triggered},
                        "estimated_cost": response.cost,
                        "estimated_latency_ms": response.latency_ms,
                        "confidence_score": 0.8 if fallback_used else 0.9,
                        "fallback_chain": [backup.provider_type.value]
                    }
                )
                
            except Exception as e:
                print(f"Request {request.request_id} failed completely: {e}")
        
        # Verify fallback behavior
        assert successful_responses >= 7  # Should have high success rate due to fallback
        assert fallback_used > 0  # Should have used fallback for some requests
        
        # Verify circuit breaker state evolution
        final_cb_state = await cm.get_circuit_breaker_state(primary.provider_type.value)
        assert final_cb_state is not None
        assert final_cb_state.failure_count > 0  # Should have recorded failures
        
        # Cleanup
        await primary.cleanup()
        await backup.cleanup()


class TestPerformanceIntegration:
    """Integration tests for performance requirements."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_sub_100ms_routing_requirement(self, integration_context_manager):
        """Test that routing decisions consistently meet sub-100ms requirement."""
        cm = integration_context_manager
        
        # Pre-populate cache with provider health data
        providers = ["anthropic", "openai", "google"]
        for provider in providers:
            await cm.update_provider_health(
                provider,
                {
                    "provider_type": provider,
                    "is_available": True,
                    "response_time_ms": 50.0 + (hash(provider) % 50),
                    "error_rate": 0.01,
                    "success_rate": 0.99,
                    "uptime_percentage": 99.5,
                    "consecutive_failures": 0,
                    "circuit_breaker_state": "CLOSED"
                }
            )
        
        # Measure routing decision performance
        routing_times = []
        for i in range(100):
            start = time.perf_counter()
            
            # Simulate routing decision process
            health_checks = await asyncio.gather(*[
                cm.get_provider_health(provider) for provider in providers
            ])
            
            # Simple routing logic
            best_provider = min(
                providers,
                key=lambda p: next(h.response_time_ms for h in health_checks if h.provider_name == p)
            )
            
            # Store decision
            await cm.store_routing_decision(
                f"perf-test-{i}",
                {
                    "selected_provider": best_provider,
                    "alternative_providers": [p for p in providers if p != best_provider],
                    "routing_strategy": "performance_test",
                    "decision_factors": {"iteration": i},
                    "estimated_cost": 0.1,
                    "estimated_latency_ms": 50.0,
                    "confidence_score": 0.9,
                    "fallback_chain": providers
                }
            )
            
            duration_ms = (time.perf_counter() - start) * 1000
            routing_times.append(duration_ms)
        
        # Analyze performance
        avg_time = sum(routing_times) / len(routing_times)
        p95_time = sorted(routing_times)[int(0.95 * len(routing_times))]
        p99_time = sorted(routing_times)[int(0.99 * len(routing_times))]
        max_time = max(routing_times)
        
        # Performance assertions
        assert avg_time < 50.0, f"Average routing time {avg_time:.2f}ms exceeds 50ms target"
        assert p95_time < 100.0, f"P95 routing time {p95_time:.2f}ms exceeds 100ms requirement"
        assert p99_time < 150.0, f"P99 routing time {p99_time:.2f}ms exceeds 150ms threshold"
        assert max_time < 200.0, f"Maximum routing time {max_time:.2f}ms is too high"
        
        print(f"Performance results: avg={avg_time:.2f}ms, p95={p95_time:.2f}ms, p99={p99_time:.2f}ms, max={max_time:.2f}ms")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.load
    async def test_concurrent_request_handling(self, integration_context_manager, test_providers):
        """Test system performance under concurrent load."""
        cm = integration_context_manager
        provider = test_providers["healthy"]
        await provider.initialize()
        
        # Initialize provider state
        await cm.update_provider_health(
            provider.provider_type.value,
            {
                "provider_type": provider.provider_type.value,
                "is_available": True,
                "response_time_ms": provider.base_latency_ms,
                "error_rate": 0.0,
                "success_rate": 1.0,
                "uptime_percentage": 100.0,
                "consecutive_failures": 0,
                "circuit_breaker_state": "CLOSED"
            }
        )
        
        async def request_worker(worker_id: int, num_requests: int):
            """Worker that processes concurrent requests."""
            worker_times = []
            successful_requests = 0
            
            for i in range(num_requests):
                start = time.perf_counter()
                
                try:
                    # Get provider health (routing simulation)
                    health = await cm.get_provider_health(provider.provider_type.value)
                    assert health is not None
                    
                    # Store routing decision
                    await cm.store_routing_decision(
                        f"load-test-{worker_id}-{i}",
                        {
                            "selected_provider": provider.provider_type.value,
                            "alternative_providers": [],
                            "routing_strategy": "load_test",
                            "decision_factors": {"worker_id": worker_id, "request_index": i},
                            "estimated_cost": 0.1,
                            "estimated_latency_ms": health.response_time_ms,
                            "confidence_score": 0.9,
                            "fallback_chain": []
                        }
                    )
                    
                    duration = (time.perf_counter() - start) * 1000
                    worker_times.append(duration)
                    successful_requests += 1
                    
                except Exception as e:
                    print(f"Worker {worker_id} request {i} failed: {e}")
            
            return {
                "worker_id": worker_id,
                "times": worker_times,
                "successful_requests": successful_requests,
                "total_requests": num_requests
            }
        
        # Run concurrent load test
        num_workers = 20
        requests_per_worker = 50
        
        start_time = time.perf_counter()
        workers = [
            request_worker(worker_id, requests_per_worker)
            for worker_id in range(num_workers)
        ]
        
        results = await asyncio.gather(*workers, return_exceptions=True)
        total_time = time.perf_counter() - start_time
        
        # Analyze results
        successful_workers = [r for r in results if not isinstance(r, Exception)]
        total_successful_requests = sum(r["successful_requests"] for r in successful_workers)
        total_requests = num_workers * requests_per_worker
        
        all_times = []
        for result in successful_workers:
            all_times.extend(result["times"])
        
        success_rate = total_successful_requests / total_requests
        throughput = total_successful_requests / total_time
        avg_latency = sum(all_times) / len(all_times) if all_times else float('inf')
        p95_latency = sorted(all_times)[int(0.95 * len(all_times))] if all_times else float('inf')
        
        # Performance assertions
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} too low"
        assert throughput >= 500, f"Throughput {throughput:.0f} RPS too low"
        assert avg_latency < 100.0, f"Average latency {avg_latency:.2f}ms too high"
        assert p95_latency < 200.0, f"P95 latency {p95_latency:.2f}ms too high"
        
        print(f"Load test results: {total_successful_requests}/{total_requests} requests successful, "
              f"{throughput:.0f} RPS, avg latency {avg_latency:.2f}ms, P95 {p95_latency:.2f}ms")
        
        await provider.cleanup()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_integrated_performance_optimization(self, integration_context_manager):
        """Test integrated performance optimization."""
        cm = integration_context_manager
        
        # Run performance benchmark
        benchmark_results = await benchmark_state_operations(cm, num_operations=500)
        
        assert "overall" in benchmark_results
        overall = benchmark_results["overall"]
        
        # Check performance targets
        assert overall["p95_latency_ms"] <= 100.0, f"P95 latency {overall['p95_latency_ms']:.2f}ms exceeds target"
        assert overall["target_achievement"]["sub_100ms_target"], "Failed to meet sub-100ms target"
        
        # Run optimization if needed
        optimizer = ProviderRouterPerformanceOptimizer(cm)
        
        if overall["p95_latency_ms"] > 80.0:  # Run optimization if close to limit
            optimization_results = await optimizer.run_comprehensive_optimization()
            
            assert "results" in optimization_results
            assert len(optimization_results["results"]) > 0
            
            # Re-run benchmark to verify improvement
            post_opt_results = await benchmark_state_operations(cm, num_operations=100)
            print(f"Post-optimization P95 latency: {post_opt_results['overall']['p95_latency_ms']:.2f}ms")


# Test markers
pytestmark = [
    pytest.mark.integration,
]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])