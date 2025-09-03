"""
Failure Scenario Tests for Provider Router with Fallback System.

This module provides comprehensive testing for failure scenarios including:
- Provider timeout and failure simulation
- Network connectivity issues
- Rate limit handling and circuit breaker triggers
- Database connection failures
- Memory pressure and resource constraints
- Concurrent request handling under stress
- Graceful degradation and recovery scenarios

Test Coverage:
- Circuit breaker pattern implementation under various failure modes
- Fallback chain execution with multiple failure points
- System resilience under resource constraints
- Recovery behavior after failures
- Error propagation and isolation
"""

import asyncio
import gc
import json
import psutil
import random
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio

from src.ai_providers.models import (
    AIRequest, AIResponse, ProviderType, ProviderCapability,
    ModelSpec, StreamingChunk
)
from src.ai_providers.abstract_provider import AbstractProvider
from src.ai_providers.error_handler import AIProviderError, RetryStrategy
from src.context.provider_router_context_manager import (
    ProviderRouterContextManager,
    ProviderHealthState,
    CircuitBreakerState,
    RoutingDecision,
    StateConsistencyLevel,
    InMemoryStateCache,
    StateSynchronizer,
    StateRecoveryManager
)


class FailureSimulationProvider(AbstractProvider):
    """Provider that simulates various failure scenarios."""
    
    def __init__(self, provider_type: ProviderType, failure_config: Dict[str, Any]):
        from src.ai_providers.models import ProviderConfig
        config = ProviderConfig(provider_type=provider_type, api_key="failure-test-key")
        super().__init__(config)
        
        self.failure_config = failure_config
        self.request_count = 0
        self.failure_count = 0
        self.is_healthy = True
        
        # Configuration options
        self.timeout_probability = failure_config.get("timeout_probability", 0.0)
        self.error_probability = failure_config.get("error_probability", 0.0)
        self.latency_spike_probability = failure_config.get("latency_spike_probability", 0.0)
        self.base_latency_ms = failure_config.get("base_latency_ms", 100.0)
        self.timeout_duration_s = failure_config.get("timeout_duration_s", 30.0)
        self.rate_limit_probability = failure_config.get("rate_limit_probability", 0.0)
        self.memory_pressure_simulation = failure_config.get("memory_pressure_simulation", False)
        
        self._models = {
            f"{provider_type.value}-failure-test": ModelSpec(
                model_id=f"{provider_type.value}-failure-test",
                provider_type=provider_type,
                display_name=f"{provider_type.value.title()} Failure Test Model",
                capabilities=[ProviderCapability.TEXT_GENERATION, ProviderCapability.STREAMING],
                cost_per_input_token=0.001,
                cost_per_output_token=0.002,
                supports_streaming=True,
                supports_tools=False,
                context_length=4096,
                max_output_tokens=2048,
                is_available=True
            )
        }
    
    async def _initialize_client(self):
        if random.random() < self.failure_config.get("init_failure_probability", 0.0):
            raise Exception(f"Initialization failed for {self.provider_type.value}")
        await asyncio.sleep(0.01)
    
    async def _cleanup_client(self):
        await asyncio.sleep(0.01)
    
    async def _load_models(self):
        if random.random() < self.failure_config.get("model_load_failure_probability", 0.0):
            raise Exception(f"Model loading failed for {self.provider_type.value}")
        await asyncio.sleep(0.01)
    
    async def _generate_response_impl(self, request: AIRequest) -> AIResponse:
        self.request_count += 1
        
        # Simulate memory pressure
        if self.memory_pressure_simulation:
            # Create temporary large objects to simulate memory pressure
            temp_data = [list(range(1000)) for _ in range(100)]
            await asyncio.sleep(0.001)
            del temp_data
        
        # Simulate timeout
        if random.random() < self.timeout_probability:
            await asyncio.sleep(self.timeout_duration_s)
        
        # Simulate rate limiting
        if random.random() < self.rate_limit_probability:
            self.failure_count += 1
            raise AIProviderError(
                f"Rate limit exceeded for {self.provider_type.value}",
                error_code="RATE_LIMIT_EXCEEDED",
                retry_after=60
            )
        
        # Simulate latency spikes
        latency = self.base_latency_ms
        if random.random() < self.latency_spike_probability:
            latency *= 5  # 5x latency spike
        
        await asyncio.sleep(latency / 1000.0)
        
        # Simulate general errors
        if random.random() < self.error_probability:
            self.failure_count += 1
            error_types = [
                ("API_ERROR", "Internal API error"),
                ("AUTHENTICATION_ERROR", "Authentication failed"),
                ("QUOTA_EXCEEDED", "Quota exceeded"),
                ("MODEL_UNAVAILABLE", "Model temporarily unavailable"),
                ("NETWORK_ERROR", "Network connection failed")
            ]
            error_code, error_message = random.choice(error_types)
            raise AIProviderError(
                f"{error_message} for {self.provider_type.value}",
                error_code=error_code
            )
        
        return AIResponse(
            request_id=request.request_id,
            provider_type=self.provider_type,
            model=request.model,
            content=f"Failure test response from {self.provider_type.value}",
            usage={"input_tokens": 50, "output_tokens": 25, "total_tokens": 75},
            cost=0.075,
            latency_ms=latency,
            metadata={"failure_test": True, "request_count": self.request_count}
        )
    
    async def _stream_response_impl(self, request: AIRequest):
        # Simulate streaming failures
        if random.random() < self.error_probability:
            self.failure_count += 1
            raise AIProviderError(f"Streaming failed for {self.provider_type.value}")
        
        chunks = ["Failure", " test", " streaming"]
        for i, chunk in enumerate(chunks):
            # Random mid-stream failures
            if i > 0 and random.random() < self.error_probability / 2:
                raise AIProviderError(f"Mid-stream failure for {self.provider_type.value}")
            
            await asyncio.sleep(self.base_latency_ms / len(chunks) / 1000.0)
            yield StreamingChunk(
                request_id=request.request_id,
                content=chunk,
                is_complete=(i == len(chunks) - 1),
                finish_reason="stop" if i == len(chunks) - 1 else None
            )
    
    def _get_supported_capabilities(self):
        return [ProviderCapability.TEXT_GENERATION, ProviderCapability.STREAMING]
    
    async def _perform_health_check(self):
        if not self.is_healthy:
            raise Exception(f"Health check failed for {self.provider_type.value}")
        
        # Simulate intermittent health check failures
        if random.random() < self.failure_config.get("health_check_failure_probability", 0.0):
            raise Exception(f"Intermittent health check failure for {self.provider_type.value}")


class NetworkFailureSimulator:
    """Simulates various network failure scenarios."""
    
    def __init__(self):
        self.network_down = False
        self.latency_multiplier = 1.0
        self.packet_loss_rate = 0.0
        self.connection_refused_probability = 0.0
    
    def simulate_network_down(self, duration_s: float):
        """Simulate complete network outage."""
        self.network_down = True
        asyncio.create_task(self._restore_network(duration_s))
    
    async def _restore_network(self, duration_s: float):
        """Restore network after specified duration."""
        await asyncio.sleep(duration_s)
        self.network_down = False
    
    def simulate_high_latency(self, multiplier: float):
        """Simulate high network latency."""
        self.latency_multiplier = multiplier
    
    def simulate_packet_loss(self, loss_rate: float):
        """Simulate packet loss."""
        self.packet_loss_rate = loss_rate
    
    def should_fail_connection(self) -> bool:
        """Check if connection should fail."""
        if self.network_down:
            return True
        
        if random.random() < self.connection_refused_probability:
            return True
        
        if random.random() < self.packet_loss_rate:
            return True
        
        return False
    
    def get_network_delay(self, base_delay: float) -> float:
        """Get network delay with simulated conditions."""
        return base_delay * self.latency_multiplier


@pytest.fixture
def network_simulator():
    """Network failure simulator fixture."""
    return NetworkFailureSimulator()


@pytest.fixture
def failure_providers():
    """Create providers with different failure characteristics."""
    return {
        "timeout_prone": FailureSimulationProvider(
            ProviderType.ANTHROPIC,
            {
                "timeout_probability": 0.3,
                "timeout_duration_s": 5.0,
                "base_latency_ms": 50.0
            }
        ),
        "error_prone": FailureSimulationProvider(
            ProviderType.OPENAI,
            {
                "error_probability": 0.4,
                "base_latency_ms": 75.0
            }
        ),
        "rate_limited": FailureSimulationProvider(
            ProviderType.GOOGLE,
            {
                "rate_limit_probability": 0.2,
                "base_latency_ms": 100.0
            }
        ),
        "unstable": FailureSimulationProvider(
            ProviderType.ANTHROPIC,
            {
                "error_probability": 0.2,
                "timeout_probability": 0.1,
                "latency_spike_probability": 0.3,
                "base_latency_ms": 80.0
            }
        ),
        "memory_pressure": FailureSimulationProvider(
            ProviderType.OPENAI,
            {
                "memory_pressure_simulation": True,
                "base_latency_ms": 120.0
            }
        )
    }


@pytest.fixture
async def failure_context_manager():
    """Context manager configured for failure testing."""
    # Mock external dependencies with failure simulation
    mock_redis = AsyncMock()
    mock_chroma = Mock()
    
    # Configure Redis to occasionally fail
    async def redis_get_with_failures(key):
        if random.random() < 0.1:  # 10% failure rate
            raise Exception("Redis connection failed")
        return None
    
    async def redis_set_with_failures(key, value, *args, **kwargs):
        if random.random() < 0.05:  # 5% failure rate
            raise Exception("Redis write failed")
    
    mock_redis.get = redis_get_with_failures
    mock_redis.set = redis_set_with_failures
    mock_redis.setex = redis_set_with_failures
    mock_redis.mget = AsyncMock(return_value=[])
    mock_redis.keys = AsyncMock(return_value=[])
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.publish = AsyncMock()
    mock_redis.pipeline = AsyncMock()
    
    with patch('redis.asyncio.from_url', return_value=mock_redis), \
         patch('chromadb.HttpClient', return_value=mock_chroma):
        
        cm = ProviderRouterContextManager(
            chroma_host="localhost",
            chroma_port=8000,
            redis_url="redis://localhost:6379/0",
            enable_recovery=False,  # Disable recovery for failure testing
            cache_size=100  # Smaller cache for failure testing
        )
        
        await cm.initialize()
        await cm.start()
        
        yield cm
        
        await cm.stop()


class TestCircuitBreakerFailures:
    """Test circuit breaker behavior under various failure scenarios."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_basic_flow(self, failure_context_manager, failure_providers):
        """Test basic circuit breaker state transitions."""
        cm = failure_context_manager
        provider = failure_providers["error_prone"]
        
        # Initialize circuit breaker in CLOSED state
        await cm.update_circuit_breaker_state(
            provider.provider_type.value,
            {
                "state": "CLOSED",
                "failure_count": 0,
                "success_count": 0,
                "last_failure_time": None,
                "next_attempt_time": None,
                "failure_threshold": 3,
                "success_threshold": 2,
                "timeout_duration_s": 30,
                "half_open_max_calls": 2,
                "current_half_open_calls": 0
            }
        )
        
        # Generate requests to trigger failures
        requests = [
            AIRequest(
                model=f"{provider.provider_type.value}-failure-test",
                messages=[{"role": "user", "content": f"Test request {i}"}],
                max_tokens=100
            )
            for i in range(10)
        ]
        
        failure_count = 0
        circuit_opened = False
        
        for request in requests:
            try:
                response = await provider._generate_response_impl(request)
                # Success - reset failure count
                cb_state = await cm.get_circuit_breaker_state(provider.provider_type.value)
                if cb_state:
                    await cm.update_circuit_breaker_state(
                        provider.provider_type.value,
                        {
                            **cb_state.__dict__,
                            "failure_count": max(0, cb_state.failure_count - 1),
                            "success_count": cb_state.success_count + 1
                        }
                    )
                
            except AIProviderError:
                failure_count += 1
                cb_state = await cm.get_circuit_breaker_state(provider.provider_type.value)
                
                if cb_state:
                    new_failure_count = cb_state.failure_count + 1
                    new_state = "OPEN" if new_failure_count >= cb_state.failure_threshold else "CLOSED"
                    
                    if new_state == "OPEN" and cb_state.state != "OPEN":
                        circuit_opened = True
                    
                    await cm.update_circuit_breaker_state(
                        provider.provider_type.value,
                        {
                            **cb_state.__dict__,
                            "state": new_state,
                            "failure_count": new_failure_count,
                            "last_failure_time": datetime.now(timezone.utc),
                            "next_attempt_time": datetime.now(timezone.utc) + timedelta(seconds=30) if new_state == "OPEN" else None
                        }
                    )
        
        # Verify circuit breaker behavior
        final_state = await cm.get_circuit_breaker_state(provider.provider_type.value)
        assert final_state is not None
        assert final_state.failure_count > 0, "Should have recorded failures"
        
        if circuit_opened:
            assert final_state.state in ["OPEN", "HALF_OPEN"], "Circuit should be open or half-open after failures"
            assert final_state.last_failure_time is not None, "Should record last failure time"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self, failure_context_manager, failure_providers):
        """Test circuit breaker recovery through HALF_OPEN state."""
        cm = failure_context_manager
        provider = failure_providers["unstable"]
        
        # Set circuit breaker to OPEN state (after failures)
        open_state = CircuitBreakerState(
            provider_name=provider.provider_type.value,
            state="OPEN",
            failure_count=5,
            success_count=0,
            last_failure_time=datetime.now(timezone.utc) - timedelta(seconds=35),  # Past timeout
            next_attempt_time=datetime.now(timezone.utc) - timedelta(seconds=5),  # Ready for retry
            failure_threshold=3,
            success_threshold=2,
            timeout_duration_s=30,
            half_open_max_calls=3,
            current_half_open_calls=0
        )
        
        await cm.update_circuit_breaker_state(
            provider.provider_type.value,
            open_state.__dict__
        )
        
        # Simulate recovery scenario
        recovery_request = AIRequest(
            model=f"{provider.provider_type.value}-failure-test",
            messages=[{"role": "user", "content": "Recovery test"}],
            max_tokens=100
        )
        
        # First attempt after timeout should move to HALF_OPEN
        cb_state = await cm.get_circuit_breaker_state(provider.provider_type.value)
        assert cb_state.state == "OPEN"
        
        # Transition to HALF_OPEN for test
        await cm.update_circuit_breaker_state(
            provider.provider_type.value,
            {
                **cb_state.__dict__,
                "state": "HALF_OPEN",
                "current_half_open_calls": 0
            }
        )
        
        # Attempt requests in HALF_OPEN state
        successful_recoveries = 0
        max_attempts = 5
        
        for i in range(max_attempts):
            try:
                # Temporarily reduce error probability for recovery test
                original_error_prob = provider.error_probability
                provider.error_probability = 0.1  # Lower error rate for recovery
                
                response = await provider._generate_response_impl(recovery_request)
                successful_recoveries += 1
                
                # Update circuit breaker with success
                cb_state = await cm.get_circuit_breaker_state(provider.provider_type.value)
                if cb_state:
                    new_success_count = cb_state.success_count + 1
                    new_state = "CLOSED" if new_success_count >= cb_state.success_threshold else "HALF_OPEN"
                    
                    await cm.update_circuit_breaker_state(
                        provider.provider_type.value,
                        {
                            **cb_state.__dict__,
                            "state": new_state,
                            "success_count": new_success_count,
                            "failure_count": 0 if new_state == "CLOSED" else cb_state.failure_count,
                            "current_half_open_calls": cb_state.current_half_open_calls + 1
                        }
                    )
                    
                    if new_state == "CLOSED":
                        break
                
                provider.error_probability = original_error_prob
                
            except AIProviderError:
                # Failed recovery - back to OPEN
                cb_state = await cm.get_circuit_breaker_state(provider.provider_type.value)
                if cb_state:
                    await cm.update_circuit_breaker_state(
                        provider.provider_type.value,
                        {
                            **cb_state.__dict__,
                            "state": "OPEN",
                            "failure_count": cb_state.failure_count + 1,
                            "last_failure_time": datetime.now(timezone.utc),
                            "next_attempt_time": datetime.now(timezone.utc) + timedelta(seconds=30)
                        }
                    )
                break
        
        final_state = await cm.get_circuit_breaker_state(provider.provider_type.value)
        assert final_state is not None
        
        if successful_recoveries >= final_state.success_threshold:
            assert final_state.state == "CLOSED", "Circuit should be closed after successful recovery"
        else:
            assert final_state.state in ["OPEN", "HALF_OPEN"], "Circuit should remain open/half-open after failed recovery"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_timeout_scenarios(self, failure_context_manager, failure_providers):
        """Test circuit breaker behavior with timeout scenarios."""
        cm = failure_context_manager
        provider = failure_providers["timeout_prone"]
        
        # Initialize circuit breaker
        await cm.update_circuit_breaker_state(
            provider.provider_type.value,
            {
                "state": "CLOSED",
                "failure_count": 0,
                "success_count": 0,
                "last_failure_time": None,
                "next_attempt_time": None,
                "failure_threshold": 2,  # Low threshold for timeout testing
                "success_threshold": 1,
                "timeout_duration_s": 10,
                "half_open_max_calls": 1,
                "current_half_open_calls": 0
            }
        )
        
        timeout_count = 0
        max_timeout_test_requests = 5
        
        for i in range(max_timeout_test_requests):
            request = AIRequest(
                model=f"{provider.provider_type.value}-failure-test",
                messages=[{"role": "user", "content": f"Timeout test {i}"}],
                max_tokens=100
            )
            
            try:
                # Set a shorter timeout for testing
                start_time = time.time()
                response = await asyncio.wait_for(
                    provider._generate_response_impl(request),
                    timeout=2.0  # 2 second timeout
                )
                
            except asyncio.TimeoutError:
                timeout_count += 1
                
                # Update circuit breaker for timeout
                cb_state = await cm.get_circuit_breaker_state(provider.provider_type.value)
                if cb_state:
                    new_failure_count = cb_state.failure_count + 1
                    new_state = "OPEN" if new_failure_count >= cb_state.failure_threshold else "CLOSED"
                    
                    await cm.update_circuit_breaker_state(
                        provider.provider_type.value,
                        {
                            **cb_state.__dict__,
                            "state": new_state,
                            "failure_count": new_failure_count,
                            "last_failure_time": datetime.now(timezone.utc),
                            "next_attempt_time": datetime.now(timezone.utc) + timedelta(seconds=10) if new_state == "OPEN" else None
                        }
                    )
                    
                    # If circuit is open, stop trying
                    if new_state == "OPEN":
                        break
            
            except Exception as e:
                # Other errors
                pass
        
        # Verify timeout handling
        final_state = await cm.get_circuit_breaker_state(provider.provider_type.value)
        assert final_state is not None
        
        if timeout_count >= final_state.failure_threshold:
            assert final_state.state == "OPEN", "Circuit should open after timeout failures"


class TestFallbackChainExecution:
    """Test fallback chain execution with multiple failure points."""
    
    @pytest.mark.asyncio
    async def test_sequential_fallback_execution(self, failure_context_manager, failure_providers):
        """Test sequential execution through fallback chain."""
        cm = failure_context_manager
        
        # Create fallback chain: error_prone -> rate_limited -> unstable (should work eventually)
        providers = [
            failure_providers["error_prone"],      # High error rate (40%)
            failure_providers["rate_limited"],     # Rate limit issues (20%)
            failure_providers["unstable"]          # Mix of issues but lower error rate
        ]
        
        # Initialize all providers
        for provider in providers:
            await provider.initialize()
            
            # Set circuit breakers to closed initially
            await cm.update_circuit_breaker_state(
                provider.provider_type.value,
                {
                    "state": "CLOSED",
                    "failure_count": 0,
                    "success_count": 0,
                    "last_failure_time": None,
                    "next_attempt_time": None,
                    "failure_threshold": 3,
                    "success_threshold": 2,
                    "timeout_duration_s": 30,
                    "half_open_max_calls": 2,
                    "current_half_open_calls": 0
                }
            )
        
        async def attempt_with_fallback(request: AIRequest, provider_chain: List[FailureSimulationProvider]):
            """Attempt request with fallback chain."""
            last_error = None
            
            for i, provider in enumerate(provider_chain):
                try:
                    # Check circuit breaker state
                    cb_state = await cm.get_circuit_breaker_state(provider.provider_type.value)
                    if cb_state and cb_state.state == "OPEN":
                        # Circuit is open, skip this provider
                        continue
                    
                    # Attempt request
                    response = await provider._generate_response_impl(request)
                    
                    # Success! Record routing decision
                    await cm.store_routing_decision(
                        request.request_id,
                        {
                            "selected_provider": provider.provider_type.value,
                            "alternative_providers": [p.provider_type.value for p in provider_chain[i+1:]],
                            "routing_strategy": "fallback_chain",
                            "decision_factors": {"fallback_position": i, "total_attempts": i + 1},
                            "estimated_cost": response.cost,
                            "estimated_latency_ms": response.latency_ms,
                            "confidence_score": 1.0 - (i * 0.2),  # Lower confidence for later fallbacks
                            "fallback_chain": [p.provider_type.value for p in provider_chain[i+1:]]
                        }
                    )
                    
                    return response, i  # Return response and fallback position
                    
                except Exception as e:
                    last_error = e
                    
                    # Update circuit breaker for failure
                    if cb_state:
                        new_failure_count = cb_state.failure_count + 1
                        new_state = "OPEN" if new_failure_count >= cb_state.failure_threshold else "CLOSED"
                        
                        await cm.update_circuit_breaker_state(
                            provider.provider_type.value,
                            {
                                **cb_state.__dict__,
                                "state": new_state,
                                "failure_count": new_failure_count,
                                "last_failure_time": datetime.now(timezone.utc),
                                "next_attempt_time": datetime.now(timezone.utc) + timedelta(seconds=30) if new_state == "OPEN" else None
                            }
                        )
                    
                    continue  # Try next provider in chain
            
            # All providers failed
            raise Exception(f"All providers in fallback chain failed. Last error: {last_error}")
        
        # Test fallback chain with multiple requests
        successful_requests = 0
        fallback_usage = {0: 0, 1: 0, 2: 0}  # Count usage of each position in chain
        total_requests = 20
        
        for i in range(total_requests):
            request = AIRequest(
                model="fallback-test",
                messages=[{"role": "user", "content": f"Fallback test request {i}"}],
                max_tokens=100
            )
            
            try:
                response, fallback_position = await attempt_with_fallback(request, providers)
                successful_requests += 1
                fallback_usage[fallback_position] += 1
                
            except Exception as e:
                print(f"Request {i} failed completely: {e}")
        
        # Verify fallback behavior
        success_rate = successful_requests / total_requests
        assert success_rate >= 0.5, f"Success rate {success_rate:.2%} too low with fallback chain"
        
        # Verify fallback chain was actually used
        total_fallbacks = sum(fallback_usage.values())
        assert total_fallbacks > 0, "No fallback usage recorded"
        
        # Primary provider should have been tried most, but fallbacks should be used
        assert fallback_usage[1] + fallback_usage[2] > 0, "Fallback providers were never used"
        
        print(f"Fallback usage: Primary={fallback_usage[0]}, Secondary={fallback_usage[1]}, Tertiary={fallback_usage[2]}")
        
        # Cleanup
        for provider in providers:
            await provider.cleanup()
    
    @pytest.mark.asyncio
    async def test_parallel_fallback_with_timeout(self, failure_context_manager, failure_providers):
        """Test parallel fallback execution with timeout scenarios."""
        cm = failure_context_manager
        
        providers = [
            failure_providers["timeout_prone"],    # High timeout probability
            failure_providers["unstable"],         # Mix of issues
        ]
        
        for provider in providers:
            await provider.initialize()
        
        async def parallel_attempt_with_timeout(request: AIRequest, provider_list: List[FailureSimulationProvider], timeout_s: float = 3.0):
            """Attempt request in parallel with timeout."""
            
            async def try_provider(provider: FailureSimulationProvider):
                try:
                    return await provider._generate_response_impl(request), provider
                except Exception as e:
                    return e, provider
            
            # Start all providers in parallel
            tasks = [try_provider(provider) for provider in provider_list]
            
            try:
                # Wait for first successful response or timeout
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=timeout_s,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                
                # Check results
                for task in done:
                    result, provider = await task
                    if not isinstance(result, Exception):
                        # Success
                        await cm.store_routing_decision(
                            request.request_id,
                            {
                                "selected_provider": provider.provider_type.value,
                                "alternative_providers": [p.provider_type.value for p in provider_list if p != provider],
                                "routing_strategy": "parallel_fallback",
                                "decision_factors": {"parallel_execution": True, "timeout_s": timeout_s},
                                "estimated_cost": result.cost,
                                "estimated_latency_ms": result.latency_ms,
                                "confidence_score": 0.9,
                                "fallback_chain": []
                            }
                        )
                        return result, provider
                
                # All completed tasks were failures
                raise Exception("All parallel attempts failed")
                
            except asyncio.TimeoutError:
                # All tasks timed out
                for task in tasks:
                    task.cancel()
                raise Exception(f"All providers timed out after {timeout_s}s")
        
        # Test parallel fallback
        successful_parallel = 0
        total_parallel_requests = 10
        
        for i in range(total_parallel_requests):
            request = AIRequest(
                model="parallel-fallback-test",
                messages=[{"role": "user", "content": f"Parallel test {i}"}],
                max_tokens=100
            )
            
            try:
                response, selected_provider = await parallel_attempt_with_timeout(
                    request, providers, timeout_s=2.0
                )
                successful_parallel += 1
                
            except Exception as e:
                print(f"Parallel request {i} failed: {e}")
        
        parallel_success_rate = successful_parallel / total_parallel_requests
        print(f"Parallel fallback success rate: {parallel_success_rate:.2%}")
        
        # Should have better success rate than individual providers due to parallelism
        assert parallel_success_rate >= 0.3, "Parallel fallback should improve success rate"
        
        # Cleanup
        for provider in providers:
            await provider.cleanup()


class TestResourceConstraintFailures:
    """Test system behavior under resource constraints."""
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, failure_context_manager, failure_providers):
        """Test system behavior under memory pressure."""
        cm = failure_context_manager
        provider = failure_providers["memory_pressure"]
        
        await provider.initialize()
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create memory pressure scenario
        memory_intensive_requests = []
        max_memory_mb = initial_memory + 100  # Allow 100MB increase
        
        for i in range(50):  # Many requests to create memory pressure
            request = AIRequest(
                model=f"{provider.provider_type.value}-failure-test",
                messages=[{"role": "user", "content": f"Memory pressure test {i}" * 100}],  # Larger content
                max_tokens=500
            )
            memory_intensive_requests.append(request)
        
        # Process requests and monitor memory
        successful_under_pressure = 0
        memory_violations = 0
        
        for i, request in enumerate(memory_intensive_requests):
            try:
                # Check memory before request
                current_memory = process.memory_info().rss / 1024 / 1024
                
                if current_memory > max_memory_mb:
                    memory_violations += 1
                    # Trigger garbage collection
                    gc.collect()
                    
                    # Skip request if memory is too high
                    if current_memory > max_memory_mb * 1.2:
                        continue
                
                response = await provider._generate_response_impl(request)
                successful_under_pressure += 1
                
                # Store routing decision with memory info
                await cm.store_routing_decision(
                    request.request_id,
                    {
                        "selected_provider": provider.provider_type.value,
                        "alternative_providers": [],
                        "routing_strategy": "memory_pressure_test",
                        "decision_factors": {
                            "memory_mb": current_memory,
                            "memory_violation": current_memory > max_memory_mb
                        },
                        "estimated_cost": response.cost,
                        "estimated_latency_ms": response.latency_ms,
                        "confidence_score": 0.8,
                        "fallback_chain": []
                    }
                )
                
            except Exception as e:
                print(f"Memory pressure request {i} failed: {e}")
            
            # Small delay to allow memory cleanup
            if i % 10 == 0:
                await asyncio.sleep(0.1)
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        print(f"Memory growth: {memory_growth:.1f}MB, violations: {memory_violations}")
        
        # Verify system handled memory pressure reasonably
        assert memory_growth < 200, f"Memory growth {memory_growth:.1f}MB too high"
        assert successful_under_pressure > 0, "No requests succeeded under memory pressure"
        
        success_rate = successful_under_pressure / len(memory_intensive_requests)
        assert success_rate >= 0.3, f"Success rate {success_rate:.2%} too low under memory pressure"
        
        await provider.cleanup()
    
    @pytest.mark.asyncio
    async def test_concurrent_connection_limits(self, failure_context_manager, failure_providers):
        """Test behavior when hitting concurrent connection limits."""
        cm = failure_context_manager
        
        # Create multiple provider instances to simulate connection limits
        providers = [
            failure_providers["unstable"],
            failure_providers["error_prone"]
        ]
        
        for provider in providers:
            await provider.initialize()
        
        # Simulate high concurrency
        max_concurrent = 50
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_request(provider: FailureSimulationProvider, request_id: int):
            """Make request with concurrency limit."""
            async with semaphore:
                request = AIRequest(
                    model=f"{provider.provider_type.value}-failure-test",
                    messages=[{"role": "user", "content": f"Concurrent test {request_id}"}],
                    max_tokens=100
                )
                
                try:
                    response = await provider._generate_response_impl(request)
                    return "success", provider.provider_type.value, response.latency_ms
                
                except Exception as e:
                    return "failure", provider.provider_type.value, str(e)
        
        # Launch concurrent requests
        total_requests = 100
        tasks = []
        
        for i in range(total_requests):
            provider = providers[i % len(providers)]  # Round-robin
            task = limited_request(provider, i)
            tasks.append(task)
        
        # Execute with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            results = [Exception("Timeout")] * len(tasks)
        
        # Analyze results
        successes = [r for r in results if not isinstance(r, Exception) and r[0] == "success"]
        failures = [r for r in results if isinstance(r, Exception) or r[0] == "failure"]
        
        success_rate = len(successes) / total_requests
        avg_latency = sum(r[2] for r in successes) / len(successes) if successes else float('inf')
        
        print(f"Concurrent test: {len(successes)}/{total_requests} successful, avg latency: {avg_latency:.1f}ms")
        
        # Verify reasonable performance under concurrency
        assert success_rate >= 0.4, f"Success rate {success_rate:.2%} too low under high concurrency"
        assert avg_latency < 5000, f"Average latency {avg_latency:.1f}ms too high under concurrency"
        
        # Cleanup
        for provider in providers:
            await provider.cleanup()


class TestGracefulDegradation:
    """Test graceful degradation and recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_sequence(self, failure_context_manager, failure_providers):
        """Test system behavior during progressive degradation."""
        cm = failure_context_manager
        
        # Start with all providers healthy
        providers = [
            failure_providers["unstable"],      # Start healthy
            failure_providers["error_prone"],   # Moderate issues  
            failure_providers["rate_limited"]   # Backup
        ]
        
        for provider in providers:
            await provider.initialize()
            # Start with low error rates
            provider.error_probability = 0.1
            provider.rate_limit_probability = 0.05
        
        # Progressive degradation scenario
        degradation_phases = [
            {"duration": 2, "error_rates": [0.1, 0.1, 0.1], "description": "Healthy"},
            {"duration": 3, "error_rates": [0.4, 0.2, 0.1], "description": "Primary degraded"},
            {"duration": 3, "error_rates": [0.8, 0.5, 0.2], "description": "Multiple degraded"},
            {"duration": 3, "error_rates": [0.9, 0.8, 0.4], "description": "Severe degradation"},
            {"duration": 2, "error_rates": [0.2, 0.1, 0.1], "description": "Recovery"}
        ]
        
        async def degradation_phase_test(phase: Dict[str, Any]):
            """Test a single degradation phase."""
            # Set error rates for this phase
            for i, provider in enumerate(providers):
                provider.error_probability = phase["error_rates"][i]
            
            # Run requests during this phase
            phase_requests = 10
            successful = 0
            
            for j in range(phase_requests):
                request = AIRequest(
                    model="degradation-test",
                    messages=[{"role": "user", "content": f"{phase['description']} test {j}"}],
                    max_tokens=100
                )
                
                # Try providers in order (fallback chain)
                request_successful = False
                for provider in providers:
                    try:
                        response = await provider._generate_response_impl(request)
                        successful += 1
                        request_successful = True
                        
                        # Record routing decision
                        await cm.store_routing_decision(
                            request.request_id,
                            {
                                "selected_provider": provider.provider_type.value,
                                "alternative_providers": [p.provider_type.value for p in providers if p != provider],
                                "routing_strategy": "degradation_test",
                                "decision_factors": {
                                    "phase": phase["description"],
                                    "provider_error_rate": provider.error_probability
                                },
                                "estimated_cost": response.cost,
                                "estimated_latency_ms": response.latency_ms,
                                "confidence_score": 1.0 - provider.error_probability,
                                "fallback_chain": [p.provider_type.value for p in providers if p != provider]
                            }
                        )
                        break
                        
                    except Exception:
                        continue
                
                if not request_successful:
                    print(f"Phase '{phase['description']}' request {j} failed on all providers")
            
            success_rate = successful / phase_requests
            return success_rate, successful
        
        # Execute degradation phases
        phase_results = []
        
        for phase in degradation_phases:
            print(f"Starting phase: {phase['description']}")
            success_rate, successful_count = await degradation_phase_test(phase)
            phase_results.append({
                "phase": phase["description"],
                "success_rate": success_rate,
                "successful_count": successful_count
            })
            print(f"Phase '{phase['description']}' completed: {success_rate:.2%} success rate")
            
            await asyncio.sleep(0.1)  # Brief pause between phases
        
        # Verify graceful degradation behavior
        healthy_success = next(r["success_rate"] for r in phase_results if r["phase"] == "Healthy")
        degraded_success = next(r["success_rate"] for r in phase_results if r["phase"] == "Severe degradation")
        recovery_success = next(r["success_rate"] for r in phase_results if r["phase"] == "Recovery")
        
        # Even under severe degradation, should have some success due to fallback
        assert degraded_success >= 0.1, f"Severe degradation success rate {degraded_success:.2%} too low"
        
        # Recovery should show improved performance
        assert recovery_success > degraded_success, "Recovery should improve success rate"
        
        # System should maintain basic functionality throughout
        overall_success = sum(r["successful_count"] for r in phase_results)
        total_requests = len(degradation_phases) * 10
        overall_rate = overall_success / total_requests
        
        assert overall_rate >= 0.3, f"Overall success rate {overall_rate:.2%} too low during degradation test"
        
        print(f"Degradation test completed: {overall_rate:.2%} overall success rate")
        
        # Cleanup
        for provider in providers:
            await provider.cleanup()


# Test markers for different failure scenario categories
pytestmark = [
    pytest.mark.stress,
    pytest.mark.slow
]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])  # Stop on first failure for debugging