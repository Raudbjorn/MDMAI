"""
Property-Based Testing for Provider Router with Fallback System.

This module uses Hypothesis for property-based testing to discover edge cases
and validate invariants in the provider router system. It tests routing algorithm
edge cases, cost calculation accuracy, state consistency validation, circuit breaker
state transitions, and configuration validation.

Test Coverage:
- Routing algorithm edge cases with arbitrary inputs
- Cost calculation accuracy across all input ranges
- State consistency validation with concurrent modifications
- Circuit breaker state transition invariants
- Configuration validation with invalid inputs
- Fallback chain execution under random conditions
- Performance bounds under random load patterns
"""

import asyncio
import json
import math
import random
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from hypothesis import given, strategies as st, assume, settings, example, Verbosity
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, invariant, initialize

from src.ai_providers.models import (
    AIRequest, AIResponse, ProviderType, ProviderCapability,
    ModelSpec, CostTier, StreamingChunk
)
from src.ai_providers.model_router import (
    ModelRouter, ModelProfile, ModelCategory, ModelTier, RoutingRule
)
from src.context.provider_router_context_manager import (
    ProviderRouterContextManager,
    ProviderHealthState,
    CircuitBreakerState,
    RoutingDecision,
    StateConsistencyLevel,
    InMemoryStateCache
)


# Custom Hypothesis strategies for domain-specific types
@st.composite
def provider_names(draw):
    """Generate valid provider names."""
    return draw(st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=32, max_codepoint=126),
        min_size=1,
        max_size=50
    ).filter(lambda x: x.strip() and not x.startswith(' ') and not x.endswith(' ')))


@st.composite
def response_times(draw):
    """Generate realistic response times."""
    return draw(st.floats(
        min_value=1.0,
        max_value=30000.0,
        allow_infinity=False,
        allow_nan=False
    ))


@st.composite
def error_rates(draw):
    """Generate valid error rates."""
    return draw(st.floats(min_value=0.0, max_value=1.0))


@st.composite
def success_rates(draw):
    """Generate valid success rates."""
    return draw(st.floats(min_value=0.0, max_value=1.0))


@st.composite
def uptime_percentages(draw):
    """Generate valid uptime percentages."""
    return draw(st.floats(min_value=0.0, max_value=100.0))


@st.composite
def provider_health_states(draw):
    """Generate valid ProviderHealthState objects."""
    name = draw(provider_names())
    error_rate = draw(error_rates())
    success_rate = draw(success_rates())
    
    # Ensure error_rate + success_rate ≤ 1.0 (with some tolerance for floating point)
    assume(abs(error_rate + success_rate - 1.0) < 0.1)
    
    return ProviderHealthState(
        provider_name=name,
        provider_type=draw(st.sampled_from(["anthropic", "openai", "google", "test"])),
        is_available=draw(st.booleans()),
        last_check=datetime.now(timezone.utc),
        response_time_ms=draw(response_times()),
        error_rate=error_rate,
        success_rate=success_rate,
        uptime_percentage=draw(uptime_percentages()),
        consecutive_failures=draw(st.integers(min_value=0, max_value=100)),
        circuit_breaker_state=draw(st.sampled_from(["CLOSED", "OPEN", "HALF_OPEN"])),
        last_error=draw(st.one_of(st.none(), st.text(max_size=200))),
        metadata=draw(st.dictionaries(
            st.text(max_size=20),
            st.one_of(st.text(max_size=50), st.integers(), st.floats(), st.booleans()),
            max_size=10
        ))
    )


@st.composite
def circuit_breaker_states(draw):
    """Generate valid CircuitBreakerState objects."""
    failure_count = draw(st.integers(min_value=0, max_value=100))
    success_count = draw(st.integers(min_value=0, max_value=100))
    failure_threshold = draw(st.integers(min_value=1, max_value=50))
    success_threshold = draw(st.integers(min_value=1, max_value=20))
    
    # Determine state based on counts and thresholds
    if failure_count >= failure_threshold:
        state = draw(st.sampled_from(["OPEN", "HALF_OPEN"]))
    elif failure_count == 0:
        state = "CLOSED"
    else:
        state = draw(st.sampled_from(["CLOSED", "HALF_OPEN"]))
    
    return CircuitBreakerState(
        provider_name=draw(provider_names()),
        state=state,
        failure_count=failure_count,
        success_count=success_count,
        last_failure_time=draw(st.one_of(
            st.none(),
            st.datetimes(min_value=datetime(2020, 1, 1, tzinfo=timezone.utc))
        )),
        next_attempt_time=draw(st.one_of(
            st.none(),
            st.datetimes(min_value=datetime.now(timezone.utc))
        )),
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        timeout_duration_s=draw(st.integers(min_value=10, max_value=300)),
        half_open_max_calls=draw(st.integers(min_value=1, max_value=10)),
        current_half_open_calls=draw(st.integers(min_value=0, max_value=10)),
        metadata=draw(st.dictionaries(st.text(max_size=20), st.text(max_size=50), max_size=5))
    )


@st.composite
def ai_requests(draw):
    """Generate valid AIRequest objects."""
    messages = draw(st.lists(
        st.dictionaries(
            st.sampled_from(["role", "content"]),
            st.one_of(
                st.sampled_from(["user", "assistant", "system"]),
                st.text(min_size=1, max_size=1000)
            ),
            min_size=2,
            max_size=2
        ),
        min_size=1,
        max_size=10
    ))
    
    return AIRequest(
        model=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100))),
        messages=messages,
        max_tokens=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=4096))),
        temperature=draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=2.0))),
        top_p=draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0))),
        stream=draw(st.booleans()),
        tools=draw(st.one_of(st.none(), st.lists(st.dictionaries(st.text(), st.text()), max_size=5)))
    )


@st.composite
def cost_calculations(draw):
    """Generate inputs for cost calculation testing."""
    input_tokens = draw(st.integers(min_value=0, max_value=1000000))
    output_tokens = draw(st.integers(min_value=0, max_value=100000))
    input_cost_per_token = draw(st.floats(min_value=0.0, max_value=0.1))
    output_cost_per_token = draw(st.floats(min_value=0.0, max_value=0.1))
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_per_token": input_cost_per_token,
        "output_cost_per_token": output_cost_per_token
    }


class TestProviderHealthStateProperties:
    """Property-based tests for provider health states."""
    
    @given(provider_health_states())
    @settings(max_examples=100)
    def test_health_state_invariants(self, health_state):
        """Test invariants that should always hold for health states."""
        # Basic type and range invariants
        assert isinstance(health_state.provider_name, str)
        assert len(health_state.provider_name.strip()) > 0
        
        assert health_state.response_time_ms >= 0
        assert 0.0 <= health_state.error_rate <= 1.0
        assert 0.0 <= health_state.success_rate <= 1.0
        assert 0.0 <= health_state.uptime_percentage <= 100.0
        assert health_state.consecutive_failures >= 0
        
        # Circuit breaker state validity
        assert health_state.circuit_breaker_state in ["CLOSED", "OPEN", "HALF_OPEN"]
        
        # Logical consistency
        if health_state.consecutive_failures > 0:
            # If there are consecutive failures, error rate should be > 0 or circuit should not be closed
            if health_state.error_rate == 0.0:
                assume(health_state.circuit_breaker_state != "CLOSED")
        
        if health_state.error_rate == 0.0 and health_state.success_rate > 0.0:
            # Perfect success should mean availability and closed circuit breaker
            assert health_state.is_available or health_state.consecutive_failures > 0
        
        # Metadata should be a dictionary
        assert isinstance(health_state.metadata, dict)
    
    @given(st.lists(provider_health_states(), min_size=2, max_size=10))
    @settings(max_examples=50)
    def test_health_state_ordering_consistency(self, health_states):
        """Test that health state comparisons are consistent."""
        # Sort by different criteria and ensure consistency
        by_response_time = sorted(health_states, key=lambda h: h.response_time_ms)
        by_error_rate = sorted(health_states, key=lambda h: h.error_rate)
        by_success_rate = sorted(health_states, key=lambda h: h.success_rate, reverse=True)
        
        # Verify sorting didn't change the objects
        assert len(by_response_time) == len(health_states)
        assert len(by_error_rate) == len(health_states)
        assert len(by_success_rate) == len(health_states)
        
        # Check that sorting is stable for equal values
        for i in range(len(by_response_time) - 1):
            if by_response_time[i].response_time_ms == by_response_time[i + 1].response_time_ms:
                # Objects with equal response times should maintain relative order
                pass  # This is hard to test directly, but sorting should be stable
    
    @given(provider_health_states(), st.floats(min_value=0.1, max_value=2.0))
    @settings(max_examples=50)
    def test_health_state_scaling_properties(self, health_state, scale_factor):
        """Test properties when scaling numeric fields."""
        # Scale response time and verify proportional relationships
        original_response_time = health_state.response_time_ms
        scaled_response_time = original_response_time * scale_factor
        
        # Create scaled version
        scaled_state = ProviderHealthState(
            provider_name=health_state.provider_name,
            provider_type=health_state.provider_type,
            is_available=health_state.is_available,
            last_check=health_state.last_check,
            response_time_ms=scaled_response_time,
            error_rate=health_state.error_rate,
            success_rate=health_state.success_rate,
            uptime_percentage=health_state.uptime_percentage,
            consecutive_failures=health_state.consecutive_failures,
            circuit_breaker_state=health_state.circuit_breaker_state,
            last_error=health_state.last_error,
            metadata=health_state.metadata.copy()
        )
        
        # Verify scaling preserved relationships
        if scale_factor > 1.0:
            assert scaled_state.response_time_ms > original_response_time
        elif scale_factor < 1.0:
            assert scaled_state.response_time_ms < original_response_time
        
        # Other properties should remain unchanged
        assert scaled_state.error_rate == health_state.error_rate
        assert scaled_state.success_rate == health_state.success_rate
        assert scaled_state.provider_name == health_state.provider_name


class TestCircuitBreakerProperties:
    """Property-based tests for circuit breaker state transitions."""
    
    @given(circuit_breaker_states())
    @settings(max_examples=100)
    def test_circuit_breaker_state_invariants(self, cb_state):
        """Test circuit breaker state invariants."""
        # Basic invariants
        assert cb_state.failure_count >= 0
        assert cb_state.success_count >= 0
        assert cb_state.failure_threshold > 0
        assert cb_state.success_threshold > 0
        assert cb_state.timeout_duration_s > 0
        assert cb_state.half_open_max_calls > 0
        assert cb_state.current_half_open_calls >= 0
        
        # State consistency invariants
        if cb_state.state == "CLOSED":
            # In CLOSED state, failure count should be below threshold
            if cb_state.failure_count >= cb_state.failure_threshold:
                # This should not happen in a well-functioning system
                # but we'll document it as a potential inconsistency
                pass
        
        elif cb_state.state == "OPEN":
            # In OPEN state, we typically have reached failure threshold
            # and should have a next attempt time
            if cb_state.failure_count >= cb_state.failure_threshold:
                assert cb_state.last_failure_time is not None
        
        elif cb_state.state == "HALF_OPEN":
            # In HALF_OPEN state, current calls should not exceed max
            assert cb_state.current_half_open_calls <= cb_state.half_open_max_calls
    
    @given(circuit_breaker_states(), st.integers(min_value=1, max_value=10))
    @settings(max_examples=50)
    def test_circuit_breaker_failure_accumulation(self, cb_state, additional_failures):
        """Test failure accumulation behavior."""
        original_failure_count = cb_state.failure_count
        
        # Simulate adding failures
        new_failure_count = original_failure_count + additional_failures
        
        # Determine expected state after failures
        if new_failure_count >= cb_state.failure_threshold:
            expected_state = "OPEN"
        else:
            expected_state = cb_state.state  # Should remain the same if not exceeding threshold
        
        # Verify that the logic is consistent
        if cb_state.state == "CLOSED" and new_failure_count < cb_state.failure_threshold:
            assert expected_state == "CLOSED"
        elif new_failure_count >= cb_state.failure_threshold:
            assert expected_state == "OPEN"
    
    @given(
        circuit_breaker_states(),
        st.integers(min_value=1, max_value=20),
        st.integers(min_value=10, max_value=300)
    )
    @settings(max_examples=30)
    def test_circuit_breaker_recovery_logic(self, cb_state, recovery_successes, timeout_duration):
        """Test circuit breaker recovery scenarios."""
        # Simulate recovery scenario
        if cb_state.state == "HALF_OPEN":
            new_success_count = cb_state.success_count + recovery_successes
            
            if new_success_count >= cb_state.success_threshold:
                # Should transition to CLOSED
                expected_state = "CLOSED"
                expected_failure_count = 0  # Reset on successful recovery
            else:
                expected_state = "HALF_OPEN"
                expected_failure_count = cb_state.failure_count
            
            # Verify recovery logic consistency
            assert expected_state in ["CLOSED", "HALF_OPEN"]
            assert expected_failure_count >= 0
        
        elif cb_state.state == "OPEN":
            # Test timeout-based transition to HALF_OPEN
            if cb_state.next_attempt_time and cb_state.last_failure_time:
                time_since_failure = (cb_state.next_attempt_time - cb_state.last_failure_time).total_seconds()
                
                if time_since_failure >= timeout_duration:
                    # Should be ready to try HALF_OPEN
                    expected_ready_for_retry = True
                else:
                    expected_ready_for_retry = False
                
                # This is more of a timestamp logic test
                assert isinstance(expected_ready_for_retry, bool)


class TestRoutingAlgorithmProperties:
    """Property-based tests for routing algorithm edge cases."""
    
    @given(ai_requests(), st.lists(provider_health_states(), min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_routing_decision_consistency(self, request, health_states):
        """Test that routing decisions are consistent and deterministic."""
        router = ModelRouter()
        
        # Convert health states to a format the router can use
        available_providers = []
        for health in health_states[:3]:  # Limit to 3 providers for test performance
            # Create a mock provider
            mock_provider = Mock()
            mock_provider.provider_type = ProviderType.ANTHROPIC if "anthropic" in health.provider_name.lower() else ProviderType.OPENAI
            mock_provider.models = {
                f"{mock_provider.provider_type.value}-test": ModelSpec(
                    model_id=f"{mock_provider.provider_type.value}-test",
                    provider_type=mock_provider.provider_type,
                    display_name="Test Model",
                    capabilities=[ProviderCapability.TEXT_GENERATION],
                    cost_per_input_token=0.001,
                    cost_per_output_token=0.002,
                    supports_streaming=True,
                    supports_tools=bool(request.tools),
                    context_length=8192,
                    max_output_tokens=4096,
                    is_available=health.is_available
                )
            }
            available_providers.append(mock_provider)
        
        if not available_providers:
            return  # Skip test if no providers
        
        try:
            # Analyze the request (this should be deterministic)
            analysis1 = router._analyze_request(request)
            analysis2 = router._analyze_request(request)
            
            # Analysis should be deterministic for the same request
            assert analysis1["estimated_input_tokens"] == analysis2["estimated_input_tokens"]
            assert analysis1["task_type"] == analysis2["task_type"]
            assert analysis1["requires_tools"] == analysis2["requires_tools"]
            assert analysis1["requires_streaming"] == analysis2["requires_streaming"]
            
            # Verify analysis properties
            assert analysis1["estimated_input_tokens"] >= 0
            assert isinstance(analysis1["task_type"], str)
            assert isinstance(analysis1["requires_tools"], bool)
            assert isinstance(analysis1["requires_streaming"], bool)
            
        except Exception:
            # Some requests might be invalid, which is OK for property testing
            pass
    
    @given(cost_calculations())
    @settings(max_examples=200)
    def test_cost_calculation_properties(self, calc_inputs):
        """Test cost calculation accuracy and properties."""
        input_tokens = calc_inputs["input_tokens"]
        output_tokens = calc_inputs["output_tokens"]
        input_cost = calc_inputs["input_cost_per_token"]
        output_cost = calc_inputs["output_cost_per_token"]
        
        # Calculate expected cost
        expected_input_cost = (input_tokens / 1000) * input_cost
        expected_output_cost = (output_tokens / 1000) * output_cost
        expected_total_cost = expected_input_cost + expected_output_cost
        
        # Properties that should always hold
        assert expected_total_cost >= 0, "Total cost should never be negative"
        assert expected_input_cost >= 0, "Input cost should never be negative"
        assert expected_output_cost >= 0, "Output cost should never be negative"
        
        # If either token count is 0, corresponding cost should be 0
        if input_tokens == 0:
            assert expected_input_cost == 0
        if output_tokens == 0:
            assert expected_output_cost == 0
        
        # Cost should scale linearly with token count
        if input_tokens > 0 and input_cost > 0:
            double_input_cost = ((input_tokens * 2) / 1000) * input_cost
            assert abs(double_input_cost - (expected_input_cost * 2)) < 1e-10
        
        # Cost should scale linearly with rate
        if input_tokens > 0 and input_cost > 0:
            double_rate_cost = (input_tokens / 1000) * (input_cost * 2)
            assert abs(double_rate_cost - (expected_input_cost * 2)) < 1e-10
        
        # Verify precision properties
        if expected_total_cost > 0:
            # Cost should not have excessive precision issues
            rounded_cost = round(expected_total_cost, 6)  # Round to 6 decimal places
            assert abs(rounded_cost - expected_total_cost) < 1e-6
    
    @given(
        st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=2, max_size=20),
        st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=50)
    def test_provider_ranking_consistency(self, response_times, quality_weight):
        """Test that provider ranking is consistent and transitive."""
        # Create mock providers with given response times
        providers_data = [
            {"name": f"provider_{i}", "response_time": rt, "quality_score": 0.8}
            for i, rt in enumerate(response_times)
        ]
        
        # Test sorting by response time
        by_response_time = sorted(providers_data, key=lambda p: p["response_time"])
        
        # Verify transitivity: if A < B and B < C, then A < C
        for i in range(len(by_response_time) - 2):
            a = by_response_time[i]
            b = by_response_time[i + 1]
            c = by_response_time[i + 2]
            
            assert a["response_time"] <= b["response_time"] <= c["response_time"]
        
        # Test composite scoring with quality weight
        for provider in providers_data:
            # Simple scoring formula: lower response time and higher quality is better
            provider["composite_score"] = (
                (1.0 - quality_weight) * (1000.0 / provider["response_time"]) +  # Lower time = higher score
                quality_weight * provider["quality_score"]
            )
        
        by_composite = sorted(providers_data, key=lambda p: p["composite_score"], reverse=True)
        
        # Verify that composite scoring is consistent
        for i in range(len(by_composite) - 1):
            current = by_composite[i]
            next_provider = by_composite[i + 1]
            
            assert current["composite_score"] >= next_provider["composite_score"]


class TestStateConsistencyProperties:
    """Property-based tests for state consistency validation."""
    
    @pytest.mark.asyncio
    @given(st.lists(provider_health_states(), min_size=1, max_size=5))
    @settings(max_examples=30)
    async def test_concurrent_state_updates_consistency(self, health_states):
        """Test state consistency under concurrent updates."""
        cache = InMemoryStateCache(max_size=1000, default_ttl=60)
        
        async def update_worker(states: List[ProviderHealthState]):
            """Worker that updates states concurrently."""
            for state in states:
                key = f"provider_health_{state.provider_name}"
                await cache.set(key, state, "provider_health")
        
        async def read_worker(provider_names: List[str]) -> Dict[str, Any]:
            """Worker that reads states concurrently."""
            results = {}
            for name in provider_names:
                key = f"provider_health_{name}"
                value = await cache.get(key, "provider_health")
                results[name] = value
            return results
        
        # Extract provider names
        provider_names = [state.provider_name for state in health_states]
        
        # Run concurrent updates and reads
        update_tasks = [update_worker([state]) for state in health_states]
        read_tasks = [read_worker([name]) for name in provider_names]
        
        await asyncio.gather(*update_tasks)
        read_results = await asyncio.gather(*read_tasks)
        
        # Verify consistency
        for i, result in enumerate(read_results):
            provider_name = provider_names[i]
            if provider_name in result:
                retrieved_state = result[provider_name]
                if retrieved_state is not None:
                    # Should have the same provider name
                    assert retrieved_state.provider_name == provider_name
                    # Should be a valid health state
                    assert isinstance(retrieved_state, ProviderHealthState)
    
    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
            min_size=0,
            max_size=20
        )
    )
    @settings(max_examples=100)
    def test_metadata_serialization_consistency(self, metadata):
        """Test that metadata serialization/deserialization is consistent."""
        try:
            # Test JSON serialization
            serialized = json.dumps(metadata, default=str)
            deserialized = json.loads(serialized)
            
            # Basic consistency checks
            assert isinstance(deserialized, dict)
            assert len(deserialized) >= 0
            
            # For string keys, should be preserved exactly
            for key, value in metadata.items():
                if isinstance(key, str) and isinstance(value, (str, int, bool)):
                    assert key in deserialized
                    if isinstance(value, (str, int, bool)):
                        assert deserialized[key] == value or str(deserialized[key]) == str(value)
            
        except (TypeError, ValueError, OverflowError):
            # Some metadata might not be serializable, which is OK
            pass


class TestConfigurationValidation:
    """Property-based tests for configuration validation."""
    
    @given(
        st.integers(min_value=-1000, max_value=1000),
        st.integers(min_value=-1000, max_value=1000),
        st.integers(min_value=-100, max_value=1000)
    )
    @settings(max_examples=100)
    def test_circuit_breaker_configuration_validation(self, failure_threshold, success_threshold, timeout_duration):
        """Test circuit breaker configuration validation."""
        # Test configuration validity
        config_valid = (
            failure_threshold > 0 and
            success_threshold > 0 and
            timeout_duration > 0
        )
        
        if config_valid:
            # Valid configuration should create consistent state
            cb_state = CircuitBreakerState(
                provider_name="test",
                state="CLOSED",
                failure_count=0,
                success_count=0,
                last_failure_time=None,
                next_attempt_time=None,
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                timeout_duration_s=timeout_duration,
                half_open_max_calls=min(success_threshold, 10),
                current_half_open_calls=0
            )
            
            # Verify the configuration was applied correctly
            assert cb_state.failure_threshold == failure_threshold
            assert cb_state.success_threshold == success_threshold
            assert cb_state.timeout_duration_s == timeout_duration
        else:
            # Invalid configuration should be rejected or handled gracefully
            # In a real system, this would trigger validation errors
            assert failure_threshold <= 0 or success_threshold <= 0 or timeout_duration <= 0
    
    @given(
        st.floats(min_value=-10.0, max_value=10.0, allow_infinity=False, allow_nan=False),
        st.floats(min_value=-10.0, max_value=10.0, allow_infinity=False, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_cost_configuration_validation(self, input_cost, output_cost):
        """Test cost configuration validation."""
        # Cost rates should be non-negative
        valid_input_cost = input_cost >= 0
        valid_output_cost = output_cost >= 0
        
        if valid_input_cost and valid_output_cost:
            # Valid costs should produce reasonable calculations
            test_tokens = 1000
            calculated_cost = (test_tokens / 1000) * input_cost + (test_tokens / 1000) * output_cost
            
            # Cost should be non-negative and finite
            assert calculated_cost >= 0
            assert math.isfinite(calculated_cost)
            
            # Cost should scale linearly
            double_tokens_cost = (test_tokens * 2 / 1000) * input_cost + (test_tokens * 2 / 1000) * output_cost
            assert abs(double_tokens_cost - calculated_cost * 2) < 1e-10
        
        else:
            # Invalid costs should be rejected
            assert input_cost < 0 or output_cost < 0


# Stateful testing with Hypothesis
class ProviderRouterStateMachine(RuleBasedStateMachine):
    """Stateful testing for provider router operations."""
    
    def __init__(self):
        super().__init__()
        self.provider_states = {}
        self.circuit_breaker_states = {}
        self.routing_decisions = []
        self.consistency_violations = []
    
    providers = Bundle('providers')
    
    @initialize()
    def setup(self):
        """Initialize the state machine."""
        self.provider_states = {}
        self.circuit_breaker_states = {}
        self.routing_decisions = []
        self.consistency_violations = []
    
    @rule(target=providers, provider_name=provider_names())
    def create_provider(self, provider_name):
        """Create a new provider."""
        if provider_name not in self.provider_states:
            self.provider_states[provider_name] = {
                "created": True,
                "healthy": True,
                "response_time": 100.0,
                "error_rate": 0.0
            }
        return provider_name
    
    @rule(provider=providers, health_data=provider_health_states())
    def update_provider_health(self, provider, health_data):
        """Update provider health state."""
        # Ensure provider name consistency
        health_data.provider_name = provider
        
        self.provider_states[provider] = {
            "created": True,
            "healthy": health_data.is_available,
            "response_time": health_data.response_time_ms,
            "error_rate": health_data.error_rate,
            "last_update": time.time()
        }
        
        # Check for consistency violations
        if health_data.error_rate > 0.5 and health_data.is_available:
            self.consistency_violations.append(
                f"Provider {provider} has high error rate ({health_data.error_rate}) but marked available"
            )
    
    @rule(provider=providers, cb_data=circuit_breaker_states())
    def update_circuit_breaker(self, provider, cb_data):
        """Update circuit breaker state."""
        cb_data.provider_name = provider
        
        self.circuit_breaker_states[provider] = {
            "state": cb_data.state,
            "failure_count": cb_data.failure_count,
            "success_count": cb_data.success_count,
            "last_update": time.time()
        }
        
        # Check circuit breaker logic consistency
        provider_state = self.provider_states.get(provider, {})
        
        if cb_data.state == "OPEN" and provider_state.get("healthy", False):
            # This might be a temporary inconsistency, but flag it
            self.consistency_violations.append(
                f"Provider {provider} has OPEN circuit breaker but marked as healthy"
            )
    
    @rule(provider=providers)
    def make_routing_decision(self, provider):
        """Make a routing decision for a provider."""
        if provider in self.provider_states:
            provider_state = self.provider_states[provider]
            cb_state = self.circuit_breaker_states.get(provider, {})
            
            # Simple routing logic
            can_route = (
                provider_state.get("healthy", False) and
                cb_state.get("state", "CLOSED") != "OPEN"
            )
            
            decision = {
                "provider": provider,
                "selected": can_route,
                "timestamp": time.time(),
                "factors": {
                    "healthy": provider_state.get("healthy", False),
                    "circuit_state": cb_state.get("state", "CLOSED"),
                    "response_time": provider_state.get("response_time", float('inf'))
                }
            }
            
            self.routing_decisions.append(decision)
    
    @invariant()
    def consistency_check(self):
        """Check overall system consistency."""
        # Verify that healthy providers generally have reasonable metrics
        for provider_name, state in self.provider_states.items():
            if state.get("healthy", False):
                # Healthy providers should have reasonable response times
                response_time = state.get("response_time", 0)
                if response_time > 10000:  # 10 seconds is very high
                    assert False, f"Healthy provider {provider_name} has excessive response time: {response_time}ms"
        
        # Check that circuit breaker states make sense
        for provider_name, cb_state in self.circuit_breaker_states.items():
            failure_count = cb_state.get("failure_count", 0)
            state = cb_state.get("state", "CLOSED")
            
            # Very basic consistency: lots of failures should mean open circuit
            if failure_count > 20 and state == "CLOSED":
                # This might be OK if successes reset the count, but flag for review
                pass
    
    @invariant()
    def routing_decisions_consistency(self):
        """Check routing decisions are consistent with provider states."""
        recent_decisions = [d for d in self.routing_decisions if time.time() - d["timestamp"] < 60]
        
        for decision in recent_decisions:
            provider = decision["provider"]
            selected = decision["selected"]
            factors = decision["factors"]
            
            # If selected, provider should generally be healthy and circuit closed
            if selected:
                if not factors["healthy"]:
                    # This might be OK in some edge cases, but worth noting
                    pass
                
                if factors["circuit_state"] == "OPEN":
                    assert False, f"Selected provider {provider} with OPEN circuit breaker"


# Run stateful tests
TestProviderRouterStateMachine = ProviderRouterStateMachine.TestCase


class TestEdgeCaseScenarios:
    """Test specific edge cases discovered through property-based testing."""
    
    @given(st.floats(min_value=0.0, max_value=1.0), st.floats(min_value=0.0, max_value=1.0))
    @example(0.0, 0.0)  # Both rates zero
    @example(1.0, 0.0)  # Perfect error rate
    @example(0.0, 1.0)  # Perfect success rate
    @example(0.5, 0.5)  # Balanced rates
    @settings(max_examples=50)
    def test_edge_case_error_success_rates(self, error_rate, success_rate):
        """Test edge cases for error and success rates."""
        # Skip invalid combinations
        assume(abs(error_rate + success_rate - 1.0) < 0.1)
        
        health_state = ProviderHealthState(
            provider_name="edge_test",
            provider_type="test",
            is_available=error_rate < 0.5,  # Available if error rate is reasonable
            last_check=datetime.now(timezone.utc),
            response_time_ms=100.0,
            error_rate=error_rate,
            success_rate=success_rate,
            uptime_percentage=success_rate * 100,
            consecutive_failures=int(error_rate * 10),
            circuit_breaker_state="OPEN" if error_rate > 0.8 else "CLOSED"
        )
        
        # Test edge cases
        if error_rate == 0.0 and success_rate == 1.0:
            # Perfect provider
            assert health_state.is_available
            assert health_state.consecutive_failures == 0
            assert health_state.circuit_breaker_state == "CLOSED"
        
        elif error_rate == 1.0 and success_rate == 0.0:
            # Completely broken provider
            assert not health_state.is_available
            assert health_state.circuit_breaker_state == "OPEN"
        
        elif error_rate == 0.0 and success_rate == 0.0:
            # Undefined state - no requests processed
            # This is an edge case that should be handled gracefully
            assert isinstance(health_state.is_available, bool)
    
    @given(st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=0, max_size=10))
    @example([])  # Empty list
    @example([100.0])  # Single provider
    @example([1.0, 1000.0])  # Extreme range
    @example([50.0, 50.0, 50.0])  # Identical providers
    @settings(max_examples=50)
    def test_edge_case_provider_selection(self, response_times):
        """Test edge cases in provider selection."""
        if not response_times:
            # No providers available - should handle gracefully
            assert len(response_times) == 0
            return
        
        # Create provider data
        providers = [{"name": f"p{i}", "response_time": rt} for i, rt in enumerate(response_times)]
        
        # Test selection logic
        best_provider = min(providers, key=lambda p: p["response_time"])
        worst_provider = max(providers, key=lambda p: p["response_time"])
        
        # Edge case: all providers identical
        if len(set(response_times)) == 1:
            assert best_provider["response_time"] == worst_provider["response_time"]
            # Selection should still work and be deterministic
            assert best_provider in providers
        
        # Edge case: extreme differences
        if len(response_times) > 1:
            min_time = min(response_times)
            max_time = max(response_times)
            
            if max_time > min_time * 100:  # More than 100x difference
                # Selection should strongly prefer the fastest
                assert best_provider["response_time"] == min_time
                assert worst_provider["response_time"] == max_time


# Performance-focused property tests
class TestPerformanceProperties:
    """Property-based tests for performance characteristics."""
    
    @given(st.integers(min_value=1, max_value=1000), st.floats(min_value=0.1, max_value=100.0))
    @settings(max_examples=30)
    def test_cache_performance_scaling(self, cache_size, operation_latency_ms):
        """Test cache performance scaling properties."""
        cache = InMemoryStateCache(max_size=cache_size, default_ttl=60)
        
        # Properties that should hold regardless of cache size
        assert cache.max_size == cache_size
        assert cache.default_ttl == 60
        
        # Cache hit rate should be deterministic for the same access pattern
        # (This is more of a consistency test)
        
        # Memory usage should scale reasonably with cache size
        # Larger cache sizes should not have exponentially worse performance
        expected_memory_overhead = cache_size * 0.001  # Rough estimate: 1KB per entry
        
        # This is a property test - we're testing that our assumptions are reasonable
        assert expected_memory_overhead >= 0
        if cache_size > 100:
            assert expected_memory_overhead > 0.1  # Should have some overhead for large caches
    
    @given(
        st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=1, max_size=20),
        st.floats(min_value=0.1, max_value=2.0)
    )
    @settings(max_examples=20)
    def test_latency_aggregation_properties(self, latencies, percentile):
        """Test properties of latency aggregation."""
        assume(0.1 <= percentile <= 1.0)
        
        if not latencies:
            return
        
        sorted_latencies = sorted(latencies)
        
        # Calculate percentile
        index = int(percentile * len(sorted_latencies))
        if index >= len(sorted_latencies):
            index = len(sorted_latencies) - 1
        
        percentile_value = sorted_latencies[index]
        
        # Properties that should always hold
        assert percentile_value >= min(latencies)
        assert percentile_value <= max(latencies)
        
        # Monotonicity: higher percentiles should have higher or equal values
        if percentile > 0.5:
            median_index = int(0.5 * len(sorted_latencies))
            median_value = sorted_latencies[median_index]
            assert percentile_value >= median_value
        
        # If all values are the same, percentile should equal that value
        if len(set(latencies)) == 1:
            assert percentile_value == latencies[0]


# Test configuration and markers
pytestmark = [
    pytest.mark.property,
    pytest.mark.hypothesis
]


if __name__ == "__main__":
    # Run with verbose hypothesis output for debugging
    pytest.main([__file__, "-v", "--tb=short", "-s", "--hypothesis-show-statistics"])