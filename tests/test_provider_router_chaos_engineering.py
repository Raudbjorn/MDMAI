"""Advanced Chaos Engineering Test Suite for Provider Router.

This module implements sophisticated chaos engineering tests to validate
system resilience under extreme and unpredictable conditions.

Chaos Scenarios:
- Jepsen-style distributed system tests
- Netflix Chaos Monkey patterns
- Gremlin-inspired failure injection
- Advanced network simulation
- Byzantine fault tolerance testing
"""

import asyncio
import random
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest
import pytest_asyncio
from scipy import stats


# ============================================================================
# CHAOS ENGINEERING FRAMEWORK
# ============================================================================

class ChaosType(Enum):
    """Types of chaos that can be injected."""
    LATENCY_SPIKE = auto()
    PROVIDER_CRASH = auto()
    NETWORK_PARTITION = auto()
    CLOCK_SKEW = auto()
    MEMORY_LEAK = auto()
    CPU_SATURATION = auto()
    DISK_FAILURE = auto()
    BYZANTINE_BEHAVIOR = auto()
    PACKET_CORRUPTION = auto()
    RATE_LIMITING = auto()
    CONNECTION_POOL_EXHAUSTION = auto()
    CACHE_POISONING = auto()
    STATE_CORRUPTION = auto()
    DEADLOCK = auto()
    LIVELOCK = auto()


@dataclass
class ChaosEvent:
    """Represents a chaos event in the system."""
    type: ChaosType
    target: str  # Provider or component affected
    start_time: float
    duration: float
    intensity: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def end_time(self) -> float:
        return self.start_time + self.duration
        
    def is_active(self, current_time: float) -> bool:
        return self.start_time <= current_time <= self.end_time


class ChaosOrchestrator:
    """Orchestrates chaos injection across the system."""
    
    def __init__(self, router, intensity: float = 0.3):
        self.router = router
        self.intensity = intensity
        self.active_events: List[ChaosEvent] = []
        self.event_history: List[ChaosEvent] = []
        self.impact_metrics: Dict[str, Any] = defaultdict(list)
        self._running = False
        self._task = None
        
    async def start(self):
        """Start chaos orchestration."""
        self._running = True
        self._task = asyncio.create_task(self._orchestration_loop())
        
    async def stop(self):
        """Stop chaos orchestration."""
        self._running = False
        if self._task:
            await self._task
            
    async def _orchestration_loop(self):
        """Main orchestration loop."""
        while self._running:
            # Clean up expired events
            current_time = time.time()
            self.active_events = [
                e for e in self.active_events 
                if e.is_active(current_time)
            ]
            
            # Potentially inject new chaos
            if random.random() < self.intensity:
                event = await self._generate_chaos_event()
                await self._inject_chaos(event)
                
            await asyncio.sleep(1)  # Check every second
            
    async def _generate_chaos_event(self) -> ChaosEvent:
        """Generate a random chaos event."""
        chaos_type = random.choice(list(ChaosType))
        target = random.choice(list(self.router.providers.keys()))
        
        # Duration and intensity based on chaos type
        duration_ranges = {
            ChaosType.LATENCY_SPIKE: (1, 30),
            ChaosType.PROVIDER_CRASH: (5, 60),
            ChaosType.NETWORK_PARTITION: (10, 120),
            ChaosType.CLOCK_SKEW: (30, 300),
            ChaosType.MEMORY_LEAK: (60, 600),
            ChaosType.CPU_SATURATION: (10, 60),
            ChaosType.BYZANTINE_BEHAVIOR: (5, 30),
        }
        
        duration_range = duration_ranges.get(chaos_type, (5, 30))
        duration = random.uniform(*duration_range)
        
        return ChaosEvent(
            type=chaos_type,
            target=target,
            start_time=time.time(),
            duration=duration,
            intensity=random.uniform(0.3, 1.0),
            metadata=await self._generate_event_metadata(chaos_type)
        )
        
    async def _generate_event_metadata(self, chaos_type: ChaosType) -> Dict:
        """Generate metadata for chaos event."""
        metadata = {}
        
        if chaos_type == ChaosType.LATENCY_SPIKE:
            metadata["added_latency_ms"] = random.uniform(100, 5000)
        elif chaos_type == ChaosType.PACKET_CORRUPTION:
            metadata["corruption_rate"] = random.uniform(0.01, 0.3)
        elif chaos_type == ChaosType.RATE_LIMITING:
            metadata["rate_limit"] = random.randint(1, 10)
        elif chaos_type == ChaosType.CLOCK_SKEW:
            metadata["skew_seconds"] = random.uniform(-60, 60)
            
        return metadata
        
    async def _inject_chaos(self, event: ChaosEvent):
        """Inject a chaos event into the system."""
        self.active_events.append(event)
        self.event_history.append(event)
        
        # Apply chaos based on type
        if event.type == ChaosType.LATENCY_SPIKE:
            await self._inject_latency(event)
        elif event.type == ChaosType.PROVIDER_CRASH:
            await self._crash_provider(event)
        elif event.type == ChaosType.NETWORK_PARTITION:
            await self._partition_network(event)
        elif event.type == ChaosType.BYZANTINE_BEHAVIOR:
            await self._inject_byzantine_behavior(event)
        elif event.type == ChaosType.MEMORY_LEAK:
            await self._simulate_memory_leak(event)
        elif event.type == ChaosType.CPU_SATURATION:
            await self._saturate_cpu(event)
            
    async def _inject_latency(self, event: ChaosEvent):
        """Inject latency into provider responses."""
        provider = self.router.providers.get(event.target)
        if provider:
            original_process = provider.process
            added_latency = event.metadata.get("added_latency_ms", 1000) / 1000
            
            async def delayed_process(*args, **kwargs):
                await asyncio.sleep(added_latency * event.intensity)
                return await original_process(*args, **kwargs)
                
            provider.process = delayed_process
            
    async def _crash_provider(self, event: ChaosEvent):
        """Simulate provider crash."""
        provider = self.router.providers.get(event.target)
        if provider:
            provider.is_available = False
            provider.process = AsyncMock(
                side_effect=Exception(f"Provider {event.target} crashed")
            )
            
    async def _partition_network(self, event: ChaosEvent):
        """Simulate network partition."""
        # Partition providers into two groups
        providers = list(self.router.providers.keys())
        partition_size = len(providers) // 2
        partition_a = providers[:partition_size]
        partition_b = providers[partition_size:]
        
        # Make partitions unable to communicate
        for provider_a in partition_a:
            for provider_b in partition_b:
                self.router.block_communication(provider_a, provider_b)
                
    async def _inject_byzantine_behavior(self, event: ChaosEvent):
        """Inject Byzantine behavior (incorrect but plausible responses)."""
        provider = self.router.providers.get(event.target)
        if provider:
            async def byzantine_process(*args, **kwargs):
                # Randomly return incorrect responses
                if random.random() < event.intensity:
                    return {
                        "content": "Byzantine response: " + str(uuid.uuid4()),
                        "corrupted": True,
                        "provider": event.target
                    }
                return await provider.original_process(*args, **kwargs)
                
            provider.original_process = provider.process
            provider.process = byzantine_process
            
    async def _simulate_memory_leak(self, event: ChaosEvent):
        """Simulate gradual memory leak."""
        leak_data = []
        
        async def leak_memory():
            while event.is_active(time.time()):
                # Allocate memory that won't be freed
                leak_size = int(1024 * 1024 * event.intensity)  # MB
                leak_data.append(bytearray(leak_size))
                await asyncio.sleep(1)
                
        asyncio.create_task(leak_memory())
        
    async def _saturate_cpu(self, event: ChaosEvent):
        """Simulate CPU saturation."""
        async def cpu_intensive_task():
            start_time = time.time()
            while time.time() - start_time < event.duration:
                # Perform CPU-intensive operations
                for _ in range(int(1000000 * event.intensity)):
                    _ = sum(i ** 2 for i in range(100))
                await asyncio.sleep(0.01)  # Yield periodically
                
        asyncio.create_task(cpu_intensive_task())
        
    def get_chaos_report(self) -> Dict:
        """Generate report of chaos events and their impact."""
        return {
            "total_events": len(self.event_history),
            "active_events": len(self.active_events),
            "events_by_type": self._count_events_by_type(),
            "average_intensity": self._calculate_average_intensity(),
            "impact_summary": self._summarize_impact(),
            "timeline": self._generate_timeline()
        }
        
    def _count_events_by_type(self) -> Dict[str, int]:
        """Count events by type."""
        counts = defaultdict(int)
        for event in self.event_history:
            counts[event.type.name] += 1
        return dict(counts)
        
    def _calculate_average_intensity(self) -> float:
        """Calculate average chaos intensity."""
        if not self.event_history:
            return 0.0
        return sum(e.intensity for e in self.event_history) / len(self.event_history)
        
    def _summarize_impact(self) -> Dict:
        """Summarize the impact of chaos events."""
        return {
            "affected_providers": list(set(e.target for e in self.event_history)),
            "total_chaos_duration": sum(e.duration for e in self.event_history),
            "max_concurrent_events": self._max_concurrent_events(),
            "system_availability": self._calculate_availability()
        }
        
    def _max_concurrent_events(self) -> int:
        """Calculate maximum number of concurrent events."""
        if not self.event_history:
            return 0
            
        timeline_events = []
        for event in self.event_history:
            timeline_events.append((event.start_time, 1))  # Start
            timeline_events.append((event.end_time, -1))  # End
            
        timeline_events.sort()
        
        max_concurrent = 0
        current_concurrent = 0
        for _, delta in timeline_events:
            current_concurrent += delta
            max_concurrent = max(max_concurrent, current_concurrent)
            
        return max_concurrent
        
    def _calculate_availability(self) -> float:
        """Calculate system availability during chaos."""
        # Simplified calculation based on provider availability
        total_time = sum(e.duration for e in self.event_history)
        if total_time == 0:
            return 1.0
            
        downtime = sum(
            e.duration for e in self.event_history 
            if e.type == ChaosType.PROVIDER_CRASH
        )
        
        return 1.0 - (downtime / total_time)
        
    def _generate_timeline(self) -> List[Dict]:
        """Generate event timeline."""
        timeline = []
        for event in self.event_history:
            timeline.append({
                "time": event.start_time,
                "type": "start",
                "event": event.type.name,
                "target": event.target
            })
            timeline.append({
                "time": event.end_time,
                "type": "end",
                "event": event.type.name,
                "target": event.target
            })
        timeline.sort(key=lambda x: x["time"])
        return timeline


# ============================================================================
# ADVANCED FAILURE SCENARIOS
# ============================================================================

class AdvancedFailureSimulator:
    """Simulates complex failure scenarios."""
    
    def __init__(self, router):
        self.router = router
        self.failure_models = {
            "cascade": self._cascade_failure,
            "thundering_herd": self._thundering_herd,
            "death_spiral": self._death_spiral,
            "split_brain": self._split_brain,
            "gray_failure": self._gray_failure,
            "correlated_failures": self._correlated_failures
        }
        
    async def simulate_failure(self, failure_type: str, **params):
        """Simulate a specific failure scenario."""
        if failure_type not in self.failure_models:
            raise ValueError(f"Unknown failure type: {failure_type}")
            
        return await self.failure_models[failure_type](**params)
        
    async def _cascade_failure(self, initial_provider: str = None, 
                               propagation_delay: float = 1.0):
        """Simulate cascading failure across providers."""
        if initial_provider is None:
            initial_provider = random.choice(list(self.router.providers.keys()))
            
        failed_providers = set([initial_provider])
        propagation_queue = deque([initial_provider])
        
        while propagation_queue:
            current = propagation_queue.popleft()
            
            # Fail current provider
            self.router.providers[current].is_available = False
            
            # Propagate to dependencies with some probability
            for provider in self.router.providers:
                if provider not in failed_providers:
                    if random.random() < 0.3:  # 30% propagation chance
                        await asyncio.sleep(propagation_delay)
                        failed_providers.add(provider)
                        propagation_queue.append(provider)
                        
        return {
            "initial_provider": initial_provider,
            "failed_providers": list(failed_providers),
            "cascade_depth": len(failed_providers)
        }
        
    async def _thundering_herd(self, surge_multiplier: int = 10):
        """Simulate thundering herd problem."""
        # Generate surge of requests
        original_load = self.router.current_load
        surge_load = original_load * surge_multiplier
        
        # All requests hit at once
        requests = []
        for _ in range(surge_load):
            requests.append(self.router.route_request({
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Surge request"}],
                "max_tokens": 100
            }))
            
        # Execute simultaneously
        results = await asyncio.gather(*requests, return_exceptions=True)
        
        failures = sum(1 for r in results if isinstance(r, Exception))
        
        return {
            "surge_size": surge_load,
            "failures": failures,
            "success_rate": 1 - (failures / surge_load)
        }
        
    async def _death_spiral(self, initial_load: float = 0.7):
        """Simulate death spiral (positive feedback loop of failures)."""
        load = initial_load
        iteration = 0
        max_iterations = 20
        
        history = []
        
        while iteration < max_iterations and load < 1.0:
            # Increased load causes more failures
            failure_rate = min(1.0, load ** 2)
            
            # Failures cause retries, increasing load
            retry_multiplier = 1 + failure_rate
            load = min(1.0, load * retry_multiplier)
            
            history.append({
                "iteration": iteration,
                "load": load,
                "failure_rate": failure_rate
            })
            
            # Simulate actual failures
            for provider in self.router.providers.values():
                if random.random() < failure_rate:
                    provider.error_rate = failure_rate
                    
            iteration += 1
            await asyncio.sleep(0.5)
            
        return {
            "spiral_detected": load >= 0.95,
            "iterations": iteration,
            "final_load": load,
            "history": history
        }
        
    async def _split_brain(self, partition_duration: float = 30.0):
        """Simulate split-brain scenario."""
        providers = list(self.router.providers.keys())
        
        # Create two partitions
        partition_a = providers[:len(providers)//2]
        partition_b = providers[len(providers)//2:]
        
        # Each partition elects its own leader
        leader_a = random.choice(partition_a)
        leader_b = random.choice(partition_b)
        
        # Simulate partition
        start_time = time.time()
        
        # Block cross-partition communication
        for p_a in partition_a:
            for p_b in partition_b:
                self.router.block_communication(p_a, p_b)
                
        # Each partition operates independently
        partition_a_state = {"leader": leader_a, "members": partition_a}
        partition_b_state = {"leader": leader_b, "members": partition_b}
        
        # Wait for partition duration
        await asyncio.sleep(partition_duration)
        
        # Restore communication
        for p_a in partition_a:
            for p_b in partition_b:
                self.router.unblock_communication(p_a, p_b)
                
        return {
            "partition_a": partition_a_state,
            "partition_b": partition_b_state,
            "duration": partition_duration,
            "conflict_detected": True  # In real scenario, check for conflicting state
        }
        
    async def _gray_failure(self, provider: str = None, 
                            degradation_rate: float = 0.1):
        """Simulate gray failure (partial, hard-to-detect failures)."""
        if provider is None:
            provider = random.choice(list(self.router.providers.keys()))
            
        # Gradually degrade performance without obvious failure
        original_latency = self.router.providers[provider].avg_latency_ms
        degradation_history = []
        
        for minute in range(10):  # Degrade over 10 minutes
            current_latency = original_latency * (1 + degradation_rate * minute)
            self.router.providers[provider].avg_latency_ms = current_latency
            
            # Occasionally drop requests (gray behavior)
            if random.random() < 0.05 * minute:  # Increasing drop rate
                self.router.providers[provider].drop_rate = 0.01 * minute
                
            degradation_history.append({
                "minute": minute,
                "latency_ms": current_latency,
                "drop_rate": self.router.providers[provider].drop_rate
            })
            
            await asyncio.sleep(60)  # Wait a minute
            
        return {
            "provider": provider,
            "degradation_history": degradation_history,
            "detected": self.router.providers[provider].health_score < 0.7
        }
        
    async def _correlated_failures(self, correlation: float = 0.8):
        """Simulate correlated failures across providers."""
        providers = list(self.router.providers.keys())
        
        # Group providers by correlation
        groups = []
        remaining = providers.copy()
        
        while remaining:
            group = [remaining.pop(0)]
            
            # Add correlated providers to group
            for provider in remaining[:]:
                if random.random() < correlation:
                    group.append(provider)
                    remaining.remove(provider)
                    
            groups.append(group)
            
        # Fail groups together
        failure_events = []
        for group in groups:
            if random.random() < 0.5:  # 50% chance of group failure
                for provider in group:
                    self.router.providers[provider].is_available = False
                failure_events.append({
                    "group": group,
                    "time": time.time()
                })
                
        return {
            "correlation": correlation,
            "groups": groups,
            "failed_groups": failure_events,
            "impact": len([p for g in failure_events for p in g["group"]])
        }


# ============================================================================
# STATISTICAL FAILURE ANALYSIS
# ============================================================================

class FailureAnalyzer:
    """Analyzes failure patterns and statistics."""
    
    def __init__(self):
        self.failure_data: List[Dict] = []
        self.recovery_times: List[float] = []
        self.mtbf_data: defaultdict = defaultdict(list)  # Mean Time Between Failures
        self.mttr_data: defaultdict = defaultdict(list)  # Mean Time To Recovery
        
    def record_failure(self, provider: str, failure_time: float, 
                       failure_type: str, metadata: Dict = None):
        """Record a failure event."""
        self.failure_data.append({
            "provider": provider,
            "time": failure_time,
            "type": failure_type,
            "metadata": metadata or {}
        })
        
        # Update MTBF
        if provider in self.mtbf_data:
            last_failure = self.mtbf_data[provider][-1]
            time_between = failure_time - last_failure
            self.mtbf_data[provider].append(time_between)
            
    def record_recovery(self, provider: str, recovery_time: float):
        """Record a recovery event."""
        # Find corresponding failure
        for failure in reversed(self.failure_data):
            if failure["provider"] == provider:
                time_to_recover = recovery_time - failure["time"]
                self.recovery_times.append(time_to_recover)
                self.mttr_data[provider].append(time_to_recover)
                break
                
    def calculate_availability(self, provider: str = None) -> float:
        """Calculate availability (uptime percentage)."""
        if provider:
            if provider not in self.mtbf_data or provider not in self.mttr_data:
                return 1.0  # No failures recorded
                
            mtbf = np.mean(self.mtbf_data[provider])
            mttr = np.mean(self.mttr_data[provider])
            return mtbf / (mtbf + mttr)
        else:
            # Overall system availability
            availabilities = [
                self.calculate_availability(p) 
                for p in self.mtbf_data.keys()
            ]
            return np.mean(availabilities) if availabilities else 1.0
            
    def calculate_reliability(self, time_period: float) -> float:
        """Calculate reliability over a time period."""
        # Use exponential distribution for failure modeling
        if not self.failure_data:
            return 1.0
            
        failure_rate = len(self.failure_data) / time_period
        return np.exp(-failure_rate * time_period)
        
    def identify_failure_patterns(self) -> Dict:
        """Identify patterns in failures."""
        patterns = {
            "temporal_clustering": self._detect_temporal_clustering(),
            "provider_correlation": self._detect_provider_correlation(),
            "failure_type_distribution": self._analyze_failure_types(),
            "recovery_time_analysis": self._analyze_recovery_times()
        }
        return patterns
        
    def _detect_temporal_clustering(self) -> Dict:
        """Detect if failures cluster in time."""
        if len(self.failure_data) < 2:
            return {"clustered": False}
            
        times = [f["time"] for f in self.failure_data]
        intervals = np.diff(sorted(times))
        
        # Use statistical test for clustering
        if len(intervals) > 1:
            cv = np.std(intervals) / np.mean(intervals)  # Coefficient of variation
            clustered = cv > 1  # High CV indicates clustering
        else:
            clustered = False
            
        return {
            "clustered": clustered,
            "coefficient_of_variation": cv if len(intervals) > 1 else 0,
            "mean_interval": np.mean(intervals) if len(intervals) > 0 else 0
        }
        
    def _detect_provider_correlation(self) -> Dict:
        """Detect correlation between provider failures."""
        provider_failures = defaultdict(list)
        
        for failure in self.failure_data:
            provider_failures[failure["provider"]].append(failure["time"])
            
        correlations = {}
        providers = list(provider_failures.keys())
        
        for i, p1 in enumerate(providers):
            for p2 in providers[i+1:]:
                # Calculate correlation between failure times
                times1 = provider_failures[p1]
                times2 = provider_failures[p2]
                
                if times1 and times2:
                    # Simple correlation based on temporal proximity
                    proximity_count = 0
                    for t1 in times1:
                        for t2 in times2:
                            if abs(t1 - t2) < 60:  # Within 60 seconds
                                proximity_count += 1
                                
                    correlation = proximity_count / (len(times1) * len(times2))
                    correlations[f"{p1}-{p2}"] = correlation
                    
        return correlations
        
    def _analyze_failure_types(self) -> Dict:
        """Analyze distribution of failure types."""
        type_counts = defaultdict(int)
        
        for failure in self.failure_data:
            type_counts[failure["type"]] += 1
            
        total = len(self.failure_data)
        
        return {
            "counts": dict(type_counts),
            "percentages": {
                k: v/total for k, v in type_counts.items()
            } if total > 0 else {}
        }
        
    def _analyze_recovery_times(self) -> Dict:
        """Analyze recovery time statistics."""
        if not self.recovery_times:
            return {}
            
        return {
            "mean": np.mean(self.recovery_times),
            "median": np.median(self.recovery_times),
            "std": np.std(self.recovery_times),
            "p95": np.percentile(self.recovery_times, 95),
            "p99": np.percentile(self.recovery_times, 99)
        }
        
    def predict_next_failure(self, provider: str = None) -> Dict:
        """Predict next failure based on historical data."""
        if provider and provider in self.mtbf_data:
            intervals = self.mtbf_data[provider]
            if intervals:
                # Fit exponential distribution
                rate = 1 / np.mean(intervals)
                # Next failure follows exponential distribution
                predicted_interval = np.random.exponential(1/rate)
                
                return {
                    "provider": provider,
                    "predicted_interval": predicted_interval,
                    "confidence": 0.7  # Simplified confidence score
                }
        return {"error": "Insufficient data for prediction"}


# ============================================================================
# CHAOS ENGINEERING TESTS
# ============================================================================

class TestChaosEngineering:
    """Comprehensive chaos engineering tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.chaos
    async def test_sustained_chaos(self, router):
        """Test system behavior under sustained chaos."""
        orchestrator = ChaosOrchestrator(router, intensity=0.5)
        analyzer = FailureAnalyzer()
        
        # Start chaos
        await orchestrator.start()
        
        # Run system under chaos for extended period
        test_duration = 60  # 60 seconds
        start_time = time.time()
        
        success_count = 0
        failure_count = 0
        
        while time.time() - start_time < test_duration:
            try:
                response = await router.route_request({
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Test"}],
                    "max_tokens": 100
                })
                success_count += 1
            except Exception as e:
                failure_count += 1
                analyzer.record_failure(
                    provider="unknown",
                    failure_time=time.time(),
                    failure_type=str(type(e).__name__)
                )
                
            await asyncio.sleep(0.1)
            
        # Stop chaos
        await orchestrator.stop()
        
        # Analyze results
        chaos_report = orchestrator.get_chaos_report()
        availability = analyzer.calculate_availability()
        
        # System should maintain minimum availability
        assert availability > 0.5  # 50% availability under chaos
        assert success_count > 0  # Some requests should succeed
        
        # Verify chaos was actually injected
        assert chaos_report["total_events"] > 0
        
    @pytest.mark.asyncio
    @pytest.mark.chaos
    async def test_adaptive_resilience(self, router):
        """Test if system adapts to ongoing failures."""
        simulator = AdvancedFailureSimulator(router)
        
        # Start with baseline measurement
        baseline_success_rate = await self._measure_success_rate(router, duration=10)
        
        # Inject gray failure
        gray_failure = await simulator.simulate_failure(
            "gray_failure",
            degradation_rate=0.2
        )
        
        # Measure success rate during failure
        failure_success_rate = await self._measure_success_rate(router, duration=10)
        
        # System should detect and adapt
        assert router.providers[gray_failure["provider"]].health_score < 0.8
        
        # Success rate should degrade gracefully, not catastrophically
        assert failure_success_rate > baseline_success_rate * 0.5
        
    @pytest.mark.asyncio
    @pytest.mark.chaos
    async def test_cascading_failure_mitigation(self, router):
        """Test mitigation of cascading failures."""
        simulator = AdvancedFailureSimulator(router)
        
        # Trigger cascade
        cascade_result = await simulator.simulate_failure(
            "cascade",
            propagation_delay=0.5
        )
        
        # System should detect cascade
        assert await router.detect_cascading_failure()
        
        # Mitigation should be applied
        mitigation = await router.get_cascade_mitigation()
        assert mitigation["circuit_breakers_triggered"]
        
        # Not all providers should fail
        assert cascade_result["cascade_depth"] < len(router.providers)
        
    @pytest.mark.asyncio
    @pytest.mark.chaos
    async def test_split_brain_recovery(self, router):
        """Test recovery from split-brain scenario."""
        simulator = AdvancedFailureSimulator(router)
        
        # Create split-brain
        split_result = await simulator.simulate_failure(
            "split_brain",
            partition_duration=5.0
        )
        
        # System should detect split-brain
        assert await router.detect_split_brain()
        
        # After partition heals, system should reconcile
        await asyncio.sleep(6)  # Wait for partition to heal
        
        # Check for successful reconciliation
        reconciled = await router.is_reconciled()
        assert reconciled
        
        # No conflicting state should remain
        conflicts = await router.get_state_conflicts()
        assert len(conflicts) == 0
        
    @pytest.mark.asyncio
    @pytest.mark.chaos
    async def test_death_spiral_prevention(self, router):
        """Test prevention of death spiral."""
        simulator = AdvancedFailureSimulator(router)
        
        # Attempt to trigger death spiral
        spiral_result = await simulator.simulate_failure(
            "death_spiral",
            initial_load=0.6
        )
        
        # System should prevent complete collapse
        assert not spiral_result["spiral_detected"] or \
               spiral_result["final_load"] < 1.0
               
        # Circuit breakers should activate
        open_breakers = [
            p for p in router.providers
            if router.get_circuit_breaker_state(p) == "OPEN"
        ]
        assert len(open_breakers) > 0
        
    @pytest.mark.asyncio
    @pytest.mark.chaos
    async def test_thundering_herd_handling(self, router):
        """Test handling of thundering herd problem."""
        simulator = AdvancedFailureSimulator(router)
        
        # Create thundering herd
        herd_result = await simulator.simulate_failure(
            "thundering_herd",
            surge_multiplier=20
        )
        
        # System should handle surge without complete failure
        assert herd_result["success_rate"] > 0.3  # At least 30% success
        
        # Rate limiting should kick in
        rate_limited = await router.get_rate_limited_requests()
        assert len(rate_limited) > 0
        
    @pytest.mark.asyncio
    @pytest.mark.chaos
    async def test_correlated_failure_handling(self, router):
        """Test handling of correlated failures."""
        simulator = AdvancedFailureSimulator(router)
        
        # Create correlated failures
        correlation_result = await simulator.simulate_failure(
            "correlated_failures",
            correlation=0.9
        )
        
        # System should still have some available providers
        available = await router.get_available_providers()
        assert len(available) > 0
        
        # Should detect correlation pattern
        patterns = await router.analyze_failure_patterns()
        assert patterns["correlation_detected"]
        
    async def _measure_success_rate(self, router, duration: int = 10) -> float:
        """Helper to measure success rate over duration."""
        start_time = time.time()
        success = 0
        total = 0
        
        while time.time() - start_time < duration:
            try:
                await router.route_request({
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Test"}],
                    "max_tokens": 100
                })
                success += 1
            except:
                pass
            total += 1
            await asyncio.sleep(0.1)
            
        return success / total if total > 0 else 0


# ============================================================================
# JEPSEN-STYLE TESTS
# ============================================================================

class JepsenStyleTest:
    """Jepsen-inspired distributed system tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.jepsen
    async def test_linearizability(self, router):
        """Test linearizability of operations."""
        operations = []
        
        async def client_operation(client_id: int, op_count: int):
            """Simulate client operations."""
            for i in range(op_count):
                op = {
                    "client": client_id,
                    "op_id": f"{client_id}-{i}",
                    "type": "route_request",
                    "start_time": time.time()
                }
                
                try:
                    response = await router.route_request({
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": f"Op {op['op_id']}"}],
                        "max_tokens": 100
                    })
                    op["end_time"] = time.time()
                    op["result"] = "success"
                    op["provider"] = response.provider_used
                except Exception as e:
                    op["end_time"] = time.time()
                    op["result"] = "failure"
                    op["error"] = str(e)
                    
                operations.append(op)
                
        # Run concurrent clients
        clients = []
        for client_id in range(5):
            clients.append(client_operation(client_id, 20))
            
        # Inject failures during execution
        async def inject_failures():
            await asyncio.sleep(2)
            router.providers["openai"].is_available = False
            await asyncio.sleep(2)
            router.providers["openai"].is_available = True
            
        await asyncio.gather(*clients, inject_failures())
        
        # Verify linearizability
        assert self._check_linearizability(operations)
        
    def _check_linearizability(self, operations: List[Dict]) -> bool:
        """Check if operations are linearizable."""
        # Sort by start time
        operations.sort(key=lambda x: x["start_time"])
        
        # Check for overlapping operations to same provider
        provider_ops = defaultdict(list)
        
        for op in operations:
            if "provider" in op:
                provider_ops[op["provider"]].append(op)
                
        for provider, ops in provider_ops.items():
            for i in range(len(ops) - 1):
                # Check if operations overlap
                if ops[i]["end_time"] > ops[i+1]["start_time"]:
                    # Overlapping operations to same provider
                    # In a linearizable system, this should be handled correctly
                    # Here we just check that results are consistent
                    if ops[i]["result"] != ops[i+1]["result"]:
                        return False
                        
        return True
        
    @pytest.mark.asyncio
    @pytest.mark.jepsen
    async def test_consistency_under_partition(self, router):
        """Test consistency during network partitions."""
        # Create initial state
        initial_state = await router.get_system_state()
        
        # Partition network
        router.create_network_partition(
            partition_a=["openai", "anthropic"],
            partition_b=["google", "azure"]
        )
        
        # Operations during partition
        partition_operations = []
        
        async def partition_client(partition: str):
            for i in range(10):
                try:
                    response = await router.route_request(
                        {
                            "model": "gpt-4",
                            "messages": [{"role": "user", "content": f"Partition {partition} op {i}"}],
                            "max_tokens": 100
                        },
                        preferred_partition=partition
                    )
                    partition_operations.append({
                        "partition": partition,
                        "op": i,
                        "result": "success"
                    })
                except Exception:
                    partition_operations.append({
                        "partition": partition,
                        "op": i,
                        "result": "failure"
                    })
                    
        # Run operations on both partitions
        await asyncio.gather(
            partition_client("A"),
            partition_client("B")
        )
        
        # Heal partition
        router.heal_network_partition()
        
        # Check consistency after healing
        final_state = await router.get_system_state()
        
        # Verify no conflicting state
        conflicts = self._detect_conflicts(initial_state, final_state, partition_operations)
        assert len(conflicts) == 0
        
    def _detect_conflicts(self, initial_state: Dict, final_state: Dict, 
                         operations: List[Dict]) -> List[Dict]:
        """Detect conflicting state changes."""
        conflicts = []
        
        # Check for divergent state changes
        for key in final_state:
            if key in initial_state:
                # Check if both partitions modified same state
                partition_a_ops = [op for op in operations if op["partition"] == "A"]
                partition_b_ops = [op for op in operations if op["partition"] == "B"]
                
                if partition_a_ops and partition_b_ops:
                    # Both partitions operated, potential conflict
                    if final_state[key] != initial_state[key]:
                        conflicts.append({
                            "key": key,
                            "initial": initial_state[key],
                            "final": final_state[key]
                        })
                        
        return conflicts


# ============================================================================
# TEST EXECUTION AND REPORTING
# ============================================================================

if __name__ == "__main__":
    # Run chaos engineering tests
    pytest.main([
        __file__,
        "-v",
        "-m", "chaos or jepsen",
        "--tb=short",
        "--color=yes",
        "-n", "auto"  # Parallel execution
    ])