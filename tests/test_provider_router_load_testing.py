"""Load Testing and Performance Benchmarking Suite for Provider Router.

This module implements comprehensive load testing using Locust framework
to validate performance under various load conditions.

Performance Targets:
- Sub-100ms routing latency (P99)
- 1000+ requests per second throughput
- <5% error rate under peak load
- Graceful degradation under overload
- Efficient resource utilization
"""

import json
import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import gevent
from locust import HttpUser, TaskSet, task, between, events, LoadTestShape
from locust.contrib.fasthttp import FastHttpUser
from locust.exception import RescheduleTask
from locust.stats import stats_printer, stats_history
import numpy as np


# ============================================================================
# CUSTOM METRICS COLLECTION
# ============================================================================

class MetricsCollector:
    """Collects custom metrics during load testing."""
    
    def __init__(self):
        self.latencies = []
        self.provider_usage = {}
        self.fallback_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.circuit_breaker_trips = 0
        self.cost_total = 0.0
        self._start_time = time.time()
        
    def record_latency(self, latency_ms: float):
        """Record request latency."""
        self.latencies.append(latency_ms)
        
    def record_provider_use(self, provider: str):
        """Record provider usage."""
        if provider not in self.provider_usage:
            self.provider_usage[provider] = 0
        self.provider_usage[provider] += 1
        
    def record_fallback(self):
        """Record fallback occurrence."""
        self.fallback_count += 1
        
    def record_cache_hit(self):
        """Record cache hit."""
        self.cache_hits += 1
        
    def record_cache_miss(self):
        """Record cache miss."""
        self.cache_misses += 1
        
    def record_circuit_breaker_trip(self):
        """Record circuit breaker trip."""
        self.circuit_breaker_trips += 1
        
    def record_cost(self, cost: float):
        """Record request cost."""
        self.cost_total += cost
        
    def get_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles."""
        if not self.latencies:
            return {}
            
        sorted_latencies = sorted(self.latencies)
        return {
            "p50": np.percentile(sorted_latencies, 50),
            "p75": np.percentile(sorted_latencies, 75),
            "p90": np.percentile(sorted_latencies, 90),
            "p95": np.percentile(sorted_latencies, 95),
            "p99": np.percentile(sorted_latencies, 99),
            "p999": np.percentile(sorted_latencies, 99.9),
        }
        
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
        
    def get_summary(self) -> Dict:
        """Get summary of collected metrics."""
        runtime = time.time() - self._start_time
        
        return {
            "runtime_seconds": runtime,
            "total_requests": len(self.latencies),
            "requests_per_second": len(self.latencies) / runtime if runtime > 0 else 0,
            "latency_percentiles": self.get_percentiles(),
            "provider_distribution": self.provider_usage,
            "fallback_rate": self.fallback_count / len(self.latencies) if self.latencies else 0,
            "cache_hit_rate": self.get_cache_hit_rate(),
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "total_cost": self.cost_total,
            "cost_per_request": self.cost_total / len(self.latencies) if self.latencies else 0
        }


# Global metrics collector
metrics = MetricsCollector()


# ============================================================================
# PROVIDER ROUTER LOAD TEST USER
# ============================================================================

class ProviderRouterUser(FastHttpUser):
    """Fast HTTP user for provider router load testing."""
    
    wait_time = between(0.01, 0.1)  # Aggressive timing for load testing
    
    def on_start(self):
        """Initialize user session."""
        self.user_id = str(uuid.uuid4())
        self.request_count = 0
        self.strategies = ["cost", "speed", "balanced", "capability"]
        self.models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", 
                      "claude-3-sonnet", "gemini-pro", "gemini-flash"]
        
    @task(40)
    def route_simple_request(self):
        """Test simple routing request."""
        self.request_count += 1
        
        request_data = {
            "request": {
                "model": random.choice(self.models),
                "messages": [
                    {"role": "user", "content": f"Test message {self.request_count}"}
                ],
                "max_tokens": random.randint(10, 500),
                "temperature": random.uniform(0, 1)
            },
            "strategy": random.choice(self.strategies),
            "fallback_enabled": True
        }
        
        start_time = time.time()
        
        with self.client.post(
            "/api/v1/route",
            json=request_data,
            catch_response=True
        ) as response:
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Record metrics
                    metrics.record_latency(latency_ms)
                    metrics.record_provider_use(data.get("provider_used", "unknown"))
                    
                    if data.get("fallback_occurred"):
                        metrics.record_fallback()
                        
                    if data.get("cache_hit"):
                        metrics.record_cache_hit()
                    else:
                        metrics.record_cache_miss()
                        
                    if data.get("cost_estimate"):
                        metrics.record_cost(data["cost_estimate"])
                        
                    # Validate response
                    if latency_ms > 100:  # Check SLA
                        response.failure(f"Latency {latency_ms:.2f}ms exceeds 100ms SLA")
                    else:
                        response.success()
                        
                except Exception as e:
                    response.failure(f"Invalid response: {str(e)}")
            else:
                response.failure(f"Status {response.status_code}")
                
    @task(20)
    def route_streaming_request(self):
        """Test streaming request routing."""
        request_data = {
            "request": {
                "model": random.choice(self.models),
                "messages": [
                    {"role": "user", "content": "Generate a story about space"}
                ],
                "max_tokens": random.randint(100, 1000),
                "stream": True
            },
            "strategy": "speed",  # Prefer fast providers for streaming
            "timeout": 30.0
        }
        
        with self.client.post(
            "/api/v1/route/stream",
            json=request_data,
            stream=True,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                chunks_received = 0
                first_chunk_time = None
                
                for chunk in response.iter_lines():
                    if chunk:
                        chunks_received += 1
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                            
                if chunks_received > 0:
                    response.success()
                else:
                    response.failure("No chunks received")
            else:
                response.failure(f"Status {response.status_code}")
                
    @task(10)
    def route_with_fallback_chain(self):
        """Test explicit fallback chain configuration."""
        request_data = {
            "request": {
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": "Test fallback chain"}
                ],
                "max_tokens": 100
            },
            "preferred_providers": ["openai", "anthropic", "google"],
            "exclude_providers": ["azure"],
            "max_retries": 3
        }
        
        with self.client.post(
            "/api/v1/route",
            json=request_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                
                # Verify provider selection
                if data.get("provider_used") not in ["openai", "anthropic", "google"]:
                    response.failure(f"Invalid provider: {data.get('provider_used')}")
                elif data.get("provider_used") == "azure":
                    response.failure("Excluded provider was used")
                else:
                    response.success()
            else:
                response.failure(f"Status {response.status_code}")
                
    @task(10)
    def route_with_cost_limit(self):
        """Test cost-limited routing."""
        cost_limit = random.uniform(0.01, 0.10)
        
        request_data = {
            "request": {
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": "Cost limited request"}
                ],
                "max_tokens": 1000
            },
            "strategy": "cost",
            "cost_limit": cost_limit
        }
        
        with self.client.post(
            "/api/v1/route",
            json=request_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                
                if data.get("cost_estimate", 0) > cost_limit:
                    response.failure(f"Cost {data.get('cost_estimate')} exceeds limit {cost_limit}")
                else:
                    response.success()
            elif response.status_code == 402:  # Payment Required
                response.success()  # Expected when cost limit too low
            else:
                response.failure(f"Status {response.status_code}")
                
    @task(5)
    def check_provider_health(self):
        """Check provider health status."""
        with self.client.get(
            "/api/v1/providers/health",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                
                # Verify health data structure
                if not isinstance(data, dict):
                    response.failure("Invalid health data format")
                else:
                    unhealthy_providers = [
                        p for p, h in data.items() 
                        if h.get("health_score", 1.0) < 0.5
                    ]
                    
                    if len(unhealthy_providers) >= len(data) * 0.5:
                        response.failure("Too many unhealthy providers")
                    else:
                        response.success()
            else:
                response.failure(f"Status {response.status_code}")
                
    @task(5)
    def get_routing_stats(self):
        """Get routing statistics."""
        with self.client.get(
            "/api/v1/stats",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
                
    @task(5)
    def configure_routing(self):
        """Test routing configuration update."""
        config_data = {
            "strategy": random.choice(self.strategies),
            "fallback_chain": random.sample(
                ["openai", "anthropic", "google", "azure"], 
                k=random.randint(2, 4)
            ),
            "circuit_breaker_threshold": random.randint(3, 10),
            "timeout": random.uniform(10, 60)
        }
        
        with self.client.put(
            "/api/v1/config/routing",
            json=config_data,
            catch_response=True
        ) as response:
            if response.status_code in [200, 204]:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
                
    @task(5)
    def test_provider_chain(self):
        """Test provider chain validation."""
        test_data = {
            "chain": ["openai", "anthropic", "google"],
            "test_request": {
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": "Chain test"}
                ],
                "max_tokens": 10
            }
        }
        
        with self.client.post(
            "/api/v1/test/chain",
            json=test_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                
                # Verify all providers were tested
                if len(data.get("results", [])) != 3:
                    response.failure("Not all providers tested")
                else:
                    response.success()
            else:
                response.failure(f"Status {response.status_code}")


# ============================================================================
# SPECIALIZED LOAD TEST SCENARIOS
# ============================================================================

class BurstLoadUser(FastHttpUser):
    """User simulating burst load patterns."""
    
    wait_time = between(0.001, 0.01)  # Very aggressive
    
    @task
    def burst_request(self):
        """Send burst of requests."""
        burst_size = random.randint(5, 20)
        
        for _ in range(burst_size):
            self.client.post(
                "/api/v1/route",
                json={
                    "request": {
                        "model": "gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": "Burst"}],
                        "max_tokens": 50
                    },
                    "strategy": "speed"
                }
            )
            
        # Brief pause between bursts
        gevent.sleep(random.uniform(1, 5))


class SustainedLoadUser(FastHttpUser):
    """User simulating sustained steady load."""
    
    wait_time = between(0.05, 0.15)  # Steady rate
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_id = 0
        
    @task
    def steady_request(self):
        """Send steady stream of requests."""
        self.request_id += 1
        
        self.client.post(
            "/api/v1/route",
            json={
                "request": {
                    "model": "gpt-4" if self.request_id % 10 == 0 else "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": f"Request {self.request_id}"}
                    ],
                    "max_tokens": 200
                },
                "strategy": "balanced"
            }
        )


class StressTestUser(FastHttpUser):
    """User for stress testing beyond normal capacity."""
    
    wait_time = between(0, 0.001)  # No wait
    
    @task
    def stress_request(self):
        """Send requests as fast as possible."""
        # Large request to stress system
        self.client.post(
            "/api/v1/route",
            json={
                "request": {
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": "X" * 1000},  # Large system prompt
                        {"role": "user", "content": "Y" * 500}
                    ],
                    "max_tokens": 4000  # Max tokens
                },
                "strategy": "capability",
                "timeout": 5.0  # Short timeout to trigger failures
            }
        )


# ============================================================================
# CUSTOM LOAD SHAPES
# ============================================================================

class StepLoadShape(LoadTestShape):
    """Step load pattern for gradual increase."""
    
    step_time = 30  # 30 seconds per step
    step_users = 10  # Add 10 users per step
    spawn_rate = 2  # Spawn 2 users per second
    max_users = 100
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > 600:  # 10 minutes max
            return None
            
        current_step = int(run_time / self.step_time)
        target_users = min(self.step_users * current_step, self.max_users)
        
        return (target_users, self.spawn_rate)


class SpikeLoadShape(LoadTestShape):
    """Spike load pattern for sudden traffic increases."""
    
    stages = [
        {"duration": 60, "users": 10, "spawn_rate": 1},   # Warm up
        {"duration": 30, "users": 100, "spawn_rate": 10}, # Spike
        {"duration": 60, "users": 10, "spawn_rate": 1},   # Recovery
        {"duration": 30, "users": 200, "spawn_rate": 20}, # Large spike
        {"duration": 60, "users": 20, "spawn_rate": 2},   # Cool down
    ]
    
    def tick(self):
        run_time = self.get_run_time()
        
        total_duration = 0
        for stage in self.stages:
            total_duration += stage["duration"]
            if run_time < total_duration:
                return (stage["users"], stage["spawn_rate"])
                
        return None


class WaveLoadShape(LoadTestShape):
    """Wave pattern simulating daily traffic patterns."""
    
    min_users = 10
    max_users = 100
    wave_duration = 120  # 2 minutes per wave
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > 600:  # 10 minutes max
            return None
            
        # Sine wave pattern
        wave_position = (run_time % self.wave_duration) / self.wave_duration
        wave_value = (np.sin(2 * np.pi * wave_position) + 1) / 2
        
        target_users = int(self.min_users + 
                          (self.max_users - self.min_users) * wave_value)
        spawn_rate = max(1, target_users // 10)
        
        return (target_users, spawn_rate)


# ============================================================================
# EVENT HOOKS FOR CUSTOM REPORTING
# ============================================================================

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Custom event handler for requests."""
    if exception is None:
        # Record successful request
        metrics.record_latency(response_time)
    else:
        # Record circuit breaker trips
        if "CircuitBreaker" in str(exception):
            metrics.record_circuit_breaker_trip()


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test environment."""
    print("Load test starting...")
    print(f"Target host: {environment.host}")
    print(f"Users: {environment.parsed_options.num_users}")
    print(f"Spawn rate: {environment.parsed_options.spawn_rate}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Finalize test and generate report."""
    print("\nLoad test completed!")
    print("\n" + "="*60)
    print("PERFORMANCE METRICS SUMMARY")
    print("="*60)
    
    summary = metrics.get_summary()
    
    print(f"\nTotal Requests: {summary['total_requests']}")
    print(f"Requests/Second: {summary['requests_per_second']:.2f}")
    print(f"Runtime: {summary['runtime_seconds']:.2f} seconds")
    
    print("\nLatency Percentiles (ms):")
    for percentile, value in summary['latency_percentiles'].items():
        print(f"  {percentile}: {value:.2f}ms")
        
    print("\nProvider Distribution:")
    for provider, count in summary['provider_distribution'].items():
        percentage = (count / summary['total_requests']) * 100
        print(f"  {provider}: {count} ({percentage:.1f}%)")
        
    print(f"\nFallback Rate: {summary['fallback_rate']*100:.2f}%")
    print(f"Cache Hit Rate: {summary['cache_hit_rate']*100:.2f}%")
    print(f"Circuit Breaker Trips: {summary['circuit_breaker_trips']}")
    print(f"Total Cost: ${summary['total_cost']:.4f}")
    print(f"Cost/Request: ${summary['cost_per_request']:.6f}")
    
    # Performance validation
    print("\n" + "="*60)
    print("PERFORMANCE VALIDATION")
    print("="*60)
    
    p99_latency = summary['latency_percentiles'].get('p99', float('inf'))
    rps = summary['requests_per_second']
    
    targets_met = []
    targets_missed = []
    
    # Check latency target
    if p99_latency < 100:
        targets_met.append(f"✓ P99 Latency: {p99_latency:.2f}ms < 100ms")
    else:
        targets_missed.append(f"✗ P99 Latency: {p99_latency:.2f}ms > 100ms target")
        
    # Check throughput target
    if rps > 1000:
        targets_met.append(f"✓ Throughput: {rps:.2f} RPS > 1000 RPS")
    else:
        targets_missed.append(f"✗ Throughput: {rps:.2f} RPS < 1000 RPS target")
        
    # Check cache performance
    cache_hit_rate = summary['cache_hit_rate']
    if cache_hit_rate > 0.8:
        targets_met.append(f"✓ Cache Hit Rate: {cache_hit_rate*100:.1f}% > 80%")
    else:
        targets_missed.append(f"✗ Cache Hit Rate: {cache_hit_rate*100:.1f}% < 80% target")
        
    # Print results
    if targets_met:
        print("\nTargets Met:")
        for target in targets_met:
            print(f"  {target}")
            
    if targets_missed:
        print("\nTargets Missed:")
        for target in targets_missed:
            print(f"  {target}")
            
    overall_result = "PASSED" if not targets_missed else "FAILED"
    print(f"\nOverall Result: {overall_result}")
    print("="*60)


# ============================================================================
# CUSTOM STATISTICS
# ============================================================================

class CustomStats:
    """Custom statistics tracking."""
    
    def __init__(self):
        self.provider_latencies = {}
        self.strategy_performance = {}
        self.time_series_data = []
        
    def record_provider_latency(self, provider: str, latency: float):
        """Record latency per provider."""
        if provider not in self.provider_latencies:
            self.provider_latencies[provider] = []
        self.provider_latencies[provider].append(latency)
        
    def record_strategy_performance(self, strategy: str, success: bool, latency: float):
        """Record performance per routing strategy."""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                "successes": 0,
                "failures": 0,
                "latencies": []
            }
            
        if success:
            self.strategy_performance[strategy]["successes"] += 1
        else:
            self.strategy_performance[strategy]["failures"] += 1
            
        self.strategy_performance[strategy]["latencies"].append(latency)
        
    def record_time_series(self, timestamp: float, rps: float, 
                          active_users: int, error_rate: float):
        """Record time series data point."""
        self.time_series_data.append({
            "timestamp": timestamp,
            "rps": rps,
            "active_users": active_users,
            "error_rate": error_rate
        })
        
    def generate_report(self) -> Dict:
        """Generate detailed performance report."""
        report = {}
        
        # Provider performance comparison
        report["provider_performance"] = {}
        for provider, latencies in self.provider_latencies.items():
            if latencies:
                report["provider_performance"][provider] = {
                    "avg_latency": np.mean(latencies),
                    "p50_latency": np.percentile(latencies, 50),
                    "p95_latency": np.percentile(latencies, 95),
                    "p99_latency": np.percentile(latencies, 99)
                }
                
        # Strategy effectiveness
        report["strategy_effectiveness"] = {}
        for strategy, data in self.strategy_performance.items():
            total = data["successes"] + data["failures"]
            if total > 0:
                report["strategy_effectiveness"][strategy] = {
                    "success_rate": data["successes"] / total,
                    "avg_latency": np.mean(data["latencies"]) if data["latencies"] else 0,
                    "total_requests": total
                }
                
        # Time series analysis
        if self.time_series_data:
            report["peak_performance"] = {
                "max_rps": max(d["rps"] for d in self.time_series_data),
                "max_concurrent_users": max(d["active_users"] for d in self.time_series_data),
                "avg_error_rate": np.mean([d["error_rate"] for d in self.time_series_data])
            }
            
        return report


# Global custom stats
custom_stats = CustomStats()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    from locust import main
    
    # Set default arguments if not provided
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--host", "http://localhost:8000",
            "--users", "100",
            "--spawn-rate", "10",
            "--run-time", "5m",
            "--headless",
            "--only-summary",
            "--print-stats"
        ])
        
    # Run Locust
    main.main()