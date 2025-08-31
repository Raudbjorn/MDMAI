"""Load testing scripts for performance evaluation."""

import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from httpx import AsyncClient

from src.bridge.bridge_server import BridgeServer, create_bridge_app
from src.bridge.models import BridgeConfig


@dataclass
class LoadTestMetrics:
    """Metrics collected during load testing."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = field(default_factory=time.time)
    
    @property
    def duration(self) -> float:
        """Total test duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def requests_per_second(self) -> float:
        """Average requests per second."""
        if self.duration == 0:
            return 0
        return self.total_requests / self.duration
    
    @property
    def success_rate(self) -> float:
        """Percentage of successful requests."""
        if self.total_requests == 0:
            return 0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def avg_response_time(self) -> float:
        """Average response time in milliseconds."""
        if not self.response_times:
            return 0
        return statistics.mean(self.response_times) * 1000
    
    @property
    def median_response_time(self) -> float:
        """Median response time in milliseconds."""
        if not self.response_times:
            return 0
        return statistics.median(self.response_times) * 1000
    
    @property
    def p95_response_time(self) -> float:
        """95th percentile response time in milliseconds."""
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[index] * 1000 if index < len(sorted_times) else sorted_times[-1] * 1000
    
    @property
    def p99_response_time(self) -> float:
        """99th percentile response time in milliseconds."""
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        index = int(len(sorted_times) * 0.99)
        return sorted_times[index] * 1000 if index < len(sorted_times) else sorted_times[-1] * 1000
    
    def print_summary(self):
        """Print a summary of the metrics."""
        print("\n" + "=" * 60)
        print("LOAD TEST RESULTS")
        print("=" * 60)
        print(f"Duration: {self.duration:.2f} seconds")
        print(f"Total Requests: {self.total_requests}")
        print(f"Successful: {self.successful_requests}")
        print(f"Failed: {self.failed_requests}")
        print(f"Success Rate: {self.success_rate:.2f}%")
        print(f"Requests/Second: {self.requests_per_second:.2f}")
        print("\nResponse Times (ms):")
        print(f"  Average: {self.avg_response_time:.2f}")
        print(f"  Median: {self.median_response_time:.2f}")
        print(f"  95th Percentile: {self.p95_response_time:.2f}")
        print(f"  99th Percentile: {self.p99_response_time:.2f}")
        
        if self.error_messages:
            print(f"\nErrors: {len(set(self.error_messages))} unique error types")
            for error in set(self.error_messages)[:5]:  # Show first 5 unique errors
                print(f"  - {error}")
        print("=" * 60 + "\n")


class LoadTestClient:
    """Client for performing load tests."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.metrics = LoadTestMetrics()
    
    async def __aenter__(self):
        """Enter async context."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        if self.session:
            await self.session.close()
    
    async def make_request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict] = None,
        track_metrics: bool = True
    ) -> Tuple[int, Any]:
        """Make an HTTP request and track metrics."""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        url = f"{self.base_url}{path}"
        start_time = time.time()
        
        try:
            async with self.session.request(
                method,
                url,
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_data = await response.json() if response.content_type == "application/json" else None
                status = response.status
                
                if track_metrics:
                    elapsed = time.time() - start_time
                    self.metrics.total_requests += 1
                    self.metrics.response_times.append(elapsed)
                    
                    if 200 <= status < 300:
                        self.metrics.successful_requests += 1
                    else:
                        self.metrics.failed_requests += 1
                        self.metrics.error_messages.append(f"HTTP {status}")
                
                return status, response_data
                
        except asyncio.TimeoutError:
            if track_metrics:
                self.metrics.total_requests += 1
                self.metrics.failed_requests += 1
                self.metrics.error_messages.append("Timeout")
            return 0, None
            
        except Exception as e:
            if track_metrics:
                self.metrics.total_requests += 1
                self.metrics.failed_requests += 1
                self.metrics.error_messages.append(str(e))
            return 0, None
    
    async def create_session(self, client_id: str) -> Optional[str]:
        """Create a new session and return session ID."""
        status, data = await self.make_request(
            "POST",
            "/sessions",
            {"client_id": client_id}
        )
        
        if status == 200 and data:
            return data.get("session_id")
        return None
    
    async def call_tool(
        self,
        session_id: str,
        tool_name: str,
        params: Dict[str, Any]
    ) -> Tuple[int, Any]:
        """Call a tool with the given parameters."""
        return await self.make_request(
            "POST",
            "/tools/call",
            {
                "session_id": session_id,
                "tool": tool_name,
                "params": params
            }
        )
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        status, _ = await self.make_request(
            "DELETE",
            f"/sessions/{session_id}"
        )
        return status == 200


class LoadTestScenarios:
    """Various load testing scenarios."""
    
    @staticmethod
    async def sustained_load(
        client: LoadTestClient,
        duration_seconds: int = 60,
        requests_per_second: int = 10
    ) -> LoadTestMetrics:
        """Sustained load at a constant rate."""
        print(f"\nRunning sustained load test: {requests_per_second} req/s for {duration_seconds}s")
        
        client.metrics.start_time = time.time()
        interval = 1.0 / requests_per_second
        end_time = time.time() + duration_seconds
        
        tasks = []
        
        while time.time() < end_time:
            # Create a request task
            session_id = f"load-test-{random.randint(1000, 9999)}"
            task = asyncio.create_task(
                client.create_session(f"client-{session_id}")
            )
            tasks.append(task)
            
            # Wait for the interval
            await asyncio.sleep(interval)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        client.metrics.end_time = time.time()
        return client.metrics
    
    @staticmethod
    async def burst_load(
        client: LoadTestClient,
        burst_size: int = 100,
        burst_count: int = 5,
        delay_between_bursts: float = 5.0
    ) -> LoadTestMetrics:
        """Burst load pattern with periods of high traffic."""
        print(f"\nRunning burst load test: {burst_count} bursts of {burst_size} requests")
        
        client.metrics.start_time = time.time()
        
        for burst_num in range(burst_count):
            print(f"  Burst {burst_num + 1}/{burst_count}")
            
            # Create burst of requests
            tasks = []
            for i in range(burst_size):
                session_id = f"burst-{burst_num}-{i}"
                task = asyncio.create_task(
                    client.create_session(f"client-{session_id}")
                )
                tasks.append(task)
            
            # Wait for burst to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Delay before next burst
            if burst_num < burst_count - 1:
                await asyncio.sleep(delay_between_bursts)
        
        client.metrics.end_time = time.time()
        return client.metrics
    
    @staticmethod
    async def ramp_up_load(
        client: LoadTestClient,
        initial_rate: int = 1,
        max_rate: int = 50,
        ramp_duration: int = 60
    ) -> LoadTestMetrics:
        """Gradually increase load to find breaking point."""
        print(f"\nRunning ramp-up load test: {initial_rate} to {max_rate} req/s over {ramp_duration}s")
        
        client.metrics.start_time = time.time()
        
        steps = 10
        step_duration = ramp_duration / steps
        rate_increment = (max_rate - initial_rate) / steps
        
        for step in range(steps):
            current_rate = initial_rate + (rate_increment * step)
            print(f"  Step {step + 1}/{steps}: {current_rate:.1f} req/s")
            
            # Run at current rate for step duration
            interval = 1.0 / current_rate
            step_end = time.time() + step_duration
            
            tasks = []
            while time.time() < step_end:
                session_id = f"ramp-{step}-{random.randint(1000, 9999)}"
                task = asyncio.create_task(
                    client.create_session(f"client-{session_id}")
                )
                tasks.append(task)
                await asyncio.sleep(interval)
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check if system is still responsive
            if client.metrics.failed_requests > client.metrics.successful_requests * 0.1:
                print(f"  System showing signs of stress at {current_rate:.1f} req/s")
        
        client.metrics.end_time = time.time()
        return client.metrics
    
    @staticmethod
    async def mixed_workload(
        client: LoadTestClient,
        duration_seconds: int = 60,
        concurrent_sessions: int = 10
    ) -> LoadTestMetrics:
        """Mixed workload simulating real usage patterns."""
        print(f"\nRunning mixed workload test: {concurrent_sessions} concurrent sessions for {duration_seconds}s")
        
        client.metrics.start_time = time.time()
        end_time = time.time() + duration_seconds
        
        # Create initial sessions
        session_ids = []
        for i in range(concurrent_sessions):
            session_id = await client.create_session(f"mixed-client-{i}")
            if session_id:
                session_ids.append(session_id)
        
        # Simulate mixed operations
        operations = ["search", "analyze", "calculate", "fetch", "process"]
        
        async def worker(session_id: str):
            """Worker simulating user actions."""
            while time.time() < end_time:
                # Random operation
                operation = random.choice(operations)
                params = {
                    "input": f"test-{random.randint(1, 100)}",
                    "timestamp": time.time()
                }
                
                await client.call_tool(session_id, operation, params)
                
                # Random delay between operations (simulate thinking time)
                await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # Start workers
        workers = [
            asyncio.create_task(worker(session_id))
            for session_id in session_ids
        ]
        
        # Wait for duration
        await asyncio.sleep(duration_seconds)
        
        # Cancel workers
        for worker_task in workers:
            worker_task.cancel()
        
        await asyncio.gather(*workers, return_exceptions=True)
        
        # Clean up sessions
        cleanup_tasks = [
            client.delete_session(session_id)
            for session_id in session_ids
        ]
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        client.metrics.end_time = time.time()
        return client.metrics
    
    @staticmethod
    async def websocket_load(
        base_url: str = "ws://localhost:8000",
        concurrent_connections: int = 50,
        duration_seconds: int = 30
    ) -> LoadTestMetrics:
        """Load test WebSocket connections."""
        print(f"\nRunning WebSocket load test: {concurrent_connections} connections for {duration_seconds}s")
        
        metrics = LoadTestMetrics()
        metrics.start_time = time.time()
        
        async def websocket_client(client_id: str):
            """Individual WebSocket client."""
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(f"{base_url}/ws") as ws:
                        # Create session
                        await ws.send_json({
                            "type": "create_session",
                            "client_id": client_id
                        })
                        
                        # Receive session created
                        response = await ws.receive_json()
                        if response.get("type") != "session_created":
                            metrics.failed_requests += 1
                            return
                        
                        metrics.successful_requests += 1
                        
                        # Send periodic messages
                        end_time = time.time() + duration_seconds
                        msg_count = 0
                        
                        while time.time() < end_time:
                            start = time.time()
                            
                            await ws.send_json({
                                "jsonrpc": "2.0",
                                "id": f"msg-{msg_count}",
                                "method": "tools/ping",
                                "params": {"timestamp": time.time()}
                            })
                            
                            response = await ws.receive_json()
                            elapsed = time.time() - start
                            
                            metrics.total_requests += 1
                            metrics.response_times.append(elapsed)
                            
                            if response.get("type") == "response":
                                metrics.successful_requests += 1
                            else:
                                metrics.failed_requests += 1
                            
                            msg_count += 1
                            await asyncio.sleep(1)  # Message every second
                            
            except Exception as e:
                metrics.failed_requests += 1
                metrics.error_messages.append(str(e))
        
        # Create concurrent WebSocket connections
        tasks = [
            asyncio.create_task(websocket_client(f"ws-client-{i}"))
            for i in range(concurrent_connections)
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.end_time = time.time()
        return metrics


class TestLoadPerformance:
    """Load and performance tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_sustained_load(self):
        """Test sustained load handling."""
        async with LoadTestClient() as client:
            with patch("src.bridge.session_manager.BridgeSessionManager.create_session") as mock_create:
                with patch("src.bridge.process_manager.MCPProcessManager.create_process") as mock_process:
                    # Mock successful session creation
                    mock_create.return_value = AsyncMock(
                        session_id="test-session",
                        state="ready"
                    )
                    mock_process.return_value = AsyncMock()
                    
                    metrics = await LoadTestScenarios.sustained_load(
                        client,
                        duration_seconds=10,
                        requests_per_second=20
                    )
                    
                    metrics.print_summary()
                    
                    # Assertions
                    assert metrics.success_rate > 95  # Should handle most requests
                    assert metrics.avg_response_time < 100  # Should be responsive
    
    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_burst_load(self):
        """Test burst load handling."""
        async with LoadTestClient() as client:
            metrics = await LoadTestScenarios.burst_load(
                client,
                burst_size=50,
                burst_count=3,
                delay_between_bursts=2.0
            )
            
            metrics.print_summary()
            
            # System should handle bursts
            assert metrics.total_requests == 150
            assert metrics.success_rate > 90
    
    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_ramp_up_load(self):
        """Test gradual load increase."""
        async with LoadTestClient() as client:
            metrics = await LoadTestScenarios.ramp_up_load(
                client,
                initial_rate=5,
                max_rate=100,
                ramp_duration=30
            )
            
            metrics.print_summary()
            
            # Should handle initial load well
            assert metrics.total_requests > 0
            # Response time should degrade gracefully
            assert metrics.p99_response_time < 5000  # Under 5 seconds
    
    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_mixed_workload(self):
        """Test mixed workload patterns."""
        async with LoadTestClient() as client:
            with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
                mock_send.return_value = {"result": "success"}
                
                metrics = await LoadTestScenarios.mixed_workload(
                    client,
                    duration_seconds=20,
                    concurrent_sessions=5
                )
                
                metrics.print_summary()
                
                # Should maintain good performance
                assert metrics.success_rate > 95
                assert metrics.median_response_time < 50
    
    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_websocket_load(self):
        """Test WebSocket connection load."""
        metrics = await LoadTestScenarios.websocket_load(
            concurrent_connections=20,
            duration_seconds=10
        )
        
        metrics.print_summary()
        
        # Should handle concurrent WebSocket connections
        assert metrics.total_requests > 0
        assert metrics.success_rate > 90


class StressTestScenarios:
    """Stress testing scenarios to find breaking points."""
    
    @staticmethod
    async def memory_stress_test(
        client: LoadTestClient,
        session_count: int = 1000
    ) -> Dict[str, Any]:
        """Create many sessions to stress memory."""
        print(f"\nRunning memory stress test: Creating {session_count} sessions")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        session_ids = []
        batch_size = 100
        
        for batch in range(0, session_count, batch_size):
            tasks = []
            for i in range(batch_size):
                if batch + i >= session_count:
                    break
                task = asyncio.create_task(
                    client.create_session(f"stress-{batch + i}")
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            session_ids.extend([r for r in results if r and not isinstance(r, Exception)])
            
            print(f"  Created {len(session_ids)} sessions")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Clean up
        cleanup_tasks = [
            client.delete_session(sid)
            for sid in session_ids[:100]  # Clean up first 100
        ]
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        return {
            "sessions_created": len(session_ids),
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": memory_increase,
            "memory_per_session_kb": (memory_increase * 1024) / len(session_ids) if session_ids else 0
        }
    
    @staticmethod
    async def cpu_stress_test(
        client: LoadTestClient,
        duration_seconds: int = 30,
        complexity_level: int = 5
    ) -> Dict[str, Any]:
        """Generate CPU-intensive requests."""
        print(f"\nRunning CPU stress test: Complexity {complexity_level} for {duration_seconds}s")
        
        import psutil
        
        cpu_percentages = []
        
        async def monitor_cpu():
            """Monitor CPU usage during test."""
            while True:
                cpu_percentages.append(psutil.cpu_percent(interval=1))
                await asyncio.sleep(1)
        
        # Start CPU monitoring
        monitor_task = asyncio.create_task(monitor_cpu())
        
        # Create a session
        session_id = await client.create_session("cpu-stress-test")
        
        # Generate CPU-intensive requests
        end_time = time.time() + duration_seconds
        request_count = 0
        
        while time.time() < end_time:
            # Simulate complex computation request
            params = {
                "operation": "complex_calculation",
                "data": list(range(1000 * complexity_level)),
                "iterations": 100 * complexity_level
            }
            
            await client.call_tool(session_id, "compute", params)
            request_count += 1
        
        # Stop monitoring
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        # Clean up
        await client.delete_session(session_id)
        
        return {
            "requests_sent": request_count,
            "avg_cpu_percent": statistics.mean(cpu_percentages) if cpu_percentages else 0,
            "max_cpu_percent": max(cpu_percentages) if cpu_percentages else 0,
            "min_cpu_percent": min(cpu_percentages) if cpu_percentages else 0
        }
    
    @staticmethod
    async def connection_limit_test(
        base_url: str = "http://localhost:8000",
        max_connections: int = 10000
    ) -> Dict[str, Any]:
        """Test maximum concurrent connections."""
        print(f"\nRunning connection limit test: Up to {max_connections} connections")
        
        successful_connections = 0
        failed_connections = 0
        
        async def create_connection(conn_id: int):
            """Try to create a connection."""
            nonlocal successful_connections, failed_connections
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{base_url}/sessions",
                        json={"client_id": f"conn-{conn_id}"},
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            successful_connections += 1
                            return True
                        else:
                            failed_connections += 1
                            return False
            except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as e:
                logger.error(f"Connection {conn_id} failed: {e}")
                failed_connections += 1
                return False
        
        # Try to create connections in batches
        batch_size = 100
        max_successful = 0
        
        for batch_start in range(0, max_connections, batch_size):
            batch_end = min(batch_start + batch_size, max_connections)
            
            tasks = [
                create_connection(i)
                for i in range(batch_start, batch_end)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            current_successful = sum(1 for r in results if r is True)
            max_successful = max(max_successful, successful_connections)
            
            print(f"  Connections: {successful_connections} successful, {failed_connections} failed")
            
            # Stop if we're hitting limits
            if current_successful < batch_size * 0.5:
                print("  Reached connection limit")
                break
        
        return {
            "max_successful_connections": max_successful,
            "total_attempted": successful_connections + failed_connections,
            "success_rate": (successful_connections / (successful_connections + failed_connections)) * 100
        }


@pytest.mark.stress
class TestStressScenarios:
    """Stress testing scenarios."""
    
    @pytest.mark.asyncio
    async def test_memory_stress(self):
        """Test memory usage under stress."""
        async with LoadTestClient() as client:
            results = await StressTestScenarios.memory_stress_test(
                client,
                session_count=500
            )
            
            print("\nMemory Stress Test Results:")
            print(f"  Sessions created: {results['sessions_created']}")
            print(f"  Memory increase: {results['memory_increase_mb']:.2f} MB")
            print(f"  Memory per session: {results['memory_per_session_kb']:.2f} KB")
            
            # Memory usage should be reasonable
            assert results['memory_per_session_kb'] < 1000  # Less than 1MB per session
    
    @pytest.mark.asyncio
    async def test_cpu_stress(self):
        """Test CPU usage under stress."""
        async with LoadTestClient() as client:
            with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
                # Simulate CPU-intensive operation
                async def cpu_intensive(*args, **kwargs):
                    await asyncio.sleep(0.1)  # Simulate processing
                    return {"result": "computed"}
                
                mock_send.side_effect = cpu_intensive
                
                results = await StressTestScenarios.cpu_stress_test(
                    client,
                    duration_seconds=10,
                    complexity_level=3
                )
                
                print("\nCPU Stress Test Results:")
                print(f"  Requests sent: {results['requests_sent']}")
                print(f"  Average CPU: {results['avg_cpu_percent']:.1f}%")
                print(f"  Peak CPU: {results['max_cpu_percent']:.1f}%")
    
    @pytest.mark.asyncio
    async def test_connection_limits(self):
        """Test connection limit handling."""
        results = await StressTestScenarios.connection_limit_test(
            max_connections=1000
        )
        
        print("\nConnection Limit Test Results:")
        print(f"  Max successful connections: {results['max_successful_connections']}")
        print(f"  Success rate: {results['success_rate']:.1f}%")
        
        # Should handle a reasonable number of connections
        assert results['max_successful_connections'] > 100


if __name__ == "__main__":
    """Run load tests directly."""
    import sys
    
    async def main():
        """Main function for running load tests."""
        print("MCP Bridge Load Testing Suite")
        print("==============================\n")
        
        # Parse command line arguments
        test_type = sys.argv[1] if len(sys.argv) > 1 else "sustained"
        
        async with LoadTestClient() as client:
            if test_type == "sustained":
                metrics = await LoadTestScenarios.sustained_load(
                    client,
                    duration_seconds=60,
                    requests_per_second=20
                )
            elif test_type == "burst":
                metrics = await LoadTestScenarios.burst_load(
                    client,
                    burst_size=100,
                    burst_count=5
                )
            elif test_type == "ramp":
                metrics = await LoadTestScenarios.ramp_up_load(
                    client,
                    initial_rate=5,
                    max_rate=100,
                    ramp_duration=60
                )
            elif test_type == "mixed":
                metrics = await LoadTestScenarios.mixed_workload(
                    client,
                    duration_seconds=120,
                    concurrent_sessions=20
                )
            elif test_type == "websocket":
                metrics = await LoadTestScenarios.websocket_load(
                    concurrent_connections=50,
                    duration_seconds=60
                )
            else:
                print(f"Unknown test type: {test_type}")
                print("Available: sustained, burst, ramp, mixed, websocket")
                return
            
            metrics.print_summary()
    
    asyncio.run(main())