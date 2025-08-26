"""
Performance metrics logging for MDMAI TTRPG Assistant.

This module provides comprehensive performance monitoring, metrics collection,
and performance analysis capabilities.
"""

import asyncio
import functools
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


class MetricType(Enum):
    """Types of performance metrics."""

    DURATION = auto()  # Time duration metrics
    COUNTER = auto()  # Count metrics
    GAUGE = auto()  # Current value metrics
    HISTOGRAM = auto()  # Distribution metrics
    RATE = auto()  # Rate metrics (per second)


@dataclass
class PerformanceMetric:
    """Individual performance metric."""

    name: str
    type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "type": self.type.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata,
        }


class PerformanceTimer:
    """High-precision performance timer."""

    def __init__(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Initialize performance timer.

        Args:
            name: Timer name
            tags: Optional tags for categorization
        """
        self.name = name
        self.tags = tags or {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.checkpoints: List[tuple[str, float]] = []

    def start(self) -> "PerformanceTimer":
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self

    def checkpoint(self, name: str) -> float:
        """
        Add a checkpoint.

        Args:
            name: Checkpoint name

        Returns:
            Elapsed time since start
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        elapsed = time.perf_counter() - self.start_time
        self.checkpoints.append((name, elapsed))
        return elapsed

    def stop(self) -> float:
        """
        Stop the timer.

        Returns:
            Total elapsed time
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        self.end_time = time.perf_counter()
        return self.elapsed

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time or time.perf_counter()
        return end - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000

    def get_metric(self) -> PerformanceMetric:
        """Get performance metric."""
        return PerformanceMetric(
            name=self.name,
            type=MetricType.DURATION,
            value=self.elapsed_ms,
            tags=self.tags,
            metadata={
                "checkpoints": [
                    {"name": name, "elapsed_ms": elapsed * 1000}
                    for name, elapsed in self.checkpoints
                ],
            },
        )


class MetricsCollector:
    """Collects and aggregates performance metrics."""

    def __init__(self, window_size: int = 1000) -> None:
        """
        Initialize metrics collector.

        Args:
            window_size: Size of metrics window for aggregation
        """
        self.window_size = window_size
        self.metrics: Dict[str, List[PerformanceMetric]] = {}
        self._lock = asyncio.Lock()

    async def record(self, metric: PerformanceMetric) -> None:
        """Record a performance metric."""
        async with self._lock:
            if metric.name not in self.metrics:
                self.metrics[metric.name] = []
            
            self.metrics[metric.name].append(metric)
            
            # Maintain window size
            if len(self.metrics[metric.name]) > self.window_size:
                self.metrics[metric.name] = self.metrics[metric.name][-self.window_size:]

    async def record_duration(
        self,
        name: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a duration metric."""
        metric = PerformanceMetric(
            name=name,
            type=MetricType.DURATION,
            value=duration_ms,
            tags=tags or {},
        )
        await self.record(metric)

    async def increment_counter(
        self,
        name: str,
        value: float = 1,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        metric = PerformanceMetric(
            name=name,
            type=MetricType.COUNTER,
            value=value,
            tags=tags or {},
        )
        await self.record(metric)

    async def set_gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric."""
        metric = PerformanceMetric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            tags=tags or {},
        )
        await self.record(metric)

    async def get_statistics(self, name: str) -> Dict[str, Any]:
        """
        Get statistics for a metric.

        Args:
            name: Metric name

        Returns:
            Statistical summary
        """
        async with self._lock:
            if name not in self.metrics or not self.metrics[name]:
                return {"error": "No metrics found"}
            
            values = [m.value for m in self.metrics[name]]
            sorted_values = sorted(values)
            
            return {
                "name": name,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "median": sorted_values[len(sorted_values) // 2],
                "p95": sorted_values[int(len(sorted_values) * 0.95)],
                "p99": sorted_values[int(len(sorted_values) * 0.99)],
                "latest": values[-1],
            }

    async def get_all_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all collected metrics."""
        async with self._lock:
            return {
                name: [m.to_dict() for m in metrics]
                for name, metrics in self.metrics.items()
            }

    async def clear_metrics(self, name: Optional[str] = None) -> None:
        """Clear metrics."""
        async with self._lock:
            if name:
                self.metrics.pop(name, None)
            else:
                self.metrics.clear()


class PerformanceMonitor:
    """Monitor and track performance across the application."""

    def __init__(self) -> None:
        """Initialize performance monitor."""
        self.collector = MetricsCollector()
        self.thresholds: Dict[str, float] = {}
        self.alerts: List[Dict[str, Any]] = []
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    def set_threshold(self, metric_name: str, threshold_ms: float) -> None:
        """
        Set performance threshold for metric.

        Args:
            metric_name: Metric name
            threshold_ms: Threshold in milliseconds
        """
        self.thresholds[metric_name] = threshold_ms

    async def check_threshold(self, metric: PerformanceMetric) -> bool:
        """
        Check if metric exceeds threshold.

        Args:
            metric: Performance metric

        Returns:
            True if threshold exceeded
        """
        threshold = self.thresholds.get(metric.name)
        if threshold and metric.type == MetricType.DURATION:
            if metric.value > threshold:
                alert = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "metric": metric.name,
                    "value": metric.value,
                    "threshold": threshold,
                    "exceeded_by": metric.value - threshold,
                    "tags": metric.tags,
                }
                self.alerts.append(alert)
                return True
        return False

    async def record_operation(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record operation performance.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            success: Whether operation succeeded
            tags: Optional tags
        """
        tags = tags or {}
        tags["success"] = str(success)
        
        metric = PerformanceMetric(
            name=f"operation.{operation}",
            type=MetricType.DURATION,
            value=duration_ms,
            tags=tags,
        )
        
        await self.collector.record(metric)
        await self.check_threshold(metric)

    async def get_slow_operations(
        self,
        threshold_ms: float = 1000,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get slowest operations.

        Args:
            threshold_ms: Minimum duration to consider slow
            limit: Maximum number of results

        Returns:
            List of slow operations
        """
        all_metrics = await self.collector.get_all_metrics()
        slow_ops = []
        
        for name, metrics in all_metrics.items():
            if name.startswith("operation."):
                for metric in metrics:
                    if metric["value"] > threshold_ms:
                        slow_ops.append({
                            "operation": name.replace("operation.", ""),
                            "duration_ms": metric["value"],
                            "timestamp": metric["timestamp"],
                            "tags": metric.get("tags", {}),
                        })
        
        # Sort by duration and return top N
        slow_ops.sort(key=lambda x: x["duration_ms"], reverse=True)
        return slow_ops[:limit]

    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        all_metrics = await self.collector.get_all_metrics()
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics_summary": {},
            "alerts": self.alerts[-100:],  # Last 100 alerts
            "slow_operations": await self.get_slow_operations(),
        }
        
        # Generate summary for each metric
        for name in all_metrics:
            stats = await self.collector.get_statistics(name)
            report["metrics_summary"][name] = stats
        
        return report


def measure_performance(
    name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    threshold_ms: Optional[float] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to measure function performance.

    Args:
        name: Metric name (defaults to function name)
        tags: Optional tags
        threshold_ms: Optional performance threshold

    Returns:
        Decorated function
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        metric_name = name or f"function.{func.__name__}"
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            timer = PerformanceTimer(metric_name, tags)
            timer.start()
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception:
                success = False
                raise
            finally:
                duration = timer.stop()
                
                # Record metric (sync context, so we create a task)
                metric = PerformanceMetric(
                    name=metric_name,
                    type=MetricType.DURATION,
                    value=timer.elapsed_ms,
                    tags={**(tags or {}), "success": str(success)},
                )
                
                # Log if threshold exceeded
                if threshold_ms and timer.elapsed_ms > threshold_ms:
                    import logging
                    logging.warning(
                        f"Performance threshold exceeded for {metric_name}: "
                        f"{timer.elapsed_ms:.2f}ms > {threshold_ms}ms"
                    )

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            timer = PerformanceTimer(metric_name, tags)
            timer.start()
            
            try:
                result = await func(*args, **kwargs)  # type: ignore
                success = True
                return result
            except Exception:
                success = False
                raise
            finally:
                duration = timer.stop()
                
                # Record metric
                metric = PerformanceMetric(
                    name=metric_name,
                    type=MetricType.DURATION,
                    value=timer.elapsed_ms,
                    tags={**(tags or {}), "success": str(success)},
                )
                
                # Record in global monitor if available
                if hasattr(performance_monitor, "collector"):
                    await performance_monitor.collector.record(metric)
                
                # Log if threshold exceeded
                if threshold_ms and timer.elapsed_ms > threshold_ms:
                    import logging
                    logging.warning(
                        f"Performance threshold exceeded for {metric_name}: "
                        f"{timer.elapsed_ms:.2f}ms > {threshold_ms}ms"
                    )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator


@contextmanager
def performance_context(
    name: str,
    tags: Optional[Dict[str, str]] = None,
) -> PerformanceTimer:
    """
    Context manager for performance measurement.

    Args:
        name: Operation name
        tags: Optional tags

    Yields:
        Performance timer
    """
    timer = PerformanceTimer(name, tags)
    timer.start()
    
    try:
        yield timer
    finally:
        timer.stop()
        
        # Log performance
        import logging
        logging.info(
            f"Performance: {name} completed in {timer.elapsed_ms:.2f}ms",
            extra={
                "performance_data": {
                    "duration_ms": timer.elapsed_ms,
                    "checkpoints": timer.checkpoints,
                    "tags": tags,
                }
            },
        )


@asynccontextmanager
async def async_performance_context(
    name: str,
    tags: Optional[Dict[str, str]] = None,
) -> PerformanceTimer:
    """
    Async context manager for performance measurement.

    Args:
        name: Operation name
        tags: Optional tags

    Yields:
        Performance timer
    """
    timer = PerformanceTimer(name, tags)
    timer.start()
    
    try:
        yield timer
    finally:
        timer.stop()
        
        # Record in global monitor
        metric = PerformanceMetric(
            name=name,
            type=MetricType.DURATION,
            value=timer.elapsed_ms,
            tags=tags or {},
            metadata={"checkpoints": timer.checkpoints},
        )
        
        if hasattr(performance_monitor, "collector"):
            await performance_monitor.collector.record(metric)
        
        # Log performance
        import logging
        logging.info(
            f"Performance: {name} completed in {timer.elapsed_ms:.2f}ms",
            extra={
                "performance_data": {
                    "duration_ms": timer.elapsed_ms,
                    "checkpoints": timer.checkpoints,
                    "tags": tags,
                }
            },
        )


class ResourceMonitor:
    """Monitor system resource usage."""

    def __init__(self) -> None:
        """Initialize resource monitor."""
        self.samples: List[Dict[str, Any]] = []
        self.max_samples = 1000
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def start_monitoring(self, interval: float = 60.0) -> None:
        """
        Start resource monitoring.

        Args:
            interval: Sampling interval in seconds
        """
        if not self._monitoring:
            self._monitoring = True
            self._monitor_task = asyncio.create_task(
                self._monitor_loop(interval)
            )

    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_task:
            await self._monitor_task

    async def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        import psutil
        
        while self._monitoring:
            try:
                # Collect resource metrics
                sample = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory": {
                        "percent": psutil.virtual_memory().percent,
                        "used_mb": psutil.virtual_memory().used / (1024 * 1024),
                        "available_mb": psutil.virtual_memory().available / (1024 * 1024),
                    },
                    "disk": {
                        "percent": psutil.disk_usage("/").percent,
                        "free_gb": psutil.disk_usage("/").free / (1024 * 1024 * 1024),
                    },
                }
                
                # Add network I/O if available
                try:
                    net_io = psutil.net_io_counters()
                    sample["network"] = {
                        "bytes_sent": net_io.bytes_sent,
                        "bytes_recv": net_io.bytes_recv,
                        "packets_sent": net_io.packets_sent,
                        "packets_recv": net_io.packets_recv,
                    }
                except Exception:
                    pass
                
                self.samples.append(sample)
                
                # Maintain window size
                if len(self.samples) > self.max_samples:
                    self.samples = self.samples[-self.max_samples:]
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                import logging
                logging.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(interval)

    def get_current_resources(self) -> Optional[Dict[str, Any]]:
        """Get current resource usage."""
        if self.samples:
            return self.samples[-1]
        return None

    def get_resource_history(
        self,
        minutes: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Get resource history.

        Args:
            minutes: Number of minutes of history

        Returns:
            List of resource samples
        """
        if not self.samples:
            return []
        
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [
            s for s in self.samples
            if datetime.fromisoformat(s["timestamp"]) > cutoff
        ]


# Global instances
performance_monitor = PerformanceMonitor()
resource_monitor = ResourceMonitor()