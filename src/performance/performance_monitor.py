"""Performance monitoring system for TTRPG Assistant."""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import psutil

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric."""

    timestamp: datetime
    metric_type: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type,
            "value": self.value,
            "metadata": self.metadata,
        }


@dataclass
class SystemMetrics:
    """System-level performance metrics."""

    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_mb": self.memory_mb,
            "disk_io_read_mb": self.disk_io_read_mb,
            "disk_io_write_mb": self.disk_io_write_mb,
            "timestamp": self.timestamp.isoformat(),
        }


class PerformanceMonitor:
    """Monitors system and application performance."""

    def __init__(self, metrics_dir: Optional[Path] = None):
        """Initialize performance monitor."""
        self.metrics_dir = metrics_dir or Path("data/metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Metric storage (keep last hour of data in memory)
        self.metrics: Deque[PerformanceMetric] = deque(maxlen=3600)
        self.system_metrics: Deque[SystemMetrics] = deque(maxlen=60)

        # Operation timers
        self.operation_timers: Dict[str, List[float]] = {}

        # Thresholds for alerting
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "query_time": 2.0,
            "batch_time": 10.0,
        }

        # Monitoring state
        self._monitoring = False
        self._monitor_task = None

        # Process handle for monitoring
        self.process = psutil.Process()
        self._last_disk_io = None

    async def start_monitoring(self, interval: int = 60) -> None:
        """
        Start continuous performance monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            logger.warning("Monitoring already started")
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info(f"Performance monitoring started (interval: {interval}s)")

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Performance monitoring stopped")

    async def _monitor_loop(self, interval: int) -> None:
        """Main monitoring loop with adaptive throttling."""
        consecutive_errors = 0
        max_consecutive_errors = 3
        min_interval = max(1, interval)  # Ensure minimum 1 second interval

        while self._monitoring:
            try:
                # Adaptive interval based on system load
                current_interval = min_interval
                if hasattr(self, "_last_cpu_percent") and self._last_cpu_percent > 80:
                    # Double interval if CPU usage is high
                    current_interval = min_interval * 2

                # Collect system metrics
                metrics = await self.collect_system_metrics()
                self.system_metrics.append(metrics)
                self._last_cpu_percent = metrics.cpu_percent

                # Check thresholds
                self._check_thresholds(metrics)

                # Save metrics periodically
                if len(self.system_metrics) >= 10:
                    await self.save_metrics()

                # Reset error counter on success
                consecutive_errors = 0

                await asyncio.sleep(current_interval)

            except asyncio.CancelledError:
                # Expected when monitoring is stopped
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    f"Error in monitoring loop: {str(e)} (attempt {consecutive_errors}/{max_consecutive_errors})"
                )

                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Too many consecutive monitoring errors, stopping monitor")
                    self._monitoring = False
                    break

                if self._monitoring:  # Only sleep if still monitoring
                    # Exponential backoff on errors
                    await asyncio.sleep(min_interval * (2**consecutive_errors))

    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU usage
        cpu_percent = self.process.cpu_percent()

        # Memory usage
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        memory_percent = self.process.memory_percent()

        # Disk I/O
        try:
            io_counters = self.process.io_counters()
            if self._last_disk_io:
                read_mb = (io_counters.read_bytes - self._last_disk_io.read_bytes) / (1024 * 1024)
                write_mb = (io_counters.write_bytes - self._last_disk_io.write_bytes) / (
                    1024 * 1024
                )
            else:
                read_mb = 0
                write_mb = 0
            self._last_disk_io = io_counters
        except (AttributeError, psutil.AccessDenied):
            # Disk I/O not available on all platforms
            read_mb = 0
            write_mb = 0

        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            disk_io_read_mb=read_mb,
            disk_io_write_mb=write_mb,
        )

    def _check_thresholds(self, metrics: SystemMetrics) -> None:
        """Check if metrics exceed thresholds."""
        if metrics.cpu_percent > self.thresholds["cpu_percent"]:
            logger.warning(
                f"High CPU usage: {metrics.cpu_percent:.1f}%",
                threshold=self.thresholds["cpu_percent"],
            )

        if metrics.memory_percent > self.thresholds["memory_percent"]:
            logger.warning(
                f"High memory usage: {metrics.memory_percent:.1f}%",
                memory_mb=metrics.memory_mb,
                threshold=self.thresholds["memory_percent"],
            )

    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an operation's performance.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
            success: Whether the operation succeeded
            metadata: Additional metadata
        """
        # Store in operation timers
        if operation not in self.operation_timers:
            self.operation_timers[operation] = []
        self.operation_timers[operation].append(duration)

        # Keep only last 100 measurements per operation
        if len(self.operation_timers[operation]) > 100:
            self.operation_timers[operation] = self.operation_timers[operation][-100:]

        # Create metric
        metric = PerformanceMetric(
            timestamp=datetime.utcnow(),
            metric_type=f"operation.{operation}",
            value=duration,
            metadata={
                **(metadata or {}),
                "success": success,
            },
        )

        self.metrics.append(metric)

        # Check threshold
        if operation == "query" and duration > self.thresholds["query_time"]:
            logger.warning(
                "Slow query detected",
                operation=operation,
                duration=duration,
                threshold=self.thresholds["query_time"],
            )
        elif operation == "batch" and duration > self.thresholds["batch_time"]:
            logger.warning(
                "Slow batch operation",
                operation=operation,
                duration=duration,
                threshold=self.thresholds["batch_time"],
            )

    def get_operation_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for operations.

        Args:
            operation: Specific operation to get stats for, or None for all

        Returns:
            Operation statistics
        """
        if operation:
            if operation not in self.operation_timers:
                return {"error": f"No data for operation '{operation}'"}

            timings = self.operation_timers[operation]
            if not timings:
                return {"error": "No measurements"}

            return {
                "operation": operation,
                "count": len(timings),
                "min": min(timings),
                "max": max(timings),
                "avg": sum(timings) / len(timings),
                "median": sorted(timings)[len(timings) // 2],
                "p95": (
                    sorted(timings)[int(len(timings) * 0.95)] if len(timings) > 1 else timings[0]
                ),
                "p99": (
                    sorted(timings)[int(len(timings) * 0.99)] if len(timings) > 1 else timings[0]
                ),
            }

        # Get stats for all operations
        stats = {}
        for op_name, timings in self.operation_timers.items():
            if timings:
                stats[op_name] = {
                    "count": len(timings),
                    "min": min(timings),
                    "max": max(timings),
                    "avg": sum(timings) / len(timings),
                }

        return stats

    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        if not self.system_metrics:
            return {"error": "No system metrics available"}

        recent_metrics = list(self.system_metrics)[-10:]  # Last 10 measurements

        return {
            "current": self.system_metrics[-1].to_dict() if self.system_metrics else None,
            "averages": {
                "cpu_percent": sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                "memory_percent": sum(m.memory_percent for m in recent_metrics)
                / len(recent_metrics),
                "memory_mb": sum(m.memory_mb for m in recent_metrics) / len(recent_metrics),
            },
            "peaks": {
                "cpu_percent": max(m.cpu_percent for m in self.system_metrics),
                "memory_mb": max(m.memory_mb for m in self.system_metrics),
            },
        }

    async def save_metrics(self) -> None:
        """Save metrics to disk."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # Ensure metrics directory exists
            self.metrics_dir.mkdir(parents=True, exist_ok=True)

            # Save system metrics with proper encoding
            if self.system_metrics:
                system_file = self.metrics_dir / f"system_{timestamp}.json"
                try:
                    with open(system_file, "w", encoding="utf-8") as f:
                        json.dump(
                            [m.to_dict() for m in self.system_metrics],
                            f,
                            indent=2,
                            default=str,  # Handle non-serializable objects
                        )
                except IOError as e:
                    logger.error(f"Failed to write system metrics: {e}")

            # Save operation metrics with proper encoding
            if self.metrics:
                ops_file = self.metrics_dir / f"operations_{timestamp}.json"
                try:
                    with open(ops_file, "w", encoding="utf-8") as f:
                        json.dump(
                            [m.to_dict() for m in self.metrics],
                            f,
                            indent=2,
                            default=str,  # Handle non-serializable objects
                        )
                except IOError as e:
                    logger.error(f"Failed to write operation metrics: {e}")

            # Clean up old metrics files (keep only last 7 days)
            await self._cleanup_old_metrics()

            logger.debug(f"Metrics saved to {self.metrics_dir}")

        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")

    async def load_metrics(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Load metrics from disk.

        Args:
            date: Date to load metrics for, or None for today

        Returns:
            Loaded metrics
        """
        if not date:
            date = datetime.utcnow()

        date_str = date.strftime("%Y%m%d")
        metrics_data = {"system": [], "operations": []}

        # Load all metrics files for the date
        for file in self.metrics_dir.glob(f"*_{date_str}_*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "system" in file.name:
                    metrics_data["system"].extend(data)
                elif "operations" in file.name:
                    metrics_data["operations"].extend(data)

            except Exception as e:
                logger.error(f"Failed to load metrics from {file}: {str(e)}")

        return metrics_data

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": self.get_system_stats(),
            "operations": self.get_operation_stats(),
            "alerts": [],
            "recommendations": [],
        }

        # Check for issues and generate recommendations
        if report["system"] and "current" in report["system"]:
            current = report["system"]["current"]

            if current["cpu_percent"] > self.thresholds["cpu_percent"]:
                report["alerts"].append(
                    {
                        "type": "cpu",
                        "message": f"High CPU usage: {current['cpu_percent']:.1f}%",
                        "severity": "warning",
                    }
                )
                report["recommendations"].append(
                    "Consider optimizing CPU-intensive operations or scaling resources"
                )

            if current["memory_percent"] > self.thresholds["memory_percent"]:
                report["alerts"].append(
                    {
                        "type": "memory",
                        "message": f"High memory usage: {current['memory_percent']:.1f}%",
                        "severity": "warning",
                    }
                )
                report["recommendations"].append(
                    "Review memory usage patterns and consider implementing better caching strategies"
                )

        # Check operation performance
        for op_name, stats in report["operations"].items():
            if "query" in op_name and stats["avg"] > self.thresholds["query_time"]:
                report["recommendations"].append(
                    f"Optimize {op_name} operations - average time {stats['avg']:.2f}s exceeds threshold"
                )
            elif "batch" in op_name and stats["avg"] > self.thresholds["batch_time"]:
                report["recommendations"].append(
                    f"Review batch size for {op_name} - average time {stats['avg']:.2f}s is high"
                )

        return report

    async def _cleanup_old_metrics(self) -> None:
        """Clean up metrics files older than 7 days."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=7)

            for metric_file in self.metrics_dir.glob("*.json"):
                try:
                    # Get file modification time
                    file_mtime = datetime.fromtimestamp(metric_file.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        metric_file.unlink()
                        logger.debug(f"Deleted old metric file: {metric_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete old metric file {metric_file.name}: {e}")

        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")

    async def cleanup(self) -> None:
        """Clean up monitoring resources."""
        await self.stop_monitoring()
        await self.save_metrics()
        logger.info("Performance monitor cleaned up")


class OperationTimer:
    """Context manager for timing operations."""

    def __init__(
        self, monitor: PerformanceMonitor, operation: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize timer."""
        self.monitor = monitor
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time = None
        self.success = True

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.success = exc_type is None
            self.monitor.record_operation(
                self.operation,
                duration,
                self.success,
                self.metadata,
            )
        return False  # Don't suppress exceptions

    async def __aenter__(self):
        """Async context enter."""
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context exit."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.success = exc_type is None
            self.monitor.record_operation(
                self.operation,
                duration,
                self.success,
                self.metadata,
            )
        return False
