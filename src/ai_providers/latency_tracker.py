"""Real-time latency tracking and performance metrics for AI providers."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple

from structlog import get_logger

from .models import ProviderType

logger = get_logger(__name__)


@dataclass
class LatencyMetric:
    """Single latency measurement."""
    
    provider_type: ProviderType
    model: str
    latency_ms: float
    timestamp: datetime
    success: bool
    token_count: int = 0
    request_type: str = "generate"  # generate, stream, health_check


@dataclass 
class LatencyStats:
    """Aggregated latency statistics."""
    
    provider_type: ProviderType
    model: Optional[str] = None
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    success_rate: float = 1.0
    sample_count: int = 0
    time_window: Optional[timedelta] = None
    last_updated: datetime = field(default_factory=datetime.now)


class LatencyTracker:
    """Track and analyze real-time latency metrics for provider selection.
    
    Maintains rolling window of latency measurements and provides
    statistical analysis for intelligent provider selection.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        time_window: timedelta = timedelta(minutes=15),
    ):
        """Initialize latency tracker.
        
        Args:
            window_size: Maximum number of measurements to keep per provider
            time_window: Time window for recent measurements
        """
        self.window_size = window_size
        self.time_window = time_window
        
        # Store measurements per provider and model
        self._measurements: Dict[ProviderType, Dict[str, Deque[LatencyMetric]]] = {}
        
        # Cache computed statistics
        self._stats_cache: Dict[Tuple[ProviderType, Optional[str]], LatencyStats] = {}
        self._cache_ttl = timedelta(seconds=10)
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start background cleanup task."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Started latency tracker cleanup task")
    
    async def stop(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Stopped latency tracker cleanup task")
    
    async def record_latency(
        self,
        provider_type: ProviderType,
        model: str,
        latency_ms: float,
        success: bool = True,
        token_count: int = 0,
        request_type: str = "generate",
    ) -> None:
        """Record a latency measurement.
        
        Args:
            provider_type: Provider that handled the request
            model: Model used
            latency_ms: Request latency in milliseconds
            success: Whether request succeeded
            token_count: Number of tokens processed
            request_type: Type of request
        """
        metric = LatencyMetric(
            provider_type=provider_type,
            model=model,
            latency_ms=latency_ms,
            timestamp=datetime.now(),
            success=success,
            token_count=token_count,
            request_type=request_type,
        )
        
        # Initialize storage if needed
        if provider_type not in self._measurements:
            self._measurements[provider_type] = {}
        
        if model not in self._measurements[provider_type]:
            self._measurements[provider_type][model] = deque(maxlen=self.window_size)
        
        # Add measurement
        self._measurements[provider_type][model].append(metric)
        
        # Invalidate cache for this provider/model
        cache_keys_to_remove = [
            key for key in self._stats_cache
            if key[0] == provider_type and (key[1] is None or key[1] == model)
        ]
        for key in cache_keys_to_remove:
            del self._stats_cache[key]
        
        logger.debug(
            "Recorded latency metric",
            provider=provider_type.value,
            model=model,
            latency_ms=latency_ms,
            success=success,
        )
    
    def get_stats(
        self,
        provider_type: ProviderType,
        model: Optional[str] = None,
        time_window: Optional[timedelta] = None,
    ) -> LatencyStats:
        """Get latency statistics for a provider/model.
        
        Args:
            provider_type: Provider to get stats for
            model: Optional specific model
            time_window: Optional custom time window
            
        Returns:
            Latency statistics
        """
        cache_key = (provider_type, model)
        
        # Check cache
        if cache_key in self._stats_cache:
            cached_stats = self._stats_cache[cache_key]
            if datetime.now() - cached_stats.last_updated < self._cache_ttl:
                return cached_stats
        
        # Compute fresh statistics
        stats = self._compute_stats(provider_type, model, time_window)
        
        # Cache results
        self._stats_cache[cache_key] = stats
        
        return stats
    
    def get_fastest_provider(
        self,
        providers: List[ProviderType],
        model: Optional[str] = None,
        min_success_rate: float = 0.95,
    ) -> Optional[ProviderType]:
        """Get the fastest provider from a list based on recent metrics.
        
        Args:
            providers: List of providers to compare
            model: Optional specific model
            min_success_rate: Minimum required success rate
            
        Returns:
            Fastest provider or None if no suitable provider
        """
        best_provider = None
        best_latency = float('inf')
        
        for provider in providers:
            stats = self.get_stats(provider, model)
            
            # Skip if not enough data or poor success rate
            if stats.sample_count < 5 or stats.success_rate < min_success_rate:
                continue
            
            # Use p50 (median) for comparison to avoid outliers
            if stats.p50_latency_ms < best_latency:
                best_latency = stats.p50_latency_ms
                best_provider = provider
        
        if best_provider:
            logger.debug(
                "Selected fastest provider",
                provider=best_provider.value,
                latency_ms=best_latency,
            )
        
        return best_provider
    
    def get_provider_ranking(
        self,
        providers: List[ProviderType],
        model: Optional[str] = None,
    ) -> List[Tuple[ProviderType, float]]:
        """Rank providers by performance.
        
        Args:
            providers: List of providers to rank
            model: Optional specific model
            
        Returns:
            List of (provider, score) tuples sorted by performance
        """
        rankings = []
        
        for provider in providers:
            stats = self.get_stats(provider, model)
            
            if stats.sample_count == 0:
                # No data, assign worst score
                score = float('inf')
            else:
                # Score based on latency and success rate
                # Lower is better
                latency_score = stats.p50_latency_ms
                reliability_penalty = (1 - stats.success_rate) * 10000  # Heavy penalty for failures
                score = latency_score + reliability_penalty
            
            rankings.append((provider, score))
        
        # Sort by score (lower is better)
        rankings.sort(key=lambda x: x[1])
        
        return rankings
    
    def get_adaptive_timeout(
        self,
        provider_type: ProviderType,
        model: Optional[str] = None,
        percentile: int = 99,
    ) -> float:
        """Get adaptive timeout based on historical latencies.
        
        Args:
            provider_type: Provider to get timeout for
            model: Optional specific model
            percentile: Percentile to use (e.g., 99 for p99)
            
        Returns:
            Suggested timeout in seconds
        """
        stats = self.get_stats(provider_type, model)
        
        if stats.sample_count == 0:
            # No data, use conservative default
            return 30.0
        
        # Use specified percentile
        if percentile >= 99:
            timeout_ms = stats.p99_latency_ms
        elif percentile >= 95:
            timeout_ms = stats.p95_latency_ms
        else:
            timeout_ms = stats.p50_latency_ms
        
        # Add buffer and convert to seconds
        timeout_s = (timeout_ms * 1.2) / 1000.0
        
        # Clamp to reasonable range
        return max(5.0, min(60.0, timeout_s))
    
    def _compute_stats(
        self,
        provider_type: ProviderType,
        model: Optional[str] = None,
        time_window: Optional[timedelta] = None,
    ) -> LatencyStats:
        """Compute statistics from measurements.
        
        Args:
            provider_type: Provider to compute stats for
            model: Optional specific model
            time_window: Optional custom time window
            
        Returns:
            Computed statistics
        """
        stats = LatencyStats(provider_type=provider_type, model=model)
        
        # Collect relevant measurements
        measurements = []
        window = time_window or self.time_window
        cutoff = datetime.now() - window
        
        if provider_type in self._measurements:
            if model:
                # Specific model
                if model in self._measurements[provider_type]:
                    measurements = [
                        m for m in self._measurements[provider_type][model]
                        if m.timestamp >= cutoff
                    ]
            else:
                # All models for provider
                for model_measurements in self._measurements[provider_type].values():
                    measurements.extend([
                        m for m in model_measurements
                        if m.timestamp >= cutoff
                    ])
        
        if not measurements:
            return stats
        
        # Calculate statistics
        latencies = [m.latency_ms for m in measurements]
        successful = [m for m in measurements if m.success]
        
        stats.sample_count = len(measurements)
        stats.success_rate = len(successful) / len(measurements) if measurements else 0
        
        if latencies:
            latencies.sort()
            stats.min_latency_ms = latencies[0]
            stats.max_latency_ms = latencies[-1]
            stats.avg_latency_ms = sum(latencies) / len(latencies)
            
            # Calculate percentiles
            stats.p50_latency_ms = self._percentile(latencies, 50)
            stats.p95_latency_ms = self._percentile(latencies, 95)
            stats.p99_latency_ms = self._percentile(latencies, 99)
        
        stats.time_window = window
        stats.last_updated = datetime.now()
        
        return stats
    
    def _percentile(self, sorted_list: List[float], percentile: int) -> float:
        """Calculate percentile from sorted list.
        
        Args:
            sorted_list: Sorted list of values
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value
        """
        if not sorted_list:
            return 0.0
        
        index = (len(sorted_list) - 1) * (percentile / 100)
        lower = int(index)
        upper = lower + 1
        
        if upper >= len(sorted_list):
            return sorted_list[lower]
        
        weight = index - lower
        return sorted_list[lower] * (1 - weight) + sorted_list[upper] * weight
    
    async def _cleanup_loop(self) -> None:
        """Periodically clean up old measurements."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                self._cleanup_old_measurements()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in latency tracker cleanup", error=str(e))
    
    def _cleanup_old_measurements(self) -> None:
        """Remove measurements older than time window."""
        cutoff = datetime.now() - self.time_window
        cleaned = 0
        
        for provider_measurements in self._measurements.values():
            for model, measurements in provider_measurements.items():
                # Remove old measurements from left side of deque
                while measurements and measurements[0].timestamp < cutoff:
                    measurements.popleft()
                    cleaned += 1
        
        if cleaned > 0:
            logger.debug(f"Cleaned {cleaned} old latency measurements")
            # Clear stats cache after cleanup
            self._stats_cache.clear()
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for analysis or persistence.
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        for provider_type, models in self._measurements.items():
            provider_metrics = {}
            
            for model, measurements in models.items():
                recent_measurements = [
                    m for m in measurements
                    if datetime.now() - m.timestamp < self.time_window
                ]
                
                if recent_measurements:
                    stats = self.get_stats(provider_type, model)
                    provider_metrics[model] = {
                        "stats": {
                            "avg_latency_ms": stats.avg_latency_ms,
                            "min_latency_ms": stats.min_latency_ms,
                            "max_latency_ms": stats.max_latency_ms,
                            "p50_latency_ms": stats.p50_latency_ms,
                            "p95_latency_ms": stats.p95_latency_ms,
                            "p99_latency_ms": stats.p99_latency_ms,
                            "success_rate": stats.success_rate,
                            "sample_count": stats.sample_count,
                        },
                        "recent_samples": [
                            {
                                "latency_ms": m.latency_ms,
                                "timestamp": m.timestamp.isoformat(),
                                "success": m.success,
                                "token_count": m.token_count,
                                "request_type": m.request_type,
                            }
                            for m in recent_measurements[-10:]  # Last 10 samples
                        ]
                    }
            
            if provider_metrics:
                metrics[provider_type.value] = provider_metrics
        
        return metrics


class RequestTimer:
    """Context manager for timing requests."""
    
    def __init__(
        self,
        tracker: LatencyTracker,
        provider_type: ProviderType,
        model: str,
        token_count: int = 0,
        request_type: str = "generate",
    ):
        """Initialize request timer.
        
        Args:
            tracker: Latency tracker instance
            provider_type: Provider handling the request
            model: Model being used
            token_count: Number of tokens
            request_type: Type of request
        """
        self.tracker = tracker
        self.provider_type = provider_type
        self.model = model
        self.token_count = token_count
        self.request_type = request_type
        self.start_time = None
        self.success = True
    
    async def __aenter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record metric."""
        if self.start_time:
            latency_ms = (time.perf_counter() - self.start_time) * 1000
            self.success = exc_type is None
            
            await self.tracker.record_latency(
                provider_type=self.provider_type,
                model=self.model,
                latency_ms=latency_ms,
                success=self.success,
                token_count=self.token_count,
                request_type=self.request_type,
            )