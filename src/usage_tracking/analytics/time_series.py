"""
Time-series analytics engine for usage tracking data.

This module provides:
- Efficient time-series aggregation and rollups
- Real-time metrics computation
- Trend analysis and forecasting
- Cost optimization insights
- Performance monitoring

TODO: CRITICAL - This module has significant functional overlap with:
- src/cost_optimization/advanced_optimizer.py
- src/ai_providers/enhanced_usage_tracker.py
The cost optimization analysis here duplicates logic from the cost_optimization package.
This should be refactored to create a single source of truth for analytics, trend analysis,
and cost optimization logic to prevent inconsistencies and reduce maintenance burden.
"""

import calendar
import logging
import math
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict
from dataclasses import dataclass
import statistics

from ..storage.models import (
    UsageRecord, TimeAggregation
)

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesPoint:
    """Single point in a time series."""
    timestamp: datetime
    value: Union[int, float, Decimal]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TimeWindow:
    """Time window for aggregation."""
    start: datetime
    end: datetime
    duration_seconds: int

    @property
    def duration_hours(self) -> float:
        return self.duration_seconds / 3600

    @property
    def duration_days(self) -> float:
        return self.duration_seconds / 86400

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within this window."""
        return self.start <= timestamp < self.end

    def overlap_with(self, other: 'TimeWindow') -> Optional['TimeWindow']:
        """Find overlap with another time window."""
        overlap_start = max(self.start, other.start)
        overlap_end = min(self.end, other.end)

        if overlap_start < overlap_end:
            duration = int((overlap_end - overlap_start).total_seconds())
            return TimeWindow(overlap_start, overlap_end, duration)

        return None


class TimeBucket:
    """Container for time-series data within a specific time window."""

    def __init__(self, window: TimeWindow):
        self.window = window
        self.points: List[TimeSeriesPoint] = []
        self.is_finalized = False

        # Cached aggregations
        self._cached_sum: Optional[Decimal] = None
        self._cached_avg: Optional[float] = None
        self._cached_count: Optional[int] = None
        self._cached_min: Optional[Union[int, float, Decimal]] = None
        self._cached_max: Optional[Union[int, float, Decimal]] = None

    def add_point(self, point: TimeSeriesPoint) -> bool:
        """Add a point to this bucket if it fits in the time window."""
        if not self.window.contains(point.timestamp):
            return False

        self.points.append(point)
        self._invalidate_cache()
        return True

    def _invalidate_cache(self) -> None:
        """Invalidate cached aggregations."""
        self._cached_sum = None
        self._cached_avg = None
        self._cached_count = None
        self._cached_min = None
        self._cached_max = None

    def finalize(self) -> None:
        """Mark bucket as finalized (no more data will be added)."""
        self.is_finalized = True
        # Pre-compute common aggregations
        _ = self.sum()
        _ = self.average()
        _ = self.count()
        _ = self.min_value()
        _ = self.max_value()

    def sum(self) -> Decimal:
        """Get sum of all values in bucket."""
        if self._cached_sum is not None:
            return self._cached_sum

        total = Decimal("0")
        for point in self.points:
            if isinstance(point.value, Decimal):
                total += point.value
            else:
                total += Decimal(str(point.value))

        if self.is_finalized:
            self._cached_sum = total
        return total

    def average(self) -> float:
        """Get average of all values in bucket."""
        if self._cached_avg is not None:
            return self._cached_avg

        if not self.points:
            return 0.0

        total = float(self.sum())
        avg = total / len(self.points)

        if self.is_finalized:
            self._cached_avg = avg
        return avg

    def count(self) -> int:
        """Get count of points in bucket."""
        if self._cached_count is not None:
            return self._cached_count

        count = len(self.points)
        if self.is_finalized:
            self._cached_count = count
        return count

    def min_value(self) -> Optional[Union[int, float, Decimal]]:
        """Get minimum value in bucket."""
        if self._cached_min is not None:
            return self._cached_min

        if not self.points:
            return None

        min_val = min(point.value for point in self.points)
        if self.is_finalized:
            self._cached_min = min_val
        return min_val

    def max_value(self) -> Optional[Union[int, float, Decimal]]:
        """Get maximum value in bucket."""
        if self._cached_max is not None:
            return self._cached_max

        if not self.points:
            return None

        max_val = max(point.value for point in self.points)
        if self.is_finalized:
            self._cached_max = max_val
        return max_val

    def percentile(self, percentile: float) -> Optional[float]:
        """Calculate percentile of values in bucket."""
        if not self.points:
            return None

        values = [float(point.value) for point in self.points]
        values.sort()

        if percentile <= 0:
            return values[0]
        if percentile >= 100:
            return values[-1]

        index = (percentile / 100) * (len(values) - 1)
        lower_index = int(math.floor(index))
        upper_index = int(math.ceil(index))

        if lower_index == upper_index:
            return values[lower_index]

        # Linear interpolation
        weight = index - lower_index
        return values[lower_index] * (1 - weight) + values[upper_index] * weight

    def standard_deviation(self) -> float:
        """Calculate standard deviation of values in bucket."""
        if len(self.points) < 2:
            return 0.0

        values = [float(point.value) for point in self.points]
        return statistics.stdev(values)


class TimeSeriesAggregator:
    """Efficient time-series aggregation engine."""

    def __init__(self):
        self.buckets_by_period: Dict[TimeAggregation, List[TimeBucket]] = {}
        self.rollup_cache: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self.aggregation_stats = defaultdict(lambda: {
            "count": 0, "total_time": 0.0, "cache_hits": 0
        })

    def _add_months(self, date: datetime, months: int) -> datetime:
        """Add accurate number of months to a datetime, handling variable month lengths."""
        month = date.month - 1 + months
        year = date.year + month // 12
        month = month % 12 + 1

        # Handle day overflow (e.g., Jan 31 + 1 month = Feb 28/29)
        max_days_in_month = calendar.monthrange(year, month)[1]
        day = min(date.day, max_days_in_month)

        return date.replace(year=year, month=month, day=day)

    def _add_years(self, date: datetime, years: int) -> datetime:
        """Add accurate number of years to a datetime, handling leap years."""
        try:
            return date.replace(year=date.year + years)
        except ValueError:
            # Handle leap year edge case (Feb 29 -> Feb 28)
            return date.replace(year=date.year + years, day=28)

    def _create_time_windows(
        self,
        start_time: datetime,
        end_time: datetime,
        aggregation_type: TimeAggregation
    ) -> List[TimeWindow]:
        """Create time windows for the specified period and aggregation type."""
        windows = []
        current = start_time

        # Align to period boundaries
        current = self._align_to_period(current, aggregation_type)

        while current < end_time:
            if aggregation_type == TimeAggregation.HOURLY:
                window_end = current + timedelta(hours=1)
            elif aggregation_type == TimeAggregation.DAILY:
                window_end = current + timedelta(days=1)
            elif aggregation_type == TimeAggregation.WEEKLY:
                window_end = current + timedelta(weeks=1)
            elif aggregation_type == TimeAggregation.MONTHLY:
                # Use accurate month calculation
                window_end = self._add_months(current, 1)
            elif aggregation_type == TimeAggregation.YEARLY:
                # Use accurate year calculation
                window_end = self._add_years(current, 1)
            else:
                window_end = current + timedelta(hours=1)  # Default

            duration_seconds = int((window_end - current).total_seconds())

            window = TimeWindow(current, window_end, duration_seconds)
            windows.append(window)
            current = window_end

        return windows

    def _align_to_period(self, timestamp: datetime, aggregation_type: TimeAggregation) -> datetime:
        """Align timestamp to period boundary."""
        if aggregation_type == TimeAggregation.HOURLY:
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif aggregation_type == TimeAggregation.DAILY:
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif aggregation_type == TimeAggregation.WEEKLY:
            # Align to Monday
            days_since_monday = timestamp.weekday()
            monday = timestamp - timedelta(days=days_since_monday)
            return monday.replace(hour=0, minute=0, second=0, microsecond=0)
        elif aggregation_type == TimeAggregation.MONTHLY:
            return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif aggregation_type == TimeAggregation.YEARLY:
            return timestamp.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return timestamp

    async def aggregate_usage_records(
        self,
        records: List[UsageRecord],
        aggregation_type: TimeAggregation,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        group_by: Optional[List[str]] = None
    ) -> Dict[str, List[TimeBucket]]:
        """Aggregate usage records into time buckets."""
        import time as time_module
        start_time_perf = time_module.time()

        try:
            # Determine time range
            if not start_time or not end_time:
                if records:
                    timestamps = [r.timestamp for r in records]
                    start_time = start_time or min(timestamps)
                    end_time = end_time or max(timestamps)
                else:
                    start_time = datetime.utcnow() - timedelta(days=7)
                    end_time = datetime.utcnow()

            # Create time windows
            windows = self._create_time_windows(start_time, end_time, aggregation_type)

            # Group records
            groups = self._group_records(records, group_by or [])

            # Create buckets for each group
            grouped_buckets = {}
            for group_key, group_records in groups.items():
                buckets = []
                for window in windows:
                    bucket = TimeBucket(window)
                    buckets.append(bucket)

                # Distribute records into buckets
                await self._distribute_records_to_buckets(group_records, buckets)

                # Finalize buckets
                for bucket in buckets:
                    bucket.finalize()

                grouped_buckets[group_key] = buckets

            # Track performance
            execution_time = time_module.time() - start_time_perf
            stats = self.aggregation_stats[f"aggregate_{aggregation_type.value}"]
            stats["count"] += 1
            stats["total_time"] += execution_time

            return grouped_buckets

        except Exception as e:
            logger.error(f"Failed to aggregate usage records: {e}")
            raise

    def _group_records(
        self,
        records: List[UsageRecord],
        group_by: List[str]
    ) -> Dict[str, List[UsageRecord]]:
        """Group records by specified fields."""
        if not group_by:
            return {"all": records}

        groups = defaultdict(list)

        for record in records:
            group_key_parts = []
            for field in group_by:
                if hasattr(record, field):
                    value = getattr(record, field)
                    if hasattr(value, 'value'):  # Enum
                        value = value.value
                    group_key_parts.append(str(value))
                else:
                    group_key_parts.append("unknown")

            group_key = "|".join(group_key_parts)
            groups[group_key].append(record)

        return dict(groups)

    def _create_time_series_point(
        self,
        record: UsageRecord,
        metric: str,
        value: float
    ) -> TimeSeriesPoint:
        """Helper method to create time series points with common parameters."""
        return TimeSeriesPoint(
            timestamp=record.timestamp,
            value=value,
            metadata={"metric": metric, "record_id": record.record_id}
        )

    async def _distribute_records_to_buckets(
        self,
        records: List[UsageRecord],
        buckets: List[TimeBucket]
    ) -> None:
        """Distribute records into appropriate time buckets."""
        for record in records:
            # Create time series points for different metrics using helper
            cost_point = self._create_time_series_point(record, "cost", record.cost_usd)
            token_point = self._create_time_series_point(record, "tokens", record.token_count)
            request_point = self._create_time_series_point(record, "requests", 1)

            # Find appropriate bucket
            for bucket in buckets:
                if bucket.window.contains(record.timestamp):
                    bucket.add_point(cost_point)
                    bucket.add_point(token_point)
                    bucket.add_point(request_point)
                    break

    async def compute_rollups(
        self,
        buckets: List[TimeBucket],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Compute rollup statistics from time buckets."""
        if not buckets:
            return {}

        metrics = metrics or ["cost", "tokens", "requests"]
        rollup_data = {}

        # Separate points by metric type
        points_by_metric = defaultdict(list)
        for bucket in buckets:
            for point in bucket.points:
                metric_type = point.metadata.get("metric", "unknown")
                points_by_metric[metric_type].append(point)

        # Compute rollups for each metric
        for metric in metrics:
            if metric not in points_by_metric:
                continue

            points = points_by_metric[metric]
            values = [float(point.value) for point in points]

            if not values:
                continue

            rollup_data[metric] = {
                "total": sum(values),
                "average": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "p50": statistics.median(values),
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99)
            }

        # Compute time-based metrics
        if buckets:
            time_spans = []
            for bucket in buckets:
                if bucket.points:
                    timestamps = [p.timestamp for p in bucket.points]
                    span = max(timestamps) - min(timestamps)
                    time_spans.append(span.total_seconds())

            if time_spans:
                rollup_data["time_metrics"] = {
                    "avg_time_span": statistics.mean(time_spans),
                    "total_time_span": sum(time_spans),
                    "active_buckets": len([b for b in buckets if b.points])
                }

        return rollup_data

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        lower_index = int(math.floor(index))
        upper_index = int(math.ceil(index))

        if lower_index == upper_index:
            return sorted_values[lower_index]

        # Linear interpolation
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight


class TrendAnalyzer:
    """Analyzes trends and patterns in time-series data."""

    def __init__(self):
        self.trend_cache = {}
        self.pattern_cache = {}

    async def detect_trends(
        self,
        buckets: List[TimeBucket],
        metric: str = "cost",
        min_periods: int = 5
    ) -> Dict[str, Any]:
        """Detect trends in time-series data."""
        if len(buckets) < min_periods:
            return {"trend": "insufficient_data", "periods": len(buckets)}

        # Extract metric values over time
        values = []
        timestamps = []

        for bucket in buckets:
            # Get metric-specific values from bucket
            metric_points = [p for p in bucket.points if p.metadata.get("metric") == metric]
            if metric_points:
                bucket_value = sum(float(p.value) for p in metric_points)
                values.append(bucket_value)
                timestamps.append(bucket.window.start)

        if len(values) < min_periods:
            return {"trend": "insufficient_data", "periods": len(values)}

        # Calculate trend using linear regression
        trend_analysis = self._calculate_trend(values, timestamps)

        # Detect patterns
        patterns = await self._detect_patterns(values, timestamps)

        # Calculate volatility
        volatility = self._calculate_volatility(values)

        return {
            "trend": trend_analysis["direction"],
            "slope": trend_analysis["slope"],
            "r_squared": trend_analysis["r_squared"],
            "confidence": trend_analysis["confidence"],
            "patterns": patterns,
            "volatility": volatility,
            "periods_analyzed": len(values),
            "value_range": {"min": min(values), "max": max(values)},
            "latest_value": values[-1] if values else 0,
            "change_from_first": ((values[-1] - values[0]) / values[0] * 100) if values and values[0] != 0 else 0
        }

    def _calculate_trend(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> Dict[str, Any]:
        """Calculate trend using linear regression."""
        n = len(values)
        if n < 2:
            return {"direction": "flat", "slope": 0, "r_squared": 0, "confidence": 0}

        # Convert timestamps to numeric values (hours since first timestamp)
        x_values = [(ts - timestamps[0]).total_seconds() / 3600 for ts in timestamps]

        # Calculate linear regression
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return {"direction": "flat", "slope": 0, "r_squared": 0, "confidence": 0}

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Calculate R-squared
        predicted_values = [slope * x + intercept for x in x_values]
        ss_res = sum((y - pred) ** 2 for y, pred in zip(values, predicted_values))
        ss_tot = sum((y - y_mean) ** 2 for y in values)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Determine trend direction
        if abs(slope) < 0.01:  # Threshold for "flat"
            direction = "flat"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        # Calculate confidence based on R-squared and number of points
        confidence = min(r_squared * (n / 10), 1.0)  # Scale by number of points

        return {
            "direction": direction,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "confidence": confidence
        }

    async def _detect_patterns(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> List[Dict[str, Any]]:
        """Detect patterns like cycles, spikes, etc."""
        patterns = []

        if len(values) < 3:
            return patterns

        # Detect spikes (values significantly above average)
        mean_value = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        spike_threshold = mean_value + 2 * std_dev

        spikes = []
        for i, (value, timestamp) in enumerate(zip(values, timestamps)):
            if value > spike_threshold:
                spikes.append({
                    "index": i,
                    "timestamp": timestamp,
                    "value": value,
                    "deviation": (value - mean_value) / std_dev if std_dev > 0 else 0
                })

        if spikes:
            patterns.append({
                "type": "spikes",
                "count": len(spikes),
                "locations": spikes,
                "severity": max(spike["deviation"] for spike in spikes)
            })

        # Detect valleys (values significantly below average)
        valley_threshold = mean_value - 2 * std_dev
        valleys = []
        for i, (value, timestamp) in enumerate(zip(values, timestamps)):
            if value < valley_threshold:
                valleys.append({
                    "index": i,
                    "timestamp": timestamp,
                    "value": value,
                    "deviation": (mean_value - value) / std_dev if std_dev > 0 else 0
                })

        if valleys:
            patterns.append({
                "type": "valleys",
                "count": len(valleys),
                "locations": valleys,
                "severity": max(valley["deviation"] for valley in valleys)
            })

        # Detect potential cycles (simplified)
        if len(values) >= 7:  # Need at least a week of data
            cycle_pattern = self._detect_weekly_cycle(values, timestamps)
            if cycle_pattern:
                patterns.append(cycle_pattern)

        return patterns

    def _detect_weekly_cycle(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> Optional[Dict[str, Any]]:
        """Detect weekly cyclical patterns."""
        # Group values by day of week
        day_values = defaultdict(list)
        for value, timestamp in zip(values, timestamps):
            day_of_week = timestamp.weekday()  # 0 = Monday, 6 = Sunday
            day_values[day_of_week].append(value)

        # Calculate average for each day
        day_averages = {}
        for day, day_vals in day_values.items():
            if day_vals:
                day_averages[day] = statistics.mean(day_vals)

        if len(day_averages) < 5:  # Need at least 5 days
            return None

        # Check if there's a significant pattern
        avg_values = list(day_averages.values())
        overall_mean = statistics.mean(avg_values)
        variations = [abs(val - overall_mean) for val in avg_values]

        if max(variations) > overall_mean * 0.2:  # 20% variation threshold
            return {
                "type": "weekly_cycle",
                "day_averages": {
                    ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"][day]: avg
                    for day, avg in day_averages.items()
                },
                "peak_day": ["Monday", "Tuesday", "Wednesday", "Thursday",
                           "Friday", "Saturday", "Sunday"][max(day_averages, key=day_averages.get)],
                "low_day": ["Monday", "Tuesday", "Wednesday", "Thursday",
                          "Friday", "Saturday", "Sunday"][min(day_averages, key=day_averages.get)],
                "variation_coefficient": max(variations) / overall_mean
            }

        return None

    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (coefficient of variation)."""
        if len(values) < 2:
            return 0.0

        mean_value = statistics.mean(values)
        if mean_value == 0:
            return 0.0

        std_dev = statistics.stdev(values)
        return std_dev / mean_value


class CostOptimizationAnalyzer:
    """Analyzes usage patterns for cost optimization opportunities."""

    def __init__(self):
        self.optimization_cache = {}

    async def analyze_cost_opportunities(
        self,
        usage_records: List[UsageRecord],
        time_window_hours: int = 24 * 7  # Default: 1 week
    ) -> Dict[str, Any]:
        """Analyze cost optimization opportunities."""
        if not usage_records:
            return {"opportunities": [], "total_potential_savings": 0.0}

        opportunities = []
        total_potential_savings = Decimal("0.00")

        # Analyze by provider
        provider_analysis = await self._analyze_provider_efficiency(usage_records)
        if provider_analysis["opportunities"]:
            opportunities.extend(provider_analysis["opportunities"])
            total_potential_savings += provider_analysis["potential_savings"]

        # Analyze by model
        model_analysis = await self._analyze_model_efficiency(usage_records)
        if model_analysis["opportunities"]:
            opportunities.extend(model_analysis["opportunities"])
            total_potential_savings += model_analysis["potential_savings"]

        # Analyze usage patterns
        pattern_analysis = await self._analyze_usage_patterns(usage_records)
        if pattern_analysis["opportunities"]:
            opportunities.extend(pattern_analysis["opportunities"])
            total_potential_savings += pattern_analysis["potential_savings"]

        # Analyze error rates
        error_analysis = await self._analyze_error_costs(usage_records)
        if error_analysis["opportunities"]:
            opportunities.extend(error_analysis["opportunities"])
            total_potential_savings += error_analysis["potential_savings"]

        return {
            "opportunities": opportunities,
            "total_potential_savings": float(total_potential_savings),
            "analysis_period": f"{time_window_hours} hours",
            "records_analyzed": len(usage_records)
        }

    async def _analyze_provider_efficiency(
        self,
        records: List[UsageRecord]
    ) -> Dict[str, Any]:
        """Analyze efficiency across different providers."""
        provider_stats = defaultdict(lambda: {
            "total_cost": Decimal("0.00"),
            "total_tokens": 0,
            "request_count": 0,
            "avg_cost_per_token": Decimal("0.00"),
            "error_rate": 0.0
        })

        # Collect stats by provider
        for record in records:
            stats = provider_stats[record.provider.value]
            stats["total_cost"] += record.cost_usd
            stats["total_tokens"] += record.token_count
            stats["request_count"] += 1
            if not record.success:
                stats["error_rate"] += 1

        # Calculate derived metrics
        for provider, stats in provider_stats.items():
            if stats["total_tokens"] > 0:
                stats["avg_cost_per_token"] = stats["total_cost"] / stats["total_tokens"]
            if stats["request_count"] > 0:
                stats["error_rate"] = stats["error_rate"] / stats["request_count"]

        # Find optimization opportunities
        opportunities = []
        potential_savings = Decimal("0.00")

        if len(provider_stats) > 1:
            # Find most and least efficient providers
            providers_by_efficiency = sorted(
                provider_stats.items(),
                key=lambda x: float(x[1]["avg_cost_per_token"])
            )

            most_efficient = providers_by_efficiency[0]
            least_efficient = providers_by_efficiency[-1]

            if float(most_efficient[1]["avg_cost_per_token"]) > 0:
                cost_difference = least_efficient[1]["avg_cost_per_token"] - most_efficient[1]["avg_cost_per_token"]
                if cost_difference > 0:
                    # Calculate potential savings by switching
                    potential_token_savings = least_efficient[1]["total_tokens"] * cost_difference
                    potential_savings += potential_token_savings

                    opportunities.append({
                        "type": "provider_switch",
                        "description": f"Switch from {least_efficient[0]} to {most_efficient[0]}",
                        "current_provider": least_efficient[0],
                        "recommended_provider": most_efficient[0],
                        "potential_savings": float(potential_token_savings),
                        "cost_per_token_improvement": float(cost_difference),
                        "affected_tokens": least_efficient[1]["total_tokens"],
                        "confidence": 0.8  # High confidence for provider switching
                    })

        return {
            "opportunities": opportunities,
            "potential_savings": potential_savings,
            "provider_stats": dict(provider_stats)
        }

    async def _analyze_model_efficiency(
        self,
        records: List[UsageRecord]
    ) -> Dict[str, Any]:
        """Analyze efficiency across different models."""
        model_stats = defaultdict(lambda: {
            "total_cost": Decimal("0.00"),
            "total_tokens": 0,
            "request_count": 0,
            "avg_cost_per_token": Decimal("0.00"),
            "success_rate": 1.0
        })

        # Collect stats by model
        for record in records:
            model_name = record.model_name or "unknown"
            stats = model_stats[model_name]
            stats["total_cost"] += record.cost_usd
            stats["total_tokens"] += record.token_count
            stats["request_count"] += 1
            if not record.success:
                stats["success_rate"] -= 1.0 / stats["request_count"]

        # Calculate derived metrics
        for model, stats in model_stats.items():
            if stats["total_tokens"] > 0:
                stats["avg_cost_per_token"] = stats["total_cost"] / stats["total_tokens"]

        opportunities = []
        potential_savings = Decimal("0.00")

        if len(model_stats) > 1:
            # Find opportunities to use cheaper models for similar tasks
            models_by_cost = sorted(
                [(k, v) for k, v in model_stats.items() if v["success_rate"] > 0.9],
                key=lambda x: float(x[1]["avg_cost_per_token"])
            )

            if len(models_by_cost) > 1:
                cheapest_model = models_by_cost[0]

                for model_name, stats in models_by_cost[1:]:
                    cost_difference = stats["avg_cost_per_token"] - cheapest_model[1]["avg_cost_per_token"]
                    if cost_difference > 0:
                        potential_token_savings = stats["total_tokens"] * cost_difference
                        potential_savings += potential_token_savings

                        opportunities.append({
                            "type": "model_downgrade",
                            "description": f"Use {cheapest_model[0]} instead of {model_name} for cost-sensitive tasks",
                            "current_model": model_name,
                            "recommended_model": cheapest_model[0],
                            "potential_savings": float(potential_token_savings),
                            "cost_per_token_improvement": float(cost_difference),
                            "affected_tokens": stats["total_tokens"],
                            "confidence": 0.6  # Medium confidence - depends on task similarity
                        })

        return {
            "opportunities": opportunities,
            "potential_savings": potential_savings,
            "model_stats": dict(model_stats)
        }

    async def _analyze_usage_patterns(
        self,
        records: List[UsageRecord]
    ) -> Dict[str, Any]:
        """Analyze usage patterns for optimization opportunities."""
        opportunities = []
        potential_savings = Decimal("0.00")

        # Analyze peak vs off-peak usage (if providers offer different pricing)
        hourly_usage = defaultdict(lambda: {"cost": Decimal("0.00"), "requests": 0})

        for record in records:
            hour = record.timestamp.hour
            hourly_usage[hour]["cost"] += record.cost_usd
            hourly_usage[hour]["requests"] += 1

        # Find peak hours
        peak_hours = sorted(
            hourly_usage.items(),
            key=lambda x: float(x[1]["cost"]),
            reverse=True
        )[:6]  # Top 6 hours

        total_peak_cost = sum(float(hour_data[1]["cost"]) for hour_data in peak_hours)
        total_cost = sum(float(data["cost"]) for data in hourly_usage.values())

        if total_cost > 0 and total_peak_cost / total_cost > 0.6:
            # High concentration in peak hours - suggest load balancing
            estimated_savings = Decimal(str(total_peak_cost * 0.15))  # 15% savings estimate
            potential_savings += estimated_savings

            opportunities.append({
                "type": "load_balancing",
                "description": "Distribute usage more evenly throughout the day",
                "peak_hours": [hour for hour, _ in peak_hours],
                "peak_cost_percentage": (total_peak_cost / total_cost) * 100,
                "potential_savings": float(estimated_savings),
                "confidence": 0.4  # Lower confidence - depends on flexibility
            })

        return {
            "opportunities": opportunities,
            "potential_savings": potential_savings,
            "hourly_usage": dict(hourly_usage)
        }

    async def _analyze_error_costs(
        self,
        records: List[UsageRecord]
    ) -> Dict[str, Any]:
        """Analyze costs associated with errors and failures."""
        failed_records = [r for r in records if not r.success]
        if not failed_records:
            return {"opportunities": [], "potential_savings": Decimal("0.00")}

        total_failed_cost = sum(r.cost_usd for r in failed_records)
        error_rate = len(failed_records) / len(records)

        opportunities = []
        potential_savings = Decimal("0.00")

        if error_rate > 0.05:  # More than 5% error rate
            # Assume 50% of error costs could be saved with better error handling
            estimated_savings = total_failed_cost * Decimal("0.5")
            potential_savings += estimated_savings

            opportunities.append({
                "type": "error_reduction",
                "description": "Implement better error handling and retry logic",
                "error_rate": error_rate * 100,
                "failed_requests": len(failed_records),
                "total_error_cost": float(total_failed_cost),
                "potential_savings": float(estimated_savings),
                "confidence": 0.7  # Good confidence - error reduction is usually effective
            })

        return {
            "opportunities": opportunities,
            "potential_savings": potential_savings
        }