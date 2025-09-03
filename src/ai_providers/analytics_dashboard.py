"""Usage analytics dashboard with comprehensive data models and aggregation strategies."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict
import statistics
from structlog import get_logger

from .models import ProviderType, UsageRecord
from .user_usage_tracker import UserUsageTracker, UserUsageAggregation
from .pricing_engine import PricingEngine
from ..core.database import ChromaDBManager, get_db_manager

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics for analytics."""
    
    USAGE = "usage"  # Token usage, request counts
    COST = "cost"  # Cost-related metrics
    PERFORMANCE = "performance"  # Latency, success rates
    USER = "user"  # User behavior metrics
    PROVIDER = "provider"  # Provider-specific metrics


class AggregationPeriod(Enum):
    """Time periods for data aggregation."""
    
    HOUR = "hour"
    DAY = "day" 
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class ChartType(Enum):
    """Chart types for dashboard visualization."""
    
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"
    HEATMAP = "heatmap"
    TABLE = "table"
    GAUGE = "gauge"
    SCATTER = "scatter"


@dataclass
class MetricDefinition:
    """Definition of a dashboard metric."""
    
    metric_id: str
    name: str
    description: str
    metric_type: MetricType
    chart_type: ChartType
    aggregation_method: str  # sum, avg, count, max, min, percentile
    
    # Data source configuration
    source_field: str  # Field name in usage records
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # Display configuration
    unit: str = ""  # USD, tokens, requests, ms, etc.
    decimal_places: int = 2
    show_trend: bool = True
    show_comparison: bool = True
    
    # Thresholds and alerts
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    enabled: bool = True


@dataclass 
class DashboardWidget:
    """Dashboard widget configuration."""
    
    widget_id: str
    title: str
    description: str
    metric_ids: List[str]  # Metrics to display in this widget
    chart_type: ChartType
    
    # Layout configuration
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "width": 4, "height": 3})
    
    # Display options
    time_range: str = "24h"  # 1h, 24h, 7d, 30d, 90d, 1y
    refresh_interval: int = 60  # seconds
    show_legend: bool = True
    show_filters: bool = False
    
    # Data configuration
    aggregation_period: AggregationPeriod = AggregationPeriod.HOUR
    max_data_points: int = 100
    
    created_at: datetime = field(default_factory=datetime.now)
    enabled: bool = True


@dataclass
class DashboardLayout:
    """Complete dashboard layout definition."""
    
    dashboard_id: str
    name: str
    description: str
    user_tier: str  # Which users can see this dashboard
    
    widgets: List[DashboardWidget] = field(default_factory=list)
    
    # Access control
    public: bool = False
    allowed_users: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"


@dataclass
class MetricDataPoint:
    """Single data point for a metric."""
    
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "metadata": self.metadata
        }


@dataclass
class MetricTimeSeries:
    """Time series data for a metric."""
    
    metric_id: str
    data_points: List[MetricDataPoint] = field(default_factory=list)
    aggregation_period: AggregationPeriod = AggregationPeriod.HOUR
    
    # Summary statistics
    total: float = 0.0
    average: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    trend_percentage: float = 0.0  # % change from previous period
    
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric_id": self.metric_id,
            "data_points": [dp.to_dict() for dp in self.data_points],
            "aggregation_period": self.aggregation_period.value,
            "summary": {
                "total": self.total,
                "average": self.average,
                "minimum": self.minimum,
                "maximum": self.maximum,
                "trend_percentage": self.trend_percentage,
            },
            "generated_at": self.generated_at.isoformat()
        }


class AnalyticsDashboard:
    """Analytics dashboard with comprehensive metrics and visualizations."""
    
    def __init__(
        self,
        usage_tracker: UserUsageTracker,
        pricing_engine: PricingEngine,
        storage_path: Optional[str] = None,
        use_chromadb: bool = True
    ):
        self.usage_tracker = usage_tracker
        self.pricing_engine = pricing_engine
        
        self.storage_path = Path(storage_path) if storage_path else Path("./data/analytics")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.use_chromadb = use_chromadb
        if use_chromadb:
            self.db_manager = get_db_manager()
        
        # Dashboard configuration
        self.metrics: Dict[str, MetricDefinition] = {}
        self.widgets: Dict[str, DashboardWidget] = {}
        self.dashboards: Dict[str, DashboardLayout] = {}
        
        # Cached metric data
        self.metric_cache: Dict[str, MetricTimeSeries] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_update: Dict[str, datetime] = {}
        
        # Performance tracking
        self._aggregation_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize default metrics and dashboards
        self._initialize_default_metrics()
        self._initialize_default_dashboards()
        
        logger.info("Analytics dashboard initialized", metrics=len(self.metrics))
    
    def _initialize_default_metrics(self) -> None:
        """Initialize default metrics for the dashboard."""
        default_metrics = [
            # Usage metrics
            MetricDefinition(
                metric_id="total_requests",
                name="Total Requests",
                description="Total number of AI requests",
                metric_type=MetricType.USAGE,
                chart_type=ChartType.LINE,
                aggregation_method="count",
                source_field="request_id",
                unit="requests"
            ),
            MetricDefinition(
                metric_id="successful_requests",
                name="Successful Requests",
                description="Number of successful AI requests", 
                metric_type=MetricType.USAGE,
                chart_type=ChartType.LINE,
                aggregation_method="count",
                source_field="request_id",
                filters={"success": True},
                unit="requests"
            ),
            MetricDefinition(
                metric_id="error_rate",
                name="Error Rate",
                description="Percentage of failed requests",
                metric_type=MetricType.PERFORMANCE,
                chart_type=ChartType.GAUGE,
                aggregation_method="avg",
                source_field="success",
                unit="%",
                warning_threshold=5.0,
                critical_threshold=10.0
            ),
            
            # Token usage metrics
            MetricDefinition(
                metric_id="total_tokens",
                name="Total Tokens",
                description="Total tokens processed (input + output)",
                metric_type=MetricType.USAGE,
                chart_type=ChartType.AREA,
                aggregation_method="sum",
                source_field="total_tokens",
                unit="tokens"
            ),
            MetricDefinition(
                metric_id="input_tokens",
                name="Input Tokens",
                description="Total input tokens",
                metric_type=MetricType.USAGE,
                chart_type=ChartType.LINE,
                aggregation_method="sum",
                source_field="input_tokens",
                unit="tokens"
            ),
            MetricDefinition(
                metric_id="output_tokens",
                name="Output Tokens", 
                description="Total output tokens",
                metric_type=MetricType.USAGE,
                chart_type=ChartType.LINE,
                aggregation_method="sum",
                source_field="output_tokens",
                unit="tokens"
            ),
            
            # Cost metrics
            MetricDefinition(
                metric_id="total_cost",
                name="Total Cost",
                description="Total cost of AI requests",
                metric_type=MetricType.COST,
                chart_type=ChartType.LINE,
                aggregation_method="sum",
                source_field="cost",
                unit="USD",
                decimal_places=4
            ),
            MetricDefinition(
                metric_id="avg_cost_per_request",
                name="Average Cost per Request",
                description="Average cost per request",
                metric_type=MetricType.COST,
                chart_type=ChartType.LINE,
                aggregation_method="avg",
                source_field="cost",
                unit="USD",
                decimal_places=4
            ),
            MetricDefinition(
                metric_id="cost_per_token",
                name="Cost per Token",
                description="Average cost per token",
                metric_type=MetricType.COST,
                chart_type=ChartType.LINE,
                aggregation_method="avg",
                source_field="cost_per_token",
                unit="USD/token",
                decimal_places=6
            ),
            
            # Performance metrics
            MetricDefinition(
                metric_id="avg_latency",
                name="Average Latency",
                description="Average response latency",
                metric_type=MetricType.PERFORMANCE,
                chart_type=ChartType.LINE,
                aggregation_method="avg",
                source_field="latency_ms",
                unit="ms",
                decimal_places=0,
                warning_threshold=2000.0,
                critical_threshold=5000.0
            ),
            MetricDefinition(
                metric_id="p95_latency",
                name="95th Percentile Latency",
                description="95th percentile response latency",
                metric_type=MetricType.PERFORMANCE,
                chart_type=ChartType.LINE,
                aggregation_method="percentile_95",
                source_field="latency_ms",
                unit="ms",
                decimal_places=0
            ),
            
            # Provider metrics
            MetricDefinition(
                metric_id="provider_distribution",
                name="Provider Distribution",
                description="Distribution of requests by provider",
                metric_type=MetricType.PROVIDER,
                chart_type=ChartType.PIE,
                aggregation_method="count",
                source_field="provider_type",
                unit="requests"
            ),
            MetricDefinition(
                metric_id="model_distribution",
                name="Model Distribution", 
                description="Distribution of requests by model",
                metric_type=MetricType.PROVIDER,
                chart_type=ChartType.BAR,
                aggregation_method="count",
                source_field="model",
                unit="requests"
            ),
            
            # User metrics
            MetricDefinition(
                metric_id="active_users",
                name="Active Users",
                description="Number of active users",
                metric_type=MetricType.USER,
                chart_type=ChartType.GAUGE,
                aggregation_method="count_distinct",
                source_field="user_id",
                unit="users"
            ),
            MetricDefinition(
                metric_id="requests_per_user",
                name="Requests per User",
                description="Average requests per active user",
                metric_type=MetricType.USER,
                chart_type=ChartType.LINE,
                aggregation_method="avg",
                source_field="requests_per_user",
                unit="req/user"
            ),
        ]
        
        for metric in default_metrics:
            self.metrics[metric.metric_id] = metric
    
    def _initialize_default_dashboards(self) -> None:
        """Initialize default dashboard layouts."""
        # Overview Dashboard
        overview_widgets = [
            DashboardWidget(
                widget_id="overview_requests",
                title="Request Volume",
                description="Total and successful requests over time",
                metric_ids=["total_requests", "successful_requests"],
                chart_type=ChartType.LINE,
                position={"x": 0, "y": 0, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="overview_cost",
                title="Cost Tracking",
                description="Total cost and cost per request",
                metric_ids=["total_cost", "avg_cost_per_request"],
                chart_type=ChartType.LINE,
                position={"x": 6, "y": 0, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="overview_tokens",
                title="Token Usage",
                description="Input and output token usage",
                metric_ids=["input_tokens", "output_tokens"],
                chart_type=ChartType.AREA,
                position={"x": 0, "y": 4, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="overview_performance",
                title="Performance Metrics", 
                description="Latency and error rates",
                metric_ids=["avg_latency", "error_rate"],
                chart_type=ChartType.LINE,
                position={"x": 6, "y": 4, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="overview_providers",
                title="Provider Distribution",
                description="Usage distribution by AI provider",
                metric_ids=["provider_distribution"],
                chart_type=ChartType.PIE,
                position={"x": 0, "y": 8, "width": 4, "height": 4}
            ),
            DashboardWidget(
                widget_id="overview_users",
                title="User Activity",
                description="Active users and usage per user",
                metric_ids=["active_users", "requests_per_user"],
                chart_type=ChartType.GAUGE,
                position={"x": 4, "y": 8, "width": 4, "height": 4}
            ),
        ]
        
        overview_dashboard = DashboardLayout(
            dashboard_id="overview",
            name="Overview Dashboard",
            description="High-level overview of AI usage and costs",
            user_tier="all",
            widgets=overview_widgets,
            public=True
        )
        
        # Cost Analysis Dashboard
        cost_widgets = [
            DashboardWidget(
                widget_id="cost_trend",
                title="Cost Trend",
                description="Daily cost trend over time",
                metric_ids=["total_cost"],
                chart_type=ChartType.LINE,
                time_range="30d",
                aggregation_period=AggregationPeriod.DAY,
                position={"x": 0, "y": 0, "width": 12, "height": 4}
            ),
            DashboardWidget(
                widget_id="cost_breakdown",
                title="Cost Breakdown by Provider",
                description="Cost distribution across providers",
                metric_ids=["total_cost"],
                chart_type=ChartType.BAR,
                position={"x": 0, "y": 4, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="cost_efficiency",
                title="Cost Efficiency", 
                description="Cost per token and per request trends",
                metric_ids=["cost_per_token", "avg_cost_per_request"],
                chart_type=ChartType.LINE,
                position={"x": 6, "y": 4, "width": 6, "height": 4}
            ),
        ]
        
        cost_dashboard = DashboardLayout(
            dashboard_id="cost_analysis",
            name="Cost Analysis",
            description="Detailed cost analysis and optimization insights",
            user_tier="premium",
            widgets=cost_widgets
        )
        
        # User Analytics Dashboard  
        user_widgets = [
            DashboardWidget(
                widget_id="user_activity",
                title="User Activity Heatmap",
                description="User activity patterns by time",
                metric_ids=["active_users"],
                chart_type=ChartType.HEATMAP,
                time_range="7d",
                aggregation_period=AggregationPeriod.HOUR,
                position={"x": 0, "y": 0, "width": 12, "height": 6}
            ),
            DashboardWidget(
                widget_id="user_costs",
                title="Top Users by Cost",
                description="Users with highest AI usage costs",
                metric_ids=["total_cost"],
                chart_type=ChartType.TABLE,
                position={"x": 0, "y": 6, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="user_models",
                title="Popular Models",
                description="Most used models across users",
                metric_ids=["model_distribution"],
                chart_type=ChartType.BAR,
                position={"x": 6, "y": 6, "width": 6, "height": 4}
            ),
        ]
        
        user_dashboard = DashboardLayout(
            dashboard_id="user_analytics",
            name="User Analytics",
            description="User behavior and usage patterns",
            user_tier="enterprise",
            widgets=user_widgets
        )
        
        self.dashboards["overview"] = overview_dashboard
        self.dashboards["cost_analysis"] = cost_dashboard
        self.dashboards["user_analytics"] = user_dashboard
    
    async def generate_metric_data(
        self,
        metric_id: str,
        start_time: datetime,
        end_time: datetime,
        aggregation_period: AggregationPeriod = AggregationPeriod.HOUR,
        user_id: Optional[str] = None
    ) -> MetricTimeSeries:
        """Generate time series data for a specific metric."""
        if metric_id not in self.metrics:
            raise ValueError(f"Unknown metric: {metric_id}")
        
        metric = self.metrics[metric_id]
        
        # Check cache first
        cache_key = f"{metric_id}:{start_time.isoformat()}:{end_time.isoformat()}:{aggregation_period.value}:{user_id or 'all'}"
        if cache_key in self.metric_cache:
            cached_data = self.metric_cache[cache_key]
            if datetime.now() - self._last_cache_update.get(cache_key, datetime.min) < timedelta(seconds=self._cache_ttl):
                return cached_data
        
        # Generate time series
        time_series = await self._aggregate_metric_data(metric, start_time, end_time, aggregation_period, user_id)
        
        # Cache the result
        self.metric_cache[cache_key] = time_series
        self._last_cache_update[cache_key] = datetime.now()
        
        return time_series
    
    async def _aggregate_metric_data(
        self,
        metric: MetricDefinition,
        start_time: datetime,
        end_time: datetime,
        aggregation_period: AggregationPeriod,
        user_id: Optional[str] = None
    ) -> MetricTimeSeries:
        """Aggregate raw usage data into metric time series."""
        
        # Generate time buckets
        time_buckets = self._generate_time_buckets(start_time, end_time, aggregation_period)
        data_points: List[MetricDataPoint] = []
        
        # Get raw usage data
        usage_data = await self._get_usage_data(start_time, end_time, user_id, metric.filters)
        
        # Aggregate data into time buckets
        for bucket_start, bucket_end in time_buckets:
            bucket_data = [
                record for record in usage_data
                if bucket_start <= record.timestamp < bucket_end
            ]
            
            # Apply aggregation method
            value = self._apply_aggregation(bucket_data, metric)
            
            data_point = MetricDataPoint(
                timestamp=bucket_start,
                value=value,
                metadata={"bucket_end": bucket_end.isoformat(), "record_count": len(bucket_data)}
            )
            data_points.append(data_point)
        
        # Calculate summary statistics
        values = [dp.value for dp in data_points if dp.value is not None]
        
        summary_stats = {
            "total": sum(values) if values else 0.0,
            "average": statistics.mean(values) if values else 0.0,
            "minimum": min(values) if values else 0.0,
            "maximum": max(values) if values else 0.0,
        }
        
        # Calculate trend
        trend_percentage = self._calculate_trend(data_points)
        
        time_series = MetricTimeSeries(
            metric_id=metric.metric_id,
            data_points=data_points,
            aggregation_period=aggregation_period,
            total=summary_stats["total"],
            average=summary_stats["average"],
            minimum=summary_stats["minimum"],
            maximum=summary_stats["maximum"],
            trend_percentage=trend_percentage
        )
        
        return time_series
    
    def _generate_time_buckets(
        self,
        start_time: datetime,
        end_time: datetime,
        period: AggregationPeriod
    ) -> List[Tuple[datetime, datetime]]:
        """Generate time buckets for aggregation."""
        buckets = []
        
        # Calculate bucket size
        if period == AggregationPeriod.HOUR:
            delta = timedelta(hours=1)
        elif period == AggregationPeriod.DAY:
            delta = timedelta(days=1)
        elif period == AggregationPeriod.WEEK:
            delta = timedelta(weeks=1)
        elif period == AggregationPeriod.MONTH:
            delta = timedelta(days=30)  # Approximation
        elif period == AggregationPeriod.QUARTER:
            delta = timedelta(days=90)  # Approximation
        elif period == AggregationPeriod.YEAR:
            delta = timedelta(days=365)  # Approximation
        else:
            delta = timedelta(hours=1)  # Default
        
        # Generate buckets
        current_time = start_time
        while current_time < end_time:
            bucket_end = min(current_time + delta, end_time)
            buckets.append((current_time, bucket_end))
            current_time = bucket_end
        
        return buckets
    
    async def _get_usage_data(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[UsageRecord]:
        """Get raw usage data from storage."""
        # This is a simplified version - in practice, you'd query your storage
        # For now, we'll use the in-memory usage data
        
        all_records = []
        
        # Get data from daily usage aggregations
        if user_id:
            user_daily = self.usage_tracker.daily_usage.get(user_id, {})
            for date_str, agg in user_daily.items():
                record_date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                if start_time <= record_date <= end_time:
                    # Convert aggregation back to individual records (approximation)
                    # In practice, you'd store and retrieve actual usage records
                    for _ in range(agg.total_requests):
                        record = UsageRecord(
                            request_id=f"agg_{date_str}_{_}",
                            provider_type=ProviderType.ANTHROPIC,  # Default
                            session_id=None,
                            model="unknown",
                            input_tokens=agg.total_input_tokens // max(agg.total_requests, 1),
                            output_tokens=agg.total_output_tokens // max(agg.total_requests, 1),
                            cost=agg.total_cost / max(agg.total_requests, 1),
                            latency_ms=agg.avg_latency_ms,
                            timestamp=record_date,
                            success=agg.successful_requests > agg.failed_requests,
                            metadata={"user_id": user_id}
                        )
                        all_records.append(record)
        else:
            # Aggregate across all users
            for user_id_iter, user_daily in self.usage_tracker.daily_usage.items():
                for date_str, agg in user_daily.items():
                    record_date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                    if start_time <= record_date <= end_time:
                        for _ in range(agg.total_requests):
                            record = UsageRecord(
                                request_id=f"agg_{user_id_iter}_{date_str}_{_}",
                                provider_type=ProviderType.ANTHROPIC,  # Default
                                session_id=None,
                                model="unknown",
                                input_tokens=agg.total_input_tokens // max(agg.total_requests, 1),
                                output_tokens=agg.total_output_tokens // max(agg.total_requests, 1),
                                cost=agg.total_cost / max(agg.total_requests, 1),
                                latency_ms=agg.avg_latency_ms,
                                timestamp=record_date,
                                success=agg.successful_requests > agg.failed_requests,
                                metadata={"user_id": user_id_iter}
                            )
                            all_records.append(record)
        
        # Apply filters
        if filters:
            filtered_records = []
            for record in all_records:
                include_record = True
                for filter_key, filter_value in filters.items():
                    if hasattr(record, filter_key):
                        if getattr(record, filter_key) != filter_value:
                            include_record = False
                            break
                    elif filter_key in record.metadata:
                        if record.metadata[filter_key] != filter_value:
                            include_record = False
                            break
                
                if include_record:
                    filtered_records.append(record)
            
            all_records = filtered_records
        
        return all_records
    
    def _apply_aggregation(self, records: List[UsageRecord], metric: MetricDefinition) -> float:
        """Apply aggregation method to a list of usage records."""
        if not records:
            return 0.0
        
        # Get values based on source field
        values = []
        for record in records:
            if metric.source_field == "request_id":
                values.append(1)  # Count requests
            elif metric.source_field == "total_tokens":
                values.append(record.input_tokens + record.output_tokens)
            elif metric.source_field == "cost_per_token":
                total_tokens = record.input_tokens + record.output_tokens
                if total_tokens > 0:
                    values.append(record.cost / total_tokens)
            elif metric.source_field == "success":
                # For error rate calculation
                values.append(0 if record.success else 1)
            elif metric.source_field == "user_id":
                values.append(record.metadata.get("user_id", "unknown"))
            elif hasattr(record, metric.source_field):
                values.append(getattr(record, metric.source_field))
            else:
                values.append(0)
        
        # Apply aggregation method
        if metric.aggregation_method == "sum":
            return sum(v for v in values if isinstance(v, (int, float)))
        elif metric.aggregation_method == "avg":
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            return statistics.mean(numeric_values) if numeric_values else 0.0
        elif metric.aggregation_method == "count":
            return len(values)
        elif metric.aggregation_method == "count_distinct":
            return len(set(values))
        elif metric.aggregation_method == "max":
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            return max(numeric_values) if numeric_values else 0.0
        elif metric.aggregation_method == "min":
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            return min(numeric_values) if numeric_values else 0.0
        elif metric.aggregation_method == "percentile_95":
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if numeric_values:
                return statistics.quantiles(numeric_values, n=20)[18]  # 95th percentile
            return 0.0
        else:
            return 0.0
    
    def _calculate_trend(self, data_points: List[MetricDataPoint]) -> float:
        """Calculate trend percentage from data points."""
        if len(data_points) < 2:
            return 0.0
        
        # Simple trend: compare first half to second half
        mid_point = len(data_points) // 2
        first_half = data_points[:mid_point]
        second_half = data_points[mid_point:]
        
        first_avg = statistics.mean([dp.value for dp in first_half if dp.value is not None])
        second_avg = statistics.mean([dp.value for dp in second_half if dp.value is not None])
        
        if first_avg == 0:
            return 0.0
        
        return ((second_avg - first_avg) / first_avg) * 100
    
    async def get_dashboard_data(
        self,
        dashboard_id: str,
        time_range: str = "24h",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get complete dashboard data for rendering."""
        if dashboard_id not in self.dashboards:
            raise ValueError(f"Unknown dashboard: {dashboard_id}")
        
        dashboard = self.dashboards[dashboard_id]
        
        # Parse time range
        end_time = datetime.now(timezone.utc)
        if time_range == "1h":
            start_time = end_time - timedelta(hours=1)
            aggregation_period = AggregationPeriod.HOUR
        elif time_range == "24h":
            start_time = end_time - timedelta(hours=24)
            aggregation_period = AggregationPeriod.HOUR
        elif time_range == "7d":
            start_time = end_time - timedelta(days=7)
            aggregation_period = AggregationPeriod.DAY
        elif time_range == "30d":
            start_time = end_time - timedelta(days=30)
            aggregation_period = AggregationPeriod.DAY
        elif time_range == "90d":
            start_time = end_time - timedelta(days=90)
            aggregation_period = AggregationPeriod.WEEK
        elif time_range == "1y":
            start_time = end_time - timedelta(days=365)
            aggregation_period = AggregationPeriod.MONTH
        else:
            start_time = end_time - timedelta(hours=24)
            aggregation_period = AggregationPeriod.HOUR
        
        dashboard_data = {
            "dashboard_id": dashboard_id,
            "name": dashboard.name,
            "description": dashboard.description,
            "time_range": time_range,
            "generated_at": datetime.now().isoformat(),
            "widgets": []
        }
        
        # Generate data for each widget
        for widget in dashboard.widgets:
            widget_data = {
                "widget_id": widget.widget_id,
                "title": widget.title,
                "description": widget.description,
                "chart_type": widget.chart_type.value,
                "position": widget.position,
                "metrics": {}
            }
            
            # Generate metric data for this widget
            for metric_id in widget.metric_ids:
                try:
                    metric_time_series = await self.generate_metric_data(
                        metric_id, start_time, end_time, aggregation_period, user_id
                    )
                    widget_data["metrics"][metric_id] = metric_time_series.to_dict()
                except Exception as e:
                    logger.error(f"Failed to generate metric data", metric_id=metric_id, error=str(e))
                    widget_data["metrics"][metric_id] = {"error": str(e)}
            
            dashboard_data["widgets"].append(widget_data)
        
        return dashboard_data
    
    async def get_real_time_metrics(self, metric_ids: List[str]) -> Dict[str, Any]:
        """Get real-time metric values."""
        real_time_data = {}
        current_time = datetime.now(timezone.utc)
        
        for metric_id in metric_ids:
            try:
                # Get last hour of data
                start_time = current_time - timedelta(hours=1)
                time_series = await self.generate_metric_data(
                    metric_id, start_time, current_time, AggregationPeriod.HOUR
                )
                
                # Get latest value
                latest_value = time_series.data_points[-1].value if time_series.data_points else 0.0
                
                real_time_data[metric_id] = {
                    "current_value": latest_value,
                    "trend": time_series.trend_percentage,
                    "timestamp": current_time.isoformat()
                }
                
            except Exception as e:
                logger.error(f"Failed to get real-time metric", metric_id=metric_id, error=str(e))
                real_time_data[metric_id] = {"error": str(e)}
        
        return real_time_data
    
    def add_custom_metric(self, metric: MetricDefinition) -> None:
        """Add a custom metric definition."""
        self.metrics[metric.metric_id] = metric
        logger.info("Custom metric added", metric_id=metric.metric_id)
    
    def add_custom_dashboard(self, dashboard: DashboardLayout) -> None:
        """Add a custom dashboard layout."""
        self.dashboards[dashboard.dashboard_id] = dashboard
        logger.info("Custom dashboard added", dashboard_id=dashboard.dashboard_id)
    
    def get_available_dashboards(self, user_tier: str = "free") -> List[Dict[str, Any]]:
        """Get list of available dashboards for a user tier."""
        available = []
        
        for dashboard in self.dashboards.values():
            if dashboard.public or dashboard.user_tier == "all" or dashboard.user_tier == user_tier:
                available.append({
                    "dashboard_id": dashboard.dashboard_id,
                    "name": dashboard.name,
                    "description": dashboard.description,
                    "widget_count": len(dashboard.widgets)
                })
        
        return available
    
    async def export_dashboard_data(
        self,
        dashboard_id: str,
        format_type: str = "json",
        time_range: str = "30d"
    ) -> Union[Dict[str, Any], str]:
        """Export dashboard data in specified format."""
        dashboard_data = await self.get_dashboard_data(dashboard_id, time_range)
        
        if format_type == "json":
            return dashboard_data
        elif format_type == "csv":
            # Convert to CSV format (simplified)
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(["widget_id", "metric_id", "timestamp", "value"])
            
            # Write data
            for widget in dashboard_data["widgets"]:
                for metric_id, metric_data in widget["metrics"].items():
                    if "data_points" in metric_data:
                        for point in metric_data["data_points"]:
                            writer.writerow([
                                widget["widget_id"],
                                metric_id,
                                point["timestamp"],
                                point["value"]
                            ])
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    async def cleanup(self) -> None:
        """Clean up resources and cancel background tasks."""
        # Cancel aggregation tasks
        for task in self._aggregation_tasks.values():
            if not task.done():
                task.cancel()
        
        # Clear cache
        self.metric_cache.clear()
        self._last_cache_update.clear()
        
        logger.info("Analytics dashboard cleanup completed")