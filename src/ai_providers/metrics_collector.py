"""Metrics collection and retention management system."""

import asyncio
import json
import gzip
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import statistics
import hashlib
from structlog import get_logger

from .models import ProviderType, UsageRecord
from .user_usage_tracker import UserUsageTracker
from ..core.database import ChromaDBManager, get_db_manager

logger = get_logger(__name__)


class RetentionPeriod(Enum):
    """Data retention periods."""
    
    HOURS_1 = "1h"
    HOURS_6 = "6h"
    HOURS_24 = "24h"
    DAYS_7 = "7d"
    DAYS_30 = "30d"
    DAYS_90 = "90d"
    DAYS_365 = "365d"
    PERMANENT = "permanent"


class AggregationLevel(Enum):
    """Levels of data aggregation."""
    
    RAW = "raw"  # Individual records
    MINUTE = "minute"  # Per-minute aggregation
    HOUR = "hour"  # Per-hour aggregation
    DAY = "day"  # Per-day aggregation
    WEEK = "week"  # Per-week aggregation
    MONTH = "month"  # Per-month aggregation


class CompressionType(Enum):
    """Data compression types."""
    
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"


@dataclass
class RetentionPolicy:
    """Data retention policy configuration."""
    
    policy_id: str
    name: str
    description: str
    
    # Retention rules
    raw_data_retention: RetentionPeriod = RetentionPeriod.DAYS_7
    aggregated_data_retention: RetentionPeriod = RetentionPeriod.DAYS_365
    
    # Aggregation rules
    aggregation_levels: List[AggregationLevel] = field(default_factory=lambda: [AggregationLevel.HOUR, AggregationLevel.DAY])
    aggregation_delay: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    
    # Compression settings
    compression_after: timedelta = field(default_factory=lambda: timedelta(days=1))
    compression_type: CompressionType = CompressionType.GZIP
    
    # Storage settings
    max_file_size_mb: int = 100
    backup_enabled: bool = True
    backup_retention: RetentionPeriod = RetentionPeriod.DAYS_30
    
    # Data types to retain
    include_user_data: bool = True
    include_provider_data: bool = True
    include_cost_data: bool = True
    include_performance_data: bool = True
    include_error_data: bool = True
    
    created_at: datetime = field(default_factory=datetime.now)
    enabled: bool = True


@dataclass
class MetricSnapshot:
    """Point-in-time metric snapshot."""
    
    timestamp: datetime
    metric_name: str
    metric_value: float
    aggregation_level: AggregationLevel
    
    # Context information
    user_id: Optional[str] = None
    provider_type: Optional[str] = None
    model: Optional[str] = None
    session_id: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "aggregation_level": self.aggregation_level.value,
            "user_id": self.user_id,
            "provider_type": self.provider_type,
            "model": self.model,
            "session_id": self.session_id,
            "metadata": self.metadata
        }


@dataclass
class AggregatedMetric:
    """Aggregated metric data."""
    
    metric_name: str
    aggregation_level: AggregationLevel
    time_bucket: datetime  # Start of the time bucket
    
    # Statistical aggregations
    count: int = 0
    sum_value: float = 0.0
    avg_value: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    std_dev: float = 0.0
    
    # Percentiles
    p50: float = 0.0  # Median
    p95: float = 0.0
    p99: float = 0.0
    
    # Context aggregations
    unique_users: int = 0
    unique_sessions: int = 0
    unique_providers: int = 0
    unique_models: int = 0
    
    # Error tracking
    error_count: int = 0
    error_rate: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "aggregation_level": self.aggregation_level.value,
            "time_bucket": self.time_bucket.isoformat(),
            "statistics": {
                "count": self.count,
                "sum": self.sum_value,
                "avg": self.avg_value,
                "min": self.min_value,
                "max": self.max_value,
                "std_dev": self.std_dev,
                "p50": self.p50,
                "p95": self.p95,
                "p99": self.p99
            },
            "context": {
                "unique_users": self.unique_users,
                "unique_sessions": self.unique_sessions,
                "unique_providers": self.unique_providers,
                "unique_models": self.unique_models
            },
            "errors": {
                "error_count": self.error_count,
                "error_rate": self.error_rate
            },
            "created_at": self.created_at.isoformat()
        }


class MetricsCollector:
    """Advanced metrics collection and retention management system."""
    
    def __init__(
        self,
        usage_tracker: UserUsageTracker,
        storage_path: Optional[str] = None,
        use_chromadb: bool = True
    ):
        self.usage_tracker = usage_tracker
        
        self.storage_path = Path(storage_path) if storage_path else Path("./data/metrics")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        (self.storage_path / "raw").mkdir(exist_ok=True)
        (self.storage_path / "aggregated").mkdir(exist_ok=True)
        (self.storage_path / "compressed").mkdir(exist_ok=True)
        (self.storage_path / "backups").mkdir(exist_ok=True)
        
        self.use_chromadb = use_chromadb
        if use_chromadb:
            self.db_manager = get_db_manager()
            self._initialize_chromadb_collections()
        
        # Retention policies
        self.retention_policies: Dict[str, RetentionPolicy] = {}
        self._load_default_policies()
        
        # In-memory metric buffers for real-time collection
        self.metric_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregated_metrics: Dict[str, Dict[str, AggregatedMetric]] = {}
        
        # Background processing
        self._collection_tasks: Dict[str, asyncio.Task] = {}
        self._aggregation_tasks: Dict[str, asyncio.Task] = {}
        self._retention_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.collection_stats = {
            "metrics_collected": 0,
            "aggregations_performed": 0,
            "files_compressed": 0,
            "files_cleaned": 0,
            "last_cleanup": None,
            "buffer_overflows": 0
        }
        
        # Start background processes
        asyncio.create_task(self._start_background_processes())
        
        logger.info("Metrics collector initialized", storage_path=str(self.storage_path))
    
    def _initialize_chromadb_collections(self) -> None:
        """Initialize ChromaDB collections for metrics storage."""
        try:
            # Metrics collection
            if "metrics" not in self.db_manager.collections:
                self.db_manager.client.create_collection(
                    name=f"{self.db_manager.client._settings.chroma_collection_prefix}metrics",
                    embedding_function=self.db_manager.embedding_function,
                    metadata={"description": "System metrics and performance data"}
                )
                self.db_manager.collections["metrics"] = self.db_manager.client.get_collection("metrics")
            
            # Aggregated metrics collection
            if "aggregated_metrics" not in self.db_manager.collections:
                self.db_manager.client.create_collection(
                    name=f"{self.db_manager.client._settings.chroma_collection_prefix}aggregated_metrics",
                    embedding_function=self.db_manager.embedding_function,
                    metadata={"description": "Aggregated system metrics"}
                )
                self.db_manager.collections["aggregated_metrics"] = self.db_manager.client.get_collection("aggregated_metrics")
                
        except Exception as e:
            logger.warning(f"Failed to initialize ChromaDB collections: {e}")
            self.use_chromadb = False
    
    def _load_default_policies(self) -> None:
        """Load default retention policies."""
        default_policies = [
            RetentionPolicy(
                policy_id="development",
                name="Development Environment",
                description="Short retention for development and testing",
                raw_data_retention=RetentionPeriod.DAYS_7,
                aggregated_data_retention=RetentionPeriod.DAYS_30,
                aggregation_levels=[AggregationLevel.HOUR, AggregationLevel.DAY],
                compression_after=timedelta(hours=6),
                compression_type=CompressionType.GZIP,
                backup_enabled=False
            ),
            RetentionPolicy(
                policy_id="production",
                name="Production Environment",
                description="Standard production retention policy",
                raw_data_retention=RetentionPeriod.DAYS_30,
                aggregated_data_retention=RetentionPeriod.DAYS_365,
                aggregation_levels=[AggregationLevel.MINUTE, AggregationLevel.HOUR, AggregationLevel.DAY, AggregationLevel.MONTH],
                compression_after=timedelta(days=7),
                compression_type=CompressionType.GZIP,
                backup_enabled=True,
                backup_retention=RetentionPeriod.DAYS_90
            ),
            RetentionPolicy(
                policy_id="enterprise",
                name="Enterprise Environment",
                description="Extended retention for enterprise customers",
                raw_data_retention=RetentionPeriod.DAYS_90,
                aggregated_data_retention=RetentionPeriod.PERMANENT,
                aggregation_levels=[AggregationLevel.MINUTE, AggregationLevel.HOUR, AggregationLevel.DAY, AggregationLevel.WEEK, AggregationLevel.MONTH],
                compression_after=timedelta(days=30),
                compression_type=CompressionType.LZMA,
                backup_enabled=True,
                backup_retention=RetentionPeriod.DAYS_365
            ),
        ]
        
        for policy in default_policies:
            self.retention_policies[policy.policy_id] = policy
        
        # Set default policy (would be configurable in production)
        self.active_policy = self.retention_policies["production"]
    
    async def _start_background_processes(self) -> None:
        """Start background collection and processing tasks."""
        try:
            # Start metric collection from usage tracker
            self._collection_tasks["usage_metrics"] = asyncio.create_task(
                self._collect_usage_metrics()
            )
            
            # Start aggregation tasks for different levels
            for level in self.active_policy.aggregation_levels:
                task_name = f"aggregation_{level.value}"
                self._collection_tasks[task_name] = asyncio.create_task(
                    self._aggregate_metrics(level)
                )
            
            # Start retention/cleanup task
            self._retention_task = asyncio.create_task(self._run_retention_cleanup())
            
            logger.info("Background metric processes started")
            
        except Exception as e:
            logger.error("Failed to start background processes", error=str(e))
    
    async def collect_metric(
        self,
        metric_name: str,
        metric_value: float,
        user_id: Optional[str] = None,
        provider_type: Optional[str] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Collect a single metric point."""
        try:
            snapshot = MetricSnapshot(
                timestamp=datetime.now(timezone.utc),
                metric_name=metric_name,
                metric_value=metric_value,
                aggregation_level=AggregationLevel.RAW,
                user_id=user_id,
                provider_type=provider_type,
                model=model,
                session_id=session_id,
                metadata=metadata or {}
            )
            
            # Add to buffer
            self.metric_buffers[metric_name].append(snapshot)
            
            # Check for buffer overflow
            if len(self.metric_buffers[metric_name]) >= self.metric_buffers[metric_name].maxlen - 1:
                self.collection_stats["buffer_overflows"] += 1
                logger.warning("Metric buffer near capacity", metric=metric_name)
            
            self.collection_stats["metrics_collected"] += 1
            
        except Exception as e:
            logger.error("Failed to collect metric", metric=metric_name, error=str(e))
    
    async def _collect_usage_metrics(self) -> None:
        """Continuously collect metrics from usage tracker."""
        while True:
            try:
                # Collect metrics from recent usage data
                await self._extract_usage_metrics()
                
                # Wait before next collection cycle
                await asyncio.sleep(30)  # Every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in usage metrics collection", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _extract_usage_metrics(self) -> None:
        """Extract metrics from usage tracker data."""
        current_time = datetime.now(timezone.utc)
        
        # Extract per-user metrics
        for user_id, daily_usage in self.usage_tracker.daily_usage.items():
            today = current_time.date().isoformat()
            if today in daily_usage:
                agg = daily_usage[today]
                
                # Collect various metrics
                await self.collect_metric("requests_total", float(agg.total_requests), user_id=user_id)
                await self.collect_metric("requests_successful", float(agg.successful_requests), user_id=user_id)
                await self.collect_metric("requests_failed", float(agg.failed_requests), user_id=user_id)
                await self.collect_metric("tokens_input", float(agg.total_input_tokens), user_id=user_id)
                await self.collect_metric("tokens_output", float(agg.total_output_tokens), user_id=user_id)
                await self.collect_metric("cost_total", agg.total_cost, user_id=user_id)
                await self.collect_metric("latency_avg", agg.avg_latency_ms, user_id=user_id)
                
                if agg.total_requests > 0:
                    error_rate = (agg.failed_requests / agg.total_requests) * 100
                    await self.collect_metric("error_rate", error_rate, user_id=user_id)
                    
                    cost_per_request = agg.total_cost / agg.total_requests
                    await self.collect_metric("cost_per_request", cost_per_request, user_id=user_id)
                
                # Provider-specific metrics
                for provider, count in agg.providers_used.items():
                    await self.collect_metric("provider_requests", float(count), 
                                            user_id=user_id, provider_type=provider)
                
                # Model-specific metrics
                for model, count in agg.models_used.items():
                    await self.collect_metric("model_requests", float(count),
                                            user_id=user_id, model=model)
    
    async def _aggregate_metrics(self, aggregation_level: AggregationLevel) -> None:
        """Aggregate metrics at specified level."""
        while True:
            try:
                await self._perform_aggregation(aggregation_level)
                
                # Determine sleep interval based on aggregation level
                if aggregation_level == AggregationLevel.MINUTE:
                    sleep_time = 60
                elif aggregation_level == AggregationLevel.HOUR:
                    sleep_time = 300  # 5 minutes
                elif aggregation_level == AggregationLevel.DAY:
                    sleep_time = 1800  # 30 minutes
                else:
                    sleep_time = 3600  # 1 hour
                
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in {aggregation_level.value} aggregation", error=str(e))
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _perform_aggregation(self, aggregation_level: AggregationLevel) -> None:
        """Perform metric aggregation for a specific time level."""
        current_time = datetime.now(timezone.utc)
        
        # Determine time bucket based on aggregation level
        if aggregation_level == AggregationLevel.MINUTE:
            bucket_start = current_time.replace(second=0, microsecond=0) - timedelta(minutes=1)
            bucket_end = bucket_start + timedelta(minutes=1)
        elif aggregation_level == AggregationLevel.HOUR:
            bucket_start = current_time.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            bucket_end = bucket_start + timedelta(hours=1)
        elif aggregation_level == AggregationLevel.DAY:
            bucket_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            bucket_end = bucket_start + timedelta(days=1)
        elif aggregation_level == AggregationLevel.WEEK:
            days_since_monday = current_time.weekday()
            bucket_start = (current_time.replace(hour=0, minute=0, second=0, microsecond=0) - 
                          timedelta(days=days_since_monday + 7))
            bucket_end = bucket_start + timedelta(weeks=1)
        elif aggregation_level == AggregationLevel.MONTH:
            bucket_start = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            bucket_start = bucket_start.replace(day=1)
            if bucket_start.month == 12:
                bucket_end = bucket_start.replace(year=bucket_start.year + 1, month=1)
            else:
                bucket_end = bucket_start.replace(month=bucket_start.month + 1)
        else:
            return
        
        # Aggregate metrics for each metric type
        for metric_name, buffer in self.metric_buffers.items():
            # Filter metrics within time bucket
            bucket_metrics = [
                snapshot for snapshot in buffer
                if bucket_start <= snapshot.timestamp < bucket_end
            ]
            
            if not bucket_metrics:
                continue
            
            # Calculate aggregations
            values = [m.metric_value for m in bucket_metrics]
            unique_users = set(m.user_id for m in bucket_metrics if m.user_id)
            unique_sessions = set(m.session_id for m in bucket_metrics if m.session_id)
            unique_providers = set(m.provider_type for m in bucket_metrics if m.provider_type)
            unique_models = set(m.model for m in bucket_metrics if m.model)
            
            aggregated = AggregatedMetric(
                metric_name=metric_name,
                aggregation_level=aggregation_level,
                time_bucket=bucket_start,
                count=len(values),
                sum_value=sum(values),
                avg_value=statistics.mean(values),
                min_value=min(values),
                max_value=max(values),
                std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
                p50=statistics.median(values),
                p95=statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                p99=statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values),
                unique_users=len(unique_users),
                unique_sessions=len(unique_sessions),
                unique_providers=len(unique_providers),
                unique_models=len(unique_models)
            )
            
            # Calculate error metrics
            if metric_name == "requests_total":
                failed_metrics = [m for m in bucket_metrics if m.metric_name == "requests_failed"]
                if failed_metrics:
                    aggregated.error_count = sum(m.metric_value for m in failed_metrics)
                    aggregated.error_rate = (aggregated.error_count / aggregated.sum_value) * 100
            
            # Store aggregated metric
            bucket_key = f"{aggregation_level.value}_{bucket_start.isoformat()}"
            if bucket_key not in self.aggregated_metrics:
                self.aggregated_metrics[bucket_key] = {}
            
            self.aggregated_metrics[bucket_key][metric_name] = aggregated
            
            # Persist to storage
            await self._persist_aggregated_metric(aggregated)
        
        self.collection_stats["aggregations_performed"] += 1
        logger.debug(f"Completed {aggregation_level.value} aggregation", bucket_start=bucket_start.isoformat())
    
    async def _persist_aggregated_metric(self, aggregated: AggregatedMetric) -> None:
        """Persist aggregated metric to storage."""
        try:
            # Save to JSON file
            date_str = aggregated.time_bucket.strftime("%Y%m%d")
            level_dir = self.storage_path / "aggregated" / aggregated.aggregation_level.value
            level_dir.mkdir(exist_ok=True)
            
            file_path = level_dir / f"{aggregated.metric_name}_{date_str}.json"
            
            # Load existing data or create new
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                data = {"metric_name": aggregated.metric_name, "aggregations": []}
            
            # Add new aggregation
            data["aggregations"].append(aggregated.to_dict())
            
            # Save back to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Save to ChromaDB if enabled
            if self.use_chromadb and "aggregated_metrics" in self.db_manager.collections:
                await self._save_to_chromadb(aggregated)
                
        except Exception as e:
            logger.error("Failed to persist aggregated metric", 
                        metric=aggregated.metric_name, error=str(e))
    
    async def _save_to_chromadb(self, aggregated: AggregatedMetric) -> None:
        """Save aggregated metric to ChromaDB."""
        try:
            collection = self.db_manager.collections["aggregated_metrics"]
            
            # Create document content
            content = (
                f"Metric: {aggregated.metric_name} "
                f"Level: {aggregated.aggregation_level.value} "
                f"Time: {aggregated.time_bucket.isoformat()} "
                f"Count: {aggregated.count} "
                f"Avg: {aggregated.avg_value:.4f} "
                f"Users: {aggregated.unique_users}"
            )
            
            metadata = {
                "metric_name": aggregated.metric_name,
                "aggregation_level": aggregated.aggregation_level.value,
                "time_bucket": aggregated.time_bucket.isoformat(),
                "count": aggregated.count,
                "avg_value": aggregated.avg_value,
                "unique_users": aggregated.unique_users,
                "created_at": aggregated.created_at.isoformat()
            }
            
            doc_id = f"{aggregated.metric_name}_{aggregated.aggregation_level.value}_{aggregated.time_bucket.strftime('%Y%m%d_%H%M')}"
            
            collection.upsert(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata]
            )
            
        except Exception as e:
            logger.error("Failed to save to ChromaDB", error=str(e))
    
    async def _run_retention_cleanup(self) -> None:
        """Run periodic retention cleanup."""
        while True:
            try:
                await self._perform_retention_cleanup()
                
                # Run cleanup daily
                await asyncio.sleep(86400)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in retention cleanup", error=str(e))
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _perform_retention_cleanup(self) -> None:
        """Perform data retention cleanup based on active policy."""
        current_time = datetime.now(timezone.utc)
        policy = self.active_policy
        
        cleanup_stats = {
            "files_deleted": 0,
            "files_compressed": 0,
            "bytes_freed": 0
        }
        
        # Clean up raw data
        await self._cleanup_raw_data(current_time, policy, cleanup_stats)
        
        # Clean up aggregated data
        await self._cleanup_aggregated_data(current_time, policy, cleanup_stats)
        
        # Compress old data
        await self._compress_old_data(current_time, policy, cleanup_stats)
        
        # Clean up backups
        if policy.backup_enabled:
            await self._cleanup_backups(current_time, policy, cleanup_stats)
        
        # Update collection stats
        self.collection_stats.update(cleanup_stats)
        self.collection_stats["last_cleanup"] = current_time.isoformat()
        
        logger.info("Retention cleanup completed", **cleanup_stats)
    
    async def _cleanup_raw_data(self, current_time: datetime, policy: RetentionPolicy, stats: Dict) -> None:
        """Clean up raw data files based on retention policy."""
        cutoff_time = current_time - self._retention_to_timedelta(policy.raw_data_retention)
        
        raw_dir = self.storage_path / "raw"
        if not raw_dir.exists():
            return
        
        for file_path in raw_dir.rglob("*.json"):
            try:
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                if file_time < cutoff_time:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    stats["files_deleted"] += 1
                    stats["bytes_freed"] += file_size
            except Exception as e:
                logger.warning(f"Failed to delete raw file {file_path}: {e}")
    
    async def _cleanup_aggregated_data(self, current_time: datetime, policy: RetentionPolicy, stats: Dict) -> None:
        """Clean up aggregated data files based on retention policy."""
        if policy.aggregated_data_retention == RetentionPeriod.PERMANENT:
            return
        
        cutoff_time = current_time - self._retention_to_timedelta(policy.aggregated_data_retention)
        
        aggregated_dir = self.storage_path / "aggregated"
        if not aggregated_dir.exists():
            return
        
        for file_path in aggregated_dir.rglob("*.json"):
            try:
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                if file_time < cutoff_time:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    stats["files_deleted"] += 1
                    stats["bytes_freed"] += file_size
            except Exception as e:
                logger.warning(f"Failed to delete aggregated file {file_path}: {e}")
    
    async def _compress_old_data(self, current_time: datetime, policy: RetentionPolicy, stats: Dict) -> None:
        """Compress old data files."""
        compress_before = current_time - policy.compression_after
        
        for data_dir in ["raw", "aggregated"]:
            dir_path = self.storage_path / data_dir
            if not dir_path.exists():
                continue
            
            for file_path in dir_path.rglob("*.json"):
                try:
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                    if file_time < compress_before:
                        await self._compress_file(file_path, policy.compression_type, stats)
                except Exception as e:
                    logger.warning(f"Failed to compress file {file_path}: {e}")
    
    async def _compress_file(self, file_path: Path, compression_type: CompressionType, stats: Dict) -> None:
        """Compress a single file."""
        if compression_type == CompressionType.NONE:
            return
        
        try:
            # Read original file
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Compress data
            if compression_type == CompressionType.GZIP:
                compressed_data = gzip.compress(data)
                extension = ".gz"
            elif compression_type == CompressionType.BZIP2:
                import bz2
                compressed_data = bz2.compress(data)
                extension = ".bz2"
            elif compression_type == CompressionType.LZMA:
                import lzma
                compressed_data = lzma.compress(data)
                extension = ".xz"
            else:
                return
            
            # Save compressed file
            compressed_path = self.storage_path / "compressed" / f"{file_path.stem}{extension}"
            compressed_path.parent.mkdir(exist_ok=True)
            
            with open(compressed_path, 'wb') as f:
                f.write(compressed_data)
            
            # Remove original file
            original_size = file_path.stat().st_size
            file_path.unlink()
            
            stats["files_compressed"] += 1
            stats["bytes_freed"] += original_size - len(compressed_data)
            
        except Exception as e:
            logger.error(f"Failed to compress {file_path}: {e}")
    
    async def _cleanup_backups(self, current_time: datetime, policy: RetentionPolicy, stats: Dict) -> None:
        """Clean up old backup files."""
        cutoff_time = current_time - self._retention_to_timedelta(policy.backup_retention)
        
        backup_dir = self.storage_path / "backups"
        if not backup_dir.exists():
            return
        
        for file_path in backup_dir.rglob("*"):
            try:
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                if file_time < cutoff_time:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    stats["files_deleted"] += 1
                    stats["bytes_freed"] += file_size
            except Exception as e:
                logger.warning(f"Failed to delete backup file {file_path}: {e}")
    
    def _retention_to_timedelta(self, retention: RetentionPeriod) -> timedelta:
        """Convert retention period to timedelta."""
        mapping = {
            RetentionPeriod.HOURS_1: timedelta(hours=1),
            RetentionPeriod.HOURS_6: timedelta(hours=6),
            RetentionPeriod.HOURS_24: timedelta(hours=24),
            RetentionPeriod.DAYS_7: timedelta(days=7),
            RetentionPeriod.DAYS_30: timedelta(days=30),
            RetentionPeriod.DAYS_90: timedelta(days=90),
            RetentionPeriod.DAYS_365: timedelta(days=365),
            RetentionPeriod.PERMANENT: timedelta(days=36500)  # 100 years
        }
        return mapping.get(retention, timedelta(days=30))
    
    async def get_metrics_summary(self, time_range: timedelta = None) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if time_range is None:
            time_range = timedelta(hours=24)
        
        cutoff_time = datetime.now(timezone.utc) - time_range
        
        summary = {
            "collection_stats": dict(self.collection_stats),
            "buffer_stats": {
                metric: len(buffer) for metric, buffer in self.metric_buffers.items()
            },
            "aggregation_stats": {
                "total_buckets": len(self.aggregated_metrics),
                "recent_buckets": sum(
                    1 for bucket_key in self.aggregated_metrics.keys()
                    if datetime.fromisoformat(bucket_key.split('_', 1)[1]) >= cutoff_time
                )
            },
            "storage_stats": await self._get_storage_stats(),
            "active_policy": {
                "policy_id": self.active_policy.policy_id,
                "name": self.active_policy.name,
                "raw_retention": self.active_policy.raw_data_retention.value,
                "aggregated_retention": self.active_policy.aggregated_data_retention.value
            }
        }
        
        return summary
    
    async def _get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        stats = {
            "total_size_bytes": 0,
            "raw_size_bytes": 0,
            "aggregated_size_bytes": 0,
            "compressed_size_bytes": 0,
            "backup_size_bytes": 0,
            "file_counts": {
                "raw": 0,
                "aggregated": 0,
                "compressed": 0,
                "backup": 0
            }
        }
        
        try:
            for subdir in ["raw", "aggregated", "compressed", "backups"]:
                dir_path = self.storage_path / subdir
                if dir_path.exists():
                    size = 0
                    count = 0
                    for file_path in dir_path.rglob("*"):
                        if file_path.is_file():
                            file_size = file_path.stat().st_size
                            size += file_size
                            count += 1
                    
                    stats[f"{subdir}_size_bytes"] = size
                    stats["file_counts"][subdir] = count
                    stats["total_size_bytes"] += size
                    
        except Exception as e:
            logger.error("Failed to get storage stats", error=str(e))
        
        return stats
    
    def set_retention_policy(self, policy_id: str) -> None:
        """Set active retention policy."""
        if policy_id in self.retention_policies:
            self.active_policy = self.retention_policies[policy_id]
            logger.info("Retention policy updated", policy_id=policy_id)
        else:
            raise ValueError(f"Unknown retention policy: {policy_id}")
    
    def add_retention_policy(self, policy: RetentionPolicy) -> None:
        """Add a new retention policy."""
        self.retention_policies[policy.policy_id] = policy
        logger.info("Retention policy added", policy_id=policy.policy_id)
    
    async def export_metrics(
        self,
        metric_names: List[str],
        start_time: datetime,
        end_time: datetime,
        aggregation_level: AggregationLevel = AggregationLevel.HOUR,
        format_type: str = "json"
    ) -> Union[Dict[str, Any], str]:
        """Export metrics data in specified format."""
        exported_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "aggregation_level": aggregation_level.value,
            "metrics": {}
        }
        
        # Collect requested metrics
        for metric_name in metric_names:
            metric_data = []
            
            # Find relevant aggregated metrics
            for bucket_key, bucket_metrics in self.aggregated_metrics.items():
                level, bucket_time_str = bucket_key.split('_', 1)
                if level != aggregation_level.value:
                    continue
                
                bucket_time = datetime.fromisoformat(bucket_time_str)
                if start_time <= bucket_time <= end_time:
                    if metric_name in bucket_metrics:
                        metric_data.append(bucket_metrics[metric_name].to_dict())
            
            exported_data["metrics"][metric_name] = metric_data
        
        if format_type == "json":
            return exported_data
        elif format_type == "csv":
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "metric_name", "time_bucket", "count", "sum", "avg", "min", "max",
                "unique_users", "unique_sessions", "error_rate"
            ])
            
            # Write data
            for metric_name, data_points in exported_data["metrics"].items():
                for point in data_points:
                    writer.writerow([
                        metric_name,
                        point["time_bucket"],
                        point["statistics"]["count"],
                        point["statistics"]["sum"],
                        point["statistics"]["avg"],
                        point["statistics"]["min"],
                        point["statistics"]["max"],
                        point["context"]["unique_users"],
                        point["context"]["unique_sessions"],
                        point["errors"]["error_rate"]
                    ])
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    async def cleanup(self) -> None:
        """Clean up resources and stop background tasks."""
        # Cancel all background tasks
        for task in self._collection_tasks.values():
            if not task.done():
                task.cancel()
        
        if self._retention_task and not self._retention_task.done():
            self._retention_task.cancel()
        
        # Wait for tasks to complete
        all_tasks = list(self._collection_tasks.values())
        if self._retention_task:
            all_tasks.append(self._retention_task)
        
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Flush remaining metrics to storage
        await self._perform_aggregation(AggregationLevel.HOUR)
        
        logger.info("Metrics collector cleanup completed")