"""Advanced state synchronization between ChromaDB, JSON storage, and PostgreSQL."""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
import json
from concurrent.futures import ThreadPoolExecutor

from ..ai_providers.models import UsageRecord, ProviderType
from ..context.persistence import ContextPersistenceLayer
from .chroma_extensions import UsageTrackingChromaExtensions, UsageAnalyticsType
from .json_persistence import JsonPersistenceManager
from config.logging_config import get_logger

logger = get_logger(__name__)


class SyncStatus(Enum):
    """Synchronization status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"


class StorageBackend(Enum):
    """Storage backend types."""
    POSTGRESQL = "postgresql"
    CHROMADB = "chromadb"
    JSON_FILES = "json_files"


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    LATEST_WINS = "latest_wins"
    MERGE = "merge"
    MANUAL = "manual"
    SOURCE_PRIORITY = "source_priority"


@dataclass
class SyncOperation:
    """Represents a synchronization operation."""
    operation_id: str
    operation_type: str  # insert, update, delete
    record_id: str
    user_id: str
    source_backend: StorageBackend
    target_backends: List[StorageBackend]
    data: Dict[str, Any]
    timestamp: datetime
    status: SyncStatus
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    dependencies: List[str] = None  # Other operation IDs this depends on
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "record_id": self.record_id,
            "user_id": self.user_id,
            "source_backend": self.source_backend.value,
            "target_backends": [b.value for b in self.target_backends],
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "error_message": self.error_message,
            "dependencies": self.dependencies or []
        }


@dataclass
class SyncConflict:
    """Represents a synchronization conflict."""
    conflict_id: str
    record_id: str
    user_id: str
    backends_involved: List[StorageBackend]
    conflicting_data: Dict[StorageBackend, Dict[str, Any]]
    detected_at: datetime
    resolution_strategy: ConflictResolution
    resolved: bool = False
    resolution_data: Optional[Dict[str, Any]] = None


@dataclass
class SyncConfiguration:
    """Configuration for state synchronization."""
    # Sync intervals
    real_time_sync: bool = True
    batch_sync_interval: int = 300  # seconds
    full_sync_interval: int = 3600  # seconds
    
    # Conflict resolution
    default_conflict_resolution: ConflictResolution = ConflictResolution.LATEST_WINS
    backend_priority: List[StorageBackend] = None
    
    # Performance settings
    max_concurrent_operations: int = 10
    operation_batch_size: int = 100
    sync_queue_size: int = 10000
    
    # Reliability settings
    enable_checksums: bool = True
    enable_conflict_detection: bool = True
    sync_validation_enabled: bool = True
    
    # Recovery settings
    failed_operation_retention_hours: int = 24
    auto_retry_failed_operations: bool = True
    
    def __post_init__(self):
        """Set default backend priority if not provided."""
        if self.backend_priority is None:
            self.backend_priority = [
                StorageBackend.POSTGRESQL,
                StorageBackend.JSON_FILES,
                StorageBackend.CHROMADB
            ]


class StateSynchronizationManager:
    """Advanced state synchronization manager for usage tracking data."""
    
    def __init__(
        self,
        postgres_persistence: ContextPersistenceLayer,
        chroma_extensions: UsageTrackingChromaExtensions,
        json_persistence: JsonPersistenceManager,
        config: Optional[SyncConfiguration] = None
    ):
        self.postgres_persistence = postgres_persistence
        self.chroma_extensions = chroma_extensions
        self.json_persistence = json_persistence
        self.config = config or SyncConfiguration()
        
        # Synchronization queues and state
        self.sync_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.sync_queue_size)
        self.failed_operations: deque = deque(maxlen=1000)
        self.active_conflicts: Dict[str, SyncConflict] = {}
        
        # Background tasks
        self._sync_tasks: List[asyncio.Task] = []
        self._running = False
        
        # Performance tracking
        self.sync_metrics = {
            "operations_processed": 0,
            "operations_successful": 0,
            "operations_failed": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "avg_sync_time": 0.0,
            "last_full_sync": None,
            "last_batch_sync": None
        }
        
        # Thread safety
        self.operation_locks: Dict[str, asyncio.Lock] = {}
        self.metrics_lock = asyncio.Lock()
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        logger.info("State synchronization manager initialized")
    
    async def start(self) -> None:
        """Start the synchronization manager."""
        if self._running:
            return
        
        self._running = True
        
        # Start background synchronization tasks
        if self.config.real_time_sync:
            self._sync_tasks.append(asyncio.create_task(self._real_time_sync_worker()))
        
        self._sync_tasks.append(asyncio.create_task(self._batch_sync_worker()))
        self._sync_tasks.append(asyncio.create_task(self._full_sync_worker()))
        self._sync_tasks.append(asyncio.create_task(self._conflict_resolution_worker()))
        
        if self.config.auto_retry_failed_operations:
            self._sync_tasks.append(asyncio.create_task(self._retry_failed_operations_worker()))
        
        logger.info("State synchronization manager started")
    
    async def stop(self) -> None:
        """Stop the synchronization manager."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all background tasks
        for task in self._sync_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._sync_tasks:
            await asyncio.gather(*self._sync_tasks, return_exceptions=True)
        
        # Process remaining operations in queue
        await self._drain_sync_queue()
        
        logger.info("State synchronization manager stopped")
    
    async def sync_usage_record(
        self,
        record: UsageRecord,
        source_backend: StorageBackend,
        target_backends: Optional[List[StorageBackend]] = None,
        force_immediate: bool = False
    ) -> str:
        """Synchronize a usage record across storage backends."""
        if target_backends is None:
            # Sync to all other backends
            target_backends = [b for b in StorageBackend if b != source_backend]
        
        # Create sync operation
        operation = SyncOperation(
            operation_id=f"sync_{record.request_id}_{int(time.time() * 1000)}",
            operation_type="insert" if record.request_id not in await self._get_existing_record_ids() else "update",
            record_id=record.request_id,
            user_id=record.metadata.get("user_id", "unknown"),
            source_backend=source_backend,
            target_backends=target_backends,
            data=self._usage_record_to_dict(record),
            timestamp=datetime.now(),
            status=SyncStatus.PENDING
        )
        
        if force_immediate or not self.config.real_time_sync:
            # Execute immediately
            await self._execute_sync_operation(operation)
            return operation.operation_id
        else:
            # Queue for background processing
            try:
                await self.sync_queue.put(operation)
                return operation.operation_id
            except asyncio.QueueFull:
                logger.warning("Sync queue full, executing immediately", operation_id=operation.operation_id)
                await self._execute_sync_operation(operation)
                return operation.operation_id
    
    async def _execute_sync_operation(self, operation: SyncOperation) -> bool:
        """Execute a single synchronization operation."""
        operation_start = time.time()
        operation.status = SyncStatus.IN_PROGRESS
        
        try:
            # Get or create lock for this record
            if operation.record_id not in self.operation_locks:
                self.operation_locks[operation.record_id] = asyncio.Lock()
            
            async with self.operation_locks[operation.record_id]:
                success = await self._sync_to_backends(operation)
                
                if success:
                    operation.status = SyncStatus.COMPLETED
                    await self._emit_event("sync_completed", operation)
                    
                    # Update metrics
                    async with self.metrics_lock:
                        self.sync_metrics["operations_successful"] += 1
                        
                        execution_time = time.time() - operation_start
                        self.sync_metrics["avg_sync_time"] = (
                            self.sync_metrics["avg_sync_time"] * 0.9 + execution_time * 0.1
                        )
                else:
                    operation.status = SyncStatus.FAILED
                    self.failed_operations.append(operation)
                    await self._emit_event("sync_failed", operation)
                    
                    async with self.metrics_lock:
                        self.sync_metrics["operations_failed"] += 1
                
                async with self.metrics_lock:
                    self.sync_metrics["operations_processed"] += 1
                
                return success
        
        except Exception as e:
            operation.status = SyncStatus.FAILED
            operation.error_message = str(e)
            self.failed_operations.append(operation)
            
            logger.error(
                "Sync operation failed",
                operation_id=operation.operation_id,
                record_id=operation.record_id,
                error=str(e)
            )
            
            await self._emit_event("sync_error", operation)
            return False
    
    async def _sync_to_backends(self, operation: SyncOperation) -> bool:
        """Sync data to target backends."""
        success_count = 0
        total_backends = len(operation.target_backends)
        
        # Convert operation data to UsageRecord
        usage_record = self._dict_to_usage_record(operation.data)
        
        for backend in operation.target_backends:
            try:
                if backend == StorageBackend.POSTGRESQL:
                    await self._sync_to_postgresql(usage_record, operation)
                elif backend == StorageBackend.CHROMADB:
                    await self._sync_to_chromadb(usage_record, operation)
                elif backend == StorageBackend.JSON_FILES:
                    await self._sync_to_json_files(usage_record, operation)
                
                success_count += 1
                
                logger.debug(
                    "Synced to backend",
                    backend=backend.value,
                    operation_id=operation.operation_id,
                    record_id=operation.record_id
                )
            
            except Exception as e:
                logger.error(
                    "Failed to sync to backend",
                    backend=backend.value,
                    operation_id=operation.operation_id,
                    record_id=operation.record_id,
                    error=str(e)
                )
        
        # Consider successful if majority of backends succeeded
        return success_count >= (total_backends // 2 + 1)
    
    async def _sync_to_postgresql(self, record: UsageRecord, operation: SyncOperation) -> None:
        """Sync usage record to PostgreSQL."""
        # Use existing context persistence layer
        # Create a context representation of the usage record
        context_data = {
            "usage_record": self._usage_record_to_dict(record),
            "sync_metadata": {
                "operation_id": operation.operation_id,
                "sync_timestamp": operation.timestamp.isoformat(),
                "source_backend": operation.source_backend.value
            }
        }
        
        # Store in a usage tracking context type
        if operation.operation_type == "insert":
            # Create new context for usage record
            from ..context.models import Context, ContextType, ContextState
            
            usage_context = Context(
                context_id=f"usage_{record.request_id}",
                context_type=ContextType.PROVIDER_SPECIFIC,
                title=f"Usage Record {record.request_id}",
                description=f"Usage tracking for {record.provider_type.value} - {record.model}",
                data=context_data,
                metadata={
                    "usage_tracking": True,
                    "user_id": record.metadata.get("user_id"),
                    "provider_type": record.provider_type.value,
                    "model": record.model,
                    "cost": record.cost,
                    "timestamp": record.timestamp.isoformat()
                },
                owner_id=record.metadata.get("user_id"),
                state=ContextState.ACTIVE
            )
            
            await self.postgres_persistence.create_context(usage_context)
        
        else:  # update
            # Update existing context
            updates = {
                "data": context_data,
                "last_modified": operation.timestamp
            }
            
            await self.postgres_persistence.update_context(
                f"usage_{record.request_id}",
                updates,
                user_id=record.metadata.get("user_id")
            )
    
    async def _sync_to_chromadb(self, record: UsageRecord, operation: SyncOperation) -> None:
        """Sync usage record to ChromaDB for analytics."""
        # Store various analytics types
        analytics_types = [
            UsageAnalyticsType.USAGE_PATTERNS,
            UsageAnalyticsType.PERFORMANCE_METRICS
        ]
        
        # Generate analysis results based on the record
        analysis_results = {
            "patterns": {
                "frequency": self._calculate_usage_frequency(record),
                "time_of_day": record.timestamp.hour,
                "usage_type": self._classify_usage_type(record)
            },
            "performance": {
                "latency_ms": record.latency_ms,
                "throughput": self._calculate_throughput(record),
                "error_rate": 0.0 if record.success else 1.0,
                "percentile_rank": self._calculate_latency_percentile(record)
            },
            "optimization": {
                "cost_per_token": record.cost / max(record.input_tokens + record.output_tokens, 1),
                "efficiency_score": self._calculate_efficiency_score(record),
                "potential_savings": self._calculate_potential_savings(record)
            }
        }
        
        # Store analytics for each type
        for analytics_type in analytics_types:
            await self.chroma_extensions.store_usage_analytics(
                record, analytics_type, analysis_results
            )
    
    async def _sync_to_json_files(self, record: UsageRecord, operation: SyncOperation) -> None:
        """Sync usage record to JSON files."""
        await self.json_persistence.store_usage_record(record)
    
    def _calculate_usage_frequency(self, record: UsageRecord) -> str:
        """Calculate usage frequency classification."""
        # Simplified implementation - would use historical data in practice
        hour = record.timestamp.hour
        if 9 <= hour <= 17:
            return "business_hours"
        elif 18 <= hour <= 22:
            return "evening"
        else:
            return "off_hours"
    
    def _classify_usage_type(self, record: UsageRecord) -> str:
        """Classify the type of usage based on record characteristics."""
        if record.input_tokens > 1000:
            return "heavy_processing"
        elif record.latency_ms > 2000:
            return "complex_request"
        elif record.cost > 0.01:
            return "premium_usage"
        else:
            return "standard_usage"
    
    def _calculate_throughput(self, record: UsageRecord) -> float:
        """Calculate throughput (tokens per second)."""
        if record.latency_ms > 0:
            total_tokens = record.input_tokens + record.output_tokens
            return (total_tokens * 1000) / record.latency_ms
        return 0.0
    
    def _calculate_latency_percentile(self, record: UsageRecord) -> float:
        """Calculate latency percentile rank (simplified)."""
        # This would use historical data in practice
        if record.latency_ms < 500:
            return 95.0
        elif record.latency_ms < 1000:
            return 75.0
        elif record.latency_ms < 2000:
            return 50.0
        else:
            return 25.0
    
    def _calculate_efficiency_score(self, record: UsageRecord) -> float:
        """Calculate efficiency score (0-100)."""
        base_score = 100.0
        
        # Penalize high latency
        if record.latency_ms > 1000:
            base_score -= (record.latency_ms - 1000) / 100
        
        # Penalize high cost per token
        cost_per_token = record.cost / max(record.input_tokens + record.output_tokens, 1)
        if cost_per_token > 0.001:
            base_score -= (cost_per_token - 0.001) * 10000
        
        # Penalize failures
        if not record.success:
            base_score -= 50
        
        return max(0.0, min(100.0, base_score))
    
    def _calculate_potential_savings(self, record: UsageRecord) -> float:
        """Calculate potential cost savings with optimization."""
        # Simplified calculation - would use more sophisticated analysis in practice
        base_cost = record.cost
        potential_savings = 0.0
        
        # If using expensive model, suggest cheaper alternative
        if "gpt-4" in record.model.lower():
            potential_savings += base_cost * 0.7  # 70% cheaper with gpt-3.5
        
        # If high latency, suggest optimization
        if record.latency_ms > 2000:
            potential_savings += base_cost * 0.1  # 10% from optimization
        
        return potential_savings
    
    async def detect_conflicts(self, record_id: str) -> Optional[SyncConflict]:
        """Detect conflicts between different storage backends for a record."""
        if not self.config.enable_conflict_detection:
            return None
        
        try:
            # Fetch record from all backends
            records = {}
            
            # PostgreSQL
            try:
                pg_record = await self._fetch_from_postgresql(record_id)
                if pg_record:
                    records[StorageBackend.POSTGRESQL] = pg_record
            except Exception as e:
                logger.debug(f"Could not fetch from PostgreSQL: {e}")
            
            # JSON Files
            try:
                json_records = await self.json_persistence.query_usage_records(
                    limit=1, offset=0
                )
                json_record = next((r for r in json_records if r.get("request_id") == record_id), None)
                if json_record:
                    records[StorageBackend.JSON_FILES] = json_record
            except Exception as e:
                logger.debug(f"Could not fetch from JSON files: {e}")
            
            # ChromaDB (more complex as it's analytics-based)
            # Skip for now as ChromaDB contains processed analytics, not raw records
            
            if len(records) < 2:
                return None  # No conflict possible with fewer than 2 records
            
            # Compare records for conflicts
            conflicts_found = False
            reference_record = list(records.values())[0]
            
            for backend, record in records.items():
                if not self._records_match(reference_record, record):
                    conflicts_found = True
                    break
            
            if conflicts_found:
                conflict = SyncConflict(
                    conflict_id=f"conflict_{record_id}_{int(time.time())}",
                    record_id=record_id,
                    user_id=reference_record.get("metadata", {}).get("user_id", "unknown"),
                    backends_involved=list(records.keys()),
                    conflicting_data=records,
                    detected_at=datetime.now(),
                    resolution_strategy=self.config.default_conflict_resolution
                )
                
                self.active_conflicts[conflict.conflict_id] = conflict
                
                async with self.metrics_lock:
                    self.sync_metrics["conflicts_detected"] += 1
                
                await self._emit_event("conflict_detected", conflict)
                
                logger.warning(
                    "Conflict detected",
                    conflict_id=conflict.conflict_id,
                    record_id=record_id,
                    backends=len(records)
                )
                
                return conflict
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to detect conflicts for record {record_id}: {e}")
            return None
    
    async def resolve_conflict(self, conflict: SyncConflict) -> bool:
        """Resolve a synchronization conflict."""
        try:
            resolved_data = None
            
            if conflict.resolution_strategy == ConflictResolution.LATEST_WINS:
                # Find the record with the latest timestamp
                latest_backend = None
                latest_timestamp = None
                
                for backend, data in conflict.conflicting_data.items():
                    timestamp_str = data.get("timestamp", "")
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if latest_timestamp is None or timestamp > latest_timestamp:
                                latest_timestamp = timestamp
                                latest_backend = backend
                                resolved_data = data
                        except ValueError:
                            continue
            
            elif conflict.resolution_strategy == ConflictResolution.SOURCE_PRIORITY:
                # Use priority defined in configuration
                for backend in self.config.backend_priority:
                    if backend in conflict.conflicting_data:
                        resolved_data = conflict.conflicting_data[backend]
                        break
            
            elif conflict.resolution_strategy == ConflictResolution.MERGE:
                # Merge data from all sources (simplified implementation)
                resolved_data = {}
                for backend, data in conflict.conflicting_data.items():
                    for key, value in data.items():
                        if key not in resolved_data or value is not None:
                            resolved_data[key] = value
            
            if resolved_data:
                # Create sync operation to propagate resolved data
                usage_record = self._dict_to_usage_record(resolved_data)
                
                # Sync to all backends involved in the conflict
                await self.sync_usage_record(
                    usage_record,
                    source_backend=StorageBackend.POSTGRESQL,  # Treat as authoritative source
                    target_backends=conflict.backends_involved,
                    force_immediate=True
                )
                
                # Mark conflict as resolved
                conflict.resolved = True
                conflict.resolution_data = resolved_data
                
                async with self.metrics_lock:
                    self.sync_metrics["conflicts_resolved"] += 1
                
                await self._emit_event("conflict_resolved", conflict)
                
                logger.info(
                    "Conflict resolved",
                    conflict_id=conflict.conflict_id,
                    strategy=conflict.resolution_strategy.value
                )
                
                return True
            
            else:
                logger.error(
                    "Failed to resolve conflict - no resolution data",
                    conflict_id=conflict.conflict_id
                )
                return False
        
        except Exception as e:
            logger.error(
                "Failed to resolve conflict",
                conflict_id=conflict.conflict_id,
                error=str(e)
            )
            return False
    
    def _records_match(self, record1: Dict[str, Any], record2: Dict[str, Any]) -> bool:
        """Check if two records match (ignoring minor differences)."""
        # Fields to compare for conflicts
        important_fields = [
            "provider_type", "model", "input_tokens", "output_tokens",
            "cost", "success", "timestamp"
        ]
        
        for field in important_fields:
            val1 = record1.get(field)
            val2 = record2.get(field)
            
            # Handle floating point comparison for cost
            if field == "cost" and val1 is not None and val2 is not None:
                if abs(float(val1) - float(val2)) > 0.0001:
                    return False
            elif val1 != val2:
                return False
        
        return True
    
    async def _fetch_from_postgresql(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a usage record from PostgreSQL."""
        try:
            context_id = f"usage_{record_id}"
            context = await self.postgres_persistence.get_context(context_id)
            
            if context and "usage_record" in context.data:
                return context.data["usage_record"]
            
            return None
        
        except Exception as e:
            logger.debug(f"Failed to fetch from PostgreSQL: {e}")
            return None
    
    async def _get_existing_record_ids(self) -> Set[str]:
        """Get set of existing record IDs across all backends."""
        # Simplified implementation - would be more comprehensive in practice
        record_ids = set()
        
        # This would query all backends to get existing record IDs
        # For now, return empty set
        
        return record_ids
    
    def _usage_record_to_dict(self, record: UsageRecord) -> Dict[str, Any]:
        """Convert UsageRecord to dictionary."""
        return {
            "request_id": record.request_id,
            "session_id": record.session_id,
            "provider_type": record.provider_type.value,
            "model": record.model,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "cost": record.cost,
            "latency_ms": record.latency_ms,
            "success": record.success,
            "error_message": record.error_message,
            "timestamp": record.timestamp.isoformat(),
            "metadata": record.metadata
        }
    
    def _dict_to_usage_record(self, data: Dict[str, Any]) -> UsageRecord:
        """Convert dictionary to UsageRecord."""
        return UsageRecord(
            request_id=data["request_id"],
            session_id=data.get("session_id"),
            provider_type=ProviderType(data["provider_type"]),
            model=data["model"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            cost=data["cost"],
            latency_ms=data["latency_ms"],
            success=data["success"],
            error_message=data.get("error_message"),
            timestamp=datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00')),
            metadata=data.get("metadata", {})
        )
    
    async def _real_time_sync_worker(self) -> None:
        """Background worker for real-time synchronization."""
        while self._running:
            try:
                # Process sync operations from queue
                try:
                    operation = await asyncio.wait_for(self.sync_queue.get(), timeout=1.0)
                    await self._execute_sync_operation(operation)
                    self.sync_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in real-time sync worker", error=str(e))
                await asyncio.sleep(1)
    
    async def _batch_sync_worker(self) -> None:
        """Background worker for batch synchronization."""
        while self._running:
            try:
                await asyncio.sleep(self.config.batch_sync_interval)
                
                if not self._running:
                    break
                
                # Perform batch synchronization
                await self._perform_batch_sync()
                
                async with self.metrics_lock:
                    self.sync_metrics["last_batch_sync"] = datetime.now().isoformat()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in batch sync worker", error=str(e))
                await asyncio.sleep(30)
    
    async def _full_sync_worker(self) -> None:
        """Background worker for full synchronization."""
        while self._running:
            try:
                await asyncio.sleep(self.config.full_sync_interval)
                
                if not self._running:
                    break
                
                # Perform full synchronization
                await self._perform_full_sync()
                
                async with self.metrics_lock:
                    self.sync_metrics["last_full_sync"] = datetime.now().isoformat()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in full sync worker", error=str(e))
                await asyncio.sleep(300)
    
    async def _conflict_resolution_worker(self) -> None:
        """Background worker for conflict resolution."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check for conflicts every 30 seconds
                
                if not self._running:
                    break
                
                # Process unresolved conflicts
                unresolved_conflicts = [
                    conflict for conflict in self.active_conflicts.values()
                    if not conflict.resolved
                ]
                
                for conflict in unresolved_conflicts:
                    await self.resolve_conflict(conflict)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in conflict resolution worker", error=str(e))
                await asyncio.sleep(60)
    
    async def _retry_failed_operations_worker(self) -> None:
        """Background worker to retry failed operations."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Retry every 5 minutes
                
                if not self._running:
                    break
                
                # Retry failed operations
                failed_ops_to_retry = []
                current_time = datetime.now()
                
                for operation in list(self.failed_operations):
                    # Only retry if within retention period and hasn't exceeded max retries
                    age_hours = (current_time - operation.timestamp).total_seconds() / 3600
                    
                    if (age_hours < self.config.failed_operation_retention_hours and
                        operation.retry_count < operation.max_retries):
                        failed_ops_to_retry.append(operation)
                    
                    # Remove old or exhausted operations
                    if (age_hours >= self.config.failed_operation_retention_hours or
                        operation.retry_count >= operation.max_retries):
                        self.failed_operations.remove(operation)
                
                # Retry operations
                for operation in failed_ops_to_retry:
                    operation.retry_count += 1
                    operation.status = SyncStatus.PENDING
                    operation.error_message = None
                    
                    await self._execute_sync_operation(operation)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in retry failed operations worker", error=str(e))
                await asyncio.sleep(300)
    
    async def _perform_batch_sync(self) -> None:
        """Perform batch synchronization of recent data."""
        logger.info("Starting batch sync")
        
        # This would implement batch synchronization logic
        # For example, syncing data from the last batch interval
        
        pass
    
    async def _perform_full_sync(self) -> None:
        """Perform full synchronization across all backends."""
        logger.info("Starting full sync")
        
        # This would implement comprehensive synchronization
        # Comparing all data across backends and resolving differences
        
        pass
    
    async def _drain_sync_queue(self) -> None:
        """Process remaining operations in sync queue."""
        while not self.sync_queue.empty():
            try:
                operation = await asyncio.wait_for(self.sync_queue.get(), timeout=0.1)
                await self._execute_sync_operation(operation)
                self.sync_queue.task_done()
            except asyncio.TimeoutError:
                break
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add event handler for sync events."""
        self.event_handlers[event_type].append(handler)
    
    async def _emit_event(self, event_type: str, data: Any) -> None:
        """Emit synchronization event to registered handlers."""
        for handler in self.event_handlers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, data)
                else:
                    handler(event_type, data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
    
    async def get_sync_status(self) -> Dict[str, Any]:
        """Get comprehensive synchronization status."""
        async with self.metrics_lock:
            return {
                "running": self._running,
                "queue_size": self.sync_queue.qsize(),
                "failed_operations": len(self.failed_operations),
                "active_conflicts": len(self.active_conflicts),
                "metrics": dict(self.sync_metrics),
                "configuration": {
                    "real_time_sync": self.config.real_time_sync,
                    "batch_sync_interval": self.config.batch_sync_interval,
                    "full_sync_interval": self.config.full_sync_interval,
                    "default_conflict_resolution": self.config.default_conflict_resolution.value
                }
            }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False