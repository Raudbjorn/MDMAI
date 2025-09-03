"""Thread-safe concurrent access patterns for usage tracking systems."""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
from contextlib import asynccontextmanager, contextmanager
from concurrent.futures import ThreadPoolExecutor, Future
import weakref
from collections import defaultdict, deque
import queue

from ..ai_providers.models import UsageRecord
from config.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class LockType(Enum):
    """Types of locks for different concurrency patterns."""
    READ_WRITE = "read_write"
    EXCLUSIVE = "exclusive"
    SHARED = "shared"
    DISTRIBUTED = "distributed"


class ConcurrencyPattern(Enum):
    """Concurrency access patterns."""
    SINGLE_WRITER_MULTIPLE_READERS = "single_writer_multiple_readers"
    MULTIPLE_WRITERS_QUEUE = "multiple_writers_queue"
    PARTITION_BASED = "partition_based"
    OPTIMISTIC_LOCKING = "optimistic_locking"
    ACTOR_MODEL = "actor_model"


@dataclass
class LockMetrics:
    """Metrics for lock usage and contention."""
    lock_name: str
    total_acquisitions: int = 0
    total_contentions: int = 0
    avg_hold_time_ms: float = 0.0
    max_hold_time_ms: float = 0.0
    current_holders: int = 0
    waiting_count: int = 0


@dataclass 
class ConcurrentOperation:
    """Represents a concurrent operation."""
    operation_id: str
    operation_type: str
    resource_id: str
    user_id: str
    started_at: datetime
    timeout_seconds: float
    retry_count: int = 0
    max_retries: int = 3


class ReadWriteLock:
    """Advanced read-write lock with priority and timeout support."""
    
    def __init__(self, name: str = "unnamed", writer_priority: bool = True):
        self.name = name
        self.writer_priority = writer_priority
        
        # Internal locks
        self._readers_lock = threading.Lock()
        self._writers_lock = threading.Lock()
        self._readers_count = 0
        self._writers_count = 0
        self._writers_waiting = 0
        
        # Condition variables
        self._readers_cv = threading.Condition(self._readers_lock)
        self._writers_cv = threading.Condition(self._writers_lock)
        
        # Metrics
        self.metrics = LockMetrics(lock_name=name)
        self._hold_times: deque = deque(maxlen=1000)
    
    @contextmanager
    def read_lock(self, timeout: Optional[float] = None):
        """Acquire read lock."""
        acquired_at = time.time()
        
        try:
            if not self._acquire_read_lock(timeout):
                raise TimeoutError(f"Failed to acquire read lock on {self.name}")
            
            self.metrics.total_acquisitions += 1
            yield
            
        finally:
            self._release_read_lock()
            hold_time = (time.time() - acquired_at) * 1000
            self._update_hold_time_metrics(hold_time)
    
    @contextmanager
    def write_lock(self, timeout: Optional[float] = None):
        """Acquire write lock."""
        acquired_at = time.time()
        
        try:
            if not self._acquire_write_lock(timeout):
                raise TimeoutError(f"Failed to acquire write lock on {self.name}")
            
            self.metrics.total_acquisitions += 1
            yield
            
        finally:
            self._release_write_lock()
            hold_time = (time.time() - acquired_at) * 1000
            self._update_hold_time_metrics(hold_time)
    
    def _acquire_read_lock(self, timeout: Optional[float]) -> bool:
        """Acquire read lock with timeout."""
        with self._readers_cv:
            end_time = None if timeout is None else time.time() + timeout
            
            while True:
                # Check if we can acquire read lock
                if self._can_acquire_read_lock():
                    self._readers_count += 1
                    self.metrics.current_holders += 1
                    return True
                
                # Wait for condition
                self.metrics.total_contentions += 1
                
                if end_time is None:
                    self._readers_cv.wait()
                else:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        return False
                    self._readers_cv.wait(remaining)
    
    def _acquire_write_lock(self, timeout: Optional[float]) -> bool:
        """Acquire write lock with timeout."""
        with self._writers_cv:
            end_time = None if timeout is None else time.time() + timeout
            
            # Increment waiting count
            self._writers_waiting += 1
            
            try:
                while True:
                    # Check if we can acquire write lock
                    if self._can_acquire_write_lock():
                        self._writers_count = 1
                        self.metrics.current_holders = 1
                        return True
                    
                    # Wait for condition
                    self.metrics.total_contentions += 1
                    
                    if end_time is None:
                        self._writers_cv.wait()
                    else:
                        remaining = end_time - time.time()
                        if remaining <= 0:
                            return False
                        self._writers_cv.wait(remaining)
                        
            finally:
                self._writers_waiting -= 1
    
    def _can_acquire_read_lock(self) -> bool:
        """Check if read lock can be acquired."""
        # No writers and (no writer priority or no waiting writers)
        return (self._writers_count == 0 and 
                (not self.writer_priority or self._writers_waiting == 0))
    
    def _can_acquire_write_lock(self) -> bool:
        """Check if write lock can be acquired."""
        # No readers and no other writers
        return self._readers_count == 0 and self._writers_count == 0
    
    def _release_read_lock(self) -> None:
        """Release read lock."""
        with self._readers_cv:
            self._readers_count -= 1
            self.metrics.current_holders -= 1
            
            if self._readers_count == 0:
                # Notify waiting writers
                with self._writers_cv:
                    self._writers_cv.notify_all()
    
    def _release_write_lock(self) -> None:
        """Release write lock."""
        with self._writers_cv:
            self._writers_count = 0
            self.metrics.current_holders = 0
            
            # Notify all waiting threads
            self._writers_cv.notify_all()
            
        # Also notify waiting readers
        with self._readers_cv:
            self._readers_cv.notify_all()
    
    def _update_hold_time_metrics(self, hold_time_ms: float) -> None:
        """Update hold time metrics."""
        self._hold_times.append(hold_time_ms)
        
        # Update metrics
        self.metrics.avg_hold_time_ms = sum(self._hold_times) / len(self._hold_times)
        self.metrics.max_hold_time_ms = max(self.metrics.max_hold_time_ms, hold_time_ms)
    
    def get_metrics(self) -> LockMetrics:
        """Get lock metrics."""
        return LockMetrics(
            lock_name=self.metrics.lock_name,
            total_acquisitions=self.metrics.total_acquisitions,
            total_contentions=self.metrics.total_contentions,
            avg_hold_time_ms=self.metrics.avg_hold_time_ms,
            max_hold_time_ms=self.metrics.max_hold_time_ms,
            current_holders=self.metrics.current_holders,
            waiting_count=self._writers_waiting + max(0, len(threading.enumerate()) - self.metrics.current_holders)
        )


class AsyncReadWriteLock:
    """Async version of read-write lock."""
    
    def __init__(self, name: str = "async_unnamed"):
        self.name = name
        self._readers_count = 0
        self._writers_count = 0
        self._writers_waiting = 0
        
        # Events for coordination
        self._readers_event = asyncio.Event()
        self._writers_event = asyncio.Event()
        
        # Lock for atomic operations
        self._lock = asyncio.Lock()
        
        # Metrics
        self.metrics = LockMetrics(lock_name=name)
        
        # Initially, both readers and writers can proceed
        self._readers_event.set()
        self._writers_event.set()
    
    @asynccontextmanager
    async def read_lock(self, timeout: Optional[float] = None):
        """Acquire async read lock."""
        acquired_at = time.time()
        
        try:
            await self._acquire_read_lock(timeout)
            self.metrics.total_acquisitions += 1
            yield
            
        finally:
            await self._release_read_lock()
            hold_time = (time.time() - acquired_at) * 1000
            self._update_hold_time_metrics(hold_time)
    
    @asynccontextmanager
    async def write_lock(self, timeout: Optional[float] = None):
        """Acquire async write lock."""
        acquired_at = time.time()
        
        try:
            await self._acquire_write_lock(timeout)
            self.metrics.total_acquisitions += 1
            yield
            
        finally:
            await self._release_write_lock()
            hold_time = (time.time() - acquired_at) * 1000
            self._update_hold_time_metrics(hold_time)
    
    async def _acquire_read_lock(self, timeout: Optional[float]) -> None:
        """Acquire async read lock."""
        end_time = None if timeout is None else time.time() + timeout
        
        while True:
            async with self._lock:
                if self._writers_count == 0 and self._writers_waiting == 0:
                    self._readers_count += 1
                    self.metrics.current_holders += 1
                    if self._readers_count == 1:
                        self._writers_event.clear()
                    return
                
                self.metrics.total_contentions += 1
            
            # Wait for writers to finish
            try:
                if end_time is None:
                    await self._writers_event.wait()
                else:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError()
                    await asyncio.wait_for(self._writers_event.wait(), remaining)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Failed to acquire async read lock on {self.name}")
    
    async def _acquire_write_lock(self, timeout: Optional[float]) -> None:
        """Acquire async write lock."""
        end_time = None if timeout is None else time.time() + timeout
        
        async with self._lock:
            self._writers_waiting += 1
            
        try:
            while True:
                async with self._lock:
                    if self._readers_count == 0 and self._writers_count == 0:
                        self._writers_count = 1
                        self._writers_waiting -= 1
                        self.metrics.current_holders = 1
                        self._readers_event.clear()
                        return
                    
                    self.metrics.total_contentions += 1
                
                # Wait for readers and other writers to finish
                try:
                    if end_time is None:
                        await self._readers_event.wait()
                    else:
                        remaining = end_time - time.time()
                        if remaining <= 0:
                            raise asyncio.TimeoutError()
                        await asyncio.wait_for(self._readers_event.wait(), remaining)
                except asyncio.TimeoutError:
                    async with self._lock:
                        self._writers_waiting -= 1
                    raise TimeoutError(f"Failed to acquire async write lock on {self.name}")
                    
        except Exception:
            async with self._lock:
                if self._writers_waiting > 0:
                    self._writers_waiting -= 1
            raise
    
    async def _release_read_lock(self) -> None:
        """Release async read lock."""
        async with self._lock:
            self._readers_count -= 1
            self.metrics.current_holders -= 1
            
            if self._readers_count == 0:
                self._readers_event.set()
                self._writers_event.set()
    
    async def _release_write_lock(self) -> None:
        """Release async write lock."""
        async with self._lock:
            self._writers_count = 0
            self.metrics.current_holders = 0
            
            self._readers_event.set()
            self._writers_event.set()
    
    def _update_hold_time_metrics(self, hold_time_ms: float) -> None:
        """Update hold time metrics."""
        # Use a simplified average calculation for async context
        alpha = 0.1
        self.metrics.avg_hold_time_ms = (
            alpha * hold_time_ms + (1 - alpha) * self.metrics.avg_hold_time_ms
        )
        self.metrics.max_hold_time_ms = max(self.metrics.max_hold_time_ms, hold_time_ms)


class PartitionedLockManager:
    """Lock manager with partitioned locking for better concurrency."""
    
    def __init__(self, partition_count: int = 16):
        self.partition_count = partition_count
        self.sync_locks = [ReadWriteLock(f"partition_{i}") for i in range(partition_count)]
        self.async_locks = [AsyncReadWriteLock(f"async_partition_{i}") for i in range(partition_count)]
        
    def _get_partition(self, key: str) -> int:
        """Get partition number for a key."""
        return hash(key) % self.partition_count
    
    @contextmanager
    def read_lock(self, key: str, timeout: Optional[float] = None):
        """Acquire read lock for a key."""
        partition = self._get_partition(key)
        with self.sync_locks[partition].read_lock(timeout):
            yield
    
    @contextmanager
    def write_lock(self, key: str, timeout: Optional[float] = None):
        """Acquire write lock for a key."""
        partition = self._get_partition(key)
        with self.sync_locks[partition].write_lock(timeout):
            yield
    
    @asynccontextmanager
    async def async_read_lock(self, key: str, timeout: Optional[float] = None):
        """Acquire async read lock for a key."""
        partition = self._get_partition(key)
        async with self.async_locks[partition].read_lock(timeout):
            yield
    
    @asynccontextmanager
    async def async_write_lock(self, key: str, timeout: Optional[float] = None):
        """Acquire async write lock for a key."""
        partition = self._get_partition(key)
        async with self.async_locks[partition].write_lock(timeout):
            yield
    
    def get_all_metrics(self) -> List[LockMetrics]:
        """Get metrics for all partitions."""
        return [lock.get_metrics() for lock in self.sync_locks]


class OptimisticLockManager:
    """Optimistic locking with version control."""
    
    def __init__(self):
        self.versions: Dict[str, int] = defaultdict(int)
        self.version_lock = threading.Lock()
        
    def get_version(self, resource_id: str) -> int:
        """Get current version of a resource."""
        with self.version_lock:
            return self.versions[resource_id]
    
    def try_update(self, resource_id: str, expected_version: int, update_func: Callable) -> bool:
        """Try to update resource with optimistic locking."""
        with self.version_lock:
            current_version = self.versions[resource_id]
            
            if current_version != expected_version:
                # Version mismatch - update failed
                return False
            
            # Execute update
            try:
                update_func()
                self.versions[resource_id] = current_version + 1
                return True
            except Exception as e:
                logger.error(f"Optimistic update failed for {resource_id}: {e}")
                return False
    
    async def async_try_update(
        self, 
        resource_id: str, 
        expected_version: int, 
        update_func: Callable
    ) -> bool:
        """Async version of optimistic update."""
        # Use thread pool for thread-safe version checking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.try_update, resource_id, expected_version, update_func
        )


class ConcurrentQueue(Generic[T]):
    """Thread-safe queue with priority and timeout support."""
    
    def __init__(self, maxsize: int = 0, priority_func: Optional[Callable[[T], int]] = None):
        self.maxsize = maxsize
        self.priority_func = priority_func
        
        if priority_func:
            self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize)
        else:
            self._queue: queue.Queue = queue.Queue(maxsize)
        
        self._item_counter = 0
        self._counter_lock = threading.Lock()
    
    def put(self, item: T, timeout: Optional[float] = None) -> bool:
        """Put item in queue."""
        try:
            if self.priority_func:
                with self._counter_lock:
                    priority = -self.priority_func(item)  # Negative for highest first
                    self._queue.put((priority, self._item_counter, item), timeout=timeout)
                    self._item_counter += 1
            else:
                self._queue.put(item, timeout=timeout)
            return True
        except queue.Full:
            return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """Get item from queue."""
        try:
            if self.priority_func:
                priority, counter, item = self._queue.get(timeout=timeout)
                return item
            else:
                return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def qsize(self) -> int:
        """Get queue size."""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()


class ActorSystem:
    """Simple actor model implementation for concurrent operations."""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.actors: Dict[str, 'Actor'] = {}
        self.running = False
    
    def start(self) -> None:
        """Start the actor system."""
        self.running = True
        logger.info("Actor system started")
    
    def stop(self) -> None:
        """Stop the actor system."""
        self.running = False
        
        # Stop all actors
        for actor in self.actors.values():
            actor.stop()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        logger.info("Actor system stopped")
    
    def create_actor(self, actor_id: str, actor_class: type, *args, **kwargs) -> 'Actor':
        """Create and register an actor."""
        if actor_id in self.actors:
            raise ValueError(f"Actor {actor_id} already exists")
        
        actor = actor_class(actor_id, self, *args, **kwargs)
        self.actors[actor_id] = actor
        actor.start()
        
        return actor
    
    def get_actor(self, actor_id: str) -> Optional['Actor']:
        """Get actor by ID."""
        return self.actors.get(actor_id)
    
    def send_message(self, actor_id: str, message: Any) -> bool:
        """Send message to actor."""
        actor = self.actors.get(actor_id)
        if actor:
            return actor.send_message(message)
        return False


class Actor:
    """Base actor class for actor model concurrency."""
    
    def __init__(self, actor_id: str, actor_system: ActorSystem):
        self.actor_id = actor_id
        self.actor_system = actor_system
        self.message_queue = ConcurrentQueue()
        self.running = False
        self.worker_future: Optional[Future] = None
    
    def start(self) -> None:
        """Start the actor."""
        if self.running:
            return
        
        self.running = True
        self.worker_future = self.actor_system.executor.submit(self._message_loop)
    
    def stop(self) -> None:
        """Stop the actor."""
        self.running = False
        if self.worker_future:
            self.worker_future.cancel()
    
    def send_message(self, message: Any) -> bool:
        """Send message to actor."""
        return self.message_queue.put(message, timeout=1.0)
    
    def _message_loop(self) -> None:
        """Main message processing loop."""
        while self.running:
            try:
                message = self.message_queue.get(timeout=1.0)
                if message is not None:
                    self.handle_message(message)
            except Exception as e:
                logger.error(f"Error in actor {self.actor_id}: {e}")
    
    def handle_message(self, message: Any) -> None:
        """Handle incoming message - override in subclasses."""
        logger.debug(f"Actor {self.actor_id} received message: {message}")


class UsageRecordActor(Actor):
    """Actor for processing usage records concurrently."""
    
    def __init__(self, actor_id: str, actor_system: ActorSystem, storage_backend):
        super().__init__(actor_id, actor_system)
        self.storage_backend = storage_backend
        self.processed_count = 0
        self.error_count = 0
    
    def handle_message(self, message: Any) -> None:
        """Handle usage record processing message."""
        try:
            if isinstance(message, dict) and message.get("type") == "store_usage_record":
                usage_record = message.get("record")
                if usage_record:
                    # Process the usage record
                    asyncio.run(self.storage_backend.store_usage_record(usage_record))
                    self.processed_count += 1
                    
            elif isinstance(message, dict) and message.get("type") == "get_stats":
                # Return processing statistics
                return {
                    "actor_id": self.actor_id,
                    "processed_count": self.processed_count,
                    "error_count": self.error_count
                }
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error processing usage record in actor {self.actor_id}: {e}")


class ConcurrencyManager:
    """Main concurrency management coordinator."""
    
    def __init__(self, pattern: ConcurrencyPattern = ConcurrencyPattern.PARTITION_BASED):
        self.pattern = pattern
        
        # Initialize concurrency primitives based on pattern
        if pattern == ConcurrencyPattern.SINGLE_WRITER_MULTIPLE_READERS:
            self.rw_lock = ReadWriteLock("main_usage_lock")
            self.async_rw_lock = AsyncReadWriteLock("async_main_usage_lock")
            
        elif pattern == ConcurrencyPattern.PARTITION_BASED:
            self.partition_manager = PartitionedLockManager(partition_count=16)
            
        elif pattern == ConcurrencyPattern.OPTIMISTIC_LOCKING:
            self.optimistic_manager = OptimisticLockManager()
            
        elif pattern == ConcurrencyPattern.ACTOR_MODEL:
            self.actor_system = ActorSystem(max_workers=20)
            
        elif pattern == ConcurrencyPattern.MULTIPLE_WRITERS_QUEUE:
            self.write_queue = ConcurrentQueue(
                maxsize=10000,
                priority_func=lambda op: op.get("priority", 0)
            )
        
        # Metrics and monitoring
        self.operation_metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_operation_time_ms": 0.0,
            "lock_contentions": 0
        }
        self.metrics_lock = threading.Lock()
        
        # Background monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start(self) -> None:
        """Start the concurrency manager."""
        if self.running:
            return
        
        self.running = True
        
        if self.pattern == ConcurrencyPattern.ACTOR_MODEL:
            self.actor_system.start()
        
        # Start monitoring
        self.monitoring_task = asyncio.create_task(self._monitoring_worker())
        
        logger.info(f"Concurrency manager started with pattern: {self.pattern.value}")
    
    async def stop(self) -> None:
        """Stop the concurrency manager."""
        if not self.running:
            return
        
        self.running = False
        
        if self.pattern == ConcurrencyPattern.ACTOR_MODEL:
            self.actor_system.stop()
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Concurrency manager stopped")
    
    @asynccontextmanager
    async def read_access(self, resource_id: str, timeout: Optional[float] = None):
        """Acquire read access to a resource."""
        start_time = time.time()
        
        try:
            if self.pattern == ConcurrencyPattern.SINGLE_WRITER_MULTIPLE_READERS:
                async with self.async_rw_lock.read_lock(timeout):
                    yield
                    
            elif self.pattern == ConcurrencyPattern.PARTITION_BASED:
                async with self.partition_manager.async_read_lock(resource_id, timeout):
                    yield
                    
            else:
                # For other patterns, use simple async lock
                yield
                
        finally:
            self._record_operation_time(time.time() - start_time)
    
    @asynccontextmanager
    async def write_access(self, resource_id: str, timeout: Optional[float] = None):
        """Acquire write access to a resource."""
        start_time = time.time()
        
        try:
            if self.pattern == ConcurrencyPattern.SINGLE_WRITER_MULTIPLE_READERS:
                async with self.async_rw_lock.write_lock(timeout):
                    yield
                    
            elif self.pattern == ConcurrencyPattern.PARTITION_BASED:
                async with self.partition_manager.async_write_lock(resource_id, timeout):
                    yield
                    
            elif self.pattern == ConcurrencyPattern.OPTIMISTIC_LOCKING:
                # Return version for optimistic update
                version = self.optimistic_manager.get_version(resource_id)
                yield version
                
            else:
                # For other patterns, use simple async lock
                yield
                
        finally:
            self._record_operation_time(time.time() - start_time)
    
    async def execute_with_retry(
        self,
        operation: ConcurrentOperation,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with retry logic."""
        start_time = time.time()
        
        for attempt in range(operation.max_retries + 1):
            try:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > operation.timeout_seconds:
                    raise TimeoutError(f"Operation {operation.operation_id} timed out")
                
                # Execute operation
                if self.pattern == ConcurrencyPattern.OPTIMISTIC_LOCKING:
                    return await self._execute_optimistic(operation, operation_func, *args, **kwargs)
                else:
                    return await operation_func(*args, **kwargs)
                    
            except (TimeoutError, Exception) as e:
                operation.retry_count = attempt
                
                if attempt == operation.max_retries:
                    with self.metrics_lock:
                        self.metrics["failed_operations"] += 1
                    logger.error(f"Operation {operation.operation_id} failed after {attempt + 1} attempts: {e}")
                    raise
                
                # Exponential backoff
                backoff_time = min(2 ** attempt, 10)  # Max 10 seconds
                await asyncio.sleep(backoff_time)
        
        with self.metrics_lock:
            self.metrics["successful_operations"] += 1
        
        return None
    
    async def _execute_optimistic(
        self, 
        operation: ConcurrentOperation, 
        operation_func: Callable,
        *args, 
        **kwargs
    ) -> Any:
        """Execute operation with optimistic locking."""
        resource_id = operation.resource_id
        version = self.optimistic_manager.get_version(resource_id)
        
        # Execute operation
        result = await operation_func(*args, **kwargs)
        
        # Try to commit with version check
        def commit_func():
            # This would contain the actual commit logic
            pass
        
        success = await self.optimistic_manager.async_try_update(
            resource_id, version, commit_func
        )
        
        if not success:
            raise Exception("Optimistic lock conflict - version mismatch")
        
        return result
    
    def _record_operation_time(self, time_seconds: float) -> None:
        """Record operation execution time."""
        time_ms = time_seconds * 1000
        
        with self.metrics_lock:
            total_ops = self.operation_metrics["total_operations"]
            current_avg = self.operation_metrics["avg_operation_time_ms"]
            
            # Update metrics
            self.operation_metrics["total_operations"] += 1
            self.operation_metrics["avg_operation_time_ms"] = (
                (current_avg * total_ops + time_ms) / (total_ops + 1)
            )
    
    async def _monitoring_worker(self) -> None:
        """Background worker for monitoring concurrency health."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for high lock contention
                if self.pattern == ConcurrencyPattern.PARTITION_BASED:
                    metrics = self.partition_manager.get_all_metrics()
                    high_contention_partitions = [
                        m for m in metrics 
                        if m.total_contentions > m.total_acquisitions * 0.1
                    ]
                    
                    if high_contention_partitions:
                        logger.warning("High lock contention detected",
                                     partitions=len(high_contention_partitions))
                
                # Check operation performance
                with self.metrics_lock:
                    avg_time = self.operation_metrics["avg_operation_time_ms"]
                    if avg_time > 1000:  # More than 1 second
                        logger.warning("Slow operations detected",
                                     avg_time_ms=avg_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def get_concurrency_metrics(self) -> Dict[str, Any]:
        """Get comprehensive concurrency metrics."""
        with self.metrics_lock:
            base_metrics = dict(self.operation_metrics)
        
        base_metrics["pattern"] = self.pattern.value
        base_metrics["running"] = self.running
        
        # Add pattern-specific metrics
        if self.pattern == ConcurrencyPattern.PARTITION_BASED:
            partition_metrics = self.partition_manager.get_all_metrics()
            base_metrics["partition_metrics"] = [
                {
                    "partition": i,
                    "total_acquisitions": m.total_acquisitions,
                    "contentions": m.total_contentions,
                    "avg_hold_time_ms": m.avg_hold_time_ms,
                    "current_holders": m.current_holders
                }
                for i, m in enumerate(partition_metrics)
            ]
        
        elif self.pattern == ConcurrencyPattern.ACTOR_MODEL and hasattr(self, 'actor_system'):
            base_metrics["actor_count"] = len(self.actor_system.actors)
        
        return base_metrics
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False