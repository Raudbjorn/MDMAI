"""Parallel processing system for concurrent operations."""

import asyncio
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import multiprocessing as mp
from functools import partial
import time
import weakref
from contextlib import asynccontextmanager
import threading

from config.logging_config import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Status of a parallel processing task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingTask:
    """Represents a task to be processed in parallel."""
    id: str
    type: str
    data: Any
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "retry_count": self.retry_count,
            "error": self.error,
        }


@dataclass
class ResourceLimits:
    """Resource limits for parallel processing."""
    max_workers: int = field(default_factory=lambda: mp.cpu_count())
    max_memory_mb: int = 1024
    max_queue_size: int = 1000
    task_timeout: int = 300  # seconds
    
    def __post_init__(self):
        """Validate resource limits."""
        if self.max_workers <= 0:
            self.max_workers = mp.cpu_count()
        if self.max_memory_mb <= 0:
            self.max_memory_mb = 1024
        if self.task_timeout <= 0:
            self.task_timeout = 300


class ParallelProcessor:
    """Manages parallel processing of tasks with resource management."""
    
    def __init__(self, resource_limits: Optional[ResourceLimits] = None):
        """Initialize the parallel processor."""
        self.limits = resource_limits or ResourceLimits()
        self.executor = None
        self.async_executor = None
        self.tasks: Dict[str, ProcessingTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=self.limits.max_queue_size)
        self._shutdown = False
        self._workers = []
        self._monitor_task = None
        self._task_lock = asyncio.Lock()
        self._executor_lock = threading.Lock()
        self._initialized = False
        self._finalizer = weakref.finalize(self, self._cleanup_resources)
        
    async def initialize(self) -> None:
        """Initialize the parallel processing system."""
        async with self._task_lock:
            if self._initialized:
                logger.debug("Processor already initialized")
                return
                
            if self._shutdown:
                raise RuntimeError("Processor has been shut down")
            
            with self._executor_lock:
                # Create thread pool for CPU-bound tasks
                self.executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.limits.max_workers,
                    thread_name_prefix="ttrpg-worker"
                )
                
                # Create process pool for heavy CPU-bound tasks
                # Use spawn context to avoid fork issues
                    max_workers=self.limits.max_process_workers,
                    mp_context=ctx
                )
            
            # Start worker tasks
            for i in range(self.limits.max_workers):
                worker = asyncio.create_task(self._worker(f"worker-{i}"))
                self._workers.append(worker)
            
            # Start resource monitor
            self._monitor_task = asyncio.create_task(self._resource_monitor())
            
            self._initialized = True
            logger.info(
                f"Parallel processor initialized with {self.limits.max_workers} workers"
            )
    
    async def _worker(self, worker_id: str) -> None:
        """Worker task that processes items from the queue."""
        while not self._shutdown:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Process the task with timeout
                try:
                    await asyncio.wait_for(
                        self._process_task(task, worker_id),
                        timeout=self.limits.task_timeout
                    )
                except asyncio.TimeoutError:
                    task.status = TaskStatus.FAILED
                    task.error = f"Task timed out after {self.limits.task_timeout} seconds"
                    logger.error(f"Task {task.id} timed out")
                
            except asyncio.TimeoutError:
                continue  # Check shutdown flag
            except asyncio.CancelledError:
                break  # Worker cancelled
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}", exc_info=True)
    
    async def _process_task(self, task: ProcessingTask, worker_id: str) -> None:
        """Process a single task."""
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.utcnow()
        
        try:
            # Validate task data
            if not task.data:
                raise ValueError("Task data is empty")
            
            # Determine processing method based on task type
            if task.type == "pdf_processing":
                result = await self._process_pdf_task(task.data)
            elif task.type == "embedding_generation":
                result = await self._process_embedding_task(task.data)
            elif task.type == "search":
                result = await self._process_search_task(task.data)
            elif task.type == "batch_operation":
                result = await self._process_batch_task(task.data)
            else:
                raise ValueError(f"Unknown task type: {task.type}")
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            raise
        except Exception as e:
            task.error = str(e)
            task.retry_count += 1
            
            if task.retry_count < task.max_retries:
                # Exponential backoff for retries
                backoff = min(2 ** task.retry_count, 30)  # Max 30 seconds
                await asyncio.sleep(backoff)
                
                # Retry the task
                task.status = TaskStatus.PENDING
                await self.task_queue.put(task)
                logger.warning(
                    f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries}) after {backoff}s backoff"
                )
            else:
                task.status = TaskStatus.FAILED
                logger.error(f"Task {task.id} failed after {task.max_retries} retries: {e}")
        
        finally:
            task.end_time = datetime.utcnow()
    
    async def _process_pdf_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process PDF in parallel."""
        # Validate required fields
        if "pdf_path" not in data:
            raise ValueError("pdf_path is required for PDF processing")
        
        pdf_path = Path(data["pdf_path"])
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        from src.pdf_processing.pdf_parser import PDFParser, PDFProcessingError
        from src.pdf_processing.content_chunker import ContentChunker
        
        loop = asyncio.get_event_loop()
        
        try:
            # Parse PDF in thread pool
            parser = PDFParser()
            pdf_content = await loop.run_in_executor(
                self.executor,
                parser.parse_pdf,
                str(pdf_path)
            )
            
            # Chunk content in parallel
            chunker = ContentChunker()
            chunks = await loop.run_in_executor(
                self.executor,
                chunker.chunk_content,
                pdf_content["text"],
                pdf_content.get("metadata", {})
            )
            
            return {
                "pdf_path": str(pdf_path),
                "chunks": len(chunks),
                "pages": pdf_content.get("num_pages", 0),
                "content": chunks,
            }
        except PDFProcessingError as e:
            logger.error(f"PDF processing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing PDF: {e}")
            raise
    
    async def _process_embedding_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings in parallel."""
        from src.pdf_processing.embedding_generator import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        chunks = data.get("chunks", [])
        
        # Process embeddings in batches
        batch_size = 10
        embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = await asyncio.gather(*[
                self._generate_single_embedding(generator, chunk)
                for chunk in batch
            ])
            embeddings.extend(batch_embeddings)
        
        return {
            "embeddings": embeddings,
            "count": len(embeddings),
        }
    
    async def _generate_single_embedding(self, generator, chunk: str) -> List[float]:
        """Generate embedding for a single chunk."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            generator.generate_embedding,
            chunk
        )
    
    async def _process_search_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process search operations in parallel."""
        from src.search.search_service import SearchService
        
        service = SearchService()
        queries = data.get("queries", [])
        
        # Process multiple searches in parallel
        results = await asyncio.gather(*[
            service.search(
                query=q.get("query"),
                collection=q.get("collection", "rulebooks"),
                max_results=q.get("max_results", 5)
            )
            for q in queries
        ])
        
        return {
            "results": results,
            "query_count": len(queries),
        }
    
    async def _process_batch_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process batch operations in parallel."""
        operations = data.get("operations", [])
        batch_size = data.get("batch_size", 10)
        
        results = []
        errors = []
        
        # Process in batches
        for i in range(0, len(operations), batch_size):
            batch = operations[i:i + batch_size]
            
            # Process batch operations in parallel
            batch_results = await asyncio.gather(*[
                self._execute_operation(op)
                for op in batch
            ], return_exceptions=True)
            
            # Collect results and errors
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    errors.append({
                        "operation": batch[j],
                        "error": str(result),
                    })
                else:
                    results.append(result)
        
        return {
            "processed": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
        }
    
    async def _execute_operation(self, operation: Dict[str, Any]) -> Any:
        """Execute a single operation."""
        op_type = operation.get("type")
        op_data = operation.get("data")
        
        if op_type == "add_document":
            from src.core.database import get_db_manager
            db = get_db_manager()
            return db.add_document(
                collection_name=op_data["collection"],
                document_id=op_data["id"],
                content=op_data["content"],
                metadata=op_data["metadata"],
            )
        else:
            raise ValueError(f"Unknown operation type: {op_type}")
    
    async def _resource_monitor(self) -> None:
        """Monitor resource usage and adjust processing."""
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not available, resource monitoring disabled")
            return
        
        process = psutil.Process()
        consecutive_high_memory = 0
        
        while not self._shutdown:
            try:
                # Check memory usage
                memory_mb = process.memory_info().rss / (1024 * 1024)
                cpu_percent = process.cpu_percent(interval=1)
                
                if memory_mb > self.limits.max_memory_mb:
                    consecutive_high_memory += 1
                    logger.warning(
                        f"Memory usage ({memory_mb:.1f}MB) exceeds limit "
                        f"({self.limits.max_memory_mb}MB) - occurrence {consecutive_high_memory}"
                    )
                    
                    # Reduce worker count if memory is consistently high
                    if consecutive_high_memory >= 3 and len(self._workers) > 1:
                        worker = self._workers.pop()
                        worker.cancel()
                        logger.info("Reduced worker count due to sustained memory pressure")
                        consecutive_high_memory = 0
                else:
                    consecutive_high_memory = 0
                
                # Check CPU usage
                if cpu_percent > 90:
                    logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                
                # Check queue size
                queue_size = self.task_queue.qsize()
                if queue_size > self.limits.max_queue_size * 0.9:
                    logger.warning(
                        f"Task queue nearly full: {queue_size}/{self.limits.max_queue_size}"
                    )
                
                # Dynamic adjustment interval based on load
                if queue_size > self.limits.max_queue_size * 0.5:
                    await asyncio.sleep(5)  # Check more frequently under load
                else:
                    await asyncio.sleep(10)  # Normal interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitor error: {str(e)}")
                await asyncio.sleep(10)
    
    async def submit_task(
        self,
        task_type: str,
        data: Any,
        task_id: Optional[str] = None,
    ) -> ProcessingTask:
        """Submit a task for parallel processing."""
        if self._shutdown:
            raise RuntimeError("Processor has been shut down")
        
        # Create task
        import uuid
        task = ProcessingTask(
            id=task_id or str(uuid.uuid4()),
            type=task_type,
            data=data,
        )
        
        # Add to tracking
        self.tasks[task.id] = task
        
        # Queue for processing
        await self.task_queue.put(task)
        
        logger.debug(f"Submitted task {task.id} of type {task_type}")
        return task
    
    async def submit_batch(
        self,
        tasks: List[Tuple[str, Any]],
    ) -> List[ProcessingTask]:
        """Submit multiple tasks for parallel processing."""
        submitted = []
        
        for task_type, data in tasks:
            task = await self.submit_task(task_type, data)
            submitted.append(task)
        
        return submitted
    
    async def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> ProcessingTask:
        """Wait for a task to complete."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        start_time = time.time()
        
        while task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} timed out")
            
            await asyncio.sleep(0.1)
        
        return task
    
    async def wait_for_all(
        self,
        task_ids: List[str],
        timeout: Optional[float] = None,
    ) -> List[ProcessingTask]:
        """Wait for multiple tasks to complete."""
        tasks = await asyncio.gather(*[
            self.wait_for_task(task_id, timeout)
            for task_id in task_ids
        ])
        return tasks
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task."""
        task = self.tasks.get(task_id)
        return task.status if task else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = {
            "total_tasks": len(self.tasks),
            "pending": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING),
            "running": sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING),
            "completed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED),
            "cancelled": sum(1 for t in self.tasks.values() if t.status == TaskStatus.CANCELLED),
            "queue_size": self.task_queue.qsize(),
            "worker_count": len(self._workers),
        }
        
        # Calculate average processing time
        completed_tasks = [
            t for t in self.tasks.values()
            if t.status == TaskStatus.COMPLETED and t.start_time and t.end_time
        ]
        
        if completed_tasks:
            total_time = sum(
                (t.end_time - t.start_time).total_seconds()
                for t in completed_tasks
            )
            stats["avg_processing_time"] = total_time / len(completed_tasks)
        
        return stats
    
    async def shutdown(self) -> None:
        """Shutdown the parallel processor."""
        async with self._task_lock:
            if self._shutdown:
                return
            
            self._shutdown = True
            
            # Cancel all pending tasks
            for task in self.tasks.values():
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    task.status = TaskStatus.CANCELLED
            
            # Cancel workers
            for worker in self._workers:
                worker.cancel()
            
            # Wait for workers to finish with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Workers did not shutdown gracefully")
            
            # Cancel monitor
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await asyncio.wait_for(self._monitor_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Shutdown executors
            with self._executor_lock:
                if self.executor:
                    self.executor.shutdown(wait=True, cancel_futures=True)
                if self.async_executor:
                    self.async_executor.shutdown(wait=True, cancel_futures=True)
            
            self._initialized = False
            logger.info("Parallel processor shut down")
    
    @staticmethod
    def _cleanup_resources():
        """Cleanup resources when object is garbage collected."""
        logger.debug("Parallel processor resources cleaned up")
    
    @asynccontextmanager
    async def managed_processor(self):
        """Context manager for automatic initialization and cleanup."""
        await self.initialize()
        try:
            yield self
        finally:
            await self.shutdown()


class ParallelPDFProcessor:
    """Specialized parallel processor for PDF operations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize PDF parallel processor."""
        self.processor = ParallelProcessor(
            ResourceLimits(max_workers=max_workers or mp.cpu_count())
        )
    
    async def process_multiple_pdfs(
        self,
        pdf_files: List[Dict[str, Any]],
        generate_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """Process multiple PDFs in parallel."""
        await self.processor.initialize()
        
        try:
            # Submit PDF processing tasks
            pdf_tasks = []
            for pdf_info in pdf_files:
                task = await self.processor.submit_task(
                    "pdf_processing",
                    pdf_info
                )
                pdf_tasks.append(task)
            
            # Wait for PDF processing to complete
            completed_tasks = await self.processor.wait_for_all(
                [t.id for t in pdf_tasks]
            )
            
            # Generate embeddings if requested
            if generate_embeddings:
                embedding_tasks = []
                for task in completed_tasks:
                    if task.status == TaskStatus.COMPLETED and task.result:
                        emb_task = await self.processor.submit_task(
                            "embedding_generation",
                            {"chunks": task.result.get("content", [])}
                        )
                        embedding_tasks.append(emb_task)
                
                # Wait for embeddings
                await self.processor.wait_for_all(
                    [t.id for t in embedding_tasks]
                )
            
            # Collect results
            stats = self.processor.get_statistics()
            
            return {
                "processed": stats["completed"],
                "failed": stats["failed"],
                "statistics": stats,
                "results": [
                    t.to_dict() for t in completed_tasks
                ],
            }
            
        finally:
            await self.processor.shutdown()


class ResourceManager:
    """Manages system resources for parallel processing."""
    
    def __init__(self):
        """Initialize resource manager."""
        self.cpu_count = mp.cpu_count()
        self.memory_limit_mb = 2048  # Default 2GB limit
        self.active_processors: Dict[str, ParallelProcessor] = {}
    
    def get_optimal_workers(self, task_type: str) -> int:
        """Get optimal number of workers for a task type."""
        if task_type in ["pdf_processing", "embedding_generation"]:
            # CPU-intensive tasks
            return max(1, self.cpu_count - 1)
        elif task_type == "search":
            # I/O-bound tasks
            return min(self.cpu_count * 2, 16)
        else:
            return self.cpu_count
    
    def allocate_processor(
        self,
        processor_id: str,
        task_type: str,
    ) -> ParallelProcessor:
        """Allocate a new parallel processor."""
        if processor_id in self.active_processors:
            raise ValueError(f"Processor {processor_id} already exists")
        
        # Determine resource limits
        limits = ResourceLimits(
            max_workers=self.get_optimal_workers(task_type),
            max_memory_mb=self.memory_limit_mb // max(1, len(self.active_processors) + 1),
        )
        
        # Create processor
        processor = ParallelProcessor(limits)
        self.active_processors[processor_id] = processor
        
        logger.info(
            f"Allocated processor {processor_id} with {limits.max_workers} workers"
        )
        
        return processor
    
    async def release_processor(self, processor_id: str) -> None:
        """Release a parallel processor."""
        processor = self.active_processors.pop(processor_id, None)
        if processor:
            await processor.shutdown()
            logger.info(f"Released processor {processor_id}")
    
    async def shutdown_all(self) -> None:
        """Shutdown all active processors."""
        for processor_id in list(self.active_processors.keys()):
            await self.release_processor(processor_id)
        logger.info("All processors shut down")