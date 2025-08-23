"""Database performance optimization for ChromaDB."""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class IndexMetrics:
    """Metrics for index optimization."""

    collection_name: str
    document_count: int
    index_time: float
    query_time: float
    memory_usage: int
    last_optimized: datetime = field(default_factory=datetime.utcnow)
    optimization_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.document_count,
            "index_time": self.index_time,
            "query_time": self.query_time,
            "memory_usage": self.memory_usage,
            "last_optimized": self.last_optimized.isoformat(),
            "optimization_count": self.optimization_count,
        }


@dataclass
class QueryMetrics:
    """Metrics for query performance."""

    query: str
    collection: str
    execution_time: float
    result_count: int
    cache_hit: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query[:100],  # Truncate long queries
            "collection": self.collection,
            "execution_time": self.execution_time,
            "result_count": self.result_count,
            "cache_hit": self.cache_hit,
            "timestamp": self.timestamp.isoformat(),
        }


class DatabaseOptimizer:
    """Optimizes ChromaDB performance through various strategies."""

    def __init__(self, db_manager: Any):
        """Initialize the optimizer."""
        self.db = db_manager
        self.index_metrics: Dict[str, IndexMetrics] = {}
        self.query_metrics: List[QueryMetrics] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._optimization_thresholds = {
            "document_count": 10000,  # Optimize when docs exceed this
            "query_time": 1.0,  # Optimize when avg query time exceeds this (seconds)
            "days_since_optimization": 7,  # Re-optimize after this many days
        }
        self._shutdown = False

    def shutdown(self):
        """Shutdown the optimizer and cleanup resources."""
        if not self._shutdown:
            self._shutdown = True
            self.executor.shutdown(wait=True)
            logger.info("Database optimizer shutdown complete")

    async def optimize_indices(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize indices for better search performance.

        Args:
            collection_name: Specific collection to optimize, or None for all

        Returns:
            Optimization results
        """
        start_time = time.time()
        results = {"optimized": [], "errors": [], "metrics": {}}

        collections_to_optimize = (
            [collection_name] if collection_name else list(self.db.collections.keys())
        )

        for coll_name in collections_to_optimize:
            try:
                # Check if optimization is needed
                if not self._needs_optimization(coll_name):
                    logger.debug(f"Collection {coll_name} does not need optimization")
                    continue

                # Perform optimization
                metrics = await self._optimize_collection_index(coll_name)

                # Update metrics
                self.index_metrics[coll_name] = metrics
                results["optimized"].append(coll_name)
                results["metrics"][coll_name] = metrics.to_dict()

                logger.info(
                    f"Optimized collection {coll_name}",
                    document_count=metrics.document_count,
                    optimization_time=metrics.index_time,
                )

            except Exception as e:
                logger.error(f"Failed to optimize {coll_name}: {str(e)}")
                results["errors"].append(
                    {
                        "collection": coll_name,
                        "error": str(e),
                    }
                )

        results["total_time"] = time.time() - start_time
        return results

    def _needs_optimization(self, collection_name: str) -> bool:
        """Check if a collection needs optimization."""
        collection = self.db.collections.get(collection_name)
        if not collection:
            return False

        # Get current document count
        doc_count = collection.count()

        # Check if we have previous metrics
        if collection_name in self.index_metrics:
            metrics = self.index_metrics[collection_name]

            # Check various conditions
            days_since_last = (datetime.utcnow() - metrics.last_optimized).days

            if days_since_last >= self._optimization_thresholds["days_since_optimization"]:
                return True

            if doc_count > self._optimization_thresholds["document_count"]:
                # Significant growth since last optimization
                if doc_count > metrics.document_count * 1.5:
                    return True

            # Check recent query performance
            recent_queries = [
                q
                for q in self.query_metrics[-100:]  # Last 100 queries
                if q.collection == collection_name
            ]
            if recent_queries:
                avg_time = sum(q.execution_time for q in recent_queries) / len(recent_queries)
                if avg_time > self._optimization_thresholds["query_time"]:
                    return True
        else:
            # No previous metrics, optimize if large collection
            if doc_count > self._optimization_thresholds["document_count"] / 2:
                return True

        return False

    async def _optimize_collection_index(self, collection_name: str) -> IndexMetrics:
        """Optimize a specific collection's index."""
        start_time = time.time()
        collection = self.db.collections[collection_name]

        # Get collection statistics
        doc_count = collection.count()

        # ChromaDB specific optimizations
        # Note: ChromaDB handles most indexing internally, but we can:

        # 1. Trigger a compaction/cleanup if supported
        try:
            # This is a placeholder - actual implementation depends on ChromaDB version
            if hasattr(collection, "_compact"):
                await asyncio.get_event_loop().run_in_executor(self.executor, collection._compact)
        except Exception as e:
            logger.debug(f"Compaction not available: {str(e)}")

        # 2. Re-index by rebuilding the collection (if necessary)
        # This would involve backing up and restoring data
        # Only do this for collections with performance issues

        # 3. Measure current query performance
        query_time = await self._measure_query_performance(collection_name)

        # Calculate memory usage (approximate)
        # Prevent integer overflow and provide reasonable estimate
        memory_usage = min(doc_count * 1024, 2**31 - 1)  # Rough estimate: 1KB per doc, cap at 2GB

        index_time = time.time() - start_time

        return IndexMetrics(
            collection_name=collection_name,
            document_count=doc_count,
            index_time=index_time,
            query_time=query_time,
            memory_usage=memory_usage,
            optimization_count=self.index_metrics.get(
                collection_name, IndexMetrics(collection_name, 0, 0, 0, 0)
            ).optimization_count
            + 1,
        )

    async def _measure_query_performance(self, collection_name: str) -> float:
        """Measure average query performance for a collection."""
        collection = self.db.collections[collection_name]

        # Run sample queries
        test_queries = [
            "test query performance",
            "sample search term",
            "benchmark query",
        ]

        total_time = 0
        for query in test_queries:
            start = time.time()
            try:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    collection.query,
                    [query],  # query_texts
                    1,  # n_results
                )
            except Exception as e:
                logger.warning("Benchmark query failed", error=str(e), exc_info=True)
            total_time += time.time() - start

        return total_time / len(test_queries)

    async def optimize_queries(self, queries: List[str]) -> Dict[str, Any]:
        """
        Optimize a list of queries for better performance.

        Args:
            queries: List of queries to optimize

        Returns:
            Optimization suggestions and rewritten queries
        """
        results = {"original": [], "optimized": [], "suggestions": []}

        for query in queries:
            # Analyze query structure
            analysis = self._analyze_query(query)

            # Generate optimized version
            optimized = self._rewrite_query(query, analysis)

            results["original"].append(query)
            results["optimized"].append(optimized)
            results["suggestions"].append(analysis["suggestions"])

        return results

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query for optimization opportunities."""
        # Validate and sanitize query
        if query is None:
            return {
                "suggestions": ["Invalid query"],
                "length": 0,
                "word_count": 0,
                "has_special_chars": False,
            }
        if not isinstance(query, str):
            return {
                "suggestions": ["Invalid query"],
                "length": 0,
                "word_count": 0,
                "has_special_chars": False,
            }
        if not query:
            return {
                "suggestions": ["Invalid query"],
                "length": 0,
                "word_count": 0,
                "has_special_chars": False,
            }

        # Limit query length to prevent DOS
        MAX_QUERY_LENGTH = 5000
        if len(query) > MAX_QUERY_LENGTH:
            query = query[:MAX_QUERY_LENGTH]

        analysis = {
            "length": len(query),
            "word_count": len(query.split()),
            "has_special_chars": any(c in query for c in ["*", "?", "[", "]"]),
            "suggestions": [],
        }

        # Provide optimization suggestions
        if analysis["length"] > 500:
            analysis["suggestions"].append(
                "Query is very long - consider breaking into multiple searches"
            )

        if analysis["word_count"] > 20:
            analysis["suggestions"].append("Many words in query - consider focusing on key terms")

        if analysis["has_special_chars"]:
            analysis["suggestions"].append(
                "Special characters may slow search - use simple terms when possible"
            )

        return analysis

    def _rewrite_query(self, query: str, analysis: Dict[str, Any]) -> str:
        """Rewrite a query for better performance."""
        optimized = query

        # Apply optimizations based on analysis
        if analysis["word_count"] > 20:
            # Keep only the most important words (simple heuristic)
            words = query.split()
            # Remove common stop words
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
            important_words = [w for w in words if w.lower() not in stop_words]
            if len(important_words) < len(words):
                optimized = " ".join(important_words[:15])  # Limit to 15 words

        return optimized

    async def batch_process(
        self,
        operations: List[Tuple[str, Dict[str, Any]]],
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Process database operations in batches for better performance.

        Args:
            operations: List of (operation_type, operation_data) tuples
            batch_size: Number of operations per batch

        Returns:
            Processing results
        """
        results = {
            "total": len(operations),
            "processed": 0,
            "failed": 0,
            "batch_times": [],
        }

        # Process in batches
        for i in range(0, len(operations), batch_size):
            batch = operations[i : i + batch_size]
            batch_start = time.time()

            # Process batch concurrently
            tasks = []
            for op_type, op_data in batch:
                if op_type == "add":
                    task = self._batch_add(op_data)
                elif op_type == "update":
                    task = self._batch_update(op_data)
                elif op_type == "delete":
                    task = self._batch_delete(op_data)
                else:
                    continue
                tasks.append(task)

            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count results
            for result in batch_results:
                if isinstance(result, Exception):
                    results["failed"] += 1
                    logger.error(f"Batch operation failed: {str(result)}")
                else:
                    results["processed"] += 1

            batch_time = time.time() - batch_start
            results["batch_times"].append(batch_time)

            logger.debug(
                f"Processed batch {i // batch_size + 1}",
                size=len(batch),
                time=batch_time,
            )

        # Calculate statistics
        if results["batch_times"]:
            results["avg_batch_time"] = sum(results["batch_times"]) / len(results["batch_times"])
            results["total_time"] = sum(results["batch_times"])

        return results

    async def _batch_add(self, data: Dict[str, Any]) -> None:
        """Add document in batch operation."""
        try:
            # Validate required fields
            if not all(k in data for k in ["collection", "id", "content", "metadata"]):
                raise ValueError(f"Missing required fields in batch add data")

            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.db.add_document,
                data["collection"],
                data["id"],
                data["content"],
                data["metadata"],
                data.get("embedding"),
            )
        except Exception as e:
            logger.error(f"Batch add failed for document {data.get('id', 'unknown')}: {e}")
            raise

    async def _batch_update(self, data: Dict[str, Any]) -> None:
        """Update document in batch operation."""
        try:
            # Validate required fields
            if not all(k in data for k in ["collection", "id"]):
                raise ValueError(f"Missing required fields in batch update data")

            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.db.update_document,
                data["collection"],
                data["id"],
                data.get("content"),
                data.get("metadata"),
            )
        except Exception as e:
            logger.error(f"Batch update failed for document {data.get('id', 'unknown')}: {e}")
            raise

    async def _batch_delete(self, data: Dict[str, Any]) -> None:
        """Delete document in batch operation."""
        try:
            # Validate required fields
            if not all(k in data for k in ["collection", "id"]):
                raise ValueError(f"Missing required fields in batch delete data")

            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.db.delete_document,
                data["collection"],
                data["id"],
            )
        except Exception as e:
            logger.error(f"Batch delete failed for document {data.get('id', 'unknown')}: {e}")
            raise

    def track_query_performance(
        self,
        query: str,
        collection: str,
        execution_time: float,
        result_count: int,
        cache_hit: bool = False,
    ) -> None:
        """Track query performance metrics."""
        metric = QueryMetrics(
            query=query,
            collection=collection,
            execution_time=execution_time,
            result_count=result_count,
            cache_hit=cache_hit,
        )

        self.query_metrics.append(metric)

        # Keep only recent metrics (last 1000)
        if len(self.query_metrics) > 1000:
            self.query_metrics = self.query_metrics[-1000:]

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report."""
        report = {
            "index_metrics": {
                name: metrics.to_dict() for name, metrics in self.index_metrics.items()
            },
            "query_metrics": {
                "total_queries": len(self.query_metrics),
                "collections": {},
            },
            "recommendations": [],
        }

        # Aggregate query metrics by collection
        collection_metrics = {}
        for metric in self.query_metrics:
            if metric.collection not in collection_metrics:
                collection_metrics[metric.collection] = {
                    "count": 0,
                    "total_time": 0,
                    "cache_hits": 0,
                    "avg_results": 0,
                }

            cm = collection_metrics[metric.collection]
            cm["count"] += 1
            cm["total_time"] += metric.execution_time
            cm["avg_results"] += metric.result_count
            if metric.cache_hit:
                cm["cache_hits"] += 1

        # Calculate averages
        for coll_name, cm in collection_metrics.items():
            if cm["count"] > 0:
                report["query_metrics"]["collections"][coll_name] = {
                    "query_count": cm["count"],
                    "avg_time": cm["total_time"] / cm["count"],
                    "cache_hit_rate": cm["cache_hits"] / cm["count"],
                    "avg_results": cm["avg_results"] / cm["count"],
                }

                # Generate recommendations
                avg_time = cm["total_time"] / cm["count"]
                if avg_time > 1.0:
                    report["recommendations"].append(
                        f"Collection '{coll_name}' has slow queries (avg {avg_time:.2f}s) - consider optimization"
                    )

                cache_rate = cm["cache_hits"] / cm["count"]
                if cache_rate < 0.3 and cm["count"] > 10:
                    report["recommendations"].append(
                        f"Collection '{coll_name}' has low cache hit rate ({cache_rate:.1%}) - consider increasing cache size"
                    )

        return report

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._shutdown = True
        if self.executor:
            try:
                # Python 3.9+ has cancel_futures parameter
                import sys

                if sys.version_info >= (3, 9):
                    self.executor.shutdown(wait=True, cancel_futures=False)
                else:
                    self.executor.shutdown(wait=True)
            except Exception as e:
                logger.error(f"Error shutting down executor: {e}")

        # Clear metrics to free memory
        self.query_metrics.clear()
        logger.info("Database optimizer cleaned up")
