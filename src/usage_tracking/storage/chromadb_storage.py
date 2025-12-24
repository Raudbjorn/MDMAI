"""
ChromaDB storage implementation for usage tracking with vector embeddings.

This module provides:
- Optimized ChromaDB collections for usage data
- Vector embeddings for usage pattern analysis
- Semantic search capabilities
- Efficient indexing and querying
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np

from .models import (
    UsageRecord, UsagePattern, UsageMetrics, CostBreakdown,
    ChromaDBConfig, ProviderType, UsageEventType, TimeAggregation
)

logger = logging.getLogger(__name__)


class UsageEmbeddingFunction:
    """Custom embedding function for usage records."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
    
    def generate_usage_text(self, record: UsageRecord) -> str:
        """Generate text representation of usage record for embedding."""
        components = [
            f"User {record.user_id}",
            f"Event {record.event_type.value}",
            f"Provider {record.provider.value}",
        ]
        
        if record.model_name:
            components.append(f"Model {record.model_name}")
        
        if record.operation:
            components.append(f"Operation {record.operation}")
        
        if record.context_id:
            components.append(f"Context {record.context_id}")
        
        # Add temporal context
        hour = record.timestamp.hour
        day_of_week = record.timestamp.strftime('%A')
        components.extend([
            f"Hour {hour}",
            f"Day {day_of_week}",
        ])
        
        # Add quantitative context
        if record.token_count > 0:
            components.append(f"Tokens {record.token_count}")
        
        if record.cost_usd > 0:
            components.append(f"Cost {float(record.cost_usd):.4f}")
        
        if record.duration_ms:
            components.append(f"Duration {record.duration_ms}ms")
        
        # Add success/failure context
        status = "Success" if record.success else "Failed"
        components.append(status)
        
        return " ".join(components)
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text list."""
        return self.embedding_fn(texts)


class ChromaDBUsageStorage:
    """ChromaDB-based storage for usage tracking data."""
    
    def __init__(self, config: ChromaDBConfig):
        self.config = config
        self.client = None
        self.collections = {}
        self.embedding_fn = UsageEmbeddingFunction(config.embedding_model)
        
        # Performance tracking
        self._query_stats = {}
        self._connection_stats = {"connects": 0, "errors": 0}
        
        # Initialize connection
        self._connect()
    
    def _connect(self) -> None:
        """Initialize ChromaDB connection and collections."""
        try:
            # Create client with persistent storage
            self.client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            
            # Initialize collections
            self._initialize_collections()
            
            self._connection_stats["connects"] += 1
            logger.info(
                "ChromaDB connection established",
                persist_directory=self.config.persist_directory,
                collections=list(self.collections.keys())
            )
            
        except Exception as e:
            self._connection_stats["errors"] += 1
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    
    def _initialize_collections(self) -> None:
        """Initialize ChromaDB collections with optimized configurations."""
        
        # Usage Records Collection - for detailed usage tracking
        self.collections["usage"] = self.client.get_or_create_collection(
            name=self.config.usage_collection,
            embedding_function=self.embedding_fn,
            metadata={
                "description": "Individual usage records with full context",
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:M": 16,
                "schema_version": "1.0.0"
            }
        )
        
        # Usage Patterns Collection - for analyzed patterns
        self.collections["patterns"] = self.client.get_or_create_collection(
            name=self.config.patterns_collection,
            embedding_function=self.embedding_fn,
            metadata={
                "description": "Analyzed usage patterns and insights",
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 100,
                "hnsw:M": 8,
                "schema_version": "1.0.0"
            }
        )
        
        # Analytics Collection - for aggregated metrics
        self.collections["analytics"] = self.client.get_or_create_collection(
            name=self.config.analytics_collection,
            embedding_function=self.embedding_fn,
            metadata={
                "description": "Aggregated usage analytics and metrics",
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 50,
                "hnsw:M": 4,
                "schema_version": "1.0.0"
            }
        )
        
        logger.info("ChromaDB collections initialized successfully")
    
    async def store_usage_record(self, record: UsageRecord) -> str:
        """Store a single usage record with embedding."""
        start_time = time.time()
        
        try:
            # Generate embedding text
            text = self.embedding_fn.generate_usage_text(record)
            
            # Prepare metadata for efficient querying
            metadata = {
                "user_id": record.user_id,
                "event_type": record.event_type.value,
                "provider": record.provider.value,
                "timestamp": record.timestamp.isoformat(),
                "date": record.timestamp.date().isoformat(),
                "hour": record.timestamp.hour,
                "day_of_week": record.timestamp.weekday(),
                "success": record.success,
                "token_count": record.token_count,
                "cost_usd": float(record.cost_usd),
                "has_context": record.context_id is not None,
            }
            
            # Add optional fields
            if record.session_id:
                metadata["session_id"] = record.session_id
            if record.model_name:
                metadata["model_name"] = record.model_name
            if record.context_id:
                metadata["context_id"] = record.context_id
            if record.operation:
                metadata["operation"] = record.operation
            if record.duration_ms:
                metadata["duration_ms"] = record.duration_ms
            
            # Store in ChromaDB
            self.collections["usage"].add(
                ids=[record.record_id],
                documents=[text],
                metadatas=[metadata]
            )
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_query_performance("store_usage_record", execution_time)
            
            logger.debug(
                "Usage record stored",
                record_id=record.record_id,
                user_id=record.user_id,
                event_type=record.event_type.value,
                execution_time=execution_time
            )
            
            return record.record_id
            
        except Exception as e:
            logger.error(f"Failed to store usage record: {e}")
            raise
    
    async def store_usage_records_batch(self, records: List[UsageRecord]) -> List[str]:
        """Store multiple usage records efficiently."""
        start_time = time.time()
        
        try:
            if not records:
                return []
            
            # Prepare batch data
            ids = [r.record_id for r in records]
            documents = [self.embedding_fn.generate_usage_text(r) for r in records]
            metadatas = []
            
            for record in records:
                metadata = {
                    "user_id": record.user_id,
                    "event_type": record.event_type.value,
                    "provider": record.provider.value,
                    "timestamp": record.timestamp.isoformat(),
                    "date": record.timestamp.date().isoformat(),
                    "hour": record.timestamp.hour,
                    "day_of_week": record.timestamp.weekday(),
                    "success": record.success,
                    "token_count": record.token_count,
                    "cost_usd": float(record.cost_usd),
                    "has_context": record.context_id is not None,
                }
                
                # Add optional fields
                if record.session_id:
                    metadata["session_id"] = record.session_id
                if record.model_name:
                    metadata["model_name"] = record.model_name
                if record.context_id:
                    metadata["context_id"] = record.context_id
                if record.operation:
                    metadata["operation"] = record.operation
                if record.duration_ms:
                    metadata["duration_ms"] = record.duration_ms
                
                metadatas.append(metadata)
            
            # Store batch
            self.collections["usage"].add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_query_performance("store_usage_records_batch", execution_time)
            
            logger.info(
                "Usage records batch stored",
                record_count=len(records),
                execution_time=execution_time,
                records_per_second=len(records) / execution_time if execution_time > 0 else 0
            )
            
            return ids
            
        except Exception as e:
            logger.error(f"Failed to store usage records batch: {e}")
            raise
    
    async def query_usage_records(
        self,
        user_id: Optional[str] = None,
        event_types: Optional[List[UsageEventType]] = None,
        providers: Optional[List[ProviderType]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        semantic_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query usage records with filtering and semantic search."""
        start_time = time.time()
        
        try:
            # Build where clause for filtering
            where_clause = {}
            
            if user_id:
                where_clause["user_id"] = user_id
            
            if event_types:
                where_clause["event_type"] = {"$in": [et.value for et in event_types]}
            
            if providers:
                where_clause["provider"] = {"$in": [p.value for p in providers]}
            
            if start_date:
                where_clause["timestamp"] = {"$gte": start_date.isoformat()}
            
            if end_date:
                if "timestamp" in where_clause:
                    where_clause["timestamp"]["$lte"] = end_date.isoformat()
                else:
                    where_clause["timestamp"] = {"$lte": end_date.isoformat()}
            
            # Perform query
            if semantic_query:
                # Semantic search with filters
                results = self.collections["usage"].query(
                    query_texts=[semantic_query],
                    n_results=limit,
                    where=where_clause if where_clause else None,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                # Filter-only query
                results = self.collections["usage"].get(
                    where=where_clause if where_clause else None,
                    limit=limit,
                    include=["documents", "metadatas"]
                )
            
            # Process results
            processed_results = []
            
            if semantic_query:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    result = {
                        "document": doc,
                        "metadata": metadata,
                        "semantic_distance": distance,
                        "relevance_score": 1.0 - distance  # Convert distance to relevance
                    }
                    processed_results.append(result)
            else:
                for doc, metadata in zip(results["documents"], results["metadatas"]):
                    result = {
                        "document": doc,
                        "metadata": metadata,
                        "relevance_score": 1.0  # Max relevance for exact matches
                    }
                    processed_results.append(result)
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_query_performance("query_usage_records", execution_time)
            
            logger.debug(
                "Usage records queried",
                result_count=len(processed_results),
                has_semantic_query=semantic_query is not None,
                execution_time=execution_time
            )
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to query usage records: {e}")
            raise
    
    async def store_usage_pattern(self, pattern: UsagePattern) -> str:
        """Store analyzed usage pattern."""
        start_time = time.time()
        
        try:
            # Generate embedding text for pattern
            pattern_text = self._generate_pattern_text(pattern)
            
            # Prepare metadata
            metadata = {
                "pattern_id": pattern.pattern_id,
                "user_id": pattern.user_id,
                "pattern_type": pattern.pattern_type,
                "frequency_score": pattern.frequency_score,
                "cost_impact": float(pattern.cost_impact),
                "efficiency_score": pattern.efficiency_score,
                "start_time": pattern.start_time.isoformat(),
                "end_time": pattern.end_time.isoformat(),
                "duration_hours": pattern.duration_hours,
                "record_count": pattern.record_count,
                "avg_cost_per_record": float(pattern.avg_cost_per_record),
                "insights_count": len(pattern.insights),
                "recommendations_count": len(pattern.recommendations),
                "last_analyzed": pattern.last_analyzed.isoformat()
            }
            
            # Store pattern
            self.collections["patterns"].add(
                ids=[pattern.pattern_id],
                documents=[pattern_text],
                metadatas=[metadata]
            )
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_query_performance("store_usage_pattern", execution_time)
            
            logger.debug(
                "Usage pattern stored",
                pattern_id=pattern.pattern_id,
                pattern_type=pattern.pattern_type,
                execution_time=execution_time
            )
            
            return pattern.pattern_id
            
        except Exception as e:
            logger.error(f"Failed to store usage pattern: {e}")
            raise
    
    def _generate_pattern_text(self, pattern: UsagePattern) -> str:
        """Generate text representation of usage pattern for embedding."""
        components = [
            f"Pattern {pattern.pattern_type}",
            f"User {pattern.user_id}",
            f"Frequency {pattern.frequency_score:.2f}",
            f"Efficiency {pattern.efficiency_score:.2f}",
            f"Cost impact {float(pattern.cost_impact):.4f}",
            f"Duration {pattern.duration_hours:.1f} hours",
            f"Records {pattern.record_count}",
        ]
        
        # Add insights and recommendations
        if pattern.insights:
            components.extend([f"Insight: {insight}" for insight in pattern.insights])
        
        if pattern.recommendations:
            components.extend([f"Recommendation: {rec}" for rec in pattern.recommendations])
        
        return " ".join(components)
    
    async def find_similar_patterns(
        self,
        pattern_type: Optional[str] = None,
        user_id: Optional[str] = None,
        query_text: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find similar usage patterns."""
        start_time = time.time()
        
        try:
            where_clause = {}
            
            if pattern_type:
                where_clause["pattern_type"] = pattern_type
            
            if user_id:
                where_clause["user_id"] = user_id
            
            if query_text:
                results = self.collections["patterns"].query(
                    query_texts=[query_text],
                    n_results=limit,
                    where=where_clause if where_clause else None,
                    include=["documents", "metadatas", "distances"]
                )
                
                processed_results = []
                for doc, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    processed_results.append({
                        "document": doc,
                        "metadata": metadata,
                        "similarity_score": 1.0 - distance
                    })
                
            else:
                results = self.collections["patterns"].get(
                    where=where_clause if where_clause else None,
                    limit=limit,
                    include=["documents", "metadatas"]
                )
                
                processed_results = []
                for doc, metadata in zip(results["documents"], results["metadatas"]):
                    processed_results.append({
                        "document": doc,
                        "metadata": metadata,
                        "similarity_score": 1.0
                    })
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_query_performance("find_similar_patterns", execution_time)
            
            logger.debug(
                "Similar patterns found",
                result_count=len(processed_results),
                execution_time=execution_time
            )
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to find similar patterns: {e}")
            raise
    
    async def store_usage_metrics(self, metrics: UsageMetrics) -> str:
        """Store aggregated usage metrics."""
        start_time = time.time()
        
        try:
            # Generate embedding text for metrics
            metrics_text = self._generate_metrics_text(metrics)
            
            # Prepare metadata
            metadata = {
                "metric_id": metrics.metric_id,
                "user_id": metrics.user_id,
                "period_start": metrics.period_start.isoformat(),
                "period_end": metrics.period_end.isoformat(),
                "aggregation_type": metrics.aggregation_type.value,
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "total_tokens": metrics.total_tokens,
                "total_cost": float(metrics.total_cost),
                "avg_cost_per_request": float(metrics.avg_cost_per_request),
                "avg_latency_ms": metrics.avg_latency_ms,
                "error_rate": metrics.error_rate,
                "calculated_at": metrics.calculated_at.isoformat()
            }
            
            # Store metrics
            self.collections["analytics"].add(
                ids=[metrics.metric_id],
                documents=[metrics_text],
                metadatas=[metadata]
            )
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_query_performance("store_usage_metrics", execution_time)
            
            logger.debug(
                "Usage metrics stored",
                metric_id=metrics.metric_id,
                aggregation_type=metrics.aggregation_type.value,
                execution_time=execution_time
            )
            
            return metrics.metric_id
            
        except Exception as e:
            logger.error(f"Failed to store usage metrics: {e}")
            raise
    
    def _generate_metrics_text(self, metrics: UsageMetrics) -> str:
        """Generate text representation of metrics for embedding."""
        components = [
            f"Period {metrics.aggregation_type.value}",
            f"Requests {metrics.total_requests}",
            f"Success rate {metrics.successful_requests / max(metrics.total_requests, 1):.2f}",
            f"Tokens {metrics.total_tokens}",
            f"Cost {float(metrics.total_cost):.4f}",
            f"Avg latency {metrics.avg_latency_ms:.1f}ms",
        ]
        
        if metrics.user_id:
            components.append(f"User {metrics.user_id}")
        else:
            components.append("Global metrics")
        
        # Add provider breakdown
        for provider, cost in metrics.cost_by_provider.items():
            components.append(f"{provider} cost {float(cost):.4f}")
        
        return " ".join(components)
    
    def _track_query_performance(self, operation: str, execution_time: float) -> None:
        """Track query performance metrics."""
        if operation not in self._query_stats:
            self._query_stats[operation] = {
                "count": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "max_time": 0.0,
                "min_time": float('inf')
            }
        
        stats = self._query_stats[operation]
        stats["count"] += 1
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["count"]
        stats["max_time"] = max(stats["max_time"], execution_time)
        stats["min_time"] = min(stats["min_time"], execution_time)
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive ChromaDB collection statistics."""
        stats = {}
        
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[name] = {
                    "document_count": count,
                    "collection_name": collection.name,
                    "metadata": collection.metadata or {}
                }
            except Exception as e:
                logger.error(f"Failed to get stats for collection {name}: {e}")
                stats[name] = {"error": str(e)}
        
        return {
            "collections": stats,
            "query_performance": self._query_stats,
            "connection_stats": self._connection_stats
        }
    
    async def cleanup_old_records(self, days_to_keep: int = 90) -> int:
        """Clean up old usage records beyond retention period."""
        start_time = time.time()
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        try:
            # Query old records
            old_records = self.collections["usage"].get(
                where={"timestamp": {"$lt": cutoff_date.isoformat()}},
                include=["metadatas"]
            )
            
            if not old_records["ids"]:
                return 0
            
            # Delete old records
            self.collections["usage"].delete(ids=old_records["ids"])
            
            deleted_count = len(old_records["ids"])
            execution_time = time.time() - start_time
            
            logger.info(
                "Old usage records cleaned up",
                deleted_count=deleted_count,
                cutoff_date=cutoff_date.isoformat(),
                execution_time=execution_time
            )
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
            raise
    
    async def close(self) -> None:
        """Clean up resources."""
        try:
            # ChromaDB client doesn't need explicit closing
            self.client = None
            self.collections.clear()
            logger.info("ChromaDB storage closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing ChromaDB storage: {e}")