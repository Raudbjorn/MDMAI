"""ChromaDB extensions for usage tracking and cost management analytics."""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import chromadb
from chromadb.utils import embedding_functions

from ..core.database import get_db_manager
from ..ai_providers.models import ProviderType, UsageRecord
from config.logging_config import get_logger

logger = get_logger(__name__)


class UsageAnalyticsType(Enum):
    """Types of usage analytics stored in ChromaDB."""
    USAGE_PATTERNS = "usage_patterns"
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_METRICS = "performance_metrics"
    USER_BEHAVIOR = "user_behavior"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class UsageVectorRecord:
    """Vector record for usage analytics in ChromaDB."""
    record_id: str
    user_id: str
    provider_type: str
    model: str
    timestamp: datetime
    cost: float
    tokens_total: int
    latency_ms: float
    success: bool
    analytics_type: UsageAnalyticsType
    
    # Vector-searchable content
    search_content: str
    
    # Metadata for filtering
    metadata: Dict[str, Any]
    
    def to_vector_document(self) -> Tuple[str, str, Dict[str, Any]]:
        """Convert to ChromaDB document format."""
        document_id = f"{self.analytics_type.value}_{self.record_id}"
        
        metadata = {
            "user_id": self.user_id,
            "provider_type": self.provider_type,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
            "cost": self.cost,
            "tokens_total": self.tokens_total,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "analytics_type": self.analytics_type.value,
            "date": self.timestamp.date().isoformat(),
            "hour": self.timestamp.hour,
            "day_of_week": self.timestamp.weekday(),
            **self.metadata
        }
        
        return document_id, self.search_content, metadata


class UsageTrackingChromaExtensions:
    """ChromaDB extensions for usage tracking analytics."""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.client = self.db_manager.client
        self.embedding_function = self.db_manager.embedding_function
        
        # Usage tracking specific collections
        self.usage_collections = {}
        self._initialize_usage_collections()
    
    def _initialize_usage_collections(self) -> None:
        """Initialize ChromaDB collections for usage tracking."""
        collection_configs = {
            "usage_patterns": {
                "description": "User usage patterns and behavior analysis",
                "metadata": {"type": "usage_analytics", "retention_days": 365}
            },
            "cost_optimization": {
                "description": "Cost optimization insights and recommendations", 
                "metadata": {"type": "cost_analytics", "retention_days": 180}
            },
            "performance_metrics": {
                "description": "Performance metrics and latency analysis",
                "metadata": {"type": "performance_analytics", "retention_days": 90}
            },
            "anomaly_detection": {
                "description": "Usage anomalies and pattern deviations",
                "metadata": {"type": "anomaly_analytics", "retention_days": 30}
            },
            "user_behavior": {
                "description": "Individual user behavior patterns",
                "metadata": {"type": "behavior_analytics", "retention_days": 365}
            }
        }
        
        for name, config in collection_configs.items():
            collection_name = f"usage_{name}"
            try:
                # Try to get existing collection
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                logger.debug(f"Retrieved existing usage collection: {collection_name}")
            except ValueError:
                # Create new collection
                collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata=config["metadata"]
                )
                logger.info(f"Created new usage collection: {collection_name}")
            
            self.usage_collections[name] = collection
    
    async def store_usage_analytics(
        self,
        usage_record: UsageRecord,
        analytics_type: UsageAnalyticsType,
        analysis_results: Dict[str, Any]
    ) -> None:
        """Store usage analytics in ChromaDB for vector search and pattern recognition."""
        try:
            # Generate search content based on analytics type
            search_content = self._generate_search_content(usage_record, analytics_type, analysis_results)
            
            # Create vector record
            vector_record = UsageVectorRecord(
                record_id=usage_record.request_id,
                user_id=usage_record.metadata.get("user_id", "unknown"),
                provider_type=usage_record.provider_type.value,
                model=usage_record.model,
                timestamp=usage_record.timestamp,
                cost=usage_record.cost,
                tokens_total=usage_record.input_tokens + usage_record.output_tokens,
                latency_ms=usage_record.latency_ms,
                success=usage_record.success,
                analytics_type=analytics_type,
                search_content=search_content,
                metadata=analysis_results
            )
            
            # Convert to document format
            doc_id, content, metadata = vector_record.to_vector_document()
            
            # Store in appropriate collection
            collection_key = analytics_type.value.replace("_", "_")
            if collection_key in self.usage_collections:
                collection = self.usage_collections[collection_key]
                collection.add(
                    ids=[doc_id],
                    documents=[content],
                    metadatas=[metadata]
                )
                
                logger.debug(
                    "Usage analytics stored in ChromaDB",
                    analytics_type=analytics_type.value,
                    record_id=usage_record.request_id,
                    collection=collection_key
                )
        
        except Exception as e:
            logger.error(
                "Failed to store usage analytics in ChromaDB",
                analytics_type=analytics_type.value,
                record_id=usage_record.request_id,
                error=str(e)
            )
            raise
    
    def _generate_search_content(
        self,
        usage_record: UsageRecord,
        analytics_type: UsageAnalyticsType,
        analysis_results: Dict[str, Any]
    ) -> str:
        """Generate searchable content for vector storage."""
        base_content = [
            f"user {usage_record.metadata.get('user_id', 'unknown')}",
            f"provider {usage_record.provider_type.value}",
            f"model {usage_record.model}",
            f"cost ${usage_record.cost:.4f}",
            f"tokens {usage_record.input_tokens + usage_record.output_tokens}",
            f"latency {usage_record.latency_ms:.0f}ms",
            "success" if usage_record.success else "failure"
        ]
        
        # Add analytics-specific content
        if analytics_type == UsageAnalyticsType.USAGE_PATTERNS:
            pattern_info = analysis_results.get("patterns", {})
            base_content.extend([
                f"frequency {pattern_info.get('frequency', 'unknown')}",
                f"time_of_day {pattern_info.get('time_of_day', 'unknown')}",
                f"usage_type {pattern_info.get('usage_type', 'unknown')}"
            ])
        
        elif analytics_type == UsageAnalyticsType.COST_OPTIMIZATION:
            optimization_info = analysis_results.get("optimization", {})
            base_content.extend([
                f"efficiency_score {optimization_info.get('efficiency_score', 0)}",
                f"cost_per_token {optimization_info.get('cost_per_token', 0)}",
                f"optimization_potential {optimization_info.get('potential_savings', 0)}"
            ])
        
        elif analytics_type == UsageAnalyticsType.PERFORMANCE_METRICS:
            performance_info = analysis_results.get("performance", {})
            base_content.extend([
                f"response_time_percentile {performance_info.get('percentile_rank', 0)}",
                f"throughput {performance_info.get('throughput', 0)}",
                f"error_rate {performance_info.get('error_rate', 0)}"
            ])
        
        elif analytics_type == UsageAnalyticsType.ANOMALY_DETECTION:
            anomaly_info = analysis_results.get("anomaly", {})
            base_content.extend([
                f"anomaly_score {anomaly_info.get('score', 0)}",
                f"anomaly_type {anomaly_info.get('type', 'unknown')}",
                f"deviation {anomaly_info.get('deviation', 0)}"
            ])
        
        elif analytics_type == UsageAnalyticsType.USER_BEHAVIOR:
            behavior_info = analysis_results.get("behavior", {})
            base_content.extend([
                f"session_length {behavior_info.get('session_length', 0)}",
                f"interaction_pattern {behavior_info.get('pattern', 'unknown')}",
                f"preference_score {behavior_info.get('preference_score', 0)}"
            ])
        
        return " ".join(base_content)
    
    async def search_usage_patterns(
        self,
        query: str,
        analytics_type: Optional[UsageAnalyticsType] = None,
        user_id: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search usage patterns using vector similarity."""
        try:
            # Determine which collections to search
            collections_to_search = []
            if analytics_type:
                collection_key = analytics_type.value.replace("_", "_")
                if collection_key in self.usage_collections:
                    collections_to_search = [self.usage_collections[collection_key]]
            else:
                collections_to_search = list(self.usage_collections.values())
            
            all_results = []
            
            for collection in collections_to_search:
                # Build where clause for filtering
                where_clause = {}
                if user_id:
                    where_clause["user_id"] = user_id
                
                if date_range:
                    start_date, end_date = date_range
                    where_clause["date"] = {
                        "$gte": start_date.date().isoformat(),
                        "$lte": end_date.date().isoformat()
                    }
                
                # Perform vector search
                results = collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_clause if where_clause else None
                )
                
                # Format results
                if results["ids"][0]:
                    for i in range(len(results["ids"][0])):
                        all_results.append({
                            "id": results["ids"][0][i],
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": results["distances"][0][i] if "distances" in results else None,
                            "collection": collection.name
                        })
            
            # Sort by relevance (distance) and limit results
            all_results.sort(key=lambda x: x.get("distance", float("inf")))
            return all_results[:n_results]
        
        except Exception as e:
            logger.error("Failed to search usage patterns", query=query, error=str(e))
            raise
    
    async def get_usage_insights(
        self,
        user_id: str,
        time_range: str = "7d"
    ) -> Dict[str, Any]:
        """Get comprehensive usage insights for a user using vector analytics."""
        try:
            # Parse time range
            days_map = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}
            days = days_map.get(time_range, 7)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            insights = {
                "user_id": user_id,
                "time_range": time_range,
                "timestamp": datetime.now().isoformat(),
                "patterns": [],
                "anomalies": [],
                "optimization_opportunities": [],
                "behavior_analysis": {},
                "performance_summary": {}
            }
            
            # Get usage patterns
            pattern_results = await self.search_usage_patterns(
                f"user {user_id} usage patterns behavior",
                analytics_type=UsageAnalyticsType.USAGE_PATTERNS,
                user_id=user_id,
                date_range=(start_date, end_date),
                n_results=5
            )
            insights["patterns"] = [r["metadata"] for r in pattern_results]
            
            # Get anomalies
            anomaly_results = await self.search_usage_patterns(
                f"user {user_id} anomaly detection unusual behavior",
                analytics_type=UsageAnalyticsType.ANOMALY_DETECTION,
                user_id=user_id,
                date_range=(start_date, end_date),
                n_results=3
            )
            insights["anomalies"] = [r["metadata"] for r in anomaly_results]
            
            # Get optimization opportunities
            optimization_results = await self.search_usage_patterns(
                f"user {user_id} cost optimization efficiency savings",
                analytics_type=UsageAnalyticsType.COST_OPTIMIZATION,
                user_id=user_id,
                date_range=(start_date, end_date),
                n_results=5
            )
            insights["optimization_opportunities"] = [r["metadata"] for r in optimization_results]
            
            # Get behavior analysis
            behavior_results = await self.search_usage_patterns(
                f"user {user_id} behavior preferences interaction",
                analytics_type=UsageAnalyticsType.USER_BEHAVIOR,
                user_id=user_id,
                date_range=(start_date, end_date),
                n_results=3
            )
            if behavior_results:
                insights["behavior_analysis"] = behavior_results[0]["metadata"]
            
            # Get performance summary
            performance_results = await self.search_usage_patterns(
                f"user {user_id} performance metrics latency throughput",
                analytics_type=UsageAnalyticsType.PERFORMANCE_METRICS,
                user_id=user_id,
                date_range=(start_date, end_date),
                n_results=3
            )
            if performance_results:
                insights["performance_summary"] = performance_results[0]["metadata"]
            
            return insights
        
        except Exception as e:
            logger.error("Failed to get usage insights", user_id=user_id, error=str(e))
            raise
    
    async def cleanup_old_analytics(self, retention_days: int = 365) -> Dict[str, int]:
        """Clean up old analytics data based on retention policies."""
        cleanup_stats = {}
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        for collection_name, collection in self.usage_collections.items():
            try:
                # Get collection metadata for retention policy
                collection_metadata = collection.metadata or {}
                collection_retention = collection_metadata.get("retention_days", retention_days)
                collection_cutoff = datetime.now() - timedelta(days=collection_retention)
                
                # Use query-then-delete approach since ChromaDB doesn't support direct where-based delete
                # ChromaDB uses different query syntax than MongoDB
                cutoff_date = collection_cutoff.date().isoformat()
                
                # First, query for documents older than cutoff
                old_docs = collection.get()
                
                # Filter documents in Python since ChromaDB has limited where clause support
                # for complex date comparisons
                ids_to_delete = []
                if old_docs["ids"] and old_docs["metadatas"]:
                    for doc_id, metadata in zip(old_docs["ids"], old_docs["metadatas"]):
                        doc_date = metadata.get("date", "")
                        if doc_date < cutoff_date:  # String comparison works for ISO dates
                            ids_to_delete.append(doc_id)
                
                # Delete the identified documents
                if ids_to_delete:
                    collection.delete(ids=ids_to_delete)
                    cleanup_stats[collection_name] = len(ids_to_delete)
                    logger.info(
                        "Cleaned up old analytics data",
                        collection=collection_name,
                        deleted_count=len(ids_to_delete),
                        cutoff_date=collection_cutoff.isoformat()
                    )
                # Remove else clause as suggested in review - no action needed when no documents to delete
                cleanup_stats[collection_name] = cleanup_stats.get(collection_name, 0)
            
            except Exception as e:
                logger.error(
                    "Failed to cleanup collection",
                    collection=collection_name,
                    error=str(e)
                )
                cleanup_stats[collection_name] = -1
        
        return cleanup_stats
    
    async def get_analytics_statistics(self) -> Dict[str, Any]:
        """Get statistics about usage analytics storage."""
        stats = {
            "collections": {},
            "total_documents": 0,
            "storage_summary": {}
        }
        
        for collection_name, collection in self.usage_collections.items():
            try:
                count = collection.count()
                collection_metadata = collection.metadata or {}
                
                stats["collections"][collection_name] = {
                    "document_count": count,
                    "retention_days": collection_metadata.get("retention_days", 365),
                    "type": collection_metadata.get("type", "unknown")
                }
                
                stats["total_documents"] += count
            
            except Exception as e:
                logger.error(f"Failed to get stats for {collection_name}: {e}")
                stats["collections"][collection_name] = {"error": str(e)}
        
        return stats