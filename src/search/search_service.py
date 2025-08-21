"""Search service integrating all search components."""

from typing import Any, Dict, List, Optional
import time
from datetime import datetime

from config.logging_config import get_logger
from config.settings import settings
from src.search.hybrid_search import HybridSearchEngine, SearchResult
from src.search.query_processor import QueryProcessor

logger = get_logger(__name__)


class SearchService:
    """High-level search service for TTRPG content."""
    
    def __init__(self):
        """Initialize search service."""
        self.search_engine = HybridSearchEngine()
        self.query_processor = QueryProcessor()
        self.search_cache = {}  # Simple cache for repeated queries
        self.search_history = []  # Track search history for analytics
        
    async def search(
        self,
        query: str,
        rulebook: Optional[str] = None,
        source_type: Optional[str] = None,
        content_type: Optional[str] = None,
        max_results: int = 5,
        use_hybrid: bool = True,
        explain_results: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform an enhanced search with query processing.
        
        Args:
            query: Search query
            rulebook: Specific rulebook to search
            source_type: Type of source ('rulebook' or 'flavor')
            content_type: Content type filter
            max_results: Maximum results to return
            use_hybrid: Whether to use hybrid search
            explain_results: Whether to include result explanations
            
        Returns:
            Search results with metadata
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"{query}_{rulebook}_{source_type}_{content_type}_{max_results}"
        if cache_key in self.search_cache:
            cached_result = self.search_cache[cache_key]
            cached_result["from_cache"] = True
            return cached_result
        
        # Process query
        processed_query = self.query_processor.process_query(query)
        
        # Extract filters from query
        cleaned_query, extracted_filters = self.query_processor.extract_filters(
            processed_query["expanded"]
        )
        
        # Build metadata filter
        metadata_filter = {}
        if rulebook:
            metadata_filter["rulebook_name"] = rulebook
        if content_type:
            metadata_filter["chunk_type"] = content_type
        elif processed_query["intent"]["content_type"]:
            metadata_filter["chunk_type"] = processed_query["intent"]["content_type"]
        
        # Add extracted filters
        metadata_filter.update(extracted_filters)
        
        # Determine collection
        collection_name = "flavor_sources" if source_type == "flavor" else "rulebooks"
        
        # Perform search
        search_results = await self.search_engine.search(
            query=cleaned_query,
            collection_name=collection_name,
            max_results=max_results,
            metadata_filter=metadata_filter if metadata_filter else None,
            use_hybrid=use_hybrid,
        )
        
        # Format results
        formatted_results = self._format_results(
            search_results,
            processed_query,
            explain_results,
        )
        
        # Calculate search time
        search_time = time.time() - start_time
        
        # Build response
        response = {
            "query": query,
            "processed_query": processed_query,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "search_time": search_time,
            "filters_applied": metadata_filter,
            "suggestions": processed_query["suggestions"],
            "from_cache": False,
        }
        
        # Add to cache
        self.search_cache[cache_key] = response
        
        # Track search
        self._track_search(query, len(formatted_results), search_time)
        
        return response
    
    def _format_results(
        self,
        results: List[SearchResult],
        processed_query: Dict[str, Any],
        explain: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Format search results for presentation.
        
        Args:
            results: Raw search results
            processed_query: Processed query information
            explain: Whether to include explanations
            
        Returns:
            Formatted results
        """
        formatted = []
        
        for result in results:
            formatted_result = {
                "content": result.content,
                "source": result.metadata.get("rulebook_name", "Unknown"),
                "page": result.metadata.get("page", result.metadata.get("page_start")),
                "section": result.metadata.get("section"),
                "content_type": result.metadata.get("chunk_type", "unknown"),
                "relevance_score": result.combined_score,
            }
            
            # Add explanation if requested
            if explain:
                explanation = self._explain_result(result, processed_query)
                formatted_result["explanation"] = explanation
            
            # Add snippet with query highlighting
            snippet = self._create_snippet(result.content, processed_query["expanded"])
            formatted_result["snippet"] = snippet
            
            formatted.append(formatted_result)
        
        return formatted
    
    def _explain_result(self, result: SearchResult, processed_query: Dict[str, Any]) -> str:
        """
        Generate explanation for why a result was returned.
        
        Args:
            result: Search result
            processed_query: Processed query
            
        Returns:
            Explanation text
        """
        explanations = []
        
        # Explain semantic score
        if result.semantic_score > 0:
            explanations.append(
                f"Semantic similarity: {result.semantic_score:.2f} - "
                f"Content is conceptually related to your query"
            )
        
        # Explain keyword score
        if result.keyword_score > 0:
            explanations.append(
                f"Keyword match: {result.keyword_score:.2f} - "
                f"Contains exact terms from your search"
            )
        
        # Explain content type match
        if processed_query["intent"]["content_type"] == result.metadata.get("chunk_type"):
            explanations.append(
                f"Content type match: Looking for {processed_query['intent']['content_type']}"
            )
        
        return " | ".join(explanations)
    
    def _create_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        """
        Create a snippet with query term highlighting.
        
        Args:
            content: Full content text
            query: Query terms to highlight
            max_length: Maximum snippet length
            
        Returns:
            Snippet with highlighted terms
        """
        # Find best position for snippet
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        best_pos = 0
        best_score = 0
        
        # Score different positions based on query term density
        for i in range(0, len(content) - max_length, 50):
            snippet = content_lower[i:i + max_length]
            score = sum(1 for term in query_terms if term in snippet)
            if score > best_score:
                best_score = score
                best_pos = i
        
        # Extract snippet
        snippet = content[best_pos:best_pos + max_length]
        
        # Add ellipsis if needed
        if best_pos > 0:
            snippet = "..." + snippet
        if best_pos + max_length < len(content):
            snippet = snippet + "..."
        
        # Highlight query terms (wrap in **term**)
        for term in query_terms:
            if len(term) > 2:  # Skip very short terms
                # Case-insensitive replacement
                import re
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                snippet = pattern.sub(f"**{term}**", snippet)
        
        return snippet
    
    def _track_search(self, query: str, result_count: int, search_time: float):
        """
        Track search for analytics.
        
        Args:
            query: Search query
            result_count: Number of results
            search_time: Time taken
        """
        self.search_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "result_count": result_count,
            "search_time": search_time,
        })
        
        # Keep only last 1000 searches
        if len(self.search_history) > 1000:
            self.search_history = self.search_history[-1000:]
    
    async def search_with_context(
        self,
        query: str,
        campaign_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search with campaign/session context.
        
        Args:
            query: Search query
            campaign_id: Campaign context
            session_id: Session context
            **kwargs: Additional search parameters
            
        Returns:
            Search results with context
        """
        # Perform base search
        results = await self.search(query, **kwargs)
        
        # TODO: Add campaign/session context enrichment
        # This would cross-reference with campaign data
        
        return results
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """
        Get search analytics and statistics.
        
        Returns:
            Analytics data
        """
        if not self.search_history:
            return {
                "total_searches": 0,
                "average_search_time": 0,
                "average_results": 0,
                "popular_queries": [],
                "cache_stats": {
                    "size": len(self.search_cache),
                    "hit_rate": 0,
                },
            }
        
        # Calculate statistics
        total_searches = len(self.search_history)
        avg_time = sum(s["search_time"] for s in self.search_history) / total_searches
        avg_results = sum(s["result_count"] for s in self.search_history) / total_searches
        
        # Find popular queries
        from collections import Counter
        query_counts = Counter(s["query"] for s in self.search_history)
        popular_queries = query_counts.most_common(10)
        
        # Cache statistics
        cache_hits = sum(1 for s in self.search_history if s.get("from_cache", False))
        cache_hit_rate = cache_hits / total_searches if total_searches > 0 else 0
        
        return {
            "total_searches": total_searches,
            "average_search_time": avg_time,
            "average_results": avg_results,
            "popular_queries": popular_queries,
            "cache_stats": {
                "size": len(self.search_cache),
                "hit_rate": cache_hit_rate,
            },
            "index_stats": self.search_engine.get_index_stats(),
        }
    
    def clear_cache(self):
        """Clear the search cache."""
        self.search_cache.clear()
        logger.info("Search cache cleared")
    
    def update_indices(self):
        """Update search indices for all collections."""
        for collection in ["rulebooks", "flavor_sources"]:
            self.search_engine.update_index(collection)
        logger.info("Search indices updated")