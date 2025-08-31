"""Search service integrating all search components."""

import hashlib
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.logging_config import get_logger
from config.settings import settings
from src.core.database import ChromaDBManager
from src.search.cache_manager import SearchCacheManager
from src.search.error_handler import (
    QueryProcessingError,
    SearchValidator,
    handle_search_errors,
)
from src.search.hybrid_search import HybridSearchEngine, SearchResult
from src.search.query_clarification import QueryClarificationService
from src.search.query_completion import QueryCompletionService
from src.search.query_processor import QueryProcessor
from src.search.search_analytics import SearchAnalytics

logger = get_logger(__name__)


class SearchService:
    """High-level search service for TTRPG content."""

    def __init__(self):
        """Initialize search service."""
        self.search_engine = HybridSearchEngine()
        self.db_client = ChromaDBManager()
        self.query_processor = QueryProcessor(db_client=self.db_client)
        self.cache_manager = SearchCacheManager()  # Use proper cache manager
        self.search_history = []  # Track search history for analytics

        # Initialize new services
        self.clarification_service = QueryClarificationService()
        self.analytics_service = SearchAnalytics(persist_dir=settings.ANALYTICS_DIR)
        self.completion_service = QueryCompletionService(model_dir=settings.COMPLETION_MODELS_DIR)

    @handle_search_errors()
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

        # Validate input
        try:
            query = SearchValidator.validate_query(query)
        except QueryProcessingError as e:
            logger.warning(f"Query validation failed: {str(e)}")
            return {
                "error": True,
                "message": str(e),
                "results": [],
                "total_results": 0,
                "suggestions": ["Try a different search term", "Check your spelling"],
            }

        # Generate cache key using hash for better key management
        cache_key_data = {
            "query": query,
            "rulebook": rulebook,
            "source_type": source_type,
            "content_type": content_type,
            "max_results": max_results,
            "use_hybrid": use_hybrid,
        }
        cache_key = hashlib.md5(json.dumps(cache_key_data, sort_keys=True).encode()).hexdigest()

        # Check cache
        cached_result = self.cache_manager.get_query_result(cache_key)
        if cached_result:
            cached_result["from_cache"] = True
            self._track_search(query, cached_result.get("total_results", 0), 0, from_cache=True)
            return cached_result

        # Process query with error handling
        try:
            processed_query = self.query_processor.process_query(query)
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            # Fallback to basic processing
            processed_query = {
                "original": query,
                "cleaned": query.lower().strip(),
                "expanded": query.lower().strip(),
                "semantic_expanded": query.lower().strip(),
                "intent": {"content_type": None},
                "suggestions": [],
            }

        # Extract filters from query
        try:
            cleaned_query, extracted_filters = self.query_processor.extract_filters(
                processed_query.get("expanded", query)
            )
        except Exception as e:
            logger.error(f"Filter extraction failed: {str(e)}")
            cleaned_query = processed_query.get("expanded", query)
            extracted_filters = {}

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

        # Perform search with error handling
        try:
            search_results = await self.search_engine.search(
                query=cleaned_query,
                collection_name=collection_name,
                max_results=max_results,
                metadata_filter=metadata_filter if metadata_filter else None,
                use_hybrid=use_hybrid,
            )
        except Exception as e:
            logger.error(f"Search engine failed: {str(e)}")
            # Try fallback to basic search without hybrid
            try:
                search_results = await self.search_engine.search(
                    query=cleaned_query,
                    collection_name=collection_name,
                    max_results=max_results,
                    metadata_filter=None,
                    use_hybrid=False,
                )
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {str(fallback_error)}")
                search_results = []

        # Format results with validation
        try:
            formatted_results = self._format_results(
                search_results,
                processed_query,
                explain_results,
            )
            # Validate results
            formatted_results = SearchValidator.validate_results(formatted_results)
        except Exception as e:
            logger.error(f"Result formatting failed: {str(e)}")
            formatted_results = []

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
        self.cache_manager.cache_query_result(cache_key, response)

        # Track search
        self._track_search(query, len(formatted_results), search_time, from_cache=False)

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

    def _explain_result(
        self, result: SearchResult, processed_query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate detailed explanation for why a result was returned.

        Args:
            result: Search result
            processed_query: Processed query

        Returns:
            Detailed explanation dictionary
        """
        explanation = {
            "summary": "",
            "scores": {
                "semantic": result.semantic_score,
                "keyword": result.keyword_score,
                "combined": result.combined_score,
            },
            "matching_terms": [],
            "relevance_factors": [],
            "confidence": "low",
        }

        # Analyze semantic score
        if result.semantic_score > 0:
            semantic_level = self._categorize_score(result.semantic_score)
            explanation["relevance_factors"].append(
                {
                    "type": "semantic",
                    "score": result.semantic_score,
                    "level": semantic_level,
                    "description": f"{semantic_level.capitalize()} conceptual similarity to your query",
                }
            )

        # Analyze keyword matches
        if result.keyword_score > 0:
            keyword_level = self._categorize_score(result.keyword_score)
            matching_terms = self._find_matching_terms(result.content, processed_query["expanded"])
            explanation["matching_terms"] = matching_terms
            explanation["relevance_factors"].append(
                {
                    "type": "keyword",
                    "score": result.keyword_score,
                    "level": keyword_level,
                    "description": f"Found {len(matching_terms)} matching term(s): {', '.join(matching_terms[:3])}",
                }
            )

        # Check content type match
        query_intent_type = processed_query["intent"].get("content_type")
        result_type = result.metadata.get("chunk_type", "").lower()

        if query_intent_type and result_type:
            if query_intent_type == result_type:
                explanation["relevance_factors"].append(
                    {
                        "type": "content_type",
                        "match": True,
                        "description": f"Exact match for {query_intent_type} content",
                    }
                )
            elif query_intent_type in result_type or result_type in query_intent_type:
                explanation["relevance_factors"].append(
                    {
                        "type": "content_type",
                        "match": "partial",
                        "description": f"Related content type: {result_type}",
                    }
                )

        # Check source quality
        source = result.metadata.get("source", "")
        if "Player's Handbook" in source or "PHB" in source:
            explanation["relevance_factors"].append(
                {
                    "type": "source_quality",
                    "level": "high",
                    "description": "From core rulebook",
                }
            )

        # Analyze query expansion contribution
        if processed_query.get("semantic_expanded") != processed_query.get("expanded"):
            explanation["relevance_factors"].append(
                {
                    "type": "query_expansion",
                    "description": "Match found through semantic query expansion",
                }
            )

        # Calculate confidence level
        if result.combined_score >= 0.8:
            explanation["confidence"] = "high"
        elif result.combined_score >= 0.6:
            explanation["confidence"] = "medium"
        else:
            explanation["confidence"] = "low"

        # Generate summary
        primary_factor = (
            max(explanation["relevance_factors"], key=lambda x: x.get("score", 0.5))
            if explanation["relevance_factors"]
            else None
        )

        if primary_factor:
            explanation["summary"] = (
                f"{explanation['confidence'].capitalize()} confidence match. "
                f"{primary_factor['description']}. "
                f"Overall relevance: {result.combined_score:.1%}"
            )
        else:
            explanation["summary"] = f"Relevance score: {result.combined_score:.1%}"

        return explanation

    def _categorize_score(self, score: float) -> str:
        """Categorize a score into levels."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "moderate"
        elif score >= 0.2:
            return "weak"
        else:
            return "minimal"

    def _find_matching_terms(self, content: str, query: str) -> List[str]:
        """Find which query terms match in the content."""
        content_lower = content.lower()
        query_terms = query.lower().split()
        matching = []

        for term in query_terms:
            if term in content_lower:
                matching.append(term)

        return matching

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
            snippet = content_lower[i : i + max_length]
            score = sum(1 for term in query_terms if term in snippet)
            if score > best_score:
                best_score = score
                best_pos = i

        # Extract snippet
        snippet = content[best_pos : best_pos + max_length]

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

    def _track_search(
        self, query: str, result_count: int, search_time: float, from_cache: bool = False
    ):
        """
        Track search for analytics.

        Args:
            query: Search query
            result_count: Number of results
            search_time: Time taken
            from_cache: Whether result was from cache
        """
        self.search_history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "result_count": result_count,
                "search_time": search_time,
                "from_cache": from_cache,
            }
        )

        # Keep only last 1000 searches
        if len(self.search_history) > 1000:
            self.search_history = self.search_history[-1000:]

    @handle_search_errors()
    async def search_with_context(
        self,
        query: str,
        campaign_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Search with campaign/session context for cross-referencing.

        Args:
            query: Search query
            campaign_id: Campaign context
            session_id: Session context
            **kwargs: Additional search parameters

        Returns:
            Search results with context and cross-references
        """
        # Perform base search
        results = await self.search(query, **kwargs)

        # Add campaign context if provided
        if campaign_id:
            campaign_refs = await self._get_campaign_references(query, campaign_id, results)
            results["campaign_references"] = campaign_refs

            # Merge campaign-specific results with general results
            if campaign_refs:
                results = self._merge_cross_references(results, campaign_refs)

        # Add session context if provided
        if session_id:
            session_refs = await self._get_session_references(query, session_id, results)
            results["session_references"] = session_refs

        # Add bidirectional cross-references
        results["cross_references"] = await self._find_cross_references(
            results.get("results", []), campaign_id, session_id
        )

        return results

    async def _get_campaign_references(
        self, query: str, campaign_id: str, base_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get campaign-specific references related to the query.

        Args:
            query: Search query
            campaign_id: Campaign identifier
            base_results: Base search results

        Returns:
            Campaign-specific references
        """
        campaign_refs = []

        try:
            # Get campaign collection
            if "campaigns" not in self.db_client.collections:
                logger.warning("Campaigns collection not found")
                return campaign_refs
            campaign_collection = self.db_client.collections["campaigns"]

            # Search within campaign data
            campaign_results = campaign_collection.query(
                query_texts=[query], n_results=10, where={"campaign_id": campaign_id}
            )

            # Process campaign results
            if campaign_results and campaign_results.get("documents"):
                for i, doc in enumerate(campaign_results["documents"][0]):
                    metadata = (
                        campaign_results["metadatas"][0][i]
                        if campaign_results.get("metadatas")
                        else {}
                    )

                    campaign_refs.append(
                        {
                            "type": "campaign",
                            "content": doc,
                            "metadata": metadata,
                            "relevance": "high" if query.lower() in doc.lower() else "medium",
                            "source": f"Campaign: {campaign_id}",
                        }
                    )

            # Find NPCs, locations, and plot points related to the query
            campaign_entities = await self._extract_campaign_entities(campaign_id, query)
            campaign_refs.extend(campaign_entities)

        except Exception as e:
            logger.error(f"Failed to get campaign references: {str(e)}")

        return campaign_refs

    async def _get_session_references(
        self, query: str, session_id: str, base_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get session-specific references related to the query.

        Args:
            query: Search query
            session_id: Session identifier
            base_results: Base search results

        Returns:
            Session-specific references
        """
        session_refs = []

        try:
            # Get session collection
            if "sessions" not in self.db_client.collections:
                logger.warning("Sessions collection not found")
                return session_refs
            session_collection = self.db_client.collections["sessions"]

            # Search within session data
            session_results = session_collection.query(
                query_texts=[query], n_results=5, where={"session_id": session_id}
            )

            # Process session results
            if session_results and session_results.get("documents"):
                for i, doc in enumerate(session_results["documents"][0]):
                    metadata = (
                        session_results["metadatas"][0][i]
                        if session_results.get("metadatas")
                        else {}
                    )

                    session_refs.append(
                        {
                            "type": "session",
                            "content": doc,
                            "metadata": metadata,
                            "relevance": "high",
                            "source": f"Session: {session_id}",
                        }
                    )

        except Exception as e:
            logger.error(f"Failed to get session references: {str(e)}")

        return session_refs

    async def _find_cross_references(
        self,
        results: List[Dict[str, Any]],
        campaign_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find cross-references between rules and campaign/session content.

        Args:
            results: Search results to cross-reference
            campaign_id: Optional campaign context
            session_id: Optional session context

        Returns:
            Dictionary of cross-references by type
        """
        cross_refs = {
            "rules_to_campaign": [],
            "campaign_to_rules": [],
            "related_spells": [],
            "related_monsters": [],
            "related_items": [],
        }

        try:
            # Extract entities from results
            entities = self._extract_entities_from_results(results)

            # Find rules referenced in campaign
            if campaign_id and entities.get("rules"):
                for rule in entities["rules"]:
                    campaign_usage = await self._find_rule_usage_in_campaign(rule, campaign_id)
                    if campaign_usage:
                        cross_refs["rules_to_campaign"].extend(campaign_usage)

            # Find campaign elements that reference rules
            if campaign_id and entities.get("campaign_elements"):
                for element in entities["campaign_elements"]:
                    rule_refs = await self._find_rules_for_campaign_element(element)
                    if rule_refs:
                        cross_refs["campaign_to_rules"].extend(rule_refs)

            # Find related game elements
            if entities.get("spells"):
                cross_refs["related_spells"] = await self._find_related_spells(entities["spells"])

            if entities.get("monsters"):
                cross_refs["related_monsters"] = await self._find_related_monsters(
                    entities["monsters"]
                )

            if entities.get("items"):
                cross_refs["related_items"] = await self._find_related_items(entities["items"])

        except Exception as e:
            logger.error(f"Failed to find cross-references: {str(e)}")

        return cross_refs

    def _extract_entities_from_results(self, results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Extract game entities from search results.

        Args:
            results: Search results

        Returns:
            Dictionary of extracted entities by type
        """
        entities = {
            "rules": [],
            "spells": [],
            "monsters": [],
            "items": [],
            "campaign_elements": [],
        }

        for result in results:
            content = result.get("content", "").lower()
            metadata = result.get("metadata", {})

            # Classify content type
            content_type = metadata.get("content_type", "")

            if "spell" in content_type or "spell" in content:
                entities["spells"].append(result.get("title", content[:50]))
            elif "monster" in content_type or "creature" in content:
                entities["monsters"].append(result.get("title", content[:50]))
            elif "item" in content_type or "equipment" in content:
                entities["items"].append(result.get("title", content[:50]))
            elif "rule" in content_type or "mechanic" in content:
                entities["rules"].append(result.get("title", content[:50]))
            elif metadata.get("source_type") == "campaign":
                entities["campaign_elements"].append(result.get("title", content[:50]))

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities

    async def _extract_campaign_entities(
        self, campaign_id: str, query: str
    ) -> List[Dict[str, Any]]:
        """Extract NPCs, locations, and plot points from campaign."""
        # Placeholder for campaign entity extraction
        # This would query specific campaign data structures
        """
        TODO: Implement extraction of NPCs, locations, and plot points from campaign data structures.
        This should query the campaign database or data structures using the provided campaign_id and query,
        and return a list of dictionaries representing the extracted entities.
        """
        raise NotImplementedError(
            "Extraction of campaign entities is not yet implemented. "
            "This method should query campaign data structures for NPCs, locations, and plot points."
        )

    async def _find_rule_usage_in_campaign(
        self, rule: str, campaign_id: str
    ) -> List[Dict[str, Any]]:
        """Find where a rule is used in a campaign."""
        # Placeholder for finding rule usage
        return []

    async def _find_rules_for_campaign_element(self, element: str) -> List[Dict[str, Any]]:
        """Find rules related to a campaign element."""
        # Placeholder for finding related rules
        return []

    async def _find_related_spells(self, spells: List[str]) -> List[Dict[str, Any]]:
        """Find spells related to the given spells."""
        # Placeholder for finding related spells
        return []

    async def _find_related_monsters(self, monsters: List[str]) -> List[Dict[str, Any]]:
        """Find monsters related to the given monsters."""
        # Placeholder for finding related monsters
        return []

    async def _find_related_items(self, items: List[str]) -> List[Dict[str, Any]]:
        """Find items related to the given items."""
        # Placeholder for finding related items
        """Find where a rule is used in a campaign.
        
        NOTE: This method is not yet implemented.
        """
        raise NotImplementedError(
            "The method '_find_rule_usage_in_campaign' is not yet implemented."
        )

    async def _find_rules_for_campaign_element(self, element: str) -> List[Dict[str, Any]]:
        """Find rules related to a campaign element.

        NOTE: This method is not yet implemented.
        """
        raise NotImplementedError(
            "The method '_find_rules_for_campaign_element' is not yet implemented."
        )

    async def _find_related_spells(self, spells: List[str]) -> List[Dict[str, Any]]:
        """Find spells related to the given spells.

        NOTE: This method is not yet implemented.
        """
        raise NotImplementedError("The method '_find_related_spells' is not yet implemented.")

    async def _find_related_monsters(self, monsters: List[str]) -> List[Dict[str, Any]]:
        """Find monsters related to the given monsters.

        NOTE: This method is not yet implemented.
        """
        raise NotImplementedError("The method '_find_related_monsters' is not yet implemented.")

    async def _find_related_items(self, items: List[str]) -> List[Dict[str, Any]]:
        """Find items related to the given items.

        NOTE: This method is not yet implemented.
        """
        raise NotImplementedError("The method '_find_related_items' is not yet implemented.")

    def _merge_cross_references(
        self, base_results: Dict[str, Any], cross_refs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge cross-references with base results.

        Args:
            base_results: Original search results
            cross_refs: Cross-reference results

        Returns:
            Merged results with cross-references prioritized
        """
        # Add cross-references as a separate section
        if "results" not in base_results:
            base_results["results"] = []

        # Mark cross-references with higher priority
        for ref in cross_refs:
            ref["is_cross_reference"] = True
            ref["priority"] = "high"

        # Insert cross-references at the beginning of results
        base_results["results"] = cross_refs + base_results.get("results", [])

        # Update result count
        base_results["total_results"] = len(base_results["results"])

        return base_results

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
                    "size": self.cache_manager.query_cache.size(),
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

        # Get cache manager stats
        cache_stats = self.cache_manager.get_stats()

        return {
            "total_searches": total_searches,
            "average_search_time": avg_time,
            "average_results": avg_results,
            "popular_queries": popular_queries,
            "cache_stats": {
                "hit_rate": cache_hit_rate,
                "detailed": cache_stats,
            },
            "index_stats": self.search_engine.get_index_stats(),
        }

    def clear_cache(self):
        """Clear all search caches."""
        self.cache_manager.clear_all()
        logger.info("All search caches cleared")

    def update_indices(self):
        """Update search indices for all collections."""
        for collection in ["rulebooks", "flavor_sources"]:
            self.search_engine.update_index(collection)
        logger.info("Search indices updated")

    # New methods for query clarification
    async def search_with_clarification(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Search with interactive query clarification.

        Args:
            query: Search query
            **kwargs: Additional search parameters

        Returns:
            Search results or clarification request
        """
        # Check if query is ambiguous
        ambiguity_info = self.clarification_service.detect_ambiguity(query, search_context=kwargs)

        if ambiguity_info["is_ambiguous"]:
            # Generate clarification questions
            questions = self.clarification_service.generate_clarification_questions(
                query, ambiguity_info, available_content=self._get_available_content_types()
            )

            # Generate query refinements
            refinements = self.clarification_service.generate_query_refinements(
                query, user_intent=self.query_processor.extract_intent(query)
            )

            return {
                "requires_clarification": True,
                "original_query": query,
                "ambiguity_score": ambiguity_info["ambiguity_score"],
                "clarification_questions": questions,
                "suggested_refinements": refinements,
                "message": "Your query needs clarification for better results",
            }

        # Proceed with normal search
        results = await self.search(query, **kwargs)

        # Learn from successful query
        if results.get("total_results", 0) > 0:
            avg_relevance = (
                sum(r.get("relevance_score", 0) for r in results["results"])
                / len(results["results"])
                if results["results"]
                else 0
            )
            self.clarification_service.learn_from_feedback(query, {}, avg_relevance)

        return results

    async def apply_clarification_and_search(
        self, original_query: str, clarification_responses: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Apply clarification responses and perform search.

        Args:
            original_query: Original ambiguous query
            clarification_responses: User's clarification responses
            **kwargs: Additional search parameters

        Returns:
            Search results with refined query
        """
        # Apply clarification to refine query
        refined_query = self.clarification_service.apply_clarification(
            original_query, clarification_responses
        )

        # Perform search with refined query
        results = await self.search(refined_query, **kwargs)
        results["original_query"] = original_query
        results["refined_query"] = refined_query
        results["clarification_applied"] = True

        # Learn from the clarification
        if results.get("total_results", 0) > 0:
            avg_relevance = (
                sum(r.get("relevance_score", 0) for r in results["results"])
                / len(results["results"])
                if results["results"]
                else 0
            )
            self.clarification_service.learn_from_feedback(
                original_query, clarification_responses, avg_relevance
            )

        return results

    # New methods for query completion
    def get_query_suggestions(
        self, partial_query: str, max_suggestions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get query completion suggestions.

        Args:
            partial_query: Partial query string
            max_suggestions: Maximum number of suggestions

        Returns:
            List of query suggestions
        """
        suggestions = self.completion_service.get_suggestions(partial_query, max_suggestions)

        # Add next word predictions
        next_words = self.completion_service.predict_next_word(partial_query)
        for word in next_words:
            suggestions.append(
                {"completion": f"{partial_query} {word}", "type": "next_word", "confidence": 0.6}
            )

        return suggestions[:max_suggestions]

    # Enhanced search method with analytics tracking
    async def search_with_analytics(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Search with detailed analytics tracking.

        Args:
            query: Search query
            **kwargs: Additional search parameters

        Returns:
            Search results with analytics tracked
        """
        start_time = time.time()
        error = None

        try:
            # Perform search
            results = await self.search(query, **kwargs)

            # Track successful search
            latency = time.time() - start_time
            relevance_scores = [r.get("relevance_score", 0) for r in results.get("results", [])]

            self.analytics_service.track_search(
                query=query,
                latency=latency,
                result_count=results.get("total_results", 0),
                relevance_scores=relevance_scores,
                metadata={"filters": kwargs, "from_cache": results.get("from_cache", False)},
                from_cache=results.get("from_cache", False),
            )

            # Learn from successful query
            self.completion_service.learn(query, successful=True)

            return results

        except Exception as e:
            error = str(e)
            latency = time.time() - start_time

            # Track failed search
            self.analytics_service.track_search(
                query=query, latency=latency, result_count=0, relevance_scores=[], error=error
            )

            # Learn from failed query
            self.completion_service.learn(query, successful=False)

            raise

    # Analytics getter methods
    def get_detailed_analytics(self, report_type: str = "summary") -> Dict[str, Any]:
        """
        Get detailed search analytics.

        Args:
            report_type: Type of report ('summary', 'detailed', 'performance')

        Returns:
            Analytics report
        """
        report = self.analytics_service.generate_report(report_type)

        # Add completion service stats
        report["completion_stats"] = self.completion_service.get_stats()

        # Add clarification stats
        report["clarification_stats"] = self.clarification_service.get_clarification_stats()

        return report

    def get_performance_metrics(self, time_range: Optional[str] = None) -> Dict[str, Any]:
        """
        Get search performance metrics.

        Args:
            time_range: Time range for metrics

        Returns:
            Performance metrics
        """
        return self.analytics_service.get_performance_metrics(time_range)

    def get_query_insights(self) -> Dict[str, Any]:
        """
        Get insights about search queries.

        Returns:
            Query insights
        """
        return self.analytics_service.get_query_insights()

    def _get_available_content_types(self) -> List[str]:
        """Get list of available content types."""
        # This would query the database for available content types
        return [
            "spell",
            "monster",
            "item",
            "rule",
            "class",
            "feat",
            "race",
            "background",
            "condition",
        ]
