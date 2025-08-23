"""Hybrid search engine combining semantic and keyword search."""

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from config.logging_config import get_logger
from config.settings import settings
from src.core.database import get_db_manager
from src.pdf_processing.embedding_generator import EmbeddingGenerator
from src.search.index_persistence import IndexPersistence

logger = get_logger(__name__)


class SearchResult:
    """Represents a search result with combined scores."""

    def __init__(
        self,
        document_id: str,
        content: str,
        metadata: Dict[str, Any],
        semantic_score: float = 0.0,
        keyword_score: float = 0.0,
    ):
        """Initialize search result."""
        self.document_id = document_id
        self.content = content
        self.metadata = metadata
        self.semantic_score = semantic_score
        self.keyword_score = keyword_score
        self.combined_score = 0.0

    def calculate_combined_score(self, semantic_weight: float, keyword_weight: float):
        """Calculate weighted combined score."""
        self.combined_score = (
            self.semantic_score * semantic_weight + self.keyword_score * keyword_weight
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "combined_score": self.combined_score,
        }


class HybridSearchEngine:
    """Implements hybrid search combining semantic and keyword approaches."""

    def __init__(self):
        """Initialize the hybrid search engine."""
        self.db = get_db_manager()
        self.embedding_generator = EmbeddingGenerator()
        self.index_persistence = IndexPersistence()
        self.bm25_indices = {}  # Collection -> BM25 index
        self.document_cache = {}  # Collection -> documents
        self.tokenized_cache = {}  # Collection -> tokenized documents
        self._initialize_indices()

    def _initialize_indices(self):
        """Initialize BM25 indices for collections, using persisted indices when available."""
        collections = ["rulebooks", "flavor_sources"]

        for collection_name in collections:
            try:
                # Load documents from collection with pagination
                documents = self._load_documents_paginated(collection_name)

                if not documents:
                    logger.warning(f"No documents found in {collection_name}")
                    continue

                # Cache documents
                self.document_cache[collection_name] = documents

                # Try to load persisted index
                if self.index_persistence.is_index_valid(collection_name, documents):
                    loaded = self.index_persistence.load_index(collection_name)
                    if loaded:
                        self.bm25_indices[collection_name] = loaded["index"]
                        self.tokenized_cache[collection_name] = loaded["tokenized_docs"]
                        logger.info(
                            f"Loaded persisted index for {collection_name}",
                            document_count=len(documents),
                        )
                        continue

                # Build new index if not loaded from disk
                tokenized_docs = self._build_bm25_index(collection_name, documents)

                # Persist the new index
                if self.bm25_indices.get(collection_name):
                    self.index_persistence.save_index(
                        collection_name,
                        self.bm25_indices[collection_name],
                        documents,
                        tokenized_docs,
                    )

                logger.info(
                    f"Built and saved new index for {collection_name}",
                    document_count=len(documents),
                )

            except Exception as e:
                logger.error(f"Failed to initialize index for {collection_name}", error=str(e))

    def _load_documents_paginated(
        self, collection_name: str, batch_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Load documents from collection with pagination.

        Args:
            collection_name: Name of collection
            batch_size: Number of documents per batch

        Returns:
            List of all documents
        """
        all_documents = []
        offset = 0

        while True:
            try:
                # Load batch of documents
                batch = self.db.list_documents(
                    collection_name=collection_name, limit=batch_size, offset=offset
                )

                if not batch:
                    break

                all_documents.extend(batch)
                offset += batch_size

                # Log progress for large collections
                if offset % 5000 == 0:
                    logger.debug(f"Loaded {offset} documents from {collection_name}")

            except Exception as e:
                logger.error(f"Error loading documents at offset {offset}: {str(e)}")
                break

        return all_documents

    def _build_bm25_index(
        self, collection_name: str, documents: List[Dict[str, Any]]
    ) -> List[List[str]]:
        """
        Build BM25 index for a collection.

        Args:
            collection_name: Name of collection
            documents: Documents to index

        Returns:
            Tokenized documents
        """
        # Tokenize documents for BM25
        tokenized_docs = []
        for doc in documents:
            content = doc.get("content", "")
            tokens = self._tokenize(content)
            tokenized_docs.append(tokens)

        # Create BM25 index
        if tokenized_docs:
            self.bm25_indices[collection_name] = BM25Okapi(tokenized_docs)
            self.tokenized_cache[collection_name] = tokenized_docs
        else:
            self.bm25_indices[collection_name] = None
            self.tokenized_cache[collection_name] = []

        return tokenized_docs

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r"[^\w\s]", " ", text)

        # Split into tokens
        tokens = text.split()

        # Remove stop words (basic list)
        stop_words = {
            "the",
            "is",
            "at",
            "which",
            "on",
            "a",
            "an",
            "as",
            "are",
            "was",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "in",
            "of",
            "to",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
        }

        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

        return tokens

    async def search(
        self,
        query: str,
        collection_name: str = "rulebooks",
        max_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True,
        semantic_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search on a collection.

        Args:
            query: Search query
            collection_name: Collection to search
            max_results: Maximum results to return
            metadata_filter: Optional metadata filtering
            use_hybrid: Whether to use hybrid search
            semantic_weight: Weight for semantic search
            keyword_weight: Weight for keyword search

        Returns:
            List of search results
        """
        semantic_weight = semantic_weight or settings.semantic_weight
        keyword_weight = keyword_weight or settings.keyword_weight

        logger.info(
            f"Performing search",
            query=query[:100],
            collection=collection_name,
            use_hybrid=use_hybrid,
        )

        results = []

        if use_hybrid:
            # Perform both searches
            semantic_results = await self._semantic_search(
                query, collection_name, max_results * 2, metadata_filter
            )
            keyword_results = self._keyword_search(
                query, collection_name, max_results * 2, metadata_filter
            )

            # Merge results
            results = self._merge_results(
                semantic_results,
                keyword_results,
                semantic_weight,
                keyword_weight,
                max_results,
            )
        else:
            # Semantic search only
            results = await self._semantic_search(
                query, collection_name, max_results, metadata_filter
            )

        return results

    async def _semantic_search(
        self,
        query: str,
        collection_name: str,
        max_results: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Perform semantic similarity search.

        Args:
            query: Search query
            collection_name: Collection to search
            max_results: Maximum results
            metadata_filter: Metadata filtering

        Returns:
            List of search results
        """
        try:
            # Search database (ChromaDB will generate embeddings internally)
            db_results = self.db.search(
                collection_name=collection_name,
                query=query,
                n_results=max_results,
                metadata_filter=metadata_filter,
            )

            # Convert to SearchResult objects
            results = []
            for db_result in db_results:
                # Calculate semantic score (1 - distance for similarity)
                semantic_score = 1.0 - db_result.get("distance", 0)

                result = SearchResult(
                    document_id=db_result["id"],
                    content=db_result["content"],
                    metadata=db_result["metadata"],
                    semantic_score=semantic_score,
                    keyword_score=0.0,
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Semantic search failed", error=str(e))
            return []

    def _keyword_search(
        self,
        query: str,
        collection_name: str,
        max_results: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query
            collection_name: Collection to search
            max_results: Maximum results
            metadata_filter: Metadata filtering

        Returns:
            List of search results
        """
        try:
            if collection_name not in self.bm25_indices or not self.bm25_indices[collection_name]:
                logger.warning(f"No BM25 index for {collection_name}")
                return []

            # Get documents and index
            documents = self.document_cache.get(collection_name, [])
            bm25 = self.bm25_indices[collection_name]

            # Tokenize query
            query_tokens = self._tokenize(query)

            # Get BM25 scores
            scores = bm25.get_scores(query_tokens)

            # Apply metadata filter if provided
            filtered_docs = []
            for i, doc in enumerate(documents):
                if metadata_filter and not self._matches_filter(doc["metadata"], metadata_filter):
                    continue
                filtered_docs.append((i, doc, scores[i]))

            # Sort by score and get top results
            filtered_docs.sort(key=lambda x: x[2], reverse=True)
            top_docs = filtered_docs[:max_results]

            # Convert to SearchResult objects
            results = []
            max_score = max(scores) if scores else 1.0

            for idx, doc, score in top_docs:
                # Normalize score to 0-1 range
                normalized_score = score / max_score if max_score > 0 else 0

                result = SearchResult(
                    document_id=doc["id"],
                    content=doc["content"],
                    metadata=doc["metadata"],
                    semantic_score=0.0,
                    keyword_score=normalized_score,
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Keyword search failed", error=str(e))
            return []

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def _merge_results(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        semantic_weight: float,
        keyword_weight: float,
        max_results: int,
    ) -> List[SearchResult]:
        """
        Merge semantic and keyword search results.

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            semantic_weight: Weight for semantic scores
            keyword_weight: Weight for keyword scores
            max_results: Maximum results to return

        Returns:
            Merged and ranked results
        """
        # Create a map of document_id to result
        merged_map = {}

        # Add semantic results
        for result in semantic_results:
            merged_map[result.document_id] = result

        # Merge keyword results
        for result in keyword_results:
            if result.document_id in merged_map:
                # Update keyword score for existing result
                merged_map[result.document_id].keyword_score = result.keyword_score
            else:
                # Add new result
                merged_map[result.document_id] = result

        # Calculate combined scores
        for result in merged_map.values():
            result.calculate_combined_score(semantic_weight, keyword_weight)

        # Sort by combined score
        sorted_results = sorted(merged_map.values(), key=lambda x: x.combined_score, reverse=True)

        # Return top results
        return sorted_results[:max_results]

    def update_index(self, collection_name: str, force_rebuild: bool = False):
        """
        Update BM25 index for a collection.

        Args:
            collection_name: Collection to update
            force_rebuild: Force rebuild even if documents haven't changed
        """
        try:
            # Reload documents with pagination
            documents = self._load_documents_paginated(collection_name)

            if documents:
                # Check if update is needed
                if not force_rebuild and self.index_persistence.is_index_valid(
                    collection_name, documents
                ):
                    logger.info(f"Index for {collection_name} is already up to date")
                    return

                # Update cache
                self.document_cache[collection_name] = documents

                # Rebuild BM25 index
                tokenized_docs = self._build_bm25_index(collection_name, documents)

                # Delete old persisted index and save new one
                self.index_persistence.delete_index(collection_name)
                if self.bm25_indices.get(collection_name):
                    self.index_persistence.save_index(
                        collection_name,
                        self.bm25_indices[collection_name],
                        documents,
                        tokenized_docs,
                    )

                logger.info(
                    f"Updated index for {collection_name}",
                    document_count=len(documents),
                )
            else:
                logger.warning(f"No documents to index in {collection_name}")

        except Exception as e:
            logger.error(f"Failed to update index", error=str(e))

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about search indices."""
        stats = {}

        for collection_name in self.document_cache:
            doc_count = len(self.document_cache.get(collection_name, []))
            has_bm25 = (
                collection_name in self.bm25_indices
                and self.bm25_indices[collection_name] is not None
            )

            stats[collection_name] = {
                "document_count": doc_count,
                "has_bm25_index": has_bm25,
            }

        return stats
