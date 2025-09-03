"""Unit tests for the search engine and retrieval system."""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.search.hybrid_search import HybridSearchEngine
from src.search.query_processor import QueryProcessor
from src.search.result_ranker import ResultRanker
from src.search.search_service import SearchService


class TestQueryProcessor:
    """Test query processing functionality."""

    def test_query_normalization(self):
        """Test query normalization and cleaning."""
        processor = QueryProcessor()

        # Test basic normalization
        assert processor.normalize_query("  HELLO  World  ") == "hello world"
        assert processor.normalize_query("D&D 5e rules") == "d&d 5e rules"
        assert processor.normalize_query("  ") == ""

    def test_query_expansion(self):
        """Test query expansion with synonyms."""
        processor = QueryProcessor()

        # Test synonym expansion
        expanded = processor.expand_query("attack")
        assert "attack" in expanded
        assert any(word in expanded for word in ["strike", "hit", "assault"])

        expanded = processor.expand_query("wizard")
        assert "wizard" in expanded
        assert any(word in expanded for word in ["mage", "sorcerer", "spellcaster"])

    def test_query_tokenization(self):
        """Test query tokenization."""
        processor = QueryProcessor()

        tokens = processor.tokenize("fireball spell damage")
        assert len(tokens) == 3
        assert "fireball" in tokens
        assert "spell" in tokens
        assert "damage" in tokens

    def test_spell_correction(self):
        """Test spell correction for common misspellings."""
        processor = QueryProcessor()

        # Test common misspellings
        assert processor.correct_spelling("firebll") == "fireball"
        assert processor.correct_spelling("wizrd") == "wizard"
        assert processor.correct_spelling("correct") == "correct"

    def test_query_classification(self):
        """Test query type classification."""
        processor = QueryProcessor()

        # Test different query types
        assert processor.classify_query("what is fireball spell") == "rule_lookup"
        assert processor.classify_query("create character backstory") == "generation"
        assert processor.classify_query("goblin stats") == "monster_lookup"
        assert processor.classify_query("session notes") == "campaign_data"


class TestHybridSearchEngine:
    """Test hybrid search engine functionality."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = Mock()
        db.search = AsyncMock(
            return_value=[{"id": "1", "content": "Test content", "metadata": {}, "distance": 0.1}]
        )
        return db

    @pytest.mark.asyncio
    async def test_semantic_search(self, mock_db):
        """Test semantic vector search."""
        engine = HybridSearchEngine(mock_db)

        results = await engine.semantic_search("test query", "rulebooks", 5)

        assert len(results) > 0
        assert results[0]["content"] == "Test content"
        mock_db.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_keyword_search(self, mock_db):
        """Test keyword-based search."""
        engine = HybridSearchEngine(mock_db)

        # Mock keyword search results
        with patch.object(
            engine,
            "_keyword_search",
            return_value=[{"id": "2", "content": "Keyword match", "score": 0.9}],
        ):
            results = await engine.keyword_search("test", "rulebooks")

            assert len(results) == 1
            assert results[0]["content"] == "Keyword match"

    @pytest.mark.asyncio
    async def test_hybrid_search_combination(self, mock_db):
        """Test combination of semantic and keyword search."""
        engine = HybridSearchEngine(mock_db)

        # Mock both search types
        semantic_results = [{"id": "1", "content": "Semantic match", "score": 0.8}]
        keyword_results = [{"id": "2", "content": "Keyword match", "score": 0.7}]

        with patch.object(engine, "semantic_search", return_value=semantic_results):
            with patch.object(engine, "keyword_search", return_value=keyword_results):
                results = await engine.hybrid_search("test query", "rulebooks")

                # Should combine both result sets
                assert len(results) >= 2
                content_list = [r["content"] for r in results]
                assert "Semantic match" in content_list
                assert "Keyword match" in content_list

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_db):
        """Test search with metadata filters."""
        engine = HybridSearchEngine(mock_db)

        filters = {"system": "D&D 5e", "source_type": "rulebook"}

        await engine.semantic_search("test", "rulebooks", 5, filters)

        # Verify filter was passed to database
        call_args = mock_db.search.call_args
        assert call_args[1]["metadata_filter"] == filters


class TestResultRanker:
    """Test search result ranking."""

    def test_relevance_scoring(self):
        """Test relevance score calculation."""
        ranker = ResultRanker()

        results = [
            {"id": "1", "content": "test", "score": 0.5},
            {"id": "2", "content": "test test", "score": 0.8},
            {"id": "3", "content": "other", "score": 0.3},
        ]

        ranked = ranker.rank_results(results, "test")

        # Higher score should rank first
        assert ranked[0]["id"] == "2"
        assert ranked[-1]["id"] == "3"

    def test_source_boosting(self):
        """Test source-based score boosting."""
        ranker = ResultRanker()

        results = [
            {"id": "1", "score": 0.5, "metadata": {"source": "Player's Handbook"}},
            {"id": "2", "score": 0.5, "metadata": {"source": "Homebrew"}},
        ]

        # Core books should get boost
        ranked = ranker.rank_results(results, "test", boost_core=True)
        assert ranked[0]["metadata"]["source"] == "Player's Handbook"

    def test_duplicate_removal(self):
        """Test removal of duplicate results."""
        ranker = ResultRanker()

        results = [
            {"id": "1", "content": "Same content", "score": 0.8},
            {"id": "2", "content": "Same content", "score": 0.7},
            {"id": "3", "content": "Different", "score": 0.6},
        ]

        deduped = ranker.remove_duplicates(results)

        assert len(deduped) == 2
        contents = [r["content"] for r in deduped]
        assert contents.count("Same content") == 1

    def test_result_limit(self):
        """Test limiting number of results."""
        ranker = ResultRanker()

        results = [{"id": str(i), "score": i / 10} for i in range(20)]

        limited = ranker.limit_results(results, 5)
        assert len(limited) == 5

        # Should keep highest scores
        assert all(r["score"] >= 0.15 for r in limited)


class TestSearchService:
    """Test the main search service."""

    @pytest.fixture
    def search_service(self):
        """Create search service with mocked dependencies."""
        mock_db = Mock()
        mock_db.search = AsyncMock(return_value=[])

        service = SearchService(mock_db)
        service.query_processor = Mock(spec=QueryProcessor)
        service.search_engine = Mock(spec=HybridSearchEngine)
        service.result_ranker = Mock(spec=ResultRanker)

        return service

    @pytest.mark.asyncio
    async def test_search_workflow(self, search_service):
        """Test complete search workflow."""
        # Setup mocks
        search_service.query_processor.normalize_query.return_value = "normalized query"
        search_service.query_processor.expand_query.return_value = [
            "normalized",
            "query",
            "expanded",
        ]

        mock_results = [{"id": "1", "content": "Test", "score": 0.8}]
        search_service.search_engine.hybrid_search = AsyncMock(return_value=mock_results)
        search_service.result_ranker.rank_results.return_value = mock_results

        # Execute search
        results = await search_service.search("Test Query", "rulebooks")

        # Verify workflow
        search_service.query_processor.normalize_query.assert_called_with("Test Query")
        search_service.search_engine.hybrid_search.assert_called()
        search_service.result_ranker.rank_results.assert_called()

        assert len(results) == 1
        assert results[0]["content"] == "Test"

    @pytest.mark.asyncio
    async def test_search_with_caching(self, search_service):
        """Test search result caching."""
        # Enable caching
        search_service.enable_cache = True
        search_service.cache = {}

        mock_results = [{"id": "1", "content": "Cached", "score": 0.8}]
        search_service.search_engine.hybrid_search = AsyncMock(return_value=mock_results)
        search_service.result_ranker.rank_results.return_value = mock_results

        # First search - should hit engine
        results1 = await search_service.search("test", "rulebooks")
        assert search_service.search_engine.hybrid_search.call_count == 1

        # Second search - should hit cache
        results2 = await search_service.search("test", "rulebooks")
        assert search_service.search_engine.hybrid_search.call_count == 1  # Not called again

        assert results1 == results2

    @pytest.mark.asyncio
    async def test_search_error_handling(self, search_service):
        """Test error handling in search."""
        search_service.search_engine.hybrid_search = AsyncMock(
            side_effect=Exception("Search failed")
        )

        with pytest.raises(Exception) as exc_info:
            await search_service.search("test", "rulebooks")

        assert "Search failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, search_service):
        """Test handling of empty queries."""
        results = await search_service.search("", "rulebooks")

        # Should return empty results for empty query
        assert results == []
        search_service.search_engine.hybrid_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_statistics(self, search_service):
        """Test search statistics tracking."""
        mock_results = [{"id": "1", "content": "Test", "score": 0.8}]
        search_service.search_engine.hybrid_search = AsyncMock(return_value=mock_results)
        search_service.result_ranker.rank_results.return_value = mock_results

        # Perform searches
        await search_service.search("test1", "rulebooks")
        await search_service.search("test2", "rulebooks")

        stats = search_service.get_statistics()

        assert stats["total_searches"] >= 2
        assert "test1" in stats["recent_queries"]
        assert "test2" in stats["recent_queries"]


class TestSearchIntegration:
    """Integration tests for search functionality."""

    @pytest.mark.asyncio
    async def test_cross_reference_search(self):
        """Test cross-referencing between rulebooks and campaigns."""
        # This would test the actual integration
        # For now, we'll use mocks
        mock_db = Mock()
        mock_db.search = AsyncMock()

        service = SearchService(mock_db)

        # Mock campaign data
        with patch.object(
            service,
            "_get_campaign_context",
            return_value={
                "current_campaign": "test_campaign",
                "relevant_npcs": ["Gandalf", "Frodo"],
            },
        ):
            # Search should include campaign context
            results = await service.search(
                "wizard spells", "rulebooks", include_campaign_context=True
            )

            # Verify campaign context was considered
            # (In real implementation, this would affect ranking)
            assert service._get_campaign_context.called

    @pytest.mark.asyncio
    async def test_multi_collection_search(self):
        """Test searching across multiple collections."""
        mock_db = Mock()

        # Mock different collection results
        def search_side_effect(collection, *args, **kwargs):
            if collection == "rulebooks":
                return [{"id": "r1", "content": "Rule", "collection": "rulebooks"}]
            elif collection == "flavor_sources":
                return [{"id": "f1", "content": "Flavor", "collection": "flavor_sources"}]
            return []

        mock_db.search = AsyncMock(side_effect=search_side_effect)

        service = SearchService(mock_db)

        # Search across multiple collections
        results = await service.search_multiple_collections("test", ["rulebooks", "flavor_sources"])

        # Should have results from both collections
        collections = [r["collection"] for r in results]
        assert "rulebooks" in collections
        assert "flavor_sources" in collections


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
